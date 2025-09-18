import os
import sys
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

import tqdm
import math
import argparse
import numpy as np
import pandas as pd
import scipy.io as sio
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import logging as hf_logging
from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler
from eval_prob_adaptive import eval_prob_adaptive_differentiable
from ControlNet import ControlNet
import process_rdf as prdf
import time
from diffusion.datasets import ThzDataset


hf_logging.set_verbosity_error()
device = "cuda" if torch.cuda.is_available() else "cpu"


# ========= Dienstfunktionen =========

def set_seed(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# ========= deine ControlNet-Klasse (Adapter) einhängen =========

class ControlNetAdapterWrapper(torch.nn.Module):
    """
    Nimmt Volumen [B,1,T,H,W] in [0,1] und gibt ein Bild [B,3,512,512] in [0,1] aus.
    Reduziert T vor dem 3D-Netz per AvgPool, um Speicher zu schonen.
    """
    def __init__(self, controlnet_cfg, in_channels=2, out_size=512, target_T=512, mid_channels=8, stride_T=3):
        super().__init__()
        self.out_size = out_size
        self.target_T = target_T
        self.in_channels = in_channels

        # Lernbarer T-Downsampler: erst Feature-Anhebung, dann stride in T
        # stride_T=3 macht aus 1400 -> ~467 (1400//3); das ist schon ~64% Speicherersparnis.
        # Du kannst stride_T=2 setzen, wenn du konservativer (mehr Info, mehr RAM) sein willst (-> 1400->700).
        self.downT = nn.Sequential(
            nn.Conv3d(in_channels, mid_channels, kernel_size=(5,3,3), stride=(stride_T,1,1), padding=(2,1,1), bias=False),
            nn.SiLU(),
            nn.Conv3d(mid_channels, in_channels, kernel_size=1, stride=1, padding=0, bias=False)
        )

        num_downsamples = len(controlnet_cfg.get("num_channels", ())) - 1
        downsample_factor = 2 ** num_downsamples
        final_depth = target_T // downsample_factor

        print(f"Adapter konfiguriert: Eingangstiefe -> {stride_T}x Reduktion -> {target_T} -> ControlNet-Flaschenhals-Tiefe: {final_depth}")

        # Schritt 3: Aktualisiere die ControlNet-Konfiguration und instanziiere das Netz
        # Dies stellt sicher, dass das ControlNet immer korrekt gebaut wird.
        cfg = controlnet_cfg.copy()
        cfg['in_channels'] = self.in_channels
        cfg['conditioning_embedding_in_channels'] = self.in_channels
        cfg['final_depth'] = final_depth

        self.net = ControlNet(**cfg)

    def forward(self, vol):  # vol: [B,1,T,H,W]
        # 1) Lernbares Downsampling in T:
        vol = self.downT(vol)     # [B,1,T',H,W], T' ~= T/stride_T

        # (Optional) sanfte Angleichung auf ein einheitliches Ziel-T (z.B. 500)
        # Das ist linear (ohne Extra-Parameter) und sehr günstig:
        if self.target_T is not None and vol.shape[2] != self.target_T:
            vol = F.interpolate(vol, size=(self.target_T, vol.shape[-2], vol.shape[-1]),
                                mode="trilinear", align_corners=False)

        # 2) Stubs fürs 3D-ControlNet (ignoriert x/t/context intern)
        B, C, Tp, H, W = vol.shape
        x_stub = torch.zeros((B, C, Tp, H, W), device=vol.device, dtype=vol.dtype)
        t_stub = torch.zeros((B,), device=vol.device, dtype=torch.long)

        # 3) Dein ControlNet rechnet 3D->2D
        rgb = self.net(x_stub, t_stub, controlnet_cond=vol, conditioning_scale=1.0, context=None)  # [B,3,h,w]

        # 4) Final auf out_size bringen
        if (rgb.shape[-2], rgb.shape[-1]) != (self.out_size, self.out_size):
            rgb = F.interpolate(rgb, size=(self.out_size, self.out_size), mode="bilinear", align_corners=False)

        return torch.sigmoid(rgb)
    

# ========= SD 2.1 base laden (eingefroren) =========

def build_sd2_1_base(dtype="float16", use_xformers=True):
    model_id = "stabilityai/stable-diffusion-2-1-base"
    scheduler = EulerDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")
    torch_dtype = torch.float16 if dtype == "float16" else torch.float32
    pipe = StableDiffusionPipeline.from_pretrained(model_id, scheduler=scheduler, torch_dtype=torch_dtype)
    if use_xformers:
        try:
            pipe.enable_xformers_memory_efficient_attention()
        except Exception:
            pass
    pipe = pipe.to(device)
    vae = pipe.vae.eval()
    unet = pipe.unet.eval()
    tokenizer = pipe.tokenizer
    text_encoder = pipe.text_encoder.eval()
    try:
        unet.enable_gradient_checkpointing()
    except Exception:
        pass

    # einfrieren
    for p in list(vae.parameters()) + list(unet.parameters()) + list(text_encoder.parameters()):
        p.requires_grad = False
    return vae, unet, tokenizer, text_encoder, scheduler

# ========= Training / Evaluation =========

def load_class_text_embeds(classes, prompts, tokenizer, text_encoder):
    with torch.no_grad():
        text_in = tokenizer(prompts, padding="max_length",
                            max_length=tokenizer.model_max_length,
                            truncation=True, return_tensors="pt").to(device)
        class_embeds = text_encoder(text_in.input_ids)[0]  # [C,seq,hid]
    return class_embeds


def acc_from_errors(errors_per_class: torch.Tensor, y: int) -> float:
    pred = torch.argmin(errors_per_class).item()
    return float(pred == y)


def run_eval(dataloader, adapter, vae, unet, class_embeds, scheduler, eargs, img_size):
    adapter.eval()
    total, correct = 0, 0
    with torch.no_grad():
        for vol, label, filename in dataloader:
            vol = vol.to(device)
            label = label.to(device).long()

            if vol.dim() == 6:  # [B,1,1,T,H,W] -> [B,1,T,H,W]
                vol = vol.squeeze(1)

            use_amp = (device == "cuda" and eargs.dtype == "float16")
            with torch.amp.autocast('cuda', dtype=torch.float16, enabled=use_amp):
                img  = adapter(vol)                       # [1,3,512,512]
                x_in = (img * 2.0 - 1.0).to(dtype=vae.dtype)
                lat  = vae.encode(x_in).latent_dist.mean * 0.18215

            # pro Sample auswerten (wie DC)
            for b in range(vol.size(0)):
                pred_idx, data, errors_per_class = eval_prob_adaptive_differentiable(
                    unet=unet, latent=lat[b:b+1], text_embeds=class_embeds,
                    scheduler=scheduler, args=eargs, latent_size=img_size // 8, all_noise=None
                )
                correct += int(torch.argmin(errors_per_class).item() == label[b].item())
                total += 1
    adapter.train()
    return correct / max(1, total)


class PromptBank:
    def __init__(self, prompt_csv: str):
        dfp = pd.read_csv(prompt_csv)  # Spalten: prompt, classname, classidx
        assert {"prompt", "classname", "classidx"}.issubset(dfp.columns), \
            "Prompt-CSV muss prompt,classname,classidx enthalten."
        dfp = dfp.sort_values(["classidx", "prompt"]).reset_index(drop=True)
        self.prompt_texts = dfp["prompt"].astype(str).tolist()               # [P]
        self.prompt_to_class = torch.tensor(dfp["classidx"].tolist()).long() # [P]
        self.classnames = (
            dfp.drop_duplicates("classidx").sort_values("classidx")["classname"].tolist()
        )
        self.num_classes = len(self.classnames)

    def to_text_embeds(self, tokenizer, text_encoder, device):
        with torch.no_grad():
            batch = tokenizer(
                self.prompt_texts,
                padding="max_length",
                max_length=tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            ).to(device)
            embeds = text_encoder(batch.input_ids)[0]  # [P, seq, hid]
        return embeds


# ---- Prompt-Fehler → Klassen-Fehler poolen (mean/min) ----
def pool_prompt_errors_to_class_errors(
    errors_per_prompt: torch.Tensor,
    prompt_to_class: torch.Tensor,
    num_classes: int,
    reduce: str = "mean",
) -> torch.Tensor:
    """
    Poolt Prompt-Fehler zu Klassen-Fehlern, dtype-sicher (keine Overflows in fp16).
    """
    dev = errors_per_prompt.device
    dt  = errors_per_prompt.dtype
    finfo = torch.finfo(dt)

    # dtype-sicherer großer Wert (z.B. ~6.5e4 bei fp16, ~1e19 Deckel bei fp32)
    fill_float = min(1e6, float(finfo.max * 0.5))
    fill = torch.tensor(fill_float, device=dev, dtype=dt)

    # init mit großem Wert
    class_errors = torch.full((num_classes,), fill, device=dev, dtype=dt)

    for c in range(num_classes):
        mask = (prompt_to_class == c)
        if torch.any(mask):
            vals = errors_per_prompt[mask]
            # numerisch robust zusammenfegen
            vals = torch.nan_to_num(vals, nan=float(fill), posinf=float(fill), neginf=float(fill))
            ce = vals.mean() if reduce == "mean" else vals.min()
            class_errors[c] = ce

    class_errors = torch.nan_to_num(class_errors, nan=float(fill), posinf=float(fill), neginf=float(fill))
    return class_errors


# ---- Accuracy aus Klassen-Fehlern ----
def acc_from_class_errors(class_errors: torch.Tensor, y: int) -> float:
    pred = torch.argmin(class_errors).item()
    return float(pred == y)

def pool_prompt_errors_to_class_errors_batch(errors_per_prompt, prompt_to_class, num_classes, reduce="mean"):
    """ Vektorisierte Version, die einen Batch von Fehler-Tensoren verarbeitet. """
    B, P = errors_per_prompt.shape
    device = errors_per_prompt.device
    
    one_hot_matrix = F.one_hot(prompt_to_class, num_classes=num_classes).float().to(device)
    
    errors_grouped = errors_per_prompt.unsqueeze(2) * one_hot_matrix.unsqueeze(0)
    
    sum_errors = errors_grouped.sum(dim=1)
    prompts_per_class = one_hot_matrix.sum(dim=0)
    
    if reduce == "mean":
        class_errors = sum_errors / (prompts_per_class.unsqueeze(0) + 1e-8)
    elif reduce == "min":
        errors_grouped[errors_grouped == 0] = float('inf')
        class_errors = errors_grouped.min(dim=1).values
    else:
        raise ValueError("reduce must be 'mean' or 'min'")
        
    return class_errors

def main():
    ap = argparse.ArgumentParser()

    # Daten
    ap.add_argument("--data_train", type=str, required=True, help="Wurzelordner der .mat-Dateien")
    ap.add_argument("--data_test", type=str, required=True, help="Wurzelordner der .mat-Dateien")
    ap.add_argument("--train_csv", type=str, required=True, help="CSV: filename,label (Train)")
    ap.add_argument("--val_csv",   type=str, required=True, help="CSV: filename,label (Val)")
    ap.add_argument("--prompts_csv", type=str, required=True, help="CSV: prompt,classname,classidx")

    # SD/DC-Args
    ap.add_argument("--dtype", type=str, default="float16", choices=("float16", "float32"))
    ap.add_argument("--img_size", type=int, default=512, choices=(256, 512))
    ap.add_argument("--loss", type=str, default="l2", choices=("l1", "l2", "huber"))
    ap.add_argument("--n_trials", type=int, default=1)
    ap.add_argument("--n_samples", nargs="+", type=int, required=True, help="z. B. 8 4 2 1")
    ap.add_argument("--to_keep",   nargs="+", type=int, required=True, help="z. B. 6 3 2 1")
    ap.add_argument("--num_train_timesteps", type=int, default=1000)
    ap.add_argument("--version", type=str, default="2-1", help="Stable Diffusion Version (2-1, 2-0, etc.)", choices=("2-1", "2-0", '1-1', '1-2', '1-3', '1-4', '1-5'))

    from datetime import datetime
    from zoneinfo import ZoneInfo 

    stamp = datetime.now(ZoneInfo("Europe/Berlin")).strftime("%y%m%d_%H%M")

    # Training
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--save_dir", type=str, default=f"./runs/checkpoints_adapter_{stamp}")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--use_xformers", action="store_true")
    args = ap.parse_args()

    set_seed(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)

    # Stable Diffusion 2.1 base einfrieren
    vae, unet, tokenizer, text_encoder, scheduler = build_sd2_1_base(dtype=args.dtype, use_xformers=args.use_xformers)

    # Datasets
    train_ds = ThzDataset(args.data_train, args.train_csv, is_train=True)
    val_ds   = ThzDataset(args.data_test, args.val_csv, is_train=False)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds, batch_size=1, shuffle=False,
                              num_workers=args.num_workers, pin_memory=True)

    # Prompts (separat) laden
    pb = PromptBank(args.prompts_csv)
    prompt_embeds   = pb.to_text_embeds(tokenizer, text_encoder, device)  # [P, seq, hid]
    prompt_to_class = pb.prompt_to_class.to(device)                        # [P]
    num_classes     = pb.num_classes                                       # C

    # Kleiner Konsistenz-Check (optional): gleiche Anzahl Klassen wie im Dataset-Mapping
    # (Dataset mappt str-Labels alphabetisch. Hier legen wir classidx fest. Stelle sicher,
    # dass classidx {0..C-1} und die Label-Mapping-Reihenfolge zusammenpassen!)
    # -> Wenn du absolute Kontrolle willst, gib eine Mapping-Datei vor. Für jetzt: Warnhinweis.
    if len(train_ds.class_to_idx) != num_classes:
        print(f"[WARN] Klassenanzahl Dataset ({len(train_ds.class_to_idx)}) != Prompts ({num_classes}). "
              f"Bitte Mapping konsistent gestalten.")

    # Adapter (trainierbar)
    controlnet_cfg = dict(
        spatial_dims=3,
        num_res_blocks=(2, 2, 2, 2),
        num_channels=(32, 64, 64, 64),
        attention_levels=(False, False, False, False),
        conditioning_embedding_in_channels=2,
        conditioning_embedding_num_channels=(32, 64, 64, 64),
        with_conditioning=False,
    )
    adapter = ControlNetAdapterWrapper(
        controlnet_cfg=controlnet_cfg,
        in_channels=2,          
        out_size=args.img_size,
        target_T=512,           
        stride_T=3              
    ).to(device)
    adapter.train()
    opt = torch.optim.AdamW(adapter.parameters(), lr=args.lr)

    use_amp = (device == "cuda" and args.dtype == "float16")
    scaler = torch.amp.GradScaler('cuda',enabled=use_amp)

    # eval_prob_adaptive_differentiable-Args
    P = len(pb.prompt_texts)
    class EArgs: pass
    eargs = EArgs()
    eargs.n_samples = args.n_samples
    eargs.to_keep = [P] * len(eargs.n_samples)
    eargs.n_trials = args.n_trials
    eargs.batch_size = args.batch_size
    eargs.dtype = 'float32'
    eargs.loss = args.loss
    eargs.num_train_timesteps = args.num_train_timesteps
    eargs.version = args.version

    best_val_acc = -1.0
    best_path = os.path.join(args.save_dir, "adapter_best.pt")

    # ---------- TRAIN ----------
    for epoch in range(1, args.epochs + 1):
        running_loss, running_acc, n_seen = 0.0, 0.0, 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
        for vol, label, filename in pbar:
            vol   = vol.to(device)        # [B,1,1,T,H,W]
            label = label.to(device).long()

            if vol.dim() == 6:  # [B,1,1,T,H,W] -> [B,1,T,H,W]
                vol = vol.squeeze(1)

            opt.zero_grad(set_to_none=True)

            # Autocast nur, wenn use_amp=True (sonst no-op)
            with torch.amp.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                # 1) Adapter 
                img  = adapter(vol)         
                x_in = (img * 2.0 - 1.0).to(dtype=vae.dtype)      
                lat  = vae.encode(x_in).latent_dist.mean * 0.18215  

                pred_idx, data, errors_per_prompt_batch = eval_prob_adaptive_differentiable(
                    unet=unet, latent=lat,
                    text_embeds=prompt_embeds, scheduler=scheduler,
                    args=eargs, latent_size=args.img_size // 8, all_noise=None
                )
                print(f"[E{epoch:02d}"
                        f"Pred={int(pred_idx)} "
                        f"MinErr={float(errors_per_prompt.min()):.3e} "
                        f"MaxErr={float(errors_per_prompt.max()):.3e} "
                        f"MeanErr={float(errors_per_prompt.mean()):.3e}")
                
                class_errors_batch = pool_prompt_errors_to_class_errors_batch(
                    errors_per_prompt_batch, prompt_to_class, num_classes, reduce="mean"
                )  # [C]
                print("class_errors dtype:", class_errors_batch.dtype, " device:", class_errors_batch.device)
                print("class_errors:", class_errors_batch)
                logit_scale = 10.0
                logits = (-class_errors) * logit_scale      # [1,C]
                loss = F.cross_entropy(logits, label)

            if torch.isfinite(loss):
                scaler.scale(loss).backward()
                # Gradient Clipping zur weiteren Stabilisierung
                torch.nn.utils.clip_grad_norm_(adapter.parameters(), max_norm=1.0)
                scaler.step(opt)
                scaler.update()
            else:
                print(f" Nicht-finiter Loss ({loss.item()}) in Epoche {epoch}. Überspringe Optimizer-Step.")
                continue
            with torch.no_grad():
                running_loss += loss.item() * vol.size(0)
                preds = torch.argmin(class_errors_batch, dim=1)
                running_acc += (preds == label).sum().item()
                n_seen += vol.size(0)
                
                # Fortschrittsanzeige aktualisieren
                pbar.set_postfix(loss=f"{running_loss/n_seen:.4f}", acc=f"{running_acc/n_seen:.4f}")

        train_loss = running_loss / max(1, n_seen)
        train_acc = running_acc / max(1, n_seen)
        print(f"[E{epoch:02d}] train_loss={train_loss:.4f}  train_acc={train_acc:.4f}")

        # ---------- VALIDIERUNG ----------
        adapter.eval()
        total, correct = 0, 0
        with torch.no_grad():
            for vol, label, filename in val_loader:
                vol   = vol.to(device)
                label = label.to(device).long()
                if vol.dim() == 6:
                    vol = vol.squeeze(1)

                with torch.amp.autocast('cuda', dtype=torch.float16, enabled=use_amp):
                    img  = adapter(vol)                       # [1,3,512,512]
                    x_in = (img * 2.0 - 1.0).to(dtype=vae.dtype)
                    lat  = vae.encode(x_in).latent_dist.mean * 0.18215

                pred_idx, data, errors_per_prompt = eval_prob_adaptive_differentiable(
                    unet=unet, latent=lat, text_embeds=prompt_embeds,
                    scheduler=scheduler, args=eargs, latent_size=args.img_size // 8, all_noise=None
                )
                errors_per_prompt = torch.nan_to_num(errors_per_prompt, nan=0.0, posinf=float('inf'), neginf=0.0)

                class_errors = pool_prompt_errors_to_class_errors(
                    errors_per_prompt, prompt_to_class, num_classes, reduce="mean"
                )
                pred = torch.argmin(class_errors).item()
                correct += int(pred == label.item())
                total   += 1

        val_acc = correct / max(1, total)
        adapter.train()
        print(f"[E{epoch:02d}] val_acc={val_acc:.4f}")

        # ---------- BESTES MODELL SPEICHERN ----------
        improved = (val_acc > best_val_acc) or math.isclose(val_acc, best_val_acc, rel_tol=1e-6)
        if improved:
            best_val_acc = val_acc
            bad_epochs = 0
            torch.save({
                "epoch": epoch,
                "adapter_state_dict": adapter.state_dict(),
                "optimizer_state_dict": opt.state_dict(),
                "best_val_acc": best_val_acc,
                "classnames": pb.classnames,
                "prompts_csv": args.prompts_csv,
                "controlnet_cfg": controlnet_cfg,
                "n_samples": args.n_samples,
                "to_keep": args.to_keep,
                "img_size": args.img_size,
                "dtype": args.dtype,
            }, best_path)
            print(f"-> Neues Best-Model gespeichert: {best_path}")
        

    print(f"Training beendet. Bestes Checkpoint: {best_path} (val_acc={best_val_acc:.4f})")


if __name__ == "__main__":
    main()