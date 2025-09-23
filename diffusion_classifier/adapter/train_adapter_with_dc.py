import os
import sys
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

from tqdm import tqdm
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
from adapter.ControlNet_Adapter_wrapper import ControlNetAdapterWrapper
import process_rdf as prdf
import time
from diffusion.datasets import ThzDataset
import csv
from collections import defaultdict
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold



hf_logging.set_verbosity_error()
device = "cuda" if torch.cuda.is_available() else "cpu"

# ========= Utils =========
def set_seed(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# ========= SD 2.1 base laden (eingefroren) =========

def build_sd2_1_base(dtype="float16", use_xformers=True):
    model_id = "stabilityai/stable-diffusion-2-1-base"
    scheduler = EulerDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")
    torch_dtype = torch.float32
    if dtype == "float16":
        torch_dtype = torch.float16
    elif dtype == "bfloat16":
        torch_dtype = torch.bfloat16
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

def read_csv_pairs(csv_path):
    rows = []
    with open(csv_path, newline="") as f:
        r = csv.reader(f); header = next(r)
        i_f = header.index("filename"); i_l = header.index("label")
        for line in r:
            rows.append((line[i_f], int(line[i_l])))
    return rows  # List[(filename, label)]

def write_csv_pairs(csv_path, rows):
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["filename","label"])
        w.writerows(rows)

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
            all_embeds = []
            for i in range(0, len(self.prompt_texts), 100):
                batch_texts = self.prompt_texts[i:i+100]
                batch = tokenizer(
                    batch_texts, padding="max_length", max_length=tokenizer.model_max_length,
                    truncation=True, return_tensors="pt"
                ).to(device)
                embeds = text_encoder(batch.input_ids).last_hidden_state
                all_embeds.append(embeds)
        return torch.cat(all_embeds, dim=0)


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
    ap.add_argument("--dtype", type=str, default="float16", choices=("float16", "float32", "bfloat16"))
    ap.add_argument("--img_size", type=int, default=256, choices=(256, 512))
    ap.add_argument("--loss", type=str, default="l2", choices=("l1", "l2", "huber"))
    ap.add_argument("--n_trials", type=int, default=1)
    ap.add_argument("--n_samples", nargs="+", type=int, required=True, help="z. B. 8 4 2 1")
    ap.add_argument("--to_keep",   nargs="+", type=int, required=True, help="z. B. 6 3 2 1")
    ap.add_argument("--num_train_timesteps", type=int, default=1000)
    ap.add_argument("--logit_scale", type=int, default=50)
    ap.add_argument("--version", type=str, default="2-1", help="Stable Diffusion Version (2-1, 2-0, etc.)", choices=("2-1", "2-0", '1-1', '1-2', '1-3', '1-4', '1-5'))
    # Training
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--save_dir", type=str, default=f"./runs/checkpoints_adapter")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--use_xformers", action="store_true")
    ap.add_argument("--n_splits", type=int, default=10, help=">1 aktiviert K-Fold Cross-Validation, z. B. 5")
    args = ap.parse_args()

    from datetime import datetime
    from zoneinfo import ZoneInfo 

    stamp = datetime.now(ZoneInfo("Europe/Berlin")).strftime("%y%m%d_%H%M")
    save_dir = os.path.join(args.save_dir, f"run_{stamp}")

    os.makedirs(save_dir, exist_ok=True)

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
        target_T=64,           
        stride_T=4              
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
    eargs.to_keep = args.to_keep
    eargs.n_trials = args.n_trials
    eargs.batch_size = args.batch_size
    eargs.dtype = args.dtype
    eargs.loss = args.loss
    eargs.num_train_timesteps = args.num_train_timesteps
    eargs.version = args.version

    best_val_acc = -1.0
    best_path = os.path.join(save_dir, "adapter_best.pt")

    # ---------- TRAIN ----------
    for epoch in range(1, args.epochs + 1):
        running_loss, running_acc, n_seen = 0.0, 0.0, 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
        for vol, label, filename in pbar:
            vol   = vol.to(device)        # [B,1,1,T,H,W]
            label = label.to(device).long()

            if vol.dim() == 6:  # [B,1,1,T,H,W] -> [B,1,T,H,W]
                vol = vol.squeeze(1)

            opt.zero_grad()

            # Autocast nur, wenn use_amp=True (sonst no-op)
            with torch.amp.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                # 1) Adapter 
                img  = adapter(vol)         
                x_in = (img * 2.0 - 1.0).to(dtype=vae.dtype)      
                lat  = vae.encode(x_in).latent_dist.mean * 0.18215  

                batch_errors_list = []
                for b in range(lat.size(0)):
                    pred_idx, data, errors_per_prompt_single = eval_prob_adaptive_differentiable(
                        unet=unet, latent=lat[b:b+1],
                        text_embeds=prompt_embeds, scheduler=scheduler,
                        args=eargs, latent_size=args.img_size // 8, all_noise=None
                    )
                    batch_errors_list.append(errors_per_prompt_single)
                errors_per_prompt_batch = torch.stack(batch_errors_list, dim=0)
                class_errors_batch = pool_prompt_errors_to_class_errors_batch(
                    errors_per_prompt_batch, prompt_to_class, num_classes, reduce="mean"
                )
                logit_scale = float(args.logit_scale)
                logits = (-class_errors_batch) * logit_scale
                loss = F.cross_entropy(logits, label)
                print(f"Loss: {loss}")

            if torch.isfinite(loss):
                scaler.scale(loss).backward()
                scaler.unscale_(opt)

                print("[grad] img.grad is None?", img.grad is None)
                if img.grad is not None:
                    print("[grad] img.grad norm:", img.grad.norm().item())

                # Optional: Grad am ersten Adapter-Parameter
                first_param = next(adapter.parameters())
                print("[grad] first adapter param grad is None?", first_param.grad is None)
                if first_param.grad is not None:
                    print("[grad] first adapter param grad norm:", first_param.grad.norm().item())

                torch.nn.utils.clip_grad_norm_(adapter.parameters(), max_norm=0.5)
                scaler.step(opt)
                scaler.update()
                print(f" Finiter Loss ({loss.item()}) in Epoche {epoch}. Optimizer-Step ausgeführt.")
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

                class_errors = pool_prompt_errors_to_class_errors(
                    errors_per_prompt.squeeze(0), prompt_to_class, num_classes, reduce="mean"
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

    # ---------- END-EVAL BESTES MODELL ----------
    print("End-Eval des besten Modells...")
    ckpt = torch.load(best_path, map_location="cpu")
    adapter = ControlNetAdapterWrapper(
        controlnet_cfg=ckpt["controlnet_cfg"],
        in_channels=2,
        out_size=ckpt["img_size"],
        target_T=64,
        stride_T=4
    ).to(device)
    adapter.load_state_dict(ckpt["adapter_state_dict"], strict=True)
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

            class_errors = pool_prompt_errors_to_class_errors(
                errors_per_prompt.squeeze(0), prompt_to_class, num_classes, reduce="mean"
            )
            pred = torch.argmin(class_errors).item()
            correct += int(pred == label.item())
            total   += 1

        val_acc = correct / max(1, total)
    
    print(f"End-Eval val_acc={val_acc:.4f} (bestes Modell)")



if __name__ == "__main__":
    main()