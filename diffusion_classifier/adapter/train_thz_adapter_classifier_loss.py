# train_thz_controlnet_with_dc.py
import os
import argparse
import random
from typing import List

import numpy as np
import pandas as pd
import scipy.io

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from diffusion.utils import set_seed
from diffusion.datasets import ThzDataset
import tqdm

# ---------------------------------------------------------
# Imports AUS DEM DIFFUSION-CLASSIFIER-REPO (so viel wie möglich)
# ---------------------------------------------------------
# Repo: https://github.com/diffusion-classifier/diffusion-classifier
from diffusion.models import get_sd_model, get_scheduler_config
try:
    # Für optionale Validierung (Eval-only)
    from eval_prob_adaptive import eval_error as repo_eval_error
    HAS_REPO_EVAL = True
except Exception:
    HAS_REPO_EVAL = False

device = "cuda" if torch.cuda.is_available() else "cpu"
# -------------------------
# 3D→2D Control-Adapter (leichtgewichtiges "ControlNet")
# -------------------------
class THZControlAdapter(nn.Module):
    """
    x: [B,1,T,H,W]  ->  y: [B,3,S,S] in [0,1]
    Idee: 3D-Encoder, mittelt T adaptiv, 2D-Refinement, 1x1-Head auf 3 Kanäle.
    """
    def __init__(self, out_hw=512, base_ch=128):
        super().__init__()
        self.enc3d = nn.Sequential(
            nn.Conv3d(1, 16, 3, padding=1), nn.SiLU(),
            nn.Conv3d(16, 32, 3, stride=(2,1,1), padding=1), nn.SiLU(),   # T↓
            nn.Conv3d(32, 64, 3, stride=(2,2,2), padding=1), nn.SiLU(),   # T,H,W↓
            nn.Conv3d(64, base_ch, 3, stride=(2,2,2), padding=1), nn.SiLU(),
        )
        self.poolT = nn.AdaptiveAvgPool3d((1, None, None))  # T -> 1
        self.refine2d = nn.Sequential(
            nn.Conv2d(base_ch, base_ch, 3, padding=1), nn.SiLU(),
            nn.Conv2d(base_ch, base_ch, 3, padding=1), nn.SiLU(),
        )
        self.head = nn.Conv2d(base_ch, 3, kernel_size=1)
        self.out_hw = out_hw

    def forward(self, x):
        # x: [B,1,T,H,W]
        h = self.enc3d(x)             # [B,C,T',H',W']
        h = self.poolT(h).squeeze(2)  # [B,C,H',W']
        if (h.shape[-2], h.shape[-1]) != (self.out_hw, self.out_hw):
            h = F.interpolate(h, size=(self.out_hw, self.out_hw), mode="bilinear", align_corners=False)
        h = self.refine2d(h)
        rgb = self.head(h)            # [B,3,S,S]
        return torch.sigmoid(rgb)     # [0,1]


# -------------------------
# Differenzierbare DC-Logits: -MSE(noise_pred, noise)
# -------------------------
def logits_from_dc_trainable(unet, vae, scheduler, c_rgb_01: torch.Tensor,
                             class_embeds: torch.Tensor, n_t: int = 2, dtype: str = "float16") -> torch.Tensor:
    """
    c_rgb_01: [B,3,S,S] ∈ [0,1]
    class_embeds: [C,77,hidden]
    Rückgabe: logits [B,C] = - mean_t MSE(noise_pred, noise)
    UNet/VAE sind eingefroren (requires_grad=False), aber wir rufen OHNE no_grad() auf,
    damit Grad bis c_rgb_01 (also den Adapter) fließen kann.
    """
    device = c_rgb_01.device
    B = c_rgb_01.size(0)
    C = class_embeds.size(0)

    # VAE-Encode (Grad bis Input erlaubt)
    x = c_rgb_01 * 2.0 - 1.0
    if dtype == "float16":
        x = x.half()
    z0 = vae.encode(x).latent_dist.mean * 0.18215            # [B,4,S/8,S/8]

    logits_accum = None
    alphas = scheduler.alphas_cumprod.to(z0.device, z0.dtype)

    for _ in range(n_t):
        t = torch.randint(0, scheduler.config.num_train_timesteps, (B,), device=device).long()
        eps = torch.randn_like(z0)
        alpha_bar = alphas[t].view(B,1,1,1)
        zt = alpha_bar.sqrt() * z0 + (1.0 - alpha_bar).sqrt() * eps

        # Klassen-parallel
        text_batch = class_embeds.unsqueeze(0).expand(B, -1, -1, -1).reshape(B*C, *class_embeds.shape[1:])
        zt_batch = zt.unsqueeze(1).expand(-1, C, -1, -1, -1).reshape(B*C, *zt.shape[1:])
        t_batch = t.unsqueeze(1).expand(-1, C).reshape(B*C)

        noise_pred = unet(zt_batch, t_batch, encoder_hidden_states=text_batch).sample  # [B*C,4,h,w]
        eps_tgt = eps.unsqueeze(1).expand(-1, C, -1, -1, -1).reshape(B*C, *eps.shape[1:])

        mse = F.mse_loss(noise_pred.float(), eps_tgt.float(), reduction='none').mean(dim=(1,2,3))  # [B*C]
        mse = mse.view(B, C)
        logits = -mse

        logits_accum = logits if logits_accum is None else (logits_accum + logits)

    return logits_accum / float(n_t)

def set_seed(seed: int = 42):
    random.seed(seed)                # für Python-Standardbibliothek (random)
    np.random.seed(seed)              # für NumPy
    torch.manual_seed(seed)           # für PyTorch CPU
    torch.cuda.manual_seed_all(seed)  # für PyTorch GPU (alle GPUs)

# -------------------------
# Optionale Validierung mit Original-Repo-Funktion (no_grad)
# -------------------------
@torch.no_grad()
def validate_with_repo_eval(unet, scheduler, vae, tokenizer, text_encoder,
                            class_prompts: List[str], c_rgb_01: torch.Tensor,
                            dtype: str = "float16") -> torch.Tensor:
    """
    c_rgb_01: [B,3,S,S] in [0,1]
    Rückgabe: logits [B,C] über repo_eval_error (Eval-only)
    """
    if not HAS_REPO_EVAL:
        raise RuntimeError("eval_error aus eval_prob_adaptive.py nicht verfügbar (Import fehlgeschlagen).")

    device = c_rgb_01.device
    # Text-Embeddings
    text_input = tokenizer(class_prompts, padding="max_length",
                           max_length=tokenizer.model_max_length,
                           truncation=True, return_tensors="pt")
    text_embeds = text_encoder(text_input.input_ids.to(device))[0]  # [C,77,hidden]
    if dtype == "float16":
        scheduler.alphas_cumprod = scheduler.alphas_cumprod.half()
        text_embeds = text_embeds.half()

    # VAE-Encode
    x = c_rgb_01 * 2.0 - 1.0
    if dtype == "float16":
        x = x.half()
    x0 = vae.encode(x).latent_dist.mean * 0.18215  # [B,4,64,64] (bei S=512)

    B = x0.size(0)
    C = text_embeds.size(0)
    logits = torch.zeros(B, C, device=device)

    # Sampleweise (einfach, klar)
    for b in range(B):
        latent = x0[b:b+1]
        T = int(scheduler.config.get("num_train_timesteps", 1000))
        n_samples = 8
        start = T // n_samples // 2
        t_to_eval = list(range(start, T, max(T // n_samples, 1)))[:n_samples]
        n_trials = 1
        all_noise = torch.randn((n_samples * n_trials, 4, latent.shape[-2], latent.shape[-1]), device=device)
        if dtype == "float16":
            all_noise = all_noise.half()

        ts, noise_idxs, text_idxs = [], [], []
        for c in range(C):
            for t_idx, t in enumerate(t_to_eval):
                ts.extend([t] * n_trials)
                base = t_idx * n_trials
                noise_idxs.extend(list(range(base, base + n_trials)))
                text_idxs.extend([c] * n_trials)

        errors = repo_eval_error(
            unet, scheduler, latent, all_noise,
            ts, noise_idxs, text_embeds, text_idxs,
            batch_size=32, dtype=dtype, loss='l2'
        ).to(device)  # [len(ts)]
        errors = errors.view(C, -1).mean(dim=1)
        logits[b] = -errors
    return logits


# -------------------------
# Training
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, required=True, help="Ordner mit .mat Dateien")
    ap.add_argument("--label_csv", type=str, required=True, help="CSV: filename,label")
    ap.add_argument("--out_dir", type=str, default="ckpts_thz_controlnet")
    ap.add_argument("--img_size", type=int, default=512)
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--max_steps", type=int, default=20000)
    ap.add_argument("--save_every", type=int, default=2000)
    ap.add_argument("--dtype", type=str, default="float16", choices=["float16","float32"])
    ap.add_argument("--n_t", type=int, default=2, help="zufällige Timesteps pro Forward (stabiler bei >1)")
    ap.add_argument("--val_every", type=int, default=1000, help="Eval-Schritt mit Repo-Funktion")
    ap.add_argument("--version", type=str, default="2-0", choices=["1-1", "1-2", "1-3", "1-4", "1-5", "2-0", "2-1"],
                    help="Stable Diffusion Version")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1) SD2-Base + Scheduler via Repo laden
    vae, tokenizer, text_encoder, unet, scheduler = get_sd_model(args)
    vae.to(device); text_encoder.to(device); unet.to(device)

    # einfrieren
    for m in [vae, text_encoder, unet]:
        m.eval()
        for p in m.parameters():
            p.requires_grad_(False)

    # 2) Klassen/Prompts aus CSV
    labels_df = pd.read_csv(args.label_csv)
    classes_sorted = sorted(labels_df['label'].unique())
    class_prompts = [f"class: {c}" for c in classes_sorted]
    print("Klassen:", classes_sorted)

    # Text-Embeddings EINMAL für Training
    with torch.no_grad():
        ti = tokenizer(class_prompts, padding="max_length",
                       max_length=tokenizer.model_max_length,
                       truncation=True, return_tensors="pt")
        class_embeds = text_encoder(ti.input_ids.to(device))[0]   # [C,77,hidden]
        if args.dtype == "float16":
            class_embeds = class_embeds.half()

    # 3) Dataset / Loader
    ds = ThzDataset(args.data_dir, args.label_csv)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                    num_workers=args.workers, pin_memory=True, drop_last=True)

    # 4) Control-Adapter
    adapter = THZControlAdapter(out_hw=args.img_size).to(device)
    opt = torch.optim.AdamW(adapter.parameters(), lr=args.lr, weight_decay=1e-4)
    scaler = torch.amp.GradScaler('cuda', enabled=(args.dtype == "float16"))

    global_step = 0
    while global_step < args.max_steps:
        for vol, y in dl:
            if global_step >= args.max_steps:
                break
            vol = vol.to(device, dtype=torch.float32)  # [B,1,T,H,W]
            y   = y.to(device, dtype=torch.long)

            with torch.amp.autocast(device_type='cuda',enabled=(args.dtype == "float16")):
                # 3D -> 2D Pseudo-RGB
                c_rgb = adapter(vol)  # [B,3,S,S] in [0,1]

                # Diffusion-Classifier-Logits (trainierbar)
                logits = logits_from_dc_trainable(
                    unet=unet, vae=vae, scheduler=scheduler,
                    c_rgb_01=c_rgb, class_embeds=class_embeds,
                    n_t=args.n_t, dtype=args.dtype
                )  # [B,C]

                loss = F.cross_entropy(logits, y)

            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            global_step += 1
            if global_step % 25 == 0:
                with torch.no_grad():
                    preds = logits.argmax(dim=1)
                    acc = (preds == y).float().mean().item()
                print(f"[{global_step:>6}] loss={loss.item():.4f}  acc={acc*100:.2f}%")

            if global_step % args.save_every == 0:
                torch.save(adapter.state_dict(), os.path.join(args.out_dir, f"adapter_step{global_step}.pt"))

            # optionale Validierung mit Original-Repo-Funktion
            if HAS_REPO_EVAL and args.val_every > 0 and global_step % args.val_every == 0:
                adapter.eval()
                with torch.no_grad():
                    c_rgb_eval = c_rgb.clamp(0,1)
                    logits_eval = validate_with_repo_eval(
                        unet=unet, scheduler=scheduler, vae=vae,
                        tokenizer=tokenizer, text_encoder=text_encoder,
                        class_prompts=class_prompts, c_rgb_01=c_rgb_eval,
                        dtype=args.dtype
                    )
                    preds = logits_eval.argmax(dim=1)
                    acc = (preds == y).float().mean().item()
                print(f"[VAL {global_step:>6}] repo_eval_error-acc={acc*100:.2f}%")
                adapter.train()

    torch.save(adapter.state_dict(), os.path.join(args.out_dir, "adapter_final.pt"))
    print("Fertig. Adapter gespeichert:", os.path.join(args.out_dir, "adapter_final.pt"))


if __name__ == "__main__":
    main()