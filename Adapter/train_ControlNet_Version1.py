#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Train ControlNet Adapter (3D THz volume -> pseudo-RGB) against a frozen Stable Diffusion 2-base
Diffusion Classifier objective (supervised).

What it does:
- Dataset loads .mat files, runs your preprocessing (Hamming + FFT + |.|^2 / max + flip)
- Adapter (your ControlNet) maps [B,1,T,H,W] -> [B,3,H0,W0]
- Same preproc as eval script (resize to img_size, normalize to [-1,1])
- VAE encode -> latent x0
- Sample timestep t + noise ε; UNet predicts ε_hat for each class text embedding
- Scores = -MSE(ε, ε_hat); CrossEntropy over scores vs. ground-truth class
- Train only the adapter; SD2-base (VAE/UNet/TextEncoder) remain frozen

Run:
python train_controlnet_adapter.py \
  --train_dir /path/to/train \
  --train_csv /path/to/train_labels.csv \
  --val_dir /path/to/val \
  --val_csv /path/to/val_labels.csv \
  --prompts_csv /path/to/prompts.csv \
  --out_ckpt /path/to/adapter_best.pth \
  --batch_size 2 --epochs 10 --lr 1e-4 --img_size 512 --dtype float16

Expected prompts.csv:
    prompt,classidx
    "class A text",0
    "class B text",1
"""

import os
import os.path as osp
import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler

# --------------------
# EDIT: import-pfade anpassen!
# - prdf muss read_mat(), process_complex_data() enthalten (deine Funktionen)
# - controlnet_adapter muss ControlNet-Klasse enthalten (dein Code)
# --------------------
from THz import process_rdf as prdf                          # <- deine Datei mit read_mat / process_complex_data
from Bachelorarbeit.Adapter.VCM.models import ControlNet   # <- deine ControlNet Klasse (Adapter)

# --------------------
# Utils
# --------------------
def set_seed(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# --------------------
# Dataset: gibt Volumen-Tensor [1,1,T,H,W], label
# --------------------
class ThzDataset(Dataset):
    def __init__(self, data_dir: str, label_csv: str):
        self.data_dir = data_dir
        self.labels_df = pd.read_csv(label_csv)
        # erwartet Spalten: filename,label  (label als string oder int)
        # map labels -> indices
        labels = sorted(self.labels_df['label'].unique())
        self.class_to_idx = {lab: i for i, lab in enumerate(labels)}
        self.file_to_class = {row.filename: self.class_to_idx[row.label] for row in self.labels_df.itertuples()}
        self.files = list(self.file_to_class.keys())

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        filename = self.files[idx]
        filepath = osp.join(self.data_dir, filename)
        label = int(self.file_to_class[filename])

        # 1) Volumen laden (complex): [T,H,W]
        complex_raw, params = prdf.read_mat(filepath, device="cpu")

        # 2) Verarbeitung (Hamming, FFT, shift)
        T = int(params["NF"])
        processed, max_val = prdf.process_complex_data(complex_raw, T, device="cpu")  # [T,H,W] complex

        # 3) |·|^2 / max -> [0,1]
        vol = torch.abs(processed)**2
        vol = vol / (max_val + 1e-12)  # [T,H,W], float32

        # 4) flipud entlang H (dim=1)
        vol = torch.flip(vol, dims=[1])  # [T,H,W]

        # 5) Form für Adapter: [B,C,T,H,W] = [1,1,T,H,W]
        vol = vol.unsqueeze(0).unsqueeze(0).contiguous().float()  # [1,1,T,H,W]

        return vol, label


# --------------------
# SD2-base Laden (frozen)
# --------------------
def load_sd(dtype: str, device: str):
    if dtype == 'float32':
        torch_dtype = torch.float32
    elif dtype == 'float16':
        torch_dtype = torch.float16
    else:
        raise ValueError("dtype must be 'float32' or 'float16'.")

    model_id = "stabilityai/stable-diffusion-2-base"
    scheduler = EulerDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        scheduler=scheduler,
        torch_dtype=torch_dtype,
        safety_checker=None,
        feature_extractor=None,
    ).to(device)

    try:
        pipe.enable_xformers_memory_efficient_attention()
    except Exception:
        pass

    vae = pipe.vae.to(device).eval()
    text_encoder = pipe.text_encoder.to(device).eval()
    tokenizer = pipe.tokenizer
    unet = pipe.unet.to(device).eval()

    # freeze
    for m in (vae, text_encoder, unet):
        for p in m.parameters():
            p.requires_grad = False

    return vae, tokenizer, text_encoder, unet, scheduler, torch_dtype


# --------------------
# Prompts -> Text Embeddings (einmalig)
# --------------------
def get_text_embeds(tokenizer, text_encoder, prompts_csv: str, device: str):
    df = pd.read_csv(prompts_csv)
    # Muss Spalten 'prompt' und 'classidx' enthalten (0..K-1)
    df = df.sort_values('classidx')
    prompts = df['prompt'].tolist()

    toks = tokenizer(
        prompts,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    with torch.no_grad():
        embeds = text_encoder(toks.input_ids.to(device))[0]  # [K, L, D]
    return embeds  # [K, L, D]


# --------------------
# Preproc wie im Eval (Resize + Normalize zu [-1,1])
# --------------------
def preprocess_rgb_like_eval(pseudo_rgb: torch.Tensor, size: int) -> torch.Tensor:
    # pseudo_rgb: [B,3,H0,W0], Werte ~[0,1]
    x = pseudo_rgb.clamp(0, 1)
    x = F.interpolate(x, size=(size, size), mode='bilinear', align_corners=False)
    # Normalize([0.5],[0.5]) -> [-1,1]
    x = (x - 0.5) / 0.5
    return x


# --------------------
# Ein Trainings-Schritt (Diffusion Classifier Loss)
# --------------------
def dc_step(adapter: ControlNet,
           vae, unet, scheduler,
           text_embeds: torch.Tensor,
           vol_5d: torch.Tensor,
           labels: torch.Tensor,
           img_size: int,
           use_amp: bool,
           torch_dtype: torch.dtype):
    """
    vol_5d: [B,1,T,H,W] float32
    labels: [B] long
    returns: (loss, scores[B,K]) and a detached preview image (for optional logging)
    """
    device = vol_5d.device
    B = vol_5d.size(0)
    K = text_embeds.size(0)

    # 1) Adapter -> Pseudo-RGB
    I = adapter(x=vol_5d, timesteps=None, controlnet_cond=vol_5d)  # [B,3,H0,W0]

    # 2) Gleiches Preproc wie Evals: Resize + Normalize [-1,1]
    I = preprocess_rgb_like_eval(I, size=img_size)

    if torch_dtype == torch.float16:
        I = I.half()

    # 3) VAE encode -> latent x0
    with torch.no_grad():
        x0 = vae.encode(I).latent_dist.mean * 0.18215  # [B,4,S,S]
    # Timesteps + noise
    T_steps = scheduler.config.get("num_train_timesteps", 1000)
    # (spätere timesteps oft stabiler)
    t = torch.randint(low=T_steps // 4, high=T_steps, size=(B,), device=device, dtype=torch.long)
    noise = torch.randn_like(x0)

    # x_t = sqrt(a)*x0 + sqrt(1-a)*noise
    alphas_cumprod = scheduler.alphas_cumprod.to(device)
    a = alphas_cumprod[t].view(B, 1, 1, 1)
    x_t = x0 * a.sqrt() + noise * (1 - a).sqrt()

    # 4) Scores je Klasse: -MSE(noise, eps_hat)
    scores_list = []
    # autocast nur für UNet vorwärts, Adapter wurde schon berechnet
    autocast_ctx = torch.autocast(device_type=device.split(':')[0],
                                  enabled=use_amp and (torch_dtype == torch.float16))
    with autocast_ctx:
        for k in range(K):
            # expand text embedding auf batch
            txt = text_embeds[k:k+1].expand(B, -1, -1)  # [B,L,D]
            eps_hat = unet(x_t, t, encoder_hidden_states=txt).sample  # [B,4,S,S]
            err = F.mse_loss(noise, eps_hat, reduction='none').mean(dim=(1, 2, 3))  # [B]
            scores_list.append(-err)
    scores = torch.stack(scores_list, dim=1)  # [B,K]

    # 5) CE über Scores
    loss = F.cross_entropy(scores, labels)
    return loss, scores.detach(), I.detach()


# --------------------
# Train Loop
# --------------------
def train(args):
    set_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    vae, tokenizer, text_encoder, unet, scheduler, torch_dtype = load_sd(args.dtype, device)

    # Text Embeddings (K Klassen)
    text_embeds = get_text_embeds(tokenizer, text_encoder, args.prompts_csv, device)

    # Datasets / Loader
    train_ds = ThzDataset(args.train_dir, args.train_csv)
    val_ds   = ThzDataset(args.val_dir, args.val_csv)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=True)

    # Adapter (ControlNet) – NUR dieser wird trainiert
    adapter = ControlNet(
        spatial_dims=3,
        in_channels=1,
        num_channels=(32, 64, 64, 64),
        attention_levels=(False, False, True, True),
        conditioning_embedding_in_channels=1,
        conditioning_embedding_num_channels=(16, 32, 96, 256),
        with_conditioning=False,
    ).to(device)
    adapter.train()

    optim = torch.optim.AdamW(adapter.parameters(), lr=args.lr, weight_decay=1e-2)
    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype == 'float16'))

    best_val_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        # ---------------- Train ----------------
        adapter.train()
        running = 0.0
        ntrain = 0
        for vol, y in train_loader:
            vol = vol.to(device, non_blocking=True)   # [B,1,T,H,W]
            y = y.to(device, non_blocking=True)       # [B]

            optim.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(args.dtype == 'float16')):
                loss, _, _ = dc_step(adapter, vae, unet, scheduler, text_embeds,
                                     vol, y, img_size=args.img_size,
                                     use_amp=(args.dtype == 'float16'),
                                     torch_dtype=torch_dtype)
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(adapter.parameters(), max_norm=1.0)
            scaler.step(optim)
            scaler.update()

            running += loss.item() * y.size(0)
            ntrain += y.size(0)

        train_loss = running / max(1, ntrain)

        # ---------------- Validate ----------------
        adapter.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for vol, y in val_loader:
                vol = vol.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                # wir wollen nur scores -> argmax
                _, scores, _ = dc_step(adapter, vae, unet, scheduler, text_embeds,
                                       vol, y, img_size=args.img_size,
                                       use_amp=False, torch_dtype=torch_dtype)
                pred = scores.argmax(dim=1)
                correct += (pred == y).sum().item()
                total += y.size(0)
        val_acc = correct / max(1, total)

        print(f"[Epoch {epoch:03d}] train_loss={train_loss:.4f} | val_acc={val_acc:.4f}")

        # Save best
        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            os.makedirs(osp.dirname(args.out_ckpt), exist_ok=True)
            torch.save(adapter.state_dict(), args.out_ckpt)
            print(f"  ✔ saved best adapter to {args.out_ckpt} (val_acc={val_acc:.4f})")

    print(f"Training done. Best val_acc = {best_val_acc:.4f}")
    return adapter


# --------------------
# CLI
# --------------------
def parse_args():
    p = argparse.ArgumentParser("Train ControlNet Adapter against frozen SD2-base (Diffusion Classifier loss)")
    # data
    p.add_argument("--train_dir", type=str, required=True)
    p.add_argument("--train_csv", type=str, required=True)
    p.add_argument("--val_dir",   type=str, required=True)
    p.add_argument("--val_csv",   type=str, required=True)
    p.add_argument("--prompts_csv", type=str, required=True,
                   help="CSV with columns: prompt,classidx; classidx from 0..K-1")
    # run
    p.add_argument("--out_ckpt", type=str, required=True, help="Path to save best adapter .pth")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--img_size", type=int, default=512, choices=(256,512))
    p.add_argument("--dtype", type=str, default="float16", choices=("float32","float16"))
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    train(args)


if __name__ == "__main__":
    main()
