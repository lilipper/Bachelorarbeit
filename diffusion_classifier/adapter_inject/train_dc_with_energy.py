# train_diffusion_classifier_cv_zsdc_hinge.py
# -*- coding: utf-8 -*-
"""
Zero-Shot Diffusion Classifier style training (adaptive prompt/timestep selection)
with original ControlNet injection and a max-margin (hinge) objective on class energies.

Differences vs your CE version:
- Keep eval_prob_adaptive_differentiable (adaptive selection like in ZSDC).
- Keep original ControlNet (residuals injected into UNet blocks).
- Replace CE with Hinge: L = mean( relu( margin + E_pos - E_best_neg ) ),
  where class energies E come from pooling prompt-level denoising errors.
"""

import os
import csv
import argparse
from datetime import datetime
from zoneinfo import ZoneInfo

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import numpy as np
from sklearn.model_selection import RepeatedStratifiedKFold
from tqdm.auto import tqdm

from transformers import logging as hf_logging
from diffusers import ControlNetModel

# --- Project modules you already have ---
from diffusion.datasets import ThzDataset
from thz_front_rgb_head import THzToRGBHead
from eval_prob_adaptive import eval_prob_adaptive_differentiable
from adapter.help_functions import (
    build_sd2_1_base, read_csv_pairs, write_csv_pairs, PromptBank,
    pool_prompt_errors_to_class_errors_batch
)

hf_logging.set_verbosity_error()
device = "cuda" if torch.cuda.is_available() else "cpu"


# --------------------- Utilities ---------------------
def set_seed(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class UnetWithControl(nn.Module):
    """
    Wrap frozen SD-UNet with a trainable ControlNetModel.
    Forward injects ControlNet residuals into the UNet (official, paper-correct path).
    """
    def __init__(self, unet, controlnet, cond_scale: float = 1.0):
        super().__init__()
        self.unet = unet
        self.controlnet = controlnet
        self.cond_scale = float(cond_scale)

    def forward(self, sample, timestep, encoder_hidden_states, controlnet_cond):
        # 1) ControlNet residuals
        cn_out = self.controlnet(
            sample=sample,
            timestep=timestep,
            encoder_hidden_states=encoder_hidden_states,
            controlnet_cond=controlnet_cond,
            conditioning_scale=self.cond_scale,
            return_dict=True,
        )
        # 2) UNet with injected residuals (official wiring)
        unet_out = self.unet(
            sample,
            timestep,
            encoder_hidden_states=encoder_hidden_states,
            down_block_additional_residuals=cn_out.down_block_res_samples,
            mid_block_additional_residual=cn_out.mid_block_res_sample,
            return_dict=True,
        )
        return unet_out  # .sample


def normalize_amp_and_scaler(dtype_str: str):
    """Return (torch_dtype, use_amp, scaler) based on CLI dtype."""
    torch_dtype = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[dtype_str]
    use_amp = (device == "cuda") and (dtype_str in ("float16", "bfloat16"))
    if dtype_str == "float16" and device == "cuda":
        scaler = torch.amp.GradScaler('cuda', enabled=True)
    else:
        scaler = torch.amp.GradScaler('cuda',enabled=False)
    return torch_dtype, use_amp, scaler


# ----------------- Train / Validate -----------------
def train_one_epoch(
    loader, front, unet_ctrl, vae, prompt_embeds, prompt_to_class, num_classes,
    scheduler, eargs, img_size, torch_dtype, use_amp, opt, scaler,
    margin: float = 0.5, normalize_rank: bool = False
):
    """
    One ZSDC-style training epoch:
      - THz -> RGB via trainable front
      - Encode to latents via VAE (frozen weights, but keep autograd to front)
      - eval_prob_adaptive_differentiable (adaptive, ZSDC-style) with a callable UNet that injects ControlNet residuals
      - Pool prompt errors -> class energies
      - Hinge loss on energies: L = mean( relu( margin + E_pos - E_best_neg ) )
    """
    front.train()
    unet_ctrl.controlnet.train()
    unet_ctrl.unet.eval()  # UNet frozen

    running_loss, running_acc, seen = 0.0, 0, 0
    latent_size = img_size // 8
    out_hw = (img_size, img_size)

    for step, (vol, label, _) in enumerate(tqdm(loader, desc="[train]", leave=False), start=1):
        vol = vol.to(device)
        if vol.dim() == 6:
            vol = vol.squeeze(1)  # safety for [B,1,2,T,H,W]
        label = label.to(device).long()

        opt.zero_grad(set_to_none=True)

        with torch.autocast(device_type="cuda", dtype=torch_dtype, enabled=use_amp):
            # THz front -> RGB [0,1]
            img_rgb = front(vol)  # [B,3,H,W]
            if img_rgb.shape[-2:] != out_hw:
                img_rgb = F.interpolate(img_rgb, size=out_hw, mode="bilinear", align_corners=False)

            # ControlNet expects [-1,1] image dtype matching UNet
            cond_rgb = (img_rgb * 2.0 - 1.0).to(dtype=unet_ctrl.unet.dtype)

            # VAE encode (keep graph)
            x_in = (img_rgb * 2.0 - 1.0).to(dtype=vae.dtype)
            lat = vae.encode(x_in).latent_dist.mean * 0.18215  # [B,4,h,w]

            # Adapter to eval_prob_adaptive_differentiable API
            class _CallableUnet(nn.Module):
                def __init__(self, unet_ctrl_ref, cond):
                    super().__init__()
                    self.ref = unet_ctrl_ref
                    self.cond = cond
                def forward(self, sample, timestep, encoder_hidden_states):
                    return self.ref(sample, timestep, encoder_hidden_states, controlnet_cond=self.cond)

            # Per-sample evaluation to avoid inner batch coupling
            B = lat.size(0)
            errors_list = []
            for i in range(B):
                callable_unet_i = _CallableUnet(unet_ctrl, cond_rgb[i:i+1])
                # ZSDC-style adaptive routine (your function):
                # returns: pred_idx, data, errors_per_prompt
                _, _, epp_i = eval_prob_adaptive_differentiable(
                    unet=callable_unet_i,
                    latent=lat[i:i+1],               # [1,4,h,w]
                    text_embeds=prompt_embeds,
                    scheduler=scheduler,
                    args=eargs,
                    latent_size=latent_size,
                    all_noise=None
                )
                if epp_i.dim() == 1:
                    epp_i = epp_i.unsqueeze(0)      # [1, P]
                errors_list.append(epp_i)

            # Prompt-level errors (lower is better) -> [B,P]
            errors_per_prompt = torch.cat(errors_list, dim=0)

            # Optional per-sample rank-only z-norm (detach stats so front can't game absolute scale)
            if normalize_rank:
                mean = errors_per_prompt.mean(dim=1, keepdim=True).detach()
                std  = errors_per_prompt.std(dim=1, keepdim=True).detach().clamp_min(1e-6)
                errors_per_prompt = (errors_per_prompt - mean) / std

            # Pool to class errors (we treat them as energies)
            class_errors_batch = pool_prompt_errors_to_class_errors_batch(
                errors_per_prompt, prompt_to_class, num_classes, reduce="mean"
            )  # [B, C]
            class_energy = class_errors_batch  # naming: lower = better

            # Hinge: margin + E_pos - E_best_neg
            mask = torch.ones_like(class_energy, dtype=torch.bool)
            mask.scatter_(1, label.view(-1,1), False)
            neg_best, _ = class_energy.masked_fill(~mask, float('inf')).min(dim=1)  # [B]
            pos = class_energy[torch.arange(B), label]  # [B]
            loss = F.relu(margin + pos - neg_best).mean()

        # Optimizer step (AMP-aware)
        if scaler.is_enabled():
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        else:
            loss.backward()
            opt.step()

        # Metrics
        with torch.no_grad():
            running_loss += loss.item() * vol.size(0)
            preds = torch.argmin(class_energy, dim=1)
            running_acc += (preds == label).sum().item()
            seen += vol.size(0)

    epoch_loss = running_loss / max(1, seen)
    epoch_acc = running_acc / max(1, seen)
    print(f"[train] epoch_loss={epoch_loss:.4f}  epoch_acc={epoch_acc:.4f}")
    return epoch_loss, epoch_acc


@torch.no_grad()
def validate(
    loader, front, unet_ctrl, vae, prompt_embeds, prompt_to_class, num_classes,
    scheduler, eargs, img_size, torch_dtype, use_amp, normalize_rank: bool = False
):
    """Validation loop in ZSDC style; prediction is argmin over class energies."""
    front.eval()
    unet_ctrl.controlnet.eval()
    unet_ctrl.unet.eval()

    latent_size = img_size // 8
    out_hw = (img_size, img_size)

    total, correct = 0, 0
    for step, (vol, label, _) in enumerate(tqdm(loader, desc="[val]", leave=False), start=1):
        vol = vol.to(device)
        if vol.dim() == 6:
            vol = vol.squeeze(1)
        label = label.to(device).long()

        with torch.autocast(device_type="cuda", dtype=torch_dtype, enabled=use_amp):
            img_rgb = front(vol)  # [B,3,H,W]
            if img_rgb.shape[-2:] != out_hw:
                img_rgb = F.interpolate(img_rgb, size=out_hw, mode="bilinear", align_corners=False)

            cond_rgb = (img_rgb * 2.0 - 1.0).to(dtype=unet_ctrl.unet.dtype)
            x_in = (img_rgb * 2.0 - 1.0).to(dtype=vae.dtype)
            lat = vae.encode(x_in).latent_dist.mean * 0.18215  # [B,4,h,w]

            class _CallableUnet(nn.Module):
                def __init__(self, unet_ctrl_ref, cond):
                    super().__init__()
                    self.ref = unet_ctrl_ref
                    self.cond = cond
                def forward(self, sample, timestep, encoder_hidden_states):
                    return self.ref(sample, timestep, encoder_hidden_states, controlnet_cond=self.cond)

            B = lat.size(0)
            errors_list = []
            for i in range(B):
                callable_unet_i = _CallableUnet(unet_ctrl, cond_rgb[i:i+1])
                _, _, epp_i = eval_prob_adaptive_differentiable(
                    unet=callable_unet_i,
                    latent=lat[i:i+1],
                    text_embeds=prompt_embeds,
                    scheduler=scheduler,
                    args=eargs,
                    latent_size=latent_size,
                    all_noise=None
                )
                if epp_i.dim() == 1:
                    epp_i = epp_i.unsqueeze(0)
                errors_list.append(epp_i)
            errors_per_prompt = torch.cat(errors_list, dim=0)  # [B, P]

            if normalize_rank:
                mean = errors_per_prompt.mean(dim=1, keepdim=True)
                std  = errors_per_prompt.std(dim=1, keepdim=True).clamp_min(1e-6)
                errors_per_prompt = (errors_per_prompt - mean) / std

            class_errors_batch = pool_prompt_errors_to_class_errors_batch(
                errors_per_prompt, prompt_to_class, num_classes, reduce="mean"
            )  # [B, C]
            preds = torch.argmin(class_errors_batch, dim=1)  # [B]

        correct += (preds == label).sum().item()
        total += label.numel()

    val_acc = correct / max(1, total)
    print(f"[val] acc={val_acc:.4f}")
    return val_acc


# ----------------------- Main -----------------------
def main():
    ap = argparse.ArgumentParser(description="ZSDC-style Diffusion Classifier (SD + ControlNet) with THz front, RSKF, Hinge objective, and Final Eval")

    # Data
    ap.add_argument("--data_train", type=str, required=True)
    ap.add_argument("--train_csv", type=str, required=True)
    ap.add_argument("--data_root_eval", type=str, default=None,
                    help="Optional root for validation/test (default: use data_train).")

    # Prompts / SD / Diffusion Classifier
    ap.add_argument("--prompts_csv", type=str, required=True, help="CSV: prompt,classname,classidx")
    ap.add_argument("--version", type=str, default="2-1", choices=("2-1","2-0","1-1","1-2","1-3","1-4","1-5"))
    ap.add_argument("--dtype", type=str, default="bfloat16", choices=("float16","float32","bfloat16"))
    ap.add_argument("--img_size", type=int, default=256, choices=(256,512))
    ap.add_argument("--num_train_timesteps", type=int, default=1000)
    ap.add_argument("--n_trials", type=int, default=1)
    ap.add_argument("--n_samples", nargs="+", type=int, required=True, help="per-round timesteps")
    ap.add_argument("--to_keep",   nargs="+", type=int, required=True, help="per-round prompt pruning")
    ap.add_argument("--loss", type=str, default="l2", choices=("l1","l2","huber"))
    ap.add_argument("--use_xformers", action="store_true")

    # Control / Front
    ap.add_argument("--cond_scale", type=float, default=1.0, help="conditioning_scale for ControlNet")
    ap.add_argument("--learn_front", action="store_true")
    ap.add_argument("--lr_front", type=float, default=1e-4)
    ap.add_argument("--wd_front", type=float, default=0.0)
    ap.add_argument("--lr_controlnet", type=float, default=5e-5)
    ap.add_argument("--wd_controlnet", type=float, default=0.0)

    # Hinge settings
    ap.add_argument("--margin", type=float, default=0.5)
    ap.add_argument("--normalize_rank", action="store_true")

    # Training
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--num_workers", type=int, default=4)

    # CV
    ap.add_argument("--cv_splits", type=int, default=5)
    ap.add_argument("--cv_repeats", type=int, default=2)
    ap.add_argument("--cv_seed", type=int, default=42)

    # Final evaluation
    ap.add_argument("--final_eval", action="store_true",
                    help="After CV: load global best and evaluate on a fixed set.")
    ap.add_argument("--data_test", type=str, default=None, help="Test root dir.")
    ap.add_argument("--test_csv", type=str, default=None, help="CSV (path,label) for test. If empty, fall back to --final_val_csv.")
    ap.add_argument("--val_csv", type=str, default=None, help="Fallback CSV if --test_csv missing.")

    # IO
    ap.add_argument("--save_dir", type=str, default="./runs/checkpoints_dc_zsdc_hinge")

    args = ap.parse_args()
    assert len(args.to_keep) == len(args.n_samples), "--to_keep and --n_samples must have same length (rounds)."

    # Print config
    print("========== CONFIG ==========")
    for k, v in vars(args).items():
        print(f"{k}: {v}")
    print("============================")

    set_seed(args.cv_seed)

    # AMP / dtype
    torch_dtype, use_amp, scaler = normalize_amp_and_scaler(args.dtype)
    print(f"[AMP] device={device}  dtype={args.dtype}  use_amp={use_amp}")

    torch.backends.cudnn.benchmark = False

    # Save root
    stamp = datetime.now(ZoneInfo("Europe/Berlin")).strftime("%y%m%d_%H%M")
    save_dir = os.path.join(args.save_dir, f"sd{args.version}_img{args.img_size}_{args.dtype}_{stamp}")
    os.makedirs(save_dir, exist_ok=True)
    print(f"[IO] Save directory: {save_dir}")

    # --- SD backbone (frozen VAE/UNet/TextEncoder, ZSDC-consistent) ---
    print("[SD] Building Stable Diffusion base...")
    vae, unet, tokenizer, text_encoder, scheduler = build_sd2_1_base(
        dtype=args.dtype, use_xformers=args.use_xformers, train_all=False, version=args.version
    )
    unet.eval()
    print("[SD] Done. UNet/VAE/TextEncoder are frozen.")

    # --- ControlNet from UNet (official path) ---
    print("[ControlNet] Creating ControlNet from UNet...")
    controlnet = ControlNetModel.from_unet(unet)
    controlnet = controlnet.to(device, dtype=unet.dtype)
    try:
        controlnet.enable_gradient_checkpointing()
        print("[ControlNet] Enabled gradient checkpointing.")
    except Exception:
        print("[ControlNet] Gradient checkpointing not available.")

    # --- THz->RGB front ---
    front = THzToRGBHead(in_ch=2, base_ch=32, k_t=5, final_depth=16).to(device)

    # --- Prompt bank & text embeddings ---
    print(f"[Prompts] Loading prompts from: {args.prompts_csv}")
    pb = PromptBank(args.prompts_csv)
    prompt_embeds = pb.to_text_embeds(tokenizer, text_encoder, device)
    prompt_to_class = pb.prompt_to_class.to(device)
    num_classes = pb.num_classes
    print(f"[Prompts] Loaded {len(pb.prompt_texts)} prompts over {num_classes} classes.")

    # --- EArgs for eval_prob_adaptive_differentiable (ZSDC rounds) ---
    class EArgs: pass
    eargs = EArgs()
    eargs.n_samples = args.n_samples
    eargs.to_keep = args.to_keep
    eargs.n_trials = args.n_trials
    eargs.batch_size = args.batch_size  # inner batching in your eval fn
    eargs.dtype = args.dtype
    eargs.loss = args.loss
    eargs.num_train_timesteps = args.num_train_timesteps
    eargs.version = args.version
    eargs.learn_front = args.learn_front

    # --- Read train rows and set up RSKF ---
    print(f"[IO] Reading training pairs from: {args.train_csv}")
    all_rows = read_csv_pairs(args.train_csv)
    labels = [y for _, y in all_rows]
    print(f"[IO] Found {len(all_rows)} samples.")

    rskf = RepeatedStratifiedKFold(
        n_splits=args.cv_splits, n_repeats=args.cv_repeats, random_state=args.cv_seed
    )
    total_splits = args.cv_splits * args.cv_repeats
    print(f"[CV] splits={args.cv_splits}, repeats={args.cv_repeats} -> total={total_splits}")

    summary_rows = []
    cv_scores = []
    best_global_acc = -1.0
    best_global_ckpt = None

    # Iterate folds
    for split_idx, (idx_tr, idx_va) in enumerate(rskf.split(all_rows, labels), start=1):
        print(f"\n===== [SPLIT {split_idx:03d}/{total_splits}] =====")
        fold_dir = os.path.join(save_dir, f"split_{split_idx:03d}")
        os.makedirs(fold_dir, exist_ok=True)
        print(f"[IO] Fold directory: {fold_dir}")

        train_rows = [all_rows[i] for i in idx_tr]
        val_rows   = [all_rows[i] for i in idx_va]
        fold_train_csv = os.path.join(fold_dir, "train.csv")
        fold_val_csv   = os.path.join(fold_dir, "val.csv")
        write_csv_pairs(fold_train_csv, train_rows)
        write_csv_pairs(fold_val_csv, val_rows)
        print(f"[IO] Wrote train.csv (n={len(train_rows)}) & val.csv (n={len(val_rows)})")

        # Datasets/loaders
        root_eval = args.data_root_eval if args.data_root_eval else args.data_train
        train_ds = ThzDataset(args.data_train, fold_train_csv, is_train=True)
        val_ds   = ThzDataset(root_eval, fold_val_csv, is_train=False)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                                  num_workers=args.num_workers, pin_memory=True, drop_last=True)
        val_loader   = DataLoader(val_ds, batch_size=1, shuffle=False,
                                  num_workers=args.num_workers, pin_memory=True)
        print(f"[Data] Train batches: ~{len(train_loader)} | Val samples: {len(val_loader)}")

        # Fresh ControlNet wrapper for the fold
        unet_ctrl = UnetWithControl(unet=unet, controlnet=controlnet, cond_scale=args.cond_scale).to(device)

        # Optimizer: ControlNet (+ optional THz front)
        param_groups = [{"params": controlnet.parameters(), "lr": args.lr_controlnet, "weight_decay": args.wd_controlnet}]
        if args.learn_front:
            param_groups.append({"params": front.parameters(), "lr": args.lr_front, "weight_decay": args.wd_front})
            print("[OPT] Front parameters will be trained.")
        else:
            for p in front.parameters():
                p.requires_grad_(False)
            front.eval()
            print("[OPT] Front is frozen.")

        opt = torch.optim.AdamW(param_groups)

        # Train epochs
        best_val = -1.0
        fold_ckpt = os.path.join(fold_dir, "best_dc.pt")
        for epoch in range(1, args.epochs + 1):
            print(f"\n[Epoch] Split {split_idx:03d} | Epoch {epoch:02d}/{args.epochs}")
            tr_loss, tr_acc = train_one_epoch(
                train_loader, front, unet_ctrl, vae,
                prompt_embeds, prompt_to_class, num_classes,
                scheduler, eargs, args.img_size, torch_dtype, use_amp, opt, scaler,
                margin=args.margin, normalize_rank=args.normalize_rank
            )
            val_acc = validate(
                val_loader, front, unet_ctrl, vae,
                prompt_embeds, prompt_to_class, num_classes,
                scheduler, eargs, args.img_size, torch_dtype, use_amp,
                normalize_rank=args.normalize_rank
            )
            print(f"[Epoch] train_loss={tr_loss:.4f}  train_acc={tr_acc:.4f}  val_acc={val_acc:.4f}")

            if val_acc >= best_val:
                best_val = val_acc
                torch.save({
                    "split_idx": split_idx,
                    "epoch": epoch,
                    "best_val_acc": best_val,
                    "version": args.version,
                    "img_size": args.img_size,
                    "dtype": args.dtype,
                    "cond_scale": args.cond_scale,
                    "controlnet_state_dict": controlnet.state_dict(),
                    "front_state_dict": front.state_dict(),
                    "prompts_csv": args.prompts_csv,
                    "n_samples": args.n_samples,
                    "to_keep": args.to_keep,
                    "num_train_timesteps": args.num_train_timesteps,
                    "loss": args.loss,
                    "margin": args.margin,
                    "normalize_rank": args.normalize_rank,
                }, fold_ckpt)
                print(f"[Checkpoint] Saved best fold checkpoint: {fold_ckpt}  (val_acc={best_val:.4f})")

        # Collect fold stats
        cv_scores.append(best_val)
        summary_rows.append({"split_idx": split_idx, "best_val_acc": best_val, "fold_dir": fold_dir})
        print(f"[Split] Best val_acc for split {split_idx:03d}: {best_val:.4f}")

        # Track global best
        if best_val > best_global_acc:
            best_global_acc = best_val
            best_global_ckpt = fold_ckpt
            print(f"[Global] New global best: acc={best_global_acc:.4f}  ckpt={best_global_ckpt}")

    # Save CV summary
    summary_csv = os.path.join(save_dir, "cv_summary.csv")
    with open(summary_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["split_idx", "best_val_acc", "fold_dir"])
        writer.writeheader()
        for r in summary_rows:
            writer.writerow(r)

    print("\n========== CV SUMMARY ==========")
    for r in summary_rows:
        print(f"split={r['split_idx']:03d}  best_val_acc={r['best_val_acc']:.4f}  dir={r['fold_dir']}")
    mean_acc = float(np.mean(cv_scores)) if len(cv_scores) else 0.0
    std_acc = float(np.std(cv_scores)) if len(cv_scores) else 0.0
    print(f"CV mean acc = {mean_acc:.4f}  ± {std_acc:.4f}")
    print(f"Global best acc = {best_global_acc:.4f}")
    if best_global_ckpt:
        print(f"Global best ckpt: {best_global_ckpt}")
    print(f"Summary CSV: {summary_csv}")
    print("================================\n")

    # ------------- FINAL EVAL (optional) -------------
    if args.final_eval:
        test_csv = args.test_csv if args.test_csv is not None else args.final_val_csv
        assert args.data_test and test_csv, "--final_eval requires --data_test and (--test_csv or --final_val_csv)"
        print("[FINAL] Starting final evaluation...")
        print(f"[FINAL] Loading global best checkpoint: {best_global_ckpt}")
        ckpt = torch.load(best_global_ckpt, map_location="cpu")
        print(f"[FINAL] Global best val_acc (from CV): {ckpt.get('best_val_acc','N/A')}")

        # Rebuild SD base (frozen) exactly as during training
        print("[FINAL] Rebuilding SD base...")
        vae_f, unet_f, tokenizer_f, text_encoder_f, scheduler_f = build_sd2_1_base(
            dtype=ckpt["dtype"], use_xformers=False, train_all=False, version=ckpt["version"]
        )
        unet_f.eval()

        # Rebuild ControlNet and THz front, load weights
        print("[FINAL] Rebuilding ControlNet + Front and loading weights...")
        controlnet_f = ControlNetModel.from_unet(unet_f).to(device, dtype=unet_f.dtype)
        controlnet_f.load_state_dict(ckpt["controlnet_state_dict"], strict=False)
        controlnet_f.eval()

        front_f = THzToRGBHead(in_ch=2, base_ch=32, k_t=5, final_depth=16).to(device)
        front_f.load_state_dict(ckpt["front_state_dict"], strict=False)
        front_f.eval()

        # Wrap UNet+Control
        unet_ctrl_f = UnetWithControl(unet=unet_f, controlnet=controlnet_f, cond_scale=ckpt.get("cond_scale", 1.0)).to(device)

        # Load prompts again to get embeddings
        print(f"[FINAL] Reloading prompts from: {ckpt['prompts_csv']}")
        pb_f = PromptBank(ckpt["prompts_csv"])
        prompt_embeds_f = pb_f.to_text_embeds(tokenizer_f, text_encoder_f, device)
        prompt_to_class_f = pb_f.prompt_to_class.to(device)
        num_classes_f = pb_f.num_classes

        # Build test loader
        final_root = args.data_test
        final_ds = ThzDataset(final_root, test_csv, is_train=False)
        final_loader = DataLoader(final_ds, batch_size=1, shuffle=False,
                                  num_workers=args.num_workers, pin_memory=True)
        print(f"[FINAL] Test set: {len(final_loader)} samples | CSV={test_csv} | ROOT={final_root}")

        # AMP for final eval
        use_amp_final = (device == "cuda") and (ckpt["dtype"] in ("float16", "bfloat16"))
        final_torch_dtype = torch.float16 if ckpt["dtype"] == "float16" else (
            torch.bfloat16 if ckpt["dtype"] == "bfloat16" else torch.float32
        )
        print(f"[FINAL] AMP enabled={use_amp_final} dtype={ckpt['dtype']}  img_size={ckpt['img_size']}")

        # Rebuild eargs
        class FE: pass
        eargs_f = FE()
        eargs_f.n_samples = ckpt["n_samples"]
        eargs_f.to_keep   = ckpt["to_keep"]
        eargs_f.n_trials  = args.n_trials  # can be 1
        eargs_f.batch_size= 1
        eargs_f.dtype     = ckpt["dtype"]
        eargs_f.loss      = ckpt["loss"]
        eargs_f.num_train_timesteps = ckpt["num_train_timesteps"]
        eargs_f.version   = ckpt["version"]
        eargs_f.learn_front = False

        # Run validate() on the fixed test set
        final_acc = validate(
            final_loader, front_f, unet_ctrl_f, vae_f,
            prompt_embeds_f, prompt_to_class_f, num_classes_f,
            scheduler_f, eargs_f, ckpt["img_size"],
            torch_dtype=final_torch_dtype, use_amp=use_amp_final, normalize_rank=ckpt.get("normalize_rank", False)
        )
        print(f"[FINAL] accuracy on fixed set ({test_csv} @ {final_root}): {final_acc:.4f}")


if __name__ == "__main__":
    main()
