# -*- coding: utf-8 -*-
"""
ZSDC Diffusion Classifier (Stable Diffusion UNet + ControlNet) with Hinge loss,
USING RAW THz AS CONTROL INPUT — no front, no RGB mapping, no VAE encode.

- Dataset must yield raw tensors: thz: [B, C_thz, H, W] (values as you prefer; passed through AS-IS)
- ControlNet conditioning_channels is set to C_thz
- Latents are sampled as Gaussian noise with spatial size img_size//8 (no image encode)
- Hinge loss on class energies from adaptive prompt/timestep selection (ZSDC-style)

Everything else (RSKF CV, checkpointing, final eval structure, prints) unchanged.
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

# --- project modules you already have ---
from diffusion.datasets import ThzDataset
from eval_prob_adaptive import eval_prob_adaptive_differentiable
from adapter.help_functions import (
    build_sd2_1_base, read_csv_pairs, write_csv_pairs, PromptBank,
    pool_prompt_errors_to_class_errors_batch
)

hf_logging.set_verbosity_error()
device = "cuda" if torch.cuda.is_available() else "cpu"


def set_seed(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class UnetWithControl(nn.Module):
    """Wrap frozen SD-UNet with a trainable ControlNetModel; inject residuals in the official path."""
    def __init__(self, unet, controlnet, cond_scale: float = 1.0):
        super().__init__()
        self.unet = unet
        self.controlnet = controlnet
        self.cond_scale = float(cond_scale)

    def forward(self, sample, timestep, encoder_hidden_states, controlnet_cond):
        cn_out = self.controlnet(
            sample=sample,
            timestep=timestep,
            encoder_hidden_states=encoder_hidden_states,
            controlnet_cond=controlnet_cond,               # <- RAW THz directly
            conditioning_scale=self.cond_scale,
            return_dict=True,
        )
        unet_out = self.unet(
            sample,
            timestep,
            encoder_hidden_states=encoder_hidden_states,
            down_block_additional_residuals=cn_out.down_block_res_samples,
            mid_block_additional_residual=cn_out.mid_block_res_sample,
            return_dict=True,
        )
        return unet_out


def normalize_amp_and_scaler(dtype_str: str):
    torch_dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[dtype_str]
    use_amp = (device == "cuda") and (dtype_str in ("float16", "bfloat16"))
    scaler = torch.amp.GradScaler('cuda', enabled=(use_amp and dtype_str == "float16"))
    return torch_dtype, use_amp, scaler


# ----------------- Train / Validate -----------------
def train_one_epoch(
    loader, unet_ctrl, prompt_embeds, prompt_to_class, num_classes,
    scheduler, eargs, img_size, torch_dtype, use_amp, opt, scaler,
    margin: float = 0.5, normalize_rank: bool = False
):
    """
    ZSDC training without any front or encode:
      - thz: [B, C_thz, H, W] -> passed AS-IS into ControlNet as controlnet_cond
      - latent is Gaussian N(0,1) with shape [B,4,img_size//8,img_size//8]
      - eval_prob_adaptive_differentiable -> prompt errors -> class energies
      - Hinge: mean( relu( margin + E_pos - E_best_neg ) )
    """
    unet_ctrl.controlnet.train()
    unet_ctrl.unet.eval()

    running_loss, running_acc, seen = 0.0, 0, 0
    h8 = img_size // 8
    out_hw = (img_size, img_size)

    for step, (thz, label, _) in enumerate(tqdm(loader, desc="[train]", leave=False), start=1):
        thz = thz.to(device)                  # [B, C_thz, H, W] — NO scaling enforced
        label = label.to(device).long()

        # optional spatial resize only if needed (pure geometric, no filtering)
        if thz.shape[-2:] != out_hw:
            thz = F.interpolate(thz, size=out_hw, mode="bilinear", align_corners=False)

        # build latent noise
        B = thz.size(0)
        latent = torch.randn(B, 4, h8, h8, device=device, dtype=unet_ctrl.unet.dtype)

        opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch_dtype, enabled=use_amp):
            class _CallableUnet(nn.Module):
                def __init__(self, unet_ctrl_ref, cond):
                    super().__init__()
                    self.ref = unet_ctrl_ref
                    self.cond = cond
                def forward(self, sample, timestep, encoder_hidden_states):
                    return self.ref(sample, timestep, encoder_hidden_states, controlnet_cond=self.cond)

            errors_list = []
            for i in range(B):
                # pass RAW thz slice directly (dtype aligned to UNet)
                cond_i = thz[i:i+1].to(dtype=unet_ctrl.unet.dtype)
                callable_unet_i = _CallableUnet(unet_ctrl, cond_i)
                _, _, epp_i = eval_prob_adaptive_differentiable(
                    unet=callable_unet_i,
                    latent=latent[i:i+1],                  # [1,4,h8,w8] noise
                    text_embeds=prompt_embeds,
                    scheduler=scheduler,
                    args=eargs,
                    latent_size=h8,
                    all_noise=None
                )
                if epp_i.dim() == 1:
                    epp_i = epp_i.unsqueeze(0)
                errors_list.append(epp_i)

            errors_per_prompt = torch.cat(errors_list, dim=0)  # [B, P]

            if normalize_rank:
                mean = errors_per_prompt.mean(dim=1, keepdim=True).detach()
                std  = errors_per_prompt.std(dim=1, keepdim=True).detach().clamp_min(1e-6)
                errors_per_prompt = (errors_per_prompt - mean) / std

            class_errors_batch = pool_prompt_errors_to_class_errors_batch(
                errors_per_prompt, prompt_to_class, num_classes, reduce="mean"
            )  # [B, C]
            class_energy = class_errors_batch

            # Hinge loss
            mask = torch.ones_like(class_energy, dtype=torch.bool)
            mask.scatter_(1, label.view(-1,1), False)
            neg_best, _ = class_energy.masked_fill(~mask, float('inf')).min(dim=1)
            pos = class_energy[torch.arange(B, device=device), label]
            loss = F.relu(margin + pos - neg_best).mean()

        if scaler.is_enabled():
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        else:
            loss.backward()
            opt.step()

        with torch.no_grad():
            running_loss += loss.item() * B
            preds = torch.argmin(class_energy, dim=1)
            running_acc += (preds == label).sum().item()
            seen += B

    epoch_loss = running_loss / max(1, seen)
    epoch_acc  = running_acc / max(1, seen)
    print(f"[train] epoch_loss={epoch_loss:.4f}  epoch_acc={epoch_acc:.4f}")
    return epoch_loss, epoch_acc


@torch.no_grad()
def validate(
    loader, unet_ctrl, prompt_embeds, prompt_to_class, num_classes,
    scheduler, eargs, img_size, torch_dtype, use_amp, normalize_rank: bool = False
):
    unet_ctrl.controlnet.eval()
    unet_ctrl.unet.eval()

    h8 = img_size // 8
    out_hw = (img_size, img_size)

    total, correct = 0, 0
    for step, (thz, label, _) in enumerate(tqdm(loader, desc="[val]", leave=False), start=1):
        thz = thz.to(device)
        label = label.to(device).long()

        if thz.shape[-2:] != out_hw:
            thz = F.interpolate(thz, size=out_hw, mode="bilinear", align_corners=False)

        B = thz.size(0)
        latent = torch.randn(B, 4, h8, h8, device=device, dtype=unet_ctrl.unet.dtype)

        with torch.autocast(device_type="cuda", dtype=torch_dtype, enabled=use_amp):
            class _CallableUnet(nn.Module):
                def __init__(self, unet_ctrl_ref, cond):
                    super().__init__()
                    self.ref = unet_ctrl_ref
                    self.cond = cond
                def forward(self, sample, timestep, encoder_hidden_states):
                    return self.ref(sample, timestep, encoder_hidden_states, controlnet_cond=self.cond)

            errors_list = []
            for i in range(B):
                cond_i = thz[i:i+1].to(dtype=unet_ctrl.unet.dtype)
                callable_unet_i = _CallableUnet(unet_ctrl, cond_i)
                _, _, epp_i = eval_prob_adaptive_differentiable(
                    unet=callable_unet_i,
                    latent=latent[i:i+1],
                    text_embeds=prompt_embeds,
                    scheduler=scheduler,
                    args=eargs,
                    latent_size=h8,
                    all_noise=None
                )
                if epp_i.dim() == 1:
                    epp_i = epp_i.unsqueeze(0)
                errors_list.append(epp_i)

            errors_per_prompt = torch.cat(errors_list, dim=0)

            if normalize_rank:
                mean = errors_per_prompt.mean(dim=1, keepdim=True)
                std  = errors_per_prompt.std(dim=1, keepdim=True).clamp_min(1e-6)
                errors_per_prompt = (errors_per_prompt - mean) / std

            class_errors_batch = pool_prompt_errors_to_class_errors_batch(
                errors_per_prompt, prompt_to_class, num_classes, reduce="mean"
            )
            preds = torch.argmin(class_errors_batch, dim=1)

        correct += (preds == label).sum().item()
        total   += label.numel()

    val_acc = correct / max(1, total)
    print(f"[val] acc={val_acc:.4f}")
    return val_acc


# ----------------------- Main -----------------------
def main():
    ap = argparse.ArgumentParser(description="ZSDC Diffusion Classifier — RAW THz into ControlNet (no front/encode), Hinge objective, CV/RSKF, Final Eval")

    # Data
    ap.add_argument("--data_train", type=str, required=True)
    ap.add_argument("--train_csv", type=str, required=True)
    ap.add_argument("--data_root_eval", type=str, default=None)

    # Prompts / SD
    ap.add_argument("--prompts_csv", type=str, required=True)
    ap.add_argument("--version", type=str, default="2-1", choices=("2-1","2-0","1-1","1-2","1-3","1-4","1-5"))
    ap.add_argument("--dtype", type=str, default="bfloat16", choices=("float16","float32","bfloat16"))
    ap.add_argument("--img_size", type=int, default=256, choices=(256,512))
    ap.add_argument("--num_train_timesteps", type=int, default=1000)
    ap.add_argument("--n_trials", type=int, default=1)
    ap.add_argument("--n_samples", nargs="+", type=int, required=True)
    ap.add_argument("--to_keep",   nargs="+", type=int, required=True)
    ap.add_argument("--loss", type=str, default="l2", choices=("l1","l2","huber"))
    ap.add_argument("--use_xformers", action="store_true")

    # Control / input channels
    ap.add_argument("--cond_scale", type=float, default=1.0)
    ap.add_argument("--thz_channels", type=int, required=True, help="C_thz: number of THz channels the dataset outputs")

    # Hinge
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

    # Final eval
    ap.add_argument("--final_eval", action="store_true")
    ap.add_argument("--data_test", type=str, default=None)
    ap.add_argument("--test_csv", type=str, default=None)
    ap.add_argument("--val_csv", type=str, default=None)

    # IO
    ap.add_argument("--save_dir", type=str, default="./runs/checkpoints_dc_zsdc_hinge_raw_thz")

    args = ap.parse_args()
    assert len(args.to_keep) == len(args.n_samples), "--to_keep and --n_samples must have same length."

    print("========== CONFIG ==========")
    for k, v in vars(args).items():
        print(f"{k}: {v}")
    print("============================")

    set_seed(args.cv_seed)
    torch_dtype, use_amp, scaler = normalize_amp_and_scaler(args.dtype)
    print(f"[AMP] device={device}  dtype={args.dtype}  use_amp={use_amp}")

    torch.backends.cudnn.benchmark = False

    stamp = datetime.now(ZoneInfo("Europe/Berlin")).strftime("%y%m%d_%H%M")
    save_dir = os.path.join(args.save_dir, f"sd{args.version}_img{args.img_size}_{args.dtype}_{stamp}")
    os.makedirs(save_dir, exist_ok=True)
    print(f"[IO] Save directory: {save_dir}")

    # SD base (frozen)
    print("[SD] Building Stable Diffusion base...")
    vae, unet, tokenizer, text_encoder, scheduler = build_sd2_1_base(
        dtype=args.dtype, use_xformers=args.use_xformers, train_all=False, version=args.version
    )
    unet.eval()
    print("[SD] Done. UNet/VAE/TextEncoder are frozen.")

    # ControlNet with conditioning_channels = thz_channels
    print(f"[ControlNet] Creating ControlNet with conditioning_channels={args.thz_channels} ...")
    try:
        # Preferred path if your diffusers supports it:
        controlnet = ControlNetModel.from_unet(unet, conditioning_channels=args.thz_channels)
    except TypeError:
        # Fallback: patch input conv if from_unet lacks the argument
        controlnet = ControlNetModel.from_unet(unet)
        # Try to patch first conv to accept args.thz_channels
        if hasattr(controlnet, "controlnet_cond_embedding") and hasattr(controlnet.controlnet_cond_embedding, "conv_in"):
            old_conv = controlnet.controlnet_cond_embedding.conv_in
            controlnet.controlnet_cond_embedding.conv_in = nn.Conv2d(
                args.thz_channels, old_conv.out_channels, kernel_size=old_conv.kernel_size,
                stride=old_conv.stride, padding=old_conv.padding, bias=old_conv.bias is not None
            )
            controlnet.config.conditioning_channels = args.thz_channels
            print("[ControlNet] Patched cond embedding conv_in to THz channels.")
        else:
            raise RuntimeError("Could not set conditioning_channels to THz channels for ControlNet.")
    controlnet = controlnet.to(device, dtype=unet.dtype)
    try:
        controlnet.enable_gradient_checkpointing()
        print("[ControlNet] Enabled gradient checkpointing.")
    except Exception:
        print("[ControlNet] Gradient checkpointing not available.")

    # Prompts / text embeds
    print(f"[Prompts] Loading prompts from: {args.prompts_csv}")
    pb = PromptBank(args.prompts_csv)
    prompt_embeds = pb.to_text_embeds(tokenizer, text_encoder, device)
    prompt_to_class = pb.prompt_to_class.to(device)
    num_classes = pb.num_classes
    print(f"[Prompts] Loaded {len(pb.prompt_texts)} prompts over {num_classes} classes.")

    # EArgs for ZSDC rounds
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
    eargs.learn_front = False

    # IO: read pairs, set up RSKF
    print(f"[IO] Reading training pairs from: {args.train_csv}")
    all_rows = read_csv_pairs(args.train_csv)
    labels = [y for _, y in all_rows]
    print(f"[IO] Found {len(all_rows)} samples.")

    rskf = RepeatedStratifiedKFold(n_splits=args.cv_splits, n_repeats=args.cv_repeats, random_state=args.cv_seed)
    total_splits = args.cv_splits * args.cv_repeats
    print(f"[CV] splits={args.cv_splits}, repeats={args.cv_repeats} -> total={total_splits}")

    summary_rows, cv_scores = [], []
    best_global_acc, best_global_ckpt = -1.0, None

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

        root_eval = args.data_root_eval if args.data_root_eval else args.data_train
        train_ds = ThzDataset(args.data_train, fold_train_csv, is_train=True)   # must output RAW thz [C_thz,H,W]
        val_ds   = ThzDataset(root_eval, fold_val_csv, is_train=False)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                                  num_workers=args.num_workers, pin_memory=True, drop_last=True)
        val_loader   = DataLoader(val_ds, batch_size=1, shuffle=False,
                                  num_workers=args.num_workers, pin_memory=True)
        print(f"[Data] Train batches: ~{len(train_loader)} | Val samples: {len(val_loader)}")

        # Wrap
        unet_ctrl = UnetWithControl(unet=unet, controlnet=controlnet, cond_scale=args.cond_scale).to(device)

        # Optimizer: ControlNet only
        opt = torch.optim.AdamW([{"params": controlnet.parameters(), "lr": 5e-5, "weight_decay": 0.0}])

        best_val = -1.0
        fold_ckpt = os.path.join(fold_dir, "best_dc.pt")
        for epoch in range(1, args.epochs + 1):
            print(f"\n[Epoch] Split {split_idx:03d} | Epoch {epoch:02d}/{args.epochs}")
            tr_loss, tr_acc = train_one_epoch(
                train_loader, unet_ctrl,
                prompt_embeds, prompt_to_class, num_classes,
                scheduler, eargs, args.img_size, torch_dtype, use_amp, opt, scaler,
                margin=args.margin, normalize_rank=args.normalize_rank
            )
            val_acc = validate(
                val_loader, unet_ctrl,
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
                    "thz_channels": args.thz_channels,
                    "controlnet_state_dict": controlnet.state_dict(),
                    "prompts_csv": args.prompts_csv,
                    "n_samples": args.n_samples,
                    "to_keep": args.to_keep,
                    "num_train_timesteps": args.num_train_timesteps,
                    "loss": args.loss,
                    "margin": args.margin,
                    "normalize_rank": args.normalize_rank,
                }, fold_ckpt)
                print(f"[Checkpoint] Saved best fold checkpoint: {fold_ckpt}  (val_acc={best_val:.4f})")

        cv_scores.append(best_val)
        summary_rows.append({"split_idx": split_idx, "best_val_acc": best_val, "fold_dir": fold_dir})
        print(f"[Split] Best val_acc for split {split_idx:03d}: {best_val:.4f}")

        if best_val > best_global_acc:
            best_global_acc, best_global_ckpt = best_val, fold_ckpt
            print(f"[Global] New global best: acc={best_global_acc:.4f}  ckpt={best_global_ckpt}")

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
    std_acc  = float(np.std(cv_scores)) if len(cv_scores) else 0.0
    print(f"CV mean acc = {mean_acc:.4f}  ± {std_acc:.4f}")
    print(f"Global best acc = {best_global_acc:.4f}")
    if best_global_ckpt:
        print(f"Global best ckpt: {best_global_ckpt}")
    print(f"Summary CSV: {summary_csv}")
    print("================================\n")

    if args.final_eval:
        test_csv = args.test_csv if args.test_csv is not None else args.val_csv
        assert args.data_test and test_csv, "--final_eval requires --data_test and (--test_csv or --val_csv)"
        print("[FINAL] Starting final evaluation...")
        print(f"[FINAL] Loading global best checkpoint: {best_global_ckpt}")
        ckpt = torch.load(best_global_ckpt, map_location="cpu")
        print(f"[FINAL] Global best val_acc (from CV): {ckpt.get('best_val_acc','N/A')}")

        print("[FINAL] Rebuilding SD base...")
        vae_f, unet_f, tokenizer_f, text_encoder_f, scheduler_f = build_sd2_1_base(
            dtype=ckpt["dtype"], use_xformers=False, train_all=False, version=ckpt["version"]
        )
        unet_f.eval()

        print(f"[FINAL] Rebuilding ControlNet (conditioning_channels={ckpt['thz_channels']}) and loading weights...")
        try:
            controlnet_f = ControlNetModel.from_unet(unet_f, conditioning_channels=int(ckpt["thz_channels"]))
        except TypeError:
            controlnet_f = ControlNetModel.from_unet(unet_f)
            if hasattr(controlnet_f, "controlnet_cond_embedding") and hasattr(controlnet_f.controlnet_cond_embedding, "conv_in"):
                old_conv = controlnet_f.controlnet_cond_embedding.conv_in
                controlnet_f.controlnet_cond_embedding.conv_in = nn.Conv2d(
                    int(ckpt["thz_channels"]), old_conv.out_channels, kernel_size=old_conv.kernel_size,
                    stride=old_conv.stride, padding=old_conv.padding, bias=old_conv.bias is not None
                )
                controlnet_f.config.conditioning_channels = int(ckpt["thz_channels"])
        controlnet_f = controlnet_f.to(device, dtype=unet_f.dtype)
        controlnet_f.load_state_dict(ckpt["controlnet_state_dict"], strict=False)
        controlnet_f.eval()

        unet_ctrl_f = UnetWithControl(unet=unet_f, controlnet=controlnet_f, cond_scale=ckpt.get("cond_scale", 1.0)).to(device)

        print(f"[FINAL] Reloading prompts from: {ckpt['prompts_csv']}")
        pb_f = PromptBank(ckpt["prompts_csv"])
        prompt_embeds_f = pb_f.to_text_embeds(tokenizer_f, text_encoder_f, device)
        prompt_to_class_f = pb_f.prompt_to_class.to(device)
        num_classes_f = pb_f.num_classes

        final_root = args.data_test
        final_ds = ThzDataset(final_root, test_csv, is_train=False)  # must output RAW thz
        final_loader = DataLoader(final_ds, batch_size=1, shuffle=False,
                                  num_workers=args.num_workers, pin_memory=True)
        print(f"[FINAL] Test set: {len(final_loader)} samples | CSV={test_csv} | ROOT={final_root}")

        use_amp_final = (device == "cuda") and (ckpt["dtype"] in ("float16", "bfloat16"))
        final_acc = validate(
            final_loader, unet_ctrl_f,
            prompt_embeds_f, prompt_to_class_f, num_classes_f,
            scheduler_f, eargs, ckpt["img_size"],
            torch_dtype=(torch.float16 if ckpt["dtype"] == "float16" else torch.bfloat16 if ckpt["dtype"] == "bfloat16" else torch.float32),
            use_amp=use_amp_final,
            normalize_rank=ckpt.get("normalize_rank", False)
        )
        print(f"[FINAL] accuracy on fixed set ({test_csv} @ {final_root}): {final_acc:.4f}")


if __name__ == "__main__":
    main()
