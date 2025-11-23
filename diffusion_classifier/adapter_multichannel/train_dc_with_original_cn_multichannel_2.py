"""
Train a diffusion-based classifier on THz volumes using Stable Diffusion v2.x,
an original ControlNet, and a LatentMultiChannelAdapter. This variant keeps the
Stable Diffusion backbone (VAE, UNet, text encoder, scheduler) fixed across all
cross-validation folds and trains ControlNet (and optionally the adapter) with SGD.

The pipeline:
    1. Load THz volumes and labels from a CSV file.
    2. Convert each THz volume into a sequence of pseudo-RGB frames and encode
       them into latents using the frozen Stable Diffusion VAE.
    3. Aggregate the latent sequence with LatentMultiChannelAdapter into a
       single 2D latent representation.
    4. Run the latent through UNet + ControlNet and a prompt-based diffusion
       classifier (`eval_prob_adaptive_differentiable`) to obtain per-prompt errors.
    5. Pool prompt errors into class errors, convert them into logits, and
       train ControlNet (and optionally the adapter) using cross-entropy loss
       under Repeated Stratified K-Fold (RSKF).
    6. Save the best model per split and a CV summary, and optionally reload
       the globally best model for evaluation on a held-out test set.

How to run:
    python train_dc_with_original_cn_multichannel_2.py \
        --data_train /path/to/thz_data \
        --train_csv /path/to/train.csv \
        --prompts_csv /path/to/prompts.csv \
        --version 2-1 \
        --dtype bfloat16 \
        --img_size 256 \
        --num_train_timesteps 1000 \
        --n_trials 1 \
        --n_samples 8 4 2 \
        --to_keep 3 2 1 \
        --loss l2 \
        --logit_scale 60.0 \
        --use_xformers \
        --cond_scale 2.0 \
        --learn_adapter \
        --lr_adapter 1e-3 \
        --wd_adapter 0.0 \
        --lr_controlnet 5e-3 \
        --wd_controlnet 0.0 \
        --epochs 200 \
        --batch_size 2 \
        --cv_splits 5 \
        --cv_repeats 1 \
        --cv_seed 42 \
        --final_eval \
        --data_test /path/to/test_data \
        --test_csv /path/to/test.csv

Key arguments:
    Data:
        --data_train (str)         Root directory with THz training volumes.
        --train_csv (str)          CSV with (path,label) pairs for training.
        --data_root_eval (str)     Optional root for validation/test data
                                   (defaults to --data_train if not set).

    Prompts / Stable Diffusion:
        --prompts_csv (str)        CSV with columns: prompt, classname, classidx.
        --version (str)            SD version ("2-1", "2-0", "1-5", ...).
        --dtype (str)              float16 | float32 | bfloat16.
        --img_size (int)           Input image size for SD (256 or 512).
        --num_train_timesteps (int)Number of diffusion timesteps.
        --n_trials (int)           Number of diffusion trials per sample.
        --n_samples (list[int])    Sequence of sample counts per trial.
        --to_keep (list[int])      Number of samples kept per trial.
        --loss (str)               Error loss type (l1 | l2 | huber).
        --logit_scale (float)      Scale factor to convert errors into logits.
        --use_xformers             Enable xFormers memory-efficient attention.

    ControlNet / Adapter:
        --cond_scale (float)       Conditioning scale for ControlNet.
        --learn_adapter            If set, train the LatentMultiChannelAdapter.
        --lr_adapter (float)       Learning rate for the adapter.
        --wd_adapter (float)       Weight decay for the adapter.
        --lr_controlnet (float)    Learning rate for ControlNet (SGD).
        --wd_controlnet (float)    Weight decay for ControlNet.

    Training / CV:
        --epochs (int)             Number of epochs per CV split.
        --batch_size (int)         Training batch size.
        --num_workers (int)        DataLoader workers.
        --cv_splits (int)          Number of RSKF folds.
        --cv_repeats (int)         Number of RSKF repeats.
        --cv_seed (int)            Seed for CV and training.

    Final evaluation:
        --final_eval               If set, run final eval on a fixed test set.
        --data_test (str)          Root directory of the test dataset.
        --test_csv (str)           CSV with (path,label) for the test set.
        --val_csv (str)            Fallback CSV if --test_csv is not provided.

    IO:
        --save_dir (str)           Base directory for splits, checkpoints,
                                   and cv_summary.csv.

Side effects:
    - Creates a timestamped experiment folder under --save_dir.
    - Saves the best ControlNet+adapter checkpoint for each RSKF split.
    - Writes a CV summary CSV with validation accuracy per split.
    - Optionally reloads the global best model and reports final test accuracy.
"""


import os
import argparse
import csv
import numpy as np
from datetime import datetime
from zoneinfo import ZoneInfo

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from sklearn.model_selection import RepeatedStratifiedKFold
from tqdm.auto import tqdm

from transformers import logging as hf_logging
import diffusers
from diffusers import ControlNetModel

# --- Your modules (must exist in your project) ---
from diffusion.datasets import ThzDataset
from thz_to_vae_adapter import LatentMultiChannelAdapter
from eval_prob_adaptive import eval_prob_adaptive_differentiable
from adapter.help_functions import (
    build_sd2_1_base, read_csv_pairs, write_csv_pairs, PromptBank,
    pool_prompt_errors_to_class_errors, pool_prompt_errors_to_class_errors_batch
)

hf_logging.set_verbosity_error()
device = "cuda" if torch.cuda.is_available() else "cpu"


# --------------------- Utilities ---------------------
def set_seed(seed: int = 42):
    """Set seeds for reproducibility."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)



def normalize_amp_and_scaler(dtype_str: str):
    torch_dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[dtype_str]
    use_amp = (device == "cuda") and (dtype_str in ("float16", "bfloat16"))
    scaler = torch.amp.GradScaler('cuda', enabled=(use_amp and dtype_str == "float16"))
    return torch_dtype, use_amp, scaler

@torch.no_grad()
def encode_all_frames_with_vae(frames_vae, vae, scaling=0.18215):
    posterior = vae.encode(frames_vae).latent_dist.mean.to(torch.float32)
    latents = posterior * scaling
    return latents


# ----------------- Train / Validate -----------------
def train_one_epoch(
    loader, adapter, unet, controlnet, vae, prompt_embeds, prompt_to_class, num_classes,
    scheduler, eargs, img_size, logit_scale, torch_dtype, use_amp, opt, scaler,
    reduce="mean"
):
    if eargs.learn_adapter:
        adapter.train()
    else:
        adapter.eval()
    controlnet.train()
    unet.eval() 

    running_loss, running_acc, seen = 0.0, 0, 0
    latent_size = img_size // 8
    out_hw = (img_size, img_size)

    print("[train_one_epoch] Starting training iteration...")
    for step, (vol, label, _) in enumerate(loader, start=1):
        vol = vol.to(device)
        if vol.dim() == 6:
            vol = vol.squeeze(1)
        label = label.to(device).long()

        opt.zero_grad(set_to_none=True)

        with torch.autocast(device_type="cuda" if device=="cuda" else "cpu", dtype=torch_dtype, enabled=use_amp):
            x = vol.mean(dim=1)
            B, T, H, W = x.shape
            frames = x.unsqueeze(2).repeat(1,1,3,1,1).view(B*T,3,H,W)
            frames = F.interpolate(frames, size=out_hw, mode="bilinear", align_corners=False)
            frames = frames.clamp(0,1)*2-1
            lat_chunks = []
            bt = frames.shape[0]
            chunk = getattr(eargs, "vae_chunk", 256)
            with torch.no_grad():
                for s in range(0, bt, chunk):
                    lat_chunks.append(encode_all_frames_with_vae(frames[s:s+chunk], vae, scaling=0.18215))
            latents_flat = torch.cat(lat_chunks, dim=0)
            lat_stack = latents_flat.view(B, T, 4, latent_size, latent_size)
            lat = adapter(lat_stack)
            try:
                cn_in_ch = controlnet.controlnet_cond_embedding.conv_in.in_channels
            except Exception:
                cn_in_ch = getattr(controlnet, "in_channels", 4)

            if cn_in_ch == 3:
                with torch.no_grad():
                    control_cond_img = vae.decode(lat / 0.18215).sample
            else:
                control_cond_img = None
        
            B = lat.size(0)
            errors_list = []
            for i in range(B):
                cond_to_pass = control_cond_img[i:i+1] if control_cond_img is not None else None
                pred_idx , _, epp_i = eval_prob_adaptive_differentiable(
                    unet=unet,
                    latent=lat[i:i+1],
                    text_embeds=prompt_embeds,
                    scheduler=scheduler,
                    args=eargs,
                    latent_size=latent_size,
                    controlnet=controlnet,
                    all_noise=None,
                    controlnet_cond=cond_to_pass
                )
                if epp_i.dim() == 1:
                    epp_i = epp_i.unsqueeze(0)
                errors_list.append(epp_i)

            errors_per_prompt = torch.cat(errors_list, dim=0)  
            class_errors_batch = pool_prompt_errors_to_class_errors_batch(
                errors_per_prompt, prompt_to_class, num_classes, reduce=reduce
            )   
            logits = (-class_errors_batch) * float(logit_scale)
            loss = F.cross_entropy(logits, label)

        if scaler.is_enabled():
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(controlnet.parameters(), 1.0)
            if step % 30 == 1:
                # ---- Grad-Logging (vor dem Step!) ----
                for name, p in list(controlnet.named_parameters())[:3]:
                    g = None if p.grad is None else p.grad
                    msg = "None" if g is None else f"mean={g.abs().mean().item():.4e} max={g.abs().max().item():.4e}"
                    print(f"[DEBUG] grad ControlNet {name}: {msg}")

                if eargs.learn_adapter:
                    for name, p in list(adapter.named_parameters())[:3]:
                        g = None if p.grad is None else p.grad
                        msg = "None" if g is None else f"mean={g.abs().mean().item():.4e} max={g.abs().max().item():.4e}"
                        print(f"[DEBUG] grad Adapter {name}: {msg}")

            scaler.step(opt)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(controlnet.parameters(), 1.0)

            # ---- Grad-Logging (vor dem Step!) ----
            for name, p in list(controlnet.named_parameters())[:3]:
                g = None if p.grad is None else p.grad
                msg = "None" if g is None else f"mean={g.abs().mean().item():.4e} max={g.abs().max().item():.4e}"
                print(f"[DEBUG] grad ControlNet {name}: {msg}")

            if eargs.learn_adapter:
                for name, p in list(adapter.named_parameters())[:3]:
                    g = None if p.grad is None else p.grad
                    msg = "None" if g is None else f"mean={g.abs().mean().item():.4e} max={g.abs().max().item():.4e}"
                    print(f"[DEBUG] grad Adapter {name}: {msg}")
            snap_before, _ = update_ratio(controlnet, None)
            opt.step()
            _, ratio = update_ratio(controlnet, snap_before)
            print(f"[STAT] controlnet avg update/param = {ratio:.3e}")

        with torch.no_grad():
            running_loss += loss.item() * vol.size(0)
            preds = torch.argmin(class_errors_batch, dim=1)
            running_acc += (preds == label).sum().item()
            seen += vol.size(0)

        if step % 10 == 1:
            print(f"[train_one_epoch] step={step}  "
                  f"avg_loss={running_loss/max(1,seen):.4f}  avg_acc={running_acc/max(1,seen):.4f}")
            with torch.no_grad():
                for any_name, any_param in controlnet.named_parameters():
                    print(f"[DEBUG] sample weight after step: {any_param.view(-1)[0].item():.6f}")
                    break

    epoch_loss = running_loss / max(1, seen)
    epoch_acc = running_acc / max(1, seen)
    print(f"[train_one_epoch] Done. epoch_loss={epoch_loss:.4f}  epoch_acc={epoch_acc:.4f}")
    return epoch_loss, epoch_acc


@torch.no_grad()
def validate(
    loader, adapter, unet, controlnet, vae, prompt_embeds, prompt_to_class, num_classes,
    scheduler, eargs, img_size, torch_dtype, use_amp,  reduce="mean"
):
    """Validation loop mirroring the training inference flow."""
    adapter.eval()
    controlnet.eval()
    unet.eval()

    latent_size = img_size // 8
    out_hw = (img_size, img_size)

    total, correct = 0, 0
    print("[validate] Starting validation iteration...")
    for step, (vol, label, _) in enumerate(loader, start=1):
        vol = vol.to(device)
        if vol.dim() == 6:
            vol = vol.squeeze(1)
        label = label.to(device).long()

        with torch.autocast(device_type="cuda" if device=="cuda" else "cpu", dtype=torch_dtype, enabled=use_amp):
            x = vol.mean(dim=1)
            B, T, H, W = x.shape
            frames = x.unsqueeze(2).repeat(1,1,3,1,1).view(B*T,3,H,W)
            frames = F.interpolate(frames, size=out_hw, mode="bilinear", align_corners=False)
            frames = frames.clamp(0,1)*2-1
            lat_chunks = []
            bt = frames.shape[0]
            chunk = getattr(eargs, "vae_chunk", 256)
            with torch.no_grad():
                for s in range(0, bt, chunk):
                    lat_chunks.append(encode_all_frames_with_vae(frames[s:s+chunk], vae, scaling=0.18215))
            latents_flat = torch.cat(lat_chunks, dim=0)
            lat_stack = latents_flat.view(B, T, 4, latent_size, latent_size)
            lat = adapter(lat_stack)
            try:
                cn_in_ch = controlnet.controlnet_cond_embedding.conv_in.in_channels
            except Exception:
                cn_in_ch = getattr(controlnet, "in_channels", 4)

            if cn_in_ch == 3:
                with torch.no_grad():
                    control_cond_img = vae.decode(lat / 0.18215).sample
            else:
                control_cond_img = None

            B = lat.size(0)
            errors_list = []
            for i in range(B):
                cond_to_pass = control_cond_img[i:i+1] if control_cond_img is not None else None
                _, _, epp_i = eval_prob_adaptive_differentiable(
                    unet=unet,
                    latent=lat[i:i+1],
                    text_embeds=prompt_embeds,
                    scheduler=scheduler,
                    args=eargs,
                    latent_size=latent_size,
                    controlnet=controlnet,
                    all_noise=None,
                    controlnet_cond=cond_to_pass
                )
                if epp_i.dim() == 1:
                    epp_i = epp_i.unsqueeze(0)
                errors_list.append(epp_i)
            errors_per_prompt = torch.cat(errors_list, dim=0)

            class_errors_batch = pool_prompt_errors_to_class_errors_batch(
                errors_per_prompt, prompt_to_class, num_classes, reduce=reduce
            )
            preds = torch.argmin(class_errors_batch, dim=1)

        correct += (preds == label).sum().item()
        total += label.numel()

        if step % 20 == 1:
            print(f"[validate] step={step}  running_acc={correct/max(1,total):.4f}")

    val_acc = correct / max(1, total)
    print(f"[validate] Done. val_acc={val_acc:.4f}")
    return val_acc

def update_ratio(mod, last_params=None):
    ratios = []
    cur = {n: p.detach().clone() for n,p in mod.named_parameters() if p.requires_grad}
    if last_params is not None:
        for n,p in mod.named_parameters():
            if p.requires_grad:
                upd = (cur[n] - last_params[n])
                num = upd.norm().item()
                den = (cur[n].norm().item() + 1e-12)
                ratios.append(num / den)
    return cur, (sum(ratios)/len(ratios) if ratios else None)

# ----------------------- Main -----------------------
def main():
    ap = argparse.ArgumentParser(description="Diffusion Classifier (SD + ControlNet) with THz, RSKF, and Final Eval")

    # Data
    ap.add_argument("--data_train", type=str, required=True)
    ap.add_argument("--train_csv", type=str, required=True)
    ap.add_argument("--data_root_eval", type=str, default=None,
                    help="Optional root override for validation/test datasets (default: use data_train).")

    # Prompts / SD / Diffusion Classifier
    ap.add_argument("--prompts_csv", type=str, required=True, help="CSV with columns: prompt,classname,classidx")
    ap.add_argument("--version", type=str, default="2-1", choices=("2-1","2-0","1-1","1-2","1-3","1-4","1-5"))
    ap.add_argument("--dtype", type=str, default="float32", choices=("float16","float32","bfloat16"))
    ap.add_argument("--img_size", type=int, default=256, choices=(256,512))
    ap.add_argument("--num_train_timesteps", type=int, default=1000)
    ap.add_argument("--n_trials", type=int, default=1)
    ap.add_argument("--n_samples", nargs="+", type=int, required=True)
    ap.add_argument("--to_keep",   nargs="+", type=int, required=True)
    ap.add_argument("--loss", type=str, default="l2", choices=("l1","l2","huber"))
    ap.add_argument("--logit_scale", type=float, default=60.0)
    ap.add_argument("--use_xformers", action="store_true")

    # Control / Adapter
    ap.add_argument("--cond_scale", type=float, default=2.0, help="conditioning_scale for ControlNet")
    ap.add_argument("--learn_adapter", action="store_true")
    ap.add_argument("--lr_adapter", type=float, default=1e-3)
    ap.add_argument("--wd_adapter", type=float, default=0.0)
    ap.add_argument("--lr_controlnet", type=float, default=5e-3)
    ap.add_argument("--wd_controlnet", type=float, default=0.0)

    # Training
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--num_workers", type=int, default=4)

    # CV
    ap.add_argument("--cv_splits", type=int, default=5)
    ap.add_argument("--cv_repeats", type=int, default=1)
    ap.add_argument("--cv_seed", type=int, default=42)

    # Final evaluation
    ap.add_argument("--final_eval", action="store_true",
                    help="After CV: load global best and evaluate on a fixed set.")
    ap.add_argument("--data_test", type=str, default=None, help="Test root dir.")
    ap.add_argument("--test_csv", type=str, default=None, help="CSV (path,label) for test. If empty, fall back to --val_csv.")
    ap.add_argument("--val_csv", type=str, default=None, help="Fallback CSV for final eval if --test_csv not set.")

    # IO
    ap.add_argument("--save_dir", type=str, default="./runs/checkpoints_dc_adapter_multichannel")

    args = ap.parse_args()
    # Print config
    print("========== CONFIG ==========")
    for k, v in vars(args).items():
        print(f"{k}: {v}")
    print("============================")

    set_seed(args.cv_seed)

    torch.backends.cudnn.benchmark = False

    # Save root
    stamp = datetime.now(ZoneInfo("Europe/Berlin")).strftime("%y%m%d_%H%M")
    save_dir = os.path.join(args.save_dir, f"sd{args.version}_img{args.img_size}_{args.dtype}_{stamp}")
    os.makedirs(save_dir, exist_ok=True)
    print(f"[IO] Save directory: {save_dir}")

    # --- EArgs for eval_prob_adaptive_differentiable ---
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
    eargs.learn_adapter = args.learn_adapter
    eargs.lr_adapter = args.lr_adapter
    eargs.wd_adapter = args.wd_adapter
    eargs.cond_scale = args.cond_scale

    # --- Read all train rows and prepare RSKF ---
    print(f"[IO] Reading training pairs from: {args.train_csv}")
    all_rows = read_csv_pairs(args.train_csv)
    labels = [y for _, y in all_rows]
    print(f"[IO] Found {len(all_rows)} samples.")

    summary_rows = []
    cv_scores = []
    best_global_acc = -1.0
    best_global_ckpt = None

     # AMP / dtype
    torch_dtype, use_amp, scaler = normalize_amp_and_scaler(args.dtype)
    print(f"[AMP] device={device}  dtype={args.dtype}  use_amp={use_amp}")

    # --- Load SD backbone (frozen VAE/UNet/TextEncoder, scheduler) ---
    print("[SD] Building Stable Diffusion base...")
    vae, unet, tokenizer, text_encoder, scheduler, controlnet = build_sd2_1_base(
        dtype=args.dtype, use_xformers=args.use_xformers, train_all=False, version=args.version
    )
    print("diffusers version:", diffusers.__version__)
    print("controlnet class:", controlnet.__class__.__name__)
    print("controlnet config in/out:", getattr(controlnet, "in_channels", None), getattr(controlnet, "out_channels", None))
    print(controlnet.config)
    unet = unet.to(device).eval()
    vae = vae.to(device).eval()
    controlnet = controlnet.to(device).eval()
    scheduler.set_timesteps(args.num_train_timesteps)
    print("[SD] Done. UNet/VAEs are frozen.")

    # --- Prompt bank & text embeddings ---
    print(f"[Prompts] Loading prompts from: {args.prompts_csv}")
    pb = PromptBank(args.prompts_csv)
    prompt_embeds = pb.to_text_embeds(tokenizer, text_encoder, device)
    prompt_to_class = pb.prompt_to_class.to(device)
    num_classes = pb.num_classes
    print(f"[Prompts] Loaded {len(pb.prompt_texts)} prompts over {num_classes} classes.")

    # Iterate RSKF folds
    print("[Train] Starting epochs...")
    rskf = RepeatedStratifiedKFold(
    n_splits=args.cv_splits, n_repeats=args.cv_repeats, random_state=args.cv_seed
    )
    total_splits = args.cv_splits * args.cv_repeats
    print(f"[CV] RSKF configured: splits={args.cv_splits}, repeats={args.cv_repeats}, total={total_splits}")
    for split_idx, (idx_tr, idx_va) in enumerate(rskf.split(all_rows, labels), start=1):
        print(f"\n===== [SPLIT {split_idx:03d}/{total_splits}] =====")
        fold_dir = os.path.join(save_dir, f"split_{split_idx:03d}")
        fold_ckpt = os.path.join(fold_dir, "best_dc.pt")
        os.makedirs(fold_dir, exist_ok=True)
        print(f"[IO] Fold directory: {fold_dir}")

        train_rows = [all_rows[i] for i in idx_tr]
        val_rows = [all_rows[i] for i in idx_va]
        fold_train_csv = os.path.join(fold_dir, "train.csv")
        fold_val_csv = os.path.join(fold_dir, "val.csv")
        write_csv_pairs(fold_train_csv, train_rows)
        write_csv_pairs(fold_val_csv, val_rows)
        print(f"[IO] Wrote train.csv (n={len(train_rows)}) & val.csv (n={len(val_rows)})")

        # Datasets/loaders
        root_eval = args.data_root_eval if args.data_root_eval else args.data_train
        train_ds = ThzDataset(args.data_train, fold_train_csv, is_train=True)
        val_ds = ThzDataset(root_eval, fold_val_csv, is_train=False)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                                num_workers=args.num_workers, pin_memory=True, drop_last=True)
        val_loader = DataLoader(val_ds, batch_size=1, shuffle=False,
                                num_workers=args.num_workers, pin_memory=True)
        print(f"[Data] Train batches: ~{len(train_loader)} | Val samples: {len(val_loader)}")


        try:
            controlnet.enable_gradient_checkpointing()
            print("[ControlNet] Enabled gradient checkpointing.")
        except Exception:
            print("[ControlNet] Gradient checkpointing not available.")

        adapter = LatentMultiChannelAdapter(k_t=5, use_attn_pool=True).to(device)
        param_groups = []
        if args.learn_adapter:
            param_groups.append({"params": adapter.parameters(), "lr": args.lr_adapter, "weight_decay": args.wd_adapter})
            print("[OPT] Adapter parameters will be trained.")
        else:
            for p in adapter.parameters():
                p.requires_grad_(False)
            adapter.eval()
            print("[OPT] Adapter is frozen.")

        param_groups.append({"params": controlnet.parameters(), "lr": args.lr_controlnet, "weight_decay": args.wd_controlnet})
        print("[OPT] ControlNet parameters will be trained.")

        opt = torch.optim.SGD(param_groups, momentum=0.9)

        print("[DEBUG] Trainable parameter groups:")
        for name, p in list(adapter.named_parameters())[:3]:
            print(" Adapter:", name, p.requires_grad)
        for name, p in list(controlnet.named_parameters())[:3]:
            print(" ControlNet:", name, p.requires_grad)
        for name, p in list(unet.named_parameters())[:3]:
            print(" UNet:", name, p.requires_grad)

        # Fold training
        best_val = -1.0
        
        for epoch in range(1, args.epochs + 1):
            print(f"\n[Epoch] Split {split_idx:03d} | Epoch {epoch:02d}/{args.epochs}")
            tr_loss, tr_acc = train_one_epoch(
                train_loader, adapter, unet, controlnet, vae,
                prompt_embeds, prompt_to_class, num_classes,
                scheduler, eargs, args.img_size, args.logit_scale,
                torch_dtype, use_amp, opt, scaler, reduce="mean"
            )
            val_acc = validate(
                val_loader, adapter, unet, controlnet, vae,
                prompt_embeds, prompt_to_class, num_classes,
                scheduler, eargs, args.img_size, torch_dtype, use_amp, reduce="mean"
            )
            print(f"[Epoch] Result: train_loss={tr_loss:.4f}  train_acc={tr_acc:.4f}  val_acc={val_acc:.4f}")

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
                    "adapter_state_dict": adapter.state_dict(),
                    "prompts_csv": args.prompts_csv,
                    "n_samples": args.n_samples,
                    "to_keep": args.to_keep,
                    "num_train_timesteps": args.num_train_timesteps,
                    "loss": args.loss,
                    "logit_scale": args.logit_scale,
                }, fold_ckpt)
                print(f"[Checkpoint] Saved best fold checkpoint: {fold_ckpt}  (val_acc={best_val:.4f})")

        # Collect split stats
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
        test_csv = args.test_csv if args.test_csv is not None else args.val_csv
        assert args.data_test and test_csv, "--final_eval requires --data_test and (--test_csv or --val_csv)"
        print("[FINAL] Starting final evaluation...")
        print(f"[FINAL] Loading global best checkpoint: {best_global_ckpt}")
        ckpt = torch.load(best_global_ckpt, map_location="cpu")
        print(f"[FINAL] Global best val_acc (from CV): {ckpt.get('best_val_acc','N/A')}")

        # Rebuild SD base (frozen) exactly as during training
        print("[FINAL] Rebuilding SD base...")
        vae_f, unet_f, tokenizer_f, text_encoder_f, scheduler_f, controlnet_f = build_sd2_1_base(
            dtype=ckpt["dtype"], use_xformers=False, train_all=False, version=ckpt["version"]
        )
        unet_f.eval()

        adapter_f = LatentMultiChannelAdapter(k_t=5, use_attn_pool=True).to(device)
        adapter_f.load_state_dict(ckpt["adapter_state_dict"], strict=False)
        adapter_f.eval()

        controlnet_f.load_state_dict(ckpt["controlnet_state_dict"], strict=False)
        controlnet_f.eval()

        # Load prompts again to get embeddings (they are not stored in the ckpt)
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

        # AMP settings for final eval
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
        eargs_f.n_trials  = 1
        eargs_f.batch_size= 1
        eargs_f.dtype     = ckpt["dtype"]
        eargs_f.loss      = ckpt["loss"]
        eargs_f.num_train_timesteps = ckpt["num_train_timesteps"]
        eargs_f.version   = ckpt["version"]
        eargs_f.learn_adapter = False

        # Run validate() on the fixed test set
        final_acc = validate(
            final_loader, adapter_f, unet_f, controlnet_f, vae_f,
            prompt_embeds_f, prompt_to_class_f, num_classes_f,
            scheduler_f, eargs_f, ckpt["img_size"],
            torch_dtype=final_torch_dtype, use_amp=use_amp_final, reduce="mean"
        )
        print(f"[FINAL] accuracy on fixed set ({test_csv} @ {final_root}): {final_acc:.4f}")


if __name__ == "__main__":
    main()
