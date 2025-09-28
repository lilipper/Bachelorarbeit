import os
import sys
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

from tqdm.auto import tqdm
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
from eval_prob_adaptive import eval_prob_adaptive
from ControlNet import ControlNet
from adapter.ControlNet_Adapter_wrapper import ControlNetAdapterWrapper
import process_rdf as prdf
import time
from diffusion.datasets import ThzDataset
import csv
from collections import defaultdict
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold
from datetime import datetime
from zoneinfo import ZoneInfo
from adapter.help_functions import build_sd2_1_base, read_csv_pairs, write_csv_pairs, PromptBank, pool_prompt_errors_to_class_errors, pool_prompt_errors_to_class_errors_batch

hf_logging.set_verbosity_error()
device = "cuda" if torch.cuda.is_available() else "cpu"

# ========= Utils =========
def set_seed(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# ========= Training / Eval Loops =========
def train_one_epoch(train_loader, adapter, vae, unet, prompt_embeds, prompt_to_class, num_classes,
                    scheduler, eargs, img_size, logit_scale, use_amp, opt, scaler, reduce="mean", bar_desc=None):
    adapter.train()
    running_loss, running_acc, n_seen = 0.0, 0.0, 0
    train_iter = tqdm(train_loader, desc=bar_desc or "train", leave=False, ncols=0)
    for vol, label, filename in train_iter:
        vol   = vol.to(device)
        label = label.to(device).long()
        if vol.dim() == 6:
            vol = vol.squeeze(1)  # [B,1,T,H,W]

        opt.zero_grad()
        with torch.amp.autocast("cuda", dtype=torch.float16, enabled=use_amp):
            img  = adapter(vol)
            x_in = (img * 2.0 - 1.0).to(dtype=vae.dtype)
            lat  = vae.encode(x_in).latent_dist.mean * 0.18215

            batch_errors_list = []
            for b in range(lat.size(0)):
                _, _, errors_per_prompt_single = eval_prob_adaptive(
                    unet=unet, latent=lat[b:b+1], text_embeds=prompt_embeds,
                    scheduler=scheduler, args=eargs, latent_size=img_size // 8, all_noise=None
                )
                batch_errors_list.append(errors_per_prompt_single)

            errors_per_prompt_batch = torch.stack(batch_errors_list, dim=0)  # [B,P]
            class_errors_batch = pool_prompt_errors_to_class_errors_batch(
                errors_per_prompt_batch, prompt_to_class, num_classes, reduce=reduce
            )
            logits = (-class_errors_batch) * float(logit_scale)
            loss = F.cross_entropy(logits, label)

        if torch.isfinite(loss):
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(adapter.parameters(), max_norm=0.5)
            scaler.step(opt)
            scaler.update()

            with torch.no_grad():
                running_loss += loss.item() * vol.size(0)
                preds = torch.argmin(class_errors_batch, dim=1)
                running_acc += (preds == label).sum().item()
                n_seen += vol.size(0)
                train_iter.set_postfix(
                    loss=f"{running_loss/max(1,n_seen):.4f}",
                    acc=f"{running_acc/max(1,n_seen):.4f}"
                )

    train_loss = running_loss / max(1, n_seen)
    train_acc  = running_acc  / max(1, n_seen)
    return train_loss, train_acc

@torch.no_grad()
def validate(val_loader, adapter, vae, unet, prompt_embeds, prompt_to_class, num_classes,
             scheduler, eargs, img_size, use_amp, reduce="mean", bar_desc=None):
    adapter.eval()
    total, correct = 0, 0
    val_iter = tqdm(val_loader, desc=bar_desc or "val", leave=False, ncols=0)
    for vol, label, filename in val_iter:
        vol   = vol.to(device)
        label = label.to(device).long()
        if vol.dim() == 6:
            vol = vol.squeeze(1)

        with torch.amp.autocast('cuda', dtype=torch.float16, enabled=use_amp):
            img  = adapter(vol)
            x_in = (img * 2.0 - 1.0).to(dtype=vae.dtype)
            lat  = vae.encode(x_in).latent_dist.mean * 0.18215

        _, _, errors_per_prompt = eval_prob_adaptive(
            unet=unet, latent=lat, text_embeds=prompt_embeds,
            scheduler=scheduler, args=eargs, latent_size=img_size // 8, all_noise=None
        )
        class_errors = pool_prompt_errors_to_class_errors(
            errors_per_prompt.squeeze(0), prompt_to_class, num_classes, reduce=reduce
        )
        pred = torch.argmin(class_errors).item()
        correct += int(pred == label.item())
        total   += 1
        val_iter.set_postfix(acc=f"{correct/max(1,total):.4f}")
    val_acc = correct / max(1, total)
    return val_acc

# ========= Main mit RSKF =========
def main():
    ap = argparse.ArgumentParser()
    # Daten
    ap.add_argument("--data_train", type=str, default="/pfs/work9/workspace/scratch/ma_lilipper-lippert_bachelorthesis_ws/thz_dataset/train", help="Wurzelordner der .mat-Dateien (Train/CV)")
    ap.add_argument("--data_test",  type=str, default="/pfs/work9/workspace/scratch/ma_lilipper-lippert_bachelorthesis_ws/thz_dataset/test", help="(Optional) Finaler Val/Test-Root")
    ap.add_argument("--train_csv",  type=str, default="/pfs/work9/workspace/scratch/ma_lilipper-lippert_bachelorthesis_ws/Bachelorarbeit/jsons/train_labels.csv", help="CSV: filename,label (nur für CV)")
    ap.add_argument("--val_csv",    type=str, default="/pfs/work9/workspace/scratch/ma_lilipper-lippert_bachelorthesis_ws/Bachelorarbeit/jsons/test_labels.csv", help="(Optional) finaler Val/Test-CSV")
    ap.add_argument("--prompts_csv", type=str, default="/pfs/work9/workspace/scratch/ma_lilipper-lippert_bachelorthesis_ws/Bachelorarbeit/diffusion_classifier/prompts/thz_prompts.csv", help="CSV: prompt,classname,classidx")

    # SD/DC
    ap.add_argument("--dtype", type=str, default="bfloat16", choices=("float16", "float32", "bfloat16"))
    ap.add_argument("--img_size", type=int, default=256, choices=(256, 512))
    ap.add_argument("--loss", type=str, default="l2", choices=("l1", "l2", "huber"))
    ap.add_argument("--n_trials", type=int, default=1)
    ap.add_argument("--n_samples", nargs="+", type=int, required=True, help="z. B. 8 4 2 1")
    ap.add_argument("--to_keep",   nargs="+", type=int, required=True, help="z. B. 6 3 2 1")
    ap.add_argument("--num_train_timesteps", type=int, default=1000)
    ap.add_argument("--logit_scale", type=float, default=80.0)
    ap.add_argument("--version", type=str, default="2-1", choices=("2-1", "2-0", '1-1', '1-2', '1-3', '1-4', '1-5'))
    ap.add_argument("--reduce", type=str, default="mean", choices=("mean", "min"), help="Pooling über Prompts -> Klassen")

    # Training
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--use_xformers", action="store_true")
    ap.add_argument("--save_dir", type=str, default="./runs/checkpoints_adapter")

    # Cross-Validation (RSKF)
    ap.add_argument("--cv_splits", type=int, default=5, help="Anzahl Folds pro Wiederholung")
    ap.add_argument("--cv_repeats", type=int, default=3, help="Anzahl Wiederholungen")
    ap.add_argument("--cv_seed", type=int, default=42)

    # Final Eval
    ap.add_argument("--final_eval", action="store_true", help="Bestes CV-Modell am Ende auf data_test/val_csv evaluieren")

    args = ap.parse_args()
    set_seed(args.cv_seed)

    stamp = datetime.now(ZoneInfo("Europe/Berlin")).strftime("%y%m%d_%H%M")
    save_dir = os.path.join(args.save_dir, f"rskf_{args.cv_splits}x{args.cv_repeats}_{stamp}")
    os.makedirs(save_dir, exist_ok=True)

    # SD/Prompts EINMAL laden (frozen)
    vae, unet, tokenizer, text_encoder, scheduler = build_sd2_1_base(dtype=args.dtype, use_xformers=args.use_xformers)
    pb = PromptBank(args.prompts_csv)
    prompt_embeds   = pb.to_text_embeds(tokenizer, text_encoder, device)
    prompt_to_class = pb.prompt_to_class.to(device)
    num_classes     = pb.num_classes
    P               = len(pb.prompt_texts)

    # EArgs für eval_prob_adaptive
    class EArgs: pass
    eargs = EArgs()
    eargs.n_samples = args.n_samples
    eargs.to_keep   = args.to_keep        # wichtig: CLI respektieren!
    eargs.n_trials  = args.n_trials
    eargs.batch_size= args.batch_size
    eargs.dtype     = args.dtype
    eargs.loss      = args.loss
    eargs.num_train_timesteps = args.num_train_timesteps
    eargs.version   = args.version

    # Gesamtdaten (nur TRAIN) einlesen
    all_rows = read_csv_pairs(args.train_csv)
    labels   = [y for _, y in all_rows]

    # RSKF vorbereiten
    rskf = RepeatedStratifiedKFold(
        n_splits=args.cv_splits,
        n_repeats=args.cv_repeats,
        random_state=args.cv_seed
    )

    cv_scores = []
    best_global_acc = -1.0
    best_global_ckpt = None
    run_counter = 0
    total_splits = args.cv_splits * args.cv_repeats
    for split_idx, (idx_tr, idx_va) in tqdm(
        enumerate(rskf.split(all_rows, labels), start=1),
        total=total_splits, desc="RSKF splits", ncols=0):
        run_counter += 1
        fold_dir = os.path.join(save_dir, f"split_{split_idx:03d}")
        os.makedirs(fold_dir, exist_ok=True)

        train_rows = [all_rows[i] for i in idx_tr]
        val_rows   = [all_rows[i] for i in idx_va]

        fold_train_csv = os.path.join(fold_dir, "train.csv")
        fold_val_csv   = os.path.join(fold_dir, "val.csv")
        write_csv_pairs(fold_train_csv, train_rows)
        write_csv_pairs(fold_val_csv,   val_rows)

        # DataLoader für diesen Split (beide aus data_train!)
        train_ds = ThzDataset(args.data_train, fold_train_csv, is_train=True)
        val_ds   = ThzDataset(args.data_train, fold_val_csv,   is_train=False)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                                  num_workers=args.num_workers, pin_memory=True)
        val_loader   = DataLoader(val_ds, batch_size=1, shuffle=False,
                                  num_workers=args.num_workers, pin_memory=True)

        # Adapter/Optimizer je Split neu
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

        opt = torch.optim.AdamW(adapter.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        use_amp = (device == "cuda" and args.dtype == "float16")
        scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

        best_val_acc = -1.0
        best_path = os.path.join(fold_dir, "adapter_best.pt")

        for epoch in tqdm(range(1, args.epochs + 1),
                  desc=f"[S{split_idx:03d}] epochs",
                  leave=False, ncols=0):
            train_loss, train_acc = train_one_epoch(
                train_loader, adapter, vae, unet, prompt_embeds, prompt_to_class, num_classes,
                scheduler, eargs, args.img_size, args.logit_scale, use_amp, opt, scaler, reduce=args.reduce, bar_desc=f"[S{split_idx:03d}] E{epoch}/{args.epochs} • train"
            )
            val_acc = validate(
                val_loader, adapter, vae, unet, prompt_embeds, prompt_to_class, num_classes,
                scheduler, eargs, args.img_size, use_amp, reduce=args.reduce, bar_desc=f"[S{split_idx:03d}] E{epoch}/{args.epochs} • val"
            )
            tqdm.write(f"[Split {split_idx:03d}] [E{epoch:02d}] train_loss={train_loss:.4f}  "
                  f"train_acc={train_acc:.4f}  val_acc={val_acc:.4f}")

            if (val_acc > best_val_acc) or math.isclose(val_acc, best_val_acc, rel_tol=1e-6):
                best_val_acc = val_acc
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
                tqdm.write(f"-> [Split {split_idx:03d}] neues Best-Model: {best_path}")

        cv_scores.append(best_val_acc)
        if best_val_acc > best_global_acc:
            best_global_acc = best_val_acc
            best_global_ckpt = best_path

    # Aggregation
    mean_acc = float(np.mean(cv_scores)) if len(cv_scores) > 0 else float("nan")
    std_acc  = float(np.std(cv_scores, ddof=1)) if len(cv_scores) > 1 else 0.0
    tqdm.write(f"[RSKF] splits={args.cv_splits} repeats={args.cv_repeats}  "
          f"mean_val_acc={mean_acc:.4f}  std={std_acc:.4f}  einzel={cv_scores}")
    tqdm.write(f"[RSKF] best_overall_acc={best_global_acc:.4f}  ckpt={best_global_ckpt}")

    # Optionale finale Evaluierung auf externem, FIXEM Set
    if args.final_eval:
        assert args.data_test and args.val_csv, \
            "--final_eval verlangt --data_test und --val_csv"
        # Bestes CV-Checkpoint laden
        ckpt = torch.load(best_global_ckpt, map_location="cpu")
        adapter = ControlNetAdapterWrapper(
            controlnet_cfg=ckpt["controlnet_cfg"],
            in_channels=2,
            out_size=ckpt["img_size"],
            target_T=64,
            stride_T=4
        ).to(device)
        adapter.load_state_dict(ckpt["adapter_state_dict"], strict=True)
        adapter.eval()

        final_ds = ThzDataset(args.data_test, args.val_csv, is_train=False)
        final_loader = DataLoader(final_ds, batch_size=1, shuffle=False,
                                  num_workers=args.num_workers, pin_memory=True)
        final_acc = validate(
            final_loader, adapter, vae, unet, prompt_embeds, prompt_to_class, num_classes,
            scheduler, eargs, args.img_size,
            use_amp=(device=="cuda" and args.dtype=="float16"),
            reduce=args.reduce
        )
        tqdm.write(f"[FINAL] accuracy on fixed set ({args.val_csv} @ {args.data_test}): {final_acc:.4f}")



if __name__ == "__main__":
    main()