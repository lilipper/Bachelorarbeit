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
from torch.cuda.amp import GradScaler

from sklearn.model_selection import RepeatedStratifiedKFold
from torchvision import models as tv_models  # torchvision backbones

# Your modules
from diffusion.datasets import ThzDataset
from thz_front_rgb_head import THzToRGBHead
from adapter.help_functions import read_csv_pairs, write_csv_pairs


device = "cuda" if torch.cuda.is_available() else "cpu"


# --------------------- Utilities ---------------------
def set_seed(seed: int = 42):
    """Set Python/NumPy/Torch seeds for reproducibility."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _stats_from_weights(weights, fallback_mean=(0.485, 0.456, 0.406), fallback_std=(0.229, 0.224, 0.225)):
    """Extract mean/std from torchvision Weights meta; fallback to ImageNet stats if not present."""
    if weights is None:
        return fallback_mean, fallback_std
    meta = getattr(weights, "meta", None)
    if meta and "mean" in meta and "std" in meta:
        return tuple(meta["mean"]), tuple(meta["std"])
    return fallback_mean, fallback_std


def build_backbone(name: str, num_classes: int, pretrained: bool = True):
    """
    Build a torchvision backbone with ImageNet-1k weights:
      - resnet50  -> ResNet50_Weights.IMAGENET1K_V1
      - vit_b16   -> ViT_B_16_Weights.IMAGENET1K_SWAG_E2E_V1
      - vit_b32   -> ViT_B_32_Weights.IMAGENET1K_V1
      - convnext_tiny -> ConvNeXt_Tiny_Weights.IMAGENET1K_V1
    Returns: (model, expected_input_size, (mean, std))
    """
    if name == "resnet50":
        weights = tv_models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        model = tv_models.resnet50(weights=weights)
        in_feat = model.fc.in_features
        model.fc = nn.Linear(in_feat, num_classes)
        mean, std = _stats_from_weights(weights)
        print(f"[build_backbone] Built ResNet-50 (pretrained={pretrained}) with ImageNet mean/std: {mean}, {std}")
        return model, 224, (mean, std)

    elif name == "vit_b16":
        weights = tv_models.ViT_B_16_Weights.IMAGENET1K_SWAG_E2E_V1 if pretrained else None
        model = tv_models.vit_b_16(weights=weights)
        in_feat = model.heads.head.in_features
        model.heads.head = nn.Linear(in_feat, num_classes)
        mean, std = _stats_from_weights(weights)
        print(f"[build_backbone] Built ViT-B/16 (pretrained={pretrained}) with ImageNet mean/std: {mean}, {std}")
        return model, 384, (mean, std)

    elif name == "vit_b32":
        weights = tv_models.ViT_B_32_Weights.IMAGENET1K_V1 if pretrained else None
        model = tv_models.vit_b_32(weights=weights)
        in_feat = model.heads.head.in_features
        model.heads.head = nn.Linear(in_feat, num_classes)
        mean, std = _stats_from_weights(weights)
        print(f"[build_backbone] Built ViT-B/32 (pretrained={pretrained}) with ImageNet mean/std: {mean}, {std}")
        return model, 224, (mean, std)

    elif name == "convnext_tiny":
        weights = tv_models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None
        model = tv_models.convnext_tiny(weights=weights)  # don't set num_classes here
        in_feat = model.classifier[2].in_features
        model.classifier[2] = nn.Linear(in_feat, num_classes)
        mean, std = _stats_from_weights(weights)
        print(f"[build_backbone] Built ConvNeXt-Tiny (pretrained={pretrained}) with ImageNet mean/std: {mean}, {std}")
        return model, 224, (mean, std)

    else:
        raise ValueError(f"Unknown backbone: {name}")


def normalize_batch(x, mean, std):
    """Normalize a batch from [0,1] to ImageNet stats."""
    mean_t = x.new_tensor(mean).view(1, 3, 1, 1)
    std_t = x.new_tensor(std).view(1, 3, 1, 1)
    return (x - mean_t) / (std_t + 1e-6)


# ----------------- Train / Validate -----------------

def train_one_epoch(loader, front, backbone, criterion, opt, scaler,
                    torch_dtype, use_amp, img_out_size, learn_front: bool, mean_std):
    """One training epoch with AMP and ImageNet normalization."""
    if learn_front:
        front.train()
    else:
        front.eval()
    backbone.train()

    running_loss, running_acc, seen = 0.0, 0, 0
    print("[train_one_epoch] Starting training iteration...")
    for step, (vol, label, _) in enumerate(loader, start=1):
        vol = vol.to(device)
        if vol.dim() == 6:  # [B,1,2,T,H,W] -> [B,2,T,H,W]
            vol = vol.squeeze(1)
        label = label.to(device).long()

        opt.zero_grad(set_to_none=True)

        with torch.autocast(device_type="cuda", dtype=torch_dtype, enabled=use_amp):
            # THz -> RGB in ~[0,1]
            img_rgb = front(vol)  # [B,3,H,W]
            if img_out_size is not None and (img_rgb.shape[-2] != img_out_size or img_rgb.shape[-1] != img_out_size):
                img_rgb = F.interpolate(img_rgb, size=(img_out_size, img_out_size),
                                        mode="bilinear", align_corners=False)
            img_in = normalize_batch(img_rgb, *mean_std)
            logits = backbone(img_in)
            loss = criterion(logits, label)

        if scaler.is_enabled():
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        else:
            loss.backward()
            opt.step()

        with torch.no_grad():
            running_loss += loss.item() * vol.size(0)
            preds = torch.argmax(logits, dim=1)
            running_acc += (preds == label).sum().item()
            seen += vol.size(0)

        if step % 20 == 0 or step == 1:
            print(f"[train_one_epoch] step={step}  "
                  f"avg_loss={running_loss/max(1,seen):.4f}  avg_acc={running_acc/max(1,seen):.4f}")

    epoch_loss = running_loss / max(1, seen)
    epoch_acc = running_acc / max(1, seen)
    print(f"[train_one_epoch] Done. epoch_loss={epoch_loss:.4f}  epoch_acc={epoch_acc:.4f}")
    return epoch_loss, epoch_acc


@torch.no_grad()
def validate(loader, front, backbone, torch_dtype, use_amp, img_out_size, mean_std):
    """Validation loop with AMP and ImageNet normalization."""
    front.eval()
    backbone.eval()
    total, correct = 0, 0
    print("[validate] Starting validation iteration...")
    for step, (vol, label, _) in enumerate(loader, start=1):
        vol = vol.to(device)
        if vol.dim() == 6:
            vol = vol.squeeze(1)
        label = label.to(device).long()

        with torch.autocast(device_type="cuda", dtype=torch_dtype, enabled=use_amp):
            img_rgb = front(vol)
            if img_out_size is not None and (img_rgb.shape[-2] != img_out_size or img_rgb.shape[-1] != img_out_size):
                img_rgb = F.interpolate(img_rgb, size=(img_out_size, img_out_size),
                                        mode="bilinear", align_corners=False)
            img_in = normalize_batch(img_rgb, *mean_std)
            logits = backbone(img_in)
            preds = torch.argmax(logits, dim=1)

        correct += (preds == label).sum().item()
        total += label.numel()

        if step % 50 == 0 or step == 1:
            print(f"[validate] step={step}  running_acc={correct/max(1,total):.4f}")

    val_acc = correct / max(1, total)
    print(f"[validate] Done. val_acc={val_acc:.4f}")
    return val_acc


# ----------------------- Main -----------------------

def main():
    parser = argparse.ArgumentParser(description="THz-Adapter + (torchvision ResNet/ViT) Classifier with RSKF + FinalEval")
    # Data
    parser.add_argument("--data_train", type=str, required=True)
    parser.add_argument("--train_csv", type=str, required=True)

    # Model
    parser.add_argument("--backbone", type=str, choices=("resnet50", "vit_b16", "vit_b32", "convnext_tiny"), required=True)
    parser.add_argument("--pretrained", action="store_true", help="Load ImageNet-1k pretrained weights")
    parser.add_argument("--num_classes", type=int, default=2)

    # Training
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=4)

    # LRs / WD
    parser.add_argument("--learn_front", action="store_true", help="Train the THz adapter (front)")
    parser.add_argument("--lr_front", type=float, default=1e-4)
    parser.add_argument("--wd_front", type=float, default=0.0)

    parser.add_argument("--train_backbone", action="store_true", help="Train the backbone classifier")
    parser.add_argument("--lr_backbone", type=float, default=1e-4)
    parser.add_argument("--wd_backbone", type=float, default=0.01)

    # AMP
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=("float16", "bfloat16", "float32"))

    # CV
    parser.add_argument("--cv_splits", type=int, default=5)
    parser.add_argument("--cv_repeats", type=int, default=2)
    parser.add_argument("--cv_seed", type=int, default=42)

    # Final Eval
    parser.add_argument("--final_eval", action="store_true",
                        help="After CV, load global best checkpoint and evaluate on a fixed test set.")
    parser.add_argument("--data_test", type=str, default=None,
                        help="Root directory of the test dataset.")
    parser.add_argument("--test_csv", type=str, default=None,
                        help="CSV (path,label) for the test set. If not set, fallback to --val_csv for compatibility.")
    parser.add_argument("--val_csv", type=str, default=None,
                        help="Fallback CSV for --final_eval when --test_csv is not provided.")

    # Misc
    parser.add_argument("--save_dir", type=str, default="./runs/checkpoints_thz_adapter_cls_rskf_tv")

    args = parser.parse_args()

    print("========== CONFIG ==========")
    for k, v in vars(args).items():
        print(f"{k}: {v}")
    print("============================")

    set_seed(args.cv_seed)

    # AMP settings
    torch_dtype = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[args.dtype]
    use_amp = (device == "cuda") and (args.dtype in ("float16", "bfloat16"))
    scaler_enabled = use_amp
    print(f"[AMP] device={device}  dtype={args.dtype}  use_amp={use_amp}")

    torch.backends.cudnn.benchmark = False

    # Save root
    stamp = datetime.now(ZoneInfo("Europe/Berlin")).strftime("%y%m%d_%H%M")
    save_dir = os.path.join(args.save_dir, f"{args.backbone}_{args.dtype}_{stamp}")
    os.makedirs(save_dir, exist_ok=True)
    print(f"[IO] Save directory: {save_dir}")

    # Read all training pairs (path,label)
    print(f"[IO] Reading training pairs from: {args.train_csv}")
    all_rows = read_csv_pairs(args.train_csv)
    labels = [y for _, y in all_rows]
    print(f"[IO] Found {len(all_rows)} samples in TRAIN CSV.")

    # Prepare RSKF
    rskf = RepeatedStratifiedKFold(
        n_splits=args.cv_splits,
        n_repeats=args.cv_repeats,
        random_state=args.cv_seed
    )
    total_splits = args.cv_splits * args.cv_repeats
    print(f"[CV] Using RepeatedStratifiedKFold: splits={args.cv_splits}, repeats={args.cv_repeats}, total={total_splits}")

    summary_rows = []
    cv_scores = []
    best_global_acc = -1.0
    best_global_ckpt = None

    # ----------- Train across splits -----------
    for split_idx, (idx_tr, idx_va) in enumerate(rskf.split(all_rows, labels), start=1):
        print(f"\n===== [SPLIT {split_idx:03d}/{total_splits}] =====")
        fold_dir = os.path.join(save_dir, f"split_{split_idx:03d}")
        fold_ckpt = os.path.join(fold_dir, "best.pt")
        os.makedirs(fold_dir, exist_ok=True)
        print(f"[IO] Fold directory: {fold_dir}")

        # (Re)build models per fold
        front = THzToRGBHead(in_ch=2, base_ch=32, k_t=5, final_depth=16).to(device)
        backbone, expected_size, mean_std = build_backbone(
            args.backbone, args.num_classes, pretrained=args.pretrained
        )
        backbone = backbone.to(device)
        img_out_size = expected_size if expected_size is not None else 224

        # Optimizer params per fold
        params = []
        if args.learn_front:
            for p in front.parameters():
                p.requires_grad_(True)
            params.append({"params": front.parameters(), "lr": args.lr_front, "weight_decay": args.wd_front})
            print("[OPT] Front parameters will be trained.")
        else:
            for p in front.parameters():
                p.requires_grad_(False)
            front.eval()
            print("[OPT] Front is frozen.")

        if args.train_backbone:
            for p in backbone.parameters():
                p.requires_grad_(True)
            params.append({"params": backbone.parameters(), "lr": args.lr_backbone, "weight_decay": args.wd_backbone})
            print("[OPT] Backbone parameters will be trained.")
        else:
            for p in backbone.parameters():
                p.requires_grad_(False)
            backbone.eval()
            print("[OPT] Backbone is frozen.")

        if len(params) == 0:
            raise ValueError("Nothing to train: enable --learn_front and/or --train_backbone.")

        opt = torch.optim.AdamW(params)
        criterion = nn.CrossEntropyLoss()
        scaler = GradScaler(enabled=scaler_enabled)

        # Prepare split CSVs
        train_rows = [all_rows[i] for i in idx_tr]
        val_rows = [all_rows[i] for i in idx_va]
        fold_train_csv = os.path.join(fold_dir, "train.csv")
        fold_val_csv = os.path.join(fold_dir, "val.csv")
        write_csv_pairs(fold_train_csv, train_rows)
        write_csv_pairs(fold_val_csv, val_rows)
        print(f"[IO] Wrote fold train CSV: {fold_train_csv} (n={len(train_rows)})")
        print(f"[IO] Wrote fold val CSV:   {fold_val_csv} (n={len(val_rows)})")

        # Build datasets/loaders (always from data_train root)
        g = torch.Generator()
        g.manual_seed(args.cv_seed + split_idx)

        train_ds = ThzDataset(args.data_train, fold_train_csv, is_train=True)
        val_ds = ThzDataset(args.data_train, fold_val_csv, is_train=False)
        train_loader = DataLoader(
            train_ds, batch_size=args.batch_size, shuffle=True,
            num_workers=args.num_workers, pin_memory=(device == "cuda"), drop_last=True, generator=g
        )
        val_loader = DataLoader(
            val_ds, batch_size=1, shuffle=False,
            num_workers=args.num_workers, pin_memory=(device == "cuda")
        )
        print(f"[DataLoader] Train batches: ~{len(train_loader)} | Val samples: {len(val_loader)}")

        # Reset best metric per fold
        best_val = -1.0

        for epoch in range(1, args.epochs + 1):
            print(f"\n[Epoch] Split {split_idx:03d} | Epoch {epoch:02d}/{args.epochs}")
            tr_loss, tr_acc = train_one_epoch(
                train_loader, front, backbone, criterion, opt, scaler,
                torch_dtype, use_amp, img_out_size, learn_front=args.learn_front, mean_std=mean_std
            )
            val_acc = validate(val_loader, front, backbone, torch_dtype, use_amp, img_out_size, mean_std=mean_std)
            print(f"[Epoch] Result: train_loss={tr_loss:.4f}  train_acc={tr_acc:.4f}  val_acc={val_acc:.4f}")

            if val_acc >= best_val:
                best_val = val_acc
                torch.save({
                    "split_idx": split_idx,
                    "epoch": epoch,
                    "best_val_acc": best_val,
                    "backbone": args.backbone,
                    "dtype": args.dtype,
                    "img_out_size": img_out_size,
                    "learn_front": args.learn_front,
                    "train_backbone": args.train_backbone,
                    "front_state_dict": front.state_dict(),
                    "backbone_state_dict": backbone.state_dict(),
                    "num_classes": args.num_classes,
                    "fold_train_csv": fold_train_csv,
                    "fold_val_csv": fold_val_csv,
                    "imagenet_mean": mean_std[0],
                    "imagenet_std": mean_std[1],
                }, fold_ckpt)
                print(f"[Checkpoint] Saved best fold checkpoint: {fold_ckpt}  (val_acc={best_val:.4f})")

        # Collect split results
        cv_scores.append(best_val)
        summary_rows.append({
            "split_idx": split_idx,
            "best_val_acc": best_val,
            "fold_dir": fold_dir,
        })
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
        print(f"[FINAL] Global best val_acc (from CV): {ckpt.get('best_val_acc', 'N/A')}")

        # Rebuild front/backbone and load state_dicts
        front = THzToRGBHead(in_ch=2, base_ch=32, k_t=5, final_depth=16).to(device)
        front.load_state_dict(ckpt["front_state_dict"], strict=False)
        front.eval()
        print("[FINAL] Loaded front weights.")
        final_eval_path = os.path.join(save_dir, "final_eval")
        os.makedirs(final_eval_path, exist_ok=True)
        backbone_name = ckpt["backbone"]
        num_classes = ckpt["num_classes"]
        imagenet_mean = tuple(ckpt["imagenet_mean"])
        imagenet_std = tuple(ckpt["imagenet_std"])
        img_out_size = ckpt["img_out_size"]

        backbone, _, _ = build_backbone(backbone_name, num_classes, pretrained=args.pretrained)
        backbone.load_state_dict(ckpt["backbone_state_dict"], strict=False)
        backbone = backbone.to(device).eval()
        print(f"[FINAL] Rebuilt backbone '{backbone_name}' and loaded weights.")

        # Test loader (always from data_test)
        final_ds = ThzDataset(args.data_test, test_csv, is_train=False)
        final_loader = DataLoader(final_ds, batch_size=1, shuffle=False,
                                  num_workers=args.num_workers, pin_memory=(device == "cuda"))
        print(f"[FINAL] Test set: {len(final_loader)} samples | CSV={test_csv} | ROOT={args.data_test}")

        # AMP settings for final eval
        use_amp_final = (device == "cuda") and (ckpt["dtype"] in ("float16", "bfloat16"))
        final_dtype = torch.float16 if ckpt["dtype"] == "float16" else (torch.bfloat16 if ckpt["dtype"] == "bfloat16" else torch.float32)
        print(f"[FINAL] AMP enabled={use_amp_final} dtype={ckpt['dtype']}")

        def get_formatstr(n):
            # get the format string that pads 0s to the left for numbers up to n
            digits = 0
            while n > 0:
                digits += 1
                n //= 10
            return f"{{:0{digits}d}}"

        # Inference
        total, correct = 0, 0
        with torch.no_grad():
            for i, (vol, label, _) in enumerate(final_loader, start=1):
                vol = vol.to(device)
                if vol.dim() == 6:
                    vol = vol.squeeze(1)
                label = label.to(device).long()

                with torch.autocast(device_type="cuda", dtype=final_dtype, enabled=use_amp_final):
                    img_rgb = front(vol)
                    if img_rgb.shape[-2] != img_out_size or img_rgb.shape[-1] != img_out_size:
                        img_rgb = F.interpolate(img_rgb, size=(img_out_size, img_out_size),
                                                mode="bilinear", align_corners=False)
                    img_in = normalize_batch(img_rgb, imagenet_mean, imagenet_std)
                    logits = backbone(img_in)
                    preds = torch.argmax(logits, dim=1)

                correct += (preds == label).sum().item()
                total += label.numel()
                formatstr = get_formatstr(len(final_loader))
                torch.save(
                    dict(preds=preds.cpu(), label=label.cpu()),
                    os.path.join(final_eval_path,  formatstr.format(i) + '.pt')
                )

                if i % 50 == 0 or i == 1:
                    print(f"[FINAL] step={i}  running_acc={correct/max(1,total):.4f}")

        final_acc = correct / max(1, total)
        print(f"[FINAL] Accuracy on fixed set ({test_csv} @ {args.data_test}): {final_acc:.4f}")


if __name__ == "__main__":
    main()
