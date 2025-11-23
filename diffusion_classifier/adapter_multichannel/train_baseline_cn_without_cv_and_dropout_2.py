"""
Train a THz-to-RGB ControlNet-based adapter + torchvision backbone classifier
without cross-validation, using a two-stage optimizer schedule and optional
early stopping.

This script is a stronger baseline variant that:
  - builds a 3D ControlNet-based THz-to-RGB front-end (ControlNetAdapterWrapper),
  - builds a torchvision backbone (ResNet, ViT, ConvNeXt) with a dropout head,
  - trains on the full training set (no CV),
  - first uses AdamW + OneCycleLR, then switches to SGD + CosineAnnealingLR,
  - supports early stopping based on accuracy and loss thresholds,
  - optionally runs a final evaluation and aggregates predictions with
    `evaluate_predictions`.

Workflow:
    1. Load the THz dataset from `--data_train` and `--train_csv`.
    2. Build the ControlNetAdapterWrapper front-end and a chosen backbone
       (resnet50, vit_b16, vit_b32, convnext_tiny) with an ImageNet-style head.
    3. Configure two optimizers:
       - AdamW + OneCycleLR for the first 60% of epochs,
       - SGD + CosineAnnealingLR for the remaining 40% of epochs.
    4. Train on the full training set with AMP and ImageNet normalization,
       track best training accuracy and save the best checkpoint.
    5. Optionally, if `--final_eval` is set:
       - reload the best checkpoint,
       - evaluate on a fixed test set,
       - save per-sample prediction files and call `evaluate_predictions(...)`
         to compute evaluation metrics.

How to run:
    python train_baseline_cn_without_cv_and_dropout_2.py \\
        --data_train /path/to/thz_data \\
        --train_csv /path/to/train.csv \\
        --backbone vit_b32 \\
        --pretrained \\
        --epochs 200 \\
        --batch_size 2 \\
        --learn_front \\
        --train_backbone \\
        --lr_front 0.05 \\
        --lr_backbone 0.05 \\
        --acc_threshold 0.98 \\
        --loss_threshold 0.05 \\
        --final_eval \\
        --data_test /path/to/test_data \\
        --test_csv /path/to/test.csv

Key arguments:
    --data_train (str)      Root directory containing the THz training volumes.
    --train_csv (str)       CSV with (path,label) pairs for the training set.
    --backbone (str)        Backbone name: resnet50 | vit_b16 | vit_b32 | convnext_tiny.
    --pretrained            Load ImageNet-1k pretrained weights for the backbone.
    --num_classes (int)     Number of output classes.
    --epochs (int)          Total number of training epochs.
    --batch_size (int)      Training batch size.
    --num_workers (int)     Number of DataLoader workers.
    --acc_threshold (float) Early-stopping accuracy threshold on the training set.
    --loss_threshold (float)Early-stopping loss threshold on the training set.
    --learn_front           Train the THz adapter front-end.
    --lr_front (float)      Base learning rate for the adapter (used in both phases).
    --wd_front (float)      Weight decay for the adapter.
    --train_backbone        Train the backbone classifier.
    --lr_backbone (float)   Base learning rate for the backbone (used in both phases).
    --wd_backbone (float)   Weight decay for the backbone.
    --dtype (str)           AMP compute dtype: float16 | bfloat16 | float32.
    --seed (int)            Global random seed.
    --dropout_p (float)     Dropout probability for adapter/backbone heads.
    --final_eval            Run a final evaluation after training.
    --data_test (str)       Root directory of the test dataset.
    --test_csv (str)        CSV with (path,label) pairs for the test set.
    --val_csv (str)         Fallback CSV for final_eval when --test_csv is not provided.
    --save_dir (str)        Base directory for checkpoints, logs, and evaluation outputs.
"""


import os
import argparse
import csv
import numpy as np
from datetime import datetime
from zoneinfo import ZoneInfo

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR, CosineAnnealingLR
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import models as tv_models 

from diffusion.datasets import ThzDataset
from adapter.ControlNet import ControlNet
from pipeline_classifier_with_adapter.eval_the_pipeline_results import evaluate_predictions



device = "cuda" if torch.cuda.is_available() else "cpu"

controlnet_cfg = dict(
            spatial_dims=3,
            num_res_blocks=(2, 2, 2, 2),
            num_channels=(32, 64, 64, 64),
            attention_levels=(False, False, False, False),
            conditioning_embedding_in_channels=2,
            conditioning_embedding_num_channels=(32, 64, 64, 64),
            with_conditioning=False,
        )

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


def build_backbone(name: str, num_classes: int, pretrained: bool = True, dropout_p: float = 0.2):
    if name == "resnet50":
        weights = tv_models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        model = tv_models.resnet50(weights=weights)
        in_feat = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(dropout_p),
            nn.Linear(in_feat, num_classes)
        )
        mean, std = _stats_from_weights(weights)
        print(f"[build_backbone] Built ResNet-50 (pretrained={pretrained}) with ImageNet mean/std: {mean}, {std}")
        return model, 224, (mean, std)

    elif name == "vit_b16":
        weights = tv_models.ViT_B_16_Weights.IMAGENET1K_SWAG_E2E_V1 if pretrained else None
        model = tv_models.vit_b_16(weights=weights)
        in_feat = model.heads.head.in_features
        model.heads.head = nn.Sequential(
            nn.Dropout(dropout_p),
            nn.Linear(in_feat, num_classes)
        )
        mean, std = _stats_from_weights(weights)
        print(f"[build_backbone] Built ViT-B/16 (pretrained={pretrained}) with ImageNet mean/std: {mean}, {std}")
        return model, 384, (mean, std)

    elif name == "vit_b32":
        weights = tv_models.ViT_B_32_Weights.IMAGENET1K_V1 if pretrained else None
        model = tv_models.vit_b_32(weights=weights)
        in_feat = model.heads.head.in_features
        model.heads.head = nn.Sequential(
            nn.Dropout(dropout_p),
            nn.Linear(in_feat, num_classes)
        )
        mean, std = _stats_from_weights(weights)
        print(f"[build_backbone] Built ViT-B/32 (pretrained={pretrained}) with ImageNet mean/std: {mean}, {std}")
        return model, 224, (mean, std)

    elif name == "convnext_tiny":
        weights = tv_models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None
        model = tv_models.convnext_tiny(weights=weights) 
        in_feat = model.classifier[2].in_features
        model.classifier[2] = nn.Sequential(
            nn.Dropout(dropout_p),
            nn.Linear(in_feat, num_classes)
        )
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

class ControlNetAdapterWrapper(torch.nn.Module):
    """
    Nimmt Volumen [B,1,T,H,W] in [0,1] und gibt ein Bild [B,3,512,512] in [0,1] aus.
    Reduziert T vor dem 3D-Netz per AvgPool, um Speicher zu schonen.
    """
    def __init__(self, controlnet_cfg, in_channels=2, out_size=512, target_T=256, mid_channels=8, stride_T=3, dropout_p=0.2):
        super().__init__()
        self.out_size = out_size
        self.target_T = target_T
        self.in_channels = in_channels
        self.p = dropout_p
        self.dropout = nn.Dropout3d(p=self.p)

        self.downT = nn.Sequential(
            nn.Conv3d(in_channels, mid_channels, kernel_size=(5,3,3), stride=(stride_T,1,1), padding=(2,1,1), bias=False),
            nn.SiLU(),
            nn.Conv3d(mid_channels, in_channels, kernel_size=1, stride=1, padding=0, bias=False)
        )

        num_downsamples = len(controlnet_cfg.get("num_channels", ())) - 1
        downsample_factor = 2 ** num_downsamples
        final_depth = target_T // downsample_factor

        print(f"Adapter konfiguriert: Eingangstiefe -> {stride_T}x Reduktion -> {target_T} -> ControlNet-Flaschenhals-Tiefe: {final_depth}")

        cfg = controlnet_cfg.copy()
        cfg['in_channels'] = self.in_channels
        cfg['conditioning_embedding_in_channels'] = self.in_channels
        cfg['final_depth'] = final_depth

        self.net = ControlNet(**cfg)

    def forward(self, vol):  
        vol = self.downT(vol)     
        vol = self.dropout(vol)


        if self.target_T is not None and vol.shape[2] != self.target_T:
            vol = F.interpolate(vol, size=(self.target_T, vol.shape[-2], vol.shape[-1]),
                                mode="trilinear", align_corners=False)

        B, C, Tp, H, W = vol.shape
        x_stub = torch.zeros((B, C, Tp, H, W), device=vol.device, dtype=vol.dtype)
        t_stub = torch.zeros((B,), device=vol.device, dtype=torch.long)

        rgb = self.net(x_stub, t_stub, controlnet_cond=vol, conditioning_scale=1.0, context=None) 

        if (rgb.shape[-2], rgb.shape[-1]) != (self.out_size, self.out_size):
            rgb = F.interpolate(rgb, size=(self.out_size, self.out_size), mode="bilinear", align_corners=False)

        return rgb

# ----------------- Train / Validate -----------------

def train_one_epoch(loader, front, backbone, criterion, opt, sched, scaler,
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
        if vol.dim() == 6:  
            vol = vol.squeeze(1)
        label = label.to(device).long()

        opt.zero_grad(set_to_none=True)

        with torch.autocast(device_type="cuda", dtype=torch_dtype, enabled=use_amp):
            img_rgb = front(vol)  
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
        if isinstance(sched, (torch.optim.lr_scheduler.OneCycleLR, torch.optim.lr_scheduler.CyclicLR)):
            sched.step()

        if step % 20 == 0 or step == 1:
            print(f"[train_one_epoch] step={step}  "
                  f"avg_loss={running_loss/max(1,seen):.4f}  avg_acc={running_acc/max(1,seen):.4f}")
    if isinstance(sched, (torch.optim.lr_scheduler.CosineAnnealingLR,
                              torch.optim.lr_scheduler.StepLR,
                              torch.optim.lr_scheduler.ReduceLROnPlateau)):
        sched.step()
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
    parser.add_argument("--acc_threshold", type=float, default=0.98)
    parser.add_argument("--loss_threshold", type=float, default=0.05)

    # LRs / WD
    parser.add_argument("--learn_front", action="store_true", help="Train the THz adapter (front)")
    parser.add_argument("--lr_front", type=float, default=5e-2)
    parser.add_argument("--wd_front", type=float, default=0.01)

    parser.add_argument("--train_backbone", action="store_true", help="Train the backbone classifier")
    parser.add_argument("--lr_backbone", type=float, default=5e-2)
    parser.add_argument("--wd_backbone", type=float, default=0.01)

    # AMP
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=("float16", "bfloat16", "float32"))
    parser.add_argument("--seed", type=int, default=42, help="Global random seed")
    parser.add_argument("--dropout_p", type=float, default=0.2, help="Dropout probability for classifier head.")

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

    set_seed(args.seed)
    acc_threshold = args.acc_threshold
    loss_threshold = args.loss_threshold

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
    
    backbone, expected_size, mean_std = build_backbone(
                args.backbone, args.num_classes, pretrained=args.pretrained, dropout_p=args.dropout_p
            )
    # (Re)build models per fold
    front = ControlNetAdapterWrapper(
        controlnet_cfg=controlnet_cfg,
        in_channels=2,
        out_size=expected_size,
        target_T=64,
        stride_T=4,
        dropout_p=args.dropout_p
    ).to(device)
    
    
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

    
    
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler('cuda', enabled=scaler_enabled)

    train_ds = ThzDataset(args.data_train, args.train_csv, is_train=True)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=(device == "cuda"), drop_last=True
    )
    print(f"[DataLoader] Train batches: ~{len(train_loader)}")

    swap_number = int(args.epochs *0.6)

    optimizer1 = optim.AdamW(params, lr=args.lr_front, weight_decay=args.wd_front)
    scheduler1 = OneCycleLR(
        optimizer1,
        max_lr=args.lr_front,
        epochs=int(swap_number),
        steps_per_epoch=len(train_loader),
        pct_start=0.3
    )

    optimizer2 = optim.SGD(params, lr=args.lr_front, momentum=0.9, weight_decay=args.wd_front)
    scheduler2 = CosineAnnealingLR(optimizer2, T_max=args.epochs - int(swap_number), eta_min=1e-5)

    def combined_optimizer(step):
        if step < swap_number:
            return optimizer1, scheduler1
        else:
            return optimizer2, scheduler2

    # Reset best metric per fold
    best_val = -1.0
    best_model_dir = os.path.join(save_dir, "best model")
    os.makedirs(best_model_dir, exist_ok=True)
    path_ckpt = os.path.join(best_model_dir, "best_checkpoint.pt")
    for epoch in range(1, args.epochs + 1):
        print(f"\n[Epoch] {epoch:02d}/{args.epochs}")
        opt, sched = combined_optimizer(epoch - 1)
        tr_loss, tr_acc = train_one_epoch(
            train_loader, front, backbone, criterion, opt, sched, scaler,
            torch_dtype, use_amp, img_out_size, learn_front=args.learn_front, mean_std=mean_std
        )
        if tr_acc >= best_val:
            best_val = tr_acc
            torch.save({
                "epoch": epoch,
                "best_val_acc": best_val,
                "backbone": args.backbone,
                "dtype": args.dtype,
                "img_out_size": img_out_size,
                "learn_front": args.learn_front,
                "train_backbone": args.train_backbone,
                "adapter_state_dict": front.state_dict(),
                "backbone_state_dict": backbone.state_dict(),
                "num_classes": args.num_classes,
                "imagenet_mean": mean_std[0],
                "imagenet_std": mean_std[1],
            }, path_ckpt)
            print(f"[Checkpoint] Saved best fold checkpoint: {path_ckpt}  (acc={best_val:.4f})")
        if tr_acc >= acc_threshold and tr_loss <= loss_threshold:
            print("✅ Hit stop.")
            break

    # ------------- FINAL EVAL (optional) -------------
    if args.final_eval:
        test_csv = args.test_csv if args.test_csv is not None else args.val_csv
        assert args.data_test and test_csv, "--final_eval requires --data_test and (--test_csv or --val_csv)"
        print("[FINAL] Starting final evaluation...")
        print(f"[FINAL] Loading global best checkpoint: {path_ckpt}")
        ckpt = torch.load(path_ckpt, map_location="cpu")
        print(f"[FINAL] Global best val_acc (from CV): {ckpt.get('best_val_acc', 'N/A')}")

        front = ControlNetAdapterWrapper(
            controlnet_cfg=controlnet_cfg,
            in_channels=2,
            out_size=expected_size,
            target_T=64,
            stride_T=4
        ).to(device)
        front.load_state_dict(ckpt["adapter_state_dict"], strict=False)
        front.eval()
        print("[FINAL] Loaded front weights.")
        final_eval_path = os.path.join(save_dir, "final_eval")
        os.makedirs(final_eval_path, exist_ok=True)
        backbone_name = ckpt["backbone"]
        num_classes = ckpt["num_classes"]
        imagenet_mean = tuple(ckpt["imagenet_mean"])
        imagenet_std = tuple(ckpt["imagenet_std"])

        backbone, img_out_size, _ = build_backbone(backbone_name, num_classes, pretrained=args.pretrained)
        state_backbone = ckpt["backbone_state_dict"]

        missing, unexpected = backbone.load_state_dict(state_backbone, strict=False)
        if missing or unexpected:
            print("[FINAL][WARN] load_state_dict: missing:", missing, "| unexpected:", unexpected)

        backbone = backbone.to(device).eval()
        print(f"[FINAL] Rebuilt backbone '{backbone_name}' and loaded weights.")

        final_ds = ThzDataset(args.data_test, test_csv, is_train=False)
        final_loader = DataLoader(final_ds, batch_size=1, shuffle=False,
                                  num_workers=args.num_workers, pin_memory=(device == "cuda"))
        print(f"[FINAL] Test set: {len(final_loader)} samples | CSV={test_csv} | ROOT={args.data_test}")

        use_amp_final = (device == "cuda") and (ckpt["dtype"] in ("float16", "bfloat16"))
        final_dtype = torch.float16 if ckpt["dtype"] == "float16" else (torch.bfloat16 if ckpt["dtype"] == "bfloat16" else torch.float32)
        print(f"[FINAL] AMP enabled={use_amp_final} dtype={ckpt['dtype']}")

        def get_formatstr(n):
            digits = 0
            while n > 0:
                digits += 1
                n //= 10
            return f"{{:0{digits}d}}"


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
        evaluate_predictions(final_eval_path, args.save_dir)


if __name__ == "__main__":
    main()
