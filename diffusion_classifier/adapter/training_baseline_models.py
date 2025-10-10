# train_backbone_rskf.py
import os
import sys
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)
import os, math, argparse, numpy as np
from datetime import datetime
from zoneinfo import ZoneInfo
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from torchvision import models
import torch.nn.init as init

from sklearn.model_selection import RepeatedStratifiedKFold

# ===== Abhängig von deinem Projekt =====
from diffusion.datasets import ThzDataset  
from adapter.ControlNet_Adapter_wrapper import ControlNetAdapterWrapper
from pipeline_classifier_with_adapter.classifiers.build_torchvision_backbone import build_torchvision_backbone
from adapter.help_functions import read_csv_pairs, write_csv_pairs, PromptBank, pool_prompt_errors_to_class_errors, pool_prompt_errors_to_class_errors_batch

device = "cuda" if torch.cuda.is_available() else "cpu"

# ========= Utils =========
def set_seed(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ========= Wrapper für Backbones =========
class BackboneClassifier(nn.Module):
    def __init__(self, num_classes: int, controlnet_cfg: dict, backbone: str = "resnet50", img_size: int = 224, pretrained: bool = True):
        super().__init__()
        self.adapter = ControlNetAdapterWrapper(
            controlnet_cfg=controlnet_cfg,
            in_channels=2,
            out_size=img_size,
            target_T=64,
            stride_T=4
        )
        self.backbone_name = backbone
        self.pretrained = pretrained
        self.imgnet_mean = torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1)
        self.imgnet_std  = torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1)

        if backbone == "resnet50":
            weights = models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
            self.backbone = models.resnet50(weights=weights)
            in_feat = self.backbone.fc.in_features
            self.backbone.fc = nn.Linear(in_feat, num_classes)
        elif backbone == "vit_b16":
            weights = models.ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None
            self.backbone = models.vit_b_16(weights=weights)
            in_feat = self.backbone.heads.head.in_features
            self.backbone.heads.head = nn.Linear(in_feat, num_classes)
        elif backbone == "vit_b32":
            weights = models.ViT_B_32_Weights.IMAGENET1K_V1 if pretrained else None
            self.backbone = models.vit_b_32(weights=weights)
            in_feat = self.backbone.heads.head.in_features
            self.backbone.heads.head = nn.Linear(in_feat, num_classes)
        elif backbone == "convnext_tiny":
            weights = models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None
            self.backbone = models.convnext_tiny(weights=weights, num_classes=2)
            in_feat = self.backbone.classifier[2].in_features
            self.backbone.classifier[2] = nn.Linear(in_feat, num_classes)
        else:
            raise ValueError("backbone must be one of {'resnet50','vit_b16'}")

    def freeze_backbone(self, freeze: bool = True):
        for p in self.backbone.parameters():
            p.requires_grad = not (freeze)

    def forward(self, vol: torch.Tensor) -> torch.Tensor:
        x = self.adapter(vol)  
        if self.pretrained:
            # ImageNet-Normalisierung
            mean = self.imgnet_mean.to(x.device, x.dtype)
            std  = self.imgnet_std.to(x.device, x.dtype)
            x = (x - mean) / std
        return self.backbone(x)

# ========= Train/Eval Loops =========
def train_one_epoch(train_loader, model, use_amp, opt, scaler, bar_desc=None):
    model.train()
    running_loss, running_acc, n_seen = 0.0, 0.0, 0
    it = tqdm(train_loader, desc=bar_desc or "train", leave=False, ncols=0)
    for vol, label, _ in it:
        vol = vol.to(device)
        if vol.dim() == 6:          # [B, 1, 2, T, H, W] -> [B, 2, T, H, W]
            vol = vol.squeeze(1)
        label = label.to(device).long()

        opt.zero_grad()
        with torch.amp.autocast("cuda", dtype=torch.float16, enabled=use_amp):
            logits = model(vol)
            loss = F.cross_entropy(logits, label)

        if torch.isfinite(loss):
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(opt)
            scaler.update()

        with torch.no_grad():
            running_loss += loss.item() * vol.size(0)
            preds = logits.argmax(dim=1)
            running_acc += (preds == label).sum().item()
            n_seen += vol.size(0)
            it.set_postfix(loss=f"{running_loss/max(1,n_seen):.4f}",
                           acc=f"{running_acc/max(1,n_seen):.4f}")

    return running_loss / max(1, n_seen), running_acc / max(1, n_seen)

def get_formatstr(n):
    # get the format string that pads 0s to the left of a number, which is at most n
    digits = 0
    while n > 0:
        digits += 1
        n //= 10
    return f"{{:0{digits}d}}"

@torch.no_grad()
def validate(val_loader, model, use_amp, bar_desc=None, final_eval: bool = False, run_folder: str = None):
    model.eval()
    total, correct, loss_sum = 0, 0, 0.0
    it = tqdm(val_loader, desc=bar_desc or "val", leave=False, ncols=0)
    for i, (vol, label, _) in enumerate(it):
        vol = vol.to(device)
        if vol.dim() == 6:          # [B, 1, 2, T, H, W] -> [B, 2, T, H, W]
            vol = vol.squeeze(1)
        label = label.to(device).long()
        with torch.amp.autocast("cuda", dtype=torch.float16, enabled=use_amp):
            logits = model(vol)
            loss = F.cross_entropy(logits, label)
        loss_sum += loss.item()
        correct += (logits.argmax(1) == label).sum().item()
        total += label.size(0)
        it.set_postfix(acc=f"{correct/max(1,total):.4f}")
        if final_eval:
            formatstr = get_formatstr(len(it) - 1)
            torch.save(dict(logits=logits.cpu(), label=label.cpu()), os.path.join(run_folder, formatstr.format(i) + '.pt'))
    return correct / max(1, total), loss_sum / max(1, total)

# ========= Main mit RSKF =========
def main():
    ap = argparse.ArgumentParser()
    # Daten
    ap.add_argument("--data_train", type=str, required=True)
    ap.add_argument("--data_test",  type=str, default="")
    ap.add_argument("--train_csv",  type=str, required=True)
    ap.add_argument("--val_csv",    type=str, default="")
    # Backbone
    ap.add_argument("--backbone", type=str, default="resnet50", choices=("resnet50","vit_b16","vit_b32","convnext_tiny"))
    ap.add_argument("--img_size", type=int, default=512, choices=(256, 512))
    ap.add_argument("--pretrained", action="store_true", help="ImageNet-Pretrained Gewichte")
    ap.add_argument("--train_all", action="store_true", help="Backbone komplett finetunen (statt nur Kopf)")
    ap.add_argument("--train_adapter", action="store_true", help="Adapter mittrainieren (sonst eingefroren)")
    # Training
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--dtype", type=str, default="float16", choices=("float16","float32","bfloat16"))
    ap.add_argument("--save_dir", type=str, default="./runs/checkpoints_backbone")
    # Cross-Validation
    ap.add_argument("--cv_splits", type=int, default=5)
    ap.add_argument("--cv_repeats", type=int, default=3)
    ap.add_argument("--cv_seed", type=int, default=42)
    # Finale Evaluation
    ap.add_argument("--final_eval", action="store_true")

    args = ap.parse_args()
    set_seed(args.cv_seed)
    print("_" * 60, "\n")
    
    print("Train Baseline Backbone mit RSKF")
    print(f"Backbone: {args.backbone}  img_size: {args.img_size}  pretrained: {args.pretrained} train_all: {args.train_all} train_adapter: {args.train_adapter}")
    # Klassenanzahl aus CSV ableiten
    all_rows = read_csv_pairs(args.train_csv)
    labels = [y for _, y in all_rows]

    stamp = datetime.now(ZoneInfo("Europe/Berlin")).strftime("%y%m%d_%H%M")
    save_dir = os.path.join(args.save_dir, f"{args.backbone}_rskf_{args.cv_splits}x{args.cv_repeats}_{stamp}")
    os.makedirs(save_dir, exist_ok=True)
    print(f"Checkpoints und Logs werden gespeichert in: {save_dir}")
    # RSKF vorbereiten
    rskf = RepeatedStratifiedKFold(
        n_splits=args.cv_splits,
        n_repeats=args.cv_repeats,
        random_state=args.cv_seed
    )

    cv_scores = []
    best_global_acc = -1.0
    best_global_ckpt = None
    total_splits = args.cv_splits * args.cv_repeats

    # AMP-Flag
    use_amp = (device == "cuda" and args.dtype == "float16")

    for split_idx, (idx_tr, idx_va) in tqdm(
        enumerate(rskf.split(all_rows, labels), start=1),
        total=total_splits, desc="RSKF splits", ncols=0):

        fold_dir = os.path.join(save_dir, f"split_{split_idx:03d}")
        os.makedirs(fold_dir, exist_ok=True)

        train_rows = [all_rows[i] for i in idx_tr]
        val_rows   = [all_rows[i] for i in idx_va]
        fold_train_csv = os.path.join(fold_dir, "train.csv")
        fold_val_csv   = os.path.join(fold_dir, "val.csv")
        write_csv_pairs(fold_train_csv, train_rows)
        write_csv_pairs(fold_val_csv,   val_rows)

        train_ds = ThzDataset(args.data_train, fold_train_csv, is_train=True)
        val_ds   = ThzDataset(args.data_train, fold_val_csv,   is_train=False)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                                  num_workers=args.num_workers, pin_memory=True)
        val_loader   = DataLoader(val_ds, batch_size=1, shuffle=False,
                                  num_workers=args.num_workers, pin_memory=True)
        num_classes = train_ds.num_classes

        # Modell je Split neu
        controlnet_cfg = dict(
            spatial_dims=3,
            num_res_blocks=(2, 2, 2, 2),
            num_channels=(32, 64, 64, 64),
            attention_levels=(False, False, False, False),
            conditioning_embedding_in_channels=2,
            conditioning_embedding_num_channels=(32, 64, 64, 64),
            with_conditioning=False,
        )
        model = BackboneClassifier(
            num_classes=num_classes,
            controlnet_cfg=controlnet_cfg,
            backbone=args.backbone,
            img_size=args.img_size,
            pretrained=args.pretrained
        ).to(device)

        # nur Kopf trainieren? (Standard) – oder alles mit --train_all
        if not args.train_all:
            model.freeze_backbone(True)  # Backbone einfrieren
            # spezifisch Kopf freigeben
            if args.backbone == "resnet50":
                for p in model.backbone.fc.parameters():
                    p.requires_grad = True
            else:
                for p in model.backbone.heads.parameters():
                    p.requires_grad = True
            if not args.train_adapter:
                for p in model.adapter.parameters():
                    p.requires_grad = True

        opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                                lr=args.lr, weight_decay=args.weight_decay)
        scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

        best_val_acc = -1.0
        best_path = os.path.join(fold_dir, "backbone_best.pt")

        for epoch in tqdm(range(1, args.epochs + 1),
                          desc=f"[S{split_idx:03d}] epochs", leave=False, ncols=0):
            train_loss, train_acc = train_one_epoch(
                train_loader, model, use_amp, opt, scaler,
                bar_desc=f"[S{split_idx:03d}] E{epoch}/{args.epochs} • train"
            )
            val_acc, val_loss = validate(
                val_loader, model, use_amp,
                bar_desc=f"[S{split_idx:03d}] E{epoch}/{args.epochs} • val"
            )
            print(f"[Split {split_idx:03d}] [E{epoch:02d}] "
                       f"train_loss={train_loss:.4f}  train_acc={train_acc:.4f}  "
                       f"val_acc={val_acc:.4f}  val_loss={val_loss:.4f}")

            if (val_acc > best_val_acc) or math.isclose(val_acc, best_val_acc, rel_tol=1e-6):
                best_val_acc = val_acc
                torch.save({
                    "epoch": epoch,
                    "state_dict": model.state_dict(),
                    "optimizer_state_dict": opt.state_dict(),
                    "best_val_acc": best_val_acc,
                    "backbone": args.backbone,
                    "img_size": args.img_size,
                    "num_classes": num_classes,
                    "train_all": args.train_all,
                    "train_adapter": args.train_adapter,
                    "controlnet_cfg": controlnet_cfg,
                    "pretrained": args.pretrained,
                }, best_path)

                # 2) Adapter-Only-Checkpoint (leichtgewichtiger)
                adapter_path = os.path.join(fold_dir, "adapter_best.pt")
                torch.save({
                    "epoch": epoch,
                    "adapter_state_dict": model.adapter.state_dict(),
                    "controlnet_cfg": controlnet_cfg,
                    "img_size": args.img_size,
                    "in_channels": 2,
                    "target_T": 64,
                    "stride_T": 4,
                }, adapter_path)
                print(f"-> [Split {split_idx:03d}] neues Best-Model: {best_path}")

        cv_scores.append(best_val_acc)
        if best_val_acc > best_global_acc:
            best_global_acc = best_val_acc
            best_global_ckpt = best_path

    mean_acc = float(np.mean(cv_scores)) if len(cv_scores) > 0 else float("nan")
    std_acc  = float(np.std(cv_scores, ddof=1)) if len(cv_scores) > 1 else 0.0
    print(f"[RSKF] splits={args.cv_splits} repeats={args.cv_repeats}  "
               f"mean_val_acc={mean_acc:.4f}  std={std_acc:.4f}  einzel={cv_scores}")
    print(f"[RSKF] best_overall_acc={best_global_acc:.4f}  ckpt={best_global_ckpt}")

    # Optionale finale Evaluierung
    if args.final_eval:
        assert args.data_test and args.val_csv, "--final_eval verlangt --data_test und --val_csv"
        ckpt = torch.load(best_global_ckpt, map_location="cpu")
        model = BackboneClassifier(
            num_classes=ckpt["num_classes"],
            backbone=ckpt["backbone"],
            img_size=ckpt["img_size"],
            controlnet_cfg=ckpt["controlnet_cfg"],
            pretrained=False
        ).to(device)
        model.load_state_dict(ckpt["state_dict"], strict=True)
        model.eval()
        adapter_only = torch.load(os.path.join(os.path.dirname(best_global_ckpt), "adapter_best.pt"), map_location="cpu")
        model.adapter.load_state_dict(adapter_only["adapter_state_dict"], strict=True)

        final_ds = ThzDataset(args.data_test, args.val_csv, is_train=False)
        final_loader = DataLoader(final_ds, batch_size=1, shuffle=False,
                                  num_workers=2, pin_memory=True)

        use_amp_final = (device=="cuda")
        final_acc, _ = validate(final_loader, model, use_amp_final, bar_desc="[FINAL] val", final_eval=True, run_folder=os.path.join(fold_dir, "final_eval"))
        print(f"[FINAL] accuracy: {final_acc:.4f}")

if __name__ == "__main__":
    main()
