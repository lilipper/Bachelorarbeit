# train_backbone_rskf.py
import os
import sys
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

import math, argparse, numpy as np
from datetime import datetime
from zoneinfo import ZoneInfo
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import models

from sklearn.model_selection import RepeatedStratifiedKFold

# ===== projekt-spezifisch =====
from diffusion.datasets import ThzDataset
from adapter.help_functions import read_csv_pairs, write_csv_pairs
from thz_front_rgb_head import THzToRGBHead  # deine Front (3D->2D RGB)

device = "cuda" if torch.cuda.is_available() else "cpu"

# -----------------------------------------------------------
# ResNet: Konditions-Adapter + Injektions-Session
# -----------------------------------------------------------
class ResNetCondAdapter(nn.Module):
    """
    Baut aus conditioning_rgb [B,3,H,W] eine Pyramid (5 Stufen) und hält 1x1-Projektoren
    auf die Stage-Kanäle des ResNet (conv1 + layer1..4).
    stage_out_channels:
      resnet18/34: [64, 64, 128, 256, 512]
      resnet50+:   [64, 256, 512, 1024, 2048]
    """
    def __init__(self, stage_out_channels, in_ch=3, base_ch=64):
        super().__init__()
        self.stage_out_channels = list(stage_out_channels)
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, base_ch, 7, 2, 3), nn.SiLU(),
            nn.Conv2d(base_ch, base_ch, 3, 1, 1), nn.SiLU(),
        )
        self.down1 = nn.Conv2d(base_ch, base_ch, 3, 2, 1)  # ~1/4
        self.down2 = nn.Conv2d(base_ch, base_ch, 3, 2, 1)  # ~1/8
        self.down3 = nn.Conv2d(base_ch, base_ch, 3, 2, 1)  # ~1/16
        self.down4 = nn.Conv2d(base_ch, base_ch, 3, 2, 1)  # ~1/32

        self.proj = nn.ModuleList([nn.Conv2d(base_ch, c, 1) for c in self.stage_out_channels])
        for p in self.proj:
            nn.init.zeros_(p.weight); nn.init.zeros_(p.bias)

    @torch.cuda.amp.autocast(enabled=False)
    def cond_pyramid(self, rgb):
        x = rgb.float()
        s0 = self.stem(x)
        s1 = self.down1(s0)
        s2 = self.down2(s1)
        s3 = self.down3(s2)
        s4 = self.down4(s3)
        return [s0, s1, s2, s3, s4]

class ResNetInjectionSession:
    """
    Hooks: conv1 (Stage0) und layer1..layer4 (Stage1..4).
    """
    def __init__(self, resnet, adapter: ResNetCondAdapter, conditioning_rgb, scale=1.0):
        self.m = resnet
        self.a = adapter
        self.rgb = conditioning_rgb
        self.scale = float(scale)
        p0 = next(resnet.parameters())
        self.dev, self.dt = p0.device, p0.dtype
        self.handles = []
        self.pyr = None

    def __enter__(self):
        self.m.eval()
        self.a.to(self.dev, dtype=self.dt)
        rgb = self.rgb.to(self.dev, dtype=self.dt)
        self.pyr = [p.to(self.dev, dtype=self.dt) for p in self.a.cond_pyramid(rgb)]

        def resize(feat, hw):
            if feat.shape[-2:] != hw:
                feat = F.interpolate(feat, size=hw, mode="bilinear", align_corners=False)
            return feat

        # conv1
        def hook_conv1(mod, inp, out):
            c = resize(self.pyr[0], out.shape[-2:])
            res = self.a.proj[0](c).to(out.dtype)
            return out + self.scale * res
        self.handles.append(self.m.conv1.register_forward_hook(hook_conv1))

        # layer1..4
        layers = [self.m.layer1, self.m.layer2, self.m.layer3, self.m.layer4]
        for i, layer in enumerate(layers, start=1):
            def make_hook(level=i):
                def _h(mod, inp, out):
                    c = resize(self.pyr[level], out.shape[-2:])
                    res = self.a.proj[level](c).to(out.dtype)
                    return out + self.scale * res
                return _h
            self.handles.append(layer.register_forward_hook(make_hook(i)))
        return self

    def __exit__(self, exc_type, exc, tb):
        for h in self.handles: h.remove()
        self.handles.clear()
        self.pyr=None
        return False

# -----------------------------------------------------------
# ViT-B/16: Konditions-Adapter + Injektions-Session (Patch-Ebene)
# -----------------------------------------------------------
class ViTCondAdapter(nn.Module):
    """
    Erzeugt Patch-Grid-Map [B,768,hp,wp] kompatibel zu vit_b_16.conv_proj-Ausgabe.
    """
    def __init__(self, embed_dim=768, base_ch=256):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(3, base_ch, 7, 2, 3), nn.SiLU(),
            nn.Conv2d(base_ch, base_ch, 3, 1, 1), nn.SiLU(),
            nn.Conv2d(base_ch, base_ch, 3, 2, 1), nn.SiLU(),  # ~1/4
            nn.Conv2d(base_ch, base_ch, 3, 2, 1), nn.SiLU(),  # ~1/8
        )
        self.to_embed = nn.Conv2d(base_ch, embed_dim, 1)
        nn.init.zeros_(self.to_embed.weight); nn.init.zeros_(self.to_embed.bias)

    @torch.cuda.amp.autocast(enabled=False)
    def cond_map(self, rgb, target_hw):
        x = self.backbone(rgb.float())
        x = F.interpolate(x, size=target_hw, mode="bilinear", align_corners=False)
        return self.to_embed(x)  # [B,embed_dim,hp,wp]

class ViTInjectionSession:
    """
    Hook auf conv_proj (vor Pos-Embedding).
    """
    def __init__(self, vit, adapter: ViTCondAdapter, conditioning_rgb, scale=1.0):
        self.m = vit
        self.a = adapter
        self.rgb = conditioning_rgb
        self.scale = float(scale)
        p0 = next(vit.parameters())
        self.dev, self.dt = p0.device, p0.dtype
        self.handles = []

    def __enter__(self):
        self.m.eval()
        self.a.to(self.dev, dtype=self.dt)
        rgb = self.rgb.to(self.dev, dtype=self.dt)

        def hook_conv_proj(mod, inp, out):
            hp, wp = out.shape[-2:]
            cmap = self.a.cond_map(rgb, (hp, wp)).to(out.dtype).to(out.device)
            return out + self.scale * cmap

        self.handles.append(self.m.conv_proj.register_forward_hook(hook_conv_proj))
        return self

    def __exit__(self, exc_type, exc, tb):
        for h in self.handles: h.remove()
        self.handles.clear()
        return False

# -----------------------------------------------------------
# BackboneClassifier: kapselt Front + Adapter + Backbone
# -----------------------------------------------------------
class BackboneClassifier(nn.Module):
    def __init__(self, num_classes: int, backbone: str = "resnet50",
                 img_size: int = 224, pretrained: bool = True,
                 cond_scale: float = 1.0):
        super().__init__()
        self.img_size = img_size
        self.backbone_name = backbone
        self.pretrained = pretrained
        self.cond_scale = float(cond_scale)

        # THz-Front (lernbar): 2×T×H×W  -> RGB 2D
        self.front = THzToRGBHead(in_ch=2, base_ch=32, k_t=5, final_depth=16)

        # Backbone + Adapter je nach Typ
        if backbone == "resnet50":
            weights = models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
            self.backbone = models.resnet50(weights=weights)
            in_feat = self.backbone.fc.in_features
            self.backbone.fc = nn.Linear(in_feat, num_classes)
            # ResNet50-Kanäle
            self.adapter = ResNetCondAdapter([64, 256, 512, 1024, 2048], in_ch=3, base_ch=64)
        elif backbone == "vit_b16":
            weights = models.ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None
            self.backbone = models.vit_b_16(weights=weights)
            in_feat = self.backbone.heads.head.in_features
            self.backbone.heads.head = nn.Linear(in_feat, num_classes)
            self.adapter = ViTCondAdapter(embed_dim=self.backbone.hidden_dim, base_ch=256)
        else:
            raise ValueError("backbone must be one of {'resnet50','vit_b16'}")

        # ImageNet-Norm fürs Input
        self.register_buffer("imgnet_mean", torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1), persistent=False)
        self.register_buffer("imgnet_std",  torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1), persistent=False)

    def freeze_backbone(self, freeze: bool = True):
        for p in self.backbone.parameters():
            p.requires_grad = not (freeze)

    def forward(self, vol: torch.Tensor) -> torch.Tensor:
        # 1) THz -> RGB
        img_rgb = self.front(vol)                                        # [B,3,H,W] in [0,1]
        if img_rgb.shape[-2:] != (self.img_size, self.img_size):
            img_rgb = F.interpolate(img_rgb, size=(self.img_size, self.img_size),
                                    mode="bilinear", align_corners=False)
        # 2) ImageNet-Norm
        mean = self.imgnet_mean.to(img_rgb.device, img_rgb.dtype)
        std  = self.imgnet_std.to(img_rgb.device, img_rgb.dtype)
        x = (img_rgb - mean) / std

        # 3) Injektion je nach Backbone
        if self.backbone_name.startswith("resnet"):
            with ResNetInjectionSession(self.backbone, self.adapter, conditioning_rgb=img_rgb, scale=self.cond_scale):
                logits = self.backbone(x)
        else:
            with ViTInjectionSession(self.backbone, self.adapter, conditioning_rgb=img_rgb, scale=self.cond_scale):
                logits = self.backbone(x)
        return logits

# -----------------------------------------------------------
# Train/Eval Loops
# -----------------------------------------------------------
def train_one_epoch(train_loader, model, use_amp, opt, scaler, bar_desc=None):
    model.train()
    running_loss, running_acc, n_seen = 0.0, 0.0, 0
    it = tqdm(train_loader, desc=bar_desc or "train", leave=False, ncols=0)
    for vol, label, _ in it:
        vol = vol.to(device)
        if vol.dim() == 6:  # [B,1,2,T,H,W] -> [B,2,T,H,W]
            vol = vol.squeeze(1)
        label = label.to(device).long()

        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", dtype=torch.float16, enabled=use_amp):
            logits = model(vol)
            loss = F.cross_entropy(logits, label)

        if torch.isfinite(loss):
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, model.parameters()), max_norm=1.0)
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

@torch.no_grad()
def validate(val_loader, model, use_amp, bar_desc=None):
    model.eval()
    total, correct, loss_sum = 0, 0, 0.0
    it = tqdm(val_loader, desc=bar_desc or "val", leave=False, ncols=0)
    for vol, label, _ in it:
        vol = vol.to(device)
        if vol.dim() == 6:
            vol = vol.squeeze(1)
        label = label.to(device).long()
        with torch.amp.autocast("cuda", dtype=torch.float16, enabled=use_amp):
            logits = model(vol)
            loss = F.cross_entropy(logits, label)
        loss_sum += loss.item()
        correct += (logits.argmax(1) == label).sum().item()
        total += label.size(0)
        it.set_postfix(acc=f"{correct/max(1,total):.4f}")
    return correct / max(1, total), loss_sum / max(1, total)

# -----------------------------------------------------------
# Main (RSKF)
# -----------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    # Daten
    ap.add_argument("--data_train", type=str, required=True)
    ap.add_argument("--data_test",  type=str, default="")
    ap.add_argument("--train_csv",  type=str, required=True)
    ap.add_argument("--val_csv",    type=str, default="")
    # Backbone & Adapter
    ap.add_argument("--backbone", type=str, default="resnet50", choices=("resnet50","vit_b16"))
    ap.add_argument("--img_size", type=int, default=512, choices=(256, 512))
    ap.add_argument("--pretrained", action="store_true")
    ap.add_argument("--train_all", action="store_true", help="Backbone mittrainieren")
    ap.add_argument("--train_adapter", action="store_true", help="Adapter (Injektion) mittrainieren")
    ap.add_argument("--learn_front", action="store_true", help="THz-Front mittrainieren")
    ap.add_argument("--cond_scale", type=float, default=1.0, help="Stärke der Injektion")
    # Optimizer
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--lr_adapter", type=float, default=None, help="falls None -> lr")
    ap.add_argument("--lr_front", type=float, default=None, help="falls None -> lr")
    ap.add_argument("--dtype", type=str, default="float16", choices=("float16","float32","bfloat16"))
    ap.add_argument("--save_dir", type=str, default="./runs/checkpoints_backbone")
    # Cross-Validation
    ap.add_argument("--cv_splits", type=int, default=5)
    ap.add_argument("--cv_repeats", type=int, default=2)
    ap.add_argument("--cv_seed", type=int, default=42)
    # Finale Evaluation
    ap.add_argument("--final_eval", action="store_true")

    args = ap.parse_args()
    # Seed
    import random
    random.seed(args.cv_seed); np.random.seed(args.cv_seed)
    torch.manual_seed(args.cv_seed); torch.cuda.manual_seed_all(args.cv_seed)

    print("Train Backbone mit Control-Injektion (RSKF)")
    print(f"Backbone: {args.backbone}  img_size: {args.img_size}  pretrained: {args.pretrained}  "
          f"train_all: {args.train_all}  train_adapter: {args.train_adapter}  learn_front: {args.learn_front}")

    # Klassenanzahl aus CSV
    all_rows = read_csv_pairs(args.train_csv)
    labels = [y for _, y in all_rows]

    stamp = datetime.now(ZoneInfo("Europe/Berlin")).strftime("%y%m%d_%H%M")
    save_dir = os.path.join(args.save_dir, f"{args.backbone}_rskf_{args.cv_splits}x{args.cv_repeats}_{stamp}")
    os.makedirs(save_dir, exist_ok=True)
    print(f"Checkpoints -> {save_dir}")

    rskf = RepeatedStratifiedKFold(n_splits=args.cv_splits, n_repeats=args.cv_repeats, random_state=args.cv_seed)
    cv_scores = []
    best_global_acc = -1.0
    best_global_ckpt = None
    total_splits = args.cv_splits * args.cv_repeats

    use_amp = (device == "cuda" and args.dtype == "float16")

    for split_idx, (idx_tr, idx_va) in tqdm(
        enumerate(rskf.split(all_rows, labels), start=1),
        total=total_splits, desc="RSKF splits", ncols=0
    ):
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

        # Modell je Split
        model = BackboneClassifier(
            num_classes=num_classes,
            backbone=args.backbone,
            img_size=args.img_size,
            pretrained=args.pretrained,
            cond_scale=args.cond_scale
        ).to(device)

        # Freeze/Unfreeze
        if not args.train_all:
            model.freeze_backbone(True)
            # Kopf dennoch trainieren
            if args.backbone == "resnet50":
                for p in model.backbone.fc.parameters(): p.requires_grad = True
            else:
                for p in model.backbone.heads.parameters(): p.requires_grad = True

        # Adapter & Front trainierbar?
        if not args.train_adapter:
            for p in model.adapter.parameters(): p.requires_grad = False
        if not args.learn_front:
            for p in model.front.parameters(): p.requires_grad = False

        # Optimizer: eigene LRs erlauben
        base_lr = args.lr
        lr_front   = base_lr if args.lr_front   is None else args.lr_front
        lr_adapter = base_lr if args.lr_adapter is None else args.lr_adapter

        param_groups = []
        # Backbone, je nach Freeze
        pg_backbone = [p for p in model.backbone.parameters() if p.requires_grad]
        if pg_backbone:
            param_groups.append({"params": pg_backbone, "lr": base_lr, "weight_decay": args.weight_decay})
        # Adapter
        pg_adapter = [p for p in model.adapter.parameters() if p.requires_grad]
        if pg_adapter:
            param_groups.append({"params": pg_adapter, "lr": lr_adapter, "weight_decay": args.weight_decay})
        # Front
        pg_front = [p for p in model.front.parameters() if p.requires_grad]
        if pg_front:
            param_groups.append({"params": pg_front, "lr": lr_front, "weight_decay": args.weight_decay})

        if len(param_groups) == 0:
            raise RuntimeError("Keine trainierbaren Parameter! Prüfe Flags --train_all/--train_adapter/--learn_front")

        opt = torch.optim.AdamW(param_groups)
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
                # Full Model
                torch.save({
                    "epoch": epoch,
                    "state_dict": model.state_dict(),
                    "best_val_acc": best_val_acc,
                    "backbone": args.backbone,
                    "img_size": args.img_size,
                    "num_classes": num_classes,
                    "pretrained": args.pretrained,
                    "train_all": args.train_all,
                    "train_adapter": args.train_adapter,
                    "learn_front": args.learn_front,
                    "cond_scale": args.cond_scale,
                }, best_path)
                # Adapter + Front separat (leichtgewichtig)
                torch.save({
                    "epoch": epoch,
                    "adapter_state_dict": model.adapter.state_dict(),
                    "front_state_dict":   model.front.state_dict(),
                    "backbone": args.backbone,
                    "img_size": args.img_size,
                    "cond_scale": args.cond_scale,
                }, os.path.join(fold_dir, "adapter_front_best.pt"))
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
            pretrained=False,
            cond_scale=ckpt.get("cond_scale", 1.0),
        ).to(device)
        model.load_state_dict(ckpt["state_dict"], strict=True)
        model.eval()

        final_ds = ThzDataset(args.data_test, args.val_csv, is_train=False)
        final_loader = DataLoader(final_ds, batch_size=1, shuffle=False, num_workers=2, pin_memory=True)
        use_amp_final = (device=="cuda" and args.dtype=="float16")
        final_acc, _ = validate(final_loader, model, use_amp_final, bar_desc="[FINAL] val")
        print(f"[FINAL] accuracy: {final_acc:.4f}")

if __name__ == "__main__":
    main()
