import torch
import torchvision.models as tvm
from torch import nn, optim
from typing import Tuple, Optional
from torchvision import datasets
from torchvision import transforms as T
from typing import Optional, Tuple, Dict, Any
from torch.utils.data import DataLoader
from diffusion.datasets import get_target_dataset

def build_torchvision_backbone(
    arch: str,
    num_classes: int,
    freeze_head: bool = True,
) -> Tuple[nn.Module, Optional[object]]:
    """
    Gibt (model, preprocess) zurück.
    """
    if arch == "resnet50":
        weights = tvm.ResNet50_Weights.IMAGENET1K_V1
        m = tvm.resnet50(weights=weights)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
        preprocess = weights

    elif arch == "convnext_tiny":
        weights = tvm.ConvNeXt_Tiny_Weights.IMAGENET1K_V1 
        m = tvm.convnext_tiny(weights=weights)
        m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, num_classes)
        preprocess = weights

    elif arch in ("vit_b_16", "vit_b_32"):
        WeightsEnum = tvm.ViT_B_16_Weights if arch == "vit_b_16" else tvm.ViT_B_32_Weights
        weights = WeightsEnum.IMAGENET1K_V1 
        ctor = tvm.vit_b_16 if arch == "vit_b_16" else tvm.vit_b_32
        m = ctor(weights=weights)
        # ViT-Kopf austauschen
        m.heads.head = nn.Linear(m.heads.head.in_features, num_classes)
        preprocess = weights

    else:
        raise ValueError(f"unknown arch: {arch}")
    
    for n, p in m.named_parameters():
        p.requires_grad = False

    if not freeze_head:
        # nur den Kopf trainieren
        if arch == "resnet50":
            for p in m.fc.parameters(): p.requires_grad = True
        elif arch == "convnext_tiny":
            for p in m.classifier[-1].parameters(): p.requires_grad = True
        else:  # ViT
            for p in m.heads.head.parameters(): p.requires_grad = True

    return m, preprocess

def make_dataloaders(
    train_dataset: str,
    val_dataset: str,
    preprocess: Optional[object],
    batch_size: int = 64,
    num_workers: int = 8,
) -> Tuple[DataLoader, DataLoader]:
    if preprocess is None:
        preprocess = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
        ])
    train_ds = get_target_dataset(train_dataset, train=True, transform=preprocess)
    val_ds   = get_target_dataset(val_dataset, train=False, transform=preprocess)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader

def train_classifier(
    train_dir: str,
    val_dir: str,
    arch: str = "resnet18",
    num_classes: int = 2,
    pretrained: bool = True,
    freeze_head: bool = False,
    epochs: int = 10,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    batch_size: int = 64,
    num_workers: int = 8,
    device: Optional[torch.device] = None,
    adapter: Optional[nn.Module] = None,
) -> Dict[str, Any]:
    """
    Trainiert einen Image-Classifier und speichert das beste Modell nach Val-Accuracy.
    Gibt History + best_acc zurück.
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Modell + Preprocess
    model, preprocess = build_torchvision_backbone(
        arch=arch,
        num_classes=num_classes,
        freeze_head=freeze_head,
    )
    model.to(device)

    # Dataloaders
    train_loader, val_loader = make_dataloaders(
        train_dir, val_dir, preprocess, batch_size=batch_size, num_workers=num_workers
    )
    best_model = model
    # Optimierung
    criterion = nn.CrossEntropyLoss()
    params_to_update = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(params_to_update, lr=lr, weight_decay=weight_decay)
    if adapter:
        adapter.to(device)
        adapter.eval()  # Adapter bleibt fix

    scaler = torch.amp.GradScaler(device='cuda', enabled=(device.type == "cuda"))

    def accuracy(logits, y):
        return (logits.argmax(1) == y).float().mean()
    
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc = 0.0

    for epoch in range(1, epochs + 1):
        # ---- Train ----
        model.train()
        run_loss = run_acc = 0.0
        nb = 0
        for x, y in train_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type="cuda", enabled=(device.type == "cuda")):
                x = adapter(x) if adapter else x  # bleibt [B,3,H,W]
                logits = model(x)
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            run_loss += loss.item()
            run_acc  += accuracy(logits.detach(), y)
            nb += 1

        train_loss = run_loss / max(1, nb)
        train_acc  = run_acc  / max(1, nb)

        # ---- Val ----
        model.eval()
        v_loss = v_acc = 0.0
        nb = 0
        with torch.no_grad(), torch.amp.autocast(device_type="cuda", enabled=(device.type == "cuda")):
            for x, y in val_loader:
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                logits = model(x)
                loss = criterion(logits, y)
                v_loss += loss.item()
                v_acc  += accuracy(logits, y)
                nb += 1

        val_loss = v_loss / max(1, nb)
        val_acc  = v_acc  / max(1, nb)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(f"[E{epoch:02d}] train_loss={train_loss:.4f}  train_acc={train_acc:.4f} | "
              f"val_loss={val_loss:.4f}  val_acc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model = model

    return best_model, preprocess