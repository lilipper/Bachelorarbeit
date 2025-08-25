from dataclasses import dataclass
from typing import Dict
import time
from unittest import loader
import torch
from torch import device, nn, optim
from torch.amp import autocast, GradScaler
from tqdm import tqdm
import pandas as pd
from pathlib import Path

from Adapter.ControlNet.annotator.uniformer.mmcv.runner import optimizer
from Adapter.ControlNet.cldm import model




@dataclass
class TrainState:
    epoch: int = 0
    best_acc: float = 0.0




def accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    correct = (preds == targets).sum().item()
    return correct / targets.size(0) * 100.0




def create_optimizer(model: nn.Module, lr: float, wd: float) -> optim.Optimizer:
    return optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)




def create_scheduler(optimizer: optim.Optimizer, epochs: int, steps_per_epoch: int):
    total_steps = epochs * steps_per_epoch
    return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)




def save_checkpoint(state: Dict, outdir: Path, name: str):
    outdir.mkdir(parents=True, exist_ok=True)
    torch.save(state, outdir / name)




def train_one_epoch(model, loader, optimizer, device, scaler, criterion):
    model.train()
    running_loss, running_acc, n = 0.0, 0.0, 0


    pbar = tqdm(loader, desc='train', leave=False)
    for imgs, targets in pbar:
        imgs, targets = imgs.to(device, non_blocking=True), targets.to(device, non_blocking=True)


        optimizer.zero_grad(set_to_none=True)
        with autocast('cuda', enabled=torch.cuda.is_available()):
            logits = model(imgs)
            loss = criterion(logits, targets)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()


        acc = accuracy(logits.detach(), targets)
        bs = targets.size(0)
        running_loss += loss.item() * bs
        running_acc += acc * bs
        n += bs
        pbar.set_postfix({"loss": f"{running_loss/n:.4f}", "acc": f"{running_acc/n:.2f}%"})


    return running_loss / n, running_acc / n




def evaluate(model, loader, device, criterion):
    model.eval()
    running_loss, running_acc, n = 0.0, 0.0, 0
    with torch.no_grad():
        for imgs, targets in tqdm(loader, desc='val', leave=False):
            imgs, targets = imgs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
            logits = model(imgs)
            loss = criterion(logits, targets)
            acc = accuracy(logits, targets)
            bs = targets.size(0)
            running_loss += loss.item() * bs
            running_acc += acc * bs
            n += bs
    return running_loss / n, running_acc / n




def fit(model, train_loader, val_loader, device, epochs, lr, wd, outdir: Path):
    scaler = GradScaler('cuda', enabled=torch.cuda.is_available())
    optimizer = create_optimizer(model, lr, wd)
    scheduler = create_scheduler(optimizer, epochs, len(train_loader))
    criterion = nn.CrossEntropyLoss()
    state = TrainState()


    log_rows = []


    for epoch in range(1, epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, device, scaler, criterion)
        val_loss, val_acc = evaluate(model, val_loader, device, criterion)
        scheduler.step()
        dt = time.time() - t0
        state.epoch = epoch


        # Logging
        log_rows.append({
        'epoch': epoch,
        'train_loss': tr_loss,
        'train_acc': tr_acc,
        'val_loss': val_loss,
        'val_acc': val_acc,
        'lr': optimizer.param_groups[0]['lr'],
        'time_sec': dt,
        })


        # Checkpointing
        ckpt = {
        'epoch': epoch,
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'val_acc': val_acc,
        }
        save_checkpoint(ckpt, outdir, 'last.pt')
        if val_acc >= state.best_acc:
            state.best_acc = val_acc
            save_checkpoint(ckpt, outdir, 'best.pt')


        print(f"Epoch {epoch:03d} | train_acc={tr_acc:.2f}% | val_acc={val_acc:.2f}% | time={dt:.1f}s")


    # CSV speichern
    pd.DataFrame(log_rows).to_csv(outdir / 'log.csv', index=False)