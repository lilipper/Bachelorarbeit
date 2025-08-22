import argparse
from pathlib import Path
import torch
from torch import nn


from models import create_model
from data.datasets import build_dataloaders
from utils.train_utils import fit




def parse_args():
    p = argparse.ArgumentParser(description='CV Baseline Trainer')
    p.add_argument('--data_root', type=str, required=True, help='Ordner mit train/ und val/')
    p.add_argument('--model', type=str, default='resnet18', choices=['resnet18', 'convnext_tiny', 'vit_b_16'])
    p.add_argument('--pretrained', type=str, default='true', choices=['true', 'false'])
    p.add_argument('--img_size', type=int, default=224)
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--epochs', type=int, default=30)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--weight_decay', type=float, default=1e-4)
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--outdir', type=str, default='runs')
    return p.parse_args()




def main():
    args = parse_args()


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loader, val_loader, num_classes = build_dataloaders(
    args.data_root, args.img_size, args.batch_size, args.workers
    )


    model, in_ch = create_model(args.model, num_classes, args.pretrained == 'true', args.img_size)


    # Falls Eingabe 1‑kanalig wäre, könnte man hier on-the-fly auf 3 Kanäle duplizieren.
    # Für ImageFolder erwarten wir 3‑Kanal‑RGB, daher keine Anpassung nötig.


    model = model.to(device)


    run_name = f"{args.model}_{'pt' if args.pretrained=='true' else 'scratch'}"
    outdir = Path(args.outdir) / run_name
    outdir.mkdir(parents=True, exist_ok=True)


    fit(model, train_loader, val_loader, device, args.epochs, args.lr, args.weight_decay, outdir)




if __name__ == '__main__':
    main()