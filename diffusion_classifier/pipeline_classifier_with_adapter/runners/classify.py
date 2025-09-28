import torch, tqdm
import os, copy, time, json, argparse, random
from pathlib import Path
from typing import Tuple, Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import models as tvm, transforms as T

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

# ==== Dein Code/Imports aus dem Projekt ====
import process_rdf as prdf
from diffusion.datasets import get_target_dataset  # falls du das brauchst

from adapter.ControlNet_Adapter_wrapper import ControlNetAdapterWrapper
from diffusion.datasets import get_target_dataset
from diffusion.models import get_sd_model
from pipeline_classifier_with_adapter.core.io import get_transform, load_prompts_csv, load_thz_indexed
from pipeline_classifier_with_adapter.classifiers.eval_prob_adaptiv_with_adapter import default_unet_forward
from pipeline_classifier_with_adapter.classifiers.build_torchvision_backbone import build_torchvision_backbone, train_classifier
from pipeline_classifier_with_adapter.classifiers.diffusion_zero_shot import DiffusionZeroShotClassifier
from pipeline_classifier_with_adapter.classifiers.torchvision_classifier import TorchvisionClassifier
from pipeline_classifier_with_adapter.adapters.rgb_adapter import load_rgb_adapter, load_thz_adapter_only
from pipeline_classifier_with_adapter.adapters.feedback_adapter import load_feedback_adapter
from diffusion.utils import LOG_DIR, get_formatstr

device = "cuda" if torch.cuda.is_available() else "cpu"

COLLECTION_ADAPTERS = {
    
}

COLLECTION_TRAINED_CLASSIFIERS = {
    'IMAGENET1K_V1 ' : 'IMAGENET1K_V1' 
}

def parse_args():
    p = argparse.ArgumentParser()
    # Daten
    p.add_argument('--dataset', required=True)
    p.add_argument('--split', default='test', choices=['train','test'])
    p.add_argument('--img_size', type=int, default=512)
    p.add_argument('--train_dataset', type=str, default='thz_for_train_adapter')
    # Auswahl
    p.add_argument('--classifier', required=True, choices=('diffusion', 'resnet50', "vit_b_16", "vit_b_32", "convnext_tiny"))       # z. B. diffusion, resnet50
    p.add_argument('--train_head', action='store_true')  # nur für torchvision-classifier
    p.add_argument('--adapter', type=str, choices=('feedback', 'rgb'))  # z. B. feedback rgb

    # Pretrained laden
    p.add_argument('--from_pretrained_classifier', action='store_true')
    p.add_argument('--from_pretrained_adapter', action='store_true')
    p.add_argument('--pretrained_classifier', type=str)
    p.add_argument('--pretrained_adapter', type=str)
    
    # Diffusion/Eval
    p.add_argument('--version',  type=str, default='2-1')
    p.add_argument('--prompt_path', required=True)
    p.add_argument('--dtype', default='float16', choices=('float16','float32'))
    p.add_argument('--n_trials', type=int, default=2)
    p.add_argument('--n_samples', nargs='+', type=int, default=[8,16,32])
    p.add_argument('--to_keep', nargs='+', type=int, default=[4,3,1])
    p.add_argument('--loss', default='l2', choices=('l1','l2','huber'))
    p.add_argument('--noise_path', default=None)
    # THz
    p.add_argument('--thz_path', default=None)
    # Adapter-Args
    p.add_argument('--feedback_ckpt', default=None)
    p.add_argument('--rgb_dir', default=None)

    #output
    p.add_argument('--output_dir', default='./data')

    #for diffusion-classifier
    p.add_argument('--batch_size', '-b', type=int, default=32)
    p.add_argument('--subset_path', type=str, default=None)
    p.add_argument('--interpolation', type=str, default='bicubic')
    p.add_argument('--extra', type=str, default=None)
    p.add_argument('--n_workers', type=int, default=1)
    p.add_argument('--worker_idx', type=int, default=0)

    return p.parse_args()

def main(args):
    
    torch.backends.cudnn.benchmark = True

    # Dataset
    ds = get_target_dataset(args.dataset, train=args.split=='train', transform=get_transform(args.img_size))

    # SD-Komponenten (werden von diffusion-classifier gebraucht)
    vae, tokenizer, text_encoder, unet_raw, scheduler = get_sd_model(args)
    vae, text_encoder = vae.to(device), text_encoder.to(device)
    unet, forward_fn = unet_raw.to(device), default_unet_forward

    # Prompts & Embeddings
    prompts_df, prompts = load_prompts_csv(args.prompt_path)
    text_input = tokenizer(prompts, padding="max_length",
                           max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")
    embeds = []
    with torch.inference_mode():
        for i in range(0, len(text_input.input_ids), 100):
            embeds.append(text_encoder(text_input.input_ids[i:i+100].to(device))[0])
    embeds = torch.cat(embeds, dim=0)
    if args.dtype=='float16': embeds = embeds.half()

    # Ausgabeordner
    from datetime import datetime
    from zoneinfo import ZoneInfo 

    stamp = datetime.now(ZoneInfo("Europe/Berlin")).strftime("%y%m%d_%H%M")
    output_dir = os.path.join(args.output_dir, args.classifier + (f"_{args.adapter}" if args.adapter else ""), stamp)
    os.makedirs(output_dir, exist_ok=True)
    # Adapter bauen
    adapter = None
    if args.adapter == "feedback":
        print("Lade Feedback-Adapter...")
        adapter = load_feedback_adapter(
            ckpt_path=args.feedback_ckpt,
            device=device,
            dtype=(torch.float16 if args.dtype=='float16' else torch.float32),
            out_size=args.img_size
        )
    elif args.adapter == "rgb":
        print("Lade RGB-Adapter...")
        rgb = load_rgb_adapter(
            output_dir=args.rgb_dir,
            device=device,
            dtype=(torch.float16 if args.dtype=='float16' else torch.float32)
        )
        adapter = rgb
    else:
        print(f"Unbekannter Adapter: {args.adapter}")

    # Classifier bauen
    'TODO: Mehrere Classifier unterstützen'
    clf = None
    if args.classifier == "diffusion":
        print("Baue Diffusion Zero-Shot Classifier...")
        classifier = DiffusionZeroShotClassifier(dtype=args.dtype, loss=args.loss)
        classifier.prepare(vae=vae, unet=unet, scheduler=scheduler, forward_fn=forward_fn)
    else:
        if args.train_head:
            model, weights = train_classifier(
                train_dir = args.train_dataset,
                val_dir= args.dataset,
                arch= args.classifier,
                num_classes = len(ds),
                pretrained= True,
                freeze_head= False,
                epochs= 10,
                lr= 1e-3,
                weight_decay= 1e-4,
                batch_size= 3,
                num_workers= 8,
                device= None,
                adapter= adapter,
            )
        else:
            model, weights = build_torchvision_backbone(
                arch=args.classifier,
                num_classes=len(ds),
                freeze_head=False
            )
        classifier = TorchvisionClassifier(model, num_classes=len(ds), weights=weights, input_adapter=adapter)
        classifier.to(device)
        classifier.eval()

    all_noise = torch.load(args.noise_path).to(device) if args.noise_path else None

    correct = total = 0
    pbar = tqdm.tqdm(range(len(ds)))
    formatstr = get_formatstr(len(ds) - 1)
    for i in pbar:
        if total>0: pbar.set_description(f'Acc: {100*correct/total:.2f}%')
        fname = os.path.join(output_dir, formatstr.format(i) + '.pt')

        image, label = ds[i]         # (3,H,W), [-1,1]
        x = image.unsqueeze(0).to(device)
        if args.dtype=='float16': x = x.half()

        # THz laden (falls Adapter benötigt)
        thz = load_thz_indexed(args.thz_path, i)
        if thz is not None:
            thz = thz.to(device, dtype=(torch.float16 if args.dtype=='float16' else torch.float32))

        # Feedback-Adapter ersetzt Bild vor VAE
        if args.adapter == "feedback" and thz is not None:
            with torch.no_grad():
                x01 = adapter(thz)             # [1,3,S,S] in [0,1]
            x = (x01*2.0 - 1.0).to(dtype=vae.dtype)         # [-1,1]

        # RGB-Adapter liefert extra_cond (ControlNet)
        if args.adapter == "rgb" and thz is not None:
            extra_cond = adapter.extra_cond(thz) 
        else: extra_cond = None

        if args.classifier == "diffusion":
            pred, pred_errors = classifier.predict(
                x_in=image, embeds=embeds, extra_cond=extra_cond,
                args=args, prompts_df=prompts_df, all_noise=all_noise
            )
            
            torch.save(dict(errors=pred_errors, pred=pred, label=label), fname)
        else:
            with torch.inference_mode():
                pred, predictions, score, category_name = classifier.predict(x, extra_cond=extra_cond)
                torch.save(dict(score=score, pred=pred, label=label), fname)
        correct += int(pred == int(label)); total += 1
        

    print(f"Final Acc: {100*correct/max(1,total):.2f}%")

if __name__ == "__main__":
    args = parse_args()
    main(args)
