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

from eval_prob_adaptive import eval_prob_adaptive_differentiable, pool_prompt_errors_to_class_errors_batch
from transformers import logging as hf_logging
from adapter_inject.thz_front_rgb_head import THzToRGBHead

# ==== Dein Code/Imports aus dem Projekt ====
import process_rdf as prdf
from diffusion.datasets import get_target_dataset  # falls du das brauchst

from diffusion.datasets import get_target_dataset
from diffusion.models import get_sd_model
from pipeline_classifier_with_adapter.core.io import get_transform, load_prompts_csv, load_thz_indexed
from adapter.help_functions import build_sd2_1_base, load_class_text_embeds, PromptBank
from diffusion.utils import LOG_DIR, get_formatstr
from adapter_inject.train_baseline_with_thz import build_backbone, normalize_batch
from pipeline_classifier_with_adapter.eval_the_pipeline_results import evaluate_predictions

device = "cuda" if torch.cuda.is_available() else "cpu"

def validate_base(loader, front, backbone, imagenet_mean, imagenet_std,
                  img_out_size, final_dtype, use_amp_final, output_dir):
    total, correct = 0, 0
    with torch.no_grad():
        for i, (vol, label, _) in enumerate(loader, start=1):
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
            formatstr = get_formatstr(len(loader) - 1)
            torch.save(dict(pred=preds.detach().cpu(), label=label.cpu()), os.path.join(output_dir,  formatstr.format(i) + '.pt'))

def validate_dc(
    loader, front, unet, controlnet, vae, prompt_embeds, prompt_to_class, num_classes,
    scheduler, eargs, img_size, torch_dtype, use_amp,  reduce="mean", output_dir=None
):
    """Validation loop mirroring the training inference flow."""
    front.eval()
    controlnet.eval()
    unet.eval()

    latent_size = img_size // 8
    out_hw = (img_size, img_size)
    formatstr = get_formatstr(len(loader) - 1)
    total, correct = 0, 0
    print("[validate] Starting validation iteration...")
    for step, (vol, label, _) in enumerate(loader, start=1):
        fname = os.path.join(output_dir, formatstr.format(step) + '.pt')
        vol = vol.to(device)
        if vol.dim() == 6:
            vol = vol.squeeze(1)
        label = label.to(device).long()

        with torch.autocast(device_type="cuda", dtype=torch_dtype, enabled=use_amp):
            img_rgb = front(vol.float())  # [B,3,H,W]
            if img_rgb.shape[-2:] != out_hw:
                img_rgb = F.interpolate(img_rgb, size=out_hw, mode="bilinear", align_corners=False)

        img_rgb = img_rgb.clamp(0, 1)

        with torch.amp.autocast('cuda', enabled=False):
            x_in = (img_rgb * 2.0 - 1.0).to(torch.float32)
            lat  = vae.encode(x_in).latent_dist.mean.to(torch.float32) * 0.18215

            # --- ControlNet-Kanalzahl erkennen & passende Kondition vorbereiten ---
            try:
                cn_in_ch = controlnet.controlnet_cond_embedding.conv_in.in_channels
            except Exception:
                cn_in_ch = getattr(controlnet, "in_channels", 4)

            latent_hw = tuple(lat.shape[-2:])
            if cn_in_ch == 3:
                control_cond_full = (img_rgb * 2.0 - 1.0)
                control_cond_lat  = F.interpolate(control_cond_full, size=latent_hw,
                                                mode="bilinear", align_corners=False)
            else:
                control_cond_lat = None

            if cn_in_ch == 3:
                control_cond_img = (img_rgb * 2.0 - 1.0)  # [B,3,img_size,img_size]
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
            errors_per_prompt = torch.cat(errors_list, dim=0)  # [B, P]

            class_errors_batch = pool_prompt_errors_to_class_errors_batch(
                errors_per_prompt, prompt_to_class, num_classes, reduce=reduce
            )  # [B, C]
            preds = torch.argmin(class_errors_batch, dim=1)  # [B]

        correct += (preds == label).sum().item()
        total += label.numel()

        if step % 20 == 1:
            print(f"[validate] step={step}  running_acc={correct/max(1,total):.4f}")

    val_acc = correct / max(1, total)
    print(f"[validate] Done. val_acc={val_acc:.4f}")
    torch.save(dict(errors=errors_per_prompt.detach().cpu(), pred=preds.detach().cpu(), label=label), fname)
    return val_acc

def parse_args():
    p = argparse.ArgumentParser()
    # Daten
    p.add_argument('--dataset', required=True)
    p.add_argument('--pretrained_path', default=None, required=True)
    p.add_argument('--split', default='test', choices=['train','test'])
    p.add_argument('--img_size', type=int, default=512)
    # Auswahl
    p.add_argument('--classifier', required=True, choices=('diffusion', 'resnet50', "vit_b_16", "vit_b_32", "convnext_tiny"))       # z. B. diffusion, resnet50
    p.add_argument('--train_head', action='store_true')  # nur für torchvision-classifier
    p.add_argument('--adapter', action='store_true')  # z. B. feedback rgb

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

    ckpt = torch.load(args.pretrained_path, map_location='cpu')
    # Dataset
    ds = get_target_dataset(args.dataset, train=args.split=='train', transform=get_transform(args.img_size))
    ds_loader = DataLoader(ds, batch_size=1, shuffle=False,
                                  num_workers=args.num_workers, pin_memory=True)
    if args.dtype=='float16': embeds = embeds.half()

    # Ausgabeordner
    from datetime import datetime
    from zoneinfo import ZoneInfo 

    stamp = datetime.now(ZoneInfo("Europe/Berlin")).strftime("%y%m%d_%H%M")
    output_dir = os.path.join(args.output_dir, args.classifier + (f"_{args.adapter}" if args.adapter else ""), stamp)
    os.makedirs(output_dir, exist_ok=True)
    result_dir = os.path.join(output_dir, 'results')
    os.makedirs(result_dir, exist_ok=True)
    # Adapter bauen
    adapter = None
    if args.adapter:
        print("Lade RGB-Adapter...")
        front_f = THzToRGBHead(in_ch=2, base_ch=32, k_t=5, final_depth=16).to(device)
        front_f.load_state_dict(ckpt["front_state_dict"], strict=False)
        front_f.eval()

    # Classifier bauen
    'TODO: Mehrere Classifier unterstützen'
    clf = None
    if args.classifier == "diffusion":
        print("Baue Diffusion Zero-Shot Classifier...")
        vae_f, unet_f, tokenizer_f, text_encoder_f, scheduler_f, controlnet_f = build_sd2_1_base(
            dtype=ckpt["dtype"], use_xformers=False, train_all=False, version=ckpt["version"]
        )
        unet_f.eval()
        pb_f = PromptBank(args.prompt_path)
        prompt_embeds_f = pb_f.to_text_embeds(tokenizer_f, text_encoder_f, device)
        prompt_to_class_f = pb_f.prompt_to_class.to(device)
        num_classes_f = pb_f.num_classes
    else:
        print("Baue Torchvision Classifier...")
        backbone_name = ckpt["backbone"]
        num_classes = ckpt["num_classes"]
        imagenet_mean = tuple(ckpt["imagenet_mean"])
        imagenet_std = tuple(ckpt["imagenet_std"])
        img_out_size = ckpt["img_out_size"]

        backbone, _, _ = build_backbone(backbone_name, num_classes, pretrained=False)
        backbone.load_state_dict(ckpt["backbone_state_dict"], strict=False)
        backbone = backbone.to(device).eval()

    if args.classifier == "diffusion":
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
        eargs_f.learn_front = False
        
        use_amp_final = (device == "cuda") and (ckpt["dtype"] in ("float16", "bfloat16"))
        final_torch_dtype = torch.float16 if ckpt["dtype"] == "float16" else (
            torch.bfloat16 if ckpt["dtype"] == "bfloat16" else torch.float32
        )

        final_acc = validate_dc(
        ds_loader, front_f, unet_f, controlnet_f, vae_f,
        prompt_embeds_f, prompt_to_class_f, num_classes_f,
        scheduler_f, eargs_f, ckpt["img_size"],
        torch_dtype=final_torch_dtype, use_amp=use_amp_final, reduce="mean", output_dir=result_dir
        )
        print(f"Final Accuracy: {final_acc:.4f}")
        
        
    else:
        validate_base(
            ds_loader, front_f if args.adapter else nn.Identity(),
            backbone, imagenet_mean, imagenet_std,
            img_out_size, final_dtype=torch.float32,
            use_amp_final=False,
            output_dir=result_dir
        )
        
    try: 
        evaluate_predictions(
            result_dir,
            args.prompt_path,
            output_dir=os.path.join(output_dir, "evaluation")
        )
    except Exception as e:
        print(f"Fehler bei der Auswertung der Vorhersagen: {e}")


if __name__ == "__main__":
    args = parse_args()
    main(args)
