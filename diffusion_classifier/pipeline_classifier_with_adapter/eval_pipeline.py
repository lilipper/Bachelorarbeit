"""
Eval script for THz classifiers and diffusion-based classifiers.

Usage examples:

# Evaluate a diffusion classifier (Stable Diffusion + ControlNet + latent adapter)
python eval_pipeline.py \
    --pretrained_path /path/to/dc_checkpoint.pt \
    --classifier diffusion \
    --adapter latent \
    --dataset thz_for_adapter \
    --split test \
    --output_dir ./results_eval_pipeline

# Evaluate a baseline backbone (e.g. ViT-B/32) with ControlNet adapter
python eval_pipeline.py \
    --pretrained_path /path/to/baseline_checkpoint.pt \
    --classifier vit_b_32 \
    --adapter cn_wrapper \
    --dataset thz_for_adapter \
    --split test \
    --output_dir ./results_eval_pipeline

Arguments:
    --dataset        Name of the dataset handled by diffusion.datasets.get_target_dataset
                     (default: thz_for_adapter).
    --pretrained_path
                     Path to the checkpoint (.pt) file to evaluate. This can be a
                     diffusion-classifier checkpoint or a baseline classifier checkpoint.
    --split          Which split of the target dataset to evaluate on: 'train' or 'test'.
    --classifier     Type of classifier to evaluate:
                        - diffusion  : Stable Diffusion + ControlNet zero-shot classifier
                        - resnet50   : torchvision ResNet-50 backbone
                        - vit_b_16   : torchvision ViT-B/16 backbone
                        - vit_b_32   : torchvision ViT-B/32 backbone
                        - convnext_tiny : torchvision ConvNeXt-Tiny backbone
    --adapter        Front-end that maps THz volumes to images/latents:
                        - rgb            : (not used here, reserved)
                        - cn_wrapper     : ControlNetAdapterWrapper (new)
                        - latent         : LatentMultiChannelAdapter
                        - old_cn_wrapper : legacy ControlNet adapter wrapper
    --output_dir     Root directory where evaluation outputs are written
                     (per run, a timestamped subfolder is created).
    --n_workers      Number of DataLoader workers.
"""

import torch, tqdm
import os, copy, time, json, argparse, random
from pathlib import Path
from typing import Tuple, Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import models as tvm, transforms as T

from pytorch_grad_cam import GradCAM, GradCAMPlusPlus
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

import numpy as np
from skimage import io
import pandas as pd
import cv2

from eval_prob_adaptive import eval_prob_adaptive_differentiable
from adapter_multichannel.train_baseline_cn_without_cv_and_dropout import ControlNetAdapterWrapper
from adapter.ControlNet_Adapter_wrapper import ControlNetAdapterWrapper as old_ControlNetAdapterWrapper
from adapter_multichannel.train_dc_with_original_cn_multichannel_dropout import LatentMultiChannelAdapter


import process_rdf as prdf
import matplotlib.pyplot as plt
from diffusion.datasets import get_target_dataset  

from diffusion.datasets import get_target_dataset
from adapter.help_functions import build_sd2_1_base, load_class_text_embeds, PromptBank
from diffusion.utils import LOG_DIR, get_formatstr
from adapter_inject.train_baseline_with_thz import build_backbone, normalize_batch
from pipeline_classifier_with_adapter.eval_the_pipeline_results import evaluate_predictions

device = "cuda" if torch.cuda.is_available() else "cpu"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

def vit_reshape_transform(tensor):
   
    B, seq_len, C = tensor.shape
    n_patches = seq_len - 1  
    h = w = int(n_patches ** 0.5)  
    x = tensor[:, 1:, :]             
    x = x.reshape(B, h, w, C)        
    x = x.permute(0, 3, 1, 2)         
    return x


def compute_volume_gradcam(
    vol,
    front,
    backbone,
    imagenet_mean,
    imagenet_std,
    img_out_size,
    final_dtype,
    use_amp_final,
    target_class: Optional[int] = None,
):
    """
    Gradient-basierte Relevanzkarte pro Slice im Volumen.

    Unterstützte Formen:
        vol: [1, 2, Z, H, W]  (dein aktuelles Setup)
        alternativ auch [1, Z, H, W] oder [Z, H, W]

    Rückgabe:
        vol_slices_np: [Z, H, W]  (Visualisierungsvolumen)
        relevance_np:  [Z, H, W]  (normierte Relevanz 0..1)
        target_class:  int
    """

    vol_cam = vol.detach().clone().to(device)

    if vol_cam.dim() == 5:
        if vol_cam.size(0) != 1:
            raise ValueError(f"compute_volume_gradcam erwartet Batchsize 1, bekam {vol_cam.size(0)}")
    elif vol_cam.dim() == 4:
        if vol_cam.size(0) == 1:
            vol_cam = vol_cam.unsqueeze(1) 
        else:
            vol_cam = vol_cam.unsqueeze(0)  
    elif vol_cam.dim() == 3:
        vol_cam = vol_cam.unsqueeze(0).unsqueeze(0)
    else:
        raise ValueError(f"Unerwartete Volumen-Form: {vol_cam.shape}")

    vol_cam.requires_grad_(True)  

    with torch.autocast(device_type="cuda", dtype=final_dtype, enabled=use_amp_final):
        img_rgb = front(vol_cam)  

        if img_rgb.shape[-2] != img_out_size or img_rgb.shape[-1] != img_out_size:
            img_rgb = F.interpolate(
                img_rgb,
                size=(img_out_size, img_out_size),
                mode="bilinear",
                align_corners=False,
            )

        img_in = normalize_batch(img_rgb, imagenet_mean, imagenet_std)
        logits = backbone(img_in)  
    if target_class is None:
        target_class = int(torch.argmax(logits, dim=1).item())

    score = logits[0, target_class]

    front.zero_grad(set_to_none=True)
    backbone.zero_grad(set_to_none=True)
    if vol_cam.grad is not None:
        vol_cam.grad.zero_()
    score.backward()

    grads = vol_cam.grad.detach().cpu().squeeze(0)   
    vol_data = vol_cam.detach().cpu().squeeze(0)     

    if grads.dim() == 4:
        grads_abs = grads.abs().sum(dim=0)      
        vol_slices = vol_data.mean(dim=0)    
    elif grads.dim() == 3:
        grads_abs = grads.abs()              
        vol_slices = vol_data
    else:
        raise ValueError(f"Unerwartete Grad-Form: {grads.shape}")

    max_val = grads_abs.max()
    if max_val > 0:
        relevance = grads_abs / max_val
    else:
        relevance = torch.zeros_like(grads_abs)

    vol_slices_np = vol_slices.numpy()  
    relevance_np = relevance.numpy()   

    return vol_slices_np, relevance_np, target_class


def validate_base(loader, front, backbone, imagenet_mean, imagenet_std,
                  img_out_size, final_dtype, use_amp_final, output_dir):
    total, correct = 0, 0
    front.to(device).eval()
    backbone.to(device).eval()
    print("[validate_base] Starting validation iteration...")

    image_dir = os.path.join(output_dir, 'images')
    os.makedirs(image_dir, exist_ok=True)
    cams_dir = os.path.join(output_dir, "images_grad_cams")
    os.makedirs(cams_dir, exist_ok=True)
    
    
    formatstr = get_formatstr(len(loader) - 1)
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
        path_cam_for_step = os.path.join(cams_dir, f"{formatstr.format(i)}_lbl_{label.item()}_pred_{preds.item()}_cam")
        os.makedirs(path_cam_for_step, exist_ok=True)
        
        vol_slices_np, relevance_np, target_cls = compute_volume_gradcam(
            vol=vol,
            front=front,
            backbone=backbone,
            imagenet_mean=imagenet_mean,
            imagenet_std=imagenet_std,
            img_out_size=img_out_size,
            final_dtype=final_dtype,
            use_amp_final=use_amp_final,
            target_class=preds.item(),   
        )

        video_path = os.path.join(path_cam_for_step, "gradcam")
        save_gradcam_video(vol_slices_np, relevance_np, video_path, fps=20)

    final_acc = correct / max(1, total)
    print(f"[validate_base] Done. val_acc={final_acc:.4f}")
    return final_acc


def encode_all_frames_with_vae(frames_vae, vae, scaling=0.18215):
    if isinstance(vae, nn.DataParallel):
        vae_ = vae.module
    else:
        vae_ = vae
    posterior = vae_.encode(frames_vae).latent_dist.mean.to(torch.float32)
    latents = posterior * scaling
    return latents

def _unwrap_vae(vae):
    return vae.module if isinstance(vae, nn.DataParallel) else vae

def save_gradcam_video(
    vol_slices: np.ndarray,
    relevance: np.ndarray,
    out_path: str,
    fps: int = 20,
):
    assert vol_slices.shape == relevance.shape

    Z, H, W = vol_slices.shape

    os.makedirs(out_path, exist_ok=True)
    video_path = os.path.join(out_path, "gradcam_volume.mp4")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(video_path, fourcc, fps, (H, W))

    if not writer.isOpened():
        print(f"[save_gradcam_video] mp4v Writer konnte nicht geöffnet werden, versuche AVI/XVID...")
        video_path = os.path.join(out_path, "gradcam_volume.avi")
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        writer = cv2.VideoWriter(video_path, fourcc, fps, (H, W))

    if not writer.isOpened():
        print(f"[save_gradcam_video] VideoWriter schlägt komplett fehl, speichere nur Einzelbilder.")
        writer = None

    for z in range(Z):
        slice_img = vol_slices[z]
        rel_map   = relevance[z]

        slice_img = slice_img - slice_img.min()
        denom = slice_img.max() + 1e-6
        slice_img = (slice_img / denom * 255).astype("uint8")

        rel_map = np.clip(rel_map, 0.0, 1.0)
        rel_map = (rel_map * 255).astype("uint8")

        heat_color = cv2.applyColorMap(rel_map, cv2.COLORMAP_JET)

        slice_bgr = cv2.cvtColor(slice_img, cv2.COLOR_GRAY2BGR)
        overlay = cv2.addWeighted(slice_bgr, 0.6, heat_color, 0.4, 0)

        overlay = cv2.rotate(overlay, cv2.ROTATE_90_CLOCKWISE)

        if 695 <= z <= 705:
            cv2.imwrite(os.path.join(os.path.dirname(out_path), f"slice_{z}.png"), overlay)
            out_file = os.path.join(os.path.dirname(out_path), f"slice_{z}.pdf")
            plt.imsave(out_file, overlay, format="pdf")

        if writer is not None:
            writer.write(overlay)

    if writer is not None:
        writer.release()


def validate_dc(
    loader,
    front,
    unet,
    controlnet,
    vae,
    prompt_embeds,
    prompt_to_class,
    num_classes,
    scheduler,
    eargs,
    img_size,
    torch_dtype,
    use_amp,
    pd_f,
    reduce="mean",
    output_dir=None,
):
    image_dir = os.path.join(output_dir, "images")
    os.makedirs(image_dir, exist_ok=True)
    cams_dir = os.path.join(output_dir, "images_grad_cams")
    os.makedirs(cams_dir, exist_ok=True)

    front.to(device).eval()
    vae.to(device).eval()
    controlnet.to(device).eval()
    unet.to(device).eval()

    latent_size = img_size // 8
    out_hw = (img_size, img_size)
    formatstr = get_formatstr(len(loader) - 1)

    total, correct = 0, 0
    print("[validate] Starting validation iteration...")

    for step, (vol, label, _) in enumerate(loader, start=1):
        fname = os.path.join(output_dir, formatstr.format(step) + ".pt")
        vol = vol.to(device)
        if vol.dim() == 6:
            vol = vol.squeeze(1)
        label = label.to(device).long()

        with torch.autocast(device_type="cuda" if device == "cuda" else "cpu", dtype=torch_dtype, enabled=use_amp):
            x = vol.mean(dim=1)
            B, T, H, W = x.shape
            frames = x.unsqueeze(2).repeat(1, 1, 3, 1, 1).view(B * T, 3, H, W)
            frames = F.interpolate(frames, size=out_hw, mode="bilinear", align_corners=False)
            frames = frames.clamp(0, 1) * 2 - 1

            lat_chunks = []
            bt = frames.shape[0]
            chunk = getattr(eargs, "vae_chunk", 256)
            with torch.no_grad():
                for s in range(0, bt, chunk):
                    lat_chunks.append(encode_all_frames_with_vae(frames[s : s + chunk], vae, scaling=0.18215))
            latents_flat = torch.cat(lat_chunks, dim=0).to(device)
            lat_stack = latents_flat.view(B, T, 4, latent_size, latent_size)
            lat = front(lat_stack)
            vae_ = _unwrap_vae(vae)
            with torch.no_grad():
                img_for_cam = vae_.decode(lat / 0.18215).sample
            img_for_cam = ((img_for_cam.clamp(-1, 1) + 1) / 2).to(device)
            control_cond_img = img_for_cam

            imgs01 = img_for_cam.detach().cpu().to(torch.float32)
            for bi in range(imgs01.shape[0]):
                arr = imgs01[bi].numpy().transpose(1, 2, 0)
                arr = np.clip(arr, 0.0, 1.0)
                save_path = os.path.join(image_dir, f"{formatstr.format(step)}_{label.item()}_{bi:02d}.tiff")
                io.imsave(save_path, arr, check_contrast=False)

            errors_list = []
            for i in range(B):
                cond = control_cond_img[i : i + 1] if control_cond_img is not None else None
                _, _, class_errors_i = eval_prob_adaptive_differentiable(
                    unet=unet,
                    latent=lat[i : i + 1],
                    text_embeds=prompt_embeds,
                    scheduler=scheduler,
                    args=eargs,
                    latent_size=latent_size,
                    controlnet=controlnet,
                    all_noise=None,
                    controlnet_cond=cond,
                )
                errors_list.append(class_errors_i)

            errors = torch.stack(errors_list, dim=0)
            logits = -errors
            preds_idx = torch.argmax(logits, dim=1)
            preds = pd_f.prompt_to_class[preds_idx.item()].item()
            loss = errors[torch.arange(B, device=errors.device), preds].mean()

        correct += (preds == label).sum().item()
        total += label.numel()

        torch.save(dict(errors=errors.detach().cpu(), pred=preds, label=label), fname)

        if step % 20 == 1:
            print(f"[validate] step={step}  running_acc={correct / max(1, total):.4f}")

    val_acc = correct / max(1, total)
    print(f"[validate] Done. val_acc={val_acc:.4f}")
    return val_acc

def validate_dc_with_gradcam(
    loader, front, unet, controlnet, vae, prompt_embeds, prompt_to_class, num_classes,
    scheduler, eargs, img_size, torch_dtype, use_amp, pd_f, reduce="mean", output_dir=None
):
    """
    GradCAM mit reduziertem Speicherverbrauch: Gradienten im Latent-Raum,
    Visualisierung im Original-Volumenraum (T, H, W).
    """

    image_dir = os.path.join(output_dir, 'images')
    os.makedirs(image_dir, exist_ok=True)
    cams_dir = os.path.join(output_dir, "images_grad_cams")
    os.makedirs(cams_dir, exist_ok=True)

    front.to(device).eval()
    vae.to(device).eval()
    controlnet.to(device).eval()
    unet.to(device).eval()

    latent_size = img_size // 8
    out_hw = (img_size, img_size)
    formatstr = get_formatstr(len(loader) - 1)
    total, correct = 0, 0

    print("[validate] Starting validation iteration (GradCAM latent->volume)...")

    for step, (vol, label, _) in enumerate(loader, start=1):
        fname = os.path.join(output_dir, formatstr.format(step) + '.pt')
        vol = vol.to(device)
        if vol.dim() == 6:
            vol = vol.squeeze(1)
        label = label.to(device).long()

        with torch.autocast(device_type="cuda", dtype=torch_dtype, enabled=use_amp):
            x = vol.mean(dim=1)
            B, T, H, W = x.shape

            frames = x.unsqueeze(2).repeat(1, 1, 3, 1, 1).view(B * T, 3, H, W)
            frames = F.interpolate(frames, size=out_hw, mode="bilinear", align_corners=False)
            frames = frames.clamp(0, 1) * 2 - 1

            lat_chunks = []
            bt = frames.shape[0]
            chunk = getattr(eargs, "vae_chunk", 256)
            with torch.no_grad():
                for s in range(0, bt, chunk):
                    lat_chunks.append(
                        encode_all_frames_with_vae(frames[s:s+chunk], vae, scaling=0.18215)
                    )

            latents_flat = torch.cat(lat_chunks, dim=0).to(device)
            lat_stack = latents_flat.view(B, T, 4, latent_size, latent_size)
            lat_stack.requires_grad_(True)

            lat = front(lat_stack)

            vae_ = _unwrap_vae(vae)
            with torch.no_grad():
                img_for_cam = vae_.decode(lat / 0.18215).sample
            img_for_cam = ((img_for_cam.clamp(-1, 1) + 1) / 2).to(device)
            control_cond_img = img_for_cam

            imgs01 = img_for_cam.detach().cpu().to(torch.float32)
            for bi in range(imgs01.shape[0]):
                arr = imgs01[bi].numpy().transpose(1, 2, 0)
                arr = np.clip(arr, 0.0, 1.0)
                save_path = os.path.join(image_dir, f"{formatstr.format(step)}_{label.item()}_{bi:02d}.tiff")
                io.imsave(save_path, arr, check_contrast=False)

            errors_list = []
            for i in range(B):
                cond = control_cond_img[i:i+1]
                _, _, class_errors_i = eval_prob_adaptive_differentiable(
                    unet=unet,
                    latent=lat[i:i+1],
                    text_embeds=prompt_embeds,
                    scheduler=scheduler,
                    args=eargs,
                    latent_size=latent_size,
                    controlnet=controlnet,
                    all_noise=None,
                    controlnet_cond=cond
                )
                errors_list.append(class_errors_i)

            errors = torch.stack(errors_list, dim=0)
            logits = -errors
            preds_idx = torch.argmax(logits, dim=1)
            preds = pd_f.prompt_to_class[preds_idx.item()].item()

            loss = logits[torch.arange(B, device=logits.device), preds_idx].mean()

        correct += (preds == label).sum().item()
        total += label.numel()
        torch.save(dict(errors=errors.detach().cpu(), pred=preds, label=label), fname)

        grads_lat = torch.autograd.grad(loss, lat_stack, retain_graph=False, create_graph=False)[0]
        grads_lat_abs = grads_lat.abs()

        grads_mean = grads_lat_abs.mean(dim=2)
        vol_slices_orig = x.detach()

        for bi in range(B):
            vol_bi = vol_slices_orig[bi].detach().cpu()
            rel_bi = grads_mean[bi:bi+1]

            rel_bi_up = F.interpolate(rel_bi, size=(H, W), mode="bilinear", align_corners=False)[0]
            rel_bi_up = rel_bi_up.detach().cpu()

            max_val = rel_bi_up.max()
            if max_val > 0:
                rel_bi_up /= max_val
            else:
                rel_bi_up = torch.zeros_like(rel_bi_up)

            vol_np = vol_bi.numpy()
            rel_np = rel_bi_up.numpy()

            path_cam_for_step = os.path.join(
                cams_dir,
                f"{formatstr.format(step)}_lbl_{label.item()}_pred_{preds}_cam_b{bi:02d}"
            )
            os.makedirs(path_cam_for_step, exist_ok=True)

            video_out = os.path.join(path_cam_for_step, "gradcam_volume")
            save_gradcam_video(vol_np, rel_np, video_out, fps=20)

        if step % 20 == 1:
            print(f"[validate] step={step}  running_acc={correct/max(1,total):.4f}")

        del x, frames, latents_flat, lat_stack, lat, img_for_cam, errors, logits, grads_lat, grads_lat_abs
        torch.cuda.empty_cache()

    val_acc = correct / max(1, total)
    print(f"[validate] Done. val_acc={val_acc:.4f}")
    return val_acc


def validate_dc_with_gradcam_big(
    loader, front, unet, controlnet, vae, prompt_embeds, prompt_to_class, num_classes,
    scheduler, eargs, img_size, torch_dtype, use_amp, pd_f, reduce="mean", output_dir=None
):
    """Is not possible, due to extreme memory consumption."""

    image_dir = os.path.join(output_dir, 'images')
    os.makedirs(image_dir, exist_ok=True)
    cams_dir = os.path.join(output_dir, "images_grad_cams")
    os.makedirs(cams_dir, exist_ok=True)

    front.to(device).eval()
    vae.to(device).eval()
    controlnet.to(device).eval()
    unet.to(device).eval()

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

        with torch.autocast(
            device_type="cuda" if device == "cuda" else "cpu",
            dtype=torch_dtype,
            enabled=use_amp
        ):
            x = vol.mean(dim=1)
            x.requires_grad_(True)

            B, T, H, W = x.shape
            frames = x.unsqueeze(2).repeat(1, 1, 3, 1, 1).view(B * T, 3, H, W)
            frames = F.interpolate(frames, size=out_hw, mode="bilinear", align_corners=False)
            frames = frames.clamp(0, 1) * 2 - 1
            lat_chunks = []
            bt = frames.shape[0]
            chunk = getattr(eargs, "vae_chunk", 256)
            for s in range(0, bt, chunk):
                lat_chunks.append(
                    encode_all_frames_with_vae(
                        frames[s:s+chunk], vae, scaling=0.18215
                    )
                )
            latents_flat = torch.cat(lat_chunks, dim=0).to(device)
            lat_stack = latents_flat.view(B, T, 4, latent_size, latent_size)

            lat = front(lat_stack)
            vae_ = _unwrap_vae(vae)
            with torch.no_grad():
                img_for_cam = vae_.decode(lat / 0.18215).sample
            img_for_cam = ((img_for_cam.clamp(-1, 1) + 1) / 2).to(device)
            control_cond_img = img_for_cam


            imgs01 = img_for_cam.detach().cpu().to(torch.float32)
            for bi in range(imgs01.shape[0]):
                arr = imgs01[bi].numpy().transpose(1, 2, 0)
                arr = np.clip(arr, 0.0, 1.0)
                save_path = os.path.join(image_dir, f"{formatstr.format(step)}_{label.item()}_{bi:02d}.tiff")
                io.imsave(save_path, arr, check_contrast=False)

            errors_list = []
            for i in range(B):
                cond = control_cond_img[i:i+1] if control_cond_img is not None else None
                _, _, class_errors_i = eval_prob_adaptive_differentiable(
                    unet=unet,
                    latent=lat[i:i+1],
                    text_embeds=prompt_embeds,
                    scheduler=scheduler,
                    args=eargs,
                    latent_size=latent_size,
                    controlnet=controlnet,
                    all_noise=None,
                    controlnet_cond=cond
                )
                errors_list.append(class_errors_i)

            errors = torch.stack(errors_list, dim=0)
            logits = -errors
            preds_idx = torch.argmax(logits, dim=1)
            preds = pd_f.prompt_to_class[preds_idx.item()].item()

            loss = errors[torch.arange(B, device=errors.device), preds].mean()

            grads_x = torch.autograd.grad(
                loss,
                x,
                retain_graph=False,
                create_graph=False
            )[0] 

            grads_x_abs = grads_x.abs()

            for bi in range(B):
                vol_slices = x[bi].detach().cpu()         
                relevance = grads_x_abs[bi].detach().cpu() 

                max_val = relevance.max()
                if max_val > 0:
                    relevance /= max_val
                else:
                    relevance = torch.zeros_like(relevance)

                vol_np = vol_slices.numpy()
                rel_np = relevance.numpy()

                path_cam_for_step = os.path.join(
                    cams_dir,
                    f"{formatstr.format(step)}_lbl_{label.item()}_pred_{preds}_cam_b{bi:02d}"
                )
                os.makedirs(path_cam_for_step, exist_ok=True)

                video_out = os.path.join(path_cam_for_step, "gradcam")
                save_gradcam_video(vol_np, rel_np, video_out, fps=20)

        correct += (preds == label).sum().item()
        total += label.numel()
        torch.save(dict(errors=errors.detach().cpu(),
                        pred=preds, label=label), fname)

        if step % 20 == 1:
            print(f"[validate] step={step}  running_acc={correct/max(1,total):.4f}")

    val_acc = correct / max(1, total)
    print(f"[validate] Done. val_acc={val_acc:.4f}")
    return val_acc



def old_validate(loader, front, backbone, torch_dtype, use_amp, img_out_size, imagenet_mean, imagenet_std, output_dir=None):
    """Validation loop with AMP and ImageNet normalization."""

    image_dir = os.path.join(output_dir, 'images')
    os.makedirs(image_dir, exist_ok=True)
    cams_dir = os.path.join(output_dir, "images_grad_cams")
    os.makedirs(cams_dir, exist_ok=True)
    
    front.to(device).eval()
    backbone.to(device).eval()
    total, correct = 0, 0
    print("[validate] Starting validation iteration...")
    for step, (vol, label, _) in enumerate(loader, start=1):
        formatstr = get_formatstr(len(loader) - 1)
        vol = vol.to(device)
        if vol.dim() == 6:
            vol = vol.squeeze(1)
        label = label.to(device).long()

        with torch.autocast(device_type="cuda", dtype=torch_dtype, enabled=use_amp):
            img_rgb = front(vol)
            if img_out_size is not None and (img_rgb.shape[-2] != img_out_size or img_rgb.shape[-1] != img_out_size):
                img_rgb = F.interpolate(img_rgb, size=(img_out_size, img_out_size),
                                        mode="bilinear", align_corners=False)
            img_in = normalize_batch(img_rgb, imagenet_mean, imagenet_std)
            logits = backbone(img_in)
            preds = torch.argmax(logits, dim=1)

        correct += (preds == label).sum().item()
        total += label.numel()
        
        torch.save(dict(pred=preds.detach().cpu(), label=label.cpu()), os.path.join(output_dir,  formatstr.format(step) + '.pt'))
        
        vol_for_cam = vol
        if vol_for_cam.dim() == 3:
            vol_for_cam = vol_for_cam.unsqueeze(0)

        vol_slices_np, relevance_np, target_cls = compute_volume_gradcam(
            vol=vol_for_cam,
            front=front,
            backbone=backbone,
            imagenet_mean=imagenet_mean,
            imagenet_std=imagenet_std,
            img_out_size=img_out_size,
            final_dtype=torch_dtype,
            use_amp_final=use_amp,
            target_class=preds.item(),  
        )
        path_cam_for_step = os.path.join(
            cams_dir,
            f"{formatstr.format(step)}_lbl_{label.item()}_pred_{preds.item()}_cam"
        )
        os.makedirs(path_cam_for_step, exist_ok=True)

        video_path = os.path.join(path_cam_for_step, "gradcam_volume.mp4")
        save_gradcam_video(vol_slices_np, relevance_np, video_path, fps=20)

    val_acc = correct / max(1, total)
    print(f"[validate] Done. val_acc={val_acc:.4f}")
    return val_acc

def parse_args():
    p = argparse.ArgumentParser()
    # Daten
    p.add_argument('--dataset', default='thz_for_adapter', type=str)
    p.add_argument('--pretrained_path', default=None, required=True)
    p.add_argument('--split', default='test', choices=['train','test'])
    # Auswahl
    p.add_argument('--classifier', required=True, choices=('diffusion', 'resnet50', "vit_b_16", "vit_b_32", "convnext_tiny")) 
    p.add_argument('--adapter', required=True, choices=("rgb", "cn_wrapper", "latent", "old_cn_wrapper"))  

    #output
    p.add_argument('--output_dir', default='./results_eval_pipeline', type=str)

    p.add_argument('--n_workers', type=int, default=1)

    return p.parse_args()

def main(args):
    
    torch.backends.cudnn.benchmark = True

    ckpt = torch.load(args.pretrained_path, map_location='cpu')
    img_size_ckpt = ckpt["img_size"] if "img_size" in ckpt else ckpt.get("img_out_size", 224)
    # Dataset
    ds = get_target_dataset(args.dataset, train=args.split=='train')
    ds_loader = DataLoader(ds, batch_size=1, shuffle=False,
                                  num_workers=args.n_workers, pin_memory=True)
    ckpt["dtype"]='bfloat16'

    # Ausgabeordner
    from datetime import datetime
    from zoneinfo import ZoneInfo 

    stamp = datetime.now(ZoneInfo("Europe/Berlin")).strftime("%y%m%d_%H%M")
    output_dir = os.path.join(args.output_dir, args.classifier + (f"_{args.adapter}" if args.adapter else ""), stamp)
    os.makedirs(output_dir, exist_ok=True)
    result_dir = os.path.join(output_dir, 'results')
    os.makedirs(result_dir, exist_ok=True)
    

    # Classifier bauen
    'TODO: Mehrere Classifier unterstützen'
    clf = None
    if args.classifier == "diffusion":
        print("Baue Diffusion Zero-Shot Classifier...")
        vae_f, unet_f, tokenizer_f, text_encoder_f, scheduler_f, controlnet_f = build_sd2_1_base(
            dtype=ckpt["dtype"], use_xformers=False, train_all=False, version=ckpt["version"]
        )
        unet_f = unet_f.to(device).eval()
        vae_f = vae_f.to(device).eval()
        controlnet_f.load_state_dict(ckpt["controlnet_state_dict"], strict=False)
        controlnet_f = controlnet_f.to(device).eval()
        pb_f = PromptBank(ckpt["prompts_csv"])
        prompt_embeds_f = pb_f.to_text_embeds(tokenizer_f, text_encoder_f, device)
        prompt_to_class_f = pb_f.prompt_to_class.to(device)
        num_classes_f = pb_f.num_classes
    else:
        print("Baue Torchvision Classifier...")
        backbone_name = ckpt["backbone"]
        num_classes = ckpt["num_classes"]
        imagenet_mean = tuple(ckpt["imagenet_mean"])
        imagenet_std = tuple(ckpt["imagenet_std"])

        backbone, expected_size, mean_std = build_backbone(backbone_name, num_classes, pretrained=True)
        backbone.load_state_dict(ckpt["backbone_state_dict"], strict=False)
        backbone = backbone.to(device).eval()

    # Adapter bauen
    adapter = None
    weights_adapter = ckpt["adapter_state_dict"] if "adapter_state_dict" in ckpt else ckpt["front_state_dict"]
    if args.adapter == "old_cn_wrapper":
        controlnet_cfg = dict(
            spatial_dims=3,
            num_res_blocks=(2, 2, 2, 2),
            num_channels=(32, 64, 64, 64),
            attention_levels=(False, False, False, False),
            conditioning_embedding_in_channels=2,
            conditioning_embedding_num_channels=(32, 64, 64, 64),
            with_conditioning=False,
        )
        front_f = old_ControlNetAdapterWrapper(
            controlnet_cfg=controlnet_cfg,
            in_channels=2,
            out_size=expected_size,
            target_T=64,
            stride_T=4
        ).to(device)
        front_f.load_state_dict(weights_adapter, strict=False, assign=True)
        front_f.eval()
    elif args.adapter == "cn_wrapper":
        print("Lade ControlNet-Wrapper Adapter...")
        controlnet_cfg = dict(
            spatial_dims=3,
            num_res_blocks=(2, 2, 2, 2),
            num_channels=(32, 64, 64, 64),
            attention_levels=(False, False, False, False),
            conditioning_embedding_in_channels=2,
            conditioning_embedding_num_channels=(32, 64, 64, 64),
            with_conditioning=False,
        )
        front_f = ControlNetAdapterWrapper(
            controlnet_cfg=controlnet_cfg,
            in_channels=2,
            out_size=256,
            target_T=64,
            stride_T=4
        ).to(device)
        front_f.load_state_dict(weights_adapter, strict=False, assign=True)
        front_f.eval()
    elif args.adapter == "latent":
        print("Lade Latent-Adapter...")
        front_f = LatentMultiChannelAdapter(k_t=5, use_attn_pool=True).to(device)
        front_f.load_state_dict(weights_adapter, strict=False, assign=True)
        front_f.eval()
    else:
        raise ValueError(f"Unbekannter Adapter-Typ: {args.adapter}")

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
        eargs_f.cond_scale = 1.0
        
        use_amp_final = (device == "cuda") and (ckpt["dtype"] in ("float16", "bfloat16"))
        final_torch_dtype = torch.float16 if ckpt["dtype"] == "float16" else (
            torch.bfloat16 if ckpt["dtype"] == "bfloat16" else torch.float32
        )
        if torch.cuda.device_count() > 1:
            print("Using", torch.cuda.device_count(), "GPUs")
            front_f = torch.nn.DataParallel(front_f)
            unet_f = torch.nn.DataParallel(unet_f)
            controlnet_f = torch.nn.DataParallel(controlnet_f)
            vae_f = torch.nn.DataParallel(vae_f)
        else:
            print("Using a single GPU")

        final_acc = validate_dc_with_gradcam(
        ds_loader, front_f, unet_f, controlnet_f, vae_f,
        prompt_embeds_f, prompt_to_class_f, num_classes_f,
        scheduler_f, eargs_f, img_size_ckpt,
        torch_dtype=final_torch_dtype, use_amp=use_amp_final, pd_f=pb_f,
        reduce="mean", output_dir=result_dir
        )
        print(f"Final Accuracy: {final_acc:.4f}")
        
        
    else:
        if args.adapter == "old_cn_wrapper":
            final_acc = old_validate(ds_loader, front_f, backbone, torch.float32, False, img_size_ckpt, imagenet_mean, imagenet_std, output_dir=result_dir)
        else:    
            final_acc = validate_base(
                ds_loader, front_f,
                backbone, imagenet_mean, imagenet_std,
                img_size_ckpt, final_dtype=torch.float32,
                use_amp_final=False,
                output_dir=result_dir
            )
        print(f"Final Accuracy: {final_acc:.4f}")
    try: 
        evaluate_predictions(
            result_dir,
            output_dir=os.path.join(output_dir, "evaluation"),
            prompts_csv_path=ckpt.get("prompts_csv", None)
        )
        print(f"Results evaluation completed successfully.")
    except Exception as e:
        print(f"Error in evaluation: {e}")


if __name__ == "__main__":
    args = parse_args()
    main(args)