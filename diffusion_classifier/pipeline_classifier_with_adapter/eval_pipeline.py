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

def _is_vit_backbone(backbone):
    name = backbone.__class__.__name__.lower()
    return ("visiontransformer" in name) or ("vit" in name)


def vit_reshape_transform(tensor):
   
    B, seq_len, C = tensor.shape
    n_patches = seq_len - 1  
    h = w = int(n_patches ** 0.5)  
    x = tensor[:, 1:, :]             
    x = x.reshape(B, h, w, C)        
    x = x.permute(0, 3, 1, 2)         
    return x


def _get_cam_target_layers(backbone):
    name = backbone.__class__.__name__.lower()

    if _is_vit_backbone(backbone):
        try:
            last_block = list(backbone.encoder.layers.children())[-1]
            return [last_block.ln_1]
        except Exception:
            return [backbone.encoder]
    if hasattr(backbone, "layer4"):
        return [backbone.layer4[-1]]
    if "convnext" in name and hasattr(backbone, "features"):
        return [backbone.features[-1]]
    if hasattr(backbone, "features"):
        return [backbone.features]
    return [backbone]


def _save_cam_tiffs(cam01, rgb01, cam_path, overlay_path, alpha=0.35):
    cam = cam01
    cam = cam - cam.min()
    cam = cam / (cam.max() + 1e-8)
    io.imsave(cam_path, (cam * 65535).astype(np.uint16), check_contrast=False)
    cam_rgb = np.repeat(cam[..., None], 3, axis=2)
    over = (1 - alpha) * rgb01 + alpha * cam_rgb
    over = np.clip(over, 0.0, 1.0)
    io.imsave(overlay_path, (over * 65535).astype(np.uint16), check_contrast=False)
    plt.imshow(rgb01)
    plt.imshow(cam01, cmap='jet', alpha=0.35)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(cam_path.replace('.tiff', '.pdf'), bbox_inches='tight', pad_inches=0)
    plt.close()

def validate_base(loader, front, backbone, imagenet_mean, imagenet_std,
                  img_out_size, final_dtype, use_amp_final, output_dir):
    total, correct = 0, 0
    image_dir = os.path.join(output_dir, 'images')
    os.makedirs(image_dir, exist_ok=True)
    cams_dir = os.path.join(output_dir, "images_grad_cams")
    os.makedirs(cams_dir, exist_ok=True)
    target_layers = _get_cam_target_layers(backbone)
    CamCls = GradCAMPlusPlus
    is_vit = _is_vit_backbone(backbone)
    cam = CamCls(
        model=backbone,
        target_layers=target_layers,
        reshape_transform=vit_reshape_transform if is_vit else None
    )
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
            img_np = img_rgb.squeeze(0).detach().cpu().numpy()  
            img_np = np.transpose(img_np, (1, 2, 0)) 
            img_np = img_np - img_np.min()
            img_np = img_np / (img_np.max() + 1e-8)
            save_path = os.path.join(image_dir, f"sample_{label.item()}.tiff")
            io.imsave(save_path, img_np, check_contrast=False)
            logits = backbone(img_in)
            preds = torch.argmax(logits, dim=1)
            targets = [ClassifierOutputTarget(int(preds.item()))]
            grayscale_cam = cam(input_tensor=img_in, targets=targets)[0]
            rgb01 = img_rgb.squeeze(0).detach().cpu().numpy()
            rgb01 = np.transpose(rgb01, (1, 2, 0))
            rgb01 = np.clip(rgb01, 0.0, 1.0)
            cam_path = os.path.join(cams_dir, f"{formatstr.format(i)}_lbl{label.item()}_cam.tiff")
            overlay_path = os.path.join(cams_dir, f"{formatstr.format(i)}_lbl{label.item()}_overlay.tiff")
            _save_cam_tiffs(grayscale_cam, rgb01, cam_path, overlay_path)

        correct += (preds == label).sum().item()
        total += label.numel()
        formatstr = get_formatstr(len(loader) - 1)
        torch.save(dict(pred=preds.detach().cpu(), label=label.cpu()), os.path.join(output_dir,  formatstr.format(i) + '.pt'))
    final_acc = correct / max(1, total)
    print(f"[validate_base] Done. val_acc={final_acc:.4f}")
    return final_acc


def encode_all_frames_with_vae(frames_vae, vae, scaling=0.18215):
    posterior = vae.encode(frames_vae).latent_dist.mean.to(torch.float32)
    latents = posterior * scaling
    return latents

def validate_dc(
    loader, front, unet, controlnet, vae, prompt_embeds, prompt_to_class, num_classes,
    scheduler, eargs, img_size, torch_dtype, use_amp,  reduce="mean", output_dir=None
):
    image_dir = os.path.join(output_dir, 'images')
    os.makedirs(image_dir, exist_ok=True)
    cams_dir = os.path.join(output_dir, "images_grad_cams")
    os.makedirs(cams_dir, exist_ok=True)

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

        with torch.autocast(device_type="cuda" if device=="cuda" else "cpu", dtype=torch_dtype, enabled=use_amp):
            x = vol.mean(dim=1)
            B, T, H, W = x.shape
            frames = x.unsqueeze(2).repeat(1,1,3,1,1).view(B*T,3,H,W)
            frames = F.interpolate(frames, size=out_hw, mode="bilinear", align_corners=False)
            frames = frames.clamp(0,1)*2-1
            lat_chunks = []
            bt = frames.shape[0]
            chunk = getattr(eargs, "vae_chunk", 256)
            with torch.no_grad():
                for s in range(0, bt, chunk):
                    lat_chunks.append(encode_all_frames_with_vae(frames[s:s+chunk], vae, scaling=0.18215))
            latents_flat = torch.cat(lat_chunks, dim=0)
            lat_stack = latents_flat.view(B, T, 4, latent_size, latent_size)
            lat = front(lat_stack)
            try:
                cn_in_ch = controlnet.controlnet_cond_embedding.conv_in.in_channels
            except Exception:
                cn_in_ch = getattr(controlnet, "in_channels", 4)

            if cn_in_ch == 3:
                control_cond_img = vae.decode(lat / 0.18215).sample  # ohne no_grad
                control_cond_img = ((control_cond_img.clamp(-1, 1) + 1.0) / 2.0)
                control_cond_img = control_cond_img.to(lat.device)
                control_cond_img.requires_grad_(True)
            else:
                control_cond_img = None

            imgs01 = ((control_cond_img.detach().float().clamp(-1, 1) + 1.0) / 2.0).cpu() 
            Bc = imgs01.shape[0]
            for bi in range(Bc):
                arr = imgs01[bi].numpy() 
                arr = np.transpose(arr, (1, 2, 0))  
                save_path = os.path.join(image_dir, f"{formatstr.format(step)}_{label.item()}_{bi:02d}.tiff")
                io.imsave(save_path, arr, check_contrast=False)

            B = lat.size(0)
            errors_list = []
            for i in range(B):
                cond_to_pass = control_cond_img[i:i+1] if control_cond_img is not None else None
                _, _, class_errors_i = eval_prob_adaptive_differentiable(
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
                errors_list.append(class_errors_i)

            errors = torch.stack(errors_list, dim=0).to(label.dtype if label.is_floating_point() else torch.float32)
            logits = -errors
            preds = torch.argmax(logits, dim=1) 

            if control_cond_img is None:
                with torch.no_grad():
                    decoded = vae.decode(lat / 0.18215).sample
                img_for_cam = ((decoded.clamp(-1, 1) + 1.0) / 2.0).detach()
            else:
                img_for_cam = ((control_cond_img.clamp(-1, 1) + 1.0) / 2.0).detach()

            img_for_cam = img_for_cam.clone().to(lat.device).requires_grad_(True)

            B = errors.size(0)  
            loss = errors[torch.arange(B, device=errors.device), preds].mean()

            for p in unet.parameters():
                p.requires_grad_(False)
            if controlnet is not None:
                for p in controlnet.parameters():
                    p.requires_grad_(False)

            grads = torch.autograd.grad(loss, img_for_cam, retain_graph=False, create_graph=False)[0]
            sal = grads.abs().sum(dim=1)
            sal = sal / (sal.amax(dim=(1, 2), keepdim=True) + 1e-8)

            for bi in range(sal.size(0)):
                cam01 = sal[bi].detach().cpu().numpy()
                rgb01 = img_for_cam[bi].detach().cpu().numpy()
                rgb01 = np.transpose(rgb01, (1, 2, 0))
                cam_path = os.path.join(cams_dir, f"{formatstr.format(step)}_{bi:02d}_cam_errgrad.tiff")
                overlay_path = os.path.join(cams_dir, f"{formatstr.format(step)}_{bi:02d}_overlay_errgrad.tiff")
                _save_cam_tiffs(cam01, rgb01, cam_path, overlay_path)

        correct += (preds == label).sum().item()
        total += label.numel()
        torch.save(dict(errors=errors.detach().cpu(), pred=preds.detach().cpu(), label=label), fname)
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
    target_layers = _get_cam_target_layers(backbone)
    CamCls = GradCAMPlusPlus
    is_vit = _is_vit_backbone(backbone)
    cam = CamCls(
        model=backbone,
        target_layers=target_layers,
        reshape_transform=vit_reshape_transform if is_vit else None
    )
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
            img_np = img_rgb.squeeze(0).detach().cpu().numpy()  
            img_np = np.transpose(img_np, (1, 2, 0)) 
            img_np = img_np - img_np.min()
            img_np = img_np / (img_np.max() + 1e-8)
            save_path = os.path.join(image_dir, f"sample_{label.item()}.tiff")
            io.imsave(save_path, img_np, check_contrast=False)
            logits = backbone(img_in)
            preds = torch.argmax(logits, dim=1)
            targets = [ClassifierOutputTarget(int(preds.item()))]
            grayscale_cam = cam(input_tensor=img_in, targets=targets)[0]
            rgb01 = img_rgb.squeeze(0).detach().cpu().numpy()
            rgb01 = np.transpose(rgb01, (1, 2, 0))
            rgb01 = np.clip(rgb01, 0.0, 1.0)
            cam_path = os.path.join(cams_dir, f"{formatstr.format(step)}_lbl{label.item()}_cam.tiff")
            overlay_path = os.path.join(cams_dir, f"{formatstr.format(step)}_lbl{label.item()}_overlay.tiff")
            _save_cam_tiffs(grayscale_cam, rgb01, cam_path, overlay_path)

        correct += (preds == label).sum().item()
        total += label.numel()
        
        torch.save(dict(pred=preds.detach().cpu(), label=label.cpu()), os.path.join(output_dir,  formatstr.format(step) + '.pt'))
        if step % 50 == 0 or step == 1:
            print(f"[validate] step={step}  running_acc={correct/max(1,total):.4f}")

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
    if ckpt["dtype"]=='float16': embeds = embeds.half()

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
        controlnet_f.base.load_state_dict(ckpt["controlnet_state_dict"], strict=False)
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
            out_size=expected_size,
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
        
        use_amp_final = (device == "cuda") and (ckpt["dtype"] in ("float16", "bfloat16"))
        final_torch_dtype = torch.float16 if ckpt["dtype"] == "float16" else (
            torch.bfloat16 if ckpt["dtype"] == "bfloat16" else torch.float32
        )

        final_acc = validate_dc(
        ds_loader, front_f, unet_f, controlnet_f, vae_f,
        prompt_embeds_f, prompt_to_class_f, num_classes_f,
        scheduler_f, eargs_f, img_size_ckpt,
        torch_dtype=final_torch_dtype, use_amp=use_amp_final, reduce="mean", output_dir=result_dir
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