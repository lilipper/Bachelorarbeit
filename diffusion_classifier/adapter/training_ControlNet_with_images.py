#!/usr/bin/env python
"""
Train ControlNet (Stable Diffusion 2.1) with Terahertz-Volumen als Conditioning.

Voraussetzungen (getestet mit PyTorch 2.x + CUDA, Diffusers >= 0.27):
  pip install -U diffusers transformers accelerate safetensors bitsandbytes
  # Optional für Speed/VRAM:
  pip install -U xformers

Datensatz-Format (JSONL): pro Zeile ein JSON-Objekt mit Schlüsseln:
  {
    "thz_path": "/pfad/zum/scan.npy|.pt|.pth",      # Volumen (D,H,W) oder (1,D,H,W) float32
    "image_path": "/pfad/zum/target_rgb.jpg",        # RGB-Zielbild
    "prompt": "a photo of a ..."                      # Textbeschreibung
  }

Beispiel-Aufruf (eine GPU):
  accelerate launch train_thz_controlnet_sd21.py \
    --dataset_jsonl /data/train.jsonl \
    --output_dir ./runs/thz_controlnet_sd21 \
    --version 2-1 \
    --resolution 512 --train_batch_size 1 --gradient_accumulation_steps 4 \
    --learning_rate 1e-5 --num_train_epochs 10 --enable_xformers

Hinweis: Dieses Skript friert VAE, Textencoder und UNet ein. Trainiert werden
ControlNet + THzAdapter (3D→2D). Das ist stabil bei kleinem Datensatz.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
from dataclasses import dataclass
from typing import List

import numpy as np
from PIL import Image
import pandas as pd
import process_rdf as prdf
import tiffile as tiff
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed

from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    ControlNetModel,
)
from transformers import AutoTokenizer, CLIPTextModel
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# ------------------------- THz → 2D Adapter ------------------------- #
class THzAdapter(nn.Module):
    """
    Low-Mem 3D→2D-Adapter:
      - optionales Depth/Spatial-Downsampling vor 3D-Conv (riesige Einsparung)
      - Mixed Precision freundlich
      - am Ende Upscale auf Targetauflösung und 3-Kanal-Head
    """
    def __init__(self, ch: int = 8,   # war 16 → halbiert Kanäle
                 max_T: int = 64,     # depth subsample, z.B. auf 64 Frames
                 hw: int = 128,       # 3D-Processing auf 128x128 statt 512x512
                 out_hw: int = 512):  # Zielauflösung für Control-Bild
        super().__init__()
        self.max_T = max_T
        self.hw = hw
        self.out_hw = out_hw

        # sehr leichte 3D-Feature-Extraktion
        self.enc3d = nn.Sequential(
            nn.Conv3d(1, ch,   3, padding=1), nn.SiLU(),
            nn.Conv3d(ch, ch*2, 3, stride=2, padding=1), nn.SiLU(),  # halbiert T/H/W
            nn.Conv3d(ch*2, ch*4, 3, stride=2, padding=1), nn.SiLU(),# erneut halbiert
        )
        self.pool = nn.AdaptiveAvgPool3d((1, None, None))  # Depth → 1
        self.refine2d = nn.Sequential(
            nn.Conv2d(ch*4, ch*2, 3, padding=1), nn.SiLU(),
            nn.Conv2d(ch*2, ch,   3, padding=1), nn.SiLU(),
        )
        self.head = nn.Conv2d(ch, 3, 1)

    def forward(self, vol: torch.Tensor) -> torch.Tensor:
        """
        vol: (B,1,T,H,W), float16/32
        returns: (B,3,out_hw,out_hw)
        """
        B, C, T, H, W = vol.shape

        # 1) Downsample Depth/Spatial VOR 3D-Conv
        T_ds = min(T, self.max_T)
        if (T != T_ds) or (H != self.hw) or (W != self.hw):
            vol = F.interpolate(
                vol, size=(T_ds, self.hw, self.hw),
                mode="trilinear", align_corners=False
            )

        # 2) 3D-Encoder
        x = self.enc3d(vol)               # (B, C', T', H', W')
        x = self.pool(x).squeeze(2)       # (B, C', H', W'), Depth kollabiert

        # 3) 2D-Refine + Upscale auf Zielauflösung
        x = self.refine2d(x)              # (B, ch, H', W')
        if x.shape[-1] != self.out_hw:
            x = F.interpolate(x, size=(self.out_hw, self.out_hw), mode="bilinear", align_corners=False)
        rgb = self.head(x)                # (B,3, out_hw, out_hw)
        return rgb

# ------------------------------ Dataset ----------------------------- #
class ThzPromptsDataset(Dataset):
    """
    Lädt Daten aus einer JSON-Datei mit Struktur:
    {
      "prompts": [
        {"source": "/path/file.mat", "target": "/path/image.tiff", "prompt": "a scan with..."},
        ...
      ]
    }

    Gibt pro Sample zurück:
      - "thz":  (1, T, H, W)  Float16  (normiert, geflippt)
      - "pixel_values": (3, R, R) Float16 in [-1, 1]
      - "prompt": str
    """
    def __init__(
        self,
        json_path: str,
        resolution: int = 512,
        transform_vol=None,
        transform_img=None,
    ):
        super().__init__()
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert "prompts" in data and isinstance(data["prompts"], list), "JSON muss 'prompts' Liste enthalten."
        self.items = data["prompts"]
        self.resolution = int(resolution)
        self.transform_vol = transform_vol  # Callable(tensor[B,C,T,H,W]) -> tensor
        self.transform_img = transform_img  # Callable(tensor[3,H,W]) -> tensor

    def __len__(self):
        return len(self.items)

    @torch.no_grad()
    def _load_target_image(self, path: str) -> torch.Tensor:
        # (3, R, R) in [-1, 1]
        if (path.lower().endswith(".tif") or path.lower().endswith(".tiff")):
            arr = tiff.imread(path)
            # in RGB bringen (manche Tiffs sind graustufig)
            if arr.ndim == 2:
                arr = np.stack([arr, arr, arr], axis=-1)
            elif arr.ndim == 3 and arr.shape[-1] == 1:
                arr = np.repeat(arr, 3, axis=-1)
            img = Image.fromarray(arr.astype(np.uint8))
        else:
            img = Image.open(path).convert("RGB")

        img = img.resize((self.resolution, self.resolution), Image.BICUBIC)
        img = np.asarray(img).astype(np.float16) / 255.0  # [0,1]
        img = img.transpose(2, 0, 1)                      # (3,H,W)
        img = img * 2.0 - 1.0                             # [-1,1]
        return torch.from_numpy(img)

    @torch.no_grad()
    def _load_thz_volume(self, path: str) -> torch.Tensor:
        """
        Lädt .mat → Prozesskette:
          read_mat -> process_complex_data -> |.|^2 / max -> flip Höhe
        Bringt auf Auflösung (R,R) via trilinear, Form (1, T, H, W).
        """
        device = "cpu"
        complex_raw_data, parameters = prdf.read_mat(path, device=device)
        T = int(parameters["NF"])
        processed_data, max_val_abs = prdf.process_complex_data(complex_raw_data, T, device=device)  # [T,H,W] complex

        vol = torch.abs(processed_data) ** 2             # [T,H,W]
        vol = vol / (max_val_abs)
        vol = torch.flipud(vol)                  # Höhe flippen (dim=1)

        # Resize H,W auf resolution (T unverändert), dann Form (1,T,H,W)
        T_, H, W = vol.shape
        if (H != self.resolution) or (W != self.resolution):
            vol5d = vol.unsqueeze(0).unsqueeze(0)        # (1,1,T,H,W)
            vol5d = F.interpolate(
                vol5d,
                size=(T_, self.resolution, self.resolution),
                mode="trilinear",
                align_corners=False,
            )
            vol = vol5d.squeeze(0).squeeze(0)            # (T,R,R)

        vol = vol.unsqueeze(0)                            # (1,T,R,R)  (C=1)
        vol = vol.contiguous().float()
        return vol

    def __getitem__(self, idx: int):
        item = self.items[idx]
        src_path = item["source"]
        tgt_path = item["target"]
        prompt  = item.get("prompt", "")

        # THz Volumen
        thz = self._load_thz_volume(src_path)            # (1,T,R,R)
        if self.transform_vol is not None:
            # Erwartet Eingabe-Form (B,C,T,H,W) -> packe Batch-Dim für Transform
            thz = self.transform_vol(thz.unsqueeze(0)).squeeze(0)

        # Zielbild
        pixel_values = self._load_target_image(tgt_path) # (3,R,R)
        if self.transform_img is not None:
            pixel_values = self.transform_img(pixel_values)

        return {
            "thz": thz,                         # (1,T,R,R)  Float32
            "pixel_values": pixel_values,       # (3,R,R)    Float32 in [-1,1]
            "prompt": prompt,
        }

# -------------------------- Collate & Tokenizer ---------------------- #
@dataclass
class Batch:
    pixel_values: torch.FloatTensor
    thz: torch.FloatTensor
    input_ids: torch.LongTensor
    attention_mask: torch.LongTensor


def collate_fn(examples: List[dict], tokenizer: AutoTokenizer) -> Batch:
    pixel_values = torch.stack([e["pixel_values"] for e in examples])
    thz = torch.stack([e["thz"] for e in examples])
    texts = [e["prompt"] for e in examples]
    # Dropout für CFG (10% leere Prompts)
    texts = [t if random.random() > 0.1 else "" for t in texts]
    tok = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=tokenizer.model_max_length,
        return_tensors="pt",
    )
    return Batch(
        pixel_values=pixel_values,
        thz=thz,
        input_ids=tok.input_ids,
        attention_mask=tok.attention_mask,
    )

# ------------------------------ Training ---------------------------- #

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_jsonl", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default=f"./runs/thz_controlnet_sd21_{time.time()}")
    parser.add_argument("--version", type=str, default="2-1", choices=["2-1", "2-0", "1-5", "1-4", "1-3", "1-2", "1-1"], help="Stable Diffusion Version")
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--train_batch_size", type=int, default=1)
    parser.add_argument("--num_train_epochs", type=int, default=10)
    parser.add_argument("--max_train_steps", type=int, default=10000, help="Optional, überschreibt Epochen")
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2)
    parser.add_argument("--adam_epsilon", type=float, default=1e-08)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--mixed_precision", type=str, default="no", choices=["no","fp16","bf16"]) 
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpointing_steps", type=int, default=1000)
    parser.add_argument("--enable_xformers", action="store_true")
    parser.add_argument("--dtype", type=str, default="float16", choices=["float16", "float32"])  # wird zu torch.dtype gemappt
    return parser.parse_args()


def main():
    args = parse_args()
    accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps, mixed_precision=args.mixed_precision)
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    MODEL_IDS = {
        '1-1': "CompVis/stable-diffusion-v1-1",
        '1-2': "CompVis/stable-diffusion-v1-2",
        '1-3': "CompVis/stable-diffusion-v1-3",
        '1-4': "CompVis/stable-diffusion-v1-4",
        '1-5': "runwayml/stable-diffusion-v1-5",
        '2-0': "stabilityai/stable-diffusion-2-base",
        '2-1': "stabilityai/stable-diffusion-2-1-base",
    }

    model_id = MODEL_IDS[args.version]

    # dtype mapping
    torch_dtype = torch.float16 if args.dtype == "float16" else torch.float32
    logger.debug(f"Verwende {torch_dtype} für die Tensoren.")

    # ----------- Load SD components direkt (ohne Pipeline) ----------- #
    logger.info("Lade Stable Diffusion Komponenten…")
    tokenizer = AutoTokenizer.from_pretrained(model_id, subfolder="tokenizer", use_fast=False)
    text_encoder = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae", torch_dtype=torch_dtype)
    from diffusers import UNet2DConditionModel
    unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet", torch_dtype=torch_dtype)

    # einfrieren
    vae.requires_grad_(False); vae.eval()
    text_encoder.requires_grad_(False); text_encoder.eval()
    unet.requires_grad_(False); unet.eval()

    if args.enable_xformers:
        try:
            unet.enable_xformers_memory_efficient_attention()
        except Exception as e:
            logger.warning(f"Konnte xFormers nicht aktivieren: {e}")

    # ----------- Build ControlNet from UNet ----------- #
    logger.info("Initialisiere ControlNet aus UNet-Konfiguration…")
    controlnet = ControlNetModel.from_unet(unet).to(dtype=torch_dtype)
    controlnet.train()

    # ----------- THz Adapter ----------- #
    thz_adapter = THzAdapter(ch=16).to(dtype=torch_dtype)
    thz_adapter.train()

    # ----------- Noise Scheduler (Training) ----------- #
    noise_scheduler = DDPMScheduler.from_pretrained(model_id, subfolder="scheduler")

    # ----------- Dataset & Dataloader ----------- #
    train_dataset = ThzPromptsDataset(args.dataset_jsonl, resolution=args.resolution)

    def _collate(examples):
        return collate_fn(examples, tokenizer)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=_collate,
        pin_memory=True,
    )

    # ----------- Optimizer ----------- #
    try:
        import bitsandbytes as bnb
        optimizer = bnb.optim.AdamW8bit(
            list(controlnet.parameters()) + list(thz_adapter.parameters()),
            lr=args.learning_rate,
            betas=(args.adam_beta1, args.adam_beta2),
            weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon,
        )
        logger.info("Nutze AdamW 8-bit (bitsandbytes)")
    except Exception:
        optimizer = torch.optim.AdamW(
            list(controlnet.parameters()) + list(thz_adapter.parameters()),
            lr=args.learning_rate,
            betas=(args.adam_beta1, args.adam_beta2),
            weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon,
        )
        logger.info("Nutze Standard AdamW")

    # ----------- Prepare with Accelerator ----------- #
    (
        controlnet,
        thz_adapter,
        optimizer,
        train_dataloader,
        text_encoder,
        vae,
        unet,
    ) = accelerator.prepare(
        controlnet,
        thz_adapter,
        optimizer,
        train_dataloader,
        text_encoder,
        vae,
        unet,
    )

    # ----------- Training steps calc ----------- #
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps > 0:
        max_train_steps = args.max_train_steps
        num_train_epochs = math.ceil(max_train_steps / num_update_steps_per_epoch)
    else:
        num_train_epochs = args.num_train_epochs
        max_train_steps = num_train_epochs * num_update_steps_per_epoch

    logger.info(f"Start Training: {num_train_epochs} Epochen, {max_train_steps} Updates")

    # ----------- Training Loop ----------- #
    global_step = 0
    vae_scale = 0.18215  # SD2.1 Skalar

    for epoch in range(num_train_epochs):
        controlnet.train(); thz_adapter.train()
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(controlnet):
                # 1) Encode RGB → latents  (Tensor auf dasselbe Device & dtype wie VAE)
                pixel_values = batch.pixel_values.to(device=latents.device if 'latents' in locals() else accelerator.device,
                                                      dtype=torch.float16, non_blocking=True)
                latents = vae.encode(pixel_values).latent_dist.sample() * vae_scale  # (B,4,H/8,W/8)

                # 2) Noise + timesteps
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,),device=latents.device, dtype=torch.long)
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # 3) Text encodings
                input_ids = batch.input_ids.to(accelerator.device, non_blocking=True)
                attention_mask = batch.attention_mask.to(accelerator.device, non_blocking=True)
                enc = text_encoder(input_ids, attention_mask=attention_mask)
                encoder_hidden_states = enc.last_hidden_state.to(device=latents.device, dtype=latents.dtype)

                # 4) Terahertz → Control-Bild via Adapter (B,1,D,H,W) → (B,3,H,W)
                thz_vol = batch.thz.to(device=accelerator.device, dtype=latents.dtype, non_blocking=True)
                control_image = thz_adapter(thz_vol)  # (B,3,H,W)

                # 5) ControlNet Vorwärtslauf
                down_samples, mid_sample = controlnet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=encoder_hidden_states,
                    controlnet_cond=control_image,
                    conditioning_scale=1.0,
                    return_dict=False,
                )

                # 6) UNet Vorwärtslauf mit zusätzlichen Residuals
                model_pred = unet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=encoder_hidden_states,
                    down_block_additional_residuals=down_samples,
                    mid_block_additional_residual=mid_sample,
                ).sample

                # 7) Loss (epsilon prädiktion)
                loss = F.mse_loss(model_pred.float(), noise.float(), reduction="mean")

                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            if accelerator.is_main_process and (global_step + 1) % 10 == 0:
                logger.info(f"epoch {epoch} step {step} | loss {loss.item():.4f}")

            # Checkpointing
            if accelerator.is_main_process and args.checkpointing_steps > 0 and (global_step + 1) % args.checkpointing_steps == 0:
                save_dir = os.path.join(args.output_dir, f"checkpoint-{global_step+1}")
                os.makedirs(save_dir, exist_ok=True)
                accelerator.unwrap_model(controlnet).save_pretrained(os.path.join(save_dir, "controlnet"))
                torch.save(accelerator.unwrap_model(thz_adapter).state_dict(), os.path.join(save_dir, "thz_adapter.pt"))
                logger.info(f"Checkpoint gespeichert: {save_dir}")

            global_step += 1
            if global_step >= max_train_steps:
                break

        if global_step >= max_train_steps:
            break

    # ----------- Final Save ----------- #
    if accelerator.is_main_process:
        accelerator.unwrap_model(controlnet).save_pretrained(os.path.join(args.output_dir, "controlnet"))
        torch.save(accelerator.unwrap_model(thz_adapter).state_dict(), os.path.join(args.output_dir, "thz_adapter.pt"))
        logger.info(f"Fertig. Modelle gespeichert in {args.output_dir}")


if __name__ == "__main__":
    main()
