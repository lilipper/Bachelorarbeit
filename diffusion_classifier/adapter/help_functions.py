import os
import sys
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

from tqdm.auto import tqdm
import math
import argparse
import numpy as np
import pandas as pd
import scipy.io as sio
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import logging as hf_logging
from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler, StableDiffusionControlNetPipeline, ControlNetModel
from eval_prob_adaptive import eval_prob_adaptive_differentiable
import process_rdf as prdf
import time
from diffusion.datasets import ThzDataset
import csv
from collections import defaultdict
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold
from datetime import datetime
from zoneinfo import ZoneInfo

device = "cuda" if torch.cuda.is_available() else "cpu"


# ========= SD 2.1 base laden (eingefroren) =========

def build_sd2_1_base(dtype="float16", use_xformers=True, train_all=False, version="2-1",):
    MODEL_IDS = {
    '1-1': "CompVis/stable-diffusion-v1-1",
    '1-2': "CompVis/stable-diffusion-v1-2",
    '1-3': "CompVis/stable-diffusion-v1-3",
    '1-4': "CompVis/stable-diffusion-v1-4",
    '1-5': "runwayml/stable-diffusion-v1-5",
    '2-0': "stabilityai/stable-diffusion-2-base",
    '2-1': "stabilityai/stable-diffusion-2-1-base"
    }
    assert version in MODEL_IDS.keys()
    model_id = MODEL_IDS[version]
    scheduler = EulerDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")
    torch_dtype = torch.float32
    if dtype == "float16":
        torch_dtype = torch.float16
    elif dtype == "bfloat16":
        torch_dtype = torch.bfloat16
    pipe = StableDiffusionPipeline.from_pretrained(model_id, scheduler=scheduler, torch_dtype=torch_dtype)
    if use_xformers:
        try:
            pipe.enable_xformers_memory_efficient_attention()
        except Exception:
            pass
    pipe = pipe.to(device)
    vae = pipe.vae.eval()
    unet = pipe.unet.eval()
    controlnet = ControlNetModel.from_unet(
            unet,
            conditioning_channels=3,
            controlnet_conditioning_channel_order="rgb",
            load_weights_from_unet=True
        ).to(device).eval()

    print("[CN] cond_ch:", controlnet.config.conditioning_channels) 
    tokenizer = pipe.tokenizer
    text_encoder = pipe.text_encoder.eval()
    try:
        unet.enable_gradient_checkpointing()
    except Exception:
        pass

    # einfrieren
    if not train_all:
        for p in list(vae.parameters()) + list(unet.parameters()) + list(text_encoder.parameters()):
            p.requires_grad = False
    return vae, unet, tokenizer, text_encoder, scheduler, controlnet

# ========= Training / Evaluation =========

def read_csv_pairs(csv_path):
    rows = []
    with open(csv_path, newline="") as f:
        r = csv.reader(f); header = next(r)
        i_f = header.index("filename")
        i_l = header.index("label")
        for line in r:
            rows.append((line[i_f], line[i_l]))
    return rows  # List[(filename, label)]

def write_csv_pairs(csv_path, rows):
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["filename","label"])
        w.writerows(rows)

def load_class_text_embeds(classes, prompts, tokenizer, text_encoder):
    with torch.no_grad():
        text_in = tokenizer(prompts, padding="max_length",
                            max_length=tokenizer.model_max_length,
                            truncation=True, return_tensors="pt").to(device)
        class_embeds = text_encoder(text_in.input_ids)[0]  # [C,seq,hid]
    return class_embeds


def acc_from_errors(errors_per_class: torch.Tensor, y: int) -> float:
    pred = torch.argmin(errors_per_class).item()
    return float(pred == y)


def run_eval(dataloader, adapter, vae, unet, class_embeds, scheduler, eargs, img_size):
    adapter.eval()
    total, correct = 0, 0
    with torch.no_grad():
        for vol, label, filename in dataloader:
            vol = vol.to(device)
            label = label.to(device).long()

            if vol.dim() == 6:  # [B,1,1,T,H,W] -> [B,1,T,H,W]
                vol = vol.squeeze(1)

            use_amp = (device == "cuda" and eargs.dtype == "float16")
            with torch.amp.autocast('cuda', dtype=torch.float16, enabled=use_amp):
                img  = adapter(vol)                       # [1,3,512,512]
                x_in = (img * 2.0 - 1.0).to(dtype=vae.dtype)
                lat  = vae.encode(x_in).latent_dist.mean * 0.18215

            # pro Sample auswerten (wie DC)
            for b in range(vol.size(0)):
                pred_idx, data, errors_per_class = eval_prob_adaptive_differentiable(
                    unet=unet, latent=lat[b:b+1], text_embeds=class_embeds,
                    scheduler=scheduler, args=eargs, latent_size=img_size // 8, all_noise=None
                )
                correct += int(torch.argmin(errors_per_class).item() == label[b].item())
                total += 1
    adapter.train()
    return correct / max(1, total)


class PromptBank:
    def __init__(self, prompt_csv: str):
        dfp = pd.read_csv(prompt_csv)  # Spalten: prompt, classname, classidx
        assert {"prompt", "classname", "classidx"}.issubset(dfp.columns), \
            "Prompt-CSV muss prompt,classname,classidx enthalten."
        dfp = dfp.sort_values(["classidx", "prompt"]).reset_index(drop=True)
        self.prompt_texts = dfp["prompt"].astype(str).tolist()               # [P]
        self.prompt_to_class = torch.tensor(dfp["classidx"].tolist()).long() # [P]
        self.classnames = (
            dfp.drop_duplicates("classidx").sort_values("classidx")["classname"].tolist()
        )
        self.num_classes = len(self.classnames)

    def to_text_embeds(self, tokenizer, text_encoder, device):
        with torch.no_grad():
            all_embeds = []
            for i in range(0, len(self.prompt_texts), 100):
                batch_texts = self.prompt_texts[i:i+100]
                batch = tokenizer(
                    batch_texts, padding="max_length", max_length=tokenizer.model_max_length,
                    truncation=True, return_tensors="pt"
                ).to(device)
                embeds = text_encoder(batch.input_ids).last_hidden_state
                all_embeds.append(embeds)
        return torch.cat(all_embeds, dim=0)


# ---- Prompt-Fehler → Klassen-Fehler poolen (mean/min) ----
def pool_prompt_errors_to_class_errors(
    errors_per_prompt: torch.Tensor,
    prompt_to_class: torch.Tensor,
    num_classes: int,
    reduce: str = "mean",
) -> torch.Tensor:
    """
    Poolt Prompt-Fehler zu Klassen-Fehlern, dtype-sicher (keine Overflows in fp16).
    """
    dev = errors_per_prompt.device
    dt  = errors_per_prompt.dtype
    finfo = torch.finfo(dt)

    # dtype-sicherer großer Wert (z.B. ~6.5e4 bei fp16, ~1e19 Deckel bei fp32)
    fill_float = min(1e6, float(finfo.max * 0.5))
    fill = torch.tensor(fill_float, device=dev, dtype=dt)

    # init mit großem Wert
    class_errors = torch.full((num_classes,), fill, device=dev, dtype=dt)

    for c in range(num_classes):
        mask = (prompt_to_class == c)
        if torch.any(mask):
            vals = errors_per_prompt[mask]
            # numerisch robust zusammenfegen
            vals = torch.nan_to_num(vals, nan=float(fill), posinf=float(fill), neginf=float(fill))
            ce = vals.mean() if reduce == "mean" else vals.min()
            class_errors[c] = ce

    class_errors = torch.nan_to_num(class_errors, nan=float(fill), posinf=float(fill), neginf=float(fill))
    return class_errors


# ---- Accuracy aus Klassen-Fehlern ----
def acc_from_class_errors(class_errors: torch.Tensor, y: int) -> float:
    pred = torch.argmin(class_errors).item()
    return float(pred == y)

def pool_prompt_errors_to_class_errors_batch(errors_per_prompt, prompt_to_class, num_classes, reduce="mean"):
    """ Vektorisierte Version, die einen Batch von Fehler-Tensoren verarbeitet. """
    B, P = errors_per_prompt.shape
    device = errors_per_prompt.device
    
    one_hot_matrix = F.one_hot(prompt_to_class, num_classes=num_classes).float().to(device)
    
    errors_grouped = errors_per_prompt.unsqueeze(2) * one_hot_matrix.unsqueeze(0)
    
    sum_errors = errors_grouped.sum(dim=1)
    prompts_per_class = one_hot_matrix.sum(dim=0)
    
    if reduce == "mean":
        class_errors = sum_errors / (prompts_per_class.unsqueeze(0) + 1e-8)
    elif reduce == "min":
        errors_grouped[errors_grouped == 0] = float('inf')
        class_errors = errors_grouped.min(dim=1).values
    else:
        raise ValueError("reduce must be 'mean' or 'min'")
        
    return class_errors