import os
import argparse
import csv
import numpy as np
from datetime import datetime
from zoneinfo import ZoneInfo

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR, CosineAnnealingLR
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from transformers import logging as hf_logging
import diffusers
from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler, ControlNetModel

from diffusion.datasets import ThzDataset
from eval_prob_adaptive import eval_prob_adaptive_differentiable
from adapter.help_functions import PromptBank, pool_prompt_errors_to_class_errors_batch

from pipeline_classifier_with_adapter.eval_the_pipeline_results import evaluate_predictions

hf_logging.set_verbosity_error()
device = "cuda" if torch.cuda.is_available() else "cpu"


# --------------------- Utilities ---------------------
def set_seed(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class ControlNetWithDropout(nn.Module):
    def __init__(self, base_controlnet: nn.Module, p: float = 0.1, spatial: bool = True):
        super().__init__()
        self.base = base_controlnet
        self.dropout = nn.Dropout2d(p) if spatial else nn.Dropout(p)

    def __getattr__(self, name):
        if name in {"base", "dropout"}:
            return super().__getattr__(name)
        try:
            return getattr(self.base, name)
        except AttributeError:
            cfg = getattr(self.base, "config", None)
            if cfg is not None and hasattr(cfg, name):
                return getattr(cfg, name)
            raise

    def forward(self, *args, **kwargs):
        out = self.base(*args, **kwargs)

        if isinstance(out, tuple):
            if len(out) != 2:
                raise RuntimeError(f"Unexpected ControlNet output tuple length {len(out)}")
            down, mid = out
            if down is not None:
                down = [self.dropout(x) for x in down]
            if mid is not None:
                mid = self.dropout(mid)
            return (down, mid)

        down = getattr(out, "down_block_res_samples", None)
        mid  = getattr(out, "mid_block_res_sample", None)
        if down is not None:
            down = [self.dropout(x) for x in down]
        if mid is not None:
            mid = self.dropout(mid)

        out_cls = type(out)
        try:
            return out_cls(down_block_res_samples=down, mid_block_res_sample=mid)
        except TypeError:
            if hasattr(out, "__dict__"):
                d = vars(out).copy()
                if down is not None: d["down_block_res_samples"] = down
                if mid  is not None: d["mid_block_res_sample"]   = mid
                return out_cls(**d)
            return out



def build_sd2_1_base(dtype="float16", use_xformers=True, train_all=False, version="2-1", dropout_p=0.1, controlnet_spatial_dropout=True):
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
    controlnet_base = ControlNetModel.from_unet(
            unet,
            conditioning_channels=3,
            controlnet_conditioning_channel_order="rgb",
            load_weights_from_unet=True
        ).to(device).eval()

    print("[CN] cond_ch:", controlnet_base.config.conditioning_channels) 
    tokenizer = pipe.tokenizer
    text_encoder = pipe.text_encoder.eval()
    try:
        unet.enable_gradient_checkpointing()
    except Exception:
        pass

    if not train_all:
        for p in list(vae.parameters()) + list(unet.parameters()) + list(text_encoder.parameters()):
            p.requires_grad = False

    controlnet = ControlNetWithDropout(
        controlnet_base,
        p=dropout_p,
        spatial=controlnet_spatial_dropout
    )
    return vae, unet, tokenizer, text_encoder, scheduler, controlnet

class LatentMultiChannelAdapter(nn.Module):
    def __init__(self, k_t=5, use_attn_pool=True, reduce_T_stride=1, hidden_channels=8, dropout_p=0.1):
        super().__init__()
        self.use_attn_pool = use_attn_pool
        self.reduce_T_stride = reduce_T_stride
        # self.dropout = nn.Dropout3d(p=dropout_p)
        self.dw_t = nn.Conv3d(4, 4, kernel_size=(k_t,1,1), padding=(k_t//2,0,0), groups=4, bias=False)
        self.pw_mix = nn.Conv3d(4, 4, kernel_size=1)
        self.norm = nn.GroupNorm(4, 4)
        self.proj_t_down = None
        if reduce_T_stride and reduce_T_stride > 1:
            self.proj_t_down = nn.AvgPool3d(kernel_size=(reduce_T_stride,1,1), stride=(reduce_T_stride,1,1))
        if use_attn_pool:
            self.attn_mlp = nn.Sequential(
                nn.Conv3d(4, hidden_channels, 1, bias=True),
                nn.SiLU(),
                nn.Conv3d(hidden_channels, 1, 1, bias=True)
            )
        else:
            self.attn_mlp = None
        self.out_mix = nn.Conv2d(4, 4, kernel_size=1, bias=True)
        self.dropout = nn.Dropout2d(p=dropout_p)
        

    def forward(self, latents):
        x = latents.permute(0, 2, 1, 3, 4).contiguous()
        residual = x
        # x = self.dropout(x)
        x = self.dw_t(x)
        x = self.pw_mix(x)
        x = self.norm(F.gelu(x) + residual)
        if self.proj_t_down is not None:
            x = self.proj_t_down(x)
        if self.use_attn_pool:
            logits = self.attn_mlp(x)
            weights = torch.softmax(logits, dim=2)
            x = (weights * x).sum(dim=2)
        else:
            x = F.adaptive_avg_pool3d(x, output_size=(1, None, None)).squeeze(2)
        x = self.out_mix(x)
        x = self.dropout(x)
        return x


def normalize_amp_and_scaler(dtype_str: str):
    torch_dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[dtype_str]
    use_amp = (device == "cuda") and (dtype_str in ("float16", "bfloat16"))
    scaler = torch.amp.GradScaler('cuda', enabled=(use_amp and dtype_str == "float16"))
    return torch_dtype, use_amp, scaler

@torch.no_grad()
def encode_all_frames_with_vae(frames_vae, vae, scaling=0.18215):
    posterior = vae.encode(frames_vae).latent_dist.mean.to(torch.float32)
    latents = posterior * scaling
    return latents


# ----------------- Train / Validate -----------------
def train_one_epoch(
    loader, adapter, unet, controlnet, vae, prompt_embeds, prompt_to_class, num_classes,
    sd_scheduler, eargs, img_size, logit_scale, torch_dtype, use_amp, opt, scheduler, scaler,
    reduce="mean"
):
    if eargs.learn_adapter:
        adapter.train()
    else:
        adapter.eval()
    controlnet.train()
    unet.eval() 

    running_loss, running_acc, seen = 0.0, 0, 0
    latent_size = img_size // 8
    out_hw = (img_size, img_size)
    criterion = nn.CrossEntropyLoss()

    print("[train_one_epoch] Starting training iteration...")
    for step, (vol, label, _) in enumerate(loader, start=1):
        vol = vol.to(device)
        if vol.dim() == 6:
            vol = vol.squeeze(1)
        label = label.to(device).long()

        opt.zero_grad(set_to_none=True)

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
            lat = adapter(lat_stack)
            try:
                cn_in_ch = controlnet.controlnet_cond_embedding.conv_in.in_channels
            except Exception:
                cn_in_ch = getattr(controlnet, "in_channels", 4)

            if cn_in_ch == 3:
                with torch.no_grad():
                    control_cond_img = vae.decode(lat / 0.18215).sample
            else:
                control_cond_img = None
        
            B = lat.size(0)
            errors_list = []
            for i in range(B):
                cond_to_pass = control_cond_img[i:i+1] if control_cond_img is not None else None
                pred_idx , _, class_errors_i = eval_prob_adaptive_differentiable(
                    unet=unet,
                    latent=lat[i:i+1],
                    text_embeds=prompt_embeds,
                    scheduler=sd_scheduler,
                    args=eargs,
                    latent_size=latent_size,
                    controlnet=controlnet,
                    all_noise=None,
                    controlnet_cond=cond_to_pass
                )
                errors_list.append(class_errors_i)

            errors = torch.stack(errors_list, dim=0).to(label.dtype if label.is_floating_point() else torch.float32)
            logits = -errors
            loss = criterion(logits, label)

        if scaler.is_enabled():
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(controlnet.parameters(), 1.0)
            if eargs.learn_adapter:
                torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1.0)
            if step % 30 == 1:
                for name, p in list(controlnet.named_parameters())[:3]:
                    g = None if p.grad is None else p.grad
                    msg = "None" if g is None else f"mean={g.abs().mean().item():.4e} max={g.abs().max().item():.4e}"
                    print(f"[DEBUG] grad ControlNet {name}: {msg}")

                if eargs.learn_adapter:
                    for name, p in list(adapter.named_parameters())[:3]:
                        g = None if p.grad is None else p.grad
                        msg = "None" if g is None else f"mean={g.abs().mean().item():.4e} max={g.abs().max().item():.4e}"
                        print(f"[DEBUG] grad Adapter {name}: {msg}")

            scaler.step(opt)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(controlnet.parameters(), 1.0)
            if eargs.learn_adapter:
                torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1.0)

            for name, p in list(controlnet.named_parameters())[:3]:
                g = None if p.grad is None else p.grad
                msg = "None" if g is None else f"mean={g.abs().mean().item():.4e} max={g.abs().max().item():.4e}"
                print(f"[DEBUG] grad ControlNet {name}: {msg}")

            if eargs.learn_adapter:
                for name, p in list(adapter.named_parameters())[:3]:
                    g = None if p.grad is None else p.grad
                    msg = "None" if g is None else f"mean={g.abs().mean().item():.4e} max={g.abs().max().item():.4e}"
                    print(f"[DEBUG] grad Adapter {name}: {msg}")
            snap_before, _ = update_ratio(controlnet, None)
            opt.step()
            _, ratio = update_ratio(controlnet, snap_before)
            print(f"[STAT] controlnet avg update/param = {ratio:.3e}")

        if isinstance(scheduler, (torch.optim.lr_scheduler.OneCycleLR, torch.optim.lr_scheduler.CyclicLR)):
            scheduler.step()

        with torch.no_grad():
            running_loss += loss.item() * vol.size(0)
            preds = torch.argmax(logits, dim=1)
            running_acc += (preds == label).sum().item()
            seen += vol.size(0)

        if step % 10 == 1:
            print(f"[train_one_epoch] step={step}  "
                  f"avg_loss={running_loss/max(1,seen):.4f}  avg_acc={running_acc/max(1,seen):.4f}")
            with torch.no_grad():
                for any_name, any_param in controlnet.named_parameters():
                    print(f"[DEBUG] sample weight after step: {any_param.view(-1)[0].item():.6f}")
                    break
    if isinstance(scheduler, (torch.optim.lr_scheduler.CosineAnnealingLR,
                            torch.optim.lr_scheduler.StepLR,
                            torch.optim.lr_scheduler.ReduceLROnPlateau)):
        scheduler.step()
    epoch_loss = running_loss / max(1, seen)
    epoch_acc = running_acc / max(1, seen)
    print(f"[train_one_epoch] Done. epoch_loss={epoch_loss:.4f}  epoch_acc={epoch_acc:.4f}")
    return epoch_loss, epoch_acc


@torch.no_grad()
def validate(
    loader, adapter, unet, controlnet, vae, prompt_embeds, prompt_to_class, num_classes,
    sd_scheduler, eargs, img_size, torch_dtype, use_amp, save_dir, reduce="mean", save_preds=False
):
    """Validation loop mirroring the training inference flow."""
    adapter.eval()
    controlnet.eval()
    unet.eval()

    def get_formatstr(n):
            digits = 0
            while n > 0:
                digits += 1
                n //= 10
            return f"{{:0{digits}d}}"
    
    latent_size = img_size // 8
    out_hw = (img_size, img_size)

    total, correct = 0, 0
    print("[validate] Starting validation iteration...")
    for step, (vol, label, _) in enumerate(loader, start=1):
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
            lat = adapter(lat_stack)
            try:
                cn_in_ch = controlnet.controlnet_cond_embedding.conv_in.in_channels
            except Exception:
                cn_in_ch = getattr(controlnet, "in_channels", 4)

            if cn_in_ch == 3:
                with torch.no_grad():
                    control_cond_img = vae.decode(lat / 0.18215).sample
            else:
                control_cond_img = None

            B = lat.size(0)
            errors_list = []
            for i in range(B):
                cond_to_pass = control_cond_img[i:i+1] if control_cond_img is not None else None
                pred_idx, _, class_errors_i = eval_prob_adaptive_differentiable(
                    unet=unet,
                    latent=lat[i:i+1],
                    text_embeds=prompt_embeds,
                    scheduler=sd_scheduler,
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
            if save_preds:
                formatstr = get_formatstr(len(loader))
                torch.save(
                        dict(preds=preds.cpu(), label=label.cpu()),
                        os.path.join(save_dir,  formatstr.format(i) + '.pt')
                    )
        correct += (preds == label).sum().item()
        total += label.numel()

        if step % 20 == 1:
            print(f"[validate] step={step}  running_acc={correct/max(1,total):.4f}")

    val_acc = correct / max(1, total)
    print(f"[validate] Done. val_acc={val_acc:.4f}")
    return val_acc

def update_ratio(mod, last_params=None):
    ratios = []
    cur = {n: p.detach().clone() for n,p in mod.named_parameters() if p.requires_grad}
    if last_params is not None:
        for n,p in mod.named_parameters():
            if p.requires_grad:
                upd = (cur[n] - last_params[n])
                num = upd.norm().item()
                den = (cur[n].norm().item() + 1e-12)
                ratios.append(num / den)
    return cur, (sum(ratios)/len(ratios) if ratios else None)

# ----------------------- Main -----------------------
def main():
    ap = argparse.ArgumentParser(description="Diffusion Classifier (SD + ControlNet) with THz, RSKF, and Final Eval")

    # Data
    ap.add_argument("--data_train", type=str, required=True)
    ap.add_argument("--train_csv", type=str, required=True)
    ap.add_argument("--data_root_eval", type=str, default=None,
                    help="Optional root override for validation/test datasets (default: use data_train).")

    # Prompts / SD / Diffusion Classifier
    ap.add_argument("--prompts_csv", type=str, required=True, help="CSV with columns: prompt,classname,classidx")
    ap.add_argument("--version", type=str, default="2-1", choices=("2-1","2-0","1-1","1-2","1-3","1-4","1-5"))
    ap.add_argument("--dtype", type=str, default="float32", choices=("float16","float32","bfloat16"))
    ap.add_argument("--img_size", type=int, default=256, choices=(256,512))
    ap.add_argument("--num_train_timesteps", type=int, default=1000)
    ap.add_argument("--n_trials", type=int, default=1)
    ap.add_argument("--n_samples", nargs="+", type=int, required=True)
    ap.add_argument("--to_keep",   nargs="+", type=int, required=True)
    ap.add_argument("--loss", type=str, default="l2", choices=("l1","l2","huber"))
    ap.add_argument("--logit_scale", type=float, default=60.0)
    ap.add_argument("--use_xformers", action="store_true")
    
    ap.add_argument("--acc_threshold", type=float, default=0.98)
    ap.add_argument("--loss_threshold", type=float, default=0.5)

    # Control / Adapter
    ap.add_argument("--cond_scale", type=float, default=2.0, help="conditioning_scale for ControlNet")
    ap.add_argument("--learn_adapter", action="store_true")
    ap.add_argument("--lr_adapter", type=float, default=1e-2)
    ap.add_argument("--wd_adapter", type=float, default=1e-2)
    ap.add_argument("--lr_controlnet", type=float, default=5e-3)
    ap.add_argument("--wd_controlnet", type=float, default=3e-3)
    ap.add_argument("--dropout_p_controlnet", type=float, default=0.1,
                help="Dropout auf ControlNet-Residuals")
    ap.add_argument("--controlnet_spatial_dropout", action="store_true",
                help="Nutze Dropout2d (spatial) auf HxW")

    # Training
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--dropout_p", type=float, default=0.1, help="Dropout probability for adapter dropout layer.")

    # CV
    ap.add_argument("--cv_seed", type=int, default=42)

    # Final evaluation
    ap.add_argument("--final_eval", action="store_true",
                    help="After CV: load global best and evaluate on a fixed set.")
    ap.add_argument("--data_test", type=str, default=None, help="Test root dir.")
    ap.add_argument("--test_csv", type=str, default=None, help="CSV (path,label) for test. If empty, fall back to --val_csv.")
    ap.add_argument("--val_csv", type=str, default=None, help="Fallback CSV for final eval if --test_csv not set.")

    # IO
    ap.add_argument("--save_dir", type=str, default="./runs/checkpoints_dc_adapter_multichannel")

    args = ap.parse_args()
    # Print config
    print("========== CONFIG ==========")
    for k, v in vars(args).items():
        print(f"{k}: {v}")
    print("============================")

    acc_threshold = args.acc_threshold
    loss_threshold = args.loss_threshold

    set_seed(args.cv_seed)

    torch.backends.cudnn.benchmark = False

    # Save root
    stamp = datetime.now(ZoneInfo("Europe/Berlin")).strftime("%y%m%d_%H%M")
    save_dir = os.path.join(args.save_dir, f"sd{args.version}_img{args.img_size}_{args.dtype}_{stamp}")
    os.makedirs(save_dir, exist_ok=True)
    print(f"[IO] Save directory: {save_dir}")

    # --- EArgs for eval_prob_adaptive_differentiable ---
    class EArgs: pass
    eargs = EArgs()
    eargs.n_samples = args.n_samples
    eargs.to_keep = args.to_keep
    eargs.n_trials = args.n_trials
    eargs.batch_size = args.batch_size
    eargs.dtype = args.dtype
    eargs.loss = args.loss
    eargs.num_train_timesteps = args.num_train_timesteps
    eargs.version = args.version
    eargs.learn_adapter = args.learn_adapter
    eargs.lr_adapter = args.lr_adapter
    eargs.wd_adapter = args.wd_adapter
    eargs.cond_scale = args.cond_scale

    path_ckpt = os.path.join(save_dir, "best_model")
    os.makedirs(path_ckpt, exist_ok=True)
    print(f"[IO] Fold directory: {path_ckpt}")

    # Datasets/loaders
    train_ds = ThzDataset(args.data_train, args.train_csv, is_train=True)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                            num_workers=args.num_workers, pin_memory=True, drop_last=True)
    print(f"[Data] Train batches: ~{len(train_loader)}")

        # AMP / dtype
    torch_dtype, use_amp, scaler = normalize_amp_and_scaler(args.dtype)
    print(f"[AMP] device={device}  dtype={args.dtype}  use_amp={use_amp}")

    # --- Load SD backbone (frozen VAE/UNet/TextEncoder, scheduler) ---
    print("[SD] Building Stable Diffusion base...")
    vae, unet, tokenizer, text_encoder, sd_scheduler, controlnet = build_sd2_1_base(
        dtype=args.dtype, use_xformers=args.use_xformers, train_all=False, version=args.version,
        dropout_p=args.dropout_p_controlnet, controlnet_spatial_dropout=args.controlnet_spatial_dropout
    )
    print("diffusers version:", diffusers.__version__)
    print("controlnet class:", controlnet.__class__.__name__)
    print("controlnet config in/out:", getattr(controlnet, "in_channels", None), getattr(controlnet, "out_channels", None))
    print(controlnet.config)
    unet = unet.to(device).eval()
    vae = vae.to(device).eval()
    controlnet = controlnet.to(device).eval()
    sd_scheduler.set_timesteps(args.num_train_timesteps)
    print("[SD] Done. UNet/VAEs are frozen.")

    # --- Prompt bank & text embeddings ---
    print(f"[Prompts] Loading prompts from: {args.prompts_csv}")
    pb = PromptBank(args.prompts_csv)
    prompt_embeds = pb.to_text_embeds(tokenizer, text_encoder, device)
    prompt_to_class = pb.prompt_to_class.to(device)
    num_classes = pb.num_classes
    print(f"[Prompts] Loaded {len(pb.prompt_texts)} prompts over {num_classes} classes.")

    try:
        controlnet.enable_gradient_checkpointing()
        print("[ControlNet] Enabled gradient checkpointing.")
    except Exception:
        print("[ControlNet] Gradient checkpointing not available.")
    print("controlnet config in/out:",
        getattr(controlnet, "in_channels", None),
        getattr(controlnet, "out_channels", None))
    print(controlnet.config)

    adapter = LatentMultiChannelAdapter(k_t=5, use_attn_pool=True, dropout_p=args.dropout_p).to(device)
    param_groups = []
    if args.learn_adapter:
        param_groups.append({"params": adapter.parameters(), "lr": args.lr_adapter, "weight_decay": args.wd_adapter})
        print("[OPT] Adapter parameters will be trained.")
    else:
        for p in adapter.parameters():
            p.requires_grad_(False)
        adapter.eval()
        print("[OPT] Adapter is frozen.")

    param_groups.append({"params": controlnet.base.parameters(), "lr": args.lr_controlnet, "weight_decay": args.wd_controlnet})
    print("[OPT] ControlNet parameters will be trained.")

    swap_number = int(args.epochs *0.6)

    optimizer1 = optim.AdamW(param_groups, lr=args.lr_adapter, weight_decay=args.wd_adapter)
    scheduler1 = OneCycleLR(
        optimizer1,
        max_lr=args.lr_adapter,
        epochs=int(swap_number),
        steps_per_epoch=len(train_loader),
        pct_start=0.3
    )

    optimizer2 = optim.SGD(param_groups, lr=args.lr_adapter, momentum=0.9, weight_decay=args.wd_adapter)
    scheduler2 = CosineAnnealingLR(optimizer2, T_max=args.epochs - int(swap_number), eta_min=1e-5)

    
    def combined_optimizer(step):
        if step < swap_number:
            return optimizer1, scheduler1
        else:
            return optimizer2, scheduler2

    print("[DEBUG] Trainable parameter groups:")
    for name, p in list(adapter.named_parameters())[:3]:
        print(" Adapter:", name, p.requires_grad)
    for name, p in list(controlnet.named_parameters())[:3]:
        print(" ControlNet:", name, p.requires_grad)
    for name, p in list(unet.named_parameters())[:3]:
        print(" UNet:", name, p.requires_grad)

    # Fold training
    best_val = -1.0
    model_path = os.path.join(path_ckpt, "best_model.pt")
    for epoch in range(1, args.epochs + 1):
        print(f"\n[Epoch] {epoch:02d}/{args.epochs}")
        opt, scheduler = combined_optimizer(epoch)
        tr_loss, tr_acc = train_one_epoch(
            train_loader, adapter, unet, controlnet, vae,
            prompt_embeds, prompt_to_class, num_classes,
            sd_scheduler, eargs, args.img_size, args.logit_scale,
            torch_dtype, use_amp, opt, scheduler, scaler, reduce="mean"
        )
        print(f"[Epoch] Result: train_loss={tr_loss:.4f}  train_acc={tr_acc:.4f}")
        
        if tr_acc >= best_val:
            best_val = tr_acc
            torch.save({
                "epoch": epoch,
                "best_val_acc": best_val,
                "version": args.version,
                "img_size": args.img_size,
                "dtype": args.dtype,
                "cond_scale": args.cond_scale,
                "controlnet_state_dict": controlnet.base.state_dict(),
                "adapter_state_dict": adapter.state_dict(),
                "prompts_csv": args.prompts_csv,
                "n_samples": args.n_samples,
                "to_keep": args.to_keep,
                "num_train_timesteps": args.num_train_timesteps,
                "loss": args.loss,
                "logit_scale": args.logit_scale,
            }, model_path)
            print(f"[Checkpoint] Saved best fold checkpoint: {model_path}  (val_acc={best_val:.4f})")
        if tr_acc >= acc_threshold and tr_loss <= loss_threshold:
            print(f"[Epoch {epoch}] Reached thresholds: acc {tr_acc:.4f} >= {acc_threshold}, loss {tr_loss:.4f} <= {loss_threshold}. Stopping training.")
            break

    # ------------- FINAL EVAL (optional) -------------
    if args.final_eval:
        test_csv = args.test_csv if args.test_csv is not None else args.val_csv
        assert args.data_test and test_csv, "--final_eval requires --data_test and (--test_csv or --val_csv)"
        print("[FINAL] Starting final evaluation...")
        print(f"[FINAL] Loading global best checkpoint: {model_path}")
        ckpt = torch.load(model_path, map_location="cpu")
        print(f"[FINAL] Global best val_acc (from CV): {ckpt.get('best_val_acc','N/A')}")

        print("[FINAL] Rebuilding SD base...")
        vae_f, unet_f, tokenizer_f, text_encoder_f, sd_scheduler_f, controlnet_f = build_sd2_1_base(
            dtype=ckpt["dtype"], use_xformers=False, train_all=False, version=ckpt["version"]
        )
        unet_f.eval()

        unet_f = unet_f.to(device).eval()
        vae_f = vae_f.to(device).eval()
        controlnet_f = controlnet_f.to(device).eval()

        sd_scheduler_f.set_timesteps(ckpt["num_train_timesteps"])

        adapter_f = LatentMultiChannelAdapter(k_t=5, use_attn_pool=True).to(device)
        adapter_f.load_state_dict(ckpt["adapter_state_dict"], strict=False)
        adapter_f.eval()

        controlnet_f.base.load_state_dict(ckpt["controlnet_state_dict"], strict=False)
        controlnet_f.eval()

        # Load prompts again to get embeddings (they are not stored in the ckpt)
        print(f"[FINAL] Reloading prompts from: {ckpt['prompts_csv']}")
        pb_f = PromptBank(ckpt["prompts_csv"])
        prompt_embeds_f = pb_f.to_text_embeds(tokenizer_f, text_encoder_f, device)
        prompt_to_class_f = pb_f.prompt_to_class.to(device)
        num_classes_f = pb_f.num_classes

        # Build test loader
        final_root = args.data_test
        final_ds = ThzDataset(final_root, test_csv, is_train=False)
        final_loader = DataLoader(final_ds, batch_size=1, shuffle=False,
                                  num_workers=args.num_workers, pin_memory=True)
        print(f"[FINAL] Test set: {len(final_loader)} samples | CSV={test_csv} | ROOT={final_root}")

        # AMP settings for final eval
        use_amp_final = (device == "cuda") and (ckpt["dtype"] in ("float16", "bfloat16"))
        final_torch_dtype = torch.float16 if ckpt["dtype"] == "float16" else (
            torch.bfloat16 if ckpt["dtype"] == "bfloat16" else torch.float32
        )
        print(f"[FINAL] AMP enabled={use_amp_final} dtype={ckpt['dtype']}  img_size={ckpt['img_size']}")

        # Rebuild eargs
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
        eargs_f.learn_adapter = False
        eargs_f.cond_scale = ckpt["cond_scale"]

        result_dir = os.path.join(save_dir, "final_results")
        os.makedirs(result_dir, exist_ok=True)
        print(f"[FINAL] Result directory: {result_dir}")
        # Run validate() on the fixed test set
        final_acc = validate(
            final_loader, adapter_f, unet_f, controlnet_f, vae_f,
            prompt_embeds_f, prompt_to_class_f, num_classes_f,
            sd_scheduler_f, eargs_f, ckpt["img_size"],
            torch_dtype=final_torch_dtype, use_amp=use_amp_final, reduce="mean", save_dir=result_dir, save_preds=True
        )
        print(f"[FINAL] accuracy on fixed set ({test_csv} @ {final_root}): {final_acc:.4f}")
        eval_dir=os.path.join(save_dir, "eval_results")
        os.makedirs(eval_dir, exist_ok=True)
        print(f"[FINAL] Evaluation directory: {eval_dir}")
        evaluate_predictions(result_dir, eval_dir, test_csv)

if __name__ == "__main__":
    main()
