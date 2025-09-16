import os, sys
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

#!/usr/bin/env python
import os.path as osp, argparse, pandas as pd, tqdm, torch
from torch.utils.data import Dataset
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode

# ---- Projekt-Helfer ----
from diffusion.datasets import get_target_dataset
from diffusion.models   import get_sd_model

from typing import Optional

# ---- Adapter & Wrapper ----
from adapters.feedback_adapter import load_feedback_adapter
from adapters.rgb_adapter      import load_rgb_adapter   # lädt THz-Adapter + ControlNet
from classifiers.unet_with_adapter         import UNetWithAdapters
from classifiers.adapter_forward           import unet_with_adapters_forward

# ---- Eval (eine Quelle der Wahrheit) ----
from classifiers.eval_prob_adaptiv_with_adapter import default_unet_forward, eval_prob_adaptive

device = "cuda" if torch.cuda.is_available() else "cpu"

# -------------------- kleine Helfer --------------------
def ensure_thz_shape(thz: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    """Bringt THz in [1,1,T,H,W], sonst Fehler."""
    if thz is None: return None
    if thz.dim()==3: thz = thz.unsqueeze(0).unsqueeze(0)
    elif thz.dim()==4:
        if thz.shape[0]!=1: thz = thz[:1]
        thz = thz.unsqueeze(0)
    elif thz.dim()==5:
        if thz.shape[0]!=1: thz = thz[:1]
    else:
        raise ValueError(f"Unerwartete THz-Shape: {tuple(thz.shape)}")
    return thz.contiguous().float()

def split_sample(sample):
    """
    Akzeptiert:
      (img,label)  mit img: [3,H,W]  (z.B. Depthlayer-Dataset)
      (vol,label)  mit vol: [T,H,W] / [1,T,H,W] / [B,1,T,H,W]
      dict mit ('image'|'img'|'pixel_values') oder ('thz'|'volume') auch ok.
    Liefert: (img_rgb_or_None, thz_or_None, label_int)
    """
    img = thz = label = None
    if isinstance(sample, dict):
        img = sample.get('image') or sample.get('img') or sample.get('pixel_values')
        thz = sample.get('thz')   or sample.get('volume')
        label = sample.get('label') or sample.get('y')
    elif isinstance(sample, (tuple, list)):
        x, label = sample[0], sample[1]
        if isinstance(x, torch.Tensor) and x.dim()==3 and x.shape[0]==3:
            img = x
        else:
            thz = x
    else:
        raise ValueError("Unbekanntes Sample-Format")
    if isinstance(thz, torch.Tensor):
        thz = ensure_thz_shape(thz)  # [1,1,T,H,W]
    return img, thz, int(label)

# -------------------- Args --------------------
def parse_args():
    p = argparse.ArgumentParser()
    # Dataset-Quelle (ohne split)
    p.add_argument('--dataset',   type=str, default='thz_for_adapter',
                   help='Alias für get_target_dataset(...). Wenn gesetzt, wird der Alias genutzt.')

    # Prompts & Eval
    p.add_argument('--prompt_path', type=str, required=True)
    p.add_argument('--img_size',    type=int, default=512, choices=(256,512))
    p.add_argument('--dtype',       type=str, default='float16', choices=('float16','float32'))
    p.add_argument('--n_trials',    type=int, default=1)
    p.add_argument('--n_samples',   nargs='+', type=int, default=[8,16,32])
    p.add_argument('--to_keep',     nargs='+', type=int, default=[5,3,1])
    p.add_argument('--loss',        type=str, default='l2', choices=('l1','l2','huber'))
    p.add_argument('--version',     type=str, default='2-1')
    p.add_argument('--noise_path',  type=str, default=None)

    # Adapter-Auswahl
    p.add_argument('--feedback_ckpt', type=str, default=None,
                   help='Pfad zu adapter_best.pt (Feedback-Adapter: THz->RGB vor VAE)')
    p.add_argument('--rgb_dir',       type=str, default=None,
                   help='Ordner mit controlnet/ & thz_adapter.pt (RGB-Adapter: THz-Adapter + ControlNet im UNet)')
    return p.parse_args()

# -------------------- Main --------------------
def main():
    args = parse_args()
    torch.backends.cudnn.benchmark = True

    img_size    = args.img_size
    latent_size = img_size // 8

    # Dataset laden (ohne split)
    if args.dataset is not None:
        # Alias (typisch: Bild-Datasets); Standard-Transform für [-1,1]
        transform = T.Compose([
            T.Resize(img_size, interpolation=InterpolationMode.BICUBIC),
            T.CenterCrop(img_size),
            T.ToTensor(),
            T.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
        ])
        ds = get_target_dataset(args.dataset, train=False, transform=transform)
        dataset_returns_images = True
    else:
        raise ValueError("Bitte EITHER --dataset angeben.")

    # Prompts laden
    dfp = pd.read_csv(args.prompt_path)
    prompts = dfp.prompt.astype(str).tolist()

    # SD / Fusion Classifier
    vae, tokenizer, text_encoder, unet_raw, scheduler = get_sd_model(args)
    vae = vae.to(device); text_encoder = text_encoder.to(device)

    # Text-Embeddings
    text_input = tokenizer(prompts, padding="max_length",
                           max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")
    embeds = []
    with torch.inference_mode():
        for i in range(0, len(text_input.input_ids), 100):
            e = text_encoder(text_input.input_ids[i:i+100].to(device))[0]
            embeds.append(e)
    text_embeddings = torch.cat(embeds, dim=0)
    if args.dtype=='float16': text_embeddings = text_embeddings.half()

    # Adapter setzen
    forward_fn = default_unet_forward
    if args.rgb_dir:
        # WICHTIG: lädt THz-Adapter **und** ControlNet und integriert beides ins UNet
        rgb_adapter = load_rgb_adapter(args.rgb_dir, device=device,
                                       dtype=(torch.float16 if args.dtype=='float16' else torch.float32))
        unet = UNetWithAdapters(unet_raw, adapters=[rgb_adapter]).to(device)
        forward_fn = unet_with_adapters_forward
    else:
        unet = unet_raw.to(device)

    feedback_adapter = None
    if args.feedback_ckpt:
        # Ein einzelner Adapter, der THz→RGB-Bild erzeugt (vor dem VAE)
        feedback_adapter = load_feedback_adapter(args.feedback_ckpt, device=device,
                                                 dtype=(torch.float16 if args.dtype=='float16' else torch.float32),
                                                 out_size=img_size)

    all_noise = torch.load(args.noise_path).to(device) if args.noise_path else None

    # Konsistenz-Checks: was liefern Samples vs. was ist gefordert?
    if feedback_adapter is not None and dataset_returns_images and not args.data_dir:
        # Kein Fehler – du könntest auch Bild-Dataset + THz verlangen; wir checken pro Sample unten.
        pass

    total = correct = 0
    pbar = tqdm.tqdm(range(len(ds)))
    for i in pbar:
        if total>0: pbar.set_description(f'Acc: {100*correct/total:.2f}%')

        sample = ds[i]
        img_rgb, thz, label = split_sample(sample)   # img: [3,H,W] ([-1,1]); thz: [1,1,T,H,W]

        # --- VAE-Input bestimmen ---
        if feedback_adapter is not None:
            if thz is None:
                raise RuntimeError("Feedback-Adapter aktiv, aber Sample enthält kein THz-Volumen.")
            thz_dev = thz.to(device, dtype=(torch.float16 if args.dtype=='float16' else torch.float32))
            with torch.inference_mode():
                img01 = feedback_adapter(thz_dev)            # [1,3,S,S] in [0,1]
            x_in = (img01*2.0 - 1.0).to(dtype=vae.dtype)     # → [-1,1]
        else:
            if img_rgb is None:
                raise RuntimeError("Kein Feedback-Adapter aktiv und Sample liefert kein Bild (3xHxW). "
                                   "Nutz ein Bild-Dataset (z.B. Depthlayer) oder aktiviere --feedback_ckpt.")
            x_in = img_rgb.unsqueeze(0).to(device, dtype=vae.dtype)  # bereits [-1,1] durch Transform

        # --- extra_cond für RGB-Adapter (ControlNet) ---
        if args.rgb_dir:
            if thz is None:
                raise RuntimeError("RGB-Adapter aktiv (ControlNet), aber Sample enthält kein THz-Volumen.")
            extra_cond = {"thz": thz.to(device, dtype=(torch.float16 if args.dtype=='float16' else torch.float32))}
        else:
            extra_cond = None

        # Encode → latent
        with torch.no_grad():
            latent = vae.encode(x_in).latent_dist.mean * 0.18215

        # Klassifizieren via eval_prob_adaptive
        pred_idx, _ = eval_prob_adaptive(
            unet, latent, text_embeddings, scheduler, args,
            latent_size=latent.shape[-1], all_noise=all_noise,
            forward_fn=forward_fn, extra_cond=extra_cond
        )
        pred = int(dfp.classidx[pred_idx])
        correct += int(pred == int(label)); total += 1

    print(f"[SD/Fusion] Acc: {100*correct/max(1,total):.2f}%")

if __name__ == "__main__":
    main()
