import argparse, numpy as np, os, os.path as osp, pandas as pd, torch, torch.nn.functional as F, tqdm
from diffusion.datasets import get_target_dataset
from diffusion.models import get_sd_model, get_scheduler_config
from diffusion.utils import LOG_DIR, get_formatstr
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode

from classifiers.adapter_forward import unet_with_adapters_forward
from adapters.feedback_adapter import load_feedback_adapter
from adapters.rgb_adapter import load_rgb_adapter
from classifiers.unet_with_adapter import UNetWithAdapters

device = "cuda" if torch.cuda.is_available() else "cpu"
INTERPOLATIONS = {'bilinear': InterpolationMode.BILINEAR,'bicubic': InterpolationMode.BICUBIC,'lanczos': InterpolationMode.LANCZOS}

def _convert_image_to_rgb(image): return image.convert("RGB")
def get_transform(interpolation=InterpolationMode.BICUBIC, size=512):
    return T.Compose([
        T.Resize(size, interpolation=interpolation),
        T.CenterCrop(size),
        _convert_image_to_rgb,
        T.ToTensor(),
        T.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5]),
    ])

def default_unet_forward(unet, latents, timesteps, text_embeds, extra_cond=None):
    return unet(latents, timesteps, encoder_hidden_states=text_embeds).sample

def eval_prob_adaptive(unet, latent, text_embeds, scheduler, args, latent_size=64, all_noise=None,
                       forward_fn=None, extra_cond=None):
    scheduler_config = get_scheduler_config(args)
    T = scheduler_config['num_train_timesteps']
    max_n_samples = max(args.n_samples)
    scheduler.alphas_cumprod = scheduler.alphas_cumprod.to(latent.device)

    if all_noise is None:
        all_noise = torch.randn((max_n_samples * args.n_trials, 4, latent_size, latent_size), device=latent.device)
    if args.dtype == 'float16':
        all_noise = all_noise.half(); scheduler.alphas_cumprod = scheduler.alphas_cumprod.half()
    else:
        scheduler.alphas_cumprod = scheduler.alphas_cumprod.float()
    if forward_fn is None: forward_fn = default_unet_forward

    data = {}; t_evaluated = set()
    remaining_prmpt_idxs = list(range(len(text_embeds)))
    start = T // max_n_samples // 2
    t_to_eval = list(range(start, T, T // max_n_samples))[:max_n_samples]

    for n_samples, n_to_keep in zip(args.n_samples, args.to_keep):
        ts, noise_idxs, text_embed_idxs = [], [], []
        curr_t_to_eval = t_to_eval[len(t_to_eval)//n_samples//2::len(t_to_eval)//n_samples][:n_samples]
        curr_t_to_eval = [t for t in curr_t_to_eval if t not in t_evaluated]
        for prompt_i in remaining_prmpt_idxs:
            for t_idx, t in enumerate(curr_t_to_eval, start=len(t_evaluated)):
                ts.extend([t]*args.n_trials)
                noise_idxs.extend(list(range(args.n_trials*t_idx, args.n_trials*(t_idx+1))))
                text_embed_idxs.extend([prompt_i]*args.n_trials)
        t_evaluated.update(curr_t_to_eval)

        pred_errors = eval_error(
            unet, scheduler, latent, all_noise, ts, noise_idxs, text_embeds, text_embed_idxs,
            batch_size=args.batch_size, dtype=args.dtype, loss=args.loss,
            forward_fn=forward_fn, extra_cond=extra_cond
        )

        for prompt_i in remaining_prmpt_idxs:
            mask = torch.tensor(text_embed_idxs) == prompt_i
            prompt_ts = torch.tensor(ts)[mask]
            prompt_pred_errors = pred_errors[mask]
            if prompt_i not in data:
                data[prompt_i] = dict(t=prompt_ts, pred_errors=prompt_pred_errors)
            else:
                data[prompt_i]['t'] = torch.cat([data[prompt_i]['t'], prompt_ts])
                data[prompt_i]['pred_errors'] = torch.cat([data[prompt_i]['pred_errors'], prompt_pred_errors])

        errors = [-data[prompt_i]['pred_errors'].mean() for prompt_i in remaining_prmpt_idxs]
        best_idxs = torch.topk(torch.tensor(errors), k=n_to_keep, dim=0).indices.tolist()
        remaining_prmpt_idxs = [remaining_prmpt_idxs[i] for i in best_idxs]

    assert len(remaining_prmpt_idxs) == 1
    pred_idx = remaining_prmpt_idxs[0]
    return pred_idx, data

def eval_error(unet, scheduler, latent, all_noise, ts, noise_idxs,
               text_embeds, text_embed_idxs, batch_size=32, dtype='float32', loss='l2',
               forward_fn=None, extra_cond=None):
    assert len(ts) == len(noise_idxs) == len(text_embed_idxs)
    pred_errors = torch.zeros(len(ts), device='cpu')
    if forward_fn is None: forward_fn = default_unet_forward
    idx = 0
    with torch.inference_mode():
        for _ in tqdm.trange(len(ts) // batch_size + int(len(ts) % batch_size != 0), leave=False):
            batch_ts = torch.tensor(ts[idx: idx + batch_size])
            noise = all_noise[noise_idxs[idx: idx + batch_size]]
            noised_latent = latent * (scheduler.alphas_cumprod[batch_ts] ** 0.5).view(-1,1,1,1).to(device) + \
                            noise * ((1 - scheduler.alphas_cumprod[batch_ts]) ** 0.5).view(-1,1,1,1).to(device)
            t_input = batch_ts.to(device).half() if dtype == 'float16' else batch_ts.to(device)
            text_input = text_embeds[text_embed_idxs[idx: idx + batch_size]]
            noise_pred = forward_fn(unet, noised_latent, t_input, text_input, extra_cond=extra_cond)
            if loss == 'l2':
                error = F.mse_loss(noise, noise_pred, reduction='none').mean(dim=(1,2,3))
            elif loss == 'l1':
                error = F.l1_loss(noise, noise_pred, reduction='none').mean(dim=(1,2,3))
            elif loss == 'huber':
                error = F.huber_loss(noise, noise_pred, reduction='none').mean(dim=(1,2,3))
            else:
                raise NotImplementedError
            pred_errors[idx: idx + len(batch_ts)] = error.detach().cpu()
            idx += len(batch_ts)
    return pred_errors

def main():
    p = argparse.ArgumentParser()
    # dataset args
    p.add_argument('--dataset', type=str, default='pets',
                   choices=['pets','flowers','stl10','mnist','cifar10','food','caltech101','imagenet','objectnet','aircraft'])
    p.add_argument('--split', type=str, default='train', choices=['train','test'])
    # run args
    p.add_argument('--version', type=str, default='2-1')
    p.add_argument('--img_size', type=int, default=512, choices=(256,512))
    p.add_argument('--batch_size', '-b', type=int, default=32)
    p.add_argument('--n_trials', type=int, default=1)
    p.add_argument('--prompt_path', type=str, required=True)
    p.add_argument('--noise_path', type=str, default=None)
    p.add_argument('--subset_path', type=str, default=None)
    p.add_argument('--dtype', type=str, default='float16', choices=('float16','float32'))
    p.add_argument('--interpolation', type=str, default='bicubic')
    p.add_argument('--extra', type=str, default=None)
    p.add_argument('--n_workers', type=int, default=1)
    p.add_argument('--worker_idx', type=int, default=0)
    p.add_argument('--load_stats', action='store_true')
    p.add_argument('--loss', type=str, default='l2', choices=('l1','l2','huber'))
    # adaptive selection
    p.add_argument('--to_keep', nargs='+', type=int, required=True)
    p.add_argument('--n_samples', nargs='+', type=int, required=True)
    # Adapters
    p.add_argument('--feedback_ckpt', type=str, default=None, help='Pfad zu adapter_best.pt (Script 1)')
    p.add_argument('--rgb_dir', type=str, default=None, help='Ordner mit controlnet/ & thz_adapter.pt (Script 2)')
    # THz input (für Adapter)
    p.add_argument('--thz_path', type=str, default=None, help='Ordner mit THz-Volumen, z.B. {index}.pt/.npy')
    args = p.parse_args(); assert len(args.to_keep) == len(args.n_samples)

    # Output-Folder
    name = f"v{args.version}_{args.n_trials}trials_" + '_'.join(map(str,args.to_keep)) + 'keep_' + '_'.join(map(str,args.n_samples)) + 'samples'
    if args.interpolation != 'bicubic': name += f'_{args.interpolation}'
    if args.loss == 'l1': name += '_l1'
    elif args.loss == 'huber': name += '_huber'
    if args.img_size != 512: name += f'_{args.img_size}'
    run_folder = osp.join(LOG_DIR, args.dataset + ('' if args.extra is None else '_' + args.extra), name)
    os.makedirs(run_folder, exist_ok=True); print(f'Run folder: {run_folder}')

    # Dataset & Prompts
    interpolation = INTERPOLATIONS[args.interpolation]
    transform = get_transform(interpolation, args.img_size)
    latent_size = args.img_size // 8
    target_dataset = get_target_dataset(args.dataset, train=args.split == 'train', transform=transform)
    prompts_df = pd.read_csv(args.prompt_path)

    # SD-Modelle
    vae, tokenizer, text_encoder, unet_raw, scheduler = get_sd_model(args)
    vae = vae.to(device); text_encoder = text_encoder.to(device)

    # Adapter laden
    adapters = []
    forward_fn = default_unet_forward
    if args.rgb_dir:   # RGBAdapter = THzAdapter+ControlNet → Residuals im UNet
        rgb_adapter = load_rgb_adapter(args.rgb_dir, device=device, dtype=(torch.float16 if args.dtype=='float16' else torch.float32))
        unet = UNetWithAdapters(unet_raw, adapters=[rgb_adapter]).to(device)
        forward_fn = unet_with_adapters_forward
    else:
        unet = unet_raw.to(device)

    feedback_adapter = None
    if args.feedback_ckpt:  # FeedbackAdapter → RGB vor VAE
        feedback_adapter = load_feedback_adapter(
            args.feedback_ckpt, device=device, dtype=(torch.float16 if args.dtype=='float16' else torch.float32), out_size=args.img_size
        )

    torch.backends.cudnn.benchmark = True

    # Noise
    all_noise = torch.load(args.noise_path).to(device) if args.noise_path else None

    # Text-Embeddings
    text_input = tokenizer(prompts_df.prompt.tolist(), padding="max_length",
                           max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")
    embeds = []
    with torch.inference_mode():
        for i in range(0, len(text_input.input_ids), 100):
            e = text_encoder(text_input.input_ids[i:i+100].to(device))[0]
            embeds.append(e)
    text_embeddings = torch.cat(embeds, dim=0)
    if args.dtype == 'float16': text_embeddings = text_embeddings.half()
    assert len(text_embeddings) == len(prompts_df)

    # THz-Loader (einfach: {index}.pt|.npy). Passe bei Bedarf an.
    def load_thz_for_index(ii: int, base: str):
        if base is None: return None
        for ext in (".pt",".pth",".npy"):
            path = osp.join(base, f"{ii}{ext}")
            if osp.isfile(path):
                if ext == ".npy":
                    arr = np.load(path).astype(np.float32)  # (T,H,W) oder (1,T,H,W)
                    vol = torch.from_numpy(arr)
                else:
                    vol = torch.load(path, map_location="cpu")
                if vol.dim() == 3: vol = vol.unsqueeze(0)     # (1,T,H,W)
                if vol.dim() == 4 and vol.shape[0] != 1:      # (C,T,H,W) -> (1,T,H,W) falls nötig
                    vol = vol[:1]
                return vol.unsqueeze(0)  # (1,1,T,H,W)
        return None

    # Eval-Loop
    formatstr = get_formatstr(len(target_dataset) - 1)
    correct = total = 0
    pbar = tqdm.tqdm(range(len(target_dataset)))
    for i in pbar:
        if total > 0:
            pbar.set_description(f'Acc: {100 * correct / total:.2f}%')
        fname = osp.join(run_folder, formatstr.format(i) + '.pt')
        if os.path.exists(fname):
            if args.load_stats:
                data = torch.load(fname); correct += int(data['pred'] == data['label']); total += 1
            continue

        image, label = target_dataset[i]  # image: [3,H,W] in [-1,1] (nach Transform)
        img_input = image.to(device).unsqueeze(0)
        if args.dtype == 'float16': img_input = img_input.half()

        # THz laden (falls Adapter benötigt)
        thz = load_thz_for_index(i, args.thz_path)

        # 1) Falls FeedbackAdapter aktiv und THz vorhanden → Bild ersetzen
        if (feedback_adapter is not None) and (thz is not None):
            thz = thz.to(device, dtype=(torch.float16 if args.dtype=='float16' else torch.float32))
            with torch.inference_mode():
                img_from_thz = feedback_adapter(thz).to(device)       # [1,3,S,S] in [0,1]
            x_in = (img_from_thz * 2.0 - 1.0).to(dtype=vae.dtype)     # [-1,1]
        else:
            x_in = img_input.to(dtype=vae.dtype)                      # normales Dataset-Bild

        with torch.no_grad():
            x0 = vae.encode(x_in).latent_dist.mean * 0.18215

        # 2) extra_cond für RGBAdapter (falls aktiv; benötigt THz)
        extra_cond = None
        if args.rgb_dir:
            if thz is None:
                raise RuntimeError("RGBAdapter aktiv, aber kein THz-Volumen gefunden. Bitte --thz_path setzen.")
            thz = thz.to(device, dtype=(torch.float16 if args.dtype=='float16' else torch.float32))
            extra_cond = {"thz": thz}

        # 3) Klassische Eval mit Adapter-Forward
        pred_idx, pred_errors = eval_prob_adaptive(
            unet, x0, text_embeddings, scheduler, args,
            latent_size=latent_size, all_noise=all_noise,
            forward_fn=(unet_with_adapters_forward if args.rgb_dir else default_unet_forward),
            extra_cond=extra_cond
        )
        pred = prompts_df.classidx[pred_idx].item() if hasattr(prompts_df.classidx, "iloc") else prompts_df.classidx[pred_idx]
        torch.save(dict(errors=pred_errors, pred=pred, label=label), fname)
        correct += int(pred == label); total += 1

    print(f"Final Acc: {100*correct/max(1,total):.2f}%")

if __name__ == '__main__':
    main()
