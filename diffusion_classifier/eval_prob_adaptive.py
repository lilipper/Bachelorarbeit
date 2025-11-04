import argparse
import numpy as np
import os
import os.path as osp
import pandas as pd
import torch
import torch.nn.functional as F
import tqdm
from diffusion.datasets import get_target_dataset
from diffusion.models import get_sd_model, get_scheduler_config
from diffusion.utils import LOG_DIR, get_formatstr
import torchvision.transforms as torch_transforms
from torchvision.transforms.functional import InterpolationMode

device = "cuda" if torch.cuda.is_available() else "cpu"

INTERPOLATIONS = {
    'bilinear': InterpolationMode.BILINEAR,
    'bicubic': InterpolationMode.BICUBIC,
    'lanczos': InterpolationMode.LANCZOS,
}


def _convert_image_to_rgb(image):
    return image.convert("RGB")


def get_transform(interpolation=InterpolationMode.BICUBIC, size=512):
    transform = torch_transforms.Compose([
        torch_transforms.Resize(size, interpolation=interpolation),
        torch_transforms.CenterCrop(size),
        _convert_image_to_rgb,
        torch_transforms.ToTensor(),
        torch_transforms.Normalize([0.5], [0.5])
    ])
    return transform


def center_crop_resize(img, interpolation=InterpolationMode.BILINEAR):
    transform = get_transform(interpolation=interpolation)
    return transform(img)


def eval_prob_adaptive(unet, latent, text_embeds, scheduler, args, latent_size=64, all_noise=None):
    scheduler_config = get_scheduler_config(args)
    T = scheduler_config['num_train_timesteps']
    max_n_samples = max(args.n_samples)

    if all_noise is None:
        all_noise = torch.randn((max_n_samples * args.n_trials, 4, latent_size, latent_size), device=latent.device)
    if args.dtype == 'float16':
        all_noise = all_noise.half()
        scheduler.alphas_cumprod = scheduler.alphas_cumprod.half()
    elif args.dtype == 'bfloat16':
        all_noise = all_noise.bfloat16()
        scheduler.alphas_cumprod = scheduler.alphas_cumprod.bfloat16()

    data = dict()
    t_evaluated = set()
    remaining_prmpt_idxs = list(range(len(text_embeds)))
    start = T // max_n_samples // 2
    t_to_eval = list(range(start, T, T // max_n_samples))[:max_n_samples]

    for n_samples, n_to_keep in zip(args.n_samples, args.to_keep):
        ts = []
        noise_idxs = []
        text_embed_idxs = []
        curr_t_to_eval = t_to_eval[len(t_to_eval) // n_samples // 2::len(t_to_eval) // n_samples][:n_samples]
        curr_t_to_eval = [t for t in curr_t_to_eval if t not in t_evaluated]
        for prompt_i in remaining_prmpt_idxs:
            for t_idx, t in enumerate(curr_t_to_eval, start=len(t_evaluated)):
                ts.extend([t] * args.n_trials)
                noise_idxs.extend(list(range(args.n_trials * t_idx, args.n_trials * (t_idx + 1))))
                text_embed_idxs.extend([prompt_i] * args.n_trials)
        t_evaluated.update(curr_t_to_eval)
        pred_errors = eval_error(unet, scheduler, latent, all_noise, ts, noise_idxs,
                                 text_embeds, text_embed_idxs, args.batch_size, args.dtype, args.loss)
        # match up computed errors to the data
        for prompt_i in remaining_prmpt_idxs:
            mask = torch.tensor(text_embed_idxs) == prompt_i
            prompt_ts = torch.tensor(ts)[mask]
            prompt_pred_errors = pred_errors[mask]
            if prompt_i not in data:
                data[prompt_i] = dict(t=prompt_ts, pred_errors=prompt_pred_errors)
            else:
                data[prompt_i]['t'] = torch.cat([data[prompt_i]['t'], prompt_ts])
                data[prompt_i]['pred_errors'] = torch.cat([data[prompt_i]['pred_errors'], prompt_pred_errors])

        # compute the next remaining idxs
        errors = [-data[prompt_i]['pred_errors'].mean() for prompt_i in remaining_prmpt_idxs]
        best_idxs = torch.topk(torch.tensor(errors), k=n_to_keep, dim=0).indices.tolist()
        remaining_prmpt_idxs = [remaining_prmpt_idxs[i] for i in best_idxs]

    # organize the output
    assert len(remaining_prmpt_idxs) == 1
    pred_idx = remaining_prmpt_idxs[0]

    return pred_idx, data


def eval_error(unet, scheduler, latent, all_noise, ts, noise_idxs,
               text_embeds, text_embed_idxs, batch_size=32, dtype='float32', loss='l2'):
    assert len(ts) == len(noise_idxs) == len(text_embed_idxs)
    pred_errors = torch.zeros(len(ts), device='cpu')
    idx = 0
    with torch.inference_mode():
        for _ in tqdm.trange(len(ts) // batch_size + int(len(ts) % batch_size != 0), leave=False):
            batch_ts = torch.tensor(ts[idx: idx + batch_size])
            noise = all_noise[noise_idxs[idx: idx + batch_size]]
            noised_latent = latent * (scheduler.alphas_cumprod[batch_ts] ** 0.5).view(-1, 1, 1, 1).to(device) + \
                            noise * ((1 - scheduler.alphas_cumprod[batch_ts]) ** 0.5).view(-1, 1, 1, 1).to(device)
            batch_ts = batch_ts.to(device)
            if dtype == 'bfloat16':
                t_input = batch_ts.bfloat16()
            elif dtype == 'float16':
                t_input = batch_ts.half()
            else:
                t_input = batch_ts
            text_input = text_embeds[text_embed_idxs[idx: idx + batch_size]]
            noise_pred = unet(noised_latent, t_input, encoder_hidden_states=text_input).sample
            if loss == 'l2':
                error = F.mse_loss(noise, noise_pred, reduction='none').mean(dim=(1, 2, 3))
            elif loss == 'l1':
                error = F.l1_loss(noise, noise_pred, reduction='none').mean(dim=(1, 2, 3))
            elif loss == 'huber':
                error = F.huber_loss(noise, noise_pred, reduction='none').mean(dim=(1, 2, 3))
            else:
                raise NotImplementedError
            pred_errors[idx: idx + len(batch_ts)] = error.detach().cpu()
            idx += len(batch_ts)
    return pred_errors

def eval_prob_adaptive_differentiable(unet, latent, text_embeds, scheduler, args, latent_size=64, controlnet=None, all_noise=None, controlnet_cond=None):
    """
    Wie dein eval_prob_adaptive, aber:
      - verwendet eval_error_differentiable (keine no_grad/detach/cpu)
      - Rückgabe: pred_idx, data, errors_per_class (Tensor [C])
        => errors_per_class kannst du für logits/CE-Loss nutzen.
    Hinweis:
      - Control-Flow (Selektion verbleibender Prompts) ist wie im Original,
        beeinflusst aber nicht den Gradientenfluss der Fehlerwerte selbst.
    """
    scheduler_config = get_scheduler_config(args)
    T = scheduler_config['num_train_timesteps']
    max_n_samples = max(args.n_samples)

    device = latent.device

    if all_noise is None:
        all_noise = torch.randn((max_n_samples * args.n_trials, 4, latent_size, latent_size), device=device)

    if args.dtype == 'float16':
        all_noise = all_noise.half()
        scheduler.alphas_cumprod = scheduler.alphas_cumprod.to(device).half()
    elif args.dtype == 'bfloat16':
        all_noise = all_noise.bfloat16()
        scheduler.alphas_cumprod = scheduler.alphas_cumprod.to(device).bfloat16()
    else:
        scheduler.alphas_cumprod = scheduler.alphas_cumprod.to(device).float()

    if not hasattr(scheduler, "timesteps") or scheduler.timesteps is None:
        scheduler.set_timesteps(int(T))
    data = dict()
    t_evaluated = set()
    remaining_prmpt_idxs = list(range(len(text_embeds)))

    # gleiche t-Auswahl wie im Original
    start = T // max_n_samples // 2
    t_to_eval = list(range(start, T, T // max_n_samples))[:max_n_samples]

    # Wir sammeln ALLE (ts, noise_idxs, text_embed_idxs), rechnen die Errors einmal,
    # und gruppieren danach. Das reduziert Overhead und behält Gradienten.
    global_ts = []
    global_noise_idxs = []
    global_text_embed_idxs = []

    # Merker, um pro Round zu wissen, was neu ist
    per_round_spans = []  # (start_index, end_index, curr_prompt_set)

    # Round-Runden wie im Original
    for n_samples, n_to_keep in zip(args.n_samples, args.to_keep):
        curr_t_to_eval = t_to_eval[len(t_to_eval) // n_samples // 2::len(t_to_eval) // n_samples][:n_samples]
        curr_t_to_eval = [t for t in curr_t_to_eval if t not in t_evaluated]

        round_start = len(global_ts)

        # wie im Original: pro verbleibendem Prompt alle ts * n_trials
        for prompt_i in remaining_prmpt_idxs:
            for t_idx, t in enumerate(curr_t_to_eval, start=len(t_evaluated)):
                # ts: je t 'n_trials' Wiederholungen
                global_ts.extend([t] * args.n_trials)
                # noise-indizes für diesen Block
                global_noise_idxs.extend(list(range(args.n_trials * t_idx, args.n_trials * (t_idx + 1))))
                # prompt-indizes
                global_text_embed_idxs.extend([prompt_i] * args.n_trials)

        t_evaluated.update(curr_t_to_eval)

        round_end = len(global_ts)
        per_round_spans.append((round_start, round_end, remaining_prmpt_idxs.copy()))

        # Nach der Round-Selektion wird remaining_prmpt_idxs upgedated,
        # das machen wir NACH dem Fehler-Compute unten (einmalig über alle Runden).

    # 1x differenzierbar Fehler berechnen
    pred_errors_all = eval_error_differentiable(
        unet, scheduler, latent, all_noise,
        ts=global_ts, noise_idxs=global_noise_idxs,
        text_embeds=text_embeds, text_embed_idxs=global_text_embed_idxs,
        batch_size=args.batch_size, dtype=args.dtype, loss=args.loss, controlnet=controlnet,
        conditioning_scale=args.cond_scale, controlnet_cond=controlnet_cond
    ) 

    # Jetzt die Round-Logik „matchen“ (ohne detach)
    # und dabei data befüllen wie im Original
    data = dict()
    t_evaluated = set()
    remaining_prmpt_idxs = list(range(len(text_embeds)))
    pos = 0

    for round_start, round_end, prompt_set_snapshot in per_round_spans:
        # Slice der aktuellen Round
        ts_slice            = global_ts[round_start:round_end]
        noise_idxs_slice    = global_noise_idxs[round_start:round_end]
        text_embed_idxs_slice = global_text_embed_idxs[round_start:round_end]
        errors_slice        = pred_errors_all[round_start:round_end]  # [S]

        # Gruppieren wie im Original
        round_data = _group_errors_by_prompt(ts_slice, text_embed_idxs_slice, errors_slice)
        # Merge in 'data' (verkettet pro Prompt)
        for prompt_i, pack in round_data.items():
            if prompt_i not in data:
                data[prompt_i] = {'t': pack['t'], 'pred_errors': pack['pred_errors']}
            else:
                data[prompt_i]['t'] = torch.cat([data[prompt_i]['t'], pack['t']], dim=0)
                data[prompt_i]['pred_errors'] = torch.cat([data[prompt_i]['pred_errors'], pack['pred_errors']], dim=0)

        # Selektion nächster verbleibender Prompts wie im Original
        # (Nutzen die bisher gesammelten Fehler in 'data')
        mean_errs = torch.stack(
            [data[p]['pred_errors'].mean() for p in remaining_prmpt_idxs], dim=0
        )  # [len(rem)]
        # errors = [-mean]; best k größte => kleinste mean_errs
        scores = -mean_errs
        # Wir nehmen den zu dieser Runde passenden n_to_keep:
        # Finde n_to_keep dieser Runde über Matching:
        # (einfacher: iteriere erneut die args.n_samples/args.to_keep parallel)
        # Aber wir kennen ihn noch: er war beim Append der Round in per_round_spans nicht gespeichert.
        # Wir lösen es pragmatisch: recompute Index dieser Round.
        round_idx = per_round_spans.index((round_start, round_end, prompt_set_snapshot))
        n_to_keep = args.to_keep[round_idx]

        best_idxs_local = torch.topk(scores, k=n_to_keep, dim=0).indices.tolist()
        remaining_prmpt_idxs = [remaining_prmpt_idxs[i] for i in best_idxs_local]

        # t_evaluated erweitern
        # (wir könnten curr_t_to_eval für diese Runde rekonstruieren; für die nächste Runde spielt es
        #  hier aber keine Rolle mehr, da wir ts bereits gebaut haben.)
        # -> Wir lassen diese Verwaltungsvariable hier symbolisch; sie wird oben nicht mehr benötigt.

    # Organize output wie im Original:
    if len(remaining_prmpt_idxs) > 1:
        with torch.no_grad():
            mean_errs_final = torch.stack(
                [data[p]['pred_errors'].mean() for p in remaining_prmpt_idxs], dim=0
            )
        best_local = torch.argmin(mean_errs_final).item()
        remaining_prmpt_idxs = [remaining_prmpt_idxs[best_local]]

    pred_idx = remaining_prmpt_idxs[0]

    # Zusätzlich: Differenzierbare Fehler über ALLE Klassen (für Training)
    all_prompt_indices = list(range(len(text_embeds)))
    errors_per_prompt = _mean_error_per_prompt(data, all_prompt_indices)  # [P], differentiable
    errors_per_class = _mean_error_per_class(data, errors_per_prompt)  # [C], differentiable

    return pred_idx, data, errors_per_class


def eval_error_differentiable(
    unet, scheduler, latent, all_noise, ts, noise_idxs,
    text_embeds, text_embed_idxs, batch_size=32, dtype='float32', loss='l2',
    controlnet=None, conditioning_scale: float = 1.0, controlnet_cond=None
):
    """
    Differenzierbare Fehlerberechnung:
      - scheduler.scale_model_input(...) für ControlNet & UNet
      - robustes Handling für Pixel-(3ch) vs. Latent-(4ch) ControlNet
      - timesteps als long, ggf. pro-unique-t Schritt mit skalierter Eingabe
    """
    assert len(ts) == len(noise_idxs) == len(text_embed_idxs)
    device = latent.device
    pred_errors = []

    # Timesteps sicher initialisieren (für scale_model_input/index_for_timestep)
    if not hasattr(scheduler, "timesteps") or scheduler.timesteps is None:
        scheduler.set_timesteps(int(max(ts) + 1))

    # erwartete Eingangs-Kanalzahl des ControlNet bestimmen (3 == Pixel, 4 == Latent)
    cond_ch = None       # z.B. 3 (Pixel-CN)
    sample_in_ch = None  # z.B. 4 (Latent-Sample)
    if controlnet is not None:
        try:
            cond_ch = controlnet.config.conditioning_channels
        except Exception:
            cond_ch = 3
        try:
            sample_in_ch = controlnet.config.in_channels
        except Exception:
            sample_in_ch = 4

    idx_global = 0
    total = len(ts)
    while idx_global < total:
        end = min(idx_global + batch_size, total)
        batch_ts_list    = ts[idx_global:end]
        batch_noise_idxs = noise_idxs[idx_global:end]
        batch_text_idxs  = text_embed_idxs[idx_global:end]

        # Timesteps (long), Noise & Alphas
        batch_ts = torch.tensor(batch_ts_list, device=device, dtype=torch.long)   # [B]
        noise    = all_noise[batch_noise_idxs]                                     # [B,4,h,w]

        alphas = scheduler.alphas_cumprod.to(device)[batch_ts].view(-1, 1, 1, 1)
        if dtype == 'float16':
            noise  = noise.half();   alphas = alphas.half()
        elif dtype == 'bfloat16':
            noise  = noise.bfloat16(); alphas = alphas.bfloat16()
        else:
            noise  = noise.float();  alphas = alphas.float()

        noised_latent = latent * (alphas ** 0.5) + noise * ((1.0 - alphas) ** 0.5)  # [B,4,h,w]
        text_input    = text_embeds[batch_text_idxs]                                 # [B, seq, hid]

        B = noised_latent.size(0)
        errors_B = torch.empty(B, device=device, dtype=noise.dtype)

        # Pro unique timestep skalieren & vorwärts
        unique_ts = torch.unique(batch_ts)
        for t_val in unique_ts:
            sel   = (batch_ts == t_val)
            idxs  = sel.nonzero(as_tuple=True)[0]

            nl_sel    = noised_latent[idxs]     # [b_t,4,h,w]
            noise_sel = noise[idxs]             # [b_t,4,h,w]
            txt_sel   = text_input[idxs]        # [b_t,seq,hid]

            # UNet-Input skalieren (ein Skalar-t)
            lat_in  = scheduler.scale_model_input(nl_sel, t_val)
            t_batch = torch.full((nl_sel.size(0),), int(t_val.item()), device=device, dtype=batch_ts.dtype)

            if controlnet is None:
                noise_pred_sel = unet(lat_in, t_batch, encoder_hidden_states=txt_sel).sample
            else:
                down_res, mid_res = controlnet(
                    lat_in, t_batch,
                    encoder_hidden_states=txt_sel,
                    controlnet_cond=controlnet_cond,
                    conditioning_scale=conditioning_scale,
                    return_dict=False,
                )
                noise_pred_sel = unet(
                    lat_in, t_batch, encoder_hidden_states=txt_sel,
                    down_block_additional_residuals=down_res,
                    mid_block_additional_residual=mid_res
                ).sample

            # Fehler pro Sample
            if loss == 'l2':
                err_sel = F.mse_loss(noise_sel, noise_pred_sel, reduction='none').mean(dim=(1, 2, 3))
            elif loss == 'l1':
                err_sel = F.l1_loss(noise_sel, noise_pred_sel, reduction='none').mean(dim=(1, 2, 3))
            elif loss == 'huber':
                err_sel = F.huber_loss(noise_sel, noise_pred_sel, reduction='none').mean(dim=(1, 2, 3))
            else:
                raise NotImplementedError

            errors_B[idxs] = err_sel

        pred_errors.append(errors_B)
        idx_global = end

    return torch.cat(pred_errors, dim=0)


def group_errors_per_prompt_idx(ts, text_embed_idxs, pred_errors, remaining_prmpt_idxs):
    """
    Baut aus (ts, text_embed_idxs, pred_errors) die pro-Prompt aggregierten Fehler.
    Alles tensoriell, ohne detach/cpu. Rückgabe: dict[prompt_i] -> (t_tensor, error_tensor).
    """
    device = pred_errors.device
    ts_t = torch.tensor(ts, device=device, dtype=torch.long)
    tei_t = torch.tensor(text_embed_idxs, device=device, dtype=torch.long)

    data = {}
    for prompt_i in remaining_prmpt_idxs:
        mask = (tei_t == prompt_i)
        prompt_ts = ts_t[mask]                # [K]
        prompt_pred_errors = pred_errors[mask]# [K]
        data[prompt_i] = (prompt_ts, prompt_pred_errors)
    return data

def avg_error_per_class(data, class_indices):
    """
    data: dict[prompt_i] -> (t_tensor, error_tensor)
    Rückgabe: Tensor [C] mit mittlerem Fehler pro Klasse in der Reihenfolge class_indices.
    """
    errs = []
    for ci in class_indices:
        _, e = data[ci]
        errs.append(e.mean())
    return torch.stack(errs, dim=0)  

def _group_errors_by_prompt(ts, text_embed_idxs, pred_errors):
    """
    Baut ein Dict: prompt_i -> {'t': Tensor[..], 'pred_errors': Tensor[..]}
    Alles auf demselben Device, mit Gradienten.
    """
    device = pred_errors.device
    ts_t = torch.tensor(ts, device=device, dtype=torch.long)
    tei_t = torch.tensor(text_embed_idxs, device=device, dtype=torch.long)

    data = {}
    uniq = torch.unique(tei_t).tolist()
    for prompt_i in uniq:
        mask = (tei_t == prompt_i)
        data[prompt_i] = {
            't': ts_t[mask],                        # [k]
            'pred_errors': pred_errors[mask]        # [k]
        }
    return data


def _mean_error_per_class(data, class_to_prompts, fill_value=float('inf')):
    device = next(iter(data.values()))['pred_errors'].device if len(data) > 0 else torch.device('cpu')
    class_ids = sorted(class_to_prompts.keys())
    errs = []
    for cid in class_ids:
        pis = class_to_prompts[cid]
        vals = [data[pi]['pred_errors'].mean()
                for pi in pis if pi in data and 'pred_errors' in data[pi]]
        if len(vals) == 0:
            errs.append(torch.tensor(fill_value, device=device))
        else:
            errs.append(torch.stack(vals).mean())
    return torch.stack(errs, dim=0)


def _mean_error_per_prompt(data, prompt_indices):
    """
    Liefert Tensor [C] mit mittleren Fehlern pro Klasse/Prompt in Reihenfolge prompt_indices.
    Fehlende Keys (falls adaptive Auswahl nicht alle enthält) werden mit +inf gefüllt.
    """
    errs = []
    device = next(iter(data.values()))['pred_errors'].device if len(data) > 0 else torch.device('cpu')
    for pi in prompt_indices:
        if pi in data:
            errs.append(data[pi]['pred_errors'].mean())
        else:
            errs.append(torch.tensor(float('inf'), device=device))
    return torch.stack(errs, dim=0) 

def main():
    parser = argparse.ArgumentParser()

    # dataset args
    parser.add_argument('--dataset', type=str, default='pets',
                        choices=['pets', 'flowers', 'stl10', 'mnist', 'cifar10', 'food', 'caltech101', 'imagenet',
                                 'objectnet', 'aircraft', 'thz_700', 'thz_for_adapter', 'thz_raw'], help='Dataset to use')
    parser.add_argument('--split', type=str, default='train', choices=['train', 'test'], help='Name of split')

    # run args
    parser.add_argument('--version', type=str, default='2-1', help='Stable Diffusion model version')
    parser.add_argument('--img_size', type=int, default=512, choices=(256, 512), help='Number of trials per timestep')
    parser.add_argument('--batch_size', '-b', type=int, default=32)
    parser.add_argument('--n_trials', type=int, default=1, help='Number of trials per timestep')
    parser.add_argument('--prompt_path', type=str, required=True, help='Path to csv file with prompts to use')
    parser.add_argument('--noise_path', type=str, default=None, help='Path to shared noise to use')
    parser.add_argument('--subset_path', type=str, default=None, help='Path to subset of images to evaluate')
    parser.add_argument('--dtype', type=str, default='float16', choices=('float16', 'float32', 'bfloat16'),
                        help='Model data type to use')
    parser.add_argument('--interpolation', type=str, default='bicubic', help='Resize interpolation type')
    parser.add_argument('--extra', type=str, default=None, help='To append to the run folder name')
    parser.add_argument('--n_workers', type=int, default=1, help='Number of workers to split the dataset across')
    parser.add_argument('--worker_idx', type=int, default=0, help='Index of worker to use')
    parser.add_argument('--load_stats', action='store_true', help='Load saved stats to compute acc')
    parser.add_argument('--loss', type=str, default='l2', choices=('l1', 'l2', 'huber'), help='Type of loss to use')

    # args for adaptively choosing which classes to continue trying
    parser.add_argument('--to_keep', nargs='+', type=int, required=True)
    parser.add_argument('--n_samples', nargs='+', type=int, required=True)

    args = parser.parse_args()
    assert len(args.to_keep) == len(args.n_samples)

    # make run output folder
    name = f"v{args.version}_{args.n_trials}trials_"
    name += '_'.join(map(str, args.to_keep)) + 'keep_'
    name += '_'.join(map(str, args.n_samples)) + 'samples'
    if args.interpolation != 'bicubic':
        name += f'_{args.interpolation}'
    if args.loss == 'l1':
        name += '_l1'
    elif args.loss == 'huber':
        name += '_huber'
    if args.img_size != 512:
        name += f'_{args.img_size}'
    if args.extra is not None:
        run_folder = osp.join(LOG_DIR, args.dataset + '_' + args.extra, name)
    else:
        run_folder = osp.join(LOG_DIR, args.dataset, name)
    os.makedirs(run_folder, exist_ok=True)
    print(f'Run folder: {run_folder}')

    # set up dataset and prompts
    interpolation = INTERPOLATIONS[args.interpolation]
    transform = get_transform(interpolation, args.img_size)
    latent_size = args.img_size // 8
    target_dataset = get_target_dataset(args.dataset, train=args.split == 'train', transform=transform)
    prompts_df = pd.read_csv(args.prompt_path)

    # load pretrained models
    vae, tokenizer, text_encoder, unet, scheduler = get_sd_model(args)
    vae = vae.to(device)
    text_encoder = text_encoder.to(device)
    unet = unet.to(device)
    torch.backends.cudnn.benchmark = True

    # load noise
    if args.noise_path is not None:
        assert not args.zero_noise
        all_noise = torch.load(args.noise_path).to(device)
        print('Loaded noise from', args.noise_path)
    else:
        all_noise = None

    # refer to https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion.py#L276
    text_input = tokenizer(prompts_df.prompt.tolist(), padding="max_length",
                           max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")
    embeddings = []
    with torch.inference_mode():
        for i in range(0, len(text_input.input_ids), 100):
            text_embeddings = text_encoder(
                text_input.input_ids[i: i + 100].to(device),
            )[0]
            embeddings.append(text_embeddings)
    text_embeddings = torch.cat(embeddings, dim=0)
    assert len(text_embeddings) == len(prompts_df)

    # subset of dataset to evaluate
    if args.subset_path is not None:
        idxs = np.load(args.subset_path).tolist()
    else:
        idxs = list(range(len(target_dataset)))
    idxs_to_eval = idxs[args.worker_idx::args.n_workers]

    formatstr = get_formatstr(len(target_dataset) - 1)
    correct = 0
    total = 0
    pbar = tqdm.tqdm(idxs_to_eval)
    for i in pbar:
        if total > 0:
            pbar.set_description(f'Acc: {100 * correct / total:.2f}%')
        fname = osp.join(run_folder, formatstr.format(i) + '.pt')
        if os.path.exists(fname):
            print('Skipping', i)
            if args.load_stats:
                data = torch.load(fname)
                correct += int(data['pred'] == data['label'])
                total += 1
            continue
        image, label = target_dataset[i]
        with torch.no_grad():
            img_input = image.to(device).unsqueeze(0)
            if args.dtype == 'float16':
                img_input = img_input.half()
            elif args.dtype == 'bfloat16':
                img_input = img_input.bfloat16()
            x0 = vae.encode(img_input).latent_dist.mean
            x0 *= 0.18215
        pred_idx, pred_errors = eval_prob_adaptive(unet, x0, text_embeddings, scheduler, args, latent_size, all_noise)
        pred = prompts_df.classidx[pred_idx]
        torch.save(dict(errors=pred_errors, pred=pred, label=label), fname)
        if pred == label:
            correct += 1
        total += 1


if __name__ == '__main__':
    main()
