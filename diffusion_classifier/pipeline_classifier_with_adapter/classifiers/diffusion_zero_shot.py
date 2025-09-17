import torch

from pipeline_classifier_with_adapter.classifiers.eval_prob_adaptiv_with_adapter import eval_prob_adaptive, default_unet_forward


_DTYPE_ALIASES = {
    "fp16": torch.float16, "float16": torch.float16, "half": torch.float16,
    "fp32": torch.float32, "float32": torch.float32, "float": torch.float32,
    "bf16": torch.bfloat16, "bfloat16": torch.bfloat16,
}

def _parse_dtype(x):
    if isinstance(x, torch.dtype): return x
    x = str(x).lower()
    if x in _DTYPE_ALIASES: return _DTYPE_ALIASES[x]
    raise ValueError(f"Unsupported dtype: {x}")

class DiffusionZeroShotClassifier:
    """
    Zero-shot nach deinem bisherigen Paper-Setup.
    Erwartet:
      sample = {
        "image": (1,3,H,W) in [-1,1],
        "embeds": Text-Embeddings,
        "runner": {"extra_cond": dict|None, "all_noise": Tensor|None},
        "args": argparse.Namespace,
        "prompts_df": DataFrame mit .classidx
      }
    """
    def __init__(self, dtype="float16", device=None, loss="l2"):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.requested_dtype = _parse_dtype(dtype)
        self.torch_dtype = torch.float32 if (self.device.type == "cpu" and self.requested_dtype != torch.float32) else self.requested_dtype
        self.loss = loss
        self.vae = self.unet = self.scheduler = None
        self.forward_fn = default_unet_forward

    def _to_dd(self, x):
        return x.to(self.device, self.torch_dtype)

    def _to_dd_dict(self, d):
        return {k: (self._to_dd(v) if torch.is_tensor(v) else v) for k, v in d.items()}
    
    def switch_dtype(self, dtype):
        """Optional: dtype zur Laufzeit wechseln (re-castet Modelle)."""
        new_dtype = _parse_dtype(dtype)
        if self.device.type == "cpu" and new_dtype != torch.float32:
            new_dtype = torch.float32
        self.torch_dtype = new_dtype
        if self.vae is not None:  self.vae  = self.vae.to(self.device, self.torch_dtype)
        if self.unet is not None: self.unet = self.unet.to(self.device, self.torch_dtype)

    def prepare(self, vae, unet, scheduler, forward_fn=None):
        self.vae = vae.to(self.device, self.torch_dtype)
        self.unet = unet.to(self.device, self.torch_dtype)
        self.scheduler = scheduler
        self.forward_fn = forward_fn or default_unet_forward

    def predict(self, x_in, embeds, extra_cond, args, prompts_df, all_noise=None):
        with torch.no_grad():
            # 1) Alles auf einheitliches device+dtype
            x = self._to_dd(x_in)
            if torch.is_tensor(embeds): embeds = self._to_dd(embeds)
            if isinstance(extra_cond, dict): extra_cond = self._to_dd_dict(extra_cond)
            if all_noise is not None: all_noise = self._to_dd(all_noise)

            # 2) VAE-Encode -> latent (bereits korrektes dtype/device)
            latent = self.vae.encode(x.unsqueeze(0)).latent_dist.mean
            latent = latent * latent.new_tensor(0.18215)  # skaliert im selben dtype

            # 3) (falls VAE/UNet versehentlich verschieden geladen wurden)
            latent = latent.to(self.unet.device if hasattr(self.unet, "device") else next(self.unet.parameters()).device,
                               next(self.unet.parameters()).dtype)
            
        pred_idx, pred_error = eval_prob_adaptive(
            self.unet, latent, embeds, self.scheduler, args,
            latent_size=latent.shape[-1],
            all_noise=all_noise,
            forward_fn=self.forward_fn, extra_cond=extra_cond
        )
        return int(prompts_df.classidx[pred_idx]), pred_error
