import torch
from eval_prob_adaptiv_with_adapter import eval_prob_adaptive, default_unet_forward

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
    def __init__(self, dtype="float16", loss="l2"):
        self.dtype = dtype
        self.loss = loss
        self.vae = self.unet = self.scheduler = None
        self.forward_fn = default_unet_forward

    def prepare(self, resources):
        self.vae = resources["vae"]
        self.unet = resources["unet"]
        self.scheduler = resources["scheduler"]
        self.forward_fn = resources.get("forward_fn", default_unet_forward)

    def predict(self, sample):
        x_in = sample["image"]
        embeds = sample["embeds"]
        extra_cond = sample["runner"].get("extra_cond")
        with torch.no_grad():
            latent = self.vae.encode(x_in).latent_dist.mean * 0.18215
        pred_idx, _ = eval_prob_adaptive(
            self.unet, latent, embeds, self.scheduler, sample["args"],
            latent_size=latent.shape[-1],
            all_noise=sample["runner"].get("all_noise"),
            forward_fn=self.forward_fn, extra_cond=extra_cond
        )
        return int(sample["prompts_df"].classidx[pred_idx])
