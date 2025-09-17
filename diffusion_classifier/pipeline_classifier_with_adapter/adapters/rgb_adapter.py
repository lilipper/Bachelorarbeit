import os, json, torch, torch.nn as nn, torch.nn.functional as F
from typing import Dict, Any, Optional
from pipeline_classifier_with_adapter.adapters.base import BaseAdapter
from diffusers import ControlNetModel

# exakt wie im Trainingsscript 2 (damit StateDict passt)
class THzAdapter(nn.Module):
    def __init__(self, ch: int = 16, max_T: int = 64, hw: int = 128, out_hw: int = 512):
        super().__init__()
        self.max_T = max_T; self.hw = hw; self.out_hw = out_hw
        self.enc3d = nn.Sequential(
            nn.Conv3d(1, ch,   3, padding=1), nn.SiLU(),
            nn.Conv3d(ch, ch*2, 3, stride=2, padding=1), nn.SiLU(),
            nn.Conv3d(ch*2, ch*4, 3, stride=2, padding=1), nn.SiLU(),
        )
        self.pool = nn.AdaptiveAvgPool3d((1, None, None))
        self.refine2d = nn.Sequential(
            nn.Conv2d(ch*4, ch*2, 3, padding=1), nn.SiLU(),
            nn.Conv2d(ch*2, ch,   3, padding=1), nn.SiLU(),
        )
        self.head = nn.Conv2d(ch, 3, 1)

    def forward(self, vol: torch.Tensor) -> torch.Tensor:
        B, C, T, H, W = vol.shape
        T_ds = min(T, self.max_T)
        if (T != T_ds) or (H != self.hw) or (W != self.hw):
            vol = F.interpolate(vol, size=(T_ds, self.hw, self.hw), mode="trilinear", align_corners=False)
        x = self.enc3d(vol)
        x = self.pool(x).squeeze(2)        # (B,C',H',W')
        x = self.refine2d(x)               # (B,ch,H',W')
        if x.shape[-1] != self.out_hw:
            x = F.interpolate(x, size=(self.out_hw, self.out_hw), mode="bilinear", align_corners=False)
        return self.head(x)                # (B,3,out_hw,out_hw)

class RGBAdapter(BaseAdapter):
    """Komposit: THzAdapter (→ Control-Bild) + ControlNetModel (→ Residuals)."""
    def __init__(self, controlnet: ControlNetModel, thz_adapter: THzAdapter):
        super().__init__()
        self.controlnet = controlnet
        self.thz_adapter = thz_adapter

    def forward(self, latents, timesteps, text_embeds, extra_cond: Optional[Dict[str, Any]] = None, **_):
        assert extra_cond is not None and ("thz" in extra_cond), "RGBAdapter erwartet extra_cond['thz']"
        thz = extra_cond["thz"].to(device=latents.device, dtype=latents.dtype)
        control_image = self.thz_adapter(thz)  # (B,3,H,W)
        down_samples, mid_sample = self.controlnet(
            latents, timesteps,
            encoder_hidden_states=text_embeds,
            controlnet_cond=control_image,
            conditioning_scale=1.0,
            return_dict=False,
        )
        return {
            "down_block_additional_residuals": down_samples,
            "mid_block_additional_residual": mid_sample,
        }

def load_rgb_adapter(output_dir: str, device: torch.device, dtype: torch.dtype) -> RGBAdapter:
    """
    Erwartet im Ordner:
      - controlnet/   (HF-Ordner)
      - thz_adapter.pt
      - optional: config.json mit {"thz": {"ch": 16, "max_T": 64, "hw": 128, "out_hw": 512}}
    """
    # ControlNet
    cn_dir = os.path.join(output_dir, "controlnet")
    controlnet = ControlNetModel.from_pretrained(cn_dir, torch_dtype=dtype).to(device)
    controlnet.eval()

    # THzAdapter hyperparams aus config.json (falls vorhanden)
    thz_kwargs = dict(ch=16, max_T=64, hw=128, out_hw=512)
    cfg_path = os.path.join(output_dir, "config.json")
    if os.path.isfile(cfg_path):
        try:
            with open(cfg_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            if isinstance(cfg, dict) and "thz" in cfg and isinstance(cfg["thz"], dict):
                thz_kwargs.update({k: cfg["thz"][k] for k in ("ch","max_T","hw","out_hw") if k in cfg["thz"]})
        except Exception:
            pass

    # THzAdapter + StateDict
    thz = THzAdapter(**thz_kwargs).to(device=device, dtype=dtype)
    thz_sd_path = os.path.join(output_dir, "thz_adapter.pt")
    thz.load_state_dict(torch.load(thz_sd_path, map_location=device), strict=True)
    thz.eval()

    return RGBAdapter(controlnet, thz).to(device)

def load_thz_adapter_only(output_dir: str, device: torch.device, dtype: torch.dtype) -> THzAdapter:
    import json, os, torch
    thz_kwargs = dict(ch=16, max_T=64, hw=128, out_hw=512)
    cfg_path = os.path.join(output_dir, "config.json")
    if os.path.isfile(cfg_path):
        try:
            with open(cfg_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            if isinstance(cfg, dict) and "thz" in cfg and isinstance(cfg["thz"], dict):
                thz_kwargs.update({k: cfg["thz"][k] for k in ("ch","max_T","hw","out_hw") if k in cfg["thz"]})
        except Exception:
            pass
    thz = THzAdapter(**thz_kwargs).to(device=device, dtype=dtype)
    thz_sd_path = os.path.join(output_dir, "thz_adapter.pt")
    thz.load_state_dict(torch.load(thz_sd_path, map_location=device), strict=True)
    thz.eval()
    return thz