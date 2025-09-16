import torch, torch.nn as nn, torch.nn.functional as F
from typing import Optional, Dict, Any
from adapter.ControlNet import ControlNet  # deine Klasse aus Script 1

class ControlNetAdapterWrapper(nn.Module):
    """
    3D→2D Feedback-Adapter: nimmt (B,1,T,H,W) und gibt RGB (B,3,S,S) in [0,1].
    """
    def __init__(self, controlnet, out_size=512, target_T=500, mid_channels=8, stride_T=3):
        super().__init__()
        self.net = controlnet
        self.out_size = out_size
        self.target_T = target_T
        self.downT = nn.Sequential(
            nn.Conv3d(1, mid_channels, kernel_size=(5,3,3), stride=(stride_T,1,1), padding=(2,1,1), bias=False),
            nn.SiLU(),
            nn.Conv3d(mid_channels, 1, kernel_size=1, stride=1, padding=0, bias=False)
        )

    def forward(self, vol):  # vol: [B,1,T,H,W] -> rgb [B,3,S,S] in [0,1]
        vol = self.downT(vol)
        if self.target_T is not None and vol.shape[2] != self.target_T:
            vol = F.interpolate(vol, size=(self.target_T, vol.shape[-2], vol.shape[-1]), mode="trilinear", align_corners=False)
        B,_,Tp,H,W = vol.shape
        x_stub = torch.zeros((B,1,Tp,H,W), device=vol.device, dtype=vol.dtype)
        t_stub = torch.zeros((B,), device=vol.device, dtype=torch.long)
        rgb = self.net(x_stub, t_stub, controlnet_cond=vol, conditioning_scale=1.0, context=None)
        if (rgb.shape[-2], rgb.shape[-1]) != (self.out_size, self.out_size):
            rgb = F.interpolate(rgb, size=(self.out_size, self.out_size), mode="bilinear", align_corners=False)
        return rgb.clamp(0,1)

def load_feedback_adapter(ckpt_path: str, device: torch.device, dtype: torch.dtype, out_size: int = 512) -> ControlNetAdapterWrapper:
    """
    Lädt den Feedback-Adapter aus Script 1.
    Erwartet in ckpt:
      - 'adapter_state_dict' (Wrapper-Gewichte)
      - 'controlnet_cfg'     (Konstruktor-Args für ControlNet)
    """
    state = torch.load(ckpt_path, map_location=device)
    cn_cfg = state["controlnet_cfg"]
    controlnet = ControlNet(**cn_cfg).to(device=device, dtype=dtype)
    wrapper = ControlNetAdapterWrapper(controlnet, out_size=out_size).to(device=device, dtype=dtype)
    wrapper.load_state_dict(state["adapter_state_dict"], strict=True)
    wrapper.eval()
    return wrapper
