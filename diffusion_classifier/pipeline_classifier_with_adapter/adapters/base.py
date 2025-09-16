from typing import Dict, Any, Optional
import torch
from torch import nn

class BaseAdapter(nn.Module):
    """
    Vereinheitlichte API: Jeder Adapter bekommt dieselben Inputs und gibt
    Zusatz-Argumente für den UNet.forward() zurück.
    """
    def forward(
        self,
        latents: torch.Tensor,               # [B,4,H,W]
        timesteps: torch.Tensor,             # [B] (long/half)
        text_embeds: torch.Tensor,           # [B,seq,hid]
        extra_cond: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        raise NotImplementedError
