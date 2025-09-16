from typing import List, Dict, Any
import torch
from torch import nn
from adapters.base import BaseAdapter

def _merge_residual_lists(a, b):
    if a is None: return b
    if b is None: return a
    assert len(a) == len(b), f"Residual length mismatch: {len(a)} vs {len(b)}"
    return [xa + xb for xa, xb in zip(a, b)]  # (bei Bedarf torch.cat([...], dim=1))

class UNetWithAdapters(nn.Module):
    def __init__(self, unet: nn.Module, adapters: List[BaseAdapter] = None):
        super().__init__()
        self.unet = unet
        self.adapters = nn.ModuleList(adapters or [])

    def forward(self, sample, timestep, encoder_hidden_states, extra_cond: Dict[str, Any] = None, **kwargs):
        down_res = None
        mid_res  = None
        extra_kwargs_acc: Dict[str, Any] = {}

        for adapter in self.adapters:
            contrib = adapter(sample, timestep, encoder_hidden_states, extra_cond=extra_cond)
            down_res = _merge_residual_lists(down_res, contrib.get("down_block_additional_residuals"))
            if "mid_block_additional_residual" in contrib:
                mid_val = contrib["mid_block_additional_residual"]
                if mid_val is not None:
                    mid_res = mid_val if mid_res is None else (mid_res + mid_val)
            for k, v in contrib.items():
                if k in ("down_block_additional_residuals", "mid_block_additional_residual"): continue
                extra_kwargs_acc[k] = v

        return self.unet(
            sample, timestep,
            encoder_hidden_states=encoder_hidden_states,
            down_block_additional_residuals=down_res,
            mid_block_additional_residual=mid_res,
            **extra_kwargs_acc
        )
