import torch, torch.nn as nn, torch.nn.functional as F
from typing import Optional, Dict, Any
from adapter.ControlNet import ControlNet  # deine Klasse aus Script 1
from adapter.ControlNet_Adapter_wrapper import ControlNetAdapterWrapper

def load_feedback_adapter(ckpt_path: str, device: torch.device, dtype: torch.dtype, out_size: int = 512) -> ControlNetAdapterWrapper:
    """
    Lädt den Feedback-Adapter aus Script 1.
    Erwartet in ckpt:
      - 'adapter_state_dict' (Wrapper-Gewichte)
      - 'controlnet_cfg'     (Konstruktor-Args für ControlNet)
    """
    state = torch.load(ckpt_path, map_location=device)
    cn_cfg = state["controlnet_cfg"]
    wrapper = ControlNetAdapterWrapper(cn_cfg, out_size=out_size).to(device=device, dtype=dtype)
    wrapper.load_state_dict(state["adapter_state_dict"], strict=True)
    wrapper.eval()
    return wrapper
