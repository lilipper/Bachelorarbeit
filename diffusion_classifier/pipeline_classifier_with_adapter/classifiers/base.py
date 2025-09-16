from typing import Dict, Optional
import torch
from torch import nn

class BaseClassifier(nn.Module):
    def predict(
        self,
        img_tensor: torch.Tensor,          # [B,3,H,W], in [-1,1]
        *,
        extra_cond: Optional[Dict] = None  # optionale Zusatzinfos (unused hier)
    ) -> torch.Tensor:                     # [B,C]
        raise NotImplementedError
