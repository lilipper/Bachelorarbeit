from typing import Optional, Dict
import torch
from torch import nn
from pipeline_classifier_with_adapter.classifiers.base import BaseClassifier

class TorchvisionClassifier(BaseClassifier):
    def __init__(self, backbone: nn.Module, num_classes: int, input_adapter: Optional[nn.Module] = None):
        super().__init__()
        self.model = backbone
        self.input_adapter = input_adapter

    @torch.inference_mode()
    def predict(self, img_tensor: torch.Tensor, *, extra_cond: Optional[Dict] = None) -> torch.Tensor:
        x = img_tensor
        if self.input_adapter is not None:
            x = self.input_adapter(x, extra_cond=extra_cond)  # bleibt [B,3,H,W]
        return self.model(x)

