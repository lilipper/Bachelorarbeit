from typing import Optional, Dict, Callable, List
import torch
from torch import nn
from pipeline_classifier_with_adapter.classifiers.base import BaseClassifier

class TorchvisionClassifier(BaseClassifier):
    def __init__(self, backbone: nn.Module, num_classes: int, weights: Optional[object] = None, input_adapter: Optional[nn.Module] = None):
        super().__init__()
        self.model = backbone
        self.input_adapter = input_adapter
        self.num_classes = num_classes
        self.weights = weights
        self.preprocess: Optional[Callable] = None
        self.categories: Optional[List[str]] = None

        if weights is not None:
            # Fall A: torchvision Weights-Enum (hat .meta und .transforms())
            if hasattr(weights, "meta"):
                self.categories = weights.meta.get("categories")
                self.preprocess = weights.transforms() if hasattr(weights, "transforms") else None
            # Fall B: direkt eine Transform-Pipeline (callable)
            elif callable(weights):
                self.preprocess = weights

    @torch.inference_mode()
    def predict(self, img_tensor: torch.Tensor, *, extra_cond: Optional[Dict] = None) -> torch.Tensor:
        x = img_tensor
        self.model.eval()
        if self.input_adapter is not None:
            x = self.input_adapter(x, extra_cond=extra_cond)  # bleibt [B,3,H,W]

        batch = self.preprocess(x)
        prediction = self.model(batch).squeeze(0).softmax(0)

        # Make predictions
        label = prediction.argmax().item()
        score = prediction[label].item()
        
        # Use meta to get the labels
        category_name = self.weights.meta['categories'][label]
        return  label, prediction, score, category_name

