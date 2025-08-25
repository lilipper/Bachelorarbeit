from typing import Tuple
import torch
import torch.nn as nn
from torchvision.models import (
resnet18, ResNet18_Weights,
convnext_tiny, ConvNeXt_Tiny_Weights,
vit_b_16, ViT_B_16_Weights,
)




def _maybe_adjust_classifier(model: nn.Module, num_classes: int) -> nn.Module:
    """Robuste Anpassung der letzten Schicht auf num_classes –
    falls der Builder kein num_classes-Argument unterstützt (alte torchvision-Versionen)."""
    # ResNet
    if hasattr(model, 'fc') and isinstance(model.fc, nn.Linear):
        in_f = model.fc.in_features
        model.fc = nn.Linear(in_f, num_classes)
        return model
    # ConvNeXt
    if hasattr(model, 'classifier') and isinstance(model.classifier, nn.Sequential):
    # letztes Linear-Glied ersetzen
        for i in reversed(range(len(model.classifier))):
            if isinstance(model.classifier[i], nn.Linear):
                in_f = model.classifier[i].in_features
                model.classifier[i] = nn.Linear(in_f, num_classes)
                return model
    # ViT
    if hasattr(model, 'heads') and isinstance(model.heads, nn.Sequential):
    # torchvision ViT nutzt heads[-1]
        for i in reversed(range(len(model.heads))):
            if isinstance(model.heads[i], nn.Linear):
                in_f = model.heads[i].in_features
                model.heads[i] = nn.Linear(in_f, num_classes)
                return model
    if hasattr(model, 'head') and isinstance(model.head, nn.Linear):
        in_f = model.head.in_features
        model.head = nn.Linear(in_f, num_classes)
        return model
    return model




def create_model(name: str, num_classes: int, pretrained: bool, img_size: int = 224) -> Tuple[nn.Module, int]:
    name = name.lower()


    if name == 'resnet18':
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        try:
            model = resnet18(weights=weights, num_classes=num_classes)
        except TypeError:
            model = resnet18(weights=weights)
            model = _maybe_adjust_classifier(model, num_classes)
            in_channels = 3


    elif name == 'convnext_tiny':
        weights = ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
        try:
            model = convnext_tiny(weights=weights, num_classes=num_classes)
        except TypeError:
            model = convnext_tiny(weights=weights)
            model = _maybe_adjust_classifier(model, num_classes)
            in_channels = 3


    elif name == 'vit_b_16':
        weights = ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None
        try:
            model = vit_b_16(weights=weights, num_classes=num_classes, image_size=img_size)
        except TypeError:
            model = vit_b_16(weights=weights)
            model = _maybe_adjust_classifier(model, num_classes)
            in_channels = 3


    else:
        raise ValueError(f"Unbekanntes Modell: {name}. Erlaubt: resnet18, convnext_tiny, vit_b_16")


    return model, in_channels