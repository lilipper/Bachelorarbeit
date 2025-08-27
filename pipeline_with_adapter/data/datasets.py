from typing import Tuple
from pathlib import Path
from torchvision import transforms as T
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader




def build_transforms(img_size: int, is_train: bool, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    if is_train:
        return T.Compose([
        T.Resize(int(img_size * 1.15)),
        T.RandomResizedCrop(img_size),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(mean, std),
        ])
    else:
        return T.Compose([
        T.Resize(int(img_size * 1.15)),
        T.CenterCrop(img_size),
        T.ToTensor(),
        T.Normalize(mean, std),
        ])




def build_dataloaders(data_root: str, img_size: int, batch_size: int, workers: int = 8) -> Tuple[DataLoader, DataLoader, int]:
    root = Path(data_root)
    train_tf = build_transforms(img_size, True)
    val_tf = build_transforms(img_size, False)


    train_ds = ImageFolder(root / 'train', transform=train_tf)
    val_ds = ImageFolder(root / 'val', transform=val_tf)


    num_classes = len(train_ds.classes)


    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True)


    return train_loader, val_loader, num_classes