import os.path as osp, numpy as np, torch, pandas as pd
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode

def get_transform(size=512, interpolation=InterpolationMode.BICUBIC):
    return T.Compose([
        T.Resize(size, interpolation=interpolation),
        T.CenterCrop(size),
        T.ToTensor(),
        T.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5]),
    ])

def load_prompts_csv(path):
    df = pd.read_csv(path)
    return df, df.prompt.astype(str).tolist()

def load_thz_indexed(base, index):
    """
    Erwartet Dateien {index}.pt/.pth/.npy unter base.
    Gibt (1,1,T,H,W) oder None zurück.
    """
    if base is None: return None
    for ext in (".pt",".pth",".npy"):
        p = osp.join(base, f"{index}{ext}")
        if osp.isfile(p):
            if ext == ".npy":
                arr = np.load(p).astype(np.float32)
                vol = torch.from_numpy(arr)
            else:
                vol = torch.load(p, map_location="cpu")
            if vol.dim()==3: vol = vol.unsqueeze(0)   # (1,T,H,W)
            if vol.dim()==4 and vol.shape[0]!=1: vol = vol[:1]
            return vol.unsqueeze(0)                   # (1,1,T,H,W)
    return None
