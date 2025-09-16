import torch

def ensure_thz_shape(thz):
    if thz is None: return None
    if thz.dim()==3: thz = thz.unsqueeze(0).unsqueeze(0)
    elif thz.dim()==4:
        if thz.shape[0]!=1: thz = thz[:1]
        thz = thz.unsqueeze(0)
    elif thz.dim()==5:
        if thz.shape[0]!=1: thz = thz[:1]
    else:
        raise ValueError(f"Unexpected THz shape: {tuple(thz.shape)}")
    return thz.contiguous().float()
