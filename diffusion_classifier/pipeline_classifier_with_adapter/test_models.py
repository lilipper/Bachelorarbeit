import torch, tqdm
import os, copy, time, json, argparse, random
from pathlib import Path
from typing import Tuple, Optional, List


path = '/pfs/work9/workspace/scratch/ma_lilipper-lippert_bachelorthesis_ws/final_eval/vitb32_dropout_01_2099564/vit_b32_bfloat16_251112_0137/best model/best_checkpoint.pt'

ckpt = torch.load(path, map_location='cpu')
print(ckpt.keys())
model = ckpt['backbone_state_dict']
for k, i in model.items():
    print(k, i.shape)