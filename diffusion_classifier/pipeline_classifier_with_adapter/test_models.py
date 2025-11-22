import torch, tqdm
import os, copy, time, json, argparse, random
from pathlib import Path
from typing import Tuple, Optional, List


path = '/pfs/work9/workspace/scratch/ma_lilipper-lippert_bachelorthesis_ws/final_eval/resnet_dropout_pretrained_SGD_2172774/resnet50_bfloat16_251118_1336/final_eval/01.pt'

data = torch.load(path, map_location='cpu')
print(type(data))
if isinstance(data, dict):
    print("Keys:", data.keys())
else:
    print("Kein Dict, sondern:", type(data))
    print("Inhalt:", data)
