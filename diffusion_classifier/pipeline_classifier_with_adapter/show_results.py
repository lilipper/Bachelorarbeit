import torch, tqdm
import os, copy, time, json, argparse, random
from pathlib import Path
from typing import Tuple, Optional, List

from eval_pipeline import main as pipeline


models = {

}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pretrained_model_name", type=str, required=True, help="Name of pretrained model"
    )
    args = parser.parse_args()
    

    p_args = None
    pipeline(p_args)