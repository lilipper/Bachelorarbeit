import torch, tqdm
import os, copy, time, json, argparse, random
from pathlib import Path
from typing import Tuple, Optional, List

from eval_pipeline import main as pipeline


models = {
    'vit_b_32_pretrained_dropout_latent': {
        'path' :'/pfs/work9/workspace/scratch/ma_lilipper-lippert_bachelorthesis_ws/final_eval/vitb32_dropout_01_2099564/vit_b32_bfloat16_251112_0137/best model/best_checkpoint.pt',
        'classifier' : 'vit_b_32',
        'adapter' : 'cn_wrapper',
    },
    'vit_b_32_untrained_dropout_latent': {
        'path' :'/pfs/work9/workspace/scratch/ma_lilipper-lippert_bachelorthesis_ws/final_eval/vitb32_dropout_untrained_2109807/vit_b32_bfloat16_251112_1219/best model/best_checkpoint.pt',
        'classifier' : 'vit_b_32',
        'adapter' : 'cn_wrapper',
    },
}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pretrained_model_name", type=str, required=True, help="Name of pretrained model"
    )
    args = parser.parse_args()
    print(f"Starting pipeline evaluation with model: {args.pretrained_model_name}")
    model_info = models[args.pretrained_model_name]
    class FE: pass
    p_args = FE()
    p_args.pretrained_path = model_info['path']
    p_args.classifier = model_info['classifier']
    p_args.adapter = model_info['adapter']
    p_args.output_dir = "./results_eval_pipeline"
    p_args.split = "test"
    p_args.n_workers = 1
    p_args.dataset = "thz_for_adapter"
    p_args.prompts_csv = '/pfs/work9/workspace/scratch/ma_lilipper-lippert_bachelorthesis_ws/Bachelorarbeit/diffusion_classifier/prompts/thz_prompts.csv' 

    print(f"Using pretrained model from: {p_args.pretrained_path}")
    pipeline(p_args)
    print("Pipeline evaluation completed.")

if __name__ == "__main__":
    main()