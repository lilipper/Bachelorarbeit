"""
show_results.py

Small convenience wrapper to evaluate a single preconfigured model
via the eval_pipeline module. You select a model by name from the
`models` dictionary and the script forwards all required arguments
to `eval_pipeline.main`.

Example usage
-------------

# List available model keys (by simply looking into the `models` dict)
# and then call, for example:
python show_results.py \
    --pretrained_model_name vit_b32_pretrained_cn_once

# Another example for a diffusion-based model:
python show_results.py \
    --pretrained_model_name dc_latent_dropout


Command line arguments
----------------------

--pretrained_model_name : str (required)
    Name of the pretrained model to evaluate. This must be one of the
    keys defined in the global `models` dictionary in this file, e.g.:

    - "dc_cn_wrapper"
    - "dc_latent_every_split"
    - "dc_latent_once"
    - "dc_latent_dropout"
    - "vitb32_dropout_pretrained_2_2153738"
    - "vitb32_dropout_untrained_2109807"
    - "vitb32_dropout_pretrained_2_2153739"
    - "convnext_tiny_dropout_pretrained_2_2153749"
    - "convnext_tiny_dropout_untrained_2109804"
    - "vitb16_dropout_pretrained_2_2153753"
    - "vitb16_dropout_pretrained_2_2153747"
    - "resnet50_dropout_pretrained_2_2153748"
    - "resnet50_dropout_pretrained_2_2153754"
    - "resnet50_dropout_untrained_2109805"
    - "vit_b16_pretrained_cn"
    - "vit_b16_pretrained_cn_once"
    - "vit_b16_pretrained_cn_once_long_8"
    - "vit_b16_pretrained_cn_once_long_50"
    - "convnext_tiny_pretrained_cn"
    - "convnext_tiny_pretrained_cn_once"
    - "convnext_tiny_pretrained_cn_once_long"
    - "vit_b32_pretrained_cn"
    - "vit_b32_pretrained_cn_once"
    - "vit_b32_pretrained_cn_once_long"
    - "resnet50_pretrained_cn"
    - "resnet50_pretrained_cn_once"
    - "resnet50_pretrained_cn_once_long"

What the script does
--------------------

1. Looks up the entry in the `models` dict for the chosen
   `--pretrained_model_name`.

2. Builds an argument object with:
   - `pretrained_path` : path to the checkpoint
   - `classifier`      : one of {"diffusion", "resnet50", "vit_b_16",
                                 "vit_b_32", "convnext_tiny"}
   - `adapter`         : one of {"cn_wrapper", "latent", "old_cn_wrapper"}
   - fixed settings for:
       - `dataset`     = "thz_for_adapter"
       - `split`       = "test"
       - `n_workers`   = 1
       - `output_dir`  = "./results_eval_pipeline"
       - `prompts_csv` = path to THz prompts CSV

3. Calls `eval_pipeline.main(p_args)` to:
   - run inference on the chosen dataset split,
   - save prediction `.pt` files and GradCAM / visualisations,
   - call `evaluate_predictions` to create a confusion matrix and report.

This script is meant as the easiest entry point to inspect the final
results of predefined models used in the thesis.
"""


import torch, tqdm
import os, copy, time, json, argparse, random
from pathlib import Path
from typing import Tuple, Optional, List

from eval_pipeline import main as pipeline


models = {
    #Diffusion Models
    'dc_cn_wrapper': {
        'path' :'/pfs/work9/workspace/scratch/ma_lilipper-lippert_bachelorthesis_ws/Bachelorarbeit/diffusion_classifier/results_for_thesis_evaluation/a_cv/checkpoints_cn_official/sd2-1_img256_float32_251028_1228/split_001/best_dc.pt',
        'classifier' : 'diffusion',
        'adapter' : 'cn_wrapper',
    },
    'dc_latent_every_split': {
        'path' :'/pfs/work9/workspace/scratch/ma_lilipper-lippert_bachelorthesis_ws/Bachelorarbeit/diffusion_classifier/results_for_thesis_evaluation/c_dc/train_dc_with_original_cn_multichannel_2036332/sd2-1_img256_float32_251102_1054/split_003/best_dc.pt',
        'classifier' : 'diffusion',
        'adapter' : 'latent',
    },
    'dc_latent_once': {
        'path' :'/pfs/work9/workspace/scratch/ma_lilipper-lippert_bachelorthesis_ws/Bachelorarbeit/diffusion_classifier/results_for_thesis_evaluation/c_dc/train_dc_with_original_cn_multichannel_load_once_2051652/sd2-1_img256_float32_251105_0931/split_002/best_dc.pt',
        'classifier' : 'diffusion',
        'adapter' : 'latent',
    },
    'dc_latent_dropout': {
        'path' :'/pfs/work9/workspace/scratch/ma_lilipper-lippert_bachelorthesis_ws/Bachelorarbeit/diffusion_classifier/results_for_thesis_evaluation/c_dc/train_dc_dropout_2d_2112518/sd2-1_img256_float32_251112_1918/best_model/best_model.pt',
        'classifier' : 'diffusion',
        'adapter' : 'latent',
    },




    #Dropout Models

    # vit b32
    'vitb32_dropout_pretrained_2_2153738': {
        'path' :'/pfs/work9/workspace/scratch/ma_lilipper-lippert_bachelorthesis_ws/Bachelorarbeit/diffusion_classifier/results_for_thesis_evaluation/b_dropout/vitb32_dropout_pretrained_2_2153738/vit_b32_bfloat16_251117_1103/best model/best_checkpoint.pt',
        'classifier' : 'vit_b_32',
        'adapter' : 'cn_wrapper',
    },
    'vitb32_dropout_untrained_2109807': {
        'path' :'/pfs/work9/workspace/scratch/ma_lilipper-lippert_bachelorthesis_ws/Bachelorarbeit/diffusion_classifier/results_for_thesis_evaluation/b_dropout/vitb32_dropout_untrained_2109807/vit_b32_bfloat16_251112_1219/best model/best_checkpoint.pt',
        'classifier' : 'vit_b_32',
        'adapter' : 'cn_wrapper',
    },
    'vitb32_dropout_pretrained_2_2153739': {
        'path' :'/pfs/work9/workspace/scratch/ma_lilipper-lippert_bachelorthesis_ws/Bachelorarbeit/diffusion_classifier/results_for_thesis_evaluation/b_dropout/vitb32_dropout_pretrained_2_2153739/vit_b32_bfloat16_251117_1103/best model/best_checkpoint.pt',
        'classifier' : 'vit_b_32',
        'adapter' : 'cn_wrapper',
    },
    # convnext tiny
    'convnext_tiny_dropout_pretrained_2_2153749': {
        'path' :'/pfs/work9/workspace/scratch/ma_lilipper-lippert_bachelorthesis_ws/Bachelorarbeit/diffusion_classifier/results_for_thesis_evaluation/b_dropout/convnext_tiny_dropout_pretrained_2_2153749/convnext_tiny_bfloat16_251117_1112/best model/best_checkpoint.pt',
        'classifier' : 'convnext_tiny',
        'adapter' : 'cn_wrapper',
    },
    'convnext_tiny_dropout_untrained_2109804': {
        'path' :'/pfs/work9/workspace/scratch/ma_lilipper-lippert_bachelorthesis_ws/Bachelorarbeit/diffusion_classifier/results_for_thesis_evaluation/b_dropout/convnext_tiny_dropout_untrained_2109804/convnext_tiny_bfloat16_251112_1154/best model/best_checkpoint.pt',
        'classifier' : 'convnext_tiny',
        'adapter' : 'cn_wrapper',
    },

    # vit b16
    'vitb16_dropout_pretrained_2_2153753': {
        'path' :'/pfs/work9/workspace/scratch/ma_lilipper-lippert_bachelorthesis_ws/Bachelorarbeit/diffusion_classifier/results_for_thesis_evaluation/b_dropout/vitb16_dropout_pretrained_2_2153753/vit_b16_bfloat16_251117_0644/best model/best_checkpoint.pt',
        'classifier' : 'vit_b_16',
        'adapter' : 'cn_wrapper',
    },
    'vitb16_dropout_pretrained_2_2153747': {
        'path' :'/pfs/work9/workspace/scratch/ma_lilipper-lippert_bachelorthesis_ws/Bachelorarbeit/diffusion_classifier/results_for_thesis_evaluation/b_dropout/vitb16_dropout_pretrained_2_2153747/vit_b16_bfloat16_251117_1103/best model/best_checkpoint.pt',
        'classifier' : 'vit_b_16',
        'adapter' : 'cn_wrapper',
    },

    #resnet50
    'resnet50_dropout_pretrained_2_2153748': {
        'path' :'/pfs/work9/workspace/scratch/ma_lilipper-lippert_bachelorthesis_ws/Bachelorarbeit/diffusion_classifier/results_for_thesis_evaluation/b_dropout/resnet50_dropout_pretrained_2_2153748/resnet50_bfloat16_251117_1103/best model/best_checkpoint.pt',
        'classifier' : 'resnet50',
        'adapter' : 'cn_wrapper',
    },
    'resnet50_dropout_pretrained_2_2153754': {
        'path' :'/pfs/work9/workspace/scratch/ma_lilipper-lippert_bachelorthesis_ws/Bachelorarbeit/diffusion_classifier/results_for_thesis_evaluation/b_dropout/resnet50_dropout_pretrained_2_2153754/resnet50_bfloat16_251117_0717/best model/best_checkpoint.pt',
        'classifier' : 'resnet50',
        'adapter' : 'cn_wrapper',
    },
    'resnet50_dropout_untrained_2109805': {
        'path' :'/pfs/work9/workspace/scratch/ma_lilipper-lippert_bachelorthesis_ws/Bachelorarbeit/diffusion_classifier/results_for_thesis_evaluation/b_dropout/resnet50_dropout_untrained_2109805/resnet50_bfloat16_251112_1201/best model/best_checkpoint.pt',
        'classifier' : 'resnet50',
        'adapter' : 'cn_wrapper',
    },
    



    # CV Models
    
    # vit b16
    'vit_b16_pretrained_cn': {
        'path' :'/pfs/work9/workspace/scratch/ma_lilipper-lippert_bachelorthesis_ws/Bachelorarbeit/diffusion_classifier/results_for_thesis_evaluation/a_cv/vitb16_pretrained_cn_wrapper_2050294/vit_b16_bfloat16_251104_1455/split_004/best.pt',
        'classifier' : 'vit_b_16',
        'adapter' : 'old_cn_wrapper',
    },
    'vit_b16_pretrained_cn_once': {
        'path' :'/pfs/work9/workspace/scratch/ma_lilipper-lippert_bachelorthesis_ws/Bachelorarbeit/diffusion_classifier/results_for_thesis_evaluation/a_cv/vitb16_pretrained_cn_wrapper_load_once_2051710/vit_b16_bfloat16_251105_0942/split_009/best.pt',
        'classifier' : 'vit_b_16',
        'adapter' : 'old_cn_wrapper',
    },
    'vit_b16_pretrained_cn_once_long_8': {
        'path' :'/pfs/work9/workspace/scratch/ma_lilipper-lippert_bachelorthesis_ws/Bachelorarbeit/diffusion_classifier/results_for_thesis_evaluation/a_cv/vitb16_pretrained_cn_wrapper_load_once_more_repeats_2071825/vit_b16_bfloat16_251107_1450/split_008/best.pt',
        'classifier' : 'vit_b_16',
        'adapter' : 'old_cn_wrapper',
    },
    'vit_b16_pretrained_cn_once_long_50': {
        'path' :'/pfs/work9/workspace/scratch/ma_lilipper-lippert_bachelorthesis_ws/Bachelorarbeit/diffusion_classifier/results_for_thesis_evaluation/a_cv/vitb16_pretrained_cn_wrapper_load_once_more_repeats_2071825/vit_b16_bfloat16_251107_1450/split_050/best.pt',
        'classifier' : 'vit_b_16',
        'adapter' : 'old_cn_wrapper',
    },
    # convnext tiny
    'convnext_tiny_pretrained_cn': {
        'path' :'/pfs/work9/workspace/scratch/ma_lilipper-lippert_bachelorthesis_ws/Bachelorarbeit/diffusion_classifier/results_for_thesis_evaluation/a_cv/convnext_tiny_pretrained_cn_wrapper_2017198/convnext_tiny_bfloat16_251031_0810/split_004/best.pt',
        'classifier' : 'convnext_tiny',
        'adapter' : 'old_cn_wrapper',
    },
    'convnext_tiny_pretrained_cn_once': {
        'path' :'/pfs/work9/workspace/scratch/ma_lilipper-lippert_bachelorthesis_ws/Bachelorarbeit/diffusion_classifier/results_for_thesis_evaluation/a_cv/convnext_tiny_pretrained_cn_wrapper_load_once_2033949/convnext_tiny_bfloat16_251102_0114/split_004/best.pt',
        'classifier' : 'convnext_tiny',
        'adapter' : 'old_cn_wrapper',
    },
    'convnext_tiny_pretrained_cn_once_long': {
        'path' :'/pfs/work9/workspace/scratch/ma_lilipper-lippert_bachelorthesis_ws/Bachelorarbeit/diffusion_classifier/results_for_thesis_evaluation/a_cv/convnext_tiny_pretrained_cn_wrapper_load_once_more_repeats_2071823/convnext_tiny_bfloat16_251107_1029/split_004/best.pt',
        'classifier' : 'convnext_tiny',
        'adapter' : 'old_cn_wrapper',
    },
    #vit b32
    'vit_b32_pretrained_cn': {
        'path' :'/pfs/work9/workspace/scratch/ma_lilipper-lippert_bachelorthesis_ws/Bachelorarbeit/diffusion_classifier/results_for_thesis_evaluation/a_cv/vit_b_32_pretrained_cn_wrapper_2011485/vit_b32_bfloat16_251031_0810/split_025/best.pt',
        'classifier' : 'vit_b_32',
        'adapter' : 'old_cn_wrapper',
    },
    'vit_b32_pretrained_cn_once': {
        'path' :'/pfs/work9/workspace/scratch/ma_lilipper-lippert_bachelorthesis_ws/Bachelorarbeit/diffusion_classifier/results_for_thesis_evaluation/a_cv/vit_b_32_pretrained_cn_wrapper_load_once_2033952/vit_b32_bfloat16_251102_0114/split_004/best.pt',
        'classifier' : 'vit_b_32',
        'adapter' : 'old_cn_wrapper',
    },
    'vit_b32_pretrained_cn_once_long': {
        'path' :'/pfs/work9/workspace/scratch/ma_lilipper-lippert_bachelorthesis_ws/Bachelorarbeit/diffusion_classifier/results_for_thesis_evaluation/a_cv/vit_b_32_pretrained_cn_wrapper_load_once_more_repeats_2071826/vit_b32_bfloat16_251107_1451/split_038/best.pt',
        'classifier' : 'vit_b_32',
        'adapter' : 'old_cn_wrapper',
    },
    #resnet50
    'resnet50_pretrained_cn': {
        'path' :'/pfs/work9/workspace/scratch/ma_lilipper-lippert_bachelorthesis_ws/Bachelorarbeit/diffusion_classifier/results_for_thesis_evaluation/a_cv/resnet50_train_pretrained_cn_wrapper_2017199/resnet50_bfloat16_251031_0810/split_012/best.pt',
        'classifier' : 'resnet50',
        'adapter' : 'old_cn_wrapper',
    },
    'resnet50_pretrained_cn_once': {
        'path' :'/pfs/work9/workspace/scratch/ma_lilipper-lippert_bachelorthesis_ws/Bachelorarbeit/diffusion_classifier/results_for_thesis_evaluation/a_cv/resnet50_train_pretrained_cn_wrapper_load_once_2033950/resnet50_bfloat16_251102_0114/split_017/best.pt',
        'classifier' : 'resnet50',
        'adapter' : 'old_cn_wrapper',
    },
    'resnet50_pretrained_cn_once_long': {
        'path' :'/pfs/work9/workspace/scratch/ma_lilipper-lippert_bachelorthesis_ws/Bachelorarbeit/diffusion_classifier/results_for_thesis_evaluation/a_cv/resnet50_train_pretrained_cn_wrapper_load_once_more_repeats_2071824/resnet50_bfloat16_251107_1450/split_007/best.pt',
        'classifier' : 'resnet50',
        'adapter' : 'old_cn_wrapper',
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