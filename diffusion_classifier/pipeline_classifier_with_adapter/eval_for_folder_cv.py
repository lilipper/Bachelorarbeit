import os
from eval_the_pipeline_results import evaluate_predictions
import torch, tqdm
import os, copy, time, json, argparse, random
from pathlib import Path
from typing import Tuple, Optional, List

from eval_pipeline import main as pipeline


folders_cv_repeated = [ 
    r"/pfs/work9/workspace/scratch/ma_lilipper-lippert_bachelorthesis_ws/final_eval/b_cv/convnext_tiny_pretrained_cn_wrapper_2017198",
    r"/pfs/work9/workspace/scratch/ma_lilipper-lippert_bachelorthesis_ws/final_eval/b_cv/resnet50_train_pretrained_cn_wrapper_2017199",
    r"/pfs/work9/workspace/scratch/ma_lilipper-lippert_bachelorthesis_ws/final_eval/b_cv/vit_b_32_pretrained_cn_wrapper_2011485",
    r"/pfs/work9/workspace/scratch/ma_lilipper-lippert_bachelorthesis_ws/final_eval/b_cv/vitb16_pretrained_cn_wrapper_2050294",
]

folder_cv_one =[
    r"/pfs/work9/workspace/scratch/ma_lilipper-lippert_bachelorthesis_ws/final_eval/b_cv/convnext_tiny_pretrained_cn_wrapper_load_once_2033949",
    r"/pfs/work9/workspace/scratch/ma_lilipper-lippert_bachelorthesis_ws/final_eval/b_cv/resnet50_train_pretrained_cn_wrapper_load_once_2033950",
    r"/pfs/work9/workspace/scratch/ma_lilipper-lippert_bachelorthesis_ws/final_eval/b_cv/vit_b_32_pretrained_cn_wrapper_load_once_2033952",
    r"/pfs/work9/workspace/scratch/ma_lilipper-lippert_bachelorthesis_ws/final_eval/b_cv/vitb16_pretrained_cn_wrapper_load_once_2051710",
]

folder_cv_one_long =[
    r"/pfs/work9/workspace/scratch/ma_lilipper-lippert_bachelorthesis_ws/final_eval/b_cv/convnext_tiny_pretrained_cn_wrapper_load_once_more_repeats_2071823",
    r"/pfs/work9/workspace/scratch/ma_lilipper-lippert_bachelorthesis_ws/final_eval/b_cv/resnet50_train_pretrained_cn_wrapper_load_once_more_repeats_2071824",
    r"/pfs/work9/workspace/scratch/ma_lilipper-lippert_bachelorthesis_ws/final_eval/b_cv/vit_b_32_pretrained_cn_wrapper_load_once_more_repeats_2071826",
    r"/pfs/work9/workspace/scratch/ma_lilipper-lippert_bachelorthesis_ws/final_eval/b_cv/vitb16_pretrained_cn_wrapper_load_once_more_repeats_2071825",
]

all_folders = folder_cv_one + folders_cv_repeated + folder_cv_one_long

                
        



models = {
       # CV Models
    'vit_b16_pretrained_cn': {
        'path' :'/pfs/work9/workspace/scratch/ma_lilipper-lippert_bachelorthesis_ws/final_eval/b_cv/vitb16_pretrained_cn_wrapper_2050294/vit_b16_bfloat16_251104_1455/split_004/best.pt',
        'classifier' : 'vit_b_16',
        'adapter' : 'old_cn_wrapper',
    },
    'vit_b16_pretrained_cn_once': {
        'path' :'/pfs/work9/workspace/scratch/ma_lilipper-lippert_bachelorthesis_ws/final_eval/b_cv/vitb16_pretrained_cn_wrapper_load_once_2051710/vit_b16_bfloat16_251105_0942/split_009/best.pt',
        'classifier' : 'vit_b_16',
        'adapter' : 'old_cn_wrapper',
    },
    'vit_b16_pretrained_cn_once_long_8': {
        'path' :'/pfs/work9/workspace/scratch/ma_lilipper-lippert_bachelorthesis_ws/final_eval/b_cv/vitb16_pretrained_cn_wrapper_load_once_more_repeats_2071825/vit_b16_bfloat16_251107_1450/split_008/best.pt',
        'classifier' : 'vit_b_16',
        'adapter' : 'old_cn_wrapper',
    },
    'vit_b16_pretrained_cn_once_long_50': {
        'path' :'/pfs/work9/workspace/scratch/ma_lilipper-lippert_bachelorthesis_ws/final_eval/b_cv/vitb16_pretrained_cn_wrapper_load_once_more_repeats_2071825/vit_b16_bfloat16_251107_1450/split_050/best.pt',
        'classifier' : 'vit_b_16',
        'adapter' : 'old_cn_wrapper',
    },
    # convnext tiny
    'convnext_tiny_pretrained_cn': {
        'path' :'/pfs/work9/workspace/scratch/ma_lilipper-lippert_bachelorthesis_ws/final_eval/b_cv/convnext_tiny_pretrained_cn_wrapper_2017198/convnext_tiny_bfloat16_251031_0810/split_004/best.pt',
        'classifier' : 'convnext_tiny',
        'adapter' : 'old_cn_wrapper',
    },
    'convnext_tiny_pretrained_cn_once': {
        'path' :'/pfs/work9/workspace/scratch/ma_lilipper-lippert_bachelorthesis_ws/final_eval/b_cv/convnext_tiny_pretrained_cn_wrapper_load_once_2033949/convnext_tiny_bfloat16_251102_0114/split_004/best.pt',
        'classifier' : 'convnext_tiny',
        'adapter' : 'old_cn_wrapper',
    },
    'convnext_tiny_pretrained_cn_once_long': {
        'path' :'/pfs/work9/workspace/scratch/ma_lilipper-lippert_bachelorthesis_ws/final_eval/b_cv/convnext_tiny_pretrained_cn_wrapper_load_once_more_repeats_2071823/convnext_tiny_bfloat16_251107_1029/split_004/best.pt',
        'classifier' : 'convnext_tiny',
        'adapter' : 'old_cn_wrapper',
    },
    #vit b32
    'vit_b32_pretrained_cn': {
        'path' :'/pfs/work9/workspace/scratch/ma_lilipper-lippert_bachelorthesis_ws/final_eval/b_cv/vit_b_32_pretrained_cn_wrapper_2011485/vit_b32_bfloat16_251031_0810/split_025/best.pt',
        'classifier' : 'vit_b_32',
        'adapter' : 'old_cn_wrapper',
    },
    'vit_b32_pretrained_cn_once': {
        'path' :'/pfs/work9/workspace/scratch/ma_lilipper-lippert_bachelorthesis_ws/final_eval/b_cv/vit_b_32_pretrained_cn_wrapper_load_once_2033952/vit_b32_bfloat16_251102_0114/split_004/best.pt',
        'classifier' : 'vit_b_32',
        'adapter' : 'old_cn_wrapper',
    },
    'vit_b32_pretrained_cn_once_long': {
        'path' :'/pfs/work9/workspace/scratch/ma_lilipper-lippert_bachelorthesis_ws/final_eval/b_cv/vit_b_32_pretrained_cn_wrapper_load_once_more_repeats_2071826/vit_b32_bfloat16_251107_1451/split_038/best.pt',
        'classifier' : 'vit_b_32',
        'adapter' : 'old_cn_wrapper',
    },
    #resnet50
    'resnet50_pretrained_cn': {
        'path' :'/pfs/work9/workspace/scratch/ma_lilipper-lippert_bachelorthesis_ws/final_eval/b_cv/resnet50_train_pretrained_cn_wrapper_2017199/resnet50_bfloat16_251031_0810/split_012/best.pt',
        'classifier' : 'resnet50',
        'adapter' : 'old_cn_wrapper',
    },
    'resnet50_pretrained_cn_once': {
        'path' :'/pfs/work9/workspace/scratch/ma_lilipper-lippert_bachelorthesis_ws/final_eval/b_cv/resnet50_train_pretrained_cn_wrapper_load_once_2033950/resnet50_bfloat16_251102_0114/split_017/best.pt',
        'classifier' : 'resnet50',
        'adapter' : 'old_cn_wrapper',
    },
    'resnet50_pretrained_cn_once_long': {
        'path' :'/pfs/work9/workspace/scratch/ma_lilipper-lippert_bachelorthesis_ws/final_eval/b_cv/resnet50_train_pretrained_cn_wrapper_load_once_more_repeats_2071824/resnet50_bfloat16_251107_1450/split_007/best.pt',
        'classifier' : 'resnet50',
        'adapter' : 'old_cn_wrapper',
    },
    }

def main(pretrained_model_name):
    print(f"Starting pipeline evaluation with model: {pretrained_model_name}")
    model_info = models[pretrained_model_name]
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


for model_name in models.keys():
    print(f"Evaluating model: {model_name}")
    main(model_name)