'''
Script to automate some steps
'''

import shutil
import os
# from eval_the_pipeline_results import evaluate_predictions
# import torch, tqdm
# import  copy, time, json, argparse, random
# from pathlib import Path
# from typing import Tuple, Optional, List

# from eval_pipeline import main as pipeline


folders_cv_repeated = [ 
    r"/pfs/work9/workspace/scratch/ma_lilipper-lippert_bachelorthesis_ws/final_eval/a_cv/convnext_tiny_pretrained_cn_wrapper_2017198",
    r"/pfs/work9/workspace/scratch/ma_lilipper-lippert_bachelorthesis_ws/final_eval/a_cv/resnet50_train_pretrained_cn_wrapper_2017199",
    r"/pfs/work9/workspace/scratch/ma_lilipper-lippert_bachelorthesis_ws/final_eval/a_cv/vit_b_32_pretrained_cn_wrapper_2011485",
    r"/pfs/work9/workspace/scratch/ma_lilipper-lippert_bachelorthesis_ws/final_eval/a_cv/vitb16_pretrained_cn_wrapper_2050294",
]

folder_cv_one =[
    r"/pfs/work9/workspace/scratch/ma_lilipper-lippert_bachelorthesis_ws/final_eval/a_cv/convnext_tiny_pretrained_cn_wrapper_load_once_2033949",
    r"/pfs/work9/workspace/scratch/ma_lilipper-lippert_bachelorthesis_ws/final_eval/a_cv/resnet50_train_pretrained_cn_wrapper_load_once_2033950",
    r"/pfs/work9/workspace/scratch/ma_lilipper-lippert_bachelorthesis_ws/final_eval/a_cv/vit_b_32_pretrained_cn_wrapper_load_once_2033952",
    r"/pfs/work9/workspace/scratch/ma_lilipper-lippert_bachelorthesis_ws/final_eval/a_cv/vitb16_pretrained_cn_wrapper_load_once_2051710",
]

folder_cv_one_long =[
    r"/pfs/work9/workspace/scratch/ma_lilipper-lippert_bachelorthesis_ws/final_eval/a_cv/convnext_tiny_pretrained_cn_wrapper_load_once_more_repeats_2071823",
    r"/pfs/work9/workspace/scratch/ma_lilipper-lippert_bachelorthesis_ws/final_eval/a_cv/resnet50_train_pretrained_cn_wrapper_load_once_more_repeats_2071824",
    r"/pfs/work9/workspace/scratch/ma_lilipper-lippert_bachelorthesis_ws/final_eval/a_cv/vit_b_32_pretrained_cn_wrapper_load_once_more_repeats_2071826",
    r"/pfs/work9/workspace/scratch/ma_lilipper-lippert_bachelorthesis_ws/final_eval/a_cv/vitb16_pretrained_cn_wrapper_load_once_more_repeats_2071825",
]

all_folders = folder_cv_one + folders_cv_repeated + folder_cv_one_long

                
        



models = {
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
    }
    }

# def main(pretrained_model_name):
#     print(f"Starting pipeline evaluation with model: {pretrained_model_name}")
#     model_info = models[pretrained_model_name]
#     class FE: pass
#     p_args = FE()
#     p_args.pretrained_path = model_info['path']
#     p_args.classifier = model_info['classifier']
#     p_args.adapter = model_info['adapter']
#     p_args.output_dir = "./results_eval_pipeline"
#     p_args.split = "test"
#     p_args.n_workers = 1
#     p_args.dataset = "thz_for_adapter"
#     p_args.prompts_csv = '/pfs/work9/workspace/scratch/ma_lilipper-lippert_bachelorthesis_ws/Bachelorarbeit/diffusion_classifier/prompts/thz_prompts.csv' 

#     print(f"Using pretrained model from: {p_args.pretrained_path}")
#     pipeline(p_args)
#     print("Pipeline evaluation completed.")


for model_name in models.keys():
    print(f"Evaluating model: {model_name}")
    # main(model_name)