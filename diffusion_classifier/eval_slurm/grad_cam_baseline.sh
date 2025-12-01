#!/bin/bash
#SBATCH --job-name=base_eval
#SBATCH --output=gradcam_logs/base_eval_%A_%a.out
#SBATCH --error=gradcam_logs/base_eval_%A_%a.err
#SBATCH --array=0-22   
#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --mem=50GB
#SBATCH --cpus-per-task=16
#SBATCH --mail-type=ALL
#SBATCH --mail-user=linus.lippert@students.uni-mannheim.de

echo "=== Job gestartet: $(date) ==="
echo "Job-ID: $SLURM_JOB_ID auf Node: $(hostname)"
export PYTHONPATH="$(pwd):$PYTHONPATH"

set -euo pipefail

echo "=== GPU-Check ==="
nvidia-smi || true

models=(
  # Dropout-Modelle
  "vitb32_dropout_pretrained_2_2153738"
  "vitb32_dropout_untrained_2109807"
  "vitb32_dropout_pretrained_2_2153739"

  "convnext_tiny_dropout_pretrained_2_2153749"
  "convnext_tiny_dropout_untrained_2109804"

  "vitb16_dropout_pretrained_2_2153753"
  "vitb16_dropout_pretrained_2_2153747"

  "resnet50_dropout_pretrained_2_2153748"
  "resnet50_dropout_pretrained_2_2153754"
  "resnet50_dropout_untrained_2109805"

  # CV-Modelle (old_cn_wrapper)
  "vitb16_pretrained_cn_wrapper_2050294"
  "vitb16_pretrained_cn_wrapper_load_once_2051710"
  "vitb16_pretrained_cn_wrapper_load_once_more_repeats_2071825"
  "vitb16_pretrained_cn_wrapper_load_once_more_repeats_2071825_50"

  "convnext_tiny_pretrained_cn_wrapper_2017198"
  "convnext_tiny_pretrained_cn_wrapper_load_once_2033949"
  "convnext_tiny_pretrained_cn_wrapper_load_once_more_repeats_2071823"

  "vitb32_pretrained_cn_wrapper_2011485"
  "vit_b_32_pretrained_cn_wrapper_load_once_2033952"
  "vitb32_pretrained_cn_wrapper_load_once_more_repeats_2071826"

  "resnet50_train_pretrained_cn_wrapper_2017199"
  "resnet50_train_pretrained_cn_wrapper_load_once_2033950"
  "resnet50_train_pretrained_cn_wrapper_load_once_more_repeats_2071824"
)

MODEL_NAME=${models[$SLURM_ARRAY_TASK_ID]}

echo "Running base model: ${MODEL_NAME}"
python pipeline_classifier_with_adapter/show_results.py \
    --pretrained_model_name "${MODEL_NAME}"