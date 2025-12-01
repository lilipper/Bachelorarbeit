#!/bin/bash
#SBATCH --job-name=dc_eval
#SBATCH --output=gradcam_logs/dc_eval_%A_%a.out
#SBATCH --error=gradcam_logs/dc_eval_%A_%a.err
#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:2
#SBATCH --array=0-2
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
  "train_dc_dropout_2d_2112518"
  "train_dc_with_original_cn_multichannel_load_once_2051652"
  "train_dc_with_original_cn_multichannel_2036332"
)

MODEL_NAME=${models[$SLURM_ARRAY_TASK_ID]}

echo "Running diffusion model: ${MODEL_NAME}"
python pipeline_classifier_with_adapter/show_results.py \
    --pretrained_model_name "${MODEL_NAME}"
