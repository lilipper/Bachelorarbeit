#!/bin/bash
#SBATCH --job-name=thz-both-trainings
#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --mem=60GB
#SBATCH --cpus-per-task=16
#SBATCH --output=logs/both_%j.out
#SBATCH --error=logs/both_%j.err

set -euo pipefail

echo "=== Job gestartet: $(date) ==="
echo "Job-ID: $SLURM_JOB_ID  Host: $(hostname)  PWD: $PWD"

# (optional) Falls Conda im Batch nicht automatisch aktiv ist:
# source ~/.bashrc
# conda activate diffusion-classifier

# Stabileres CUDA-Alloc-Handling
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:512"

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs runs

echo "=== GPU-Check ==="
nvidia-smi || true
python - <<'PY'
import torch
print("torch.cuda.is_available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))
PY

#########################
# Training 1: Adapter DC
#########################
echo "=== [1/2] Adapter+DC startet: $(date) ==="
python adapter/train_adapter_with_dc.py \
  --data_train /pfs/work9/workspace/scratch/ma_lilipper-lippert_bachelorthesis_ws/thz_dataset/train \
  --data_test  /pfs/work9/workspace/scratch/ma_lilipper-lippert_bachelorthesis_ws/thz_dataset/test \
  --train_csv  /pfs/work9/workspace/scratch/ma_lilipper-lippert_bachelorthesis_ws/Bachelorarbeit/jsons/train_labels.csv \
  --val_csv    /pfs/work9/workspace/scratch/ma_lilipper-lippert_bachelorthesis_ws/Bachelorarbeit/jsons/test_labels.csv \
  --prompts_csv /pfs/work9/workspace/scratch/ma_lilipper-lippert_bachelorthesis_ws/Bachelorarbeit/diffusion_classifier/prompts/thz_prompts.csv \
  --n_samples 8 4 2 \
  --to_keep   3 2 1 \
  --epochs 20 \
  --batch_size 2 \
  --lr 1e-4 \
  --dtype float16 \
  --use_xformers 2>&1 | tee "logs/adapter_dc_${SLURM_JOB_ID}.log"

echo "=== [1/2] Adapter+DC fertig: $(date) ==="

#####################################
# Training 2: ControlNet mit Bildern
#####################################
echo "=== [2/2] ControlNet startet: $(date) ==="
python adapter/training_ControlNet_with_images.py \
  --dataset_jsonl /pfs/work9/workspace/scratch/ma_lilipper-lippert_bachelorthesis_ws/Bachelorarbeit/jsons/train_prompt_one_target.json \
  --output_dir ./runs/thz_controlnet_sd21 \
  --resolution 512 \
  --train_batch_size 2 \
  --gradient_accumulation_steps 4 \
  --learning_rate 1e-5 \
  --num_train_epochs 10 \
  --enable_xformers 2>&1 | tee "logs/controlnet_${SLURM_JOB_ID}.log"

echo "=== [2/2] ControlNet fertig: $(date) ==="
echo "=== Job beendet: $(date) ==="
