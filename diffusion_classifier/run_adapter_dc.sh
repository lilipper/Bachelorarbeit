#!/bin/bash
#SBATCH --job-name=dc-adapter-training-with-feedback
#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --mem=40GB
#SBATCH --cpus-per-task=16
#SBATCH --output=logs/train_adapter_with_dc_%j.out
#SBATCH --error=logs/train_adapter_with_dc_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=linus.lippert@students.uni-mannheim.de

echo "=== Job gestartet: $(date) ==="
echo "Job-ID: $SLURM_JOB_ID auf Node: $(hostname)"
export PYTHONPATH="$(pwd):$PYTHONPATH"

set -euo pipefail

# init_conda
# conda activate diffusion-classifier

# Stabilere CUDA-Allocator-Einstellungen (hilft gegen Fragmentierung)
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:512"

cd "$SLURM_SUBMIT_DIR"

echo "=== GPU-Check ==="
nvidia-smi || true
python - <<'PY'
try:
    import torch
    print("torch.cuda.is_available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("device:", torch.cuda.get_device_name(0))
        print("dtype half supported:", torch.cuda.get_device_capability())
except Exception as e:
    print("Torch check error:", e)
PY

# Dein Python-Skript aufrufen
python adapter/train_adapter_with_dc.py \
  --data_train /pfs/work9/workspace/scratch/ma_lilipper-lippert_bachelorthesis_ws/thz_dataset/train \
  --data_test /pfs/work9/workspace/scratch/ma_lilipper-lippert_bachelorthesis_ws/thz_dataset/test \
  --train_csv /pfs/work9/workspace/scratch/ma_lilipper-lippert_bachelorthesis_ws/Bachelorarbeit/jsons/train_labels.csv \
  --val_csv   /pfs/work9/workspace/scratch/ma_lilipper-lippert_bachelorthesis_ws/Bachelorarbeit/jsons/test_labels.csv \
  --prompts_csv /pfs/work9/workspace/scratch/ma_lilipper-lippert_bachelorthesis_ws/Bachelorarbeit/diffusion_classifier/prompts/thz_prompts.csv \
  --n_samples 2 4 8 \
  --to_keep   3 2 1 \
  --epochs 200 \
  --batch_size 2 \
  --lr 1e-4 \
  --dtype bfloat16  \
  --use_xformers

echo "=== Job beendet: $(date) ==="