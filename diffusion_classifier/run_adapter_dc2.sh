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
python adapter/train_adapter_with_dc_2.py \
  --n_samples 8 4 2 \
  --to_keep   3 2 1 \
  --epochs 20 \
  --batch_size 2 \
  --lr 1e-4 \
  --dtype float16 \
  --use_xformers

echo "=== Job beendet: $(date) ==="