#!/bin/bash
#SBATCH --job-name=ctrlnet-train
#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --mem=55GB
#SBATCH --cpus-per-task=16
#SBATCH --output=logs/ctrlnet_%j.out
#SBATCH --error=logs/ctrlnet_%j.err

set -euo pipefail
echo "=== Job start: $(date) on $(hostname) ==="
echo "SLURM_JOB_ID=$SLURM_JOB_ID  PARTITION=$SLURM_JOB_PARTITION  CPUS=$SLURM_CPUS_PER_TASK"

# (Falls nötig) Conda aktivieren – falls deine Umgebung im Batch nicht automatisch aktiv ist:
# source ~/.bashrc
# conda activate <deine_env>

cd "$SLURM_SUBMIT_DIR"

echo "=== GPU check ==="
nvidia-smi || true
python - <<'PY'
try:
    import torch
    print("torch.cuda.is_available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("CUDA device:", torch.cuda.get_device_name(0))
except Exception as e:
    print("Torch check error:", e)
PY

# Training starten
python tutorial_train_sd21.py

echo "=== Job end: $(date) ==="