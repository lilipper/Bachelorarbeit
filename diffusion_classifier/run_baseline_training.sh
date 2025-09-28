#!/bin/bash
#SBATCH --job-name=baseline-training
#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --mem=40GB
#SBATCH --cpus-per-task=16
#SBATCH --output=logs/baseline-training_%j.out
#SBATCH --error=logs/baseline-training_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=linus.lippert@students.uni-mannheim.de

echo "=== Job gestartet: $(date) ==="
echo "Job-ID: $SLURM_JOB_ID auf Node: $(hostname)"
export PYTHONPATH="$(pwd):$PYTHONPATH"

set -euo pipefail

# init_conda
# conda activate diffusion-classifier

# Stabilere CUDA-Allocator-Einstellungen (hilft gegen Fragmentierung)
# export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:512"
unset PYTORCH_CUDA_ALLOC_CONF
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128"

cd "$SLURM_SUBMIT_DIR"

echo "=== GPU-Check ==="
nvidia-smi || true
python - <<'PY'
import torch, os
print("torch.cuda.is_available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))
    print("capability:", torch.cuda.get_device_capability())
print("PYTORCH_CUDA_ALLOC_CONF:", os.environ.get("PYTORCH_CUDA_ALLOC_CONF"))
# schneller Smoke-Test einer 3D-Conv in half
if torch.cuda.is_available():
    x = torch.randn(1, 8, 48, 48, 48, device="cuda", dtype=torch.float16)
    m = torch.nn.Conv3d(8, 16, 3, padding=1).cuda().half()
    y = m(x)
    print("3D conv smoke test OK:", y.shape)
PY

# Dein Python-Skript aufrufen
python baseline/training_baseline_models.py \
--data_train /pfs/work9/workspace/scratch/ma_lilipper-lippert_bachelorthesis_ws/thz_dataset/train \
--data_test /pfs/work9/workspace/scratch/ma_lilipper-lippert_bachelorthesis_ws/thz_dataset/test \
--train_csv /pfs/work9/workspace/scratch/ma_lilipper-lippert_bachelorthesis_ws/Bachelorarbeit/jsons/train_labels.csv \
--val_csv   /pfs/work9/workspace/scratch/ma_lilipper-lippert_bachelorthesis_ws/Bachelorarbeit/jsons/test_labels.csv \
--epochs 20 \
--batch_size 2 \
--lr 1e-4 \
--dtype bfloat16 \
--train_all \
--final_eval

python baseline/training_baseline_models.py \
--data_train /pfs/work9/workspace/scratch/ma_lilipper-lippert_bachelorthesis_ws/thz_dataset/train \
--data_test /pfs/work9/workspace/scratch/ma_lilipper-lippert_bachelorthesis_ws/thz_dataset/test \
--train_csv /pfs/work9/workspace/scratch/ma_lilipper-lippert_bachelorthesis_ws/Bachelorarbeit/jsons/train_labels.csv \
--val_csv   /pfs/work9/workspace/scratch/ma_lilipper-lippert_bachelorthesis_ws/Bachelorarbeit/jsons/test_labels.csv \
--epochs 20 \
--batch_size 2 \
--lr 1e-4 \
--dtype bfloat16 \
--train_adapter \
--final_eval

python baseline/training_baseline_models.py \
--data_train /pfs/work9/workspace/scratch/ma_lilipper-lippert_bachelorthesis_ws/thz_dataset/train \
--data_test /pfs/work9/workspace/scratch/ma_lilipper-lippert_bachelorthesis_ws/thz_dataset/test \
--train_csv /pfs/work9/workspace/scratch/ma_lilipper-lippert_bachelorthesis_ws/Bachelorarbeit/jsons/train_labels.csv \
--val_csv   /pfs/work9/workspace/scratch/ma_lilipper-lippert_bachelorthesis_ws/Bachelorarbeit/jsons/test_labels.csv \
--backbone vit_b16 \
--epochs 20 \
--batch_size 2 \
--lr 1e-4 \
--dtype bfloat16 \
--train_all \
--final_eval

echo "=== Job beendet: $(date) ==="