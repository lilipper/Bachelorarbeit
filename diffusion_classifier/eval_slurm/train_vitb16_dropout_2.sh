#!/bin/bash
#SBATCH --job-name=vitb16_dropout_pretrained-SGD
#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --mem=40GB
#SBATCH --cpus-per-task=16
#SBATCH --output=final_logs/vitb16_dropout_pretrained_SGD_%j.out
#SBATCH --error=final_logs/vitb16_dropout_pretrained_SGD_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=linus.lippert@students.uni-mannheim.de

echo "=== Job gestartet: $(date) ==="
echo "Job-ID: $SLURM_JOB_ID auf Node: $(hostname)"
export PYTHONPATH="$(pwd):$PYTHONPATH"

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"

########## 🔧 Pfade (anpassen, falls nötig)
WORKSPACE_BASE="/pfs/work9/workspace/scratch/ma_lilipper-lippert_bachelorthesis_ws"
INPUT_TRAIN_DIR="$WORKSPACE_BASE/thz_dataset/train"
INPUT_TEST_DIR="$WORKSPACE_BASE/thz_dataset/test"
INPUT_LABELS_DIR="$WORKSPACE_BASE/Bachelorarbeit/jsons"
OUTPUT_BASE="$WORKSPACE_BASE/final_eval"                       # Ziel für Ergebnisse auf Workspace
RUN_NAME="vitb16_dropout_pretrained_SGD_${SLURM_JOB_ID}"
OUTPUT_DIR="$OUTPUT_BASE/$RUN_NAME"

########## 🧊 Lokales SSD-Arbeitsverzeichnis ($TMPDIR)
: "${TMPDIR:?TMPDIR muss vom System gesetzt sein}"
WORKDIR="$TMPDIR/job_${SLURM_JOB_ID}"
LCL_INPUT="$WORKDIR/input"
LCL_RESULTS="$WORKDIR/results"
mkdir -p "$LCL_INPUT/train" "$LCL_INPUT/test" "$LCL_INPUT/jsons" "$LCL_RESULTS"

########## 🧮 Platz prüfen (grobe Schätzung: train+test+labels)
need_bytes=$(( $(du -sb "$INPUT_TRAIN_DIR" | awk '{print $1}') \
             + $(du -sb "$INPUT_TEST_DIR"  | awk '{print $1}') \
             + $(du -sb "$INPUT_LABELS_DIR" | awk '{print $1}') ))
free_bytes=$(df -PB1 "$TMPDIR" | tail -1 | awk '{print $4}')
echo "[INFO] Benötigt: $need_bytes B, Frei: $free_bytes B auf $TMPDIR"
if (( free_bytes <= need_bytes )); then
  echo "[ERROR] Nicht genug Platz auf $TMPDIR. Reduziere Datenmenge oder splitte den Job."
  exit 2
fi

########## ⏬ Daten → lokale SSD kopieren (minimiert zentrale FS-Reads)
echo "[INFO] Kopiere Daten nach lokale SSD ..."
rsync -a --info=stats2,progress2 "$INPUT_TRAIN_DIR/" "$LCL_INPUT/train/"
rsync -a --info=stats2,progress2 "$INPUT_TEST_DIR/"  "$LCL_INPUT/test/"
rsync -a --info=stats2,progress2 "$INPUT_LABELS_DIR/" "$LCL_INPUT/jsons/"

########## 📦 (Optional) Umgebung lokal machen, um HOME-Zugriffe zu vermeiden
# module load <python/pytorch>, ODER:
# rsync -a --delete "$WORKSPACE_BASE/.venv/" "$WORKDIR/.venv/" && source "$WORKDIR/.venv/bin/activate"

########## 🛡️ Ergebnisse zuverlässig zurückkopieren – auch bei Abbruch
sync_back() {
  echo "[INFO] Kopiere Ergebnisse zurück nach $OUTPUT_DIR ..."
  mkdir -p "$OUTPUT_DIR"
  rsync -a --info=stats2,progress2 "$LCL_RESULTS/" "$OUTPUT_DIR/" || echo "[WARN] Rückkopieren teilweise fehlgeschlagen."
}
trap sync_back EXIT

########## CUDA/torch Settings (wie gehabt)
unset PYTORCH_CUDA_ALLOC_CONF
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128"

########## GPU-Check (optional)
echo "=== GPU-Check ==="
nvidia-smi || true
python - <<'PY'
import torch, os
print("torch.cuda.is_available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))
    print("capability:", torch.cuda.get_device_capability())
print("PYTORCH_CUDA_ALLOC_CONF:", os.environ.get("PYTORCH_CUDA_ALLOC_CONF"))
if torch.cuda.is_available():
    x = torch.randn(1,8,48,48,48, device="cuda", dtype=torch.float16)
    m = torch.nn.Conv3d(8,16,3,padding=1).cuda().half()
    y = m(x); print("3D conv smoke test OK:", y.shape)
PY

########## 🚀 Training – liest/schreibt NUR auf lokaler SSD
# WICHTIG: Pfade auf $LCL_INPUT und $LCL_RESULTS umbiegen
python adapter_multichannel/train_baseline_cn_without_cv_and_dropout_2.py \
  --data_train "$LCL_INPUT/train" \
  --data_test  "$LCL_INPUT/test"  \
  --train_csv  "$LCL_INPUT/jsons/train_labels.csv" \
  --val_csv    "$LCL_INPUT/jsons/test_labels.csv"  \
  --backbone vit_b16 \
  --epochs 600 \
  --dropout_p 0.1 \
  --batch_size 2 \
  --dtype bfloat16 \
  --learn_front \
  --pretrained \
  --train_backbone \
  --final_eval \
  --save_dir "$LCL_RESULTS"

echo "=== Job beendet: $(date) ==="