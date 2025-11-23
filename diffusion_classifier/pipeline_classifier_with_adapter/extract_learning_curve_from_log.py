"""
extract_learning_curve_from_log.py

Parse a training log file, extract epoch-wise training loss and accuracy,
save them as a CSV file, and plot learning curves (loss and accuracy)
as PNG and PDF.

Example usage
-------------

# Basic usage with default output directory ("./log_curves")
python extract_learning_curve_from_log.py \
    --log /path/to/logfile.out

# Custom output directory
python extract_learning_curve_from_log.py \
    --log /path/to/logfile.err \
    --outdir ./results/log_curves


Command line arguments
----------------------

Required:
- --log      : Path to the training log file (e.g. *.out / *.err).
               The script expects lines of the form:
                 "[Epoch] 12/700"
                 "[train_one_epoch] Done. epoch_loss=0.9026  epoch_acc=0.4688"

Optional:
- --outdir   : Root output directory for CSV and plots.
               Default: "./log_curves"

What the script does
--------------------

1. Parses the log and looks for:
   - epoch lines: "[Epoch] <current>/<max>"
   - metric lines: "... epoch_loss=<float>  epoch_acc=<float>"

2. Builds a pandas DataFrame with columns:
   - epoch       : current epoch index
   - max_epoch   : total number of epochs from the log
   - train_loss  : training loss at the end of the epoch
   - train_acc   : training accuracy at the end of the epoch

3. Creates a subfolder inside --outdir that is derived from the log filename
   (dots replaced with underscores, ".out"/".err" removed).

4. Saves:
   - learning_curve.csv      : epoch, train_loss, train_acc
   - train_loss.png / .pdf   : loss over epochs
   - train_acc.png / .pdf    : accuracy over epochs
"""


import re
from pathlib import Path
import os
import pandas as pd
import matplotlib.pyplot as plt
import argparse

# Regex passend zu deinem Logformat:
EPOCH_RE = re.compile(r"\[Epoch\]\s*(\d+)\s*/\s*(\d+)")
TRAIN_METRIC_RE = re.compile(r"epoch_loss=([0-9.]+)\s+epoch_acc=([0-9.]+)")


def parse_log(path: Path) -> pd.DataFrame:
    current_epoch = None
    max_epoch = None
    records = []

    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            # Epoch detection
            m_epoch = EPOCH_RE.search(line)
            if m_epoch:
                current_epoch = int(m_epoch.group(1))
                max_epoch = int(m_epoch.group(2))
                continue

            # Train metrics
            m_train = TRAIN_METRIC_RE.search(line)
            if m_train and current_epoch is not None:
                rec = {
                    "epoch": current_epoch,
                    "max_epoch": max_epoch,
                    "train_loss": float(m_train.group(1)),
                    "train_acc": float(m_train.group(2)),
                }
                records.append(rec)

    if not records:
        raise ValueError("Keine passenden Einträge im Log gefunden.")

    df = pd.DataFrame(records).sort_values("epoch")
    return df


def save_plot(fig, outdir: Path, name: str):
    """Speichert eine Matplotlib-Figur als PNG und PDF."""
    fig.savefig(outdir / f"{name}.png", dpi=200)
    fig.savefig(outdir / f"{name}.pdf")
    plt.close(fig)


def plot_curves(df: pd.DataFrame, outdir: Path):
    outdir.mkdir(parents=True, exist_ok=True)

    # Loss
    fig_loss = plt.figure()
    plt.plot(df["epoch"], df["train_loss"])
    plt.xlabel("Epoch")
    plt.ylabel("Train Loss")
    plt.title("Learning Curve – Train Loss")
    plt.tight_layout()
    save_plot(fig_loss, outdir, "train_loss")

    # Accuracy
    fig_acc = plt.figure()
    plt.plot(df["epoch"], df["train_acc"])
    plt.xlabel("Epoch")
    plt.ylabel("Train Accuracy")
    plt.title("Learning Curve – Train Accuracy")
    plt.tight_layout()
    save_plot(fig_acc, outdir, "train_acc")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", required=True, help="Pfad zum Logfile (*.out/*.err)")
    parser.add_argument("--outdir", default=r"./log_curves",
                        help="Output-Verzeichnis für CSV und Plots")
    return parser.parse_args()

def main(log,outdir):
    

    log_path = Path(log)
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    outdir = Path(os.path.join(outdir, os.path.basename(log_path).replace('.out', '').replace('.err', '').replace('.', '_')))

    df = parse_log(log_path)

    # CSV
    csv_path = outdir / "learning_curve.csv"
    outdir.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    print(f"CSV gespeichert: {csv_path}")

    # Plots (PNG + PDF)
    plot_curves(df, outdir)
    print(f"Plots gespeichert in {outdir} (PNG + PDF)")


if __name__ == "__main__":
    args = parse_args()
    main(args.log,args.outdir)
    

