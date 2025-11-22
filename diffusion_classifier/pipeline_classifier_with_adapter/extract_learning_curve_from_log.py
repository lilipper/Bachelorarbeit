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
    

