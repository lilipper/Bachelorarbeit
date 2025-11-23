"""
evaluate_predictions.py

Load saved prediction tensors from a folder, compute standard classification
metrics, and save a confusion matrix plus a text report.

The script expects `.pt` files that contain at least:
    - "pred" or "preds" : predicted class indices (tensor or scalar)
    - "label"          : ground-truth class indices

Example usage
-------------

# Minimal usage: evaluate all .pt files in a folder and write results
# back into the same folder
python evaluate_predictions.py \
    /path/to/results_folder

# With class names from prompts.csv and custom output directory
python evaluate_predictions.py \
    /path/to/results_folder \
    --prompts_csv /path/to/prompts.csv \
    --output_dir /path/to/output_dir


Command line arguments
----------------------

Positional:
- folder        : Path to the folder containing `.pt` prediction files.
                  Each file must store a dict with keys:
                  - "pred"  or "preds"
                  - "label"

Optional:
- --prompts_csv : Optional CSV to map class indices to human-readable names.
                  The file is expected to contain columns:
                  - "classidx"
                  - "classname"
                  If missing, numeric labels are used instead.

- --output_dir  : Directory to store the evaluation results.
                  If not provided, the input folder is used.

What the script does
--------------------

1. Scans the given folder for all `.pt` files.
2. Loads each file and extracts predictions and labels.
3. Computes:
   - Accuracy
   - Balanced accuracy
   - Full sklearn classification report
   - Confusion matrix

4. Prints a summary to stdout.

5. Saves into the chosen output directory:
   - "confusion_matrix.png"    : heatmap of the confusion matrix
   - "confusion_matrix.pdf"    : same as vector graphic
   - "classification_report.txt":
       - accuracy
       - balanced accuracy
       - detailed classification report
"""


import argparse
import os
from pathlib import Path
import torch
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, balanced_accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm


def evaluate_predictions(folder_path: str, output_dir: str, prompts_csv_path: str = None):
    try:
        files = sorted([f for f in os.listdir(folder_path) if f.endswith(".pt")])
        if not files:
            print(f"Error: No .pt files found in folder '{folder_path}'.")
            return
    except FileNotFoundError:
        print(f"Error: The specified folder '{folder_path}' does not exist.")
        return

    all_preds = []
    all_labels = []
    for f in tqdm(files, desc="Loading results"):
        try:
            data = torch.load(os.path.join(folder_path, f), map_location=torch.device("cpu"))
            if "preds" in data:
                all_preds.append(data["preds"])
            else:
                all_preds.append(data["pred"])
            all_labels.append(data["label"])
        except Exception as e:
            print(f"\nWarning: Could not load or parse file {f}. Error: {e}")

    if not all_labels:
        print("Error: Could not extract any valid label data from the files.")
        return

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)

    class_names = None
    if prompts_csv_path:
        try:
            df = pd.read_csv(prompts_csv_path)
            if "classidx" in df.columns and "classname" in df.columns:
                class_names = (
                    df.sort_values("classidx")
                    .drop_duplicates("classidx")["classname"]
                    .tolist()
                )
            else:
                print("Warning: Missing classidx/classname columns. Using numeric labels.")
        except FileNotFoundError:
            print(f"Warning: prompts.csv not found at '{prompts_csv_path}'. Using numeric labels.")

    if class_names is None:
        class_names = [str(i) for i in sorted(np.unique(y_true))]

    accuracy = accuracy_score(y_true, y_pred)
    balanced_acc = balanced_accuracy_score(y_true, y_pred)
    try:
        report = classification_report(y_true, y_pred, target_names=class_names, digits=3)
    except Exception:
        report = classification_report(y_true, y_pred, digits=3)
    cm = confusion_matrix(y_true, y_pred)

    print("\n" + "=" * 60)
    print(" E V A L U A T I O N   R E P O R T")
    print("=" * 60)
    print(f" Results folder: {folder_path}")
    print(f" Number of evaluated samples: {len(y_true)}")
    print("-" * 60)
    print(f" Accuracy: {accuracy:.2%}")
    print(f" Balanced Accuracy: {balanced_acc:.2%}")
    print("-" * 60)
    print(" Detailed classification report:")
    print(report)
    print("=" * 60)

    plt.figure(figsize=(max(8, len(class_names)), max(6, int(len(class_names) * 0.8))))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        annot_kws={"size": 12},
    )
    plt.ylabel("True class", fontsize=14)
    plt.xlabel("Predicted class", fontsize=14)
    plt.title("Confusion Matrix", fontsize=16)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    img_path = os.path.join(output_dir, f"confusion_matrix.png")
    pdf_path = os.path.join(output_dir, f"confusion_matrix.pdf")
    plt.savefig(img_path, dpi=300)
    plt.savefig(pdf_path, bbox_inches="tight", pad_inches=0)

    rep_path = os.path.join(output_dir, "classification_report.txt")
    with open(rep_path, "w", encoding="utf-8") as f:
        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write(f"Balanced Accuracy: {balanced_acc:.4f}\n\n")
        f.write(report)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluates saved predictions and creates a detailed report plus a confusion matrix."
    )
    parser.add_argument("folder", type=str)
    parser.add_argument("--prompts_csv", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()

    output_dir = args.output_dir if args.output_dir else args.folder
    evaluate_predictions(args.folder, args.prompts_csv, output_dir)


if __name__ == "__main__":
    main()

