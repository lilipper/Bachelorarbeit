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


def evaluate_predictions(folder_path: str, prompts_csv_path: str, output_dir: str):
    """
    Reads prediction files from a folder, computes evaluation metrics,
    prints a formatted report, and saves a confusion matrix heatmap.
    """
    # --- 1) Load data ---
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
            all_preds.append(data["pred"])
            all_labels.append(data["label"])
        except Exception as e:
            print(f"\nWarning: Could not load or parse file {f}. Error: {e}")

    if not all_labels:
        print("Error: Could not extract any valid label data from the files.")
        return

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)

    # --- 2) Load class names ---
    class_names = None
    try:
        prompts_df = pd.read_csv(prompts_csv_path)
        missing_cols = {c for c in ["classidx", "classname"] if c not in prompts_df.columns}
        if missing_cols:
            print(f"Warning: prompts CSV is missing columns: {sorted(missing_cols)}. "
                  f"Falling back to numeric class labels.")
        else:
            # Unique class names in correct order
            class_names = (prompts_df
                           .sort_values("classidx")
                           .drop_duplicates("classidx")["classname"]
                           .tolist())
    except FileNotFoundError:
        print(f"Error: prompts.csv not found at '{prompts_csv_path}'.")
        print("Class names will not be displayed; using numeric labels.")

    if class_names is None:
        class_names = [str(i) for i in sorted(np.unique(y_true))]

    # --- 3) Compute metrics ---
    accuracy = accuracy_score(y_true, y_pred)
    balanced_acc = balanced_accuracy_score(y_true, y_pred)
    try:
        report = classification_report(y_true, y_pred, target_names=class_names, digits=3)
    except Exception:
        # Fallback in case the number of classes doesn't match target_names
        report = classification_report(y_true, y_pred, digits=3)
    cm = confusion_matrix(y_true, y_pred)

    # --- 4) Pretty print results ---
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

    # --- 5) Visualize & save confusion matrix ---
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
    cm_path = os.path.join(output_dir, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=300)
    cm_path_pdf = os.path.join(output_dir, "confusion_matrix.pdf")
    plt.savefig(cm_path_pdf, bbox_inches="tight", pad_inches=0)
    print(f" Confusion matrix saved to: {cm_path}")

    # Optional: also save the text report for later reference
    rep_path = os.path.join(output_dir, "classification_report.txt")
    with open(rep_path, "w", encoding="utf-8") as f:
        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write(f"Balanced Accuracy: {balanced_acc:.4f}\n\n")
        f.write(report)
    print(f" Classification report saved to: {rep_path}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluates saved predictions and creates a detailed report plus a confusion matrix."
    )
    parser.add_argument(
        "folder",
        type=str,
        help="Path to the folder containing .pt result files."
    )
    parser.add_argument(
        "--prompts_csv",
        type=str,
        required=True,
        help="Path to the prompts.csv file to load class names."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Optional path to save the confusion matrix and report. Defaults to the input folder."
    )
    args = parser.parse_args()

    output_directory = args.output_dir if args.output_dir else args.folder
    evaluate_predictions(args.folder, args.prompts_csv, output_directory)


if __name__ == "__main__":
    main()
