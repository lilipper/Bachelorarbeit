# This is the pipeline to test and evaluate models and results.

## show_results.py
The easiest way is to use show_results.py
Here you can select one pretrained model and you get the results in return.


## eval_the_pipeline_results.py

This script loads `.pt` prediction files from a folder, evaluates them, and produces a full classification report including accuracy, balanced accuracy, and a confusion matrix.  
All results are saved into an output directory as images (PNG + PDF) and a text report.

### Features

- Automatically loads all `.pt` files in a folder
- Supports prediction dictionaries containing either `"pred"` or `"preds"`
- Extracts:
  - Ground truth labels
  - Predicted labels
- Computes:
  - Accuracy
  - Balanced accuracy
  - Full classification report
  - Confusion matrix
- Saves:
  - `confusion_matrix.png`  
  - `confusion_matrix.pdf`  
  - `classification_report.txt`
- Optional: load class names from a `prompts.csv` file (expects columns `classidx`, `classname`)

### Usage

Run the script with:

```bash
python evaluate_predictions.py /path/to/results \
    --prompts_csv /path/to/prompts.csv \
    --output_dir /path/to/output


## extract_learning_curve_from_log.py

This script parses a single training log file and extracts learning curves (loss and accuracy per epoch).  
The results are exported as a CSV file and plotted both as PNG and PDF.

### Features

- Detects epoch boundaries using log lines like:  
  `"[Epoch] 12/700"`
- Extracts:
  - `epoch`
  - `train_loss`
  - `train_acc`
- Outputs:
  - `learning_curve.csv`
  - `train_loss.png` & `train_loss.pdf`
  - `train_acc.png` & `train_acc.pdf`
- Works directly with `.out` and `.err` files (e.g., Slurm logs)

### Usage

Run it from the terminal:

```bash
python extract_learning_curve_from_log.py \
    --log /path/to/logfile.out \
    --outdir output_folder

