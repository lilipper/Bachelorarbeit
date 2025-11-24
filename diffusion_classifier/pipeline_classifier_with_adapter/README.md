# This is the pipeline to test and evaluate models and results.

## `show_results.py`

This script is a small convenience wrapper around `eval_pipeline.py`.  
It lets you evaluate one of the preconfigured models with a single command.

All available models are defined in the `models` dictionary inside the script.  
Each entry specifies:
- the checkpoint path,
- the classifier type (e.g. `diffusion`, `resnet50`, `vit_b_16`, `vit_b_32`, `convnext_tiny`),
- the adapter type (`cn_wrapper`, `old_cn_wrapper`, or `latent`).

Internally, the script builds the argument object and calls `eval_pipeline.main(...)` on:
- the `thz_for_adapter` dataset,
- the `test` split,
- with `prompts/thz_prompts.csv` as class definition file,
- and writes all outputs into `./results_eval_pipeline`.

### Usage

```bash
# Example: evaluate a diffusion-based latent adapter model
python show_results.py \
    --pretrained_model_name dc_latent_dropout

# Example: evaluate a ConvNeXt-Tiny baseline with CN-wrapper
python show_results.py \
    --pretrained_model_name convnext_tiny_pretrained_cn


## `eval_pipeline.py`

This script is a unified evaluation entry point for both baseline classifiers and the diffusion-based classifier.

Given a pretrained checkpoint, it rebuilds the corresponding model and adapter, runs inference on a chosen dataset split, and stores per-sample predictions together with visualization outputs (reconstructed images and Grad-CAM-style saliency maps). Afterwards, it optionally aggregates the results using `eval_the_pipeline_results.py`.

### Features

- Supports:
  - Torchvision backbones (`resnet50`, `vit_b_16`, `vit_b_32`, `convnext_tiny`) with THz adapters
  - Diffusion-based zero-shot classifier (Stable Diffusion + ControlNet + latent adapter)
- Uses different adapters to map THz volumes to RGB images or latents:
  - `cn_wrapper`, `old_cn_wrapper`, `latent`
- Saves:
  - Per-sample prediction `.pt` files (`pred`, `label`, optionally `errors`)
  - Reconstructed images and Grad-CAM overlays
- Can be combined with `eval_the_pipeline_results.py` to produce confusion matrices and reports

### Usage

```bash
# Diffusion zero-shot classifier with latent adapter
python eval_pipeline.py \
    --pretrained_path /path/to/dc_checkpoint.pt \
    --classifier diffusion \
    --adapter latent \
    --dataset thz_for_adapter \
    --split test \
    --output_dir ./results_eval_pipeline

# Baseline ViT-B/32 model with ControlNet-based adapter
python eval_pipeline.py \
    --pretrained_path /path/to/baseline_checkpoint.pt \
    --classifier vit_b_32 \
    --adapter cn_wrapper \
    --dataset thz_for_adapter \
    --split test \
    --output_dir ./results_eval_pipeline
```

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
```

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
```
