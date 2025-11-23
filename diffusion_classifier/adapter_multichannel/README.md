# Training Scripts Overview

This folder contains all training scripts used for the ControlNet-based adapter experiments (“cn_adapter”) and the Diffusion Classifier (“dc”). The scripts cover baseline setups as well as variants with cross-validation, dropout, and multichannel processing.

## File Structure and Purpose

### **Adapter Conversion**
- **`thz_to_vae_adapter.py`**  
  ### LatentMultiChannelAdapter

  The `LatentMultiChannelAdapter` is a compact temporal aggregation module used to convert a sequence of Stable Diffusion latents into a single 2D latent representation. It takes input of shape **(B, T, 4, H, W)** and outputs a compressed latent of shape **(B, 4, H, W)**.

### **Baseline Experiments (ControlNet Adapter)**
- **`train_baseline_cn_adapter.py`**  
  Baseline training script for the THz-to-RGB ControlNet adapter combined with a
  torchvision backbone (ResNet/ViT/ConvNeXt). For each CV split, the adapter
  front-end and backbone are reinitialized and trained independently, providing
  clean cross-validation baselines. The script saves the best checkpoint per
  split, writes a CV summary CSV, and can optionally run a final evaluation on
  a fixed test set using the globally best model.
  

- **`train_baseline_cn_adapter_load_once.py`**  
  Trains a THz-to-RGB ControlNet-based adapter together with a torchvision backbone
  classifier using Repeated Stratified K-Fold CV. In this variant, the adapter
  front-end and backbone are instantiated once before the CV loop and reused across
  all folds, so the model is updated continuously instead of being reinitialized per
  split. The script saves the best checkpoint for each split, writes a CV summary CSV,
  and optionally performs a final evaluation on a fixed test set.


- **`train_baseline_cn_without_cv_and_dropout.py`**  
  Single-run baseline for the THz-to-RGB ControlNet adapter combined with a
  torchvision backbone. The script trains on one fixed train split without
  cross-validation and uses dropout both in the 3D adapter and in the backbone
  classification head. It saves a single best checkpoint based on training
  accuracy and can optionally perform a final evaluation on a separate test set.


- **`train_baseline_cn_without_cv_and_dropout_2.py`**  
  Enhanced single-run baseline that trains the ControlNet-based THz-to-RGB adapter
  together with a torchvision backbone without cross-validation.  
  The script uses a two-phase optimization schedule (AdamW + OneCycleLR followed by
  SGD + CosineAnnealingLR) and supports early stopping based on accuracy and loss
  thresholds. It saves one best checkpoint over the full training run and can
  optionally perform a final evaluation on a fixed test set, including automatic
  aggregation of prediction files via `evaluate_predictions`.

---

### **Diffusion Classifier (DC) – Original ControlNet**

- **`train_dc_with_original_cn_multichannel.py`**  
  Trains a diffusion-based classifier on THz volumes using Stable Diffusion, an
  original ControlNet, and a multichannel latent adapter. The script runs
  Repeated Stratified K-Fold cross-validation with a prompt-based diffusion
  classifier, saves the best ControlNet+adapter checkpoints per split, writes a
  CV summary, and can optionally reload the globally best model for evaluation
  on a fixed test set. 

- **`train_dc_with_original_cn_multichannel_2.py`**  
  Variant of the diffusion classifier training script that keeps the Stable
  Diffusion backbone fixed across all CV splits and optimizes ControlNet (and
  optionally the multichannel adapter) with SGD. It runs Repeated Stratified
  K-Fold cross-validation, stores the best checkpoint per split, writes a CV
  summary, and can optionally reload the globally best model for evaluation on
  a fixed test set.   

- **`train_dc_with_original_cn_multichannel_dropout.py`**  
  Trains a diffusion-based classifier on THz volumes using Stable Diffusion v2.x,
  a ControlNet wrapped with dropout regularization, and a LatentMultiChannelAdapter
  with dropout. The script uses a two-stage optimizer schedule (AdamW + OneCycleLR
  followed by SGD + CosineAnnealingLR), early stopping based on accuracy and loss,
  saves the best model checkpoint and, optionally, runs a final evaluation on a
  held-out test set with prediction export and metric computation.
 

---

## Usage Notes
- All scripts require prepared THz data and a working diffusion-model setup.  
- Most scripts are designed for cluster environments (long runs, logging, checkpointing).  
- Multichannel variants target experiments where a trainable adapter reduces the THz volume into a latent representation.


