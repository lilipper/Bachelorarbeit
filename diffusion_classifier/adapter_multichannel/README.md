# Training Scripts Overview

This folder contains all training scripts used for the ControlNet-based adapter experiments (“cn_adapter”) and the Diffusion Classifier (“dc”). The scripts cover baseline setups as well as variants with cross-validation, dropout, and multichannel processing.

## File Structure and Purpose

### **Adapter Conversion**
- **`thz_to_vae_adapter.py`**  
  Utility script that converts THz volume data into latent representations. Used to prepare the adapter as a preprocessing step.

---

### **Baseline Experiments (ControlNet Adapter)**
- **`train_baseline_cn_adapter.py`**  
  Standard baseline: trains the ControlNet adapter without cross-validation and without dropout.

- **`train_baseline_cn_adapter_load_once.py`**  
  Variant that loads the data only once (optimized for cluster jobs).

- **`train_baseline_cn_without_cv_and_dropout.py`**  
  Minimal version without cross-validation and without dropout.

- **`train_baseline_cn_without_cv_and_dropout_2.py`**  
  Second iteration of the same setup with minor modifications.

---

### **Diffusion Classifier (DC) – Original ControlNet**
All scripts use the original ControlNet as the adapter backbone and only train the multichannel branch.

- **`train_dc_with_original_cn_multichannel.py`**  
  Main script for training the Diffusion Classifier with multichannel support.

- **`train_dc_with_original_cn_multichannel_2.py`**  
  Alternative version with small adjustments for stability or debugging.

- **`train_dc_with_original_cn_multichannel_acc.py`**  
  Version with extended accuracy logging during training.

- **`train_dc_with_original_cn_multichannel_dropout.py`**  
  Adds dropout to improve regularization.

---

## Usage Notes
- All scripts require prepared THz data and a working diffusion-model setup.  
- Most scripts are designed for cluster environments (long runs, logging, checkpointing).  
- Multichannel variants target experiments where a trainable adapter reduces the THz volume into a latent representation.

## Recommended Workflow
1. **Run the baseline scripts**  
   → Ensure the data pipeline and the adapter operate correctly.  
2. **Run the DC scripts**  
   → Perform experiments using the frozen diffusion model with a trainable adapter.  
3. **Test variants (Dropout, ACC, v2 scripts)**  
   → Explore stability improvements and generalization behavior.

---

If you want, I can also generate this README as a standalone file inside your project folder.
