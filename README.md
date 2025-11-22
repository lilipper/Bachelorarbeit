# Bachelor’s Thesis by Linus Lippert

## Zero-Shot Classification with Diffusion Models

This repository contains the code and experiments of my bachelor’s thesis at the University of Mannheim.  
The goal is to investigate **how a generative diffusion model can be used as a classifier** without being trained itself.  
For this purpose, a **trainable adapter** is developed that transforms complex 3D THz data into a format understandable for a frozen diffusion model (Stable Diffusion 2.1).  

## ⚙️ Setup & Execution

To set up the project locally, run the following commands:

```bash
# Clone the repository
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>

# Install dependencies
pip install -r requirements.txt
```

## 🎯 Project Overview

- **Zero-Shot Classification:** The diffusion model is not retrained but only adapted to the input data via a trainable adapter.  
- **Adapter Architecture:** Transforms 3D measurement data (292 × 90 × 1400) into a 2D-compatible format for the diffusion model.  
- **ControlNet Extension:** Optional conditioning to guide structural information.  
- **Comparison:** Classical models such as ViT, ResNet, and ConvNeXt serve as baselines.  
- **Evaluation:** Besides accuracy, explainability (Grad-CAM) and robustness are analyzed.  

## 📊 Dataset
You can find the dataset here: [THz Dataset](https://uni-siegen.sciebo.de/s/QdKujlTwbVhmAAX)
The dataset consists of **THz measurements of fiber-reinforced plastic samples** with various defect types, such as:  

- Fiber displacement  
- Fiber breakage  
- Delamination  
- Pores  
- and others  

Each sample was measured with high spectral resolution (up to 300 GHz).  

**Data structure:**  
- `NX = 292`  
- `NY = 90`  
- `NF = 1400`  
