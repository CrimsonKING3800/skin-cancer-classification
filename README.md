# Skin Cancer Classification using Multi-Stage Ensemble Deep Learning

[![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![TensorFlow 2.x](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://tensorflow.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red.svg)](https://pytorch.org/)

An end-to-end deep learning pipeline for **binary skin cancer classification** (malignant vs. benign) using a multi-stage ensemble of CNNs and Vision Transformers, fused with tabular clinical metadata. Trained on the **ISIC 2024 Challenge** dataset containing **400,000+ dermoscopic images**.

---

## Table of Contents

- [Problem Statement](#problem-statement)
- [Features](#features)
- [Architecture](#architecture)
- [Dataset](#dataset)
- [Results](#results)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Challenges & Limitations](#challenges--limitations)
- [Future Work](#future-work)

---

## Problem Statement

Skin cancer is the most common form of cancer worldwide. Early and accurate detection of malignant lesions is critical for patient survival. Manual dermoscopic assessment is time-consuming, subjective, and requires specialist expertise.

This project develops a **robust, automated classification system** that combines:
- **Dermoscopic image analysis** via deep CNNs and Vision Transformers
- **Clinical metadata** (patient age, lesion size, color features, symmetry, etc.)
- **Multi-stage ensemble learning** for improved generalization

The goal is to assist dermatologists in early detection by providing a reliable second opinion.

---

## Features

- **Multi-Model Ensemble** — ResNet50, EfficientNet-B3, and Vision Transformer (ViT) combined
- **Dual-Input Fusion** — Image features + 34 tabular clinical metadata features
- **Image Preprocessing Pipeline** — CLAHE enhancement, lesion segmentation, morphological operations
- **Data Augmentation** — Rotation, flipping, zooming, and shearing to handle class imbalance
- **Transfer Learning** — Pretrained backbones fine-tuned on dermoscopic images
- **Class Imbalance Handling** — Strategic oversampling of minority class (malignant)
- **Comprehensive Metrics** — Accuracy, Precision, Recall, F1-Score, ROC-AUC

---

## Architecture

The system uses a **multi-stage ensemble** approach:

```
┌─────────────────────────────────────────────────┐
│                 Input Image (224×224×3)          │
└───────────────┬────────────┬────────────┬────────┘
                │            │            │
         ┌──────▼─────┐ ┌───▼────┐ ┌─────▼──────┐
         │  ResNet50   │ │EffNet  │ │    ViT     │
         │  (CNN)      │ │ B3     │ │ (Patches)  │
         └──────┬──────┘ └───┬────┘ └─────┬──────┘
                │            │            │
                └──────┬─────┘            │
                       │                  │
                ┌──────▼──────┐    ┌──────▼──────┐
                │ CNN Ensemble│    │ Tabular MLP │◄── Clinical Metadata
                │   Features  │    │   Features  │    (34 features)
                └──────┬──────┘    └──────┬──────┘
                       │                  │
                       └────────┬─────────┘
                                │
                         ┌──────▼──────┐
                         │   Fusion    │
                         │   Layer     │
                         └──────┬──────┘
                                │
                         ┌──────▼──────┐
                         │  Dense +    │
                         │  Sigmoid    │
                         └──────┬──────┘
                                │
                         ┌──────▼──────┐
                         │  Malignant  │
                         │  / Benign   │
                         └─────────────┘
```

---

## Dataset

**Source:** [ISIC 2024 — Skin Cancer Detection with 3D-TBP](https://www.kaggle.com/competitions/isic-2024-skin-cancer-detection-with-3d-tbp)

| Property | Details |
|---|---|
| **Total Samples** | ~401,059 |
| **Image Format** | JPEG (stored in HDF5) |
| **Resolution** | Variable (resized to 128×128 or 224×224) |
| **Tabular Features** | 34 clinical metadata columns |
| **Target** | Binary — `0` (Benign), `1` (Malignant) |
| **Class Distribution** | Highly imbalanced (~0.1% malignant) |

### Preprocessing Steps

1. **Resizing** — All images resized to 128×128 (CNN) or 224×224 (ViT)
2. **Normalization** — Pixel values scaled to [0, 1]
3. **CLAHE** — Contrast Limited Adaptive Histogram Equalization for enhanced contrast
4. **Lesion Extraction** — Thresholding + morphological operations to isolate the lesion region
5. **Data Augmentation** — 5x augmentation on malignant samples (rotation, flip, shear, zoom)

> **Note:** The dataset is not included in this repository due to its size (~50GB). Download it from the [Kaggle competition page](https://www.kaggle.com/competitions/isic-2024-skin-cancer-detection-with-3d-tbp).

---

## Results

### Model Performance Comparison

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| Custom CNN (Image Only) | ~85% | 0.49 | 0.74 | 0.59 |
| ResNet50 + Tabular | ~88% | 0.65 | 0.78 | 0.71 |
| EfficientNet-B3 + Tabular | ~89% | 0.68 | 0.80 | 0.73 |
| **ViT + Tabular (Ensemble)** | **~91%** | **0.72** | **0.82** | **0.77** |

> Metrics reported on the validation set. Due to extreme class imbalance, recall and F1-score are prioritized over accuracy.

### Key Observations

- Tabular metadata significantly improved model performance over image-only baselines
- The ViT-Tabular fusion model achieved the best balance of precision and recall
- Data augmentation of malignant samples was critical for model convergence

---

## Project Structure

```
skin-cancer-classification/
│
├── README.md
├── requirements.txt
├── .gitignore
├── LICENSE
│
├── notebooks/
│   ├── cancer_prediction_models.ipynb           # Base CNN models
│   ├── image_preprocessing_pipeline.ipynb       # Image preprocessing
│   ├── skin_lesion_preprocessing.ipynb          # Lesion extraction
│   ├── resnet_efficientnet_ensemble.ipynb       # ResNet + EfficientNet
│   ├── multi_stage_ensemble_pipeline.ipynb      # Multi-stage ensemble
│   └── ensemble_architecture_experiments.ipynb  # Final ViT ensemble
│
├── src/
│   ├── train.py                                 # Training script
│   ├── evaluate.py                              # Evaluation & metrics
│   ├── inference.py                             # Single-image inference
│   ├── preprocessing.py                         # Image preprocessing
│   ├── dataset.py                               # PyTorch dataset class
│   └── models/
│       ├── resnet_model.py
│       ├── efficientnet_model.py
│       └── ensemble.py
│
├── assets/
├── outputs/
│
├── demo/
│   └── app.py
│
└── classification_flow_map.html
```

---

## Installation

```bash
# Clone the repository
git clone https://github.com/CrimsonKING3800/skin-cancer-classification.git
cd skin-cancer-classification

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## Usage

### Running Notebooks
```bash
jupyter notebook notebooks/
```

### Training
```bash
python src/train.py --data_path /path/to/data --epochs 20 --batch_size 32
```

### Inference
```bash
python src/inference.py --image /path/to/lesion_image.jpg
```

### Demo App
```bash
streamlit run demo/app.py
```

---

## Challenges & Limitations

- **Extreme Class Imbalance** — Only ~0.1% of samples are malignant, making it very challenging for models to learn discriminative features for the minority class
- **Limited GPU Memory** — Full dataset processing requires significant GPU VRAM; batch processing and image resizing were necessary trade-offs
- **Dataset Variability** — Dermoscopic images vary in lighting, magnification, and quality, which reduces generalization
- **Overfitting Risk** — Small effective training set for malignant class, even after augmentation
- **No External Validation** — Model was not validated on an independent external dataset

---

## Future Work

- **Deploy via FastAPI** — Create a REST API for real-time inference
- **Grad-CAM Explainability** — Add visual explanations showing which image regions influence predictions
- **Larger Vision Transformers** — Explore ViT-L architectures with self-supervised pretraining
- **Docker Containerization** — Package the full pipeline for reproducible deployment
- **Experiment Tracking** — Integrate MLflow or Weights & Biases for systematic tracking
- **Multi-class Extension** — Extend from binary to multi-class classification (7+ lesion types)
- **Federated Learning** — Explore privacy-preserving training across distributed hospital datasets
