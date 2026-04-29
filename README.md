# Skin Lesion Classification (CISC 484/684 Final Project)

## Project Goal
Build and evaluate machine learning pipelines that classify skin lesion images from the HAM10000 dataset into:
- `benign`
- `malignant`

The project compares multiple approaches (classical ML and deep learning) after a shared preprocessing pipeline.

## What This Project Does
- Downloads and verifies the HAM10000 dataset files.
- Converts metadata labels into binary cancer classes.
- Preprocesses images (resize, CLAHE, color normalization, hair removal, normalization).
- Builds train/validation/test splits and model-ready arrays.
- Trains and evaluates:
  - an SVM feature pipeline
  - deep learning models (ResNet/EfficientNet)
- Produces plots and artifacts in `outputs/` and `src/03_deep_learning/plots/`.

## Repository Structure
- `src/01_data_setup/` - data download, cleaning, preprocessing, dataset finalization
- `src/02_svm/` - SVM-oriented feature engineering and training workflow
- `src/03_deep_learning/` - deep learning training/evaluation scripts
- `src/04_evaluation/` - analysis/evaluation utilities
- `data/` - local dataset files and generated splits
- `outputs/` - model outputs, checkpoints, and derived files

## Setup
From repo root:

```bash
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
# macOS/Linux
# source .venv/bin/activate

pip install --upgrade pip
pip install numpy pandas pillow opencv-python scikit-learn imbalanced-learn scikit-image tqdm matplotlib torch torchvision
```

Download HAM10000 from Kaggle and place the archive in `data/` (see notes in `src/01_data_setup/01_download_verify.py`).

## How To Run The Codebase

### 1) Data setup pipeline
These scripts use relative paths that are easiest to run from inside `src/01_data_setup/`:

```bash
cd src/01_data_setup
python 01_download_verify.py
python 02_metadata_to_binary_labels.py
python 03_resize.py
python 05_clahe.py
python 06_color_normalization.py
python 07_hair_remover_filter.py
python 04_normalization.py
python 08_finalize_freeze_dataset.py.py
```

### 2) SVM pipeline
From the SVM folder:

```bash
cd src/02_svm
python 02.py
python 03.py
```

### 3) Deep learning pipeline
Run from repository root:

```bash
python src/03_deep_learning/04_resnet_setup.py
```

## Notes
- Scripts assume specific relative paths under `data/`; keep folder names consistent.
- First full preprocessing/training run can take significant time depending on hardware.
- GPU acceleration is recommended for deep learning training.

