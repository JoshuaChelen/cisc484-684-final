CISC484/684 Final Project


Skin lesion classification on **HAM10000** using classical features / SVM and deep learning. This repository separates **data acquisition**, **preprocessing**, **SVM**, and **deep learning** code under `src/`.

---

## Preprocessing pipeline (reproducibility)

Scripts `01`–`04` live in `src/01_data_setup/` and expect you to **run them from that folder** so their `../..` paths reach the repo `data/` directory:

```powershell
cd src\01_data_setup
```

Script `05_clahe.py` uses paths like `data/HAM10000_images_resized` **relative to the current working directory**, so run it from the **repository root** (where the `data/` folder lives), not from `src/01_data_setup/`.

### Environment

Install Python dependencies used by the pipeline (at minimum):

- `Pillow` — image IO and resize  
- `numpy` — float arrays and `.npy` export  
- `pandas` — metadata CSV handling  
- `opencv-python` — CLAHE (`05_clahe.py`)

Example:

```powershell
pip install pillow numpy pandas opencv-python
```

### Step order and artifacts

| Step | Script | Input | Output | Notes |
|------|--------|-------|--------|--------|
| 1 | `01_download_verify.py` | Zip path in `ZIP_PATH` inside the script | `data/HAM10000_images_part_1/`, `part_2/` | **Edit `ZIP_PATH`** in the script so it points at your local archive (the repo may still contain a teammate-specific path). Verifies **10,015** JPEGs and runs `PIL.Image.verify()`. |
| 2 | `02_metadata_to_binary_labels.py` | `data/HAM10000_metadata.csv` + image folders | `data/metadata_labeled.csv` | Maps 7 `dx` codes to **benign** / **malignant**; adds `filepath` per row. |
| 3 | `03_resize.py` | Raw images in both `HAM10000_images_part_*` | `data/HAM10000_images_resized/*.jpg` | **128×128**, LANCZOS, JPEG quality **95**. |
| 4 | `05_clahe.py` (optional but recommended for contrast) | Resized JPEGs | `data/HAM10000_images_clahe/*.jpg` | LAB **L**-channel CLAHE: `clipLimit=2.0`, `tileGridSize=(8,8)`. |
| 5 | `04_normalization.py` | Resized JPEGs (see below) | `data/HAM10000_images_normalized/*.npy` | Scales uint8 **\[0, 255\]** → float32 **\[0, 1\]**, shape **H×W×3**. |

### Important: chaining CLAHE and normalization

As written, **`04_normalization.py` reads from `HAM10000_images_resized`**, not from `HAM10000_images_clahe`. If the full pipeline should apply CLAHE **before** scaling to \[0, 1\], the team needs a single agreed artifact path (e.g. normalize from CLAHE JPEGs) and must document that choice in the shared log so SVM and DL use the same inputs.

### Binary label mapping (HAM10000)

Defined in `02_metadata_to_binary_labels.py`:

- **Benign:** `bkl`, `df`, `nv`, `vasc`  
- **Malignant:** `akiec`, `bcc`, `mel`

### Items planned but not in this folder yet

The following Week-2 checklist items may still need implementation or integration elsewhere in the repo:

- Dataset-wide **color normalization** (e.g., per-channel standardization across dermoscope sources)  
- **Dull-razor** (or similar) hair removal  
- **Train/validation/test** split, **SMOTE** or class weights, **HSV / GLCM / shape** features, **PCA**, and **ResNet** training scripts  

When those land, extend this section with exact commands, random seeds, and versions.

---

## Repository layout (high level)

- `src/01_data_setup/` — download, labels, resize, CLAHE, \[0,1\] NumPy export  
- `src/02_svm/` — SVM-related utilities  
- `src/03_deep_learning/` — DL environment helpers  
- `eval_figures/` — exploratory plotting scripts  

