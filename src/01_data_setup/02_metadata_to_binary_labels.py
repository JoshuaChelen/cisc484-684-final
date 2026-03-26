"""
this script reads the original HAM10000 metadata CSV, maps the 7 lesion types to binary labels (benign vs malignant), 
and creates a new CSV with the image_id, original dx, binary label, and filepath for each image. 
This will be used by later scripts to load images and labels for training and evaluation.

to use:
1. Ensure you have run `src/01_download_verify.py` to extract the dataset and verify the images.
2. Run this script to generate `data/metadata_labeled.csv` with the binary labels and filepaths.

output:
- A new CSV file at `data/metadata_labeled.csv` containing:
    - image_id: original image ID from the metadata
    - dx: original diagnosis label (one of the 7 types)
    - label: binary label ("benign" or "malignant")
    - filepath: full path to the corresponding image file
"""

import pandas as pd
import os

# ── Paths (relative to src/) ───────────────────────────────────────────────
METADATA_PATH = os.path.join("..", "..", "data", "HAM10000_metadata.csv")
IMG_DIR_1     = os.path.join("..", "..", "data", "HAM10000_images_part_1")
IMG_DIR_2     = os.path.join("..", "..", "data", "HAM10000_images_part_2")
OUTPUT_PATH   = os.path.join("..", "..", "data", "metadata_labeled.csv")

# ── Binary label mapping ────────────────────────────────────────────────────
BENIGN    = ["bkl", "df", "nv", "vasc"]
MALIGNANT = ["akiec", "bcc", "mel"]

def get_label(dx):
    if dx in MALIGNANT:
        return "malignant"
    elif dx in BENIGN:
        return "benign"
    else:
        return "unknown"

# ── Build filepath lookup from both image folders ───────────────────────────
filepath_map = {}
for folder in [IMG_DIR_1, IMG_DIR_2]:
    for fname in os.listdir(folder):
        if fname.endswith(".jpg"):
            image_id = fname.replace(".jpg", "")
            filepath_map[image_id] = os.path.join(folder, fname)

# ── Load metadata and attach labels + filepaths ─────────────────────────────
df = pd.read_csv(METADATA_PATH)
df["label"]    = df["dx"].apply(get_label)
df["filepath"] = df["image_id"].map(filepath_map)

# ── Sanity checks ───────────────────────────────────────────────────────────
print(df[["image_id", "dx", "label", "filepath"]].head(10))
print(f"\nTotal rows:        {len(df)}")
print(f"Benign:            {(df['label'] == 'benign').sum()}")
print(f"Malignant:         {(df['label'] == 'malignant').sum()}")
print(f"Unknown labels:    {(df['label'] == 'unknown').sum()}")
print(f"Missing filepaths: {df['filepath'].isna().sum()}")

# ── Save for use by other scripts ───────────────────────────────────────────
df.to_csv(OUTPUT_PATH, index=False)
print(f"\nSaved to {OUTPUT_PATH}")