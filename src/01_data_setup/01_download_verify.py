'''
extracts the HAM10000 dataset and verifies that all images are present and not corrupted.

to use:
1. Download the dataset from Kaggle: https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000
2. Place the downloaded zip file in the `data/` directory (so you should have `data/skin-cancer-mnist-ham10000.zip`)
3. Run this script

output:
- Extracted images in `data/HAM10000_images_part_1/` and `data/HAM10000_images_part_2/`
- Console output confirming the number of images and integrity check results
'''

import os
import zipfile
from PIL import Image

# ── Paths (relative to src/) ───────────────────────────────────────────────
ZIP_PATH    = os.path.join("..", "..", "data", "skin-cancer-mnist-ham10000.zip")
EXTRACT_DIR = os.path.join("..", "..", "data")
IMG_DIR_1   = os.path.join("..", "..", "data", "HAM10000_images_part_1")
IMG_DIR_2   = os.path.join("..", "..", "data", "HAM10000_images_part_2")

# ── Step 1: Extract zip ────────────────────────────────────────────────────
if not os.path.exists(IMG_DIR_1):
    print("Extracting dataset...")
    with zipfile.ZipFile(ZIP_PATH, "r") as z:
        z.extractall(EXTRACT_DIR)
    print("Extraction complete.\n")
else:
    print("Data already extracted, skipping.\n")

# ── Step 2: Count images ───────────────────────────────────────────────────
count_1 = len([f for f in os.listdir(IMG_DIR_1) if f.endswith(".jpg")])
count_2 = len([f for f in os.listdir(IMG_DIR_2) if f.endswith(".jpg")])

print(f"Part 1: {count_1} images")
print(f"Part 2: {count_2} images")
print(f"Total:  {count_1 + count_2} images")
assert count_1 + count_2 == 10015, "WARNING: Expected 10,015 images — count mismatch!"
print("Image count verified.\n")

# ── Step 3: Integrity check ────────────────────────────────────────────────
print("Running integrity check...")
corrupted = []

for folder in [IMG_DIR_1, IMG_DIR_2]:
    files = [f for f in os.listdir(folder) if f.endswith(".jpg")]
    for fname in files:
        fpath = os.path.join(folder, fname)
        try:
            with Image.open(fpath) as img:
                img.verify()
        except Exception as e:
            corrupted.append((fpath, str(e)))

if corrupted:
    print(f"Found {len(corrupted)} corrupted files:")
    for path, err in corrupted:
        print(f"  {path} — {err}")
else:
    print("All 10,015 images passed integrity check. No corrupted files found.")