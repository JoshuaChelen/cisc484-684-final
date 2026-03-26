'''
Resizes all images in the HAM10000 dataset to 128x128 pixels using high-quality resampling.

to use:
1. Ensure you have run `src/01_download_verify.py` to extract the dataset and verify the images.
2. Run this script to create resized images in `data/HAM10000_images_resized/`
z
output:
- Resized images saved in `data/HAM10000_images_resized/` with the same filenames as the originals.
- Console output summarizing the number of images resized and skipped (if already resized).
'''

import os
from PIL import Image

# ── Paths (relative to src/) ───────────────────────────────────────────────
IMG_DIR_1   = os.path.join("..", "..", "data", "HAM10000_images_part_1")
IMG_DIR_2   = os.path.join("..", "..", "data", "HAM10000_images_part_2")
OUTPUT_DIR  = os.path.join("..", "..", "data", "HAM10000_images_resized")
TARGET_SIZE = (128, 128)

# ── Create output directory if it doesn't exist ────────────────────────────
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Resize all images ──────────────────────────────────────────────────────
total = 0
skipped = 0

for folder in [IMG_DIR_1, IMG_DIR_2]:
    files = [f for f in os.listdir(folder) if f.endswith(".jpg")]
    for fname in files:
        src_path = os.path.join(folder, fname)
        dst_path = os.path.join(OUTPUT_DIR, fname)

        # Skip if already resized
        if os.path.exists(dst_path):
            skipped += 1
            continue

        with Image.open(src_path) as img:
            img_resized = img.resize(TARGET_SIZE, Image.LANCZOS)
            img_resized.save(dst_path, "JPEG", quality=95)

        total += 1
        if total % 500 == 0:
            print(f"  Resized {total} images...")

# ── Summary ────────────────────────────────────────────────────────────────
print(f"\nDone.")
print(f"Resized:  {total} images")
print(f"Skipped:  {skipped} images (already existed)")
print(f"Saved to: {OUTPUT_DIR}")