import os
import cv2
import numpy as np

#paths
INPUT_DIR  = os.path.join("data", "HAM10000_images_clahe")
OUTPUT_DIR = os.path.join("data", "HAM10000_images_color_normalized")

# Choose one image in the folder as the reference image
REFERENCE_IMAGE = "ISIC_0029836.jpg"

os.makedirs(OUTPUT_DIR, exist_ok=True)

#match channel of image to reference image
def match_cdf(source, reference):
    src_hist, _=np.histogram(source.flatten(), 256, [0,256])
    ref_hist, _=np.histogram(reference.flatten(), 256, [0,256])

    src_cdf=src_hist.cumsum().astype(np.float64)
    ref_cdf=ref_hist.cumsum().astype(np.float64)
    src_cdf/=src_cdf[-1]
    ref_cdf/=ref_cdf[-1]

    table=np.zeros(256, dtype=np.uint8)
    ref_i=0
    for src_i in range(256):
        while ref_i<255 and ref_cdf[ref_i]<src_cdf[src_i]:
            ref_i+=1
        table[src_i]=ref_i
    return table[source]

#match color channels of image to reference image
def match_hist_color(source, reference):
    matched=np.zeros_like(source)
    for channel in range(3):
        matched[:, :, channel]=match_cdf(source[:, :, channel], reference[:, :, channel])
    return matched

#load and check reference image
ref_path = os.path.join(INPUT_DIR, REFERENCE_IMAGE)

if not os.path.exists(ref_path):
    raise FileNotFoundError(f"Reference image not found: {ref_path}")

reference = cv2.imread(ref_path)

if reference is None:
    raise ValueError(f"Could not read reference image: {ref_path}")

#process images
total=0
skipped=0

files = [f for f in os.listdir(INPUT_DIR) if f.endswith(".jpg")]

for fname in files:
    src_path = os.path.join(INPUT_DIR, fname)
    dst_path = os.path.join(OUTPUT_DIR, fname)

    if os.path.exists(dst_path):
        skipped += 1
        continue

    source=cv2.imread(src_path)

    if source is None:
        print(f"Skipping unreadable file: {fname}")
        continue

    matched=match_hist_color(source, reference)
    cv2.imwrite(dst_path, matched)

    total+=1
    if total%500==0:
        print(f"Processed {total} images...")

    #summarize
print("\nDone.")
print(f"Processed: {total} images")
print(f"Skipped:   {skipped} images")
print(f"Saved to:  {OUTPUT_DIR}")