import os
import numpy as np
from PIL import Image

#create paths
INPUT_DIR=os.path.join("..", "..", "data", "HAM10000_images_hair_removed")
OUTPUT_DIR=os.path.join("..", "..", "data", "HAM10000_images_normalized")

#create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

#Normalize resized images
total=0
skipped=0

files = [f for f in os.listdir(INPUT_DIR) if f.endswith(".jpg")]

for file in files:
    src_path=os.path.join(INPUT_DIR, file)
    out_name= os.path.splitext(file)[0]
    dst_path=os.path.join(OUTPUT_DIR, out_name)

    #skip if already normalized
    if os.path.exists(dst_path):
        skipped+=1
        continue
    with Image.open(src_path) as img:
        img_array=np.array(img, dtype=np.float32)/255.0

    np.save(dst_path, img_array)

    total+=1
    if total%500==0:
        print(f" Normalized {total} images...")

    #summarize
print("\nDone.")
print(f"Normalized: {total} images")
print(f"Skipped:    {skipped} images (already existed)")
print(f"Saved to:   {OUTPUT_DIR}")