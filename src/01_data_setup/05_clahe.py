import os
import cv2
import numpy as np
from PIL import Image
from display_images import display_images

#paths
INPUT_DIR  = os.path.join("..", "..", "data", "HAM10000_images_resized")
OUTPUT_DIR = os.path.join("..", "..", "data", "HAM10000_images_clahe")

os.makedirs(OUTPUT_DIR, exist_ok=True)

#create clahe
clahe=cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

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

    #load image
    img=cv2.imread(src_path)
    #convert to lab color space
    lab=cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    #split channels
    l, a, b=cv2.split(lab)
    #apply clahe to l channel
    l_clahe=clahe.apply(l)

    #merge back
    lab_clahe=cv2.merge((l_clahe, a, b))

    #convert to rgb
    final_img=cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)

    #save
    cv2.imwrite(dst_path, final_img)

    total+=1
    if total%500==0:
        print(f"  Processed {total} images...")

#summarize
print("\nDone.")
print(f"Processed: {total} images")
print(f"Skipped:   {skipped} images")
print(f"Saved to:  {OUTPUT_DIR}")
display_images(OUTPUT_DIR, "clahe")