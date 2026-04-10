import os
import cv2
import numpy as np
import random

#paths
INPUT_DIR  = "data/HAM10000_images_color_normalized"
#OUTPUT_DIR = "data/HAM10000_images_hair_removed"
OUTPUT_DIR="data/test_20_hair_removed"

os.makedirs(OUTPUT_DIR, exist_ok=True)


def remove_hair(image):
    #convert to gray
    gray=cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    #using blackhat to highlight thin lines
    kernel=cv2.getStructuringElement(cv2.MORPH_RECT, (17, 17))
    blackhat= cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    #thresholding to create hair mask
    _, hair_mask=cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)
    #thickening mask
    hair_mask=cv2.dilate(hair_mask, np.ones((3,3), np.unit8), iterations=1)
    #fill in mask 
    result=cv2.inpaint(image, hair_mask, 3, cv2.INPAINT_TELEA)

    return result, hair_mask

#process images
total=0
skipped=0

#files = [f for f in os.listdir(INPUT_DIR) if f.endswith(".jpg")]
files = [f for f in os.listdir(INPUT_DIR) if f.endswith(".jpg")]
random.seed(42)
files = random.sample(files, min(20, len(files)))

print("Testing these files:")
for f in files:
    print(f)

for fname in files:
    src_path = os.path.join(INPUT_DIR, fname)
    dst_path = os.path.join(OUTPUT_DIR, fname)

    if os.path.exists(dst_path):
        skipped += 1
        continue

    image=cv2.imreaad(src_path)

    if image is None:
        print(f"Skipping unreadable file: {fname}")
        continue

    cleaned, mask=remove_hair(image)
    cv2.imwrite(dst_path, cleaned)

    total+=1
    if total%500==0:
        print(f"Processed {total} images...")

#summarize
print("\nDone.")
print(f"Processed: {total} images")
print(f"Skipped:   {skipped} images")
print(f"Saved to:  {OUTPUT_DIR}")