import os
import cv2
import numpy as np
import random
import matplotlib.pyplot as plt
from display_images import display_images


#paths
INPUT_DIR  = os.path.join("data", "HAM10000_images_color_normalized")
#OUTPUT_DIR = os.path.join("..", "..", "data", "HAM10000_images_hair_removed")
OUTPUT_DIR=os.path.join("data","HAM10000_images_hair_removed" )

os.makedirs(OUTPUT_DIR, exist_ok=True)



def remove_hair(image):

    #convert to gray
    gray=cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    #using blackhat to highlight thin lines
    kernel=cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    blackhat= cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    #thresholding to create hair mask
    _, hair_mask=cv2.threshold(blackhat, 30, 255, cv2.THRESH_BINARY)
    #thickening mask
    #hair_mask=cv2.dilate(hair_mask, np.ones((3,3), np.uint8), iterations=1)
    #fill in mask 
    result=cv2.inpaint(image, hair_mask, 1, cv2.INPAINT_TELEA)

    return result

  


#process images
total=0
skipped=0

#files = [f for f in os.listdir(INPUT_DIR) if f.endswith(".jpg")]
files = [f for f in os.listdir(INPUT_DIR) if f.endswith(".jpg")]
#random.seed(42)
#files = random.sample(files, min(20, len(files)))
#displaying testing samples

# for fname in files:
#     before_path = os.path.join(INPUT_DIR, fname)
#     after_path  = os.path.join(OUTPUT_DIR, fname)

#     before = cv2.imread(before_path)
#     after  = cv2.imread(after_path)

#     if before is None or after is None:
#         print(f"Skipping {fname}")
#         continue

#     # Convert BGR → RGB for correct display
#     before = cv2.cvtColor(before, cv2.COLOR_BGR2RGB)
#     after  = cv2.cvtColor(after, cv2.COLOR_BGR2RGB)

#     plt.figure(figsize=(8,4))

#     plt.subplot(1,2,1)
#     plt.imshow(before)
#     plt.title("Before")
#     plt.axis("off")

#     plt.subplot(1,2,2)
#     plt.imshow(after)
#     plt.title("After")
#     plt.axis("off")

#     plt.suptitle(fname)
#     plt.show()

# print("Testing these files:")
# for f in files:
#     print(f)

for fname in files:
    src_path = os.path.join(INPUT_DIR, fname)
    dst_path = os.path.join(OUTPUT_DIR, fname)

    if os.path.exists(dst_path):
        skipped += 1
        continue

    image=cv2.imread(src_path)

    if image is None:
        print(f"Skipping unreadable file: {fname}")
        continue

    cleaned=remove_hair(image)
    cv2.imwrite(dst_path, cleaned)

    total+=1
    if total%500==0:
        print(f"Processed {total} images...")

#summarize
print("\nDone.")
print(f"Processed: {total} images")
print(f"Skipped:   {skipped} images")
print(f"Saved to:  {OUTPUT_DIR}")
display_images(OUTPUT_DIR, "hair_remover")