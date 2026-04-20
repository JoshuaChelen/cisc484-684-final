import torch
import tensorflow as tf
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
import numpy as np


#---------- GPU Availability Test (CUDA for Linux/Windows, MPS for MacOS) -----------------------------
device = "gpu" if torch.cuda.is_available() or torch.backends.mps.is_available() else "cpu"
print(device)

files = glob.glob("data/HAM10000_images_part_2/*.jpg")

# display as a grid instead of one at a time
n_cols = 5
n_rows = 10  # 50 images total
fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 30))

grayscale_files = []

for ax, f in zip(axes.flatten(), files[:50]):
    img = mpimg.imread(f)
    ax.imshow(img)
    ax.axis('off')
    
    img = Image.open(f)
    # check mode — RGB is color, L is grayscale
    if img.mode != "RGB":
        grayscale_files.append((f, img.mode))
    else:
        # also check if all 3 channels are identical (looks color but is grayscale)
        arr = np.array(img)
        if np.array_equal(arr[:,:,0], arr[:,:,1]) and np.array_equal(arr[:,:,1], arr[:,:,2]):
            grayscale_files.append((f, "functionally grayscale"))

plt.tight_layout()
plt.show()


