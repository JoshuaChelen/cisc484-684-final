# =========================================
# HAM10000 EDA Script (Local Dataset Only)
# =========================================

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from PIL import Image

sns.set(style="whitegrid")  # nicer plots

#  Dataset Paths 
dataset_dir = "HAM10000_data"
metadata_csv = os.path.join(dataset_dir, "HAM10000_metadata.csv")
# =========================


images_dir = dataset_dir

# load Metadata
metadata = pd.read_csv(metadata_csv)
print("Metadata loaded. Total samples:", len(metadata))
print(metadata.head())

# Map to Binary Labels 
malignant = ["akiec", "bcc", "mel"]
metadata["binary_label"] = metadata["dx"].apply(
    lambda x: "malignant" if x in malignant else "benign"
)
print(metadata["binary_label"].value_counts())

# Create output directory for plots
plot_dir = "EDA_plots"
os.makedirs(plot_dir, exist_ok=True)

# Class Distribution Plots 

plt.figure(figsize=(8,5))
sns.countplot(data=metadata, x="dx", order=metadata["dx"].value_counts().index)
plt.title("Class Distribution of Skin Lesion Categories (HAM10000)")
plt.xlabel("Diagnosis Category")
plt.ylabel("Number of Images")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, "class_distribution_7cat.png"))
plt.show()

# Binary distribution
plt.figure(figsize=(6,4))
sns.countplot(data=metadata, x="binary_label")
plt.title("Binary Class Distribution (Benign vs Malignant)")
plt.xlabel("Class")
plt.ylabel("Number of Images")
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, "class_distribution_binary.png"))
plt.show()


# Load a sample of images for PCA (reduce size to speed up)
sample_df = metadata.sample(1000, random_state=42)

images = []
labels = []

# List of folders with images
image_subdirs = [
    os.path.join(images_dir, "HAM10000_images_part_1"),
    os.path.join(images_dir, "HAM10000_images_part_2")
]

for _, row in sample_df.iterrows():
    img_filename = row["image_id"] + ".jpg"
    img_path = None
    # Find image in either part 1 or part 2
    for subdir in image_subdirs:
        candidate = os.path.join(subdir, img_filename)
        if os.path.exists(candidate):
            img_path = candidate
            break
    if img_path is None:
        print(f"Skipping {img_filename}: not found in any folder")
        continue
    
    try:
        img = Image.open(img_path).resize((64,64))  # resize to 64x64
        img_array = np.array(img).flatten()
        images.append(img_array)
        labels.append(row["binary_label"])
    except Exception as e:
        print(f"Skipping {img_filename}: {e}")

X = np.array(images)

if len(X) == 0:
    raise RuntimeError("No images loaded. Check paths!")

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

pca_df = pd.DataFrame({
    "PC1": X_pca[:,0],
    "PC2": X_pca[:,1],
    "label": labels
})

plt.figure(figsize=(7,6))
sns.scatterplot(
    data=pca_df,
    x="PC1",
    y="PC2",
    hue="label",
    alpha=0.7
)
plt.title("PCA Projection of Skin Lesion Images")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, "pca_projection.png"))
plt.show()

print("EDA plots saved in folder:", plot_dir)