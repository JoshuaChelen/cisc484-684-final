import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# paths
PROCESSED_DIR = os.path.join("..", "..", "data", "HAM10000_images_normalized")
LABELS_CSV = os.path.join("..", "..", "data", "metadata_labeled.csv")
FINAL_DIR = os.path.join("..", "..", "data", "final_dataset")

IMAGES_OUT = os.path.join(FINAL_DIR, "images_npy")
os.makedirs(IMAGES_OUT, exist_ok=True)

# load labels
df = pd.read_csv(LABELS_CSV)

# keep only benign/malignant
df = df[df["label"].isin(["benign", "malignant"])].copy()

# map labels to 0/1
label_map = {"benign": 0, "malignant": 1}
df["y"] = df["label"].map(label_map)

# keep only rows whose processed file exists
def processed_path(image_id):
    return os.path.join(PROCESSED_DIR, f"{image_id}.npy")

df["processed_path"] = df["image_id"].apply(processed_path)
df = df[df["processed_path"].apply(os.path.exists)].copy()

print("Usable rows:", len(df))

# split
train_df, temp_df = train_test_split(
    df,
    test_size=0.3,
    stratify=df["y"],
    random_state=42
)

val_df, test_df = train_test_split(
    temp_df,
    test_size=0.5,
    stratify=temp_df["y"],
    random_state=42
)

train_df["split"] = "train"
val_df["split"] = "val"
test_df["split"] = "test"

final_df = pd.concat([train_df, val_df, test_df], ignore_index=True)

# save metadata
metadata_out = os.path.join(FINAL_DIR, "metadata_final.csv")
final_df.to_csv(metadata_out, index=False)

print("Saved metadata to:", metadata_out)

# build arrays
X = []
y = []
image_ids = []

for _, row in final_df.iterrows():
    arr = np.load(row["processed_path"]).astype(np.float32)
    X.append(arr)
    y.append(row["y"])
    image_ids.append(row["image_id"])

X = np.array(X, dtype=np.float32)
y = np.array(y, dtype=np.int64)

print("X shape:", X.shape)
print("y shape:", y.shape)

# save deep learning version
np.save(os.path.join(FINAL_DIR, "X_cnn.npy"), X)
np.save(os.path.join(FINAL_DIR, "y_cnn.npy"), y)

# save SVM version (flatten images)
X_svm = X.reshape(X.shape[0], -1)
np.save(os.path.join(FINAL_DIR, "X_svm.npy"), X_svm)
np.save(os.path.join(FINAL_DIR, "y_svm.npy"), y)

print("Saved:")
print(" - X_cnn.npy")
print(" - y_cnn.npy")
print(" - X_svm.npy")
print(" - y_svm.npy")