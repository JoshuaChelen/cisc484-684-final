"""
CISC 484/684 — Machine Learning
Skin Lesion Cancer Classification — HAM10000
Member 2: SVM / Phase 1 Lead
Week 2 Tasks:
    1. Stratified 70/15/15 train/validation/test split
    2. SMOTE oversampling (chosen over class-weighted loss — see note below)
    3. HSV color histogram extraction (16 bins per channel)
    4. GLCM texture descriptor extraction (contrast, homogeneity, energy)

Required installs:
    pip install numpy pandas opencv-python scikit-learn imbalanced-learn scikit-image tqdm
"""

# ─────────────────────────────────────────────
# DEPENDENCY CHECK — auto-installs missing packages
# ─────────────────────────────────────────────
import subprocess, sys

REQUIRED = [
    ("numpy",           "numpy"),
    ("pandas",          "pandas"),
    ("cv2",             "opencv-python"),
    ("sklearn",         "scikit-learn"),
    ("imblearn",        "imbalanced-learn"),
    ("skimage",         "scikit-image"),
    ("tqdm",            "tqdm"),
]

for import_name, pip_name in REQUIRED:
    try:
        __import__(import_name)
    except ImportError:
        print(f"Installing {pip_name}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", pip_name])

# ─────────────────────────────────────────────
# IMPORTS
# ─────────────────────────────────────────────

import os
import numpy as np
import pandas as pd
import cv2
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from skimage.feature import graycomatrix, graycoprops
from tqdm import tqdm

# ─────────────────────────────────────────────
# 0. CONFIGURATION
# ─────────────────────────────────────────────

METADATA_PATH = "../../data/metadata_labeled.csv"
IMAGE_DIR     = "../../data/HAM10000_images_clahe/"
OUTPUT_DIR    = "outputs/"
SPLIT_DIR     = "../../data/"   # save splits here so whole team can access them
RANDOM_STATE  = 42
IMG_SIZE      = 128             # must match Member 1's resizing

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(SPLIT_DIR,  exist_ok=True)

# ─────────────────────────────────────────────
# TASK 1 — Stratified 70 / 15 / 15 Split
# ─────────────────────────────────────────────
# Goal: split the dataset such that each split (train/val/test) preserves
# the same benign:malignant ratio as the full dataset.
# HAM10000 is ~75% benign / ~25% malignant, so stratification is critical.
# ─────────────────────────────────────────────

print("=" * 60)
print("TASK 1 — Stratified Train / Val / Test Split")
print("=" * 60)

df = pd.read_csv(METADATA_PATH)
print(f"Total samples in metadata: {len(df)}")
print(f"Columns: {list(df.columns)}")

# Binary label mapping — skip if metadata_labeled.csv already has binary_label
MALIGNANT_CLASSES = {"akiec", "bcc", "mel"}
if "binary_label" not in df.columns:
    df["binary_label"] = df["dx"].apply(lambda x: 1 if x in MALIGNANT_CLASSES else 0)
    print("Binary label column created from 'dx'.")
else:
    print("Binary label column already present — skipping mapping.")

print("\nFull dataset class distribution:")
full_counts = df["binary_label"].value_counts()
print(f"  Benign    (0): {full_counts[0]:>5}  ({full_counts[0]/len(df):.2%})")
print(f"  Malignant (1): {full_counts[1]:>5}  ({full_counts[1]/len(df):.2%})")

# Step 1: carve out 15% test set (stratified)
train_val_df, test_df = train_test_split(
    df,
    test_size=0.15,
    stratify=df["binary_label"],
    random_state=RANDOM_STATE,
)

# Step 2: split remaining 85% into 70% train / 15% val
# 15% of full dataset = 15/85 ≈ 17.65% of the train_val portion
train_df, val_df = train_test_split(
    train_val_df,
    test_size=0.1765,
    stratify=train_val_df["binary_label"],
    random_state=RANDOM_STATE,
)

print(f"\nSplit sizes:")
print(f"  Train : {len(train_df):>5} samples  (target ~70%)")
print(f"  Val   : {len(val_df):>5} samples  (target ~15%)")
print(f"  Test  : {len(test_df):>5} samples  (target ~15%)")
print(f"  Total : {len(train_df) + len(val_df) + len(test_df):>5} samples")

# Verify class distribution is preserved in each split
print("\nClass distribution per split (verify ratios match full dataset):")
for split_name, split_df in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
    counts = split_df["binary_label"].value_counts()
    n = len(split_df)
    benign_pct    = counts.get(0, 0) / n
    malignant_pct = counts.get(1, 0) / n
    print(f"  {split_name:<6} — Benign: {counts.get(0,0):>5} ({benign_pct:.2%})  |  "
          f"Malignant: {counts.get(1,0):>5} ({malignant_pct:.2%})")

# Save splits to CSV so the whole team uses the same exact sets
train_df.to_csv(os.path.join(SPLIT_DIR, "split_train.csv"), index=False)
val_df.to_csv(  os.path.join(SPLIT_DIR, "split_val.csv"),   index=False)
test_df.to_csv( os.path.join(SPLIT_DIR, "split_test.csv"),  index=False)
print(f"\nSplits saved to {SPLIT_DIR}")


# ─────────────────────────────────────────────
# TASK 2 — Class Imbalance: SMOTE vs Class-Weighted Loss
# ─────────────────────────────────────────────
# DECISION: We use SMOTE on the training feature matrix (after feature
# extraction in Tasks 3 & 4) rather than class-weighted loss.
#
# Rationale:
#   • SMOTE works well for SVM — it physically balances the training set,
#     giving the RBF kernel equal exposure to both classes during fitting.
#   • Class-weighted loss is more natural for neural networks (gradient-based).
#     sklearn's SVM does support class_weight='balanced', but SMOTE gives us
#     explicit control and lets us verify the balanced distribution.
#   • SMOTE is applied ONLY to the training set — never to val or test.
#     Applying it to val/test would contaminate evaluation and inflate metrics.
#
# SMOTE is applied AFTER feature extraction (see bottom of file).
# ─────────────────────────────────────────────

print("\n" + "=" * 60)
print("TASK 2 — Class Imbalance Strategy: SMOTE (applied after feature extraction)")
print("=" * 60)
print("Decision: SMOTE on training set.")
print("  • Applied AFTER HSV + GLCM features are extracted.")
print("  • Val and Test sets are NOT oversampled — raw distribution preserved.")


# ─────────────────────────────────────────────
# HELPER — Image Loader
# ─────────────────────────────────────────────

def load_image(image_id: str, image_dir: str, size: int = IMG_SIZE):
    """
    Load a preprocessed image by its image_id.
    Returns BGR numpy array at (size x size), or None if file not found.
    """
    filename = f"{image_id}.jpg"
    filepath = os.path.join(image_dir, filename)
    if not os.path.exists(filepath):
        return None
    img = cv2.imread(filepath)
    if img is None:
        return None
    if img.shape[:2] != (size, size):
        img = cv2.resize(img, (size, size))
    return img


# ─────────────────────────────────────────────
# TASK 3 — HSV Color Histogram Extraction
# ─────────────────────────────────────────────
# For each image: convert BGR → HSV, compute a histogram over each of the
# 3 channels (H, S, V) with 16 bins each, normalize, and concatenate.
# Result: 16 * 3 = 48-dimensional feature vector per image.
#
# Why HSV for skin lesions?
#   • Hue captures the actual color of the lesion independent of brightness.
#   • Saturation reflects color intensity — malignant lesions often show
#     irregular, high-saturation regions.
#   • Value (brightness) is influenced by dermoscope lighting; useful but
#     less discriminative than H and S.
# ─────────────────────────────────────────────

def extract_hsv_histogram(img_bgr: np.ndarray, bins: int = 16) -> np.ndarray:
    """
    Extract a normalized HSV color histogram from a BGR image.

    Args:
        img_bgr: Image as a (H, W, 3) BGR numpy array.
        bins:    Number of histogram bins per channel. Default 16.

    Returns:
        1D numpy array of length bins * 3 (48 values with default bins=16).
        Values are L1-normalized so they sum to 1.0 per channel.
    """
    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    # OpenCV HSV ranges: H ∈ [0,180], S ∈ [0,255], V ∈ [0,255]
    channel_ranges = [(0, 180), (0, 256), (0, 256)]
    histograms = []

    for channel_idx, (low, high) in enumerate(channel_ranges):
        hist = cv2.calcHist(
            [img_hsv],
            [channel_idx],
            None,
            [bins],
            [low, high],
        )
        hist = hist.flatten()
        # L1 normalize so each channel histogram sums to 1
        total = hist.sum()
        if total > 0:
            hist = hist / total
        histograms.append(hist)

    return np.concatenate(histograms)  # shape: (bins * 3,) = (48,)


# ─────────────────────────────────────────────
# TASK 4 — GLCM Texture Descriptor Extraction
# ─────────────────────────────────────────────
# Gray-Level Co-occurrence Matrix (GLCM) captures spatial texture by counting
# how often pairs of pixel values occur at a given offset.
#
# We extract 3 Haralick properties:
#   • Contrast    — intensity variation between neighbors.
#                   High contrast = rough/irregular texture (common in malignant).
#   • Homogeneity — how close GLCM elements are to the diagonal.
#                   High homogeneity = smooth, uniform texture (more benign).
#   • Energy      — sum of squared GLCM elements; measures textural uniformity.
#                   High energy = very regular, repeated texture patterns.
#
# Computed at 4 orientations (0°, 45°, 90°, 135°) and averaged for rotation
# invariance. Final output: 3-dimensional vector.
# ─────────────────────────────────────────────

def extract_glcm_features(img_bgr: np.ndarray) -> np.ndarray:
    """
    Extract GLCM texture descriptors from a BGR image.

    Converts image to grayscale, computes GLCM at distances=[1] and
    angles=[0°, 45°, 90°, 135°], then averages each property across angles.

    Args:
        img_bgr: Image as a (H, W, 3) BGR numpy array.

    Returns:
        1D numpy array: [contrast, homogeneity, energy] — shape (3,).
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # Compute GLCM at 4 angles for rotation invariance
    glcm = graycomatrix(
        gray,
        distances=[1],
        angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
        levels=256,
        symmetric=True,
        normed=True,
    )
    # glcm shape: (256, 256, num_distances=1, num_angles=4)

    # Average each property across all 4 angles
    contrast    = graycoprops(glcm, "contrast").mean()
    homogeneity = graycoprops(glcm, "homogeneity").mean()
    energy      = graycoprops(glcm, "energy").mean()

    return np.array([contrast, homogeneity, energy])  # shape: (3,)


# ─────────────────────────────────────────────
# FEATURE EXTRACTION PIPELINE
# ─────────────────────────────────────────────
# Runs Tasks 3 & 4 over each split.
# HSV (48) + GLCM (3) = 51 features per image.
# Shape features from Member 3 will be concatenated in Week 3 before PCA.
# ─────────────────────────────────────────────

def extract_features_for_split(
    split_df: pd.DataFrame,
    image_dir: str,
    split_name: str,
):
    """
    Extract HSV + GLCM features for every image in a dataframe split.

    Args:
        split_df:   DataFrame with columns ['image_id', 'binary_label'].
        image_dir:  Directory containing preprocessed images.
        split_name: Label for progress display (e.g. 'Train').

    Returns:
        X: Feature matrix of shape (n_samples, 51).
        y: Label array of shape (n_samples,). 0=benign, 1=malignant.
    """
    X_list = []
    y_list = []
    skipped = 0

    print(f"\nExtracting features for {split_name} split ({len(split_df)} images)...")
    for _, row in tqdm(split_df.iterrows(), total=len(split_df), desc=split_name):
        img = load_image(row["image_id"], image_dir)
        if img is None:
            skipped += 1
            continue

        hsv_feat  = extract_hsv_histogram(img, bins=16)  # (48,)
        glcm_feat = extract_glcm_features(img)            # (3,)

        feature_vec = np.concatenate([hsv_feat, glcm_feat])  # (51,)
        X_list.append(feature_vec)
        y_list.append(row["binary_label"])

    if skipped > 0:
        print(f"  Warning: {skipped} images could not be loaded and were skipped.")

    X = np.array(X_list)
    y = np.array(y_list)
    print(f"  {split_name} feature matrix shape: {X.shape}")
    return X, y


# ── Run extraction ─────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("TASK 3 & 4 — Feature Extraction (HSV + GLCM)")
print("=" * 60)

X_train_raw, y_train = extract_features_for_split(train_df, IMAGE_DIR, "Train")
X_val,       y_val   = extract_features_for_split(val_df,   IMAGE_DIR, "Val")
X_test,      y_test  = extract_features_for_split(test_df,  IMAGE_DIR, "Test")

# Save val and test — these never get oversampled
np.save(os.path.join(OUTPUT_DIR, "X_val.npy"),  X_val)
np.save(os.path.join(OUTPUT_DIR, "y_val.npy"),  y_val)
np.save(os.path.join(OUTPUT_DIR, "X_test.npy"), X_test)
np.save(os.path.join(OUTPUT_DIR, "y_test.npy"), y_test)
print(f"\nVal and Test features saved (no SMOTE applied).")


# ─────────────────────────────────────────────
# TASK 2 (continued) — Apply SMOTE to Training Set
# ─────────────────────────────────────────────
# Applied here (after feature extraction) because SMOTE operates on feature
# vectors, not raw images. Applying it to raw pixels would be computationally
# prohibitive and semantically less meaningful.
# ─────────────────────────────────────────────

print("\n" + "=" * 60)
print("TASK 2 — Applying SMOTE to Training Set")
print("=" * 60)

print("Before SMOTE — Train class distribution:")
unique, counts_before = np.unique(y_train, return_counts=True)
for cls, cnt in zip(unique, counts_before):
    label = "Benign" if cls == 0 else "Malignant"
    print(f"  {label} ({cls}): {cnt:>5} samples")

smote = SMOTE(random_state=RANDOM_STATE)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train_raw, y_train)

print("\nAfter SMOTE — Train class distribution:")
unique, counts_after = np.unique(y_train_balanced, return_counts=True)
for cls, cnt in zip(unique, counts_after):
    label = "Benign" if cls == 0 else "Malignant"
    print(f"  {label} ({cls}): {cnt:>5} samples")

print(f"\nBalanced training matrix shape: {X_train_balanced.shape}")

# Save balanced training features
np.save(os.path.join(OUTPUT_DIR, "X_train_balanced.npy"), X_train_balanced)
np.save(os.path.join(OUTPUT_DIR, "y_train_balanced.npy"), y_train_balanced)

# Also save raw (pre-SMOTE) in case needed for comparison
np.save(os.path.join(OUTPUT_DIR, "X_train_raw.npy"), X_train_raw)
np.save(os.path.join(OUTPUT_DIR, "y_train_raw.npy"), y_train)

print(f"\nAll feature arrays saved to {OUTPUT_DIR}")
print("\nHandoff notes for team:")
print("  → Member 3 : concatenate shape features onto X_train_raw before SMOTE (or re-run this script after)")
print("  → Member 4 : use split_train/val/test.csv in ../../data/ for class ratio bar charts")
print("  → Week 3   : X_train_balanced feeds directly into SVM training + PCA")
print("  → Everyone : do NOT touch X_test / y_test until Week 5 final evaluation")


# ─────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────

print("\n" + "=" * 60)
print("WEEK 2 MEMBER 2 — COMPLETE")
print("=" * 60)
print("Task 1: Stratified 70/15/15 split                          DONE")
print("Task 2: SMOTE applied to training set                      DONE")
print("Task 3: HSV color histograms (16 bins × 3 ch = 48 feats)  DONE")
print("Task 4: GLCM descriptors (contrast, homogeneity, energy)   DONE")
print(f"\nOutputs in {OUTPUT_DIR}:")
print("  X_train_balanced.npy / y_train_balanced.npy")
print("  X_train_raw.npy      / y_train_raw.npy")
print("  X_val.npy            / y_val.npy")
print("  X_test.npy           / y_test.npy")
print(f"\nSplits in {SPLIT_DIR}:")
print("  split_train.csv / split_val.csv / split_test.csv")