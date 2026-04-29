import numpy as np
import cv2
import pandas as pd
import os
from tqdm import tqdm
from imblearn.over_sampling import SMOTE

IMAGE_DIR   = "data/HAM10000_images_clahe/"
OUTPUT_DIR  = "outputs/"
SPLIT_DIR   = "data/"
RANDOM_STATE = 42
IMG_SIZE     = 128

def extract_shape_features(img_bgr: np.ndarray) -> np.ndarray:
    """
    Extract shape statistics from the dominant lesion contour.

    Pipeline:
        1. Convert to grayscale and threshold to isolate the lesion
        2. Find contours and select the largest (assumed to be the lesion)
        3. Compute aspect ratio, compactness, and border irregularity

    Args:
        img_bgr: Image as a (H, W, 3) BGR numpy array.

    Returns:
        1D numpy array: [aspect_ratio, compactness, border_irregularity]
        Returns [1.0, 1.0, 0.0] as a safe fallback if no contour is found.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # Otsu thresholding to separate lesion from background
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Morphological closing to fill small holes in the lesion mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Safe fallback if no contour detected
    if not contours:
        return np.array([1.0, 1.0, 0.0])

    # Use the largest contour as the lesion boundary
    lesion_contour = max(contours, key=cv2.contourArea)

    area      = cv2.contourArea(lesion_contour)
    perimeter = cv2.arcLength(lesion_contour, closed=True)

    # ── Aspect Ratio ──────────────────────────────────────────────────────────
    x, y, w, h = cv2.boundingRect(lesion_contour)
    aspect_ratio = w / h if h > 0 else 1.0

    # ── Compactness ───────────────────────────────────────────────────────────
    # Ranges (0, 1] — circle = 1.0, irregular shapes approach 0
    if perimeter > 0:
        compactness = (4 * np.pi * area) / (perimeter ** 2)
    else:
        compactness = 1.0

    # ── Border Irregularity ───────────────────────────────────────────────────
    # Compute centroid, then measure std/mean of radial distances
    M = cv2.moments(lesion_contour)
    if M["m00"] > 0:
        cx = M["m10"] / M["m00"]
        cy = M["m01"] / M["m00"]
        points = lesion_contour[:, 0, :]          # shape: (N, 2)
        distances = np.sqrt((points[:, 0] - cx) ** 2 + (points[:, 1] - cy) ** 2)
        mean_dist = distances.mean()
        border_irregularity = distances.std() / mean_dist if mean_dist > 0 else 0.0
    else:
        border_irregularity = 0.0

    return np.array([aspect_ratio, compactness, border_irregularity])


def load_image(image_id: str, image_dir: str, size: int = IMG_SIZE):
    """Load a preprocessed image by image_id. Returns None if not found."""
    filepath = os.path.join(image_dir, f"{image_id}.jpg")
    if not os.path.exists(filepath):
        return None
    img = cv2.imread(filepath)
    if img is None:
        return None
    if img.shape[:2] != (size, size):
        img = cv2.resize(img, (size, size))
    return img

def extract_shape_for_split(split_df: pd.DataFrame, image_dir: str, split_name: str):
    """
    Run shape extraction over an entire split.

    Returns:
        X_shape: np.ndarray of shape (n_valid_samples, 3)
        valid_ids: list of image_ids that loaded successfully
                   (used to align with HSV/GLCM arrays if needed)
    """
    X_shape = []
    valid_ids = []
    skipped = 0

    print(f"\nExtracting shape features for {split_name} ({len(split_df)} images)...")
    for _, row in tqdm(split_df.iterrows(), total=len(split_df), desc=split_name):
        img = load_image(row["image_id"], image_dir)
        if img is None:
            skipped += 1
            continue
        X_shape.append(extract_shape_features(img))
        valid_ids.append(row["image_id"])

    if skipped:
        print(f"  Warning: {skipped} images skipped.")

    X_shape = np.array(X_shape)
    print(f"  {split_name} shape feature matrix: {X_shape.shape}")
    return X_shape, valid_ids

# ─────────────────────────────────────────────
# TASK 2 — Load HSV + GLCM Arrays and Concatenate
# ─────────────────────────────────────────────
# Feature layout after concatenation:
#   Indices  0–47  → HSV color histogram  (48 features)
#   Indices 48–50  → GLCM texture         ( 3 features)
#   Indices 51–53  → Shape statistics     ( 3 features)
#   Total: 54 features per image
# ─────────────────────────────────────────────

print("=" * 60)
print("TASK 2 — Loading HSV + GLCM Feature Arrays")
print("=" * 60)

X_train_raw = np.load(os.path.join(OUTPUT_DIR, "X_train_raw.npy"))
y_train     = np.load(os.path.join(OUTPUT_DIR, "y_train_raw.npy"))
X_val       = np.load(os.path.join(OUTPUT_DIR, "X_val.npy"))
y_val       = np.load(os.path.join(OUTPUT_DIR, "y_val.npy"))
X_test      = np.load(os.path.join(OUTPUT_DIR, "X_test.npy"))
y_test      = np.load(os.path.join(OUTPUT_DIR, "y_test.npy"))

print(f"  X_train_raw : {X_train_raw.shape}")
print(f"  X_val       : {X_val.shape}")
print(f"  X_test      : {X_test.shape}")

# ── Run shape extraction on all three splits ──────────────────────────────────

print("\n" + "=" * 60)
print("TASK 1 — Shape Feature Extraction")
print("=" * 60)

train_df = pd.read_csv(os.path.join(SPLIT_DIR, "split_train.csv"))
val_df   = pd.read_csv(os.path.join(SPLIT_DIR, "split_val.csv"))
test_df  = pd.read_csv(os.path.join(SPLIT_DIR, "split_test.csv"))

X_shape_train, _ = extract_shape_for_split(train_df, IMAGE_DIR, "Train")
X_shape_val,   _ = extract_shape_for_split(val_df,   IMAGE_DIR, "Val")
X_shape_test,  _ = extract_shape_for_split(test_df,  IMAGE_DIR, "Test")

# ── Concatenate: HSV + GLCM + Shape ──────────────────────────────────────────

print("\n" + "=" * 60)
print("TASK 2 — Concatenating Feature Vectors")
print("=" * 60)

X_train_full = np.concatenate([X_train_raw, X_shape_train], axis=1)
X_val_full   = np.concatenate([X_val,       X_shape_val],   axis=1)
X_test_full  = np.concatenate([X_test,      X_shape_test],  axis=1)

print(f"  X_train_full : {X_train_full.shape}   (expected: (n, 54))")
print(f"  X_val_full   : {X_val_full.shape}")
print(f"  X_test_full  : {X_test_full.shape}")

assert X_train_full.shape[1] == 54, f"Expected 54 features, got {X_train_full.shape[1]}"
assert X_val_full.shape[1]   == 54
assert X_test_full.shape[1]  == 54
print("\n  Feature count confirmed: 54 per sample")

# ── Apply SMOTE on the full training matrix (pre-SMOTE) ──────────────────────

print("\n" + "=" * 60)
print("TASK 2 — Applying SMOTE to Full Training Matrix")
print("=" * 60)

smote = SMOTE(random_state=RANDOM_STATE)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train_full, y_train)

print(f"  Before SMOTE: {X_train_full.shape}")
print(f"  After  SMOTE: {X_train_balanced.shape}")

unique, counts = np.unique(y_train_balanced, return_counts=True)
for cls, cnt in zip(unique, counts):
    print(f"    {'Benign' if cls == 0 else 'Malignant'} ({cls}): {cnt} samples")

# ── Save all final arrays ─────────────────────────────────────────────────────

np.save(os.path.join(OUTPUT_DIR, "X_train_balanced_full.npy"), X_train_balanced)
np.save(os.path.join(OUTPUT_DIR, "y_train_balanced_full.npy"), y_train_balanced)
np.save(os.path.join(OUTPUT_DIR, "X_val_full.npy"),            X_val_full)
np.save(os.path.join(OUTPUT_DIR, "y_val.npy"),                 y_val)
np.save(os.path.join(OUTPUT_DIR, "X_test_full.npy"),           X_test_full)
np.save(os.path.join(OUTPUT_DIR, "y_test.npy"),                y_test)

print(f"\nAll arrays saved to {OUTPUT_DIR}")
print("\nHandoff notes:")
print("  → SVM training : load X_train_balanced_full.npy / y_train_balanced_full.npy")
print("  → Evaluation   : load X_val_full.npy and X_test_full.npy")
print("  → Do NOT touch X_test_full / y_test until final evaluation in Week 5")

print("\n" + "=" * 60)
print("TASKS 1 & 2 — COMPLETE")
print("=" * 60)
print("Task 1: Shape features (aspect ratio, compactness, irregularity)  DONE")
print("Task 2: HSV (48) + GLCM (3) + Shape (3) = 54 features            DONE")
print("Task 2: SMOTE applied to full training matrix                      DONE")


'''
def extract_shape_features(image_path):
    # convert image to different color spaces
    img_bgr = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img_bgr,cv2.COLOR_BGR2RGB)
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # ── Segmentation ──────────────────────────────────────────────────────────
    # Use Otsu's thresholding to separate lesion from skin background
    # Gaussian blur first to reduce noise
    blurred = cv2.GaussianBlur(img_gray, (15, 15), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None
    
    # find contours using largest one as lesion boundary
    lesion_contour = max(contours, key=cv2.contourArea)

    # ── Basic Geometry ────────────────────────────────────────────────────────
    area = cv2.contourArea(lesion_contour)
    perimeter = cv2.arcLength(lesion_contour, closed=True)
    x, y, w, h = cv2.boundingRect(lesion_contour)

    if area == 0 or perimeter == 0:
        return None

    # ── Aspect Ratio ──────────────────────────────────────────────────────────
    # width / height of the bounding box
    # values far from 1.0 indicate elongated lesions
    aspect_ratio = w / h

    # ── Compactness (Circularity) ─────────────────────────────────────────────
    # = 4π × area / perimeter²
    # perfect circle = 1.0, irregular shapes < 1.0
    compactness = (4 * np.pi * area) / (perimeter ** 2)

    # ── Extent ───────────────────────────────────────────────────────────────
    # ratio of lesion area to bounding box area
    # low extent = irregular shape that doesn't fill its bounding box
    bounding_box_area = w * h
    extent = area / bounding_box_area if bounding_box_area > 0 else 0

    # ── Solidity ─────────────────────────────────────────────────────────────
    # ratio of lesion area to convex hull area
    # low solidity = concave / irregular border
    hull        = cv2.convexHull(lesion_contour)
    hull_area   = cv2.contourArea(hull)
    solidity    = area / hull_area if hull_area > 0 else 0

    # ── Border Irregularity ───────────────────────────────────────────────────
    # how much the actual perimeter exceeds the convex hull perimeter
    # higher = more irregular/jagged border (clinically significant for melanoma)
    hull_perimeter       = cv2.arcLength(hull, closed=True)
    border_irregularity  = perimeter / hull_perimeter if hull_perimeter > 0 else 0

    # ── Equivalent Diameter ───────────────────────────────────────────────────
    # diameter of a circle with the same area as the lesion
    equiv_diameter = np.sqrt(4 * area / np.pi)

    # ── Ellipse Fit ───────────────────────────────────────────────────────────
    # fit an ellipse to the contour and extract its axes and orientation
    if len(lesion_contour) >= 5:
        (cx, cy), (major_axis, minor_axis), angle = cv2.fitEllipse(lesion_contour)
        ellipse_ratio = minor_axis / major_axis if major_axis > 0 else 0
    else:
        ellipse_ratio = 1.0
        angle         = 0.0

    # ── Hu Moments ───────────────────────────────────────────────────────────
    # 7 invariant shape descriptors — invariant to rotation, scale, translation
    moments    = cv2.moments(lesion_contour)
    hu_moments = cv2.HuMoments(moments).flatten()
    # log-transform to normalise scale
    hu_moments = -np.sign(hu_moments) * np.log10(np.abs(hu_moments) + 1e-10)

    features = {
        "aspect_ratio":        aspect_ratio,
        "compactness":         compactness,
        "extent":              extent,
        "solidity":            solidity,
        "border_irregularity": border_irregularity,
        "equiv_diameter":      equiv_diameter,
        "ellipse_ratio":       ellipse_ratio,
        "lesion_area":         area,
        "perimeter":           perimeter,
    }

    # add Hu moments as individual features
    for i, hu in enumerate(hu_moments):
        features[f"hu_moment_{i+1}"] = hu

    return features

# ── Run on all images ─────────────────────────────────────────────────────────
files = glob.glob("data/HAM10000_images_part_2/*.jpg")

all_features = []
failed       = []

for f in files:
    feats = extract_shape_features(f)
    if feats:
        feats["filename"] = f
        all_features.append(feats)
    else:
        failed.append(f)

df_shape = pd.DataFrame(all_features)
print(df_shape.shape)
print(df_shape.head())

if failed:
    print(f"\n{len(failed)} images failed contour detection:")
    for f in failed:
        print(f"  {f}")


# save for use in model pipeline
df_shape.to_csv("shape_features.csv", index=False)
'''