import numpy as np
import cv2
import glob
import pandas as pd
from PIL import Image

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