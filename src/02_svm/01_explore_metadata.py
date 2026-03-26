'''
Explores the HAM10000 metadata CSV to check for:
- Missing values
- Duplicate image IDs
- Duplicate lesion IDs (one lesion, multiple images)
- Patient overlap (same patient_id appearing across multiple records)

to use:
1. Ensure you have run src/data_setup/02_metadata_to_binary_labels.py first.
2. Run this script from src/svm/

output:
- Console report summarizing all findings
'''

import pandas as pd
import os

# ── Paths (relative to src/svm/) ───────────────────────────────────────────
METADATA_PATH = os.path.join("..", "..", "data", "metadata_labeled.csv")

# ── Load ────────────────────────────────────────────────────────────────────
df = pd.read_csv(METADATA_PATH)
print(f"Loaded {len(df)} rows\n")
print("Columns:", df.columns.tolist())
print("\nSample:")
print(df.head(5))

# ── 1. Missing values ────────────────────────────────────────────────────────
print("\n── Missing Values ──────────────────────────────────────────────────")
missing = df.isnull().sum()
print(missing[missing > 0] if missing.any() else "No missing values found.")

# ── 2. Duplicate image IDs ───────────────────────────────────────────────────
print("\n── Duplicate Image IDs ─────────────────────────────────────────────")
dup_images = df[df.duplicated(subset="image_id", keep=False)]
if len(dup_images) > 0:
    print(f"Found {len(dup_images)} duplicate image IDs:")
    print(dup_images[["image_id", "lesion_id", "dx"]])
else:
    print("No duplicate image IDs found.")

# ── 3. Duplicate lesion IDs ──────────────────────────────────────────────────
print("\n── Duplicate Lesion IDs (one lesion, multiple images) ──────────────")
lesion_counts = df.groupby("lesion_id")["image_id"].count()
multi_image_lesions = lesion_counts[lesion_counts > 1]
print(f"Lesions with multiple images: {len(multi_image_lesions)}")
print(f"Total images from multi-image lesions: {multi_image_lesions.sum()}")
print("\nTop 5 lesions by image count:")
print(multi_image_lesions.sort_values(ascending=False).head())

# ── 4. Patient overlap ───────────────────────────────────────────────────────
print("\n── Patient Overlap ─────────────────────────────────────────────────")
if "patient_id" in df.columns:
    patient_counts = df.groupby("patient_id")["image_id"].count()
    print(f"Unique patients:       {len(patient_counts)}")
    print(f"Unique images:         {len(df)}")
    print(f"Avg images/patient:    {patient_counts.mean():.2f}")
    print(f"Max images/patient:    {patient_counts.max()}")
    multi_image_patients = patient_counts[patient_counts > 1]
    print(f"Patients with 2+ images: {len(multi_image_patients)}")
else:
    print("No patient_id column found in metadata.")

# ── Summary ──────────────────────────────────────────────────────────────────
print("\n── Summary ─────────────────────────────────────────────────────────")
print(f"Total images:          {len(df)}")
print(f"Unique lesion IDs:     {df['lesion_id'].nunique()}")
print(f"Unique image IDs:      {df['image_id'].nunique()}")
print(f"Binary label counts:")
print(df["label"].value_counts())