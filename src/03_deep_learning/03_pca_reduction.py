import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

OUTPUT_DIR   = "outputs/"
PLOT_DIR     = "src/03_deep_learning/plots/"
RANDOM_STATE = 42
BATCH_SIZE   = 8
IMG_SIZE     = 224

os.makedirs(PLOT_DIR, exist_ok=True)

# ─────────────────────────────────────────────
# TASK 1 — PCA Dimensionality Reduction
# ─────────────────────────────────────────────
# PCA is applied to the balanced training matrix (54 features).
# We test n_components = 50 and 100 to find the best trade-off between
# variance retained and dimensionality reduction.
#
# Fitting the scaler and PCA on training data ONLY.
# Apply (transform) to val and test without refitting — otherwise you
# leak information from val/test into the preprocessing step.
# ─────────────────────────────────────────────

print("=" * 60)
print("TASK 1 — PCA Dimensionality Reduction")
print("=" * 60)

# ── Load feature arrays ───────────────────────────────────────────────────────

X_train = np.load(os.path.join(OUTPUT_DIR, "X_train_balanced_full.npy"))
y_train = np.load(os.path.join(OUTPUT_DIR, "y_train_balanced_full.npy"))
X_val   = np.load(os.path.join(OUTPUT_DIR, "X_val_full.npy"))
y_val   = np.load(os.path.join(OUTPUT_DIR, "y_val.npy"))
X_test  = np.load(os.path.join(OUTPUT_DIR, "X_test_full.npy"))
y_test  = np.load(os.path.join(OUTPUT_DIR, "y_test.npy"))

print(f"X_train : {X_train.shape}")
print(f"X_val   : {X_val.shape}")
print(f"X_test  : {X_test.shape}")

# ── Step 1: Standardize ───────────────────────────────────────────────────────
# PCA is sensitive to feature scale. HSV values are normalized [0,1] but GLCM
# contrast can be in the hundreds — standardizing puts all features on equal
# footing before PCA finds its principal components.

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)   # fit + transform on train
X_val_scaled   = scaler.transform(X_val)          # transform only
X_test_scaled  = scaler.transform(X_test)         # transform only

# ── Step 2: Fit full PCA to get the explained variance curve ─────────────────
# Fit with all 54 components first so we can plot the full curve and make an
# informed decision on the right number of components.

pca_full = PCA(n_components=None, random_state=RANDOM_STATE)
pca_full.fit(X_train_scaled)

cumulative_variance = np.cumsum(pca_full.explained_variance_ratio_)
n_features = X_train_scaled.shape[1]

print(f"\nTotal features before PCA : {n_features}")
for threshold in [0.80, 0.90, 0.95, 0.99]:
    n_needed = np.argmax(cumulative_variance >= threshold) + 1
    print(f"  Components needed for {threshold:.0%} variance : {n_needed}")

# ── Step 3: Plot explained variance curve ─────────────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("PCA Explained Variance — HAM10000 Feature Matrix (54 features)", fontsize=13)

# Left: individual explained variance per component
axes[0].bar(range(1, n_features + 1), pca_full.explained_variance_ratio_, color="steelblue", alpha=0.8)
axes[0].set_xlabel("Principal Component")
axes[0].set_ylabel("Explained Variance Ratio")
axes[0].set_title("Per-Component Explained Variance")
axes[0].set_xlim(0, n_features + 1)

# Right: cumulative explained variance with threshold lines
axes[1].plot(range(1, n_features + 1), cumulative_variance, marker="o", markersize=3,
             color="steelblue", linewidth=2)
for threshold, color, label in [(0.80, "orange", "80%"), (0.90, "green", "90%"), (0.95, "red", "95%")]:
    axes[1].axhline(y=threshold, color=color, linestyle="--", linewidth=1.2, label=f"{label} variance")
axes[1].set_xlabel("Number of Components")
axes[1].set_ylabel("Cumulative Explained Variance")
axes[1].set_title("Cumulative Explained Variance")
axes[1].legend()
axes[1].set_xlim(1, n_features)
axes[1].set_ylim(0.4, 1.01)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plot_path = os.path.join(PLOT_DIR, "pca_explained_variance.png")
plt.savefig(plot_path, dpi=150, bbox_inches="tight")
plt.show()
print(f"\nVariance curve saved to {plot_path}")

# ── Step 4: Apply PCA with n_components = 50 and 100 ─────────────────────────
# Note: our feature matrix only has 54 dimensions, so n=100 will be capped
# at 54 automatically. We test both for completeness and to confirm behavior.

results = {}

for n in [50, 54]:    # 54 is the max; 100 would just equal 54 here
    pca = PCA(n_components=n, random_state=RANDOM_STATE)
    X_tr = pca.fit_transform(X_train_scaled)
    X_v  = pca.transform(X_val_scaled)
    X_te = pca.transform(X_test_scaled)

    variance_retained = pca.explained_variance_ratio_.sum()
    results[n] = {
        "pca":       pca,
        "X_train":   X_tr,
        "X_val":     X_v,
        "X_test":    X_te,
        "variance":  variance_retained,
    }
    print(f"\n  PCA n={n}: shape={X_tr.shape}  |  variance retained={variance_retained:.4f} ({variance_retained:.2%})")

# ── Step 5: Select best n and save ───────────────────────────────────────────
# Since our feature matrix is only 54-dimensional, the meaningful test is
# which threshold (80%, 90%, 95%) gives the best SVM performance in Week 4.
# We save n=50 as the primary reduced matrix for SVM training.

best_n = 50
best   = results[best_n]

np.save(os.path.join(OUTPUT_DIR, "X_train_pca.npy"), best["X_train"])
np.save(os.path.join(OUTPUT_DIR, "y_train_pca.npy"), y_train)
np.save(os.path.join(OUTPUT_DIR, "X_val_pca.npy"),   best["X_val"])
np.save(os.path.join(OUTPUT_DIR, "y_val_pca.npy"),   y_val)
np.save(os.path.join(OUTPUT_DIR, "X_test_pca.npy"),  best["X_test"])
np.save(os.path.join(OUTPUT_DIR, "y_test_pca.npy"),  y_test)

print(f"\nPCA arrays (n={best_n}) saved to {OUTPUT_DIR}")
print("  X_train_pca.npy / y_train_pca.npy")
print("  X_val_pca.npy   / y_val_pca.npy")
print("  X_test_pca.npy  / y_test_pca.npy")

# ─────────────────────────────────────────────
# TASK 2 — ResNet-50 Test Forward Pass + CUDA Memory Diagnostics
# ─────────────────────────────────────────────
# Goal: confirm ResNet-50 loads correctly and a batch of 8 images passes
# through without error. Log CUDA memory before/during/after to catch
# any OOM issues early before full training in Week 4.
# ─────────────────────────────────────────────

print("\n" + "=" * 60)
print("TASK 2 — ResNet-50 Test Forward Pass (batch=8)")
print("=" * 60)

# ─────────────────────────────────────────────
# TASK 2 — ResNet-50 Test Forward Pass + CUDA Memory Diagnostics
# ─────────────────────────────────────────────
# Goal: confirm ResNet-50 loads correctly and a batch of 8 images passes
# through without error. Log CUDA memory before/during/after to catch
# any OOM issues early before full training in Week 4.
# ─────────────────────────────────────────────
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Device : Apple MPS (Metal GPU)")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
    print("Device : CPU — no GPU acceleration available")

def log_cuda_memory(label: str):
    if device.type == "mps":
        # MPS doesn't expose memory stats the same way — just note the step
        print(f"  [{label}] Running on MPS (memory stats not available)")
    elif device.type == "cuda":
        allocated = torch.cuda.memory_allocated(0) / 1e6
        reserved  = torch.cuda.memory_reserved(0) / 1e6
        print(f"  [{label}] Allocated: {allocated:.1f} MB  |  Reserved: {reserved:.1f} MB")

# ── Load ResNet-50 ────────────────────────────────────────────────────────────

print("\nLoading ResNet-50 (pretrained ImageNet weights)...")
model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

# Modify final layer for binary classification (benign vs malignant)
in_features = model.fc.in_features
model.fc = torch.nn.Linear(in_features, 2)
model = model.to(device)
model.eval()

total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters : {total_params:,}")
log_cuda_memory("After model load")

# ── Construct a synthetic batch ───────────────────────────────────────────────
# Simulates what a real DataLoader would provide — random tensors in the
# same shape and value range as normalized ImageNet images.

transform = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)

print(f"\nCreating synthetic batch: {BATCH_SIZE} × 3 × {IMG_SIZE} × {IMG_SIZE}")
dummy_batch = torch.rand(BATCH_SIZE, 3, IMG_SIZE, IMG_SIZE)
dummy_batch = torch.stack([transform(img) for img in dummy_batch])
dummy_batch = dummy_batch.to(device)
log_cuda_memory("After batch to device")

# ── Forward pass ──────────────────────────────────────────────────────────────

print("\nRunning forward pass...")
try:
    with torch.no_grad():
        output = model(dummy_batch)
    log_cuda_memory("After forward pass")

    print(f"\n  Input shape  : {dummy_batch.shape}")
    print(f"  Output shape : {output.shape}   (expected: [{BATCH_SIZE}, 2])")
    print(f"  Output (logits sample):\n{output[:2].cpu().numpy()}")
    print("\n  Forward pass SUCCESSFUL — no CUDA memory issues detected.")

except torch.cuda.OutOfMemoryError as e:
    print(f"\n  CUDA OUT OF MEMORY ERROR: {e}")
    print("  Suggestions:")
    print("    • Reduce batch size below 8")
    print("    • Use torch.cuda.empty_cache() between batches")
    print("    • Enable gradient checkpointing for training")
    print("    • Consider mixed precision: torch.cuda.amp.autocast()")

except Exception as e:
    print(f"\n  Unexpected error during forward pass: {e}")

finally:
    # Always clean up GPU memory
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps":
        torch.mps.empty_cache()