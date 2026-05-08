"""
-----------------
Loads SVM, ResNet-50, and EfficientNet-B0,
and produces head-to-head metric comparison plots
(grouped bar + radar chart).

Uses preprocessed test data from the pipeline.

Usage:
    python 03_edaplot.py
"""

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import accuracy_score, auc, f1_score, precision_recall_curve, precision_score, recall_score, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
EVAL_DIR = PROJECT_ROOT / "src" / "04_evaluation"

METADATA_PATH = DATA_DIR / "metadata_labeled.csv"
SPLIT_TEST_PATH = DATA_DIR / "split_test.csv"
IMAGE_DIR_CANDIDATES = [
    DATA_DIR / "HAM10000_images_hair_removed",
    DATA_DIR / "HAM10000_images_normalized",
    DATA_DIR / "HAM10000_images_resized",
]

SVM_PATH = PROJECT_ROOT / "src" / "02_svm" / "outputs" / "svm_best_model.joblib"
RESNET_PATH = PROJECT_ROOT / "outputs" / "checkpoints" / "resnet50_best_20_epochs.pth"
EFFICIENTNET_PATH = PROJECT_ROOT / "outputs" / "checkpoints" / "efficientnet_b0_best_20_epochs.pth"

X_TEST_PATH = OUTPUT_DIR / "X_test.npy"
Y_TEST_PATH = OUTPUT_DIR / "y_test.npy"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")


print("Loading test data from pipeline outputs...")
X_test = np.load(X_TEST_PATH)
y_test = np.load(Y_TEST_PATH)
print(f"Test data shape: X={X_test.shape}, y={y_test.shape}")
X_test_flat = X_test.reshape(X_test.shape[0], -1)


def load_svm_bundle(model_path):
    """Load the saved SVM artifact and normalize the expected structure."""
    artifact = joblib.load(model_path)

    if isinstance(artifact, dict):
        svm_model = artifact.get("model", artifact.get("svm"))
        pca = artifact.get("pca")
    else:
        svm_model = artifact
        pca = None

    if svm_model is None:
        raise ValueError(
            f"Unsupported SVM artifact format at {model_path}. Expected a saved model or a bundle with a 'model' key."
        )

    return svm_model, pca


def resolve_image_dir():
    for candidate in IMAGE_DIR_CANDIDATES:
        if candidate.exists() and any(candidate.iterdir()):
            return candidate

    raise FileNotFoundError(
        "Could not find an image directory for deep-model evaluation. "
        f"Checked: {', '.join(str(path) for path in IMAGE_DIR_CANDIDATES)}"
    )


def resolve_image_path(image_id):
    for candidate in IMAGE_DIR_CANDIDATES:
        image_path = candidate / f"{image_id}.jpg"
        if image_path.exists():
            return image_path
    return None


def load_test_dataframe():
    if SPLIT_TEST_PATH.exists():
        return pd.read_csv(SPLIT_TEST_PATH)

    if not METADATA_PATH.exists():
        raise FileNotFoundError(
            f"Missing test split CSV and metadata fallback: {SPLIT_TEST_PATH}, {METADATA_PATH}"
        )

    df = pd.read_csv(METADATA_PATH)
    if "binary_label" not in df.columns:
        if "dx" not in df.columns:
            raise ValueError("metadata_labeled.csv must contain either binary_label or dx")
        df["binary_label"] = df["dx"].apply(lambda value: 1 if value in {"akiec", "bcc", "mel"} else 0)

    _, test_df = train_test_split(
        df,
        test_size=0.15,
        stratify=df["binary_label"],
        random_state=42,
    )

    test_df = test_df.reset_index(drop=True)
    existing_mask = test_df["image_id"].apply(lambda image_id: resolve_image_path(image_id) is not None)
    missing_count = int((~existing_mask).sum())
    if missing_count:
        print(f"Skipping {missing_count} test rows with missing image files.")
    return test_df.loc[existing_mask].reset_index(drop=True)


class HAMImageDataset(Dataset):
    def __init__(self, split_df, image_dir):
        self.df = split_df.reset_index(drop=True)
        self.image_dir = Path(image_dir)
        self.image_dirs = [self.image_dir, *[candidate for candidate in IMAGE_DIR_CANDIDATES if candidate != self.image_dir]]
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        image_path = None
        for candidate_dir in self.image_dirs:
            candidate_path = candidate_dir / f"{row['image_id']}.jpg"
            if candidate_path.exists():
                image_path = candidate_path
                break

        if image_path is None:
            raise FileNotFoundError(f"Missing image file for {row['image_id']} in any known image directory")

        image = Image.open(image_path).convert("RGB")
        return self.transform(image), int(row["binary_label"])


def build_deep_test_loader(batch_size=32):
    image_dir = resolve_image_dir()
    test_df = load_test_dataframe()
    dataset = HAMImageDataset(test_df, image_dir)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)


def build_resnet50_for_eval():
    model = models.resnet50(weights=None)
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(in_features, 512),
        nn.LayerNorm(512),
        nn.ReLU(),
        nn.Dropout(p=0.5),
        nn.Linear(512, 2),
    )
    return model


def build_efficientnet_b0_for_eval():
    model = models.efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.4),
        nn.Linear(in_features, 2),
    )
    return model


def load_torch_checkpoint(model, checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=True)
    state_dict = checkpoint.get("model_state") if isinstance(checkpoint, dict) else checkpoint
    if state_dict is None:
        raise ValueError(f"Unsupported checkpoint format: {checkpoint_path}")
    model.load_state_dict(state_dict)
    return model.to(DEVICE)


def get_deep_preds(model, loader):
    """Get predictions from deep model on an image DataLoader."""
    model.eval()
    all_preds = []
    all_probs = []
    all_true = []

    with torch.no_grad():
        for batch_X, batch_y in loader:
            batch_X = batch_X.to(DEVICE)
            out = model(batch_X)
            probs = torch.softmax(out, dim=1)[:, 1].cpu().numpy()
            preds = (probs >= 0.5).astype(int)
            all_probs.extend(probs)
            all_preds.extend(preds)
            all_true.extend(batch_y.numpy())

    return np.array(all_true), np.array(all_preds), np.array(all_probs)


def compute_metrics(y_true, y_pred, y_prob):
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = auc(recall, precision)
    return {
        "Accuracy": round(accuracy_score(y_true, y_pred), 4),
        "Precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "Recall": round(recall_score(y_true, y_pred, zero_division=0), 4),
        "F1": round(f1_score(y_true, y_pred, zero_division=0), 4),
        "ROC-AUC": round(roc_auc_score(y_true, y_prob), 4),
        "PR-AUC": round(pr_auc, 4),
    }


models_results = {}
models_predictions = {}  # Store y_true, y_prob for ROC curves
deep_test_loader = None


if SVM_PATH.exists():
    print("\nEvaluating SVM...")
    svm_model, svm_pca = load_svm_bundle(SVM_PATH)
    X_test_svm = svm_pca.transform(X_test_flat) if svm_pca is not None else X_test_flat
    svm_probs = svm_model.predict_proba(X_test_svm)[:, 1]
    svm_preds = (svm_probs >= 0.5).astype(int)
    svm_metrics = compute_metrics(y_test, svm_preds, svm_probs)
    models_results["SVM"] = svm_metrics
    models_predictions["SVM"] = (y_test, svm_probs)
    print(f"  → {svm_metrics}")


if RESNET_PATH.exists():
    print("\nEvaluating ResNet-50...")
    if deep_test_loader is None:
        deep_test_loader = build_deep_test_loader()
    resnet = load_torch_checkpoint(build_resnet50_for_eval(), RESNET_PATH)
    y_true_r, y_pred_r, y_prob_r = get_deep_preds(resnet, deep_test_loader)
    resnet_metrics = compute_metrics(y_true_r, y_pred_r, y_prob_r)
    models_results["ResNet-50"] = resnet_metrics
    models_predictions["ResNet-50"] = (y_true_r, y_prob_r)
    print(f"  → {resnet_metrics}")


if EFFICIENTNET_PATH.exists():
    print("\nEvaluating EfficientNet-B0...")
    if deep_test_loader is None:
        deep_test_loader = build_deep_test_loader()
    effnet = load_torch_checkpoint(build_efficientnet_b0_for_eval(), EFFICIENTNET_PATH)
    y_true_e, y_pred_e, y_prob_e = get_deep_preds(effnet, deep_test_loader)
    effnet_metrics = compute_metrics(y_true_e, y_pred_e, y_prob_e)
    models_results["EfficientNet-B0"] = effnet_metrics
    models_predictions["EfficientNet-B0"] = (y_true_e, y_prob_e)
    print(f"  → {effnet_metrics}")


metrics_list = ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "PR-AUC"]
colors = ["#4C72B0", "#DD8452", "#55A868", "#C5B0D5"]

x = np.arange(len(metrics_list))
width = 0.25

fig, ax = plt.subplots(figsize=(12, 6))
fig.patch.set_facecolor("#F7F7F7")
ax.set_facecolor("#F7F7F7")

for i, (model_name, metrics) in enumerate(models_results.items()):
    vals = [metrics[m] for m in metrics_list]
    offset = (i - (len(models_results) - 1) / 2) * width
    bars = ax.bar(
        x + offset,
        vals,
        width,
        label=model_name,
        color=colors[i % len(colors)],
        edgecolor="white",
        linewidth=0.8,
    )
    for bar, val in zip(bars, vals):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{val:.2f}",
            ha="center",
            va="bottom",
            fontsize=8,
            color="#333333",
        )

ax.set_xticks(x)
ax.set_xticklabels(metrics_list, fontsize=11)
ax.set_ylabel("Score", fontsize=11)
ax.set_title("Model Comparison — Test Set Metrics", fontsize=14, fontweight="bold", pad=15)
ax.set_ylim(0, 1.15)
ax.legend(fontsize=10)
ax.spines[["top", "right"]].set_visible(False)
ax.yaxis.grid(True, linestyle="--", alpha=0.5)
ax.set_axisbelow(True)

plt.tight_layout()
output_path = EVAL_DIR / "comparison_bar.png"
plt.savefig(output_path, dpi=150, bbox_inches="tight")
print(f"Saved: {output_path}")
plt.close()


num_vars = len(metrics_list)
angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
angles += angles[:1]

fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
fig.patch.set_facecolor("#F7F7F7")
ax.set_facecolor("#F7F7F7")

for i, (model_name, metrics) in enumerate(models_results.items()):
    vals = [metrics[m] for m in metrics_list]
    vals_closed = vals + vals[:1]
    ax.plot(angles, vals_closed, color=colors[i % len(colors)], linewidth=2, label=model_name)
    ax.fill(angles, vals_closed, color=colors[i % len(colors)], alpha=0.1)

ax.set_thetagrids(np.degrees(angles[:-1]), metrics_list, fontsize=10)
ax.set_ylim(0, 1)
ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=8, color="grey")
ax.set_title("Model Comparison — Radar Chart", fontsize=14, fontweight="bold", pad=20)
ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.15), fontsize=10)
ax.spines["polar"].set_visible(False)

plt.tight_layout()
output_path = EVAL_DIR / "comparison_radar.png"
plt.savefig(output_path, dpi=150, bbox_inches="tight")
print(f"Saved: {output_path}")
plt.close()


# Generate ROC curves
fig, ax = plt.subplots(figsize=(10, 8))
fig.patch.set_facecolor("#FFFFFF")
ax.set_facecolor("#F7F7F7")

roc_colors = {"SVM": "#4C72B0", "ResNet-50": "#DD8452", "EfficientNet-B0": "#55A868"}

for model_name, (y_true, y_prob) in models_predictions.items():
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc_score = models_results[model_name]["ROC-AUC"]
    ax.plot(
        fpr,
        tpr,
        linewidth=2.5,
        label=f"{model_name} (AUC = {auc_score:.4f})",
        color=roc_colors.get(model_name, "#000000"),
    )

# Plot diagonal (random classifier)
ax.plot([0, 1], [0, 1], "k--", linewidth=1.5, label="Random Classifier (AUC = 0.5000)", alpha=0.7)

ax.set_xlabel("False Positive Rate", fontsize=12, fontweight="bold")
ax.set_ylabel("True Positive Rate", fontsize=12, fontweight="bold")
ax.set_title("ROC Curves — Model Comparison (Test Set)", fontsize=14, fontweight="bold", pad=15)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.grid(True, linestyle="--", alpha=0.3)
ax.legend(loc="lower right", fontsize=11)
ax.spines[["top", "right"]].set_visible(False)

plt.tight_layout()
output_path = EVAL_DIR / "roc_curves.png"
plt.savefig(output_path, dpi=150, bbox_inches="tight")
print(f"Saved: {output_path}")
plt.close()


# Generate Precision-Recall curves
fig, ax = plt.subplots(figsize=(10, 8))
fig.patch.set_facecolor("#FFFFFF")
ax.set_facecolor("#F7F7F7")

pr_colors = {"SVM": "#4C72B0", "ResNet-50": "#DD8452", "EfficientNet-B0": "#55A868"}

for model_name, (y_true, y_prob) in models_predictions.items():
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    pr_auc_score = models_results[model_name]["PR-AUC"]
    ax.plot(
        recall,
        precision,
        linewidth=2.5,
        label=f"{model_name} (PR-AUC = {pr_auc_score:.4f})",
        color=pr_colors.get(model_name, "#000000"),
    )

# Plot baseline (no-skill classifier)
no_skill = np.sum(y_test) / len(y_test)
ax.plot([0, 1], [no_skill, no_skill], "k--", linewidth=1.5, label=f"No Skill (PR-AUC = {no_skill:.4f})", alpha=0.7)

ax.set_xlabel("Recall", fontsize=12, fontweight="bold")
ax.set_ylabel("Precision", fontsize=12, fontweight="bold")
ax.set_title("Precision-Recall Curves — Model Comparison (Test Set)", fontsize=14, fontweight="bold", pad=15)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.grid(True, linestyle="--", alpha=0.3)
ax.legend(loc="lower left", fontsize=11)
ax.spines[["top", "right"]].set_visible(False)

plt.tight_layout()
output_path = EVAL_DIR / "pr_curves.png"
plt.savefig(output_path, dpi=150, bbox_inches="tight")
print(f"Saved: {output_path}")
plt.close()

print("\nEvaluation complete!")