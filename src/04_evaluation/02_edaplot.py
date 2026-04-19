"""EDA for HAM10000 using artifacts produced by src/02_svm/02.py.

This script uses
  - data/split_train.csv, split_val.csv, split_test.csv
  - X_train_raw.npy, y_train_raw.npy, X_val.npy, y_val.npy, X_test.npy, y_test.npy
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

sns.set(style="whitegrid")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
SVM_OUTPUT_DIR = PROJECT_ROOT / "outputs"
OUT_DIR = PROJECT_ROOT / "src" / "04_evaluation" / "EDA_plots_v2"


def _load_splits(split_dir: Path) -> dict[str, pd.DataFrame]:
    split_files = {
        "train": split_dir / "split_train.csv",
        "val": split_dir / "split_val.csv",
        "test": split_dir / "split_test.csv",
    }

    splits: dict[str, pd.DataFrame] = {}
    for name, path in split_files.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing split CSV: {path}")
        df = pd.read_csv(path)
        if "binary_label" not in df.columns:
            raise ValueError(f"Missing binary_label column in {path}")
        splits[name] = df

    return splits


def _load_feature_arrays(output_dir: Path) -> dict[str, np.ndarray]:
    arrays = {
        "X_train_raw": np.load(output_dir / "X_train_raw.npy"),
        "y_train_raw": np.load(output_dir / "y_train_raw.npy"),
        "X_val": np.load(output_dir / "X_val.npy"),
        "y_val": np.load(output_dir / "y_val.npy"),
        "X_test": np.load(output_dir / "X_test.npy"),
        "y_test": np.load(output_dir / "y_test.npy"),
    }
    return arrays


def task1_split_ratio_validation(splits: dict[str, pd.DataFrame], out_dir: Path) -> None:
    records = []
    for split_name, df in splits.items():
        local = df[["binary_label"]].copy()
        local["split"] = split_name
        records.append(local)
    df_plot = pd.concat(records, ignore_index=True)

    split_order = ["train", "val", "test"]

    plt.figure(figsize=(8, 5))
    sns.countplot(
        data=df_plot,
        x="split",
        hue="binary_label",
        order=split_order,
        hue_order=[0, 1],
    )
    plt.title("Task 1: Class Counts per Split (from 02.py CSVs)")
    plt.xlabel("Split")
    plt.ylabel("Image Count")
    plt.legend(title="binary_label", labels=["Benign (0)", "Malignant (1)"])
    plt.tight_layout()
    plt.savefig(out_dir / "task1_split_class_counts.png", dpi=180)
    plt.close()

    split_ratio = (
        pd.crosstab(df_plot["split"], df_plot["binary_label"], normalize="index")
        .reindex(split_order)
        .fillna(0.0)
    )
    split_ratio.columns = ["benign", "malignant"]

    overall_ratio = df_plot["binary_label"].value_counts(normalize=True)
    overall_ratio = overall_ratio.reindex([0, 1]).fillna(0.0)
    overall_ratio.index = ["benign", "malignant"]

    deviation = split_ratio.subtract(overall_ratio, axis=1) * 100.0
    dev_plot = deviation.reset_index().melt(
        id_vars="split",
        value_vars=["benign", "malignant"],
        var_name="class",
        value_name="pct_point_deviation",
    )

    plt.figure(figsize=(8, 5))
    sns.barplot(data=dev_plot, x="split", y="pct_point_deviation", hue="class")
    plt.axhline(0.0, color="black", linewidth=1)
    plt.title("Task 1: Ratio Deviation")
    plt.xlabel("Split")
    plt.ylabel("Deviation (percentage points)")
    plt.tight_layout()
    plt.savefig(out_dir / "task1_split_ratio_deviation.png", dpi=180)
    plt.close()

    ratio_table = split_ratio.copy()
    ratio_table.loc["overall"] = overall_ratio
    ratio_table.to_csv(out_dir / "task1_split_ratio_table.csv")


def task2_pca_explained_variance(X_train_raw: np.ndarray, out_dir: Path) -> None:
    X_scaled = StandardScaler().fit_transform(X_train_raw)
    pca = PCA(random_state=42)
    pca.fit(X_scaled)

    explained = pca.explained_variance_ratio_
    cumulative = np.cumsum(explained)
    n95 = int(np.searchsorted(cumulative, 0.95) + 1)

    x = np.arange(1, len(cumulative) + 1)
    plt.figure(figsize=(9, 5))
    plt.plot(x, cumulative, linewidth=2, label="Cumulative explained variance")
    plt.axhline(0.95, linestyle="--", linewidth=1.5, label="95% target")
    plt.axvline(n95, linestyle="--", linewidth=1.5, label=f"n_components={n95}")
    plt.title("Task 2: PCA on 02.py Features (HSV + GLCM)")
    plt.xlabel("Number of components")
    plt.ylabel("Cumulative explained variance")
    plt.ylim(0, 1.02)
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        out_dir / "task2_pca_explained_variance_curve.png",
        dpi=180,
        bbox_inches="tight",
        pad_inches=0.2,
    )
    plt.close()

    pd.DataFrame(
        {
            "component": x,
            "explained_variance_ratio": explained,
            "cumulative_explained_variance": cumulative,
        }
    ).to_csv(out_dir / "task2_pca_explained_variance_table.csv", index=False)


def task3_hsv_histogram_comparison(
    X_train_raw: np.ndarray,
    y_train_raw: np.ndarray,
    out_dir: Path,
    bins: int = 16,
) -> None:
    # X_train_raw columns [0:16]=H, [16:32]=S, [32:48]=V, [48:51]=GLCM.
    if X_train_raw.shape[1] < 48:
        raise ValueError("Expected at least 48 HSV features in X_train_raw.")

    hsv = X_train_raw[:, : 3 * bins]
    channel_names = ["H", "S", "V"]
    class_names = {0: "benign", 1: "malignant"}

    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)

    for c_idx, ax in enumerate(axes):
        start = c_idx * bins
        end = start + bins
        bin_x = np.arange(1, bins + 1)

        for label in [0, 1]:
            cls_mask = y_train_raw == label
            if np.any(cls_mask):
                mean_hist = hsv[cls_mask, start:end].mean(axis=0)
                ax.plot(bin_x, mean_hist, linewidth=2, label=class_names[label])

        ax.set_title(f"{channel_names[c_idx]} histogram bins")
        ax.set_xlabel("Bin index")
        if c_idx == 0:
            ax.set_ylabel("Average normalized frequency")
        ax.grid(alpha=0.3)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.04))
    fig.suptitle("Task 3: HSV Histogram Distribution by Class (from 02.py features)", y=1.09)
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.9])
    fig.savefig(
        out_dir / "task3_hsv_histogram_benign_vs_malignant.png",
        dpi=180,
        bbox_inches="tight",
        pad_inches=0.25,
    )
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    svm_output_dir = SVM_OUTPUT_DIR
    splits = _load_splits(DATA_DIR)
    arrays = _load_feature_arrays(svm_output_dir)

    task1_split_ratio_validation(splits, OUT_DIR)
    task2_pca_explained_variance(arrays["X_train_raw"], OUT_DIR)
    task3_hsv_histogram_comparison(
        arrays["X_train_raw"],
        arrays["y_train_raw"],
        OUT_DIR,
        bins=16,
    )


if __name__ == "__main__":
    main()
