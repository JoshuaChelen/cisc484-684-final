"""
CISC 484/684 — Machine Learning
Skin Lesion Cancer Classification — HAM10000
Member 2: SVM / Phase 1 Lead
Week 3 Tasks:
    1. PCA dimensionality reduction on training features
    2. Train SVM with RBF kernel; run 5-fold stratified cross-validation
    3. Grid search over C (0.1, 1, 10, 100) and gamma ('scale', 'auto', 1e-3)
    4. Record 5-fold CV mean ± std for Precision, Recall, F1, ROC-AUC
    5. Retrain final SVM on full training set with best hyperparameters
    6. Save trained SVM model to disk using joblib

Inputs (from 02.py outputs):
    02_svm/outputs/X_train_balanced.npy   — SMOTE-balanced training features (HSV+GLCM, 51-dim)
    02_svm/outputs/y_train_balanced.npy   — balanced training labels
    02_svm/outputs/X_val.npy              — validation features (raw, no SMOTE)
    02_svm/outputs/y_val.npy              — validation labels

Outputs:
    02_svm/outputs/pca_model.joblib           — fitted PCA transformer
    02_svm/outputs/svm_best_model.joblib      — final trained SVM bundle
    02_svm/outputs/cv_results_table.csv       — full grid search CV results
    02_svm/outputs/svm_val_metrics.txt        — validation set metrics summary
    02_svm/outputs/confusion_matrix_svm.png   — confusion matrix (val set)
    02_svm/outputs/roc_curve_svm.png          — ROC curve (val set)
"""

# ─────────────────────────────────────────────
# DEPENDENCY CHECK
# ─────────────────────────────────────────────
import subprocess, sys

REQUIRED = [
    ("numpy",      "numpy"),
    ("pandas",     "pandas"),
    ("sklearn",    "scikit-learn"),
    ("joblib",     "joblib"),
    ("matplotlib", "matplotlib"),
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
import warnings
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, ConfusionMatrixDisplay, roc_curve, make_scorer,
)

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# 0.  CONFIGURATION
# ─────────────────────────────────────────────

# Resolve paths relative to this file's location so the script works
# regardless of where it's launched from
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR  = os.path.join(SCRIPT_DIR, "outputs")

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# ─────────────────────────────────────────────
# 1.  LOAD DATA
# ─────────────────────────────────────────────

def load_data():
    print("=" * 60)
    print("Loading Week 2 outputs…")

    paths = {
        "X_train": os.path.join(OUTPUT_DIR, "X_train_balanced.npy"),
        "y_train": os.path.join(OUTPUT_DIR, "y_train_balanced.npy"),
        "X_val":   os.path.join(OUTPUT_DIR, "X_val.npy"),
        "y_val":   os.path.join(OUTPUT_DIR, "y_val.npy"),
    }

    missing = [k for k, v in paths.items() if not os.path.exists(v)]
    if missing:
        raise FileNotFoundError(
            f"Missing files: {missing}\n"
            "Run 02.py first to generate the feature arrays."
        )

    X_train = np.load(paths["X_train"])
    y_train = np.load(paths["y_train"])
    X_val   = np.load(paths["X_val"])
    y_val   = np.load(paths["y_val"])

    print(f"  X_train (balanced) : {X_train.shape}  — "
          f"benign={( y_train==0).sum()}  malignant={(y_train==1).sum()}")
    print(f"  X_val              : {X_val.shape}  — "
          f"benign={(y_val==0).sum()}  malignant={(y_val==1).sum()}")

    return X_train, y_train, X_val, y_val


# ─────────────────────────────────────────────
# 2.  PCA — Dimensionality Reduction
# ─────────────────────────────────────────────
# Our feature vector is only 51-dim (48 HSV + 3 GLCM), so PCA here serves
# two purposes:
#   • Removes any redundant correlation between HSV bins
#   • Keeps the pipeline consistent with the plan so Week 3 Member 3
#     tasks (shape features) can be concatenated and re-run cleanly
#
# We retain enough components to explain 95% of variance.
# ─────────────────────────────────────────────

def apply_pca(X_train, X_val, variance_target=0.95):
    print("\n" + "=" * 60)
    print("PCA — Dimensionality Reduction")
    print("=" * 60)

    # Fit ONLY on training data — transform val separately to prevent leakage
    pca = PCA(n_components=variance_target, random_state=RANDOM_STATE)
    X_train_pca = pca.fit_transform(X_train)
    X_val_pca   = pca.transform(X_val)

    n_components = pca.n_components_
    explained    = pca.explained_variance_ratio_.sum()

    print(f"  Components retained : {n_components}  "
          f"(target ≥{variance_target:.0%} variance)")
    print(f"  Variance explained  : {explained:.4f}")
    print(f"  X_train after PCA   : {X_train_pca.shape}")
    print(f"  X_val   after PCA   : {X_val_pca.shape}")

    # Save PCA model so DL team / Member 4 can apply the same transform
    pca_path = os.path.join(OUTPUT_DIR, "pca_model.joblib")
    joblib.dump(pca, pca_path)
    print(f"  PCA model saved → {pca_path}")

    # ── Explained variance plot ──────────────────────────────────
    fig, ax = plt.subplots(figsize=(6, 4))
    cumulative = np.cumsum(pca.explained_variance_ratio_)
    ax.plot(range(1, len(cumulative) + 1), cumulative,
            marker="o", color="steelblue", lw=2, markersize=4)
    ax.axhline(variance_target, color="tomato", ls="--", lw=1.5,
               label=f"{variance_target:.0%} variance threshold")
    ax.set_xlabel("Number of Components", fontsize=11)
    ax.set_ylabel("Cumulative Explained Variance", fontsize=11)
    ax.set_title("PCA — Cumulative Explained Variance", fontsize=12, fontweight="bold")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    ax.legend(fontsize=10)
    plt.tight_layout()
    plot_path = os.path.join(OUTPUT_DIR, "pca_explained_variance.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"  PCA variance plot   → {plot_path}")

    return X_train_pca, X_val_pca, pca


# ─────────────────────────────────────────────
# 3.  GRID SEARCH + 5-FOLD CROSS-VALIDATION
# ─────────────────────────────────────────────

def run_grid_search(X_train, y_train):
    """
    Grid search over C × gamma using 5-fold stratified CV.
    Primary refit metric: Recall — per TA guidance, False Negatives
    (missed malignant diagnoses) carry the highest clinical risk.
    """
    print("\n" + "=" * 60)
    print("Grid Search — 5-Fold Stratified Cross-Validation")
    print("=" * 60)
    print("  Param grid:")
    print("    C     : [0.1, 1, 10, 100]")
    print("    gamma : ['scale', 'auto', 1e-3]")
    print("  Refit metric: Recall  (minimize False Negatives)")

    param_grid = {
        "C":     [0.1, 1, 10, 100],
        "gamma": ["scale", "auto", 1e-3],
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    # Track all four metrics across every fold
    scoring = {
        "precision": make_scorer(precision_score,  zero_division=0),
        "recall":    make_scorer(recall_score,      zero_division=0),
        "f1":        make_scorer(f1_score,          zero_division=0),
        # decision_function is faster than predict_proba for AUC during CV
        "roc_auc":   make_scorer(roc_auc_score,
                                 response_method="decision_function"),
    }

    svm_base = SVC(
        kernel="rbf",
        probability=True,       # needed for predict_proba at evaluation time
        random_state=RANDOM_STATE,
        # NOTE: class_weight not set here — training data is already balanced
        # via SMOTE from Week 2. Setting class_weight='balanced' on top of
        # SMOTE would double-penalize the majority class.
    )

    grid_search = GridSearchCV(
        estimator=svm_base,
        param_grid=param_grid,
        scoring=scoring,
        refit="recall",
        cv=cv,
        n_jobs=-1,
        verbose=2,
        return_train_score=False,
    )

    grid_search.fit(X_train, y_train)

    print(f"\n  Best params    : {grid_search.best_params_}")
    print(f"  Best CV Recall : {grid_search.best_score_:.4f}")

    return grid_search


# ─────────────────────────────────────────────
# 4.  SAVE & DISPLAY CV RESULTS TABLE
# ─────────────────────────────────────────────

def save_cv_results(grid_search):
    print("\n" + "=" * 60)
    print("Cross-Validation Results")
    print("=" * 60)

    cv_res = grid_search.cv_results_
    rows = []

    for i in range(len(cv_res["params"])):
        rows.append({
            "C":              cv_res["params"][i]["C"],
            "gamma":          cv_res["params"][i]["gamma"],
            "Precision_mean": cv_res["mean_test_precision"][i],
            "Precision_std":  cv_res["std_test_precision"][i],
            "Recall_mean":    cv_res["mean_test_recall"][i],
            "Recall_std":     cv_res["std_test_recall"][i],
            "F1_mean":        cv_res["mean_test_f1"][i],
            "F1_std":         cv_res["std_test_f1"][i],
            "ROC_AUC_mean":   cv_res["mean_test_roc_auc"][i],
            "ROC_AUC_std":    cv_res["std_test_roc_auc"][i],
            "rank_recall":    cv_res["rank_test_recall"][i],
        })

    df = pd.DataFrame(rows).sort_values("rank_recall").reset_index(drop=True)
    csv_path = os.path.join(OUTPUT_DIR, "cv_results_table.csv")
    df.to_csv(csv_path, index=False)
    print(f"  Full CV table saved → {csv_path}")

    # ── Console summary: all 12 combos ───────────────────────────
    print(f"\n  {'C':>6}  {'gamma':>8}  "
          f"{'Recall':>13}  {'F1':>13}  {'Precision':>13}  {'ROC-AUC':>13}")
    print("  " + "-" * 72)
    for _, row in df.iterrows():
        print(f"  {row['C']:>6}  {str(row['gamma']):>8}  "
              f"{row['Recall_mean']:.4f}±{row['Recall_std']:.4f}  "
              f"{row['F1_mean']:.4f}±{row['F1_std']:.4f}  "
              f"{row['Precision_mean']:.4f}±{row['Precision_std']:.4f}  "
              f"{row['ROC_AUC_mean']:.4f}±{row['ROC_AUC_std']:.4f}")

    best = df.iloc[0]
    print(f"\n  ── Best combination (ranked by Recall) ───────────────")
    print(f"     C            : {best['C']}")
    print(f"     gamma        : {best['gamma']}")
    print(f"     Recall       : {best['Recall_mean']:.4f} ± {best['Recall_std']:.4f}  ← PRIMARY")
    print(f"     Precision    : {best['Precision_mean']:.4f} ± {best['Precision_std']:.4f}")
    print(f"     F1-score     : {best['F1_mean']:.4f} ± {best['F1_std']:.4f}")
    print(f"     ROC-AUC      : {best['ROC_AUC_mean']:.4f} ± {best['ROC_AUC_std']:.4f}")
    print(f"  ──────────────────────────────────────────────────────")

    return df


# ─────────────────────────────────────────────
# 5.  RETRAIN FINAL SVM ON FULL TRAINING SET
# ─────────────────────────────────────────────

def train_final_svm(best_params, X_train, y_train):
    """
    Retrain a fresh SVM on ALL training data (not just CV fold subsets)
    using the best hyperparameters from grid search.
    """
    print("\n" + "=" * 60)
    print("Retraining Final SVM on Full Training Set")
    print("=" * 60)
    print(f"  C={best_params['C']},  gamma={best_params['gamma']}")

    final_svm = SVC(
        kernel="rbf",
        C=best_params["C"],
        gamma=best_params["gamma"],
        probability=True,
        random_state=RANDOM_STATE,
    )
    final_svm.fit(X_train, y_train)
    print("  Training complete.")
    return final_svm


# ─────────────────────────────────────────────
# 6.  EVALUATE ON VALIDATION SET
# ─────────────────────────────────────────────

def evaluate_on_validation(model, X_val, y_val):
    print("\n" + "=" * 60)
    print("Validation Set Evaluation")
    print("(Test set stays locked until Week 5)")
    print("=" * 60)

    y_pred = model.predict(X_val)
    y_prob = model.predict_proba(X_val)[:, 1]   # P(malignant)

    precision = precision_score(y_val, y_pred, zero_division=0)
    recall    = recall_score(y_val,    y_pred, zero_division=0)
    f1        = f1_score(y_val,        y_pred, zero_division=0)
    roc_auc   = roc_auc_score(y_val, y_prob)
    cm        = confusion_matrix(y_val, y_pred)
    tn, fp, fn, tp = cm.ravel()
    fn_rate   = fn / (fn + tp) if (fn + tp) > 0 else float("nan")

    summary = (
        "\n  ── Validation Metrics (Final SVM) ───────────────────\n"
        f"     Precision      : {precision:.4f}\n"
        f"     Recall (Sens.) : {recall:.4f}   ← PRIMARY METRIC\n"
        f"     F1-Score       : {f1:.4f}\n"
        f"     ROC-AUC        : {roc_auc:.4f}\n"
        f"  ── Confusion Matrix ─────────────────────────────────\n"
        f"     TP={tp}  FP={fp}  TN={tn}  FN={fn}\n"
        f"     False Negative Rate : {fn_rate:.4f}  "
        f"← missed cancers, minimize this!\n"
        "  ──────────────────────────────────────────────────────\n"
    )
    print(summary)

    txt_path = os.path.join(OUTPUT_DIR, "svm_val_metrics.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(summary)
    print(f"  Metrics saved → {txt_path}")

    # ── Confusion matrix figure ────────────────────────────────────
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay(confusion_matrix=cm,
                           display_labels=["Benign", "Malignant"]).plot(
        ax=ax, colorbar=False, cmap="Blues"
    )
    ax.set_title("SVM Confusion Matrix — Validation Set",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    cm_path = os.path.join(OUTPUT_DIR, "confusion_matrix_svm.png")
    plt.savefig(cm_path, dpi=150)
    plt.close()
    print(f"  Confusion matrix   → {cm_path}")

    # ── ROC curve figure ──────────────────────────────────────────
    fpr, tpr, _ = roc_curve(y_val, y_prob)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color="steelblue", lw=2,
            label=f"SVM RBF  (AUC = {roc_auc:.4f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random classifier")
    ax.set_xlabel("False Positive Rate", fontsize=11)
    ax.set_ylabel("True Positive Rate (Recall)", fontsize=11)
    ax.set_title("ROC Curve — SVM (Validation Set)",
                 fontsize=12, fontweight="bold")
    ax.legend(loc="lower right", fontsize=10)
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    plt.tight_layout()
    roc_path = os.path.join(OUTPUT_DIR, "roc_curve_svm.png")
    plt.savefig(roc_path, dpi=150)
    plt.close()
    print(f"  ROC curve          → {roc_path}")

    return {
        "Precision": precision, "Recall": recall,
        "F1": f1, "ROC_AUC": roc_auc,
        "TP": int(tp), "FP": int(fp), "TN": int(tn), "FN": int(fn),
        "FN_Rate": fn_rate,
    }


# ─────────────────────────────────────────────
# 7.  SAVE MODEL
# ─────────────────────────────────────────────

def save_model(model, pca, best_params, val_metrics):
    """
    Save a bundle containing the SVM, its PCA transformer, best params,
    and validation metrics. Bundling PCA ensures teammates can reproduce
    predictions without hunting for the right transform.
    """
    bundle = {
        "model":       model,
        "pca":         pca,
        "best_params": best_params,
        "val_metrics": val_metrics,
        "notes": (
            "SVM RBF trained on SMOTE-balanced training split of HAM10000. "
            "Features: 48-dim HSV histogram + 3-dim GLCM = 51 raw, then PCA. "
            "To predict: apply bundle['pca'].transform(X_raw), then "
            "bundle['model'].predict(X_pca). "
            "DO NOT call model.predict() on raw features directly."
        ),
    }
    model_path = os.path.join(OUTPUT_DIR, "svm_best_model.joblib")
    joblib.dump(bundle, model_path, compress=3)
    size_kb = os.path.getsize(model_path) / 1024
    print(f"\n  Model bundle saved → {model_path}  ({size_kb:.1f} KB)")
    print("  Load with:")
    print("    bundle = joblib.load('02_svm/outputs/svm_best_model.joblib')")
    print("    svm    = bundle['model']")
    print("    pca    = bundle['pca']")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    print("\n" + "=" * 60)
    print("  CISC 484 — Week 3  |  Member 2: SVM / Phase 1 Lead")
    print("=" * 60)

    X_train, y_train, X_val, y_val = load_data()

    X_train_pca, X_val_pca, pca = apply_pca(X_train, X_val)

    grid_search = run_grid_search(X_train_pca, y_train)

    save_cv_results(grid_search)

    best_params = grid_search.best_params_
    final_svm   = train_final_svm(best_params, X_train_pca, y_train)

    val_metrics = evaluate_on_validation(final_svm, X_val_pca, y_val)

    print("\n" + "=" * 60)
    print("Saving model…")
    print("=" * 60)
    save_model(final_svm, pca, best_params, val_metrics)

    print("\n" + "=" * 60)
    print("  WEEK 3 COMPLETE — outputs in 02_svm/outputs/")
    print("=" * 60)
    print("  pca_model.joblib          — PCA transformer")
    print("  svm_best_model.joblib     — final SVM bundle (includes PCA)")
    print("  cv_results_table.csv      — all 12 param combos + metrics")
    print("  svm_val_metrics.txt       — validation metrics summary")
    print("  confusion_matrix_svm.png  — confusion matrix figure")
    print("  roc_curve_svm.png         — ROC curve figure")
    print("  pca_explained_variance.png— PCA variance curve")
    print("\n  Handoff notes:")
    print("  → Member 4 : figures are in outputs/ ready for the report")
    print("  → DL team  : svm val metrics are the baseline to beat")
    print("  → Week 5   : X_test / y_test stay locked until final eval")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()