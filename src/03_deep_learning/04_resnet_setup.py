import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import recall_score
from PIL import Image

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────

IMAGE_DIR    = "data/HAM10000_images_hair_removed"
SPLIT_DIR    = "data/"
OUTPUT_DIR   = "outputs/"
PLOT_DIR     = "src/03_deep_learning/plots/"
CKPT_DIR     = "outputs/checkpoints/"
RANDOM_STATE = 42
IMG_SIZE     = 224        # ResNet-50 / EfficientNet-B0 both expect 224×224
NUM_CLASSES  = 2          # benign (0) / malignant (1)
NUM_WORKERS  = 0          # set to 0 on Mac to avoid multiprocessing issues

# Batch size ladder — script tries each in order, falls back on OOM
BATCH_SIZE_LADDER = [32, 16, 8, 4]

# Week 5 — full training
NUM_EPOCHS       = 20
BATCH_SIZE       = 16         # reduce to 8 if MPS crashes
LR_HEAD          = 1e-3       # LR while backbone is frozen
LR_FINETUNE      = 1e-5       # LR after backbone is unfrozen
UNFREEZE_EPOCH   = 3          # unfreeze backbone after this epoch
MALIGNANT_WEIGHT = 2.0        # upweight malignant loss — tune to 3.0 if recall too low
RUN_EFFICIENTNET = True       # set False to skip stretch goal

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(PLOT_DIR,   exist_ok=True)
os.makedirs(CKPT_DIR,   exist_ok=True)

# ─────────────────────────────────────────────
# DEVICE SETUP — CUDA / MPS / CPU
# ─────────────────────────────────────────────

def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        name   = torch.cuda.get_device_name(0)
        vram   = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"Device : CUDA — {name} ({vram:.1f} GB VRAM)")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Device : Apple MPS (Metal GPU)")
        print("  NOTE : MPS memory stats unavailable; OOM appears as a Python crash.")
        print("         If the script crashes, reduce BATCH_SIZE_LADDER values.")
    else:
        device = torch.device("cpu")
        print("Device : CPU — no GPU detected. Smoke test will be slow but functional.")
    return device

device = get_device()

def log_memory(label: str):
    """Log GPU memory usage. CUDA only — MPS and CPU are no-ops."""
    if device.type == "cuda":
        allocated = torch.cuda.memory_allocated() / 1e6
        reserved  = torch.cuda.memory_reserved() / 1e6
        print(f"  [{label}] Allocated: {allocated:.1f} MB | Reserved: {reserved:.1f} MB")
    else:
        print(f"  [{label}] (memory logging only available on CUDA)")

def clear_memory():
    """Free cached GPU memory between retries."""
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps":
        torch.mps.empty_cache()

# ─────────────────────────────────────────────
# DATASET
# ─────────────────────────────────────────────

class HAMDataset(Dataset):
    """
    PyTorch Dataset for HAM10000.
    Returns (image_tensor, label, image_id) so Week 5 prediction
    test can map predictions back to filenames for display.
    """
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD  = [0.229, 0.224, 0.225]

    def __init__(self, split_csv: str, image_dir: str, augment: bool = False):
        self.df        = pd.read_csv(split_csv)
        self.image_dir = image_dir
        self.transform = self._build_transform(augment)

    def _build_transform(self, augment: bool):
        norm = transforms.Normalize(self.IMAGENET_MEAN, self.IMAGENET_STD)
        if augment:
            return transforms.Compose([
                transforms.Resize((IMG_SIZE, IMG_SIZE)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(20),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
                transforms.ToTensor(),
                norm,
            ])
        return transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            norm,
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row      = self.df.iloc[idx]
        img_path = os.path.join(self.image_dir, f"{row['image_id']}.jpg")
        image    = Image.open(img_path).convert("RGB")
        label    = int(row["binary_label"])
        return self.transform(image), label, row["image_id"]


def build_loaders(batch_size: int, drop_last_train: bool = True):
    """Build train and val DataLoaders for a given batch size."""
    train_dataset = HAMDataset(os.path.join(SPLIT_DIR, "split_train.csv"), IMAGE_DIR, augment=True)
    val_dataset   = HAMDataset(os.path.join(SPLIT_DIR, "split_val.csv"),   IMAGE_DIR, augment=False)
    train_loader  = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=NUM_WORKERS, drop_last=drop_last_train,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(device.type == "cuda"),
    )
    print(f"Train: {len(train_dataset)} images | Val: {len(val_dataset)} images | batch={batch_size}")
    return train_loader, val_loader

# ─────────────────────────────────────────────
# MODEL BUILDERS
# ─────────────────────────────────────────────

def build_resnet50(freeze_backbone: bool = True, use_checkpointing: bool = False):
    """
    ResNet-50 with pretrained ImageNet weights and custom binary head.
    Uses LayerNorm instead of BatchNorm1d to avoid batch-size-1 crash.

    Head: Linear(2048→512) → LayerNorm → ReLU → Dropout(0.5) → Linear(512→2)
    """
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

    for param in model.parameters():
        param.requires_grad = not freeze_backbone

    in_features = model.fc.in_features   # 2048
    model.fc = nn.Sequential(
        nn.Linear(in_features, 512),
        nn.LayerNorm(512),               # LayerNorm: safe at any batch size
        nn.ReLU(),
        nn.Dropout(p=0.5),
        nn.Linear(512, NUM_CLASSES),
    )
    # Head parameters always require grad
    for param in model.fc.parameters():
        param.requires_grad = True

    if use_checkpointing:
        model.layer1 = torch.utils.checkpoint.checkpoint_sequential(model.layer1, 2, model.layer1[0])
        model.layer2 = torch.utils.checkpoint.checkpoint_sequential(model.layer2, 4, model.layer2[0])
        model.layer3 = torch.utils.checkpoint.checkpoint_sequential(model.layer3, 6, model.layer3[0])
        model.layer4 = torch.utils.checkpoint.checkpoint_sequential(model.layer4, 3, model.layer4[0])
        print("Gradient checkpointing ENABLED.")

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    status    = "frozen backbone" if freeze_backbone else "full fine-tune"
    print(f"ResNet-50 ({status}) — trainable: {trainable:,} / {total:,} ({trainable/total:.2%})")
    return model


def build_efficientnet_b0(freeze_backbone: bool = True):
    """EfficientNet-B0 — lighter alternative (~5.3M vs ~25M params)."""
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)

    for param in model.parameters():
        param.requires_grad = not freeze_backbone

    in_features = model.classifier[1].in_features   # 1280
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.4),
        nn.Linear(in_features, NUM_CLASSES),
    )
    for param in model.classifier.parameters():
        param.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"EfficientNet-B0 — trainable: {trainable:,} / {total:,} ({trainable/total:.2%})")
    return model


def unfreeze_backbone(model, model_name: str):
    """Unfreeze all parameters for full fine-tuning phase."""
    for param in model.parameters():
        param.requires_grad = True
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"  Backbone unfrozen — {trainable:,} / {total:,} params now trainable")

# ─────────────────────────────────────────────
# WEEK 4 — SMOKE TEST
# ─────────────────────────────────────────────

def run_smoke_test(model, train_loader, val_loader, batch_size: int):
    """
    One training epoch + one val pass to confirm the pipeline works.
    Returns True on success, False on OOM.
    """
    print(f"\n{'─'*60}")
    print(f"Smoke test — batch_size={batch_size}")
    print(f"{'─'*60}")

    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-3, weight_decay=1e-4,
    )
    criterion = nn.CrossEntropyLoss()

    # ── Training pass ──────────────────────────────────────────────────────
    model.train()
    train_losses  = []
    correct_train = 0
    total_train   = 0

    try:
        for batch_idx, (images, labels, _) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss    = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())
            preds          = outputs.argmax(dim=1)
            correct_train += (preds == labels).sum().item()
            total_train   += labels.size(0)

            if batch_idx % 10 == 0:
                print(f"  Batch {batch_idx:>4}/{len(train_loader)} — "
                      f"loss: {loss.item():.4f}  avg: {np.mean(train_losses):.4f}")
                log_memory("train step")

    except torch.cuda.OutOfMemoryError:
        print("\n  CUDA OOM — trying next fallback.")
        clear_memory()
        return False

    first_loss = train_losses[0]
    last_loss  = train_losses[-1]
    avg_loss   = np.mean(train_losses)
    train_acc  = correct_train / total_train

    print(f"\n  Train epoch complete:")
    print(f"    First batch loss : {first_loss:.4f}")
    print(f"    Last batch loss  : {last_loss:.4f}")
    print(f"    Average loss     : {avg_loss:.4f}")
    print(f"    Train accuracy   : {train_acc:.4f}")
    if last_loss < first_loss:
        print("    ✓ Loss decreased — training loop working correctly.")
    else:
        print("    ⚠ Loss did not decrease — check LR and data pipeline.")

    # ── Validation pass ────────────────────────────────────────────────────
    model.eval()
    val_losses  = []
    correct_val = 0
    total_val   = 0

    with torch.no_grad():
        for images, labels, _ in val_loader:
            images  = images.to(device)
            labels  = labels.to(device)
            outputs = model(images)
            loss    = criterion(outputs, labels)
            val_losses.append(loss.item())
            preds        = outputs.argmax(dim=1)
            correct_val += (preds == labels).sum().item()
            total_val   += labels.size(0)

    val_acc  = correct_val / total_val
    val_loss = np.mean(val_losses)
    print(f"\n  Validation complete — loss: {val_loss:.4f}  acc: {val_acc:.4f}")
    log_memory("after val")

    ckpt_path = os.path.join(OUTPUT_DIR, "resnet50_smoke_test.pth")
    torch.save({
        "epoch": 1, "model_state": model.state_dict(),
        "optim_state": optimizer.state_dict(),
        "train_loss": avg_loss, "val_loss": val_loss, "val_acc": val_acc,
    }, ckpt_path)
    print(f"  Checkpoint saved: {ckpt_path}")
    return True


def run_smoke_test_with_fallback():
    """
    Try smoke test across batch sizes and models.
    Returns (model_name, batch_size) of first passing config.
    """
    train_loader_smoke, val_loader_smoke = None, None

    for use_checkpointing in [False, True]:
        for batch_size in BATCH_SIZE_LADDER:
            print(f"\n{'='*60}")
            print(f"ResNet-50 | batch={batch_size} | "
                  f"checkpointing={'ON' if use_checkpointing else 'OFF'}")
            print(f"{'='*60}")
            clear_memory()

            model = build_resnet50(freeze_backbone=True,
                                   use_checkpointing=use_checkpointing).to(device)
            train_loader_smoke, val_loader_smoke = build_loaders(batch_size)

            if run_smoke_test(model, train_loader_smoke, val_loader_smoke, batch_size):
                print(f"\n✓ ResNet-50 smoke test PASSED (batch={batch_size}, "
                      f"checkpointing={'ON' if use_checkpointing else 'OFF'})")
                return "resnet50", batch_size

    # EfficientNet-B0 fallback
    print(f"\n{'='*60}")
    print("Falling back to EfficientNet-B0")
    print(f"{'='*60}")
    for batch_size in BATCH_SIZE_LADDER:
        clear_memory()
        model = build_efficientnet_b0(freeze_backbone=True).to(device)
        train_loader_smoke, val_loader_smoke = build_loaders(batch_size)
        if run_smoke_test(model, train_loader_smoke, val_loader_smoke, batch_size):
            print(f"\n✓ EfficientNet-B0 smoke test PASSED (batch={batch_size})")
            return "efficientnet_b0", batch_size

    return None, None

# ─────────────────────────────────────────────
# WEEK 5 — FULL TRAINING
# ─────────────────────────────────────────────

# Weighted loss — penalizes malignant misclassifications more heavily
class_weights = torch.tensor([1.0, MALIGNANT_WEIGHT], dtype=torch.float).to(device)
criterion_weighted = nn.CrossEntropyLoss(weight=class_weights)
print(f"Weighted loss : [benign=1.0, malignant={MALIGNANT_WEIGHT}]")


def run_epoch(model, loader, optimizer, phase: str):
    """One training or validation epoch. Returns (avg_loss, accuracy, mal_recall)."""
    is_train = (phase == "train")
    model.train() if is_train else model.eval()

    total_loss = 0.0
    all_preds  = []
    all_labels = []

    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        for images, labels, _ in loader:
            images = images.to(device)
            labels = labels.to(device)

            if is_train:
                optimizer.zero_grad()

            outputs = model(images)
            loss    = criterion_weighted(outputs, labels)

            if is_train:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * labels.size(0)
            all_preds.extend(outputs.argmax(dim=1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    n          = len(all_labels)
    avg_loss   = total_loss / n
    accuracy   = (np.array(all_preds) == np.array(all_labels)).mean()
    mal_recall = recall_score(all_labels, all_preds, pos_label=1, zero_division=0)
    return avg_loss, accuracy, mal_recall


def run_full_training(model, model_name: str, train_loader, val_loader):
    """10-epoch training with backbone unfreezing. Saves best checkpoint by val recall."""
    print(f"\n{'='*60}")
    print(f"Full training — {model_name} — {NUM_EPOCHS} epochs")
    print(f"{'='*60}")

    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR_HEAD, weight_decay=1e-4,
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2
    )

    history = {k: [] for k in
               ["train_loss", "val_loss", "train_acc", "val_acc", "train_recall", "val_recall"]}
    best_val_recall = 0.0
    best_ckpt_path  = os.path.join(CKPT_DIR, f"{model_name}_best.pth")

    for epoch in range(1, NUM_EPOCHS + 1):

        # Unfreeze backbone after UNFREEZE_EPOCH epochs
        if epoch == UNFREEZE_EPOCH + 1:
            print(f"\nEpoch {epoch}: Unfreezing backbone — LR → {LR_FINETUNE}")
            unfreeze_backbone(model, model_name)
            for pg in optimizer.param_groups:
                pg["lr"] = LR_FINETUNE

        tr_loss, tr_acc, tr_rec = run_epoch(model, train_loader, optimizer, "train")
        vl_loss, vl_acc, vl_rec = run_epoch(model, val_loader,   optimizer, "val")
        scheduler.step(vl_loss)

        for key, val in zip(
            ["train_loss","val_loss","train_acc","val_acc","train_recall","val_recall"],
            [tr_loss, vl_loss, tr_acc, vl_acc, tr_rec, vl_rec]
        ):
            history[key].append(val)

        marker = " ← best" if vl_rec > best_val_recall else ""
        print(f"Epoch {epoch:>2}/{NUM_EPOCHS}  "
              f"train_loss={tr_loss:.4f}  val_loss={vl_loss:.4f}  "
              f"val_acc={vl_acc:.4f}  val_recall={vl_rec:.4f}{marker}")

        if vl_rec > best_val_recall:
            best_val_recall = vl_rec
            torch.save({
                "epoch": epoch, "model_name": model_name,
                "model_state": model.state_dict(),
                "val_loss": vl_loss, "val_recall": vl_rec,
            }, best_ckpt_path)

    print(f"\nBest val malignant recall : {best_val_recall:.4f}")
    print(f"Checkpoint saved          : {best_ckpt_path}")
    return history, best_ckpt_path


# ─────────────────────────────────────────────
# PLOTTING
# ─────────────────────────────────────────────

def plot_training_history(history: dict, model_name: str):
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(f"{model_name} — Training History ({NUM_EPOCHS} epochs)", fontsize=13)

    for ax, (tr_key, vl_key), title, ylabel in zip(
        axes,
        [("train_loss","val_loss"), ("train_acc","val_acc"), ("train_recall","val_recall")],
        ["Loss", "Accuracy", "Malignant Recall"],
        ["Weighted Cross-Entropy", "Accuracy", "Recall (malignant)"],
    ):
        ax.plot(epochs, history[tr_key], label=f"Train", marker="o")
        ax.plot(epochs, history[vl_key], label=f"Val",   marker="o")
        ax.axvline(x=UNFREEZE_EPOCH + 1, color="red", linestyle="--",
                   linewidth=1, label=f"Unfreeze (ep {UNFREEZE_EPOCH+1})")
        if "recall" in tr_key:
            ax.axhline(y=0.80, color="green", linestyle=":", label="0.80 target")
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(PLOT_DIR, f"{model_name}_training_history.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Plot saved: {path}")


# ─────────────────────────────────────────────
# WEEK 5 TASK 3 — MANUAL PREDICTION TEST
# ─────────────────────────────────────────────

def run_manual_prediction_test(ckpt_path: str, model_name: str, n_samples: int = 10):
    """Load best checkpoint and visually inspect predictions on 10 test images."""
    print(f"\n{'='*60}")
    print(f"Manual Prediction Test — {model_name} (n={n_samples})")
    print(f"{'='*60}")

    ckpt  = torch.load(ckpt_path, map_location=device)
    model = build_resnet50(freeze_backbone=False) if model_name == "resnet50" \
            else build_efficientnet_b0(freeze_backbone=False)
    model.load_state_dict(ckpt["model_state"])
    model = model.to(device)
    model.eval()
    print(f"Loaded checkpoint — epoch {ckpt['epoch']}, val_recall={ckpt['val_recall']:.4f}")

    test_df      = pd.read_csv(os.path.join(SPLIT_DIR, "split_test.csv"))
    benign_df    = test_df[test_df["binary_label"] == 0].sample(
                       min(5, (test_df["binary_label"] == 0).sum()), random_state=RANDOM_STATE)
    malignant_df = test_df[test_df["binary_label"] == 1].sample(
                       min(5, (test_df["binary_label"] == 1).sum()), random_state=RANDOM_STATE)
    sample_df    = pd.concat([benign_df, malignant_df]).sample(
                       frac=1, random_state=RANDOM_STATE).reset_index(drop=True)

    infer_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    softmax   = nn.Softmax(dim=1)
    label_map = {0: "Benign", 1: "Malignant"}
    results   = []

    for _, row in sample_df.iterrows():
        img_path = os.path.join(IMAGE_DIR, f"{row['image_id']}.jpg")
        pil_img  = Image.open(img_path).convert("RGB")
        tensor   = infer_transform(pil_img).unsqueeze(0).to(device)

        with torch.no_grad():
            probs = softmax(model(tensor))[0].cpu().numpy()

        pred_class = int(probs.argmax())
        confidence = float(probs[pred_class])
        true_label = int(row["binary_label"])
        correct    = pred_class == true_label

        results.append({
            "pil_img": pil_img, "true_label": true_label,
            "pred_class": pred_class, "confidence": confidence, "correct": correct,
        })
        status = "✓" if correct else "✗"
        print(f"  {status} {row['image_id']:<20}  "
              f"true={label_map[true_label]:<10}  "
              f"pred={label_map[pred_class]:<10}  conf={confidence:.3f}")

    # ── Visualize ──────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 5, figsize=(18, 8))
    fig.suptitle(f"{model_name} — Manual Prediction Test", fontsize=13)

    for i, res in enumerate(results):
        ax = axes[i // 5][i % 5]
        ax.imshow(res["pil_img"].resize((224, 224)))
        ax.axis("off")
        color = "green" if res["correct"] else "red"
        ax.set_title(
            f"{'✓' if res['correct'] else '✗'} "
            f"True: {label_map[res['true_label']]}\n"
            f"Pred: {label_map[res['pred_class']]} ({res['confidence']:.2%})",
            fontsize=9, color=color, fontweight="bold",
        )
        for spine in ax.spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(3)
            spine.set_visible(True)

    fig.legend(
        handles=[mpatches.Patch(color="green", label="Correct"),
                 mpatches.Patch(color="red",   label="Wrong")],
        loc="lower center", ncol=2, fontsize=10, bbox_to_anchor=(0.5, -0.02),
    )
    plt.tight_layout()
    path = os.path.join(PLOT_DIR, f"{model_name}_manual_predictions.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Prediction grid saved: {path}")

    n_correct = sum(r["correct"] for r in results)
    print(f"Summary: {n_correct}/{n_samples} correct ({n_correct/n_samples:.0%})")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

print("=" * 60)
print("WEEK 4 — Smoke Test")
print("=" * 60)

smoke_model_name, smoke_batch_size = run_smoke_test_with_fallback()

if smoke_model_name is None:
    print("\n✗ Smoke test failed at all batch sizes and models.")
    print("  Use Google Colab (T4 GPU) or UD's Caviness HPC cluster.")
    raise SystemExit(1)

print(f"\n✓ Proceeding to Week 5 full training with {smoke_model_name}, batch={BATCH_SIZE}")

# ── Week 5: Full training ─────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("WEEK 5 — Full Fine-Tuning")
print("=" * 60)

train_loader, val_loader = build_loaders(BATCH_SIZE)

if smoke_model_name == "resnet50":
    model = build_resnet50(freeze_backbone=True).to(device)
else:
    model = build_efficientnet_b0(freeze_backbone=True).to(device)

history, best_ckpt = run_full_training(model, smoke_model_name, train_loader, val_loader)
plot_training_history(history, smoke_model_name)
run_manual_prediction_test(best_ckpt, smoke_model_name)

# ── Week 5 Task 4 (Stretch): EfficientNet-B0 comparison ──────────────────────

if RUN_EFFICIENTNET and smoke_model_name == "resnet50":
    print("\n" + "=" * 60)
    print("STRETCH GOAL — EfficientNet-B0 Comparison")
    print("=" * 60)
    clear_memory()
    effnet = build_efficientnet_b0(freeze_backbone=True).to(device)
    eff_history, eff_ckpt = run_full_training(effnet, "efficientnet_b0", train_loader, val_loader)
    plot_training_history(eff_history, "efficientnet_b0")
    run_manual_prediction_test(eff_ckpt, "efficientnet_b0")

    # Side-by-side recall comparison
    epochs = range(1, NUM_EPOCHS + 1)
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["val_recall"],     label="ResNet-50",       marker="o")
    plt.plot(epochs, eff_history["val_recall"], label="EfficientNet-B0", marker="s")
    plt.axhline(y=0.80, color="green", linestyle=":", label="0.80 target")
    plt.xlabel("Epoch")
    plt.ylabel("Val Malignant Recall")
    plt.title("ResNet-50 vs EfficientNet-B0 — Val Malignant Recall")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    cmp_path = os.path.join(PLOT_DIR, "model_comparison_recall.png")
    plt.savefig(cmp_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Comparison plot saved: {cmp_path}")

elif RUN_EFFICIENTNET and smoke_model_name == "efficientnet_b0":
    print("\nNOTE: Smoke test already fell back to EfficientNet-B0 — skipping duplicate comparison run.")

# ─────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────

print("\n" + "=" * 60)
print("WEEK 4 + WEEK 5 — COMPLETE")
print("=" * 60)
print("[W4] Task 1: ResNet-50 pretrained, backbone frozen              DONE")
print("[W4] Task 2: Custom binary classification head (LayerNorm)      DONE")
print("[W4] Task 3: Smoke test — 1 epoch train + val                   DONE")
print("[W4] Task 4: OOM fallback ladder                                DONE")
print("[W5] Task 1: Full fine-tuning, 10 epochs, loss + recall         DONE")
print("[W5] Task 2: Weighted cross-entropy (malignant weight=2.0)      DONE")
print("[W5] Task 3: Manual prediction test on 10 held-out images       DONE")
print("[W5] Task 4: EfficientNet-B0 comparison (stretch goal)          DONE")
print(f"\nOutputs in {OUTPUT_DIR}:")
print("  resnet50_smoke_test.pth")
print("  checkpoints/resnet50_best.pth")
print("  checkpoints/efficientnet_b0_best.pth  (if stretch ran)")
print("  plots/resnet50_training_history.png")
print("  plots/resnet50_manual_predictions.png")
print("  plots/model_comparison_recall.png     (if stretch ran)")
print("\nHandoff notes:")
print("  → Final eval  : run on split_test.csv — first and only test set usage")
print("  → SVM compare : load X_test_pca.npy alongside CNN test metrics")
print("  → Report      : use training history plots and prediction grid directly")