from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from src.utils import load_checkpoint


@torch.no_grad()
def evaluate_model(model, val_loader, config):
    checkpoint_path = Path(getattr(config, "CHECKPOINT_PATH", config.checkpoint_path))
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Best checkpoint not found: {checkpoint_path}")

    load_checkpoint(str(checkpoint_path), model=model)
    model = model.to(config.DEVICE)
    model.eval()

    y_true, y_pred = [], []
    top5_correct = 0
    total = 0

    for images, labels in val_loader:
        images = images.to(config.DEVICE, non_blocking=True)
        labels = labels.to(config.DEVICE, non_blocking=True)

        logits = model(images)
        preds = torch.argmax(logits, dim=1)

        y_true.extend(labels.cpu().tolist())
        y_pred.extend(preds.cpu().tolist())

        top5_idx = torch.topk(logits, k=min(5, config.NUM_CLASSES), dim=1).indices
        top5_correct += top5_idx.eq(labels.view(-1, 1)).any(dim=1).sum().item()
        total += labels.size(0)

    overall_acc = accuracy_score(y_true, y_pred) * 100.0
    top5_acc = (top5_correct / max(total, 1)) * 100.0

    print(f"\nValidation Accuracy: {overall_acc:.2f}%")
    print(f"Top-5 Accuracy: {top5_acc:.2f}%")
    print("\nPer-class report:")
    print(
        classification_report(
            y_true,
            y_pred,
            labels=list(range(len(config.CLASS_NAMES))),
            target_names=config.CLASS_NAMES,
            digits=4,
            zero_division=0,
        )
    )

    return y_true, y_pred, config.CLASS_NAMES


def plot_confusion_matrix(y_true, y_pred, class_names, save_path):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))), normalize="true")

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(13, 10))
    sns.heatmap(cm * 100.0, cmap="Blues", annot=False, fmt=".1f", cbar_kws={"label": "Percentage"})
    plt.title("Normalized Confusion Matrix (%)")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.xticks(np.arange(len(class_names)) + 0.5, class_names, rotation=45, ha="right")
    plt.yticks(np.arange(len(class_names)) + 0.5, class_names, rotation=0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_training_curves(log_csv_path, save_path):
    history = pd.read_csv(log_csv_path)
    best_idx = history["val_acc"].idxmax()
    best_epoch = int(history.loc[best_idx, "epoch"])

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(history["epoch"], history["train_loss"], label="Train Loss")
    axes[0].plot(history["epoch"], history["val_loss"], label="Val Loss")
    axes[0].axvline(best_epoch, linestyle="--", color="gray", label="Best Val Epoch")
    axes[0].set_title("Loss Curve")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()

    axes[1].plot(history["epoch"], history["train_acc"], label="Train Acc")
    axes[1].plot(history["epoch"], history["val_acc"], label="Val Acc")
    axes[1].axvline(best_epoch, linestyle="--", color="gray", label="Best Val Epoch")
    axes[1].set_title("Accuracy Curve")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close(fig)


@torch.no_grad()
def plot_sample_predictions(model, val_loader, class_names, device, n=16):
    model.eval()
    Path("outputs/plots").mkdir(parents=True, exist_ok=True)

    images_collected, labels_collected, preds_collected = [], [], []
    for images, labels in val_loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        outputs = model(images)
        preds = outputs.argmax(dim=1)

        images_collected.extend(images.cpu())
        labels_collected.extend(labels.cpu().tolist())
        preds_collected.extend(preds.cpu().tolist())
        if len(images_collected) >= n:
            break

    images_collected = images_collected[:n]
    labels_collected = labels_collected[:n]
    preds_collected = preds_collected[:n]

    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    for i, ax in enumerate(axes.flat):
        if i >= len(images_collected):
            ax.axis("off")
            continue

        img = images_collected[i] * std + mean
        img = torch.clamp(img, 0, 1).permute(1, 2, 0).numpy()
        true_name = class_names[labels_collected[i]]
        pred_name = class_names[preds_collected[i]]

        ax.imshow(img)
        color = "green" if labels_collected[i] == preds_collected[i] else "red"
        ax.set_title(f"True: {true_name}\nPred: {pred_name}", color=color, fontsize=8)
        ax.axis("off")

    fig.tight_layout()
    fig.savefig("outputs/plots/sample_predictions.png", dpi=300)
    plt.close(fig)
