from pathlib import Path
import csv

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from src.model import unfreeze_backbone
from src.utils import AverageMeter, save_checkpoint


def train_one_epoch(model, loader, optimizer, criterion, device, scaler):
    model.train()
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()

    for images, labels in tqdm(loader, desc="Training", leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=(device == "cuda")):
            outputs = model(images)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        preds = outputs.argmax(dim=1)
        batch_acc = (preds == labels).float().mean().item() * 100.0

        loss_meter.update(loss.item(), images.size(0))
        acc_meter.update(batch_acc, images.size(0))

    return loss_meter.avg, acc_meter.avg


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    all_preds, all_labels = [], []

    for images, labels in tqdm(loader, desc="Validation", leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        outputs = model(images)
        loss = criterion(outputs, labels)

        preds = outputs.argmax(dim=1)
        batch_acc = (preds == labels).float().mean().item() * 100.0

        loss_meter.update(loss.item(), images.size(0))
        acc_meter.update(batch_acc, images.size(0))
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    return loss_meter.avg, acc_meter.avg, all_preds, all_labels


def run_training(model, train_loader, val_loader, config, logger):
    Path(config.CHECKPOINT_DIR).mkdir(parents=True, exist_ok=True)
    Path(config.LOG_DIR).mkdir(parents=True, exist_ok=True)

    optimizer = AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=config.NUM_EPOCHS)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    scaler = torch.cuda.amp.GradScaler(enabled=(config.DEVICE == "cuda"))

    log_csv_path = Path(config.LOG_DIR) / "training_log.csv"
    with open(log_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "lr"])

    best_val_acc = 0.0
    patience = 5
    no_improve_epochs = 0

    for epoch in range(1, config.NUM_EPOCHS + 1):
        print(f"\nEpoch [{epoch}/{config.NUM_EPOCHS}]")

        if epoch == 6:
            # warmup period over, unfreeze backbone for full fine-tuning
            unfreeze_backbone(model)
            logger.info("Unfroze backbone for full fine-tuning.")

        train_loss, train_acc = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=config.DEVICE,
            scaler=scaler,
        )
        val_loss, val_acc, _, _ = validate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=config.DEVICE,
        )
        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]

        # TODO: experiment with mixup augmentation
        log_line = (
            f"Epoch {epoch:02d}/{config.NUM_EPOCHS} | "
            f"Train Loss: {train_loss:.3f} | Train Acc: {train_acc:.1f}% | "
            f"Val Loss: {val_loss:.3f} | Val Acc: {val_acc:.1f}% | LR: {current_lr:.2e}"
        )
        print(log_line)
        logger.info(log_line)

        with open(log_csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([epoch, train_loss, train_acc, val_loss, val_acc, current_lr])

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            no_improve_epochs = 0
            save_checkpoint(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "best_val_acc": best_val_acc,
                },
                str(config.checkpoint_path),
            )
            logger.info(f"Saved new best checkpoint at epoch {epoch} ({best_val_acc:.2f}%).")
        else:
            no_improve_epochs += 1
            if no_improve_epochs >= patience:
                logger.info("Early stopping triggered.")
                break

    logger.info(f"Training complete. Best validation accuracy: {best_val_acc:.2f}%")
