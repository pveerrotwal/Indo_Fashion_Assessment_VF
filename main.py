import argparse

from config import Config
from src.dataset import get_dataloaders
from src.evaluate import (
    evaluate_model,
    plot_confusion_matrix,
    plot_sample_predictions,
    plot_training_curves,
)
from src.model import get_model
from src.train import run_training
from src.utils import get_logger, set_seed


def main():
    parser = argparse.ArgumentParser(description="Indo Fashion Classifier")
    parser.add_argument("--mode", choices=["train", "eval", "both"], default="both")
    parser.add_argument("--checkpoint", type=str, default=None)
    args = parser.parse_args()

    config = Config()
    if args.checkpoint:
        config.CHECKPOINT_PATH = args.checkpoint

    set_seed(config.SEED)
    logger = get_logger("main", f"{config.LOG_DIR}/run.log")

    logger.info("=== Indo Fashion Image Classifier ===")
    logger.info(f"Device: {config.DEVICE}")

    train_loader, val_loader = get_dataloaders(config)
    model = get_model(config).to(config.DEVICE)

    if args.mode in ["train", "both"]:
        run_training(model, train_loader, val_loader, config, logger)

    if args.mode in ["eval", "both"]:
        y_true, y_pred, class_names = evaluate_model(model, val_loader, config)
        plot_confusion_matrix(y_true, y_pred, class_names, f"{config.PLOT_DIR}/confusion_matrix.png")
        plot_training_curves(f"{config.LOG_DIR}/training_log.csv", f"{config.PLOT_DIR}/training_curves.png")
        plot_sample_predictions(model, val_loader, class_names, config.DEVICE)


if __name__ == "__main__":
    main()
