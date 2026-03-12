from dataclasses import dataclass, field
from pathlib import Path
import torch


@dataclass
class Config:
    DATA_DIR: str = "data/subset"
    NUM_CLASSES: int = 15
    BATCH_SIZE: int = 32
    IMAGE_SIZE: int = 224
    NUM_EPOCHS: int = 20
    LEARNING_RATE: float = 1e-4
    WEIGHT_DECAY: float = 1e-5
    TRAIN_SPLIT: float = 0.8
    MODEL_NAME: str = "efficientnet_b0"
    PRETRAINED: bool = True
    CHECKPOINT_DIR: str = "outputs/checkpoints"
    LOG_DIR: str = "outputs/logs"
    PLOT_DIR: str = "outputs/plots"
    SEED: int = 42
    CLASS_NAMES: list[str] = field(
        default_factory=lambda: [
            "blouse",
            "dhoti_pants",
            "dupattas",
            "gowns",
            "kurta_men",
            "lehenga",
            "mojaris_men",
            "mojaris_women",
            "nehru_jacket",
            "palazzo",
            "petticoats",
            "saree",
            "sherwanis",
            "women_kurta",
            "leggings_and_salwars",
        ]
    )
    DEVICE: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")

    @property
    def checkpoint_path(self) -> Path:
        return Path(self.CHECKPOINT_DIR) / "best_model.pth"
