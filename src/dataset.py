from pathlib import Path
from typing import Optional

from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class IndoFashionDataset(Dataset):
    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        transform=None,
        class_names: Optional[list[str]] = None,
    ):
        self.root = Path(root_dir) / split
        self.transform = transform
        self.class_names = class_names

        if not self.root.exists():
            raise FileNotFoundError(f"Split directory not found: {self.root}")

        class_dirs = [d for d in self.root.iterdir() if d.is_dir()]
        if self.class_names is None:
            self.class_names = sorted(d.name for d in class_dirs)

        self.class_to_idx = {name: idx for idx, name in enumerate(self.class_names)}
        self.samples: list[tuple[Path, int]] = []

        for class_name in self.class_names:
            class_dir = self.root / class_name
            if not class_dir.exists():
                continue
            images = [p for p in class_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}]
            self.samples.extend((img_path, self.class_to_idx[class_name]) for img_path in images)

        if not self.samples:
            raise RuntimeError(f"No images found under {self.root}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, label


def get_transforms(split: str):
    if split == "train":
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.RandomRotation(15),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )

    return transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def get_dataloaders(config):
    train_dataset = IndoFashionDataset(
        root_dir=config.DATA_DIR,
        split="train",
        transform=get_transforms("train"),
        class_names=config.CLASS_NAMES,
    )
    val_dataset = IndoFashionDataset(
        root_dir=config.DATA_DIR,
        split="val",
        transform=get_transforms("val"),
        class_names=config.CLASS_NAMES,
    )

    use_cuda = config.DEVICE == "cuda" and torch.cuda.is_available()
    # pin_memory only helps with CUDA, skip on CPU to avoid warning
    pin_memory = True if use_cuda else False
    # Multiprocessing workers can be flaky on some local CPU/macOS setups.
    num_workers = 4 if use_cuda else 0

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return train_loader, val_loader
