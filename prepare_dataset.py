import argparse
import json
import random
import shutil
from collections import defaultdict
from pathlib import Path

from PIL import Image, UnidentifiedImageError

from config import Config
from src.utils import set_seed


def _find_annotation_file(dataset_root: Path, split_name: str) -> Path:
    candidates = [
        dataset_root / f"{split_name}.json",
        dataset_root / f"{split_name}_data.json",
        dataset_root / "annotations" / f"{split_name}.json",
        dataset_root / "annotations" / f"{split_name}_data.json",
        dataset_root / "labels" / f"{split_name}.json",
        dataset_root / "labels" / f"{split_name}_data.json",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"Could not find annotation for split '{split_name}' under {dataset_root}")


def _load_annotation_records(ann_path: Path) -> list[dict]:
    # Some releases store a JSON list, others store newline-delimited JSON.
    with open(ann_path, "r", encoding="utf-8") as f:
        first_char = f.read(1)
        f.seek(0)
        if first_char == "[":
            return json.load(f)

        records = []
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
        return records


def _read_annotations(dataset_root: Path) -> list[dict]:
    all_records = []
    for split_name in ["train", "val", "test"]:
        ann_path = _find_annotation_file(dataset_root, split_name)
        records = _load_annotation_records(ann_path)
        all_records.extend(records)
    return all_records


def _is_readable_image(image_path: Path) -> bool:
    try:
        with Image.open(image_path) as img:
            img.verify()
        return True
    except (FileNotFoundError, UnidentifiedImageError, OSError):
        return False


def prepare_subset(dataset_root: str, config: Config) -> None:
    source_root = Path(dataset_root)
    if not source_root.exists():
        raise FileNotFoundError(f"Dataset path does not exist: {source_root}")

    subset_root = Path(config.DATA_DIR)
    train_root = subset_root / "train"
    val_root = subset_root / "val"

    # Clean previous subset so stale files do not leak into a new run.
    if train_root.exists():
        shutil.rmtree(train_root)
    if val_root.exists():
        shutil.rmtree(val_root)
    train_root.mkdir(parents=True, exist_ok=True)
    val_root.mkdir(parents=True, exist_ok=True)

    records = _read_annotations(source_root)
    by_class = defaultdict(list)

    for row in records:
        label = row.get("label") or row.get("class_label")
        image_rel_path = row.get("image_path")
        if not label or not image_rel_path:
            continue
        image_abs_path = source_root / image_rel_path
        by_class[label].append(image_abs_path)

    summary = []
    for class_name in config.CLASS_NAMES:
        class_images = by_class.get(class_name, [])
        readable_images = [p for p in class_images if _is_readable_image(p)]
        unreadable_count = len(class_images) - len(readable_images)
        if unreadable_count > 0:
            print(f"[WARN] {class_name}: skipped {unreadable_count} unreadable/corrupt images.")

        target_count = 500
        if len(readable_images) < target_count:
            print(f"[WARN] {class_name}: only {len(readable_images)} images available (<500). Using all.")
            selected = readable_images
        else:
            selected = random.sample(readable_images, target_count)

        random.shuffle(selected)
        split_idx = int(len(selected) * config.TRAIN_SPLIT)
        train_images = selected[:split_idx]
        val_images = selected[split_idx:]

        (train_root / class_name).mkdir(parents=True, exist_ok=True)
        (val_root / class_name).mkdir(parents=True, exist_ok=True)

        for image_path in train_images:
            shutil.copy2(image_path, train_root / class_name / image_path.name)
        for image_path in val_images:
            shutil.copy2(image_path, val_root / class_name / image_path.name)

        summary.append((class_name, len(train_images), len(val_images)))

    print("\nSubset Summary")
    print("-" * 55)
    print(f"{'Class Name':<24} {'Train Count':<12} {'Val Count':<10}")
    print("-" * 55)
    for class_name, train_count, val_count in summary:
        print(f"{class_name:<24} {train_count:<12} {val_count:<10}")
    print("-" * 55)


def main():
    parser = argparse.ArgumentParser(description="Prepare Indo Fashion 500-per-class subset")
    parser.add_argument("--dataset_root", type=str, default=None, help="Path to downloaded Indo Fashion dataset root")
    parser.add_argument(
        "--dataset_path",
        type=str,
        default=None,
        help="Alias for --dataset_root (kept for CLI compatibility).",
    )
    args = parser.parse_args()

    dataset_root = args.dataset_root or args.dataset_path
    if dataset_root is None:
        parser.error("one of --dataset_root or --dataset_path is required")

    config = Config()
    set_seed(config.SEED)
    # TODO: support exporting split metadata CSV for reproducibility checks
    prepare_subset(dataset_root, config)


if __name__ == "__main__":
    main()
