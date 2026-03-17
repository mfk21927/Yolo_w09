import argparse
import csv
import random
import shutil
from pathlib import Path
from typing import List, Sequence, Tuple

from ultralytics import YOLO

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DEFAULT_CLASS_NAMES = ["car", "number_plate"]


def collect_image_label_pairs(dataset_root: Path) -> List[Tuple[Path, Path]]:
    """Collect image/label pairs from existing split folders under dataset root."""
    pairs: List[Tuple[Path, Path]] = []

    for split in ("train", "val", "test"):
        image_dir = dataset_root / "images" / split
        label_dir = dataset_root / "labels" / split
        if not image_dir.exists() or not label_dir.exists():
            continue

        for image_path in image_dir.iterdir():
            if image_path.suffix.lower() not in VALID_EXTENSIONS:
                continue

            label_path = label_dir / f"{image_path.stem}.txt"
            if label_path.exists():
                pairs.append((image_path, label_path))

    return pairs


def prepare_dataset_splits(
    source_root: Path,
    prepared_root: Path,
    val_ratio: float,
    test_ratio: float,
    seed: int,
    force_rebuild: bool,
) -> None:
    """Create train/val/test dataset folders in YOLO format from available labeled data."""
    expected = [
        prepared_root / "images" / split for split in ("train", "val", "test")
    ] + [
        prepared_root / "labels" / split for split in ("train", "val", "test")
    ]

    if not force_rebuild and all(path.exists() and any(path.iterdir()) for path in expected):
        print(f"Using existing prepared dataset at: {prepared_root}")
        return

    pairs = collect_image_label_pairs(source_root)
    if not pairs:
        raise FileNotFoundError(
            "No labeled image/label pairs found under dataset/images/* and dataset/labels/*."
        )

    if prepared_root.exists() and force_rebuild:
        shutil.rmtree(prepared_root)

    for split in ("train", "val", "test"):
        (prepared_root / "images" / split).mkdir(parents=True, exist_ok=True)
        (prepared_root / "labels" / split).mkdir(parents=True, exist_ok=True)

    random.seed(seed)
    random.shuffle(pairs)

    total = len(pairs)
    test_count = max(1, int(total * test_ratio)) if total >= 10 else max(1, total // 10)
    val_count = max(1, int(total * val_ratio)) if total >= 10 else max(1, total // 5)

    if test_count + val_count >= total:
        test_count = max(1, total // 10)
        val_count = max(1, total // 5)
        if test_count + val_count >= total:
            val_count = max(1, total - test_count - 1)

    train_count = total - val_count - test_count
    if train_count <= 0:
        raise ValueError("Not enough samples to create train/val/test splits.")

    split_map = {
        "train": pairs[:train_count],
        "val": pairs[train_count : train_count + val_count],
        "test": pairs[train_count + val_count :],
    }

    for split, items in split_map.items():
        for image_path, label_path in items:
            shutil.copy2(image_path, prepared_root / "images" / split / image_path.name)
            shutil.copy2(label_path, prepared_root / "labels" / split / label_path.name)

    print("Prepared dataset split summary:")
    for split in ("train", "val", "test"):
        img_count = len(list((prepared_root / "images" / split).iterdir()))
        print(f"  {split}: {img_count} images")


def infer_num_classes(labels_root: Path) -> int:
    max_class_id = -1

    for split in ("train", "val", "test"):
        split_dir = labels_root / split
        if not split_dir.exists():
            continue

        for label_file in split_dir.glob("*.txt"):
            lines = [line.strip() for line in label_file.read_text().splitlines() if line.strip()]
            for line in lines:
                class_id = int(line.split()[0])
                max_class_id = max(max_class_id, class_id)

    if max_class_id < 0:
        raise ValueError("No class ids found in label files.")

    return max_class_id + 1


def resolve_class_names(num_classes: int, class_names_arg: str) -> List[str]:
    if class_names_arg:
        parsed = [name.strip() for name in class_names_arg.split(",") if name.strip()]
        if len(parsed) == num_classes:
            return parsed
        print(
            f"Class name count ({len(parsed)}) does not match inferred classes ({num_classes}). "
            "Using auto-generated names instead."
        )

    if num_classes == len(DEFAULT_CLASS_NAMES):
        return DEFAULT_CLASS_NAMES

    return [f"class_{i}" for i in range(num_classes)]


def write_data_yaml(data_yaml_path: Path, prepared_root: Path, class_names: Sequence[str]) -> None:
    content = [
        f"path: {prepared_root.resolve().as_posix()}",
        "train: images/train",
        "val: images/val",
        "test: images/test",
        f"nc: {len(class_names)}",
        f"names: {list(class_names)}",
        "",
    ]
    data_yaml_path.write_text("\n".join(content), encoding="utf-8")
    print(f"Created data config: {data_yaml_path}")


def find_metric_column(columns: Sequence[str], key: str) -> str:
    for column in columns:
        if key.lower() in column.lower():
            return column
    return ""


def analyze_training_curves(results_csv: Path) -> None:
    if not results_csv.exists():
        print("results.csv not found, skipping metrics analysis.")
        return

    with results_csv.open("r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        rows = list(reader)

    if not rows:
        print("results.csv is empty, skipping metrics analysis.")
        return

    metric_col = find_metric_column(reader.fieldnames or [], "mAP50-95")
    if not metric_col:
        metric_col = find_metric_column(reader.fieldnames or [], "mAP50")

    if not metric_col:
        print("No mAP metric column found in results.csv.")
        return

    best_row = max(rows, key=lambda r: float(r.get(metric_col, 0.0) or 0.0))
    epoch_col = find_metric_column(reader.fieldnames or [], "epoch") or "epoch"

    print("Training curve analysis (from results.csv):")
    print(f"  Best epoch: {best_row.get(epoch_col, 'N/A')}")
    print(f"  Best {metric_col}: {float(best_row.get(metric_col, 0.0) or 0.0):.4f}")

    loss_cols = [
        c
        for c in (reader.fieldnames or [])
        if "loss" in c.lower() and ("train" in c.lower() or "box" in c.lower())
    ]
    for col in loss_cols[:3]:
        print(f"  Final {col}: {float(rows[-1].get(col, 0.0) or 0.0):.4f}")


def export_model_formats(best_weights: Path) -> None:
    export_model = YOLO(str(best_weights))

    print("Exporting model to ONNX...")
    export_model.export(format="onnx")

    print("Exporting model to TFLite...")
    export_model.export(format="tflite")


def test_on_new_images(best_weights: Path, image_source: Path) -> None:
    if not image_source.exists():
        print(f"Test image path not found: {image_source}. Skipping inference test.")
        return

    print(f"Running inference on new images from: {image_source}")
    infer_model = YOLO(str(best_weights))
    infer_model.predict(
        source=str(image_source),
        conf=0.25,
        save=True,
        project="runs/detect",
        name="task9_test_images",
        exist_ok=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Task 9.2 - Custom YOLOv8 Training Pipeline")
    parser.add_argument("--dataset-root", type=Path, default=Path("dataset"))
    parser.add_argument("--prepared-root", type=Path, default=Path("dataset_prepared"))
    parser.add_argument("--data-yaml", type=Path, default=Path("data.yaml"))
    parser.add_argument("--class-names", type=str, default="car,number_plate")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--force-rebuild", action="store_true")
    parser.add_argument("--test-images", type=Path, default=Path("sample_images"))
    parser.add_argument("--project", type=str, default="runs/train")
    parser.add_argument("--name", type=str, default="task9_custom")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    prepare_dataset_splits(
        source_root=args.dataset_root,
        prepared_root=args.prepared_root,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
        force_rebuild=args.force_rebuild,
    )

    num_classes = infer_num_classes(args.prepared_root / "labels")
    class_names = resolve_class_names(num_classes, args.class_names)
    write_data_yaml(args.data_yaml, args.prepared_root, class_names)

    model = YOLO("yolov8n.pt")
    print("Starting training...")
    train_result = model.train(
        data=str(args.data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        seed=args.seed,
        project=args.project,
        name=args.name,
        plots=True,
    )

    print("Running validation...")
    val_metrics = model.val(data=str(args.data_yaml), split="val")
    map50 = getattr(val_metrics.box, "map50", None)
    map5095 = getattr(val_metrics.box, "map", None)
    print(f"Validation mAP50: {map50}")
    print(f"Validation mAP50-95: {map5095}")

    save_dir = Path(train_result.save_dir)
    best_weights = save_dir / "weights" / "best.pt"
    if not best_weights.exists():
        raise FileNotFoundError(f"Best weights not found at {best_weights}")

    analyze_training_curves(save_dir / "results.csv")
    export_model_formats(best_weights)
    test_on_new_images(best_weights, args.test_images)

    print("\nTask 9.2 pipeline completed successfully.")
    print(f"Training outputs: {save_dir}")
    print(f"Best model (.pt): {best_weights}")


if __name__ == "__main__":
    main()
