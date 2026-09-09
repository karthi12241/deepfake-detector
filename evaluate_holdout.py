"""Evaluate Bidirectional, Swin-T, and Gated models on holdout_test.

Set the paths below if needed, then run:
    ./.venv/Scripts/python.exe evaluate_holdout.py
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from compare_models import SwinBinaryClassifier, load_torch_file, remove_prefixes
from test import build_model, crop_largest_face, make_transform, unpack_checkpoint


ROOT = Path(__file__).resolve().parent
# ── Edit these paths for Colab or another machine ──────────────────────────
# Example Colab path: /content/drive/MyDrive/models/best.pt
HOLDOUT_DIR = ROOT / "holdout_test"
BIDIRECTIONAL_MODEL_PATH = ROOT / "best.pt"
SWIN_MODEL_PATH = ROOT / "Swin" / "best.pth"
GATED_MODEL_PATH = ROOT / "Gated" / "best.pt"
OUTPUT_DIR = ROOT / "holdout_evaluation"

# ──────────────────────────────────────────────────────────────────────────
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def collect_records(root: Path) -> list[dict]:
    """Discover real/fake images regardless of how deeply their folders nest."""
    records = []
    for class_name, label in (("real", 0), ("fake", 1)):
        class_dir = root / class_name
        if not class_dir.is_dir():
            raise FileNotFoundError(f"Expected folder not found: {class_dir}")
        for path in sorted(class_dir.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            relative = path.relative_to(class_dir)
            method = "real" if label == 0 else (relative.parts[0] if relative.parts else "unknown")
            records.append({"path": path, "label": label, "method": method})
    return records


class HoldoutDataset(Dataset):
    def __init__(self, records: list[dict], transform, use_face_crop: bool) -> None:
        self.records, self.transform, self.use_face_crop = records, transform, use_face_crop

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        record = self.records[index]
        try:
            with Image.open(record["path"]) as opened:
                image = opened.convert("RGB")
            if self.use_face_crop:
                image = crop_largest_face(image, margin=0.35, min_face_size=48).image
            return self.transform(image), record["label"], record["method"], str(record["path"])
        except Exception as error:
            return None, str(record["path"]), str(error)


def collate_skip_errors(items):
    valid, failed = [], []
    for item in items:
        if item[0] is None:
            failed.append((item[1], item[2]))
        else:
            valid.append(item)
    if not valid:
        return None, failed
    images = torch.stack([item[0] for item in valid])
    labels = torch.tensor([item[1] for item in valid], dtype=torch.int64)
    methods = [item[2] for item in valid]
    paths = [item[3] for item in valid]
    return (images, labels, methods, paths), failed


def load_fusion_model(path: Path, device: torch.device):
    checkpoint = load_torch_file(path, device)
    state_dict, model_config = unpack_checkpoint(checkpoint)
    model = build_model(model_config, state_dict).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    saved_config = checkpoint.get("config", {}) if isinstance(checkpoint, dict) else {}
    image_size = int(saved_config.get("training", {}).get("image_size", 224)) if isinstance(saved_config, dict) else 224
    del checkpoint, state_dict
    return model, image_size


def load_swin_model(path: Path, device: torch.device):
    state_dict = load_torch_file(path, device)
    if isinstance(state_dict, dict) and "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    if not isinstance(state_dict, dict):
        raise ValueError("Swin checkpoint does not contain a state dictionary.")
    model = SwinBinaryClassifier().to(device)
    model.load_state_dict(remove_prefixes(state_dict))
    model.eval()
    del state_dict
    return model, 224


def evaluate_model(name, model, image_size, records, args, device):
    loader = DataLoader(
        HoldoutDataset(records, make_transform(image_size), args.face_crop),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        collate_fn=collate_skip_errors,
    )
    predictions, failed = [], []
    total_batches = len(loader)
    for batch_index, (batch, batch_failures) in enumerate(loader, start=1):
        failed.extend(batch_failures)
        if batch is None:
            continue
        images, labels, methods, paths = batch
        images = images.to(device, non_blocking=True)
        with torch.inference_mode():
            probabilities = torch.sigmoid(model(images)).flatten().cpu().tolist()
        for path, label, method, probability in zip(paths, labels.tolist(), methods, probabilities):
            predictions.append({
                "model": name, "path": path, "label": label, "method": method,
                "fake_probability": float(probability),
                "prediction": int(probability >= args.threshold),
            })
        if batch_index % 100 == 0 or batch_index == total_batches:
            print(f"  {name}: {batch_index}/{total_batches} batches")
    return predictions, failed


def calculate_metrics(predictions: list[dict]) -> dict:
    tp = sum(row["label"] == 1 and row["prediction"] == 1 for row in predictions)
    tn = sum(row["label"] == 0 and row["prediction"] == 0 for row in predictions)
    fp = sum(row["label"] == 0 and row["prediction"] == 1 for row in predictions)
    fn = sum(row["label"] == 1 and row["prediction"] == 0 for row in predictions)
    total = tp + tn + fp + fn
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    return {
        "evaluated": total, "accuracy": (tp + tn) / max(total, 1),
        "precision": precision, "recall": recall,
        "f1": 2 * precision * recall / max(precision + recall, 1e-12),
        "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
    }


def plot_confusion_matrices(metrics: dict, output_dir: Path) -> None:
    figure, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 4.5))
    if len(metrics) == 1:
        axes = [axes]
    for axis, (name, values) in zip(axes, metrics.items()):
        cm = values["confusion_matrix"]
        matrix = [[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]]
        image = axis.imshow(matrix, cmap="Blues")
        axis.set_xticks([0, 1], ["Pred REAL", "Pred FAKE"])
        axis.set_yticks([0, 1], ["Actual REAL", "Actual FAKE"])
        axis.set_title(f"{name}\nAccuracy: {values['accuracy'] * 100:.2f}%")
        for row in range(2):
            for column in range(2):
                color = "white" if matrix[row][column] > max(map(max, matrix)) / 2 else "black"
                axis.text(column, row, f"{matrix[row][column]:,}", ha="center", va="center", color=color, fontsize=12)
        figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    figure.tight_layout()
    figure.savefig(output_dir / "confusion_matrices.png", dpi=160, bbox_inches="tight")
    plt.close(figure)


def plot_metric_comparison(metrics: dict, output_dir: Path) -> None:
    names, fields = list(metrics), ("accuracy", "precision", "recall", "f1")
    positions = list(range(len(names)))
    width = 0.18
    figure, axis = plt.subplots(figsize=(10, 5))
    for index, field in enumerate(fields):
        offsets = [position + (index - 1.5) * width for position in positions]
        axis.bar(offsets, [metrics[name][field] * 100 for name in names], width, label=field.upper())
    axis.set_xticks(positions, names)
    axis.set_ylim(0, 100)
    axis.set_ylabel("Score (%)")
    axis.set_title("Holdout metrics: model comparison")
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_dir / "model_metrics.png", dpi=160, bbox_inches="tight")
    plt.close(figure)


def plot_per_method_recall(predictions_by_model: dict, output_dir: Path) -> None:
    methods = sorted({row["method"] for rows in predictions_by_model.values() for row in rows if row["label"] == 1})
    names = list(predictions_by_model)
    figure, axis = plt.subplots(figsize=(11, 5))
    width = 0.24
    for index, name in enumerate(names):
        rows = predictions_by_model[name]
        recalls = []
        for method in methods:
            subset = [row for row in rows if row["method"] == method and row["label"] == 1]
            recalls.append(100 * sum(row["prediction"] == 1 for row in subset) / max(len(subset), 1))
        offsets = [position + (index - (len(names) - 1) / 2) * width for position in range(len(methods))]
        axis.bar(offsets, recalls, width, label=name)
    axis.set_xticks(range(len(methods)), methods, rotation=20, ha="right")
    axis.set_ylim(0, 100)
    axis.set_ylabel("Fake recall (%)")
    axis.set_title("Fake recall by manipulation method")
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_dir / "per_method_fake_recall.png", dpi=160, bbox_inches="tight")
    plt.close(figure)


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate all three detectors on holdout_test.")
    parser.add_argument("--dataset", default=str(HOLDOUT_DIR), help="Dataset root containing real/ and fake/.")
    parser.add_argument("--output", default=str(OUTPUT_DIR), help="Folder for metrics, CSV files, and charts.")
    parser.add_argument("--bidirectional", default=str(BIDIRECTIONAL_MODEL_PATH), help="Bidirectional best.pt path.")
    parser.add_argument("--swin", default=str(SWIN_MODEL_PATH), help="Swin best.pth path.")
    parser.add_argument("--gated", default=str(GATED_MODEL_PATH), help="Gated best.pt path.")
    parser.add_argument("--batch-size", type=int, default=4, help="Inference batch size; lower it if GPU memory is insufficient.")
    parser.add_argument("--workers", type=int, default=2, help="Image-loader worker processes. Use 0 if Windows has issues.")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--face-crop", action="store_true", help="Detect and crop a face before inference; slower but matches face-crop training.")
    parser.add_argument("--limit", type=int, default=None, help="Optional limit for a quick test run.")
    args = parser.parse_args()
    if args.batch_size < 1 or args.workers < 0 or not 0 <= args.threshold <= 1:
        parser.error("Invalid batch size, worker count, or threshold.")

    dataset_root, output_dir = Path(args.dataset), Path(args.output)
    records = collect_records(dataset_root)
    if args.limit:
        records = records[:args.limit]
    counts = defaultdict(int)
    for record in records:
        counts["FAKE" if record["label"] else "REAL"] += 1
    print(f"Found {len(records):,} images: {counts['REAL']:,} real, {counts['FAKE']:,} fake")
    print(f"Face crop: {'enabled' if args.face_crop else 'disabled'}")

    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    loaders = {
        "Bidirectional": (load_fusion_model, Path(args.bidirectional)),
        "Swin-T": (load_swin_model, Path(args.swin)),
        "Gated": (load_fusion_model, Path(args.gated)),
    }
    all_predictions, all_failures, metrics = {}, [], {}
    for name, (loader, checkpoint_path) in loaders.items():
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"{name} model file not found: {checkpoint_path}")
        print(f"\nLoading {name}: {checkpoint_path}")
        model, image_size = loader(checkpoint_path, device)
        predictions, failures = evaluate_model(name, model, image_size, records, args, device)
        all_predictions[name] = predictions
        all_failures.extend(failures)
        metrics[name] = calculate_metrics(predictions)
        print(f"{name}: accuracy={metrics[name]['accuracy'] * 100:.2f}%  F1={metrics[name]['f1'] * 100:.2f}%")
        del model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    combined_predictions = [row for rows in all_predictions.values() for row in rows]
    write_csv(output_dir / "per_image_predictions.csv", combined_predictions)
    write_csv(output_dir / "skipped_images.csv", [{"path": path, "error": error} for path, error in all_failures])
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)
    metric_rows = [{"model": name, **{key: value for key, value in values.items() if key != "confusion_matrix"}, **values["confusion_matrix"]} for name, values in metrics.items()]
    write_csv(output_dir / "metrics.csv", metric_rows)
    plot_confusion_matrices(metrics, output_dir)
    plot_metric_comparison(metrics, output_dir)
    plot_per_method_recall(all_predictions, output_dir)
    print(f"\nSaved results and charts to: {output_dir}")


if __name__ == "__main__":
    main()
