"""Compare Bidirectional, Swin-T, and Gated models on one image.

Example:
    python compare_models.py --image "C:\\Users\\HP\\Downloads\\woo\\test.jpg"
"""

from __future__ import annotations

import argparse
import gc
import time
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torchvision.models import swin_t

from test import FaceCrop, build_model, crop_largest_face, make_transform, unpack_checkpoint


ROOT = Path(__file__).resolve().parent
# Paste the image you want to compare here, then run:  python compare_models.py
IMAGE_PATH = r"C:\Users\HP\Downloads\woo\test.jpg"

DEFAULT_MODELS = {
    "Bidirectional": ROOT / "best.pt",
    "Swin-T": ROOT / "Swin" / "best.pth",
    "Gated": ROOT / "Gated" / "best.pt",
}


class SwinBinaryClassifier(nn.Module):
    """The architecture corresponding to Swin/best.pth."""
    def __init__(self) -> None:
        super().__init__()
        self.backbone = swin_t(weights=None)
        self.backbone.head = nn.Sequential(
            nn.Dropout(0.30),
            nn.Linear(self.backbone.head.in_features, 1),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.backbone(images).squeeze(1)


def load_torch_file(path: Path, device: torch.device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:  # PyTorch versions before the weights_only argument
        return torch.load(path, map_location=device)


def remove_prefixes(state_dict: dict) -> dict:
    """Remove common training-wrapper prefixes from a state dictionary."""
    for prefix in ("module.", "model."):
        if state_dict and all(key.startswith(prefix) for key in state_dict):
            return {key[len(prefix):]: value for key, value in state_dict.items()}
    return state_dict


def predict_fusion(checkpoint_path: Path, image: Image.Image, device: torch.device) -> tuple[float, float]:
    checkpoint = load_torch_file(checkpoint_path, device)
    state_dict, model_config = unpack_checkpoint(checkpoint)
    model = build_model(model_config, state_dict).to(device).eval()
    model.load_state_dict(state_dict)

    saved_config = checkpoint.get("config", {}) if isinstance(checkpoint, dict) else {}
    image_size = int(saved_config.get("training", {}).get("image_size", 224)) if isinstance(saved_config, dict) else 224
    batch = make_transform(image_size)(image).unsqueeze(0).to(device)
    started = time.perf_counter()
    with torch.inference_mode():
        probability = float(torch.sigmoid(model(batch)).item())
    elapsed_ms = (time.perf_counter() - started) * 1000
    del model, checkpoint, state_dict, batch
    return probability, elapsed_ms


def predict_swin(checkpoint_path: Path, image: Image.Image, device: torch.device) -> tuple[float, float]:
    saved = load_torch_file(checkpoint_path, device)
    if isinstance(saved, dict) and "state_dict" in saved:
        saved = saved["state_dict"]
    if not isinstance(saved, dict):
        raise ValueError("Swin checkpoint does not contain a state dictionary.")

    state_dict = remove_prefixes(saved)
    model = SwinBinaryClassifier().to(device).eval()
    model.load_state_dict(state_dict)
    batch = make_transform(224)(image).unsqueeze(0).to(device)
    started = time.perf_counter()
    with torch.inference_mode():
        probability = float(torch.sigmoid(model(batch)).item())
    elapsed_ms = (time.perf_counter() - started) * 1000
    del model, saved, state_dict, batch
    return probability, elapsed_ms


def print_table(rows: list[tuple[str, str, float, float]]) -> None:
    headers = ("Model", "Prediction", "Fake probability", "Inference")
    formatted = [
        (name, verdict, f"{probability * 100:.2f}%", f"{elapsed_ms:.0f} ms")
        for name, verdict, probability, elapsed_ms in rows
    ]
    widths = [max(len(header), *(len(row[index]) for row in formatted)) for index, header in enumerate(headers)]
    border = "+" + "+".join("-" * (width + 2) for width in widths) + "+"
    print("\n" + border)
    print("|" + "|".join(f" {header:<{width}} " for header, width in zip(headers, widths)) + "|")
    print(border)
    for row in formatted:
        print("|" + "|".join(f" {value:<{width}} " for value, width in zip(row, widths)) + "|")
    print(border)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare three deepfake detectors on one image.")
    parser.add_argument(
        "--image",
        default=IMAGE_PATH,
        help="Path to the image to classify. Overrides IMAGE_PATH at the top of this file.",
    )
    parser.add_argument("--threshold", type=float, default=0.5, help="FAKE decision threshold (default: 0.5).")
    parser.add_argument("--no-face-crop", action="store_true", help="Use the full image rather than the largest detected face.")
    parser.add_argument("--bidirectional", default=str(DEFAULT_MODELS["Bidirectional"]), help="Bidirectional checkpoint path.")
    parser.add_argument("--swin", default=str(DEFAULT_MODELS["Swin-T"]), help="Swin checkpoint path.")
    parser.add_argument("--gated", default=str(DEFAULT_MODELS["Gated"]), help="Gated checkpoint path.")
    args = parser.parse_args()

    image_path = Path(args.image)
    model_paths = {
        "Bidirectional": Path(args.bidirectional),
        "Swin-T": Path(args.swin),
        "Gated": Path(args.gated),
    }
    if not args.image:
        parser.error("Set IMAGE_PATH at the top of compare_models.py, or pass --image PATH.")
    if not image_path.is_file():
        parser.error(f"Image file not found: {image_path}")
    for name, path in model_paths.items():
        if not path.is_file():
            parser.error(f"{name} model file not found: {path}")
    if not 0.0 <= args.threshold <= 1.0:
        parser.error("--threshold must be between 0 and 1")

    with Image.open(image_path) as opened:
        original = opened.convert("RGB")
    crop = FaceCrop(original, "full image requested") if args.no_face_crop else crop_largest_face(original, 0.35, 48)
    print(f"Image: {image_path.name}  |  Input used: {crop.source}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    runners = {
        "Bidirectional": predict_fusion,
        "Swin-T": predict_swin,
        "Gated": predict_fusion,
    }
    rows = []
    for name in ("Bidirectional", "Swin-T", "Gated"):
        try:
            probability, elapsed_ms = runners[name](model_paths[name], crop.image, device)
            verdict = "FAKE" if probability >= args.threshold else "REAL"
            rows.append((name, verdict, probability, elapsed_ms))
        finally:
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()

    print_table(rows)


if __name__ == "__main__":
    main()
