"""Run one-image inference with a trained cross-attention checkpoint.

Example:
    python test.py --model "C:\\path\\to\\best.pt" --image "C:\\path\\to\\image.jpg"
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import timm
import torch
import torch.nn as nn
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class BidirectionalCrossAttentionFusion(nn.Module):
    def __init__(self, cnn_dim, vit_dim, fusion_dim, num_heads=8, dropout=0.30):
        super().__init__()
        self.cnn_proj = nn.Sequential(nn.LayerNorm(cnn_dim), nn.Linear(cnn_dim, fusion_dim), nn.GELU(), nn.Dropout(dropout))
        self.vit_proj = nn.Sequential(nn.LayerNorm(vit_dim), nn.Linear(vit_dim, fusion_dim), nn.GELU(), nn.Dropout(dropout))
        self.cnn_to_vit = nn.MultiheadAttention(fusion_dim, num_heads, dropout=dropout, batch_first=True)
        self.vit_to_cnn = nn.MultiheadAttention(fusion_dim, num_heads, dropout=dropout, batch_first=True)
        self.cnn_norm, self.vit_norm = nn.LayerNorm(fusion_dim), nn.LayerNorm(fusion_dim)

    def forward(self, cnn_tokens, vit_tokens):
        cnn_projected, vit_projected = self.cnn_proj(cnn_tokens), self.vit_proj(vit_tokens)
        cnn_enriched, _ = self.cnn_to_vit(cnn_projected, vit_projected, vit_projected)
        vit_enriched, _ = self.vit_to_cnn(vit_projected, cnn_projected, cnn_projected)
        cnn_enriched = self.cnn_norm(cnn_enriched + cnn_projected)
        vit_enriched = self.vit_norm(vit_enriched + vit_projected)
        return torch.cat([cnn_enriched.mean(1), vit_enriched.mean(1)], dim=1)


class XceptionViTCrossAttentionModel(nn.Module):
    def __init__(self, cnn_name="legacy_xception", vit_name="vit_base_patch16_224", pretrained=False,
                 fusion_dim=512, num_heads=8, dropout=0.30):
        super().__init__()
        # pretrained is deliberately False: the checkpoint supplies every weight.
        self.cnn = timm.create_model(cnn_name, pretrained=False, num_classes=0, global_pool="")
        self.vit = timm.create_model(vit_name, pretrained=False, num_classes=0)
        with torch.no_grad():
            sample = torch.zeros(1, 3, 224, 224)
            cnn_dim = self.cnn(sample).shape[1]
            vit_dim = self.vit.forward_features(sample).shape[2]
        self.fusion = BidirectionalCrossAttentionFusion(cnn_dim, vit_dim, fusion_dim, num_heads, dropout)
        self.classifier = nn.Sequential(
            nn.LayerNorm(fusion_dim * 2), nn.Linear(fusion_dim * 2, fusion_dim),
            nn.GELU(), nn.Dropout(dropout), nn.Linear(fusion_dim, 1),
        )

    def forward(self, images):
        cnn_tokens = self.cnn(images).flatten(2).transpose(1, 2)
        vit_tokens = self.vit.forward_features(images)[:, 1:, :]
        return self.classifier(self.fusion(cnn_tokens, vit_tokens)).squeeze(1)


class GatedBidirectionalFusion(nn.Module):
    """Legacy fusion used by checkpoints containing ``fusion.gate.0.*`` keys.

    It retains the CNN feature, ViT feature, and a learned gated blend of both;
    that is why its classifier receives ``fusion_dim * 3`` features.
    """
    def __init__(self, cnn_dim, vit_dim, fusion_dim, dropout=0.30):
        super().__init__()
        self.cnn_proj = nn.Sequential(nn.LayerNorm(cnn_dim), nn.Linear(cnn_dim, fusion_dim), nn.GELU(), nn.Dropout(dropout))
        self.vit_proj = nn.Sequential(nn.LayerNorm(vit_dim), nn.Linear(vit_dim, fusion_dim), nn.GELU(), nn.Dropout(dropout))
        self.gate = nn.Sequential(nn.Linear(fusion_dim * 2, fusion_dim), nn.Sigmoid())

    def forward(self, cnn_features, vit_features):
        cnn_projected = self.cnn_proj(cnn_features)
        vit_projected = self.vit_proj(vit_features)
        gate = self.gate(torch.cat([cnn_projected, vit_projected], dim=1))
        blended = gate * cnn_projected + (1.0 - gate) * vit_projected
        return torch.cat([cnn_projected, vit_projected, blended], dim=1)


class XceptionViTGatedFusionModel(nn.Module):
    def __init__(self, cnn_name="legacy_xception", vit_name="vit_base_patch16_224", pretrained=False,
                 fusion_dim=512, num_heads=8, dropout=0.30):
        super().__init__()
        self.cnn = timm.create_model(cnn_name, pretrained=False, num_classes=0, global_pool="")
        self.vit = timm.create_model(vit_name, pretrained=False, num_classes=0)
        with torch.no_grad():
            sample = torch.zeros(1, 3, 224, 224)
            cnn_dim = self.cnn(sample).shape[1]
            vit_dim = self.vit.forward_features(sample).shape[2]
        self.fusion = GatedBidirectionalFusion(cnn_dim, vit_dim, fusion_dim, dropout)
        self.classifier = nn.Sequential(
            nn.LayerNorm(fusion_dim * 3), nn.Linear(fusion_dim * 3, fusion_dim),
            nn.GELU(), nn.Dropout(dropout), nn.Linear(fusion_dim, 1),
        )

    def forward(self, images):
        cnn_features = self.cnn(images).mean(dim=(2, 3))
        vit_features = self.vit.forward_features(images)[:, 0, :]
        return self.classifier(self.fusion(cnn_features, vit_features)).squeeze(1)


def build_model(config: dict, state_dict: dict | None = None) -> nn.Module:
    """Build the exact fusion version identified by checkpoint parameter names."""
    inferred_fusion_dim = None
    if state_dict and "fusion.cnn_proj.1.weight" in state_dict:
        inferred_fusion_dim = state_dict["fusion.cnn_proj.1.weight"].shape[0]
    common = dict(
        cnn_name=config.get("cnn_name", "legacy_xception"),
        vit_name=config.get("vit_name", "vit_base_patch16_224"),
        fusion_dim=int(config.get("fusion_dim", inferred_fusion_dim or 512)),
        num_heads=int(config.get("num_heads", 8)),
        dropout=float(config.get("dropout", 0.30)),
    )
    if state_dict and "fusion.gate.0.weight" in state_dict:
        return XceptionViTGatedFusionModel(**common)
    return XceptionViTCrossAttentionModel(
        **common,
    )


def unpack_checkpoint(loaded: object) -> tuple[dict, dict]:
    """Accept full notebook checkpoints and common plain state-dict formats."""
    if not isinstance(loaded, dict):
        raise ValueError("Unsupported model file: torch.load() did not return a dictionary.")

    state_dict = None
    for key in ("model_state", "model_state_dict", "state_dict"):
        value = loaded.get(key)
        if isinstance(value, dict):
            state_dict = value
            break

    # torch.save(model.state_dict(), path) stores parameter keys at the top level.
    if state_dict is None and any(torch.is_tensor(value) for value in loaded.values()):
        state_dict = loaded
    if state_dict is None:
        raise ValueError(
            "Could not find model weights. Expected model_state, model_state_dict, "
            "state_dict, or a plain PyTorch state dictionary."
        )

    # DataParallel and wrapper models commonly add one of these prefixes.
    for prefix in ("module.", "model."):
        if state_dict and all(key.startswith(prefix) for key in state_dict):
            state_dict = {key[len(prefix):]: value for key, value in state_dict.items()}

    saved_config = loaded.get("config", {})
    if not isinstance(saved_config, dict):
        saved_config = {}
    model_config = saved_config.get("model", saved_config)
    if not isinstance(model_config, dict):
        model_config = {}
    return state_dict, model_config


@dataclass(frozen=True)
class FaceCrop:
    image: Image.Image
    source: str


def crop_largest_face(image: Image.Image, margin: float, min_face_size: int) -> FaceCrop:
    """Use OpenCV when available; otherwise safely evaluate the full image."""
    try:
        import cv2
        import numpy as np
        rgb = np.asarray(image.convert("RGB"))
        gray = cv2.equalizeHist(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY))
        cascade = cv2.CascadeClassifier(str(Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"))
        faces = cascade.detectMultiScale(gray, 1.08, 5, minSize=(min_face_size, min_face_size))
        if len(faces):
            x, y, width, height = max(faces, key=lambda box: box[2] * box[3])
            pad_x, pad_y = int(width * margin), int(height * margin)
            left, top = max(0, x - pad_x), max(0, y - pad_y)
            right, bottom = min(image.width, x + width + pad_x), min(image.height, y + height + pad_y)
            return FaceCrop(image.crop((left, top, right, bottom)), "detected face")
    except ImportError:
        pass
    return FaceCrop(image, "full image fallback")


def make_transform(image_size: int):
    return transforms.Compose([
        transforms.Resize(image_size + 32, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def save_result_image(image: Image.Image, prediction: str, confidence: float, output_path: Path) -> None:
    image = image.copy()
    draw = ImageDraw.Draw(image)
    text = f"{prediction} ({confidence:.1f}%)"
    font = ImageFont.load_default()
    box = draw.textbbox((0, 0), text, font=font)
    color = (220, 38, 38) if prediction == "FAKE" else (22, 163, 74)
    draw.rectangle((8, 8, box[2] + 24, box[3] + 20), fill=color)
    draw.text((16, 14), text, fill="white", font=font)
    image.save(output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Classify one image as REAL or FAKE.")
    parser.add_argument("--model", required=True, help="Path to trained best.pt checkpoint.")
    parser.add_argument("--image", required=True, help="Path to one input image.")
    parser.add_argument("--threshold", type=float, default=0.5, help="Fake-probability threshold (default: 0.5).")
    parser.add_argument("--no-face-crop", action="store_true", help="Classify the full image instead of its largest detected face.")
    parser.add_argument("--output", default=None, help="Optional path for an annotated output image.")
    args = parser.parse_args()

    model_path, image_path = Path(args.model), Path(args.image)
    if not model_path.is_file():
        parser.error(f"Model file not found: {model_path}")
    if not image_path.is_file():
        parser.error(f"Image file not found: {image_path}")
    if not 0.0 <= args.threshold <= 1.0:
        parser.error("--threshold must be between 0 and 1")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(model_path, map_location=device)
    state_dict, model_config = unpack_checkpoint(checkpoint)
    model = build_model(model_config, state_dict).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    with Image.open(image_path) as opened:
        original = opened.convert("RGB")
    config = checkpoint.get("config", {}) if isinstance(checkpoint, dict) else {}
    if not isinstance(config, dict):
        config = {}
    face_config = config.get("inference", {}).get("face_crop", {})
    if args.no_face_crop:
        crop = FaceCrop(original, "full image requested")
    else:
        crop = crop_largest_face(original, float(face_config.get("margin", 0.35)), int(face_config.get("min_face_size", 48)))

    image_size = int(config.get("training", {}).get("image_size", 224))
    batch = make_transform(image_size)(crop.image).unsqueeze(0).to(device)
    with torch.inference_mode():
        fake_probability = float(torch.sigmoid(model(batch)).item())

    prediction = "FAKE" if fake_probability >= args.threshold else "REAL"
    confidence = max(fake_probability, 1.0 - fake_probability) * 100
    print(f"Image:            {image_path}")
    print(f"Crop used:        {crop.source}")
    print(f"Prediction:       {prediction}")
    print(f"Confidence:       {confidence:.2f}%")
    print(f"Fake probability: {fake_probability:.6f}")

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        save_result_image(original, prediction, confidence, output_path)
        print(f"Saved result:     {output_path}")


if __name__ == "__main__":
    main()
