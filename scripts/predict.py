from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from deepfake_detector.inference.predict import predict_images  # noqa: E402
from deepfake_detector.utils.config import load_config  # noqa: E402
from deepfake_detector.utils.logging import configure_logging  # noqa: E402

# ── Fixed paths ───────────────────────────────────────────────────────────────
CONFIG_PATH     = ROOT / "configs" / "train_fusion.yaml"
CHECKPOINT_PATH = ROOT / "outputs" / "fusion_baseline" / "best.pt"


def _overlay_label_on_image(image_path: str, prediction: str, confidence: float):
    """Open an image, draw the FAKE/REAL verdict on top, and display it."""
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)

    label_text = f"{prediction}  ({confidence:.1f}%)"

    # Pick a font size proportional to the image
    font_size = max(28, img.width // 16)
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except OSError:
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
        except OSError:
            font = ImageFont.load_default()

    # Measure text
    bbox = draw.textbbox((0, 0), label_text, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]

    # Colours
    if prediction == "FAKE":
        bg_color   = (220, 38, 38)     # red
        text_color = (255, 255, 255)
    else:
        bg_color   = (22, 163, 74)     # green
        text_color = (255, 255, 255)

    # Draw banner at the top
    pad_x, pad_y = 16, 10
    banner_x0 = (img.width - text_w) // 2 - pad_x
    banner_y0 = 12
    banner_x1 = banner_x0 + text_w + 2 * pad_x
    banner_y1 = banner_y0 + text_h + 2 * pad_y

    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    overlay_draw = ImageDraw.Draw(overlay)
    overlay_draw.rounded_rectangle(
        [banner_x0, banner_y0, banner_x1, banner_y1],
        radius=12,
        fill=(*bg_color, 210),
    )
    img = Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")

    draw = ImageDraw.Draw(img)
    text_x = banner_x0 + pad_x
    text_y = banner_y0 + pad_y
    draw.text((text_x, text_y), label_text, fill=text_color, font=font)

    # Show
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.imshow(img)
    ax.axis("off")
    ax.set_title(Path(image_path).name, fontsize=14, fontweight="bold", pad=12)
    plt.tight_layout()
    plt.show()


def main() -> None:
    # ── Ask for the image path ──
    image_path = input("Enter the path to the image: ").strip().strip('"').strip("'")
    if not image_path:
        print("Error: No image path provided.")
        sys.exit(1)
    if not Path(image_path).exists():
        print(f"Error: Path does not exist: {image_path}")
        sys.exit(1)

    configure_logging()
    config  = load_config(str(CONFIG_PATH))
    results = predict_images(
        config,
        checkpoint_path=str(CHECKPOINT_PATH),
        input_path=image_path,
        use_face_crop=True,
        save_crops=False,
    )

    # ── Print result & show image ──
    for r in results:
        prediction = r["prediction"].upper()
        prob_fake  = float(r["probability_fake"])
        confidence = max(prob_fake, 1.0 - prob_fake) * 100

        print(f"\nResult     : {prediction}")
        print(f"Confidence : {confidence:.1f}%")
        print(f"Fake prob  : {prob_fake:.6f}")

        _overlay_label_on_image(r["path"], prediction, confidence)


if __name__ == "__main__":
    main()
