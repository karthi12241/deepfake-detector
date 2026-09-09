from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from deepfake_detector.training.train import train  # noqa: E402
from deepfake_detector.utils.config import load_config  # noqa: E402
from deepfake_detector.utils.logging import configure_logging  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Xception+ViT deepfake fusion model.")
    parser.add_argument("--config", default="configs/train_fusion.yaml")
    parser.add_argument("--resume", default=None, help="Optional checkpoint path to continue training.")
    args = parser.parse_args()

    configure_logging()
    config = load_config(args.config)
    try:
        best_path = train(config, resume_checkpoint=args.resume)
        print(f"Best checkpoint: {best_path}")
    except Exception:
        logging.getLogger(__name__).exception("Training failed.")
        raise


if __name__ == "__main__":
    main()
