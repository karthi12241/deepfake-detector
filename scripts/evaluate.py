from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from deepfake_detector.training.evaluate import evaluate_checkpoint, metrics_to_json  # noqa: E402
from deepfake_detector.utils.config import load_config  # noqa: E402
from deepfake_detector.utils.logging import configure_logging  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained checkpoint.")
    parser.add_argument("--config", default="configs/train_fusion.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    configure_logging()
    config = load_config(args.config)
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = str(Path(args.checkpoint).resolve().parent / "reports" / f"evaluation_{args.split}")
    metrics = evaluate_checkpoint(config, args.checkpoint, args.split, output_dir=output_dir)
    print(metrics_to_json(metrics))


if __name__ == "__main__":
    main()
