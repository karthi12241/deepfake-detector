from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from deepfake_detector.data.build_manifest import (  # noqa: E402
    build_manifest_records,
    summarize_manifest,
    write_manifest,
)
from deepfake_detector.utils.config import load_config, resolve_path  # noqa: E402
from deepfake_detector.utils.logging import configure_logging  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a leakage-safe deepfake manifest.")
    parser.add_argument("--config", default="configs/train_fusion.yaml")
    args = parser.parse_args()

    configure_logging()
    config = load_config(args.config)
    records = build_manifest_records(
        archive_root=resolve_path(config["dataset"]["archive_root"]),
        split_seed=int(config["dataset"].get("split_seed", 42)),
        split_ratios=config["dataset"]["split_ratios"],
        extra_image_roots=config["dataset"].get("extra_image_roots", []),
    )
    output_path = write_manifest(records, resolve_path(config["dataset"]["manifest_path"]))
    print(json.dumps({"manifest": str(output_path), "summary": summarize_manifest(records)}, indent=2))


if __name__ == "__main__":
    main()
