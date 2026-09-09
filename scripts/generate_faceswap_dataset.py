from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import cv2  # noqa: E402
from face2face import Face2Face  # noqa: E402
from tqdm import tqdm  # noqa: E402

from deepfake_detector.data.build_manifest import IMAGE_EXTENSIONS  # noqa: E402
from deepfake_detector.utils.config import load_config, resolve_path  # noqa: E402
from deepfake_detector.utils.logging import configure_logging  # noqa: E402


def list_video_frames(real_root: Path) -> dict[str, list[Path]]:
    videos: dict[str, list[Path]] = {}
    for video_dir in sorted(p for p in real_root.iterdir() if p.is_dir()):
        frames = sorted(
            p for p in video_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        )
        if frames:
            videos[video_dir.name] = frames
    if len(videos) < 2:
        raise RuntimeError(f"Need at least two real video folders in {real_root}")
    return videos


def split_video_ids(video_ids: list[str], seed: int) -> dict[str, list[str]]:
    rng = random.Random(seed)
    shuffled = list(video_ids)
    rng.shuffle(shuffled)
    n = len(shuffled)
    train_end = int(n * 0.70)
    val_end = train_end + int(n * 0.15)
    return {
        "train": shuffled[:train_end],
        "val": shuffled[train_end:val_end],
        "test": shuffled[val_end:],
    }


def choose_frame(frames: list[Path], rng: random.Random) -> Path:
    return rng.choice(frames)


def write_image(path: Path, image) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise RuntimeError(f"Failed to save image: {path}")


def generate_split(
    f2f: Face2Face,
    videos: dict[str, list[Path]],
    split_ids: list[str],
    split: str,
    output_root: Path,
    count: int,
    rng: random.Random,
    image_extension: str,
    copy_matching_real: bool,
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    if len(split_ids) < 2:
        raise RuntimeError(f"Need at least two video ids for split={split}")

    for index in tqdm(range(count), desc=f"Generating {split}"):
        source_id, target_id = rng.sample(split_ids, 2)
        source_path = choose_frame(videos[source_id], rng)
        target_path = choose_frame(videos[target_id], rng)
        sample_id = f"{index:06d}_{source_id}_to_{target_id}"

        fake_path = output_root / split / "fake" / f"{sample_id}{image_extension}"
        real_path = output_root / split / "real" / f"{sample_id}{image_extension}"

        if not fake_path.exists():
            result = f2f.swap_img_to_img(str(source_path), str(target_path))
            write_image(fake_path, result)

        if copy_matching_real and not real_path.exists():
            target_image = cv2.imread(str(target_path), cv2.IMREAD_COLOR)
            if target_image is None:
                raise RuntimeError(f"Failed to read target image: {target_path}")
            write_image(real_path, target_image)

        rows.append(
            {
                "split": split,
                "sample_id": sample_id,
                "source_id": source_id,
                "target_id": target_id,
                "source_path": str(source_path),
                "target_path": str(target_path),
                "fake_path": str(fake_path),
                "real_path": str(real_path) if copy_matching_real else "",
            }
        )
    return rows


def write_generation_metadata(output_root: Path, rows: list[dict[str, str]]) -> None:
    metadata_path = output_root / "generation_metadata.csv"
    with metadata_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "split",
                "sample_id",
                "source_id",
                "target_id",
                "source_path",
                "target_path",
                "fake_path",
                "real_path",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a small Face2Face-style faceswap dataset.")
    parser.add_argument("--config", default="configs/train_fusion.yaml")
    parser.add_argument("--train-count", type=int, default=None)
    parser.add_argument("--val-count", type=int, default=None)
    parser.add_argument("--test-count", type=int, default=None)
    args = parser.parse_args()

    configure_logging()
    config = load_config(args.config)
    generation_cfg = config["synthetic_faceswap"]

    real_root = resolve_path(generation_cfg["archive_real_root"])
    output_root = resolve_path(generation_cfg["output_root"])
    output_root.mkdir(parents=True, exist_ok=True)

    seed = int(generation_cfg.get("seed", 42))
    rng = random.Random(seed)
    videos = list_video_frames(real_root)
    split_ids = split_video_ids(sorted(videos), seed=seed)
    counts = dict(generation_cfg.get("counts", {}))
    if args.train_count is not None:
        counts["train"] = args.train_count
    if args.val_count is not None:
        counts["val"] = args.val_count
    if args.test_count is not None:
        counts["test"] = args.test_count

    f2f = Face2Face(device_id=int(generation_cfg.get("device_id", -1)))
    image_extension = str(generation_cfg.get("image_extension", ".jpg"))
    copy_matching_real = bool(generation_cfg.get("copy_matching_real", True))

    rows: list[dict[str, str]] = []
    for split in ("train", "val", "test"):
        rows.extend(
            generate_split(
                f2f=f2f,
                videos=videos,
                split_ids=split_ids[split],
                split=split,
                output_root=output_root,
                count=int(counts.get(split, 0)),
                rng=rng,
                image_extension=image_extension,
                copy_matching_real=copy_matching_real,
            )
        )
    write_generation_metadata(output_root, rows)
    print(f"Generated {len(rows)} face swaps under {output_root}")
    print(f"Metadata: {output_root / 'generation_metadata.csv'}")


if __name__ == "__main__":
    main()

