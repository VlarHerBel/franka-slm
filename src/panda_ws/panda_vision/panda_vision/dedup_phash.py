#!/usr/bin/env python3
"""Deduplicate dataset images with perceptual hash."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

import imagehash
from PIL import Image


def _pairs(dataset_dir: Path) -> List[Tuple[str, Path, Path]]:
    items: List[Tuple[str, Path, Path]] = []
    for split in ("train", "val"):
        image_dir = dataset_dir / "images" / split
        label_dir = dataset_dir / "labels" / split
        for image_path in sorted(image_dir.glob("*.png")):
            label_path = label_dir / f"{image_path.stem}.txt"
            if label_path.is_file():
                items.append((split, image_path, label_path))
    return items


def dedup_dataset(dataset_dir: Path, threshold: int) -> Path:
    out_dir = dataset_dir / "dedup"
    if out_dir.exists():
        shutil.rmtree(out_dir)
    removed = 0
    kept_hashes: Dict[str, imagehash.ImageHash] = {}

    for split, image_path, label_path in _pairs(dataset_dir):
        image_hash = imagehash.phash(Image.open(image_path))
        duplicate = False
        for kept_name, kept_hash in kept_hashes.items():
            if image_hash - kept_hash <= threshold:
                duplicate = True
                removed += 1
                break
        if duplicate:
            continue
        kept_hashes[image_path.name] = image_hash
        target_image = out_dir / "images" / split / image_path.name
        target_label = out_dir / "labels" / split / label_path.name
        target_image.parent.mkdir(parents=True, exist_ok=True)
        target_label.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(image_path, target_image)
        shutil.copy2(label_path, target_label)

    data_yaml = dataset_dir / "data.yaml"
    if data_yaml.is_file():
        shutil.copy2(data_yaml, out_dir / "data.yaml")

    print(f"dedup_output={out_dir}")
    print(f"duplicates_removed={removed}")
    print(f"images_kept={len(kept_hashes)}")
    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--threshold", type=int, default=1)
    args = parser.parse_args()
    dedup_dataset(args.dataset.expanduser().resolve(), args.threshold)


if __name__ == "__main__":
    main()
