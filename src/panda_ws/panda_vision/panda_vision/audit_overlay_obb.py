#!/usr/bin/env python3
"""Render audit overlays for Ultralytics OBB labels."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import cv2
import numpy as np


def _collect_pairs(dataset_dir: Path) -> List[tuple[Path, Path]]:
    pairs: List[tuple[Path, Path]] = []
    for split in ("train", "val"):
        for image_path in sorted((dataset_dir / "images" / split).glob("*.png")):
            label_path = dataset_dir / "labels" / split / f"{image_path.stem}.txt"
            if label_path.is_file():
                pairs.append((image_path, label_path))
    return pairs


def _draw_label(image: np.ndarray, line: str) -> None:
    parts = line.split()
    if len(parts) != 9:
        return
    coords = np.array([float(value) for value in parts[1:]], dtype=np.float32).reshape(4, 2)
    height, width = image.shape[:2]
    coords[:, 0] *= width
    coords[:, 1] *= height
    pts = coords.astype(np.int32).reshape((-1, 1, 2))
    cv2.polylines(image, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
    anchor = tuple(pts[0, 0].tolist())
    cv2.putText(
        image,
        parts[0],
        anchor,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 0, 255),
        2,
        cv2.LINE_AA,
    )


def render_audit(dataset_dir: Path, sample_count: int) -> Path:
    out_dir = dataset_dir / "audit_overlay"
    out_dir.mkdir(parents=True, exist_ok=True)
    pairs = _collect_pairs(dataset_dir)[:sample_count]
    for image_path, label_path in pairs:
        image = cv2.imread(str(image_path))
        if image is None:
            continue
        for line in label_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                _draw_label(image, line)
        target = out_dir / f"{image_path.stem}_overlay.png"
        cv2.imwrite(str(target), image)
    print(f"audit_overlay={out_dir}")
    print(f"samples_written={len(list(out_dir.glob('*.png')))}")
    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--sample-count", type=int, default=50)
    args = parser.parse_args()
    render_audit(args.dataset.expanduser().resolve(), args.sample_count)


if __name__ == "__main__":
    main()
