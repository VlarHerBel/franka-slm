#!/usr/bin/env python3
"""Validate Ultralytics OBB labels."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List

import yaml


def _label_files(dataset_dir: Path) -> List[Path]:
    return sorted((dataset_dir / "labels").glob("*/*.txt"))


def _read_names(dataset_dir: Path) -> List[str]:
    data_yaml = dataset_dir / "data.yaml"
    if not data_yaml.is_file():
        return []
    payload = yaml.safe_load(data_yaml.read_text(encoding="utf-8")) or {}
    names = payload.get("names", {})
    if isinstance(names, dict):
        return [names[idx] for idx in sorted(names)]
    if isinstance(names, list):
        return names
    return []


def validate_labels(dataset_dir: Path, warn_empty_ratio: float) -> int:
    names = _read_names(dataset_dir)
    class_count = len(names)
    label_files = _label_files(dataset_dir)
    empty_files = 0
    errors = 0

    for label_path in label_files:
        lines = [ln.strip() for ln in label_path.read_text(encoding="utf-8").splitlines()]
        non_empty = [ln for ln in lines if ln]
        if not non_empty:
            empty_files += 1
            continue
        for line_idx, line in enumerate(non_empty, start=1):
            parts = line.split()
            if len(parts) != 9:
                print(f"ERROR {label_path}: línea {line_idx} tiene {len(parts)} campos, esperado 9")
                errors += 1
                continue
            try:
                class_id = int(parts[0])
                coords = [float(value) for value in parts[1:]]
            except ValueError:
                print(f"ERROR {label_path}: línea {line_idx} contiene valores no numéricos")
                errors += 1
                continue
            if class_count and not (0 <= class_id < class_count):
                print(f"ERROR {label_path}: línea {line_idx} class_id={class_id} fuera de rango")
                errors += 1
            if any(value < 0.0 or value > 1.0 for value in coords):
                print(f"ERROR {label_path}: línea {line_idx} coords fuera de [0,1]")
                errors += 1

    total = max(1, len(label_files))
    empty_ratio = empty_files / total
    print(
        f"Resumen labels: total={len(label_files)} empty={empty_files} "
        f"empty_ratio={empty_ratio:.3f} class_count={class_count}"
    )
    if empty_ratio > warn_empty_ratio:
        print(
            f"WARN empty_ratio={empty_ratio:.3f} supera warn_empty_ratio={warn_empty_ratio:.3f}"
        )
    return 1 if errors else 0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--warn-empty-ratio", type=float, default=0.05)
    args = parser.parse_args()
    raise SystemExit(validate_labels(args.dataset.expanduser().resolve(), args.warn_empty_ratio))


if __name__ == "__main__":
    main()
