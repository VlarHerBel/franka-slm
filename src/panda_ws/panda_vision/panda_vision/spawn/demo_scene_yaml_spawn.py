"""Spawn entries desde YAML de escena demo (panda_controller/config/demo_scenes)."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml


def _demo_scenes_dir_candidates() -> List[str]:
    dirs: List[str] = []
    try:
        from ament_index_python.packages import get_package_share_directory

        share = get_package_share_directory("panda_controller")
        dirs.append(os.path.join(share, "config", "demo_scenes"))
    except Exception:
        pass
    here = Path(__file__).resolve().parents[3] / "panda_controller" / "config" / "demo_scenes"
    dirs.append(str(here))
    ws_src = Path(__file__).resolve().parents[4]
    dirs.append(str(ws_src / "panda_ws" / "panda_controller" / "config" / "demo_scenes"))
    out: List[str] = []
    seen: set = set()
    for d in dirs:
        if d and d not in seen and os.path.isdir(d):
            seen.add(d)
            out.append(d)
    return out


def demo_scene_yaml_path(scene_id: str) -> Optional[str]:
    sid = str(scene_id or "").strip().lower()
    if not sid:
        return None
    for base in _demo_scenes_dir_candidates():
        path = os.path.join(base, f"{sid}.yaml")
        if os.path.isfile(path):
            return path
    return None


def load_demo_scene_spawn_entries(scene_id: str) -> Optional[List[Dict[str, Any]]]:
    path = demo_scene_yaml_path(scene_id)
    if not path:
        return None
    with open(path, "r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if str(raw.get("spawn_mode", "")).strip().lower() == "random":
        return None
    pick_order = [
        str(x).strip().lower()
        for x in (raw.get("pick_order") or [])
        if str(x).strip()
    ]
    objects = raw.get("objects") or {}
    entries: List[Dict[str, Any]] = []
    for idx, label in enumerate(pick_order):
        spec = objects.get(label)
        if not isinstance(spec, dict):
            continue
        pose = spec.get("pose") or {}
        entry: Dict[str, Any] = {
            "label": label,
            "x": float(pose.get("x", 0.0)),
            "y": float(pose.get("y", 0.0)),
            "yaw": float(pose.get("yaw", 0.0)),
            "order_index": idx,
        }
        seed = spec.get("seed")
        if seed is not None:
            entry["seed"] = int(seed)
        entries.append(entry)
    return entries if entries else None


_DEPOSIT_SLOT_XY: Tuple[Tuple[float, float], ...] = (
    (-0.37, 0.08),
    (-0.54, 0.08),
    (-0.37, -0.10),
    (-0.54, -0.10),
)
DEPOSIT_OBJECT_SPAWN_Z_M = 0.22


def load_demo_scene_deposit_spawn_entries(scene_id: str) -> List[Dict[str, Any]]:
    """Objetos precargados en la caja de depósito (initial_deposits)."""
    path = demo_scene_yaml_path(scene_id)
    if not path:
        return []
    with open(path, "r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    entries: List[Dict[str, Any]] = []
    for item in raw.get("initial_deposits") or []:
        if not isinstance(item, dict):
            continue
        label = str(item.get("label") or "").strip().lower()
        try:
            slot_index = int(item.get("slot_index"))
        except (TypeError, ValueError):
            continue
        if not label or not 0 <= slot_index < len(_DEPOSIT_SLOT_XY):
            continue
        x, y = _DEPOSIT_SLOT_XY[slot_index]
        entries.append(
            {
                "label": label,
                "x": float(x),
                "y": float(y),
                "yaw": float(item.get("yaw", 0.0)),
                "z": float(item.get("z", DEPOSIT_OBJECT_SPAWN_Z_M)),
                "spawn_region": "deposit_box",
                "deposit_slot_index": int(slot_index),
            }
        )
    return entries
