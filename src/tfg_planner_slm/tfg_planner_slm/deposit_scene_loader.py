"""Carga escenas demo con depósito precargado (YAML en panda_controller/config/demo_scenes)."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

_DEPOSIT_SLOT_XY: Tuple[Tuple[float, float], ...] = (
    (-0.37, 0.08),
    (-0.54, 0.08),
    (-0.37, -0.10),
    (-0.54, -0.10),
)


def _demo_scenes_dir_candidates() -> List[str]:
    dirs: List[str] = []
    try:
        from ament_index_python.packages import get_package_share_directory

        share = get_package_share_directory("panda_controller")
        dirs.append(os.path.join(share, "config", "demo_scenes"))
    except Exception:
        pass
    here = Path(__file__).resolve()
    rel_suffix = ("panda_ws", "panda_controller", "config", "demo_scenes")
    for depth in range(2, min(6, len(here.parents))):
        root = here.parents[depth]
        for prefix in ((), ("src",)):
            cand = root.joinpath(*prefix, *rel_suffix)
            dirs.append(str(cand))
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
        path = os.path.join(base, "%s.yaml" % sid)
        if os.path.isfile(path):
            return path
    return None


def load_demo_scene_yaml(scene_id: str) -> Optional[Dict[str, Any]]:
    path = demo_scene_yaml_path(scene_id)
    if not path:
        return None
    with open(path, "r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    return raw if isinstance(raw, dict) else None


def load_initial_deposit_slots(scene_id: str) -> List[Dict[str, Any]]:
    """[{label, slot_index}, ...] desde initial_deposits del YAML."""
    raw = load_demo_scene_yaml(scene_id)
    if not raw:
        return []
    out: List[Dict[str, Any]] = []
    for item in raw.get("initial_deposits") or []:
        if not isinstance(item, dict):
            continue
        label = str(item.get("label") or "").strip().lower()
        try:
            slot_index = int(item.get("slot_index"))
        except (TypeError, ValueError):
            continue
        if not label or not 0 <= slot_index < 4:
            continue
        out.append({"label": label, "slot_index": slot_index})
    return out


def resolve_table_pick_order(scene_id: str) -> Tuple[str, ...]:
    raw = load_demo_scene_yaml(scene_id)
    if not raw:
        return ()
    return tuple(
        str(x).strip().lower()
        for x in (raw.get("pick_order") or [])
        if str(x).strip()
    )


def seed_slot_occupancy_from_scene(
    occupancy: Any,
    scene_id: str,
) -> int:
    """Marca slots ocupados según initial_deposits; devuelve cuántos."""
    count = 0
    for dep in load_initial_deposit_slots(scene_id):
        if occupancy.mark_occupied(int(dep["slot_index"]), str(dep["label"])):
            count += 1
    return count


def deposit_box_full_message() -> str:
    return (
        "La caja de depósito está llena: los cuatro cajones están ocupados. "
        "No puedo depositar otro objeto hasta que liberes un hueco."
    )
