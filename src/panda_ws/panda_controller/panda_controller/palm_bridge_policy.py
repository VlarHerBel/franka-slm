"""Política conservadora de clearance vertical del puente de palma (tall_object_topdown)."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

_SOURCE_PRIORITY = {
    "mustard_mesh_local_cap_center_z": 50,
    "runtime_gt_top_face_center_base_z": 40,
    "top_surface_center_base_z": 35,
    "chosen_target_center_base_z": 30,
    "expected_top_z_table_plus_height": 25,
    "collision_top_z": 20,
    "top_z_m_payload": 10,
}


def _to_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(v):
        return None
    return v


def _z_from_xyz_field(candidate: Dict[str, Any], key: str) -> Optional[float]:
    raw = candidate.get(key)
    if isinstance(raw, (list, tuple)) and len(raw) >= 3:
        return _to_float(raw[2])
    return None


def resolve_effective_top_z_for_palm_bridge(
    candidate: Dict[str, Any],
    raw_top_z: float,
    *,
    table_z_m: float,
) -> Tuple[float, str, Dict[str, Any]]:
    """Máximo Z fiable entre fuentes operativas para clearance del palm bridge."""
    raw_top_z = float(raw_top_z)
    entries: List[Tuple[float, str]] = [(raw_top_z, "top_z_m_payload")]

    height_m = _to_float(candidate.get("height_m"))
    if height_m is None:
        height_m = _to_float(candidate.get("object_height_m"))
    if height_m is None:
        height_m = _to_float(candidate.get("db_height_m"))
    if height_m is None:
        dims = candidate.get("dims_lwh")
        if isinstance(dims, (list, tuple)) and len(dims) >= 3:
            height_m = _to_float(dims[2])
    expected_top: Optional[float] = None
    if height_m is not None:
        expected_top = float(table_z_m) + float(height_m)
        entries.append((expected_top, "expected_top_z_table_plus_height"))

    label_l = str(candidate.get("label", "")).strip().lower()
    mesh_z = _z_from_xyz_field(candidate, "mustard_mesh_local_cap_center_base")
    if mesh_z is not None and label_l != "mustard_bottle":
        entries.append((float(mesh_z), "mustard_mesh_local_cap_center_z"))

    collision_top = _to_float(candidate.get("top_z_m"))
    if collision_top is None:
        collision_top = _to_float(candidate.get("top_z_estimated"))
    if collision_top is not None:
        entries.append((float(collision_top), "collision_top_z"))

    cap_center_keys = (
        ("gt_top_face_center_base", "runtime_gt_top_face_center_base_z"),
        ("top_surface_center_base", "top_surface_center_base_z"),
        ("chosen_target_center_base", "chosen_target_center_base_z"),
    )
    for key, src in cap_center_keys:
        z = _z_from_xyz_field(candidate, key)
        if z is None:
            continue
        if label_l == "mustard_bottle" and float(z) < 0.455:
            continue
        entries.append((float(z), src))

    max_z = max(z for z, _ in entries)

    def _rank(item: Tuple[float, str]) -> Tuple[float, int]:
        z, src = item
        return (float(z), int(_SOURCE_PRIORITY.get(src, 0)))

    selected_src = max(
        [item for item in entries if abs(item[0] - max_z) <= 1e-6],
        key=_rank,
    )[1]

    meta = {
        "raw_top_z_m": raw_top_z,
        "expected_top_z_m": expected_top,
        "mesh_local_cap_center_z": mesh_z,
        "collision_top_z": collision_top,
        "candidates": {src: z for z, src in entries},
    }
    return float(max_z), selected_src, meta


def compute_palm_bridge_grasp_tcp_z(
    effective_top_z: float,
    *,
    clearance_m: float,
    palm_bridge_below_panda_hand_m: float,
    panda_hand_to_grasp_tcp_z_m: float,
) -> Tuple[float, float, float]:
    """Devuelve (grasp_tcp_z, target_panda_hand_z, desired_bridge_z)."""
    desired_bridge_z = float(effective_top_z) + float(clearance_m)
    target_panda_hand_z = desired_bridge_z + float(palm_bridge_below_panda_hand_m)
    grasp_tcp_z = target_panda_hand_z - float(panda_hand_to_grasp_tcp_z_m)
    return float(grasp_tcp_z), float(target_panda_hand_z), float(desired_bridge_z)
