"""Target XY para verificación/corrección de centrado de pinza."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from panda_controller.grasp_width_resolve import is_edge_grasp_policy


def _xy_from_xyz(xyz: Any) -> Optional[np.ndarray]:
    if not isinstance(xyz, (list, tuple)) or len(xyz) < 2:
        return None
    try:
        return np.array([float(xyz[0]), float(xyz[1])], dtype=np.float64)
    except (TypeError, ValueError):
        return None


def resolve_object_geometric_center_xy(candidate: Dict[str, Any]) -> Optional[np.ndarray]:
    """Centro geométrico del objeto (sin offset edge)."""
    for key in (
        "known_box_center_base",
        "model_box_center_base",
        "chosen_target_center_base",
        "grasp_center_base",
        "position",
    ):
        xy = _xy_from_xyz(candidate.get(key))
        if xy is not None:
            return xy
    return None


def resolve_gripper_centering_target_xy(
    candidate: Dict[str, Any],
    plan_targets: Optional[Dict[str, Any]] = None,
    commanded_hand_xy: Optional[Tuple[float, float]] = None,
) -> Tuple[Optional[np.ndarray], str]:
    """Devuelve (target_xy, source) para [GRIPPER_CENTERING_*]."""
    edge = is_edge_grasp_policy(candidate)

    if edge and commanded_hand_xy is not None:
        return (
            np.array(
                [float(commanded_hand_xy[0]), float(commanded_hand_xy[1])],
                dtype=np.float64,
            ),
            "commanded_pregrasp_hand_xy",
        )

    if edge:
        for key, source in (
            ("_runtime_pregrasp_tcp_xy", "edge_pregrasp_tcp"),
            ("_runtime_grasp_tcp_xy", "edge_grasp_tcp"),
            ("grasp_tcp", "grasp_tcp"),
            ("pregrasp_tcp", "pregrasp_tcp"),
        ):
            xy = _xy_from_xyz(candidate.get(key))
            if xy is not None:
                return xy, source

        if isinstance(plan_targets, dict):
            for tkey, source in (
                ("pregrasp_tcp", "plan_pregrasp_tcp"),
                ("grasp_tcp", "plan_grasp_tcp"),
            ):
                raw = plan_targets.get(tkey)
                if isinstance(raw, (list, tuple)) and len(raw) >= 2:
                    return (
                        np.array([float(raw[0]), float(raw[1])], dtype=np.float64),
                        source,
                    )

        gpos = candidate.get("grasp_position")
        if isinstance(gpos, (list, tuple)) and len(gpos) >= 2:
            return (
                np.array([float(gpos[0]), float(gpos[1])], dtype=np.float64),
                "grasp_position",
            )

        # grasp_center_base solo si no hay centro geométrico alternativo distinto
        gcb = _xy_from_xyz(candidate.get("grasp_center_base"))
        geom = resolve_object_geometric_center_xy(candidate)
        if gcb is not None and geom is not None:
            if float(np.linalg.norm(gcb - geom)) > 0.002:
                return gcb, "grasp_center_base_edge_offset"

    for key, source in (
        ("grasp_center_base", "grasp_center_base"),
        ("model_box_center_base", "model_box_center"),
        ("known_box_center_base", "known_box_center"),
        ("chosen_target_center_base", "chosen_target_center"),
        ("position", "position"),
    ):
        xy = _xy_from_xyz(candidate.get(key))
        if xy is not None:
            return xy, source

    return None, "unset"


def centering_error_xy_m(
    target_xy: np.ndarray, actual_xy: np.ndarray
) -> float:
    return float(np.linalg.norm(target_xy - actual_xy))
