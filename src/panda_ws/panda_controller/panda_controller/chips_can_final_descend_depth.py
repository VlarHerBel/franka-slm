"""Búsqueda de profundidad para descenso final chips_can (sin ROS)."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence

CHIPS_CAN_FINAL_DESCEND_DEPTH_FROM_TOP_M: tuple[float, ...] = (
    0.005,
    0.010,
    0.015,
    0.020,
    0.025,
    0.030,
    0.035,
)

MIN_CHIPS_CAN_HIGH_LOW_PREGRASP_DELTA_M = 0.04


def chips_can_grasp_tcp_z_from_depth(*, top_z_m: float, depth_from_top_m: float) -> float:
    return float(top_z_m) - float(depth_from_top_m)


def chips_can_high_route_contract_ok(
    *,
    object_high_tcp_z: float,
    low_pregrasp_tcp_z: float,
    min_delta_m: float = MIN_CHIPS_CAN_HIGH_LOW_PREGRASP_DELTA_M,
) -> bool:
    return float(object_high_tcp_z) >= float(low_pregrasp_tcp_z) + float(min_delta_m) - 1e-6


def chips_can_high_route_contract_delta(
    *,
    object_high_tcp_z: float,
    low_pregrasp_tcp_z: float,
) -> float:
    return float(object_high_tcp_z) - float(low_pregrasp_tcp_z)


def select_chips_can_final_descend_depth_variant(
    variants: Sequence[Dict[str, Any]],
    *,
    fraction_threshold: float,
) -> Optional[Dict[str, Any]]:
    """Elige la profundidad más profunda con cartesiano >= threshold."""
    threshold = float(fraction_threshold)
    passing: List[Dict[str, Any]] = []
    for item in variants:
        if float(item.get("cartesian_fraction", 0.0)) + 1e-6 < threshold:
            continue
        if not bool(item.get("ok", False)):
            continue
        passing.append(item)
    if not passing:
        return None
    return max(passing, key=lambda item: float(item.get("depth_from_top_m", 0.0)))


def infer_chips_can_final_descend_blocker(
    *,
    with_obstacles_fraction: float,
    without_remaining_obstacles_fraction: float,
    fraction_threshold: float,
) -> str:
    threshold = float(fraction_threshold)
    with_ok = float(with_obstacles_fraction) + 1e-6 >= threshold
    without_ok = float(without_remaining_obstacles_fraction) + 1e-6 >= threshold
    if not with_ok and without_ok:
        return "obstacle_collision"
    if not without_ok:
        if float(without_remaining_obstacles_fraction) + 1e-6 >= 0.45:
            return "kinematic_limit"
        return "table_or_hand_collision"
    return "unknown"


def format_chips_can_high_route_contract_check_log(fields: Dict[str, Any]) -> str:
    return (
        "[CHIPS_CAN_HIGH_ROUTE_CONTRACT_CHECK]\n"
        "object_high_tcp_z=%.4f\n"
        "low_pregrasp_tcp_z=%.4f\n"
        "delta=%.4f\n"
        "min_delta=%.4f\n"
        "result=%s\n"
        "reason=%s"
        % (
            float(fields.get("object_high_tcp_z", 0.0)),
            float(fields.get("low_pregrasp_tcp_z", 0.0)),
            float(fields.get("delta", 0.0)),
            float(fields.get("min_delta", MIN_CHIPS_CAN_HIGH_LOW_PREGRASP_DELTA_M)),
            fields.get("result", "FAIL"),
            fields.get("reason", ""),
        )
    )


def format_chips_can_final_descend_depth_variant_log(fields: Dict[str, Any]) -> str:
    frac = fields.get("cartesian_fraction")
    frac_s = "n/a" if frac is None else "%.5f" % float(frac)
    return (
        "[CHIPS_CAN_FINAL_DESCEND_DEPTH_VARIANT]\n"
        "depth_from_top_m=%.4f\n"
        "grasp_tcp_z=%.4f\n"
        "cartesian_fraction=%s\n"
        "result=%s\n"
        "reject_reason=%s"
        % (
            float(fields.get("depth_from_top_m", 0.0)),
            float(fields.get("grasp_tcp_z", 0.0)),
            frac_s,
            fields.get("result", "FAIL"),
            fields.get("reject_reason", ""),
        )
    )


def format_chips_can_final_descend_depth_selected_log(fields: Dict[str, Any]) -> str:
    return (
        "[CHIPS_CAN_FINAL_DESCEND_DEPTH_SELECTED]\n"
        "depth_from_top_m=%.4f\n"
        "grasp_tcp_z=%.4f\n"
        "cartesian_fraction=%.5f\n"
        "result=%s"
        % (
            float(fields.get("depth_from_top_m", 0.0)),
            float(fields.get("grasp_tcp_z", 0.0)),
            float(fields.get("cartesian_fraction", 0.0)),
            fields.get("result", "OK"),
        )
    )


def format_chips_can_final_descend_blocker_diag_log(fields: Dict[str, Any]) -> str:
    return (
        "[CHIPS_CAN_FINAL_DESCEND_BLOCKER_DIAG]\n"
        "with_obstacles_fraction=%.5f\n"
        "without_remaining_obstacles_fraction=%.5f\n"
        "suspected_blocker=%s\n"
        "nearest_obstacles=%s\n"
        "result=%s"
        % (
            float(fields.get("with_obstacles_fraction", 0.0)),
            float(fields.get("without_remaining_obstacles_fraction", 0.0)),
            fields.get("suspected_blocker", "unknown"),
            fields.get("nearest_obstacles", "[]"),
            fields.get("result", "FAIL"),
        )
    )
