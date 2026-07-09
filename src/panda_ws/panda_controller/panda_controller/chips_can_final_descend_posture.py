"""Postura IK y corte cinemático para descenso final chips_can (sin ROS)."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

CHIPS_CAN_POSTURE_SEARCH_IMPL_REV = "2026-06-07-posture-v2"

CHIPS_CAN_SHALLOW_FINAL_DESCEND_DEPTH_M = 0.005

CHIPS_CAN_HIGH_ROUTE_IK_SEED_NAMES: Tuple[str, ...] = (
    "pick_workspace_ready",
    "home",
    "transport_ready_unwind",
    "joint7_near_zero",
    "elbow_unwind_high",
    "previous_successful_chips_can_pose",
)

CHIPS_CAN_FAVORABLE_POSITION_PROBE_XY: Tuple[Tuple[float, float], ...] = (
    (0.48, -0.04),
    (0.48, 0.00),
)


def chips_can_shallow_grasp_tcp_z(*, top_z_m: float) -> float:
    return float(top_z_m) - float(CHIPS_CAN_SHALLOW_FINAL_DESCEND_DEPTH_M)


def estimate_cartesian_kinematic_cutoff_tcp_z(
    *,
    start_tcp_z: float,
    target_tcp_z: float,
    achieved_fraction: float,
) -> float:
    frac = max(0.0, min(1.0, float(achieved_fraction)))
    return float(start_tcp_z) + frac * (float(target_tcp_z) - float(start_tcp_z))


def infer_cartesian_kinematic_suspected_cause(
    *,
    achieved_fraction: float,
    with_obstacles_fraction: Optional[float] = None,
    without_obstacles_fraction: Optional[float] = None,
) -> str:
    frac = float(achieved_fraction)
    if with_obstacles_fraction is not None and without_obstacles_fraction is not None:
        if abs(float(with_obstacles_fraction) - float(without_obstacles_fraction)) > 0.05:
            return "obstacle_collision"
    if frac + 1e-6 >= 0.95:
        return "none"
    if frac + 1e-6 >= 0.45:
        return "ik_limit"
    if frac + 1e-6 >= 0.20:
        return "joint_limit"
    return "self_collision"


def format_joint_state_compact(js: Any) -> str:
    if js is None:
        return "n/a"
    try:
        pos = [float(v) for v in js.position][:7]
        return "[%s]" % ", ".join("%.4f" % v for v in pos)
    except (TypeError, ValueError, AttributeError):
        return "n/a"


def posture_variant_passes(
    probe: Dict[str, Any],
    *,
    fraction_threshold: float,
) -> bool:
    threshold = float(fraction_threshold)
    return bool(probe.get("full_route_ok")) and float(
        probe.get("low_to_grasp_fraction", 0.0)
    ) + 1e-6 >= threshold


def summarize_chips_can_posture_search(
    probes: Sequence[Dict[str, Any]],
    *,
    fraction_threshold: float,
) -> Dict[str, Any]:
    threshold = float(fraction_threshold)
    total = len(probes)
    passing = [p for p in probes if posture_variant_passes(p, fraction_threshold=threshold)]
    ok_count = len(passing)
    best_fraction = 0.0
    best_seed = ""
    best_yaw_deg = 0.0
    for probe in probes:
        frac = float(probe.get("low_to_grasp_fraction", 0.0))
        if frac + 1e-6 > best_fraction:
            best_fraction = frac
            best_seed = str(probe.get("ik_seed_name", ""))
            best_yaw_deg = math.degrees(float(probe.get("commanded_yaw_rad", 0.0)))
    return {
        "total_variants": total,
        "ok_variants": ok_count,
        "best_fraction": best_fraction,
        "best_seed": best_seed,
        "best_yaw_deg": best_yaw_deg,
        "result": "OK" if ok_count > 0 else "EXHAUSTED",
    }


def format_chips_can_posture_search_start_log(fields: Dict[str, Any]) -> str:
    seeds = fields.get("seeds", [])
    if isinstance(seeds, (list, tuple)):
        seeds_s = "[%s]" % ", ".join(str(s) for s in seeds)
    else:
        seeds_s = str(seeds)
    return (
        "[CHIPS_CAN_FINAL_DESCEND_POSTURE_SEARCH_START]\n"
        "impl_rev=%s\n"
        "seeds=%s\n"
        "yaw_candidates_count=%d\n"
        "depth_probe=%.4f"
        % (
            fields.get("impl_rev", CHIPS_CAN_POSTURE_SEARCH_IMPL_REV),
            seeds_s,
            int(fields.get("yaw_candidates_count", 0)),
            float(fields.get("depth_probe", CHIPS_CAN_SHALLOW_FINAL_DESCEND_DEPTH_M)),
        )
    )


def format_chips_can_posture_search_summary_log(fields: Dict[str, Any]) -> str:
    return (
        "[CHIPS_CAN_FINAL_DESCEND_POSTURE_SEARCH_SUMMARY]\n"
        "total_variants=%d\n"
        "ok_variants=%d\n"
        "best_fraction=%.5f\n"
        "best_seed=%s\n"
        "best_yaw_deg=%.2f\n"
        "result=%s"
        % (
            int(fields.get("total_variants", 0)),
            int(fields.get("ok_variants", 0)),
            float(fields.get("best_fraction", 0.0)),
            fields.get("best_seed", ""),
            float(fields.get("best_yaw_deg", 0.0)),
            fields.get("result", "EXHAUSTED"),
        )
    )


def format_chips_can_posture_search_exhausted_log(fields: Dict[str, Any]) -> str:
    return (
        "[CHIPS_CAN_POSTURE_SEARCH_EXHAUSTED]\n"
        "result=FAIL\n"
        "falling_back_to_pending_final_descend=%s\n"
        "reason=%s"
        % (
            str(bool(fields.get("falling_back_to_pending_final_descend", False))).lower(),
            fields.get("reason", "no_posture_variant_shallow_grasp_ok"),
        )
    )


def format_chips_can_posture_probe_cache_empty_log(fields: Dict[str, Any]) -> str:
    return (
        "[CHIPS_CAN_POSTURE_PROBE_CACHE_EMPTY]\n"
        "reason=%s"
        % fields.get("reason", "unknown")
    )


def format_chips_can_cartesian_kinematic_cutoff_log(fields: Dict[str, Any]) -> str:
    return (
        "[CHIPS_CAN_CARTESIAN_KINEMATIC_CUTOFF]\n"
        "start_tcp_z=%.4f\n"
        "target_tcp_z=%.4f\n"
        "achieved_fraction=%.5f\n"
        "estimated_cutoff_tcp_z=%.4f\n"
        "joint_state_at_start=%s\n"
        "suspected=%s"
        % (
            float(fields.get("start_tcp_z", 0.0)),
            float(fields.get("target_tcp_z", 0.0)),
            float(fields.get("achieved_fraction", 0.0)),
            float(fields.get("estimated_cutoff_tcp_z", 0.0)),
            fields.get("joint_state_at_start", "n/a"),
            fields.get("suspected", "unknown"),
        )
    )


def format_chips_can_final_descend_posture_variant_log(fields: Dict[str, Any]) -> str:
    shallow_frac = fields.get("low_to_shallow_grasp_fraction")
    if shallow_frac is None:
        shallow_frac = fields.get("cartesian_fraction")
    shallow_s = "n/a" if shallow_frac is None else "%.5f" % float(shallow_frac)
    hl_frac = fields.get("object_high_to_low_fraction")
    hl_s = "n/a" if hl_frac is None else "%.5f" % float(hl_frac)
    cutoff = fields.get("cutoff_tcp_z")
    cutoff_s = "n/a" if cutoff is None else "%.4f" % float(cutoff)
    return (
        "[CHIPS_CAN_FINAL_DESCEND_POSTURE_VARIANT]\n"
        "seed_name=%s\n"
        "yaw_deg=%.2f\n"
        "object_high_plan_ok=%s\n"
        "object_high_to_low_fraction=%s\n"
        "low_to_shallow_grasp_fraction=%s\n"
        "cutoff_tcp_z=%s\n"
        "result=%s\n"
        "reject_reason=%s"
        % (
            fields.get("seed_name", ""),
            float(fields.get("yaw_deg", 0.0)),
            str(bool(fields.get("object_high_plan_ok", False))).lower(),
            hl_s,
            shallow_s,
            cutoff_s,
            fields.get("result", "FAIL"),
            fields.get("reject_reason", ""),
        )
    )


def format_chips_can_final_descend_posture_selected_log(fields: Dict[str, Any]) -> str:
    return (
        "[CHIPS_CAN_FINAL_DESCEND_POSTURE_SELECTED]\n"
        "seed_name=%s\n"
        "yaw_deg=%.2f\n"
        "depth_from_top_m=%.4f\n"
        "cartesian_fraction=%.5f\n"
        "result=%s"
        % (
            fields.get("seed_name", ""),
            float(fields.get("yaw_deg", 0.0)),
            float(fields.get("depth_from_top_m", CHIPS_CAN_SHALLOW_FINAL_DESCEND_DEPTH_M)),
            float(fields.get("cartesian_fraction", 0.0)),
            fields.get("result", "OK"),
        )
    )


def select_chips_can_high_route_posture_variant(
    variants: Sequence[Dict[str, Any]],
    *,
    fraction_threshold: float,
) -> Optional[Dict[str, Any]]:
    """Elige postura con shallow final descend OK; preferir menor joint_dist."""
    threshold = float(fraction_threshold)
    passing: List[Dict[str, Any]] = []
    for item in variants:
        if posture_variant_passes(item, fraction_threshold=threshold):
            passing.append(item)
    if not passing:
        return None
    return min(passing, key=lambda item: float(item.get("joint_dist", math.inf)))
