"""Límites de descenso cartesiano y staging alto para objetos bajos / edge grasp."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

_DEMO_MOTION: Dict[str, Dict[str, float]] = {
    "cracker_box": {
        "recommended_grasp_depth_from_top_m": 0.033,
        "pregrasp_clearance_above_top_m": 0.080,
        "preferred_cartesian_descend_m": 0.080,
        "min_cartesian_descend_m": 0.060,
        "max_cartesian_descend_m": 0.100,
        "descend_velocity_scaling": 0.028,
        "descend_acceleration_scaling": 0.028,
        "insertion_depth_limit_m": 0.036,
        "object_high_clearance_above_top_m": 0.100,
        "min_tcp_clearance_above_table_m": 0.012,
    },
    "sugar_box": {
        "recommended_grasp_depth_from_top_m": 0.028,
        "pregrasp_clearance_above_top_m": 0.070,
        "preferred_cartesian_descend_m": 0.065,
        "min_cartesian_descend_m": 0.050,
        "max_cartesian_descend_m": 0.085,
        "descend_velocity_scaling": 0.022,
        "descend_acceleration_scaling": 0.022,
        "insertion_depth_limit_m": 0.032,
        "object_high_clearance_above_top_m": 0.120,
        "min_tcp_clearance_above_table_m": 0.012,
    },
    "pudding_box": {
        "recommended_grasp_depth_from_top_m": 0.008,
        "pregrasp_clearance_above_top_m": 0.055,
        "preferred_cartesian_descend_m": 0.060,
        "min_cartesian_descend_m": 0.045,
        "max_cartesian_descend_m": 0.085,
        "descend_velocity_scaling": 0.017,
        "descend_acceleration_scaling": 0.017,
        "insertion_depth_limit_m": 0.010,
        "object_high_clearance_above_top_m": 0.125,
        "min_tcp_clearance_above_table_m": 0.010,
    },
    "gelatin_box": {
        "recommended_grasp_depth_from_top_m": 0.008,
        "pregrasp_clearance_above_top_m": 0.055,
        "preferred_cartesian_descend_m": 0.060,
        "min_cartesian_descend_m": 0.045,
        "max_cartesian_descend_m": 0.085,
        "descend_velocity_scaling": 0.017,
        "descend_acceleration_scaling": 0.017,
        "insertion_depth_limit_m": 0.010,
        "object_high_clearance_above_top_m": 0.125,
        "min_tcp_clearance_above_table_m": 0.010,
    },
}

def motion_defaults_for_label(label: str) -> Dict[str, float]:
    key = str(label or "").strip().lower().replace(" ", "_").replace("-", "_")
    base = dict(_DEMO_MOTION.get(key, {}))
    return base


def _f(candidate: Dict[str, Any], key: str, default: Optional[float] = None) -> Optional[float]:
    if key in candidate and candidate[key] is not None:
        try:
            return float(candidate[key])
        except (TypeError, ValueError):
            pass
    return default


def should_use_low_object_high_approach(
    candidate: Dict[str, Any],
    top_z: float,
    *,
    enabled: bool = True,
) -> bool:
    if not enabled:
        return False
    if bool(candidate.get("force_object_high_stage", False)):
        return True
    # Mantener el pipeline común por defecto para todos los objetos.
    return False


def apply_descend_tcp_sequence(
    *,
    label: str,
    candidate: Dict[str, Any],
    top_z: float,
    grasp_xy: Tuple[float, float],
    min_grasp_z_from_table: float,
    max_target_z: float,
    eff_approach_m: float,
    eff_pregrasp_clear_m: float,
    eff_safe_above_m: float,
    eff_safe_extra_m: float,
    global_min_tcp_clearance_m: float,
    low_object_high_approach_enabled: bool = True,
    lift_clearance_m: float = 0.12,
) -> Dict[str, Any]:
    """Calcula grasp/pregrasp/safe/high TCP con límite de descenso cartesiano."""
    defaults = motion_defaults_for_label(label)
    depth = _f(candidate, "recommended_grasp_depth_from_top_m", defaults.get("recommended_grasp_depth_from_top_m", 0.035))
    insertion_lim = _f(
        candidate,
        "insertion_depth_limit_m",
        depth if depth is not None else defaults.get("insertion_depth_limit_m"),
    )
    max_descend = _f(
        candidate,
        "max_cartesian_descend_m",
        defaults.get("max_cartesian_descend_m", 0.070),
    )
    preferred_descend = _f(
        candidate,
        "preferred_cartesian_descend_m",
        defaults.get("preferred_cartesian_descend_m", max_descend),
    )
    min_descend = _f(
        candidate,
        "min_cartesian_descend_m",
        defaults.get("min_cartesian_descend_m", 0.040),
    )
    table_clr = _f(
        candidate,
        "min_tcp_clearance_above_table_m",
        global_min_tcp_clearance_m,
    )
    if depth is None:
        depth = 0.035
    if insertion_lim is None:
        insertion_lim = depth
    effective_depth = min(float(depth), float(insertion_lim))
    if max_descend is None:
        max_descend = 0.070
    if preferred_descend is None:
        preferred_descend = max_descend
    if min_descend is None:
        min_descend = 0.040
    preferred_descend = max(0.0, float(preferred_descend))
    min_descend = max(0.0, float(min_descend))
    max_descend = max(float(max_descend), min_descend)
    preferred_descend = min(max_descend, max(min_descend, preferred_descend))

    requested_grasp_z = float(top_z) - effective_depth
    min_z = float(min_grasp_z_from_table)
    if table_clr is not None:
        min_z = max(min_z, float(min_grasp_z_from_table) + float(table_clr) * 0.0)
    adjusted_grasp_z = max(requested_grasp_z, min_z)
    grasp_reason = "label_policy"
    if adjusted_grasp_z > requested_grasp_z + 1e-6:
        grasp_reason = "table_clearance"

    grasp_tcp = (float(grasp_xy[0]), float(grasp_xy[1]), float(adjusted_grasp_z))

    pregrasp_from_approach = grasp_tcp[2] + preferred_descend
    desired_above_top = float(top_z) + float(eff_pregrasp_clear_m)
    requested_pregrasp_z = max(pregrasp_from_approach, desired_above_top)
    requested_pregrasp_z = min(requested_pregrasp_z, float(max_target_z))

    requested_descend_m = requested_pregrasp_z - grasp_tcp[2]
    descend_reason = "within_limit"
    if requested_descend_m > float(max_descend) + 1e-6:
        pregrasp_z = grasp_tcp[2] + float(max_descend)
        final_descend_m = float(max_descend)
        descend_reason = "max_cartesian_descend_m"
    else:
        pregrasp_z = requested_pregrasp_z
        final_descend_m = requested_descend_m

    pregrasp_tcp = (grasp_tcp[0], grasp_tcp[1], float(pregrasp_z))

    safe_top_z = float(top_z) + float(eff_safe_above_m)
    safe_from_pre = pregrasp_tcp[2] + float(eff_safe_extra_m)
    safe_z = max(safe_top_z, safe_from_pre)
    safe_z = min(safe_z, float(max_target_z))
    safe_pregrasp_tcp = (grasp_tcp[0], grasp_tcp[1], float(safe_z))

    uses_high = should_use_low_object_high_approach(
        candidate, float(top_z), enabled=low_object_high_approach_enabled
    )
    high_clear = _f(
        candidate,
        "object_high_clearance_above_top_m",
        defaults.get("object_high_clearance_above_top_m", 0.120),
    )
    object_high_tcp: Optional[Tuple[float, float, float]] = None
    high_reason = ""
    if uses_high and high_clear is not None:
        high_z = min(float(top_z) + float(high_clear), float(max_target_z))
        object_high_tcp = (grasp_tcp[0], grasp_tcp[1], float(high_z))
        high_reason = "low_object_or_edge_grasp"

    return {
        "grasp_tcp": grasp_tcp,
        "pregrasp_tcp": pregrasp_tcp,
        "safe_pregrasp_tcp": safe_pregrasp_tcp,
        "lift_tcp": (grasp_tcp[0], grasp_tcp[1], grasp_tcp[2] + float(lift_clearance_m)),
        "object_high_pregrasp_tcp": object_high_tcp,
        "uses_low_object_high_approach_stage": uses_high,
        "low_object_high_reason": high_reason,
        "requested_grasp_tcp_z": requested_grasp_z,
        "adjusted_grasp_tcp_z": adjusted_grasp_z,
        "requested_descend_m": requested_descend_m,
        "desired_descend_m": preferred_descend,
        "min_cartesian_descend_m": min_descend,
        "final_descend_m": final_descend_m,
        "max_cartesian_descend_m": float(max_descend),
        "effective_depth_m": effective_depth,
        "descend_limit_reason": descend_reason,
        "grasp_z_reason": grasp_reason,
        "top_z": float(top_z),
    }


def descend_speed_from_candidate(
    candidate: Dict[str, Any],
    global_vel: float,
    global_acc: float,
) -> Tuple[float, float, str]:
    label = str(candidate.get("label", ""))
    defaults = motion_defaults_for_label(label)
    v = _f(candidate, "descend_velocity_scaling", defaults.get("descend_velocity_scaling"))
    a = _f(candidate, "descend_acceleration_scaling", defaults.get("descend_acceleration_scaling"))
    source = "object_policy"
    if v is None:
        v = float(global_vel)
        a = float(global_acc)
        source = "global_default"
    else:
        if a is None:
            a = float(global_acc)
    return float(v), float(a), source
