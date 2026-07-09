"""Microdescenso extra post-cartesiano para mustard_bottle tall_object_topdown."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

MUSTARD_EXTRA_MICRO_DESCEND_CARTESIAN_FRACTION_THRESHOLD = 0.15
MUSTARD_EXTRA_MICRO_DESCEND_STEP_Z_TOLERANCE_M = 0.004
MUSTARD_GRASP_TCP_Z_TOLERANCE_M = 0.006


def mustard_micro_step_target_reached(
    actual_tcp_z: float,
    target_tcp_z: float,
    *,
    tolerance_m: float = MUSTARD_EXTRA_MICRO_DESCEND_STEP_Z_TOLERANCE_M,
) -> bool:
    return abs(float(actual_tcp_z) - float(target_tcp_z)) <= float(tolerance_m)


def resolve_mustard_nominal_grasp_tcp_z(
    candidate: Dict[str, Any],
    *,
    top_z_m: Optional[float] = None,
    min_required_depth_from_top_m: float = 0.034,
) -> Optional[float]:
    for key in (
        "_mustard_nominal_grasp_tcp_z_m",
        "_mustard_shortfall_requested_tcp_z",
    ):
        value = candidate.get(key)
        if value is not None:
            return float(value)
    recommended = candidate.get("recommended_grasp_depth_from_top_m")
    tz = top_z_m if top_z_m is not None else candidate.get("top_z_m")
    if recommended is not None and tz is not None:
        return float(tz) - float(recommended)
    if tz is not None:
        return float(tz) - float(min_required_depth_from_top_m)
    return None


def resolve_mustard_effective_min_required_depth_m(
    *,
    configured_min_m: float,
    recommended_depth_m: Optional[float] = None,
    floor_m: float = 0.028,
) -> float:
    effective = max(float(floor_m), float(configured_min_m))
    if recommended_depth_m is not None:
        effective = min(effective, float(recommended_depth_m))
    return float(effective)


def resolve_mustard_descend_shortfall_extra_m(
    *,
    requested_tcp_z: float,
    actual_tcp_z: float,
    max_extra_m: float,
    min_shortfall_to_trigger_m: float = 0.004,
) -> Tuple[float, str]:
    shortfall = float(actual_tcp_z) - float(requested_tcp_z)
    if shortfall + 1e-6 < float(min_shortfall_to_trigger_m):
        return 0.0, "within_tolerance"
    cap = max(0.0, float(max_extra_m))
    if cap <= 0.0:
        return 0.0, "max_extra_disabled"
    return min(shortfall, cap), "auto_shortfall_compensation"


def resolve_mustard_extra_micro_descend_apply_m(
    *,
    requested_extra_m: float,
    max_extra_m: float,
    palm_clearance_observed_m: Optional[float],
    min_bridge_clearance_after_m: float,
) -> Tuple[float, str]:
    req = max(0.0, float(requested_extra_m))
    cap = max(0.0, float(max_extra_m))
    if req <= 0.0 or cap <= 0.0:
        return 0.0, "disabled_or_zero_request"
    if palm_clearance_observed_m is None:
        return min(req, cap), "no_palm_bridge_constraint"
    allowed = float(palm_clearance_observed_m) - float(min_bridge_clearance_after_m)
    if allowed <= 1e-6:
        return 0.0, "palm_bridge_clearance_insufficient"
    return min(req, cap, allowed), "palm_bridge_limited"


def build_mustard_extra_micro_descend_steps(
    *,
    start_tcp_z: float,
    applied_extra_m: float,
    step_m: float,
) -> Tuple[float, ...]:
    total = max(0.0, float(applied_extra_m))
    step = max(1e-4, float(step_m))
    if total <= 1e-6:
        return tuple()
    steps = []
    remaining = total
    z = float(start_tcp_z)
    while remaining > 1e-6:
        dz = min(step, remaining)
        z -= dz
        steps.append(float(z))
        remaining -= dz
    return tuple(steps)


def extend_mustard_micro_descend_z_targets_to_grasp(
    *,
    start_tcp_z: float,
    nominal_grasp_tcp_z: float,
    step_m: float,
) -> Tuple[float, ...]:
    """Pasos cartesianos adicionales hasta el grasp nominal (sin límite palm-bridge)."""
    z = float(start_tcp_z)
    nominal = float(nominal_grasp_tcp_z)
    if z <= nominal + 1e-4:
        return tuple()
    step = max(1e-4, float(step_m))
    out = []
    while z - nominal > 1e-4:
        dz = min(step, z - nominal)
        z -= dz
        out.append(float(z))
    return tuple(out)


def build_mustard_micro_descend_z_targets(
    *,
    start_tcp_z: float,
    applied_extra_m: float,
    step_m: float,
    nominal_grasp_tcp_z: Optional[float] = None,
) -> Tuple[float, ...]:
    """Trayectoria Z: extra palm-limited + extensión opcional hasta grasp nominal."""
    initial = build_mustard_extra_micro_descend_steps(
        start_tcp_z=float(start_tcp_z),
        applied_extra_m=float(applied_extra_m),
        step_m=float(step_m),
    )
    if nominal_grasp_tcp_z is None:
        return initial
    extend_from = float(initial[-1]) if initial else float(start_tcp_z)
    extension = extend_mustard_micro_descend_z_targets_to_grasp(
        start_tcp_z=extend_from,
        nominal_grasp_tcp_z=float(nominal_grasp_tcp_z),
        step_m=float(step_m),
    )
    if not extension:
        return initial
    return initial + extension


def evaluate_mustard_post_descend_depth_verify(
    *,
    top_z_m: float,
    actual_tcp_z: float,
    min_required_depth_from_top_m: float,
    nominal_grasp_tcp_z: Optional[float] = None,
    grasp_tcp_z_tolerance_m: float = MUSTARD_GRASP_TCP_Z_TOLERANCE_M,
    depth_tolerance_m: float = 0.0,
) -> Tuple[bool, float]:
    depth = float(top_z_m) - float(actual_tcp_z)
    depth_tol = max(0.0, float(depth_tolerance_m))
    ok = depth + depth_tol + 1e-6 >= float(min_required_depth_from_top_m)
    if (
        not ok
        and nominal_grasp_tcp_z is not None
        and float(grasp_tcp_z_tolerance_m) > 0.0
    ):
        z_err = float(actual_tcp_z) - float(nominal_grasp_tcp_z)
        z_tol = float(grasp_tcp_z_tolerance_m)
        depth_floor = float(min_required_depth_from_top_m) - z_tol
        # TCP at or below nominal grasp Z: collision top vs mesh top may differ slightly.
        if z_err <= z_tol + 1e-6 and depth + depth_tol + 1e-6 >= depth_floor:
            ok = True
    return bool(ok), float(depth)


def format_mustard_extra_micro_descend_log(fields: Dict[str, Any]) -> str:
    lines = ["[MUSTARD_EXTRA_MICRO_DESCEND]"]
    for key in (
        "requested_extra_m",
        "applied_extra_m",
        "before_tcp_z",
        "after_tcp_z",
        "top_z",
        "depth_before",
        "depth_after",
        "palm_clearance_before",
        "palm_clearance_after_estimated",
        "result",
        "reason",
    ):
        if key in fields:
            lines.append("%s=%s" % (key, fields[key]))
    return "\n".join(lines)


def format_mustard_post_descend_depth_verify_log(fields: Dict[str, Any]) -> str:
    lines = ["[MUSTARD_POST_DESCEND_DEPTH_VERIFY]"]
    for key in (
        "actual_depth_from_top",
        "min_required_depth_from_top",
        "actual_tcp_z",
        "top_z",
        "result",
        "reason",
    ):
        if key in fields:
            lines.append("%s=%s" % (key, fields[key]))
    return "\n".join(lines)
