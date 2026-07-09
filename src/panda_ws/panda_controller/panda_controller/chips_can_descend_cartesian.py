"""Política de descenso cartesiano puro en Z para chips_can (sin ROS)."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple


def chips_can_skip_gripper_centering_verify(candidate: Dict[str, Any]) -> bool:
    """Cilindro yaw-free: el centrado GT en eje del objeto es autoritativo."""
    if str(candidate.get("label", "")).strip().lower() != "chips_can":
        return False
    return bool(
        candidate.get("_chips_can_gt_centering_verified_at_pregrasp")
        or candidate.get("_chips_can_gripper_centering_ok_at_pregrasp")
    )


def chips_can_xy_drift_m(
    lock_xy: Sequence[float],
    actual_xy: Sequence[float],
) -> float:
    return float(
        math.hypot(
            float(actual_xy[0]) - float(lock_xy[0]),
            float(actual_xy[1]) - float(lock_xy[1]),
        )
    )


def chips_can_xy_drift_ok(drift_xy_m: float, max_allowed_xy_drift_m: float) -> bool:
    return float(drift_xy_m) <= float(max_allowed_xy_drift_m) + 1e-9


def chips_can_descend_use_cartesian_effective(
    *,
    param_value: bool,
    demo_fast_mode: bool,
    demo_motion_profile_active: bool,
) -> bool:
    if bool(demo_fast_mode) or bool(demo_motion_profile_active):
        return True
    return bool(param_value)


def chips_can_pre_descend_tcp_z_ok(
    actual_tcp_z: float,
    top_z_m: float,
    min_clearance_above_top_m: float,
) -> bool:
    return float(actual_tcp_z) + 1e-6 >= float(top_z_m) + float(
        min_clearance_above_top_m
    )


def chips_can_pre_descend_centering_ok(
    centering_error_xy_m: Optional[float],
    max_error_xy_m: float,
) -> bool:
    if centering_error_xy_m is None:
        return False
    return float(centering_error_xy_m) <= float(max_error_xy_m) + 1e-9


def build_chips_can_descend_z_waypoints(
    start_z: float,
    end_z: float,
    *,
    max_step_m: float,
    min_step_m: float = 0.010,
) -> List[float]:
    """Waypoints TCP-Z monótonos descendente de start_z hasta end_z (inclusive)."""
    z0 = float(start_z)
    z1 = float(end_z)
    if z0 <= z1 + 1e-6:
        return [z1]
    max_step = max(float(min_step_m), float(max_step_m))
    total_drop = z0 - z1
    n_steps = max(1, int(math.ceil(total_drop / max_step)))
    step = total_drop / float(n_steps)
    raw: List[float] = []
    z = z0
    for _ in range(n_steps):
        z -= step
        if z < z1:
            z = z1
        raw.append(round(z, 4))
    if not raw or abs(raw[-1] - z1) > 1e-4:
        if raw and raw[-1] < z1 - 1e-4:
            raw.append(round(z1, 4))
        elif not raw:
            raw = [round(z1, 4)]
        else:
            raw[-1] = round(z1, 4)
    out: List[float] = []
    for w in raw:
        if not out or abs(w - out[-1]) > 1e-4:
            out.append(w)
    return out


def subdivide_chips_can_descend_z_range(
    start_z: float,
    end_z: float,
    *,
    max_step_m: float,
    min_step_m: float = 0.010,
) -> List[float]:
    """Subdivide un tramo fallido en pasos <= max_step_m (mínimo 2 si hay recorrido)."""
    waypoints = build_chips_can_descend_z_waypoints(
        start_z, end_z, max_step_m=max_step_m, min_step_m=min_step_m
    )
    if len(waypoints) >= 2:
        return waypoints
    drop = float(start_z) - float(end_z)
    if drop <= float(min_step_m) + 1e-6:
        return waypoints if waypoints else [float(end_z)]
    half = max(float(min_step_m), drop / 2.0)
    return build_chips_can_descend_z_waypoints(
        start_z, end_z, max_step_m=half, min_step_m=min_step_m
    )


def chips_can_tcp_in_final_descend_relax_zone(
    actual_tcp_z: float,
    top_z_m: float,
    relax_above_top_m: float,
) -> bool:
    """Zona final: TCP ya cerca del top del cilindro (dedos pueden entrar en volumen)."""
    return float(actual_tcp_z) <= float(top_z_m) + float(relax_above_top_m) + 1e-6


def chips_can_descend_step_in_contact_zone(
    current_tcp_z: float,
    target_tcp_z: float,
    top_z_m: float,
    contact_zone_above_top_m: float,
) -> bool:
    """Paso que entra o continúa en la zona de contacto con el cilindro."""
    contact_z = float(top_z_m) + float(contact_zone_above_top_m)
    return (
        min(float(current_tcp_z), float(target_tcp_z)) <= contact_z + 1e-6
    )


def chips_can_final_descend_avoid_collisions_effective(
    *,
    policy_enabled: bool,
    avoid_collisions_in_contact_zone: bool,
    current_tcp_z: float,
    target_tcp_z: float,
    top_z_m: float,
    contact_zone_above_top_m: float,
    contact_zone_latched: bool,
) -> Optional[bool]:
    """
    None: usar avoid_collisions global de MoveIt.
    bool: override explícito para computeCartesianPath en este micropaso.
    """
    if not bool(policy_enabled):
        return None
    in_zone = bool(contact_zone_latched) or chips_can_descend_step_in_contact_zone(
        current_tcp_z,
        target_tcp_z,
        top_z_m,
        contact_zone_above_top_m,
    )
    if in_zone:
        return bool(avoid_collisions_in_contact_zone)
    return None


def chips_can_plan_blocked_for_relax_retry(
    fail_reason: str,
    fraction: Optional[float],
    fraction_threshold: float,
) -> bool:
    reason = str(fail_reason or "").strip().lower()
    if reason in (
        "trajectory_empty",
        "trajectory_single_point",
        "cartesian_null_response",
    ):
        return True
    if fraction is not None:
        try:
            return float(fraction) + 1e-6 < float(fraction_threshold)
        except (TypeError, ValueError):
            pass
    return False


def chips_can_descend_tcp_above_table_floor(
    target_tcp_z: float,
    table_z_m: float,
    table_clearance_m: float,
) -> bool:
    floor_z = float(table_z_m) + float(table_clearance_m)
    return float(target_tcp_z) + 1e-6 >= floor_z


def chips_can_pre_descend_pose_gate_ok(
    *,
    actual_tcp_z: float,
    top_z_m: float,
    min_clearance_above_top_m: float,
    centering_error_xy_m: Optional[float],
    max_centering_error_xy_m: float,
) -> Tuple[bool, bool, bool]:
    z_ok = chips_can_pre_descend_tcp_z_ok(
        actual_tcp_z, top_z_m, min_clearance_above_top_m
    )
    center_ok = chips_can_pre_descend_centering_ok(
        centering_error_xy_m, max_centering_error_xy_m
    )
    return z_ok and center_ok, z_ok, center_ok
