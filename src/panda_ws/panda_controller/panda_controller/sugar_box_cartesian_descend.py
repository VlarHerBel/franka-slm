"""Fallback y bloqueo de orientación para descenso cartesiano sugar_box (sin ROS)."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

SUGAR_BOX_DESCEND_MICROSTEP_DZ_M: Tuple[float, ...] = (
    0.010,
    0.020,
    0.030,
    0.040,
    0.055,
)

SUGAR_BOX_DESCEND_SEGMENT_STEP_M = 0.020


def sugar_box_cartesian_soft_fail_eligible(
    fraction: Optional[float],
    threshold: float,
    *,
    soft_min_fraction: float = 0.80,
) -> bool:
    """True si fraction está en [soft_min, threshold) — candidato a fallback."""
    if fraction is None:
        return False
    f = float(fraction)
    t = float(threshold)
    sm = float(soft_min_fraction)
    return sm - 1e-6 <= f < t - 1e-6


def sugar_box_fallback_grasp_tcp_z_proportional(
    pregrasp_tcp_z: float,
    nominal_grasp_tcp_z: float,
    cartesian_fraction: float,
    threshold: float,
    *,
    safety_scale: float = 0.98,
    min_descend_retention: float = 0.75,
) -> float:
    """Sube grasp_tcp_z para que el tramo cartesiano restante quepa en fraction observada."""
    drop = float(pregrasp_tcp_z) - float(nominal_grasp_tcp_z)
    if drop <= 1e-6:
        return float(nominal_grasp_tcp_z)
    ratio = max(
        float(min_descend_retention),
        float(cartesian_fraction) / max(float(threshold), 1e-6),
    )
    ratio = min(1.0, ratio * float(safety_scale))
    return float(pregrasp_tcp_z) - drop * ratio


def sugar_box_fallback_grasp_tcp_z_raise_steps(
    nominal_grasp_tcp_z: float,
    steps_m: Sequence[float],
) -> List[float]:
    """Candidatos de grasp_tcp_z más altos (descenso reducido)."""
    base = float(nominal_grasp_tcp_z)
    out: List[float] = []
    for step in steps_m:
        z = base + float(step)
        if z > base + 1e-6 and (not out or abs(z - out[-1]) > 1e-6):
            out.append(z)
    return out


def build_sugar_box_cartesian_fallback_grasp_z_candidates(
    pregrasp_tcp_z: float,
    nominal_grasp_tcp_z: float,
    cartesian_fraction: float,
    threshold: float,
    *,
    z_raise_steps_m: Sequence[float] = (0.005, 0.010, 0.015),
) -> List[float]:
    """Orden: proporcional a fraction, luego micro-elevaciones fijas."""
    candidates: List[float] = []
    prop = sugar_box_fallback_grasp_tcp_z_proportional(
        pregrasp_tcp_z,
        nominal_grasp_tcp_z,
        cartesian_fraction,
        threshold,
    )
    if prop > nominal_grasp_tcp_z + 1e-6:
        candidates.append(prop)
    for z in sugar_box_fallback_grasp_tcp_z_raise_steps(
        nominal_grasp_tcp_z, z_raise_steps_m
    ):
        if z not in candidates:
            candidates.append(z)
    return candidates


def sugar_box_cartesian_fallback_accept(
    fraction: Optional[float],
    threshold: float,
    *,
    traj_points: int = 0,
) -> bool:
    return (
        fraction is not None
        and float(fraction) + 1e-6 >= float(threshold)
        and int(traj_points) >= 2
    )


def apply_grasp_tcp_z_to_plan(
    grasp_plan: Tuple[float, float, float],
    grasp_tcp_z: float,
) -> Tuple[float, float, float]:
    return (float(grasp_plan[0]), float(grasp_plan[1]), float(grasp_tcp_z))


def build_sugar_box_locked_descend_target_tcp(
    current_tcp: Tuple[float, float, float],
    selected_grasp_tcp_z: float,
) -> Tuple[float, float, float]:
    """Objetivo TCP: XY real tras joint7, Z del grasp seleccionado en depth_search."""
    return (
        float(current_tcp[0]),
        float(current_tcp[1]),
        float(selected_grasp_tcp_z),
    )


def build_sugar_box_locked_descend_hand_goal(
    orientation_lock: dict,
    target_tcp_z: float,
    *,
    target_tcp_xy: Optional[Tuple[float, float]] = None,
) -> Tuple[float, float, float]:
    """Hand goal vertical con orientación bloqueada (offset TCP-hand del estado actual)."""
    hand_pos = orientation_lock["current_hand_position"]
    tcp_pos = orientation_lock.get("current_tcp_position")
    if not isinstance(tcp_pos, (list, tuple)) or len(tcp_pos) < 3:
        tcp_pos = hand_pos
    tx = float(target_tcp_xy[0]) if target_tcp_xy is not None else float(tcp_pos[0])
    ty = float(target_tcp_xy[1]) if target_tcp_xy is not None else float(tcp_pos[1])
    tz = float(target_tcp_z)
    return (
        float(hand_pos[0]) + (tx - float(tcp_pos[0])),
        float(hand_pos[1]) + (ty - float(tcp_pos[1])),
        float(hand_pos[2]) + (tz - float(tcp_pos[2])),
    )


def build_sugar_box_segment_descent_waypoints(
    current_tcp_z: float,
    grasp_tcp_z: float,
    *,
    step_m: float = SUGAR_BOX_DESCEND_SEGMENT_STEP_M,
) -> List[float]:
    """Waypoints Z intermedios (descenso por segmentos)."""
    start_z = float(current_tcp_z)
    end_z = float(grasp_tcp_z)
    if start_z <= end_z + 1e-6:
        return [end_z]
    waypoints: List[float] = []
    z = start_z
    step = max(float(step_m), 1e-4)
    while z - step > end_z + 1e-6:
        z -= step
        waypoints.append(float(z))
    if not waypoints or abs(waypoints[-1] - end_z) > 1e-6:
        waypoints.append(end_z)
    return waypoints


def format_sugar_box_descend_orientation_lock_log(fields: Dict[str, Any]) -> str:
    current_tcp = fields.get("current_tcp")
    target_tcp = fields.get("target_tcp")
    tcp_str = (
        "n/a"
        if not isinstance(current_tcp, (list, tuple)) or len(current_tcp) < 3
        else "(%.4f, %.4f, %.4f)"
        % (float(current_tcp[0]), float(current_tcp[1]), float(current_tcp[2]))
    )
    tgt_str = (
        "n/a"
        if not isinstance(target_tcp, (list, tuple)) or len(target_tcp) < 3
        else "(%.4f, %.4f, %.4f)"
        % (float(target_tcp[0]), float(target_tcp[1]), float(target_tcp[2]))
    )
    return (
        "[DESCEND_ORIENTATION_LOCK]\n"
        "label=sugar_box\n"
        "source=current_tf_after_joint7\n"
        "old_candidate_yaw=%.4f\n"
        "current_hand_yaw=%.4f\n"
        "orientation_delta_deg=%.2f\n"
        "current_tcp=%s\n"
        "target_tcp=%s\n"
        "selected_grasp_tcp_z=%.4f\n"
        "result=%s"
        % (
            float(fields.get("old_candidate_yaw_rad", 0.0)),
            float(fields.get("current_hand_yaw_rad", 0.0)),
            float(fields.get("orientation_delta_deg", 0.0)),
            tcp_str,
            tgt_str,
            float(fields.get("selected_grasp_tcp_z", 0.0)),
            str(fields.get("result", "OK")),
        )
    )


def format_sugar_box_descend_microstep_diag_log(fields: Dict[str, Any]) -> str:
    frac = fields.get("fraction")
    frac_s = "n/a" if frac is None else "%.5f" % float(frac)
    return (
        "[SUGAR_BOX_DESCEND_MICROSTEP_DIAG]\n"
        "dz=%.3f\n"
        "target_tcp_z=%.4f\n"
        "fraction=%s\n"
        "trajectory_points=%d\n"
        "result=%s"
        % (
            float(fields.get("dz", 0.0)),
            float(fields.get("target_tcp_z", 0.0)),
            frac_s,
            int(fields.get("trajectory_points", 0)),
            str(fields.get("result", "FAIL")),
        )
    )


def quaternion_yaw_delta_deg(
    quat_a: Tuple[float, float, float, float],
    quat_b: Tuple[float, float, float, float],
) -> float:
    """Delta angular aproximado entre dos quaternions (grados)."""
    ax, ay, az, aw = (float(quat_a[0]), float(quat_a[1]), float(quat_a[2]), float(quat_a[3]))
    bx, by, bz, bw = (float(quat_b[0]), float(quat_b[1]), float(quat_b[2]), float(quat_b[3]))
    dot = abs(ax * bx + ay * by + az * bz + aw * bw)
    dot = min(1.0, max(-1.0, dot))
    return float(math.degrees(2.0 * math.acos(dot)))
