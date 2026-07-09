"""Fallback descend por IK/FK guardado para sugar_box demo_scene_02 (sin ROS)."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

SUGAR_BOX_GUARDED_IK_SCENE_ID = "demo_scene_02"
SUGAR_BOX_GUARDED_IK_Z_STEP_M = 0.010
SUGAR_BOX_GUARDED_IK_XY_TOLERANCE_M = 0.002
SUGAR_BOX_GUARDED_IK_Z_TOLERANCE_M = 0.003
SUGAR_BOX_GUARDED_IK_ORIENTATION_TOLERANCE_DEG = 5.0
SUGAR_BOX_GUARDED_IK_FINAL_Z_EXTRA_CANDIDATES_M = (0.004, 0.006)


def sugar_box_guarded_ik_descend_eligible(
    *,
    label: str,
    scene_id: str,
    multiobject_safe_route: bool,
) -> bool:
    return (
        str(label).strip().lower() == "sugar_box"
        and str(scene_id).strip() == SUGAR_BOX_GUARDED_IK_SCENE_ID
        and bool(multiobject_safe_route)
    )


def build_sugar_box_guarded_ik_z_waypoints(
    current_tcp_z: float,
    final_tcp_z: float,
    *,
    step_m: float = SUGAR_BOX_GUARDED_IK_Z_STEP_M,
) -> List[float]:
    start_z = float(current_tcp_z)
    end_z = float(final_tcp_z)
    if start_z <= end_z + 1e-6:
        return [end_z]
    step = max(float(step_m), 1e-4)
    out: List[float] = []
    z = start_z
    while z - step > end_z + 1e-6:
        z -= step
        out.append(float(z))
    if not out or abs(out[-1] - end_z) > 1e-6:
        out.append(end_z)
    return out


def build_sugar_box_guarded_ik_final_z_candidates(
    nominal_grasp_tcp_z: float,
    *,
    extra_depths_m: Sequence[float] = SUGAR_BOX_GUARDED_IK_FINAL_Z_EXTRA_CANDIDATES_M,
) -> List[float]:
    """Orden: nominal primero; luego más profundo (Z menor)."""
    base = float(nominal_grasp_tcp_z)
    out: List[float] = [base]
    for depth in extra_depths_m:
        z = base - float(depth)
        if z not in out and all(abs(z - v) > 1e-6 for v in out):
            out.append(z)
    return out


def build_sugar_box_guarded_ik_hand_goal_from_tcp_delta(
    current_hand: Tuple[float, float, float],
    current_tcp: Tuple[float, float, float],
    target_tcp: Tuple[float, float, float],
) -> Tuple[float, float, float]:
    """Convierte objetivo TCP a panda_hand manteniendo offset hand-tcp del estado actual."""
    return (
        float(current_hand[0]) + (float(target_tcp[0]) - float(current_tcp[0])),
        float(current_hand[1]) + (float(target_tcp[1]) - float(current_tcp[1])),
        float(current_hand[2]) + (float(target_tcp[2]) - float(current_tcp[2])),
    )


def sugar_box_guarded_ik_tcp_z_used_as_hand_z(
    *,
    current_hand_z: float,
    current_tcp_z: float,
    target_hand_z: float,
    target_tcp_z: float,
    min_hand_tcp_offset_m: float = 0.030,
) -> bool:
    """True si target_hand_z ≈ target_tcp_z ignorando el offset hand-tcp conocido."""
    hand_minus_tcp = float(current_hand_z) - float(current_tcp_z)
    if abs(hand_minus_tcp) + 1e-9 < float(min_hand_tcp_offset_m):
        return False
    return abs(float(target_hand_z) - float(target_tcp_z)) < 1e-3


def evaluate_sugar_box_guarded_ik_fk_step(
    *,
    fk_tcp: Tuple[float, float, float],
    target_tcp_z: float,
    reference_tcp_xy: Tuple[float, float],
    orientation_error_deg: float,
    xy_tolerance_m: float = SUGAR_BOX_GUARDED_IK_XY_TOLERANCE_M,
    z_tolerance_m: float = SUGAR_BOX_GUARDED_IK_Z_TOLERANCE_M,
    orientation_tolerance_deg: float = SUGAR_BOX_GUARDED_IK_ORIENTATION_TOLERANCE_DEG,
) -> Tuple[bool, float, float, float]:
    xy_err = math.hypot(
        float(fk_tcp[0]) - float(reference_tcp_xy[0]),
        float(fk_tcp[1]) - float(reference_tcp_xy[1]),
    )
    z_err = abs(float(fk_tcp[2]) - float(target_tcp_z))
    ok = (
        xy_err + 1e-9 <= float(xy_tolerance_m)
        and z_err + 1e-9 <= float(z_tolerance_m)
        and float(orientation_error_deg) + 1e-9
        <= float(orientation_tolerance_deg)
    )
    return bool(ok), float(xy_err), float(z_err), float(orientation_error_deg)


def format_sugar_box_guarded_ik_descend_start_log(fields: Dict[str, Any]) -> str:
    return (
        "[SUGAR_BOX_GUARDED_IK_DESCEND_START]\n"
        "current_tcp_z=%.4f\n"
        "target_tcp_z=%.4f\n"
        "cartesian_fraction_failed=true\n"
        "reason=cartesian_path_zero_fraction"
        % (
            float(fields.get("current_tcp_z", 0.0)),
            float(fields.get("target_tcp_z", 0.0)),
        )
    )


def format_sugar_box_guarded_ik_descend_step_target_log(
    fields: Dict[str, Any],
) -> str:
    return (
        "[SUGAR_BOX_GUARDED_IK_DESCEND_STEP_TARGET]\n"
        "moveit_target_link=%s\n"
        "current_tcp_z=%.4f\n"
        "current_hand_z=%.4f\n"
        "target_tcp_z=%.4f\n"
        "target_hand_z=%.4f\n"
        "hand_minus_tcp_z=%.4f\n"
        "quat_source=%s"
        % (
            str(fields.get("moveit_target_link", "panda_hand")),
            float(fields.get("current_tcp_z", 0.0)),
            float(fields.get("current_hand_z", 0.0)),
            float(fields.get("target_tcp_z", 0.0)),
            float(fields.get("target_hand_z", 0.0)),
            float(fields.get("hand_minus_tcp_z", 0.0)),
            str(fields.get("quat_source", "current_tf_after_joint7")),
        )
    )


def format_sugar_box_guarded_ik_frame_bug_log() -> str:
    return (
        "[SUGAR_BOX_GUARDED_IK_FRAME_BUG]\n"
        "reason=tcp_z_used_as_hand_z\n"
        "result=FAIL_INTERNAL"
    )


def format_sugar_box_guarded_ik_descend_step_log(fields: Dict[str, Any]) -> str:
    fk = fields.get("fk_tcp")
    fk_str = (
        "n/a"
        if not isinstance(fk, (list, tuple)) or len(fk) < 3
        else "(%.4f, %.4f, %.4f)" % (float(fk[0]), float(fk[1]), float(fk[2]))
    )
    return (
        "[SUGAR_BOX_GUARDED_IK_DESCEND_STEP]\n"
        "step=%s\n"
        "target_tcp_z=%.4f\n"
        "ik_ok=%s\n"
        "fk_tcp=%s\n"
        "xy_error=%s\n"
        "z_error=%s\n"
        "orientation_error_deg=%s\n"
        "result=%s"
        % (
            str(fields.get("step", "n/a")),
            float(fields.get("target_tcp_z", 0.0)),
            str(bool(fields.get("ik_ok", False))).lower(),
            fk_str,
            "n/a"
            if fields.get("xy_error") is None
            else "%.4f" % float(fields.get("xy_error")),
            "n/a"
            if fields.get("z_error") is None
            else "%.4f" % float(fields.get("z_error")),
            "n/a"
            if fields.get("orientation_error_deg") is None
            else "%.2f" % float(fields.get("orientation_error_deg")),
            str(fields.get("result", "FAIL")),
        )
    )


def format_sugar_box_guarded_ik_descend_execute_log(fields: Dict[str, Any]) -> str:
    return (
        "[SUGAR_BOX_GUARDED_IK_DESCEND_EXECUTE]\n"
        "steps=%d\n"
        "target_tcp_z=%.4f\n"
        "result=%s"
        % (
            int(fields.get("steps", 0)),
            float(fields.get("target_tcp_z", 0.0)),
            str(fields.get("result", "FAIL")),
        )
    )


def format_sugar_box_guarded_ik_descend_verify_log(fields: Dict[str, Any]) -> str:
    return (
        "[SUGAR_BOX_GUARDED_IK_DESCEND_VERIFY]\n"
        "actual_tcp_z=%.4f\n"
        "target_tcp_z=%.4f\n"
        "depth_below_top=%s\n"
        "result=%s"
        % (
            float(fields.get("actual_tcp_z", 0.0)),
            float(fields.get("target_tcp_z", 0.0)),
            "n/a"
            if fields.get("depth_below_top") is None
            else "%.4f" % float(fields.get("depth_below_top")),
            str(fields.get("result", "FAIL")),
        )
    )


def format_sugar_box_guarded_ik_fail_diag_log(fields: Dict[str, Any]) -> str:
    obstacles = fields.get("obstacles") or []
    obs_s = ",".join(str(o) for o in obstacles) if obstacles else "none"
    near = fields.get("joint_limits_near") or []
    near_s = ",".join(str(n) for n in near) if near else "none"
    return (
        "[SUGAR_BOX_GUARDED_IK_FAIL_DIAG]\n"
        "step=%s\n"
        "target_tcp_z=%.4f\n"
        "target_hand_z=%.4f\n"
        "seed_source=%s\n"
        "seed_joint7=%s\n"
        "current_joint7=%s\n"
        "ik_link_name=%s\n"
        "orientation_tolerance_deg=%.2f\n"
        "position_tolerance_m=%.4f\n"
        "target_collision_present=%s\n"
        "obstacles=%s\n"
        "joint_limits_near=%s\n"
        "joint_dist_pregrasp_to_step=%s\n"
        "result=NO_IK_SOLUTION"
        % (
            str(fields.get("step", "1")),
            float(fields.get("target_tcp_z", 0.0)),
            float(fields.get("target_hand_z", 0.0)),
            str(fields.get("seed_source", "unknown")),
            "n/a"
            if fields.get("seed_joint7") is None
            else "%.4f" % float(fields.get("seed_joint7")),
            "n/a"
            if fields.get("current_joint7") is None
            else "%.4f" % float(fields.get("current_joint7")),
            str(fields.get("ik_link_name", "panda_hand")),
            float(fields.get("orientation_tolerance_deg", 5.0)),
            float(fields.get("position_tolerance_m", 0.001)),
            str(bool(fields.get("target_collision_present", False))).lower(),
            obs_s,
            near_s,
            "n/a"
            if fields.get("joint_dist_pregrasp_to_step") is None
            else "%.4f" % float(fields.get("joint_dist_pregrasp_to_step")),
        )
    )
