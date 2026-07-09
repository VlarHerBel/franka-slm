"""Política acotada collision_off en descenso final demo_scene_02 + cracker_box."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

from panda_controller.attached_transport_phases import joint_limit_margin_min
from panda_controller.demo_cracker_box_cartesian_prevalidate import (
    demo_scene_02_cracker_box_policy_active,
    vertical_descend_volume_clear_of_obstacles,
    _obs_xy,
)
from panda_controller.target_collision_policy import object_collision_radius_xy

DEMO_COLLISION_OFF_FINAL_DESCEND_SOURCE = "demo_collision_off_final_descend"
DEMO_COLLISION_OFF_FINAL_DESCEND_STAGED_SOURCE = (
    "demo_collision_off_final_descend_staged"
)

DEFAULT_DEMO_CRACKER_MAX_COLLISION_OFF_DESCEND_M = 0.14
DEFAULT_COLLISION_OFF_FRACTION_MIN = 0.98
DEFAULT_COLLISION_OFF_MAX_XY_DEVIATION_M = 0.003
DEFAULT_COLLISION_OFF_MIN_OBSTACLE_CLEARANCE_M = 0.025
DEFAULT_GRIPPER_XY_RADIUS_M = 0.055
DEFAULT_MIN_JOINT_LIMIT_MARGIN_RAD = 0.02

REASON_COLLISION_ON_FRACTION_LOW_BUT_OFF_OK = (
    "collision_on_fraction_low_but_collision_off_ok"
)
REASON_STAGED_COLLISION_OFF_OK = "staged_collision_on_then_off_ok"

DEFAULT_STAGED_COLLISION_ON_EXTRA_Z = (
    0.4975,
    0.485,
    0.470,
    0.455,
    0.440,
)


def select_collision_on_until_z(
    *,
    pregrasp_tcp_z: float,
    grasp_tcp_z: float,
    max_collision_off_descend_m: float,
    z_candidates: Sequence[float],
    probe_collision_on_fraction: Any,
    fraction_min: float = DEFAULT_COLLISION_OFF_FRACTION_MIN,
) -> Optional[float]:
    """Elige el Z intermedio más bajo alcanzable con collision ON y off-segment acotado."""
    pre_z = float(pregrasp_tcp_z)
    gr_z = float(grasp_tcp_z)
    max_off = float(max_collision_off_descend_m)
    passing: List[float] = []
    for z in sorted(float(z) for z in z_candidates):
        if z <= gr_z + 1e-6 or z > pre_z + 1e-6:
            continue
        if z - gr_z > max_off + 1e-6:
            continue
        frac = probe_collision_on_fraction(float(z))
        if frac is not None and float(frac) + 1e-6 >= float(fraction_min):
            passing.append(float(z))
    if not passing:
        return None
    return float(min(passing))


def format_demo_final_descend_collision_off_policy_skip_log(
    fields: Dict[str, Any],
) -> str:
    return (
        "[DEMO_FINAL_DESCEND_COLLISION_OFF_POLICY_SKIP]\n"
        "reason=%s\n"
        "label=%s\n"
        "scene_id=%s\n"
        "grid_mode=%s\n"
        "candidate_idx=%s"
        % (
            fields.get("reason", ""),
            fields.get("label", "n/a"),
            fields.get("scene_id", "n/a"),
            fields.get("grid_mode", "n/a"),
            fields.get("candidate_idx", "n/a"),
        )
    )


def demo_cracker_collision_off_policy_enabled(
    *,
    param_value: Any,
    demo_authoritative_scene: bool,
    scene_id: str,
    label: str,
) -> bool:
    if not demo_scene_02_cracker_box_policy_active(
        label=label,
        demo_authoritative_scene=demo_authoritative_scene,
        scene_id=scene_id,
    ):
        return False
    if isinstance(param_value, bool):
        return bool(param_value)
    ps = str(param_value or "auto").strip().lower()
    if ps in ("false", "0", "no", "off", "disabled"):
        return False
    if ps in ("true", "1", "yes", "on", "enabled"):
        return True
    return True


def min_obstacle_lateral_clearance_m(
    pre_plan: Tuple[float, float, float],
    scene_obstacles: Sequence[Dict[str, Any]],
    *,
    gripper_xy_radius_m: float = DEFAULT_GRIPPER_XY_RADIUS_M,
) -> Tuple[float, str]:
    px, py = float(pre_plan[0]), float(pre_plan[1])
    min_clear = float("inf")
    nearest = "none"
    for obs in scene_obstacles or []:
        if not isinstance(obs, dict) or bool(obs.get("is_target", False)):
            continue
        oxy = _obs_xy(obs)
        if oxy is None:
            continue
        obs_r = float(object_collision_radius_xy(obs))
        dxy = math.hypot(px - oxy[0], py - oxy[1])
        clearance = dxy - obs_r - float(gripper_xy_radius_m)
        if clearance < min_clear:
            min_clear = float(clearance)
            nearest = str(obs.get("label", "unknown"))
    if not math.isfinite(min_clear):
        return float("inf"), "none"
    return float(min_clear), nearest


def compute_final_descend_safety_metrics(
    *,
    pre_plan: Tuple[float, float, float],
    gr_plan: Tuple[float, float, float],
    scene_obstacles: Sequence[Dict[str, Any]],
    table_z_m: float,
    joint_values_7: Optional[Sequence[float]],
    target_removed_ok: bool,
    min_table_clearance_m: float = 0.012,
    max_xy_deviation_m: float = DEFAULT_COLLISION_OFF_MAX_XY_DEVIATION_M,
    max_collision_off_descend_m: float = DEFAULT_DEMO_CRACKER_MAX_COLLISION_OFF_DESCEND_M,
    gripper_xy_radius_m: float = DEFAULT_GRIPPER_XY_RADIUS_M,
    min_lateral_clearance_m: float = DEFAULT_COLLISION_OFF_MIN_OBSTACLE_CLEARANCE_M,
    min_joint_limit_margin_rad: float = DEFAULT_MIN_JOINT_LIMIT_MARGIN_RAD,
    collision_off_start_plan: Optional[Tuple[float, float, float]] = None,
) -> Dict[str, Any]:
    off_start = (
        tuple(collision_off_start_plan)
        if collision_off_start_plan is not None
        else tuple(pre_plan)
    )
    max_xy_dev = math.hypot(
        float(off_start[0]) - float(gr_plan[0]),
        float(off_start[1]) - float(gr_plan[1]),
    )
    max_descend = max(0.0, float(off_start[2]) - float(gr_plan[2]))
    total_descend = max(0.0, float(pre_plan[2]) - float(gr_plan[2]))
    min_table_clr = float(gr_plan[2]) - float(table_z_m)
    min_obs_dist, nearest_obs = min_obstacle_lateral_clearance_m(
        off_start,
        scene_obstacles,
        gripper_xy_radius_m=gripper_xy_radius_m,
    )
    corridor_ok, corridor_reason = vertical_descend_volume_clear_of_obstacles(
        off_start,
        gr_plan,
        scene_obstacles,
        gripper_xy_radius_m=gripper_xy_radius_m,
        min_lateral_clearance_m=min_lateral_clearance_m,
    )
    j_margin = None
    joint_limit_ok = True
    if joint_values_7 is not None and len(joint_values_7) >= 7:
        j_margin = float(joint_limit_margin_min(joint_values_7))
        joint_limit_ok = float(j_margin) + 1e-6 >= float(min_joint_limit_margin_rad)
    obstacle_distance_ok = (
        float(min_obs_dist) + 1e-6 >= float(min_lateral_clearance_m)
        if math.isfinite(float(min_obs_dist))
        else True
    )
    return {
        "max_xy_deviation_m": float(max_xy_dev),
        "max_descend_m": float(max_descend),
        "total_descend_m": float(total_descend),
        "collision_off_start_z": float(off_start[2]),
        "min_table_clearance_m": float(min_table_clr),
        "min_obstacle_distance_m": float(min_obs_dist),
        "nearest_obstacle_label": nearest_obs,
        "corridor_clear_ok": bool(corridor_ok),
        "corridor_reason": str(corridor_reason),
        "joint_limit_margin_rad": j_margin,
        "target_removed_ok": bool(target_removed_ok),
        "table_clearance_ok": bool(min_table_clr + 1e-6 >= float(min_table_clearance_m)),
        "xy_deviation_ok": bool(max_xy_dev <= float(max_xy_deviation_m) + 1e-6),
        "max_descend_ok": bool(
            float(max_descend) <= float(max_collision_off_descend_m) + 1e-6
        ),
        "obstacle_clearance_ok": bool(corridor_ok and obstacle_distance_ok),
        "joint_limit_ok": bool(joint_limit_ok),
    }


def evaluate_collision_off_final_descend_allow(
    *,
    demo_authoritative_scene: bool,
    scene_id: str,
    label: str,
    pre_plan: Tuple[float, float, float],
    gr_plan: Tuple[float, float, float],
    scene_obstacles: Sequence[Dict[str, Any]],
    table_z_m: float,
    joint_values_7: Optional[Sequence[float]],
    target_removed_ok: bool,
    collision_on_fraction: float,
    collision_off_fraction: Optional[float],
    start_state_honored: Any,
    endpoint_ik_ok: bool,
    traj_pts: int,
    collision_on_fraction_threshold: float = 0.95,
    max_collision_off_descend_m: float = DEFAULT_DEMO_CRACKER_MAX_COLLISION_OFF_DESCEND_M,
    collision_off_fraction_min: float = DEFAULT_COLLISION_OFF_FRACTION_MIN,
    max_xy_deviation_m: float = DEFAULT_COLLISION_OFF_MAX_XY_DEVIATION_M,
    min_obstacle_clearance_m: float = DEFAULT_COLLISION_OFF_MIN_OBSTACLE_CLEARANCE_M,
    min_joint_limit_margin_rad: float = DEFAULT_MIN_JOINT_LIMIT_MARGIN_RAD,
    min_table_clearance_m: float = 0.012,
    collision_off_start_plan: Optional[Tuple[float, float, float]] = None,
) -> Tuple[bool, str, Dict[str, Any]]:
    metrics = compute_final_descend_safety_metrics(
        pre_plan=pre_plan,
        gr_plan=gr_plan,
        scene_obstacles=scene_obstacles,
        table_z_m=table_z_m,
        joint_values_7=joint_values_7,
        target_removed_ok=target_removed_ok,
        min_table_clearance_m=min_table_clearance_m,
        min_lateral_clearance_m=min_obstacle_clearance_m,
        max_xy_deviation_m=max_xy_deviation_m,
        max_collision_off_descend_m=max_collision_off_descend_m,
        min_joint_limit_margin_rad=min_joint_limit_margin_rad,
        collision_off_start_plan=collision_off_start_plan,
    )
    metrics["collision_on_fraction"] = float(collision_on_fraction)
    metrics["collision_off_fraction"] = collision_off_fraction
    metrics["target_removed"] = bool(target_removed_ok)
    metrics["endpoint_ik_diagnostic_ok"] = bool(endpoint_ik_ok)

    if not demo_scene_02_cracker_box_policy_active(
        label=label,
        demo_authoritative_scene=demo_authoritative_scene,
        scene_id=scene_id,
    ):
        return False, "not_demo_scene_02_cracker_box", metrics
    if not bool(target_removed_ok):
        return False, "target_collision_still_present", metrics
    if not bool(metrics.get("obstacle_clearance_ok")):
        return False, str(metrics.get("corridor_reason") or "obstacle_clearance_fail"), metrics
    if not bool(metrics.get("table_clearance_ok")):
        return False, "table_clearance_fail", metrics
    if not bool(metrics.get("xy_deviation_ok")):
        return False, "xy_deviation_exceeds_limit", metrics
    if float(metrics["max_descend_m"]) > float(max_collision_off_descend_m) + 1e-6:
        return False, "max_descend_exceeds_limit", metrics
    if not bool(metrics.get("joint_limit_ok", True)):
        return False, "joint_limit_margin_low", metrics
    if start_state_honored is not True:
        return False, "start_state_not_honored", metrics
    if collision_off_fraction is None:
        return False, "collision_off_fraction_missing", metrics
    if float(collision_off_fraction) + 1e-6 < float(collision_off_fraction_min):
        return False, "collision_off_fraction_low", metrics
    if int(traj_pts) < 2:
        return False, "collision_off_trajectory_too_short", metrics
    if float(collision_on_fraction) + 1e-6 >= float(collision_on_fraction_threshold):
        return False, "collision_on_not_false_negative", metrics
    return True, REASON_COLLISION_ON_FRACTION_LOW_BUT_OFF_OK, metrics


def evaluate_staged_collision_off_final_descend_allow(
    *,
    demo_authoritative_scene: bool,
    scene_id: str,
    label: str,
    pre_plan: Tuple[float, float, float],
    gr_plan: Tuple[float, float, float],
    collision_on_until_z: float,
    scene_obstacles: Sequence[Dict[str, Any]],
    table_z_m: float,
    joint_values_7: Optional[Sequence[float]],
    target_removed_ok: bool,
    collision_on_stage_fraction: float,
    collision_off_fraction: Optional[float],
    collision_on_fraction: float,
    start_state_honored: Any,
    endpoint_ik_ok: bool,
    traj_pts: int,
    collision_on_fraction_threshold: float = 0.95,
    max_collision_off_descend_m: float = DEFAULT_DEMO_CRACKER_MAX_COLLISION_OFF_DESCEND_M,
    collision_off_fraction_min: float = DEFAULT_COLLISION_OFF_FRACTION_MIN,
    max_xy_deviation_m: float = DEFAULT_COLLISION_OFF_MAX_XY_DEVIATION_M,
    min_obstacle_clearance_m: float = DEFAULT_COLLISION_OFF_MIN_OBSTACLE_CLEARANCE_M,
    min_joint_limit_margin_rad: float = DEFAULT_MIN_JOINT_LIMIT_MARGIN_RAD,
    min_table_clearance_m: float = 0.012,
) -> Tuple[bool, str, Dict[str, Any]]:
    off_start = (
        float(pre_plan[0]),
        float(pre_plan[1]),
        float(collision_on_until_z),
    )
    allow, reason, metrics = evaluate_collision_off_final_descend_allow(
        demo_authoritative_scene=demo_authoritative_scene,
        scene_id=scene_id,
        label=label,
        pre_plan=pre_plan,
        gr_plan=gr_plan,
        scene_obstacles=scene_obstacles,
        table_z_m=table_z_m,
        joint_values_7=joint_values_7,
        target_removed_ok=target_removed_ok,
        collision_on_fraction=float(collision_on_fraction),
        collision_off_fraction=collision_off_fraction,
        start_state_honored=start_state_honored,
        endpoint_ik_ok=endpoint_ik_ok,
        traj_pts=traj_pts,
        collision_on_fraction_threshold=collision_on_fraction_threshold,
        max_collision_off_descend_m=max_collision_off_descend_m,
        collision_off_fraction_min=collision_off_fraction_min,
        max_xy_deviation_m=max_xy_deviation_m,
        min_obstacle_clearance_m=min_obstacle_clearance_m,
        min_joint_limit_margin_rad=min_joint_limit_margin_rad,
        min_table_clearance_m=min_table_clearance_m,
        collision_off_start_plan=off_start,
    )
    metrics["staged_descend"] = True
    metrics["collision_on_until_z"] = float(collision_on_until_z)
    metrics["collision_on_descend_m"] = max(
        0.0, float(pre_plan[2]) - float(collision_on_until_z)
    )
    metrics["collision_off_descend_m"] = max(
        0.0, float(collision_on_until_z) - float(gr_plan[2])
    )
    if not allow:
        return False, reason, metrics
    if float(collision_on_stage_fraction) + 1e-6 < float(collision_off_fraction_min):
        return False, "collision_on_stage_fraction_low", metrics
    return True, REASON_STAGED_COLLISION_OFF_OK, metrics


def evaluate_demo_cracker_collision_off_final_descend(
    *,
    demo_authoritative_scene: bool,
    scene_id: str,
    label: str,
    pre_plan: Tuple[float, float, float],
    gr_plan: Tuple[float, float, float],
    scene_obstacles: Sequence[Dict[str, Any]],
    table_z_m: float,
    joint_values_7: Optional[Sequence[float]],
    target_removed_ok: bool,
    collision_on_fraction: float,
    collision_off_fraction: Optional[float],
    start_state_honored: Any,
    endpoint_ik_ok: bool,
    traj_pts: int,
    collision_on_fraction_threshold: float = 0.95,
    max_collision_off_descend_m: float = DEFAULT_DEMO_CRACKER_MAX_COLLISION_OFF_DESCEND_M,
    collision_off_fraction_min: float = DEFAULT_COLLISION_OFF_FRACTION_MIN,
    max_xy_deviation_m: float = DEFAULT_COLLISION_OFF_MAX_XY_DEVIATION_M,
    min_obstacle_clearance_m: float = DEFAULT_COLLISION_OFF_MIN_OBSTACLE_CLEARANCE_M,
    min_joint_limit_margin_rad: float = DEFAULT_MIN_JOINT_LIMIT_MARGIN_RAD,
    min_table_clearance_m: float = 0.012,
) -> Tuple[bool, str, Dict[str, Any]]:
    """Evalúa fallback collision_off tras fraction baja con collision_on."""
    return evaluate_collision_off_final_descend_allow(
        demo_authoritative_scene=demo_authoritative_scene,
        scene_id=scene_id,
        label=label,
        pre_plan=pre_plan,
        gr_plan=gr_plan,
        scene_obstacles=scene_obstacles,
        table_z_m=table_z_m,
        joint_values_7=joint_values_7,
        target_removed_ok=target_removed_ok,
        collision_on_fraction=collision_on_fraction,
        collision_off_fraction=collision_off_fraction,
        start_state_honored=start_state_honored,
        endpoint_ik_ok=endpoint_ik_ok,
        traj_pts=traj_pts,
        collision_on_fraction_threshold=collision_on_fraction_threshold,
        max_collision_off_descend_m=max_collision_off_descend_m,
        collision_off_fraction_min=collision_off_fraction_min,
        max_xy_deviation_m=max_xy_deviation_m,
        min_obstacle_clearance_m=min_obstacle_clearance_m,
        min_joint_limit_margin_rad=min_joint_limit_margin_rad,
        min_table_clearance_m=min_table_clearance_m,
    )


def format_demo_final_descend_staged_policy_log(fields: Dict[str, Any]) -> str:
    return (
        "[DEMO_FINAL_DESCEND_STAGED_POLICY]\n"
        "candidate_idx=%s\n"
        "pregrasp_tcp_z=%s\n"
        "collision_on_until_z=%s\n"
        "grasp_tcp_z=%s\n"
        "collision_on_descend_m=%s\n"
        "collision_off_descend_m=%s\n"
        "max_collision_off_descend_m=%s\n"
        "collision_on_stage_fraction=%s\n"
        "collision_off_fraction=%s\n"
        "result=%s"
        % (
            fields.get("candidate_idx", "n/a"),
            fields.get("pregrasp_tcp_z", "n/a"),
            fields.get("collision_on_until_z", "n/a"),
            fields.get("grasp_tcp_z", "n/a"),
            fields.get("collision_on_descend_m", "n/a"),
            fields.get("collision_off_descend_m", "n/a"),
            fields.get("max_collision_off_descend_m", "n/a"),
            fields.get("collision_on_stage_fraction", "n/a"),
            fields.get("collision_off_fraction", "n/a"),
            str(fields.get("result", "REJECT")),
        )
    )


def format_demo_final_descend_collision_on_execute_log(fields: Dict[str, Any]) -> str:
    return (
        "[DEMO_FINAL_DESCEND_COLLISION_ON_EXECUTE]\n"
        "candidate_idx=%s\n"
        "collision_on_until_z=%s\n"
        "avoid_collisions=true\n"
        "validated_preflight=%s\n"
        "result=%s"
        % (
            fields.get("candidate_idx", "n/a"),
            fields.get("collision_on_until_z", "n/a"),
            str(bool(fields.get("validated_preflight"))).lower(),
            str(fields.get("result", "FAIL")),
        )
    )


def format_demo_final_descend_collision_off_policy_log(fields: Dict[str, Any]) -> str:
    return (
        "[DEMO_FINAL_DESCEND_COLLISION_OFF_POLICY]\n"
        "label=%s\n"
        "scene_id=%s\n"
        "reason=%s\n"
        "collision_on_fraction=%s\n"
        "collision_off_fraction=%s\n"
        "target_removed=%s\n"
        "obstacle_clearance_ok=%s\n"
        "table_clearance_ok=%s\n"
        "xy_deviation_ok=%s\n"
        "max_descend_ok=%s\n"
        "result=%s"
        % (
            fields.get("label", "n/a"),
            fields.get("scene_id", "n/a"),
            fields.get("reason", ""),
            fields.get("collision_on_fraction", "n/a"),
            fields.get("collision_off_fraction", "n/a"),
            str(bool(fields.get("target_removed"))).lower(),
            str(bool(fields.get("obstacle_clearance_ok"))).lower(),
            str(bool(fields.get("table_clearance_ok"))).lower(),
            str(bool(fields.get("xy_deviation_ok"))).lower(),
            str(bool(fields.get("max_descend_ok"))).lower(),
            str(fields.get("result", "REJECT")),
        )
    )


def format_demo_final_descend_collision_off_execute_log(fields: Dict[str, Any]) -> str:
    return (
        "[DEMO_FINAL_DESCEND_COLLISION_OFF_EXECUTE]\n"
        "candidate_idx=%s\n"
        "avoid_collisions=false\n"
        "validated_preflight=%s\n"
        "result=%s"
        % (
            fields.get("candidate_idx", "n/a"),
            str(bool(fields.get("validated_preflight"))).lower(),
            str(fields.get("result", "FAIL")),
        )
    )


def format_demo_final_descend_safety_check_log(fields: Dict[str, Any]) -> str:
    return (
        "[DEMO_FINAL_DESCEND_SAFETY_CHECK]\n"
        "min_obstacle_distance=%s\n"
        "min_obstacle_distance_threshold=%s\n"
        "min_table_clearance=%s\n"
        "max_xy_deviation=%s\n"
        "target_removed_ok=%s\n"
        "target_collision_present_source=%s\n"
        "target_collision_present=%s\n"
        "obstacle_clearance_ok=%s\n"
        "table_clearance_ok=%s\n"
        "xy_deviation_ok=%s\n"
        "max_descend_ok=%s\n"
        "joint_limit_ok=%s\n"
        "collision_off_descend_m=%s\n"
        "result=%s"
        % (
            fields.get("min_obstacle_distance", "n/a"),
            fields.get("min_obstacle_distance_threshold", "n/a"),
            fields.get("min_table_clearance", "n/a"),
            fields.get("max_xy_deviation", "n/a"),
            str(bool(fields.get("target_removed_ok"))).lower(),
            fields.get("target_collision_present_source", "n/a"),
            str(bool(fields.get("target_collision_present"))).lower(),
            str(bool(fields.get("obstacle_clearance_ok"))).lower(),
            str(bool(fields.get("table_clearance_ok"))).lower(),
            str(bool(fields.get("xy_deviation_ok"))).lower(),
            str(bool(fields.get("max_descend_ok"))).lower(),
            str(bool(fields.get("joint_limit_ok", True))).lower(),
            fields.get("collision_off_descend_m", "n/a"),
            str(fields.get("result", "FAIL")),
        )
    )
