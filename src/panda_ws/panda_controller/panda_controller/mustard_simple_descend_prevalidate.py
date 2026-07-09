"""Prevalidación descenso cartesiano mustard_bottle simple_direct (sin ROS)."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

MUSTARD_DEPTH_FROM_TOP_M: Tuple[float, ...] = (0.010, 0.015, 0.020, 0.040, 0.046)
MUSTARD_DESCEND_DELTA_M: Tuple[float, ...] = (0.030, 0.040, 0.050, 0.060, 0.075)
MUSTARD_STAGED_DESCEND_ALPHAS: Tuple[float, ...] = (0.33, 0.66, 1.0)
MUSTARD_CARTESIAN_STEP_DIAG_ALPHAS: Tuple[float, ...] = (
    0.0,
    0.16667,
    0.33333,
    0.5,
    0.66667,
    0.83333,
    1.0,
)
MUSTARD_STAGED_CARTESIAN_SOURCE = "mustard_staged_cartesian"


def format_xyz_tuple(pos: Tuple[float, float, float]) -> str:
    return "(%.4f, %.4f, %.4f)" % (float(pos[0]), float(pos[1]), float(pos[2]))


def _fmt_xyz(pos: Optional[Tuple[float, float, float]]) -> str:
    if pos is None:
        return "n/a"
    return "(%.4f, %.4f, %.4f)" % (float(pos[0]), float(pos[1]), float(pos[2]))


def interpolate_vertical_tcp(
    pre_plan: Tuple[float, float, float],
    gr_plan: Tuple[float, float, float],
    alpha: float,
) -> Tuple[float, float, float]:
    a = float(alpha)
    return (
        float(pre_plan[0]),
        float(pre_plan[1]),
        float(pre_plan[2]) + a * (float(gr_plan[2]) - float(pre_plan[2])),
    )


def build_mustard_descend_candidate_specs(
    *,
    pre_plan: Tuple[float, float, float],
    top_z_m: float,
    xy: Tuple[float, float],
    depth_candidates_m: Sequence[float] = MUSTARD_DEPTH_FROM_TOP_M,
    descend_candidates_m: Sequence[float] = MUSTARD_DESCEND_DELTA_M,
    min_grasp_z_m: Optional[float] = None,
) -> List[Dict[str, Any]]:
    """Candidatos grasp_tcp conservadores (descenso corto primero)."""
    specs: List[Dict[str, Any]] = []
    seen_z: set = set()
    pre_z = float(pre_plan[2])

    def _add(
        grasp_z: float,
        *,
        depth_from_top_m: Optional[float],
        descend_delta_m: float,
        source: str,
    ) -> None:
        gz = round(float(grasp_z), 5)
        if gz >= pre_z - 1e-4:
            return
        if min_grasp_z_m is not None and gz + 1e-6 < float(min_grasp_z_m):
            return
        key = round(gz, 3)
        if key in seen_z:
            return
        seen_z.add(key)
        specs.append(
            {
                "grasp_tcp": (float(xy[0]), float(xy[1]), gz),
                "depth_from_top_m": depth_from_top_m,
                "descend_delta_m": float(descend_delta_m),
                "source": str(source),
            }
        )

    for depth in depth_candidates_m:
        depth_f = float(depth)
        grasp_z = float(top_z_m) - depth_f
        descend_delta = pre_z - grasp_z
        _add(
            grasp_z,
            depth_from_top_m=depth_f,
            descend_delta_m=descend_delta,
            source="depth_from_top",
        )

    for delta in descend_candidates_m:
        delta_f = float(delta)
        grasp_z = pre_z - delta_f
        depth_from_top = float(top_z_m) - grasp_z
        _add(
            grasp_z,
            depth_from_top_m=depth_from_top,
            descend_delta_m=delta_f,
            source="descend_delta",
        )

    specs.sort(key=lambda s: float(s["descend_delta_m"]))
    return specs


def evaluate_mustard_contact_depth_policy(
    *,
    depth_from_top_m: Optional[float],
    insertion_depth_limit_m: Optional[float],
    recommended_depth_m: Optional[float] = None,
) -> Tuple[bool, str]:
    if depth_from_top_m is None:
        return False, "missing_depth_from_top"
    depth = float(depth_from_top_m)
    lim = insertion_depth_limit_m
    if lim is None:
        lim = recommended_depth_m
    if lim is None:
        return True, "no_insertion_limit"
    if depth + 1e-6 > float(lim):
        return False, "depth_exceeds_insertion_limit"
    if depth + 1e-6 < 0.005:
        return False, "depth_too_shallow"
    return True, "contact_depth_ok"


def format_mustard_cartesian_descend_step_diag_log(fields: Dict[str, Any]) -> str:
    return (
        "[MUSTARD_CARTESIAN_DESCEND_STEP_DIAG]\n"
        "step_idx=%s\n"
        "alpha=%s\n"
        "target_tcp=%s\n"
        "target_hand=%s\n"
        "endpoint_ik_ok=%s\n"
        "collision_ok=%s\n"
        "joint_limits_ok=%s\n"
        "reason=%s"
        % (
            fields.get("step_idx", "n/a"),
            fields.get("alpha", "n/a"),
            fields.get("target_tcp", "n/a"),
            fields.get("target_hand", "n/a"),
            fields.get("endpoint_ik_ok", "n/a"),
            fields.get("collision_ok", "n/a"),
            fields.get("joint_limits_ok", "n/a"),
            str(fields.get("reason") or ""),
        )
    )


def format_mustard_endpoint_ik_scene_log(fields: Dict[str, Any]) -> str:
    obstacles = fields.get("obstacles_present")
    if isinstance(obstacles, (list, tuple)):
        obs_str = "[" + ",".join(str(x) for x in obstacles) + "]"
    else:
        obs_str = str(obstacles or "[]")
    return (
        "[MUSTARD_ENDPOINT_IK_SCENE]\n"
        "target_collision_present_before=%s\n"
        "target_collision_present_during_endpoint=%s\n"
        "obstacles_present=%s\n"
        "result=%s"
        % (
            str(fields.get("target_collision_present_before", "n/a")),
            str(fields.get("target_collision_present_during_endpoint", "n/a")),
            obs_str,
            str(fields.get("result") or "OK"),
        )
    )


def format_mustard_grasp_endpoint_ik_validate_log(fields: Dict[str, Any]) -> str:
    return (
        "[MUSTARD_GRASP_ENDPOINT_IK_VALIDATE]\n"
        "endpoint_seed_source=%s\n"
        "endpoint_seed_pregrasp_tcp_z=%s\n"
        "endpoint_seed_joints=%s\n"
        "target_tcp=%s\n"
        "target_hand=%s\n"
        "quaternion=%s\n"
        "commanded_tcp_yaw_rad=%s\n"
        "target_collision_present=%s\n"
        "pregrasp_tcp=%s\n"
        "grasp_tcp=%s\n"
        "endpoint_ik_ok=%s\n"
        "plan_to_endpoint_ok=%s\n"
        "result=%s"
        % (
            str(fields.get("endpoint_seed_source", "n/a")),
            str(fields.get("endpoint_seed_pregrasp_tcp_z", "n/a")),
            str(fields.get("endpoint_seed_joints", "n/a")),
            str(fields.get("target_tcp", "n/a")),
            str(fields.get("target_hand", "n/a")),
            str(fields.get("quaternion", "n/a")),
            str(fields.get("commanded_tcp_yaw_rad", "n/a")),
            str(fields.get("target_collision_present", "n/a")),
            str(fields.get("pregrasp_tcp", "n/a")),
            str(fields.get("grasp_tcp", "n/a")),
            str(fields.get("endpoint_ik_ok", "n/a")),
            str(fields.get("plan_to_endpoint_ok", "n/a")),
            str(fields.get("result", "n/a")),
        )
    )


def format_mustard_simple_descend_prevalidate_log(fields: Dict[str, Any]) -> str:
    return (
        "[MUSTARD_SIMPLE_DESCEND_PREVALIDATE]\n"
        "pregrasp_tcp=%s\n"
        "grasp_tcp=%s\n"
        "descend_delta=%s\n"
        "cartesian_fraction=%s\n"
        "endpoint_ik_ok=%s\n"
        "selected_depth_from_top=%s\n"
        "result=%s\n"
        "reason=%s"
        % (
            fields.get("pregrasp_tcp", "n/a"),
            fields.get("grasp_tcp", "n/a"),
            fields.get("descend_delta", "n/a"),
            fields.get("cartesian_fraction", "n/a"),
            fields.get("endpoint_ik_ok", "n/a"),
            fields.get("selected_depth_from_top", "n/a"),
            fields.get("result", "n/a"),
            str(fields.get("reason") or ""),
        )
    )


def cartesian_accept(
    *,
    fraction: Optional[float],
    threshold: float,
    traj_points: int,
    start_state_honored: Optional[bool],
) -> bool:
    if start_state_honored is not None and not bool(start_state_honored):
        return False
    if fraction is None:
        return False
    if int(traj_points) < 2:
        return False
    return float(fraction) + 1e-6 >= float(threshold)
