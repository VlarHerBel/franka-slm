"""Auditoría de start_state para GetCartesianPath (validación paired pregrasp+descenso)."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

from geometry_msgs.msg import Pose, Quaternion
from moveit_msgs.msg import RobotState
from moveit_msgs.srv import GetCartesianPath
from sensor_msgs.msg import JointState

from panda_controller.tfg_motion_waypoints import PANDA_ARM_JOINT_NAMES, joint_values_7d_from_any

START_STATE_HONOR_MAX_ABS_RAD = 0.02

REJECT_ENDPOINT_IK_FAILED = "endpoint_ik_failed"
REJECT_ENDPOINT_IK_RAW_FAILED = "endpoint_ik_raw_failed"
REJECT_CARTESIAN_START_STATE_NOT_HONORED = "cartesian_start_state_not_honored"
REJECT_CARTESIAN_FRACTION_LOW = "cartesian_fraction_low"
REJECT_COLLISION = "collision"
REJECT_JOINT_LIMIT = "joint_limit"
REJECT_POST_LIFT_EXIT_FAIL = "post_lift_exit_fail"
REJECT_TRANSPORT_FAIL = "transport_fail"


def hand_pose_to_geometry_pose(
    hand_pos: Tuple[float, float, float],
    hand_quat: Tuple[float, float, float, float],
) -> Pose:
    pose = Pose()
    pose.position.x = float(hand_pos[0])
    pose.position.y = float(hand_pos[1])
    pose.position.z = float(hand_pos[2])
    pose.orientation = Quaternion(
        x=float(hand_quat[0]),
        y=float(hand_quat[1]),
        z=float(hand_quat[2]),
        w=float(hand_quat[3]),
    )
    return pose


def build_get_cartesian_path_request(
    *,
    planning_frame: str,
    group_name: str,
    link_name: str,
    start_js: Any,
    hand_goal: Tuple[float, float, float],
    hand_quat: Tuple[float, float, float, float],
    max_step: float = 0.0025,
    jump_threshold: float = 0.0,
    avoid_collisions: bool = True,
    joint_names: Sequence[str] = PANDA_ARM_JOINT_NAMES,
) -> Optional[GetCartesianPath.Request]:
    start_positions = joint_values_7d_from_any(
        start_js, context="get_cartesian_path_start_state"
    )
    if start_positions is None:
        return None
    req = GetCartesianPath.Request()
    req.header.frame_id = str(planning_frame)
    req.group_name = str(group_name)
    req.link_name = str(link_name)
    req.max_step = float(max_step)
    req.jump_threshold = float(jump_threshold)
    req.avoid_collisions = bool(avoid_collisions)
    req.start_state = RobotState()
    req.start_state.is_diff = False
    js = JointState()
    js.name = [str(n) for n in joint_names]
    js.position = [float(v) for v in start_positions]
    req.start_state.joint_state = js
    req.waypoints = [hand_pose_to_geometry_pose(hand_goal, hand_quat)]
    return req


def first_trajectory_joint_positions(
    response: GetCartesianPath.Response,
    joint_names: Sequence[str] = PANDA_ARM_JOINT_NAMES,
) -> Optional[List[float]]:
    traj = getattr(response, "solution", None)
    if traj is None:
        return None
    jt = getattr(traj, "joint_trajectory", None)
    if jt is None or not jt.points:
        return None
    pt0 = jt.points[0]
    if not pt0.positions:
        return None
    if jt.joint_names:
        name_to_pos = {
            str(n): float(p) for n, p in zip(jt.joint_names, pt0.positions)
        }
        try:
            return [name_to_pos[str(n)] for n in joint_names]
        except KeyError:
            pass
    if len(pt0.positions) >= len(joint_names):
        return [float(v) for v in pt0.positions[: len(joint_names)]]
    return None


def last_trajectory_joint_positions(
    response: GetCartesianPath.Response,
    joint_names: Sequence[str] = PANDA_ARM_JOINT_NAMES,
) -> Optional[List[float]]:
    traj = getattr(response, "solution", None)
    if traj is None:
        return None
    jt = getattr(traj, "joint_trajectory", None)
    if jt is None or not jt.points:
        return None
    pt_last = jt.points[-1]
    if not pt_last.positions:
        return None
    if jt.joint_names:
        name_to_pos = {
            str(n): float(p) for n, p in zip(jt.joint_names, pt_last.positions)
        }
        try:
            return [name_to_pos[str(n)] for n in joint_names]
        except KeyError:
            pass
    if len(pt_last.positions) >= len(joint_names):
        return [float(v) for v in pt_last.positions[: len(joint_names)]]
    return None


def max_abs_joint_delta(
    a: Sequence[float],
    b: Sequence[float],
) -> float:
    if not a or not b:
        return float("inf")
    n = min(len(a), len(b))
    return max(abs(float(a[i]) - float(b[i])) for i in range(n))


def wrap_to_pi(angle: float) -> float:
    a = float(angle)
    while a > math.pi:
        a -= 2.0 * math.pi
    while a < -math.pi:
        a += 2.0 * math.pi
    return a


def weighted_joint_distance(a: Sequence[float], b: Sequence[float]) -> float:
    wrist = ("panda_joint6", "panda_joint7")
    total = 0.0
    for i, name in enumerate(PANDA_ARM_JOINT_NAMES):
        if i >= len(a) or i >= len(b):
            continue
        delta = wrap_to_pi(float(a[i]) - float(b[i]))
        w = 3.0 if name in wrist else 1.0
        total += w * abs(delta)
    return float(total)


def evaluate_get_cartesian_path_start_state_audit(
    *,
    requested_start_js: Any,
    response: Optional[GetCartesianPath.Response],
    current_js: Any = None,
    honor_tol_rad: float = START_STATE_HONOR_MAX_ABS_RAD,
    joint_names: Sequence[str] = PANDA_ARM_JOINT_NAMES,
) -> Dict[str, Any]:
    requested = joint_values_7d_from_any(
        requested_start_js, context="cartesian_start_state_audit_requested"
    )
    returned_first = (
        first_trajectory_joint_positions(response, joint_names)
        if response is not None
        else None
    )
    returned_last = (
        last_trajectory_joint_positions(response, joint_names)
        if response is not None
        else None
    )
    current = joint_values_7d_from_any(
        current_js, context="cartesian_start_state_audit_current"
    )
    max_delta = (
        max_abs_joint_delta(requested, returned_first)
        if requested is not None and returned_first is not None
        else None
    )
    if response is None:
        start_state_honored = None
    else:
        start_state_honored = bool(
            requested is not None
            and returned_first is not None
            and max_delta is not None
            and float(max_delta) <= float(honor_tol_rad) + 1e-9
        )
    current_dist = (
        weighted_joint_distance(current, requested)
        if current is not None and requested is not None
        else None
    )
    fraction = None
    traj_pts = 0
    if response is not None:
        fraction = float(response.fraction)
        traj = getattr(response, "solution", None)
        jt = getattr(traj, "joint_trajectory", None) if traj is not None else None
        if jt is not None and jt.points:
            traj_pts = len(jt.points)
    result = "FAIL"
    if start_state_honored is None:
        result = "NOT_EVALUATED"
    elif start_state_honored and fraction is not None and float(fraction) + 1e-6 >= 0.95:
        result = "OK"
    elif start_state_honored and fraction is not None:
        result = "FAIL"
    elif not start_state_honored:
        result = "FAIL"
    return {
        "requested_start_js": requested,
        "returned_first_point_js": returned_first,
        "returned_last_point_js": returned_last,
        "max_abs_delta_start_vs_first": max_delta,
        "start_state_honored": start_state_honored,
        "current_js_distance_to_requested_start": current_dist,
        "fraction": fraction,
        "traj_pts": int(traj_pts),
        "result": result,
    }


def format_start_state_honored_log_value(honored: Any) -> str:
    if honored is None:
        return "not_evaluated"
    return str(bool(honored)).lower()


def format_get_cartesian_path_start_state_audit_log(
    *,
    candidate_idx: Optional[int],
    audit: Dict[str, Any],
    start_state_source: Optional[str] = None,
) -> str:
    req = audit.get("requested_start_js")
    ret = audit.get("returned_first_point_js")
    return (
        "[GET_CARTESIAN_PATH_START_STATE_AUDIT]\n"
        "candidate_idx=%s\n"
        "start_state_source=%s\n"
        "requested_start_js=%s\n"
        "returned_first_point_js=%s\n"
        "max_abs_delta_start_vs_first=%s\n"
        "start_state_honored=%s\n"
        "current_js_distance_to_requested_start=%s\n"
        "fraction=%s\n"
        "traj_pts=%s\n"
        "result=%s"
        % (
            "n/a" if candidate_idx is None else str(int(candidate_idx)),
            str(start_state_source or audit.get("start_state_source") or "n/a"),
            "n/a" if req is None else str([round(float(v), 4) for v in req]),
            "n/a" if ret is None else str([round(float(v), 4) for v in ret]),
            "n/a"
            if audit.get("max_abs_delta_start_vs_first") is None
            else "%.6f" % float(audit["max_abs_delta_start_vs_first"]),
            format_start_state_honored_log_value(audit.get("start_state_honored")),
            "n/a"
            if audit.get("current_js_distance_to_requested_start") is None
            else "%.4f" % float(audit["current_js_distance_to_requested_start"]),
            "n/a" if audit.get("fraction") is None else "%.5f" % float(audit["fraction"]),
            audit.get("traj_pts", "n/a"),
            str(audit.get("result", "FAIL")),
        )
    )


def normalize_paired_grid_reject_reason(raw: str) -> str:
    reason = str(raw or "").strip()
    if not reason:
        return ""
    if reason in (
        "endpoint_ik_fail",
        REJECT_ENDPOINT_IK_FAILED,
    ):
        return REJECT_ENDPOINT_IK_FAILED
    if reason in (
        "endpoint_ik_raw_fail",
        REJECT_ENDPOINT_IK_RAW_FAILED,
    ):
        return REJECT_ENDPOINT_IK_RAW_FAILED
    if reason == REJECT_CARTESIAN_START_STATE_NOT_HONORED:
        return REJECT_CARTESIAN_START_STATE_NOT_HONORED
    if reason.startswith("cartesian_descend_fail"):
        return REJECT_CARTESIAN_FRACTION_LOW
    if reason in ("descend_volume_collision", "collision"):
        return REJECT_COLLISION
    if reason in ("joint_limit", "joint_limit_near"):
        return REJECT_JOINT_LIMIT
    if reason == "lift_fail":
        return PAIRED_REJECT_REASON_LIFT_AFTER_VALID_DESCEND
    if reason in ("transport_exit_fail", REJECT_TRANSPORT_FAIL):
        return REJECT_TRANSPORT_FAIL
    if reason in ("post_lift_exit_fail", REJECT_POST_LIFT_EXIT_FAIL):
        return REJECT_POST_LIFT_EXIT_FAIL
    return reason


def aggregate_paired_grid_summary(
    records: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    reject_counts: Dict[str, int] = {}
    accepted = 0
    best: Optional[Dict[str, Any]] = None
    for rec in records:
        if str(rec.get("result", "")).upper() == "ACCEPT":
            accepted += 1
        norm = normalize_paired_grid_reject_reason(str(rec.get("reject_reason", "")))
        if norm:
            reject_counts[norm] = int(reject_counts.get(norm, 0)) + 1
        frac = rec.get("cartesian_fraction")
        if frac is None:
            continue
        if best is None or float(frac) > float(best.get("cartesian_fraction", -1.0)):
            best = rec
    return {
        "total_candidates": len(records),
        "accepted": accepted,
        "reject_reason_counts": reject_counts,
        "best": best,
    }


def format_paired_grid_summary_log(summary: Dict[str, Any]) -> str:
    best = summary.get("best") or {}
    counts = summary.get("reject_reason_counts") or {}
    counts_repr = "{%s}" % ", ".join(
        "%s=%d" % (k, int(v)) for k, v in sorted(counts.items())
    )
    return (
        "[PAIRED_GRID_SUMMARY]\n"
        "total_candidates=%d\n"
        "accepted=%d\n"
        "reject_reason_counts=%s\n"
        "best_cartesian_fraction=%s\n"
        "best_candidate_idx=%s\n"
        "best_yaw_deg=%s\n"
        "best_pregrasp_tcp_z=%s\n"
        "best_grasp_tcp_z=%s\n"
        "best_depth_from_top=%s\n"
        "best_endpoint_ik_ok=%s\n"
        "best_start_state_honored=%s\n"
        "best_reject_reason=%s"
        % (
            int(summary.get("total_candidates", 0)),
            int(summary.get("accepted", 0)),
            counts_repr,
            best.get("cartesian_fraction", "n/a"),
            best.get("candidate_idx", "n/a"),
            best.get("yaw_deg", "n/a"),
            best.get("pregrasp_tcp_z", "n/a"),
            best.get("grasp_tcp_z", "n/a"),
            best.get("depth_from_top", "n/a"),
            str(bool(best.get("endpoint_ik_ok"))).lower()
            if "endpoint_ik_ok" in best
            else "n/a",
            format_start_state_honored_log_value(best.get("start_state_honored"))
            if "start_state_honored" in best
            else "n/a",
            normalize_paired_grid_reject_reason(str(best.get("reject_reason", "")))
            or "n/a",
        )
    )


def cartesian_validation_invalid_when_start_state_not_honored(
    audit: Dict[str, Any],
) -> bool:
    honored = audit.get("start_state_honored")
    if honored is None:
        return False
    return not bool(honored)


def paired_cartesian_fraction_reject_reason(
    *,
    audit: Dict[str, Any],
    fraction: float,
    threshold: float,
    traj_pts: int,
) -> str:
    if cartesian_validation_invalid_when_start_state_not_honored(audit):
        return REJECT_CARTESIAN_START_STATE_NOT_HONORED
    if int(traj_pts) < 2:
        return REJECT_CARTESIAN_FRACTION_LOW
    if float(fraction) + 1e-6 < float(threshold):
        return REJECT_CARTESIAN_FRACTION_LOW
    return ""
