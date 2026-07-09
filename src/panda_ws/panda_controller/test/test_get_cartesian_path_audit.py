"""Tests auditoría start_state GetCartesianPath (paired pregrasp+descenso)."""

from moveit_msgs.msg import RobotTrajectory
from moveit_msgs.srv import GetCartesianPath
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from panda_controller.get_cartesian_path_audit import (
    REJECT_CARTESIAN_FRACTION_LOW,
    REJECT_CARTESIAN_START_STATE_NOT_HONORED,
    REJECT_ENDPOINT_IK_FAILED,
    aggregate_paired_grid_summary,
    build_get_cartesian_path_request,
    evaluate_get_cartesian_path_start_state_audit,
    format_paired_grid_summary_log,
    format_start_state_honored_log_value,
    last_trajectory_joint_positions,
    normalize_paired_grid_reject_reason,
    paired_cartesian_fraction_reject_reason,
)
from panda_controller.tfg_motion_waypoints import PANDA_ARM_JOINT_NAMES


def _virtual_pregrasp_js() -> list:
    return [0.35, -0.25, 0.10, -2.10, 0.05, 1.85, 0.42]


def _current_home_js() -> list:
    return [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785]


def _make_cartesian_response(
    first_js,
    *,
    fraction: float = 0.98,
    last_js=None,
) -> GetCartesianPath.Response:
    resp = GetCartesianPath.Response()
    resp.fraction = float(fraction)
    resp.error_code.val = 1
    jt = JointTrajectory()
    jt.joint_names = list(PANDA_ARM_JOINT_NAMES)
    pt0 = JointTrajectoryPoint()
    pt0.positions = [float(v) for v in first_js]
    pts = [pt0]
    if last_js is not None:
        pt1 = JointTrajectoryPoint()
        pt1.positions = [float(v) for v in last_js]
        pts.append(pt1)
    jt.points = pts
    resp.solution = RobotTrajectory()
    resp.solution.joint_trajectory = jt
    return resp


def test_build_request_uses_virtual_start_js() -> None:
    virtual = _virtual_pregrasp_js()
    js = JointState()
    js.name = list(PANDA_ARM_JOINT_NAMES)
    js.position = list(virtual)
    req = build_get_cartesian_path_request(
        planning_frame="panda_link0",
        group_name="panda_arm",
        link_name="panda_hand",
        start_js=js,
        hand_goal=(0.455, 0.115, 0.437),
        hand_quat=(1.0, 0.0, 0.0, 0.0),
    )
    assert req is not None
    assert req.start_state.is_diff is False
    assert list(req.start_state.joint_state.position) == virtual
    assert req.avoid_collisions is True
    assert req.link_name == "panda_hand"


def test_last_trajectory_joint_positions_returns_final_point() -> None:
    first = _virtual_pregrasp_js()
    last = [float(v) + 0.05 for v in first]
    resp = _make_cartesian_response(first, last_js=last)
    got = last_trajectory_joint_positions(resp)
    assert got is not None
    assert [round(v, 3) for v in got] == [round(v, 3) for v in last]


def test_start_state_honored_when_first_point_matches_virtual_js() -> None:
    virtual = _virtual_pregrasp_js()
    current = _current_home_js()
    js = JointState()
    js.name = list(PANDA_ARM_JOINT_NAMES)
    js.position = list(virtual)
    current_js = JointState()
    current_js.name = list(PANDA_ARM_JOINT_NAMES)
    current_js.position = list(current)
    resp = _make_cartesian_response(virtual)
    audit = evaluate_get_cartesian_path_start_state_audit(
        requested_start_js=js,
        response=resp,
        current_js=current_js,
    )
    assert audit["start_state_honored"] is True
    assert float(audit["max_abs_delta_start_vs_first"]) < 1e-6
    assert float(audit["current_js_distance_to_requested_start"]) > 0.5
    assert audit["returned_last_point_js"] == virtual


def test_start_state_not_honored_when_first_point_is_current_js() -> None:
    virtual = _virtual_pregrasp_js()
    current = _current_home_js()
    js = JointState()
    js.name = list(PANDA_ARM_JOINT_NAMES)
    js.position = list(virtual)
    current_js = JointState()
    current_js.name = list(PANDA_ARM_JOINT_NAMES)
    current_js.position = list(current)
    resp = _make_cartesian_response(current)
    audit = evaluate_get_cartesian_path_start_state_audit(
        requested_start_js=js,
        response=resp,
        current_js=current_js,
    )
    assert audit["start_state_honored"] is False
    assert paired_cartesian_fraction_reject_reason(
        audit=audit,
        fraction=0.98,
        threshold=0.95,
        traj_pts=10,
    ) == REJECT_CARTESIAN_START_STATE_NOT_HONORED


def test_fraction_low_only_when_start_state_honored() -> None:
    virtual = _virtual_pregrasp_js()
    js = JointState()
    js.name = list(PANDA_ARM_JOINT_NAMES)
    js.position = list(virtual)
    resp = _make_cartesian_response(virtual, fraction=0.50)
    audit = evaluate_get_cartesian_path_start_state_audit(
        requested_start_js=js,
        response=resp,
    )
    assert audit["start_state_honored"] is True
    assert (
        paired_cartesian_fraction_reject_reason(
            audit=audit,
            fraction=0.50,
            threshold=0.95,
            traj_pts=5,
        )
        == REJECT_CARTESIAN_FRACTION_LOW
    )


def test_paired_grid_summary_aggregates_reject_reasons() -> None:
    records = [
        {
            "result": "REJECT",
            "reject_reason": "endpoint_ik_fail",
            "cartesian_fraction": "0.50000",
            "candidate_idx": 0,
            "yaw_deg": "0.00",
            "pregrasp_tcp_z": "0.5750",
            "grasp_tcp_z": "0.4370",
            "depth_from_top": "0.0330",
            "endpoint_ik_ok": False,
            "start_state_honored": True,
        },
        {
            "result": "REJECT",
            "reject_reason": REJECT_CARTESIAN_FRACTION_LOW,
            "cartesian_fraction": "0.98000",
            "candidate_idx": 1,
            "yaw_deg": "5.00",
            "pregrasp_tcp_z": "0.5950",
            "grasp_tcp_z": "0.4370",
            "depth_from_top": "0.0330",
            "endpoint_ik_ok": True,
            "start_state_honored": True,
        },
    ]
    summary = aggregate_paired_grid_summary(records)
    assert summary["total_candidates"] == 2
    assert summary["accepted"] == 0
    assert summary["reject_reason_counts"][REJECT_ENDPOINT_IK_FAILED] == 1
    assert summary["reject_reason_counts"][REJECT_CARTESIAN_FRACTION_LOW] == 1
    log = format_paired_grid_summary_log(summary)
    assert "[PAIRED_GRID_SUMMARY]" in log
    assert "best_cartesian_fraction=0.98000" in log


def test_start_state_honored_not_evaluated_without_response() -> None:
    virtual = _virtual_pregrasp_js()
    js = JointState()
    js.name = list(PANDA_ARM_JOINT_NAMES)
    js.position = list(virtual)
    audit = evaluate_get_cartesian_path_start_state_audit(
        requested_start_js=js,
        response=None,
    )
    assert audit["start_state_honored"] is None
    assert audit["result"] == "NOT_EVALUATED"
    assert format_start_state_honored_log_value(None) == "not_evaluated"


def test_normalize_endpoint_ik_fail_alias() -> None:
    assert (
        normalize_paired_grid_reject_reason("endpoint_ik_fail")
        == REJECT_ENDPOINT_IK_FAILED
    )
