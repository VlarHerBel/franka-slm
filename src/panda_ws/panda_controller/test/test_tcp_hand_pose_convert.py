"""Tests SE(3) TCP -> panda_hand (sin ROS)."""

import math

import numpy as np

from panda_controller.math_tf_utils import quaternion_from_euler
from panda_controller.tcp_hand_pose_convert import (
    compose_world_tcp_from_hand,
    evaluate_pregrasp_endpoint_fk_contract,
    format_pregrasp_endpoint_fk_contract_log,
    format_tcp_to_hand_full_transform_log,
    hand_pose_from_desired_tcp,
    hand_pose_from_tcp_with_hand_orientation,
    hand_position_from_desired_tcp,
    invert_transform,
    naive_world_z_hand_goal,
    pose_to_matrix,
    xy_compensation_m,
)


HAND_TO_TCP_T = (0.0, 0.0, 0.100)
HAND_TO_TCP_Q = (0.0, 0.0, 0.0, 1.0)


def _top_down_tcp_quat(yaw_rad: float) -> tuple:
    roll, pitch, _ = math.pi, 0.0, float(yaw_rad)
    return quaternion_from_euler(roll, pitch, yaw_rad)


def test_hand_pose_roundtrip_top_down_with_yaw() -> None:
    desired_tcp = (0.455, 0.115, 0.562)
    desired_q = _top_down_tcp_quat(0.35)
    hand_pos, hand_q = hand_pose_from_desired_tcp(
        desired_tcp,
        desired_q,
        HAND_TO_TCP_T,
        HAND_TO_TCP_Q,
    )
    tcp_back, q_back = compose_world_tcp_from_hand(
        hand_pos,
        hand_q,
        HAND_TO_TCP_T,
        HAND_TO_TCP_Q,
    )
    assert math.dist(desired_tcp, tcp_back) < 1e-6
    # Orientación: comparar matrices de rotación.
    R_des = pose_to_matrix(desired_tcp, desired_q)[:3, :3]
    R_back = pose_to_matrix(tcp_back, q_back)[:3, :3]
    assert float(np.linalg.norm(R_des - R_back)) < 1e-5


def test_hand_position_has_xy_offset_when_approach_not_world_z() -> None:
    """Con pitch != 0 el offset local [0,0,0.1] debe proyectar XY en mundo."""
    desired_tcp = (0.455, 0.115, 0.562)
    desired_q = quaternion_from_euler(math.pi, 0.25, 0.40)
    hand_pos = hand_position_from_desired_tcp(
        desired_tcp,
        desired_q,
        HAND_TO_TCP_T,
        HAND_TO_TCP_Q,
    )
    assert abs(hand_pos[0] - desired_tcp[0]) > 0.005
    assert abs(hand_pos[1] - desired_tcp[1]) > 0.005


def test_se3_inverse_matches_explicit_compose() -> None:
    desired_tcp = (0.4791, 0.0934, 0.5676)
    desired_q = _top_down_tcp_quat(-0.2)
    T_w_tcp = pose_to_matrix(desired_tcp, desired_q)
    T_h_t = pose_to_matrix(HAND_TO_TCP_T, HAND_TO_TCP_Q)
    T_w_h = T_w_tcp @ invert_transform(T_h_t)
    hand_pos, hand_q = hand_pose_from_desired_tcp(
        desired_tcp,
        desired_q,
        HAND_TO_TCP_T,
        HAND_TO_TCP_Q,
    )
    assert math.dist(hand_pos, (float(T_w_h[0, 3]), float(T_w_h[1, 3]), float(T_w_h[2, 3]))) < 1e-9


def test_world_z_plus_offset_is_wrong_for_tilted_approach() -> None:
    """Regresión: hand != tcp + (0,0,0.1) en mundo cuando hay pitch."""
    desired_tcp = (0.455, 0.115, 0.562)
    desired_q = quaternion_from_euler(math.pi, 0.30, 0.10)
    hand_pos = hand_position_from_desired_tcp(
        desired_tcp,
        desired_q,
        HAND_TO_TCP_T,
        HAND_TO_TCP_Q,
    )
    naive_world_z = (
        desired_tcp[0],
        desired_tcp[1],
        desired_tcp[2] + 0.100,
    )
    assert math.dist(hand_pos, naive_world_z) > 0.01


def test_hand_pose_with_hand_orientation_matches_full_se3_for_pure_translation() -> None:
    desired_tcp = (0.630, -0.175, 0.470)
    desired_q = _top_down_tcp_quat(0.25)
    hand_se3, _ = hand_pose_from_desired_tcp(
        desired_tcp, desired_q, HAND_TO_TCP_T, HAND_TO_TCP_Q
    )
    hand_orient, _ = hand_pose_from_tcp_with_hand_orientation(
        desired_tcp, desired_q, HAND_TO_TCP_T
    )
    assert math.dist(hand_se3, hand_orient) < 1e-6


def test_hand_orientation_path_xy_offset_when_pitch_nonzero() -> None:
    desired_tcp = (0.630, -0.175, 0.470)
    hand_q = quaternion_from_euler(math.pi, 0.25, 0.40)
    hand_pos, _ = hand_pose_from_tcp_with_hand_orientation(
        desired_tcp, hand_q, HAND_TO_TCP_T
    )
    naive = naive_world_z_hand_goal(desired_tcp)
    assert xy_compensation_m(hand_pos, naive) > 0.005


def test_pregrasp_endpoint_fk_contract_pass_and_fail() -> None:
    desired = (0.630, -0.175, 0.470)
    desired_hand = (0.630, -0.175, 0.570)
    desired_q = _top_down_tcp_quat(0.25)
    ok_result = evaluate_pregrasp_endpoint_fk_contract(
        desired_tcp=desired,
        desired_hand_pos=desired_hand,
        desired_hand_quat=desired_q,
        fk_tcp=(0.6305, -0.1752, 0.4701),
        fk_hand_pos=desired_hand,
        fk_hand_quat=desired_q,
    )
    assert bool(ok_result["ok"])
    fail_tcp = evaluate_pregrasp_endpoint_fk_contract(
        desired_tcp=desired,
        desired_hand_pos=desired_hand,
        desired_hand_quat=desired_q,
        fk_tcp=(0.6365, -0.1928, 0.470),
        fk_hand_pos=desired_hand,
        fk_hand_quat=desired_q,
    )
    assert not bool(fail_tcp["ok"])
    assert fail_tcp["reject_reason"] == "pregrasp_endpoint_fk_contract_fail"
    fail_orient = evaluate_pregrasp_endpoint_fk_contract(
        desired_tcp=desired,
        desired_hand_pos=desired_hand,
        desired_hand_quat=desired_q,
        fk_tcp=(0.6305, -0.1752, 0.4701),
        fk_hand_pos=desired_hand,
        fk_hand_quat=quaternion_from_euler(math.pi, 0.20, 0.25),
        hand_orientation_tol_deg=2.0,
    )
    assert not bool(fail_orient["ok"])
    assert fail_orient["reject_reason"] == "pregrasp_endpoint_orientation_contract_fail"
    log = format_pregrasp_endpoint_fk_contract_log(fail_tcp, result="FAIL")
    assert "[PREGRASP_ENDPOINT_FK_CONTRACT]" in log
    assert "hand_orientation_error_deg=" in log
    assert "expected_hand_to_tcp_base=" in log


def test_tcp_to_hand_full_transform_log_format() -> None:
    log = format_tcp_to_hand_full_transform_log(
        target_link="panda_hand",
        tcp_goal=(0.630, -0.175, 0.470),
        hand_goal=(0.625, -0.180, 0.565),
        hand_to_tcp_local=(0.0, 0.0, 0.1),
        hand_to_tcp_base=(0.01, -0.02, -0.09),
        old_world_z_hand_goal=(0.630, -0.175, 0.570),
        xy_compensation_m=0.036,
        result="OK",
    )
    assert "[TCP_TO_HAND_FULL_TRANSFORM]" in log
    assert "xy_compensation_m=0.0360" in log
