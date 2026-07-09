"""Tests postura IK y corte cinemático descenso final chips_can."""

import math

from panda_controller.chips_can_final_descend_posture import (
    CHIPS_CAN_FAVORABLE_POSITION_PROBE_XY,
    CHIPS_CAN_POSTURE_SEARCH_IMPL_REV,
    CHIPS_CAN_SHALLOW_FINAL_DESCEND_DEPTH_M,
    chips_can_shallow_grasp_tcp_z,
    estimate_cartesian_kinematic_cutoff_tcp_z,
    format_chips_can_posture_search_start_log,
    infer_cartesian_kinematic_suspected_cause,
    select_chips_can_high_route_posture_variant,
    summarize_chips_can_posture_search,
)


def test_shallow_grasp_tcp_z() -> None:
    assert chips_can_shallow_grasp_tcp_z(top_z_m=0.510) == 0.505


def test_kinematic_cutoff_matches_runtime_diagnosis() -> None:
    start_z = 0.610
    target_z = 0.505
    frac = 0.62791
    cutoff = estimate_cartesian_kinematic_cutoff_tcp_z(
        start_tcp_z=start_z,
        target_tcp_z=target_z,
        achieved_fraction=frac,
    )
    assert math.isclose(cutoff, 0.544, rel_tol=0, abs_tol=0.002)


def test_infer_kinematic_suspected_cause() -> None:
    assert (
        infer_cartesian_kinematic_suspected_cause(
            achieved_fraction=0.628,
            with_obstacles_fraction=0.628,
            without_obstacles_fraction=0.628,
        )
        == "ik_limit"
    )
    assert (
        infer_cartesian_kinematic_suspected_cause(
            achieved_fraction=0.49,
            with_obstacles_fraction=0.49,
            without_obstacles_fraction=0.98,
        )
        == "obstacle_collision"
    )


def test_select_posture_variant_prefers_lower_joint_dist() -> None:
    variants = [
        {
            "full_route_ok": True,
            "low_to_grasp_fraction": 0.98,
            "joint_dist": 2.0,
            "ik_seed_name": "home",
        },
        {
            "full_route_ok": True,
            "low_to_grasp_fraction": 0.96,
            "joint_dist": 0.5,
            "ik_seed_name": "pick_workspace_ready",
        },
        {
            "full_route_ok": False,
            "low_to_grasp_fraction": 0.99,
            "joint_dist": 0.1,
            "ik_seed_name": "fail",
        },
    ]
    selected = select_chips_can_high_route_posture_variant(
        variants, fraction_threshold=0.95
    )
    assert selected is not None
    assert selected["ik_seed_name"] == "pick_workspace_ready"


def test_favorable_position_probe_coordinates() -> None:
    assert (0.48, -0.04) in CHIPS_CAN_FAVORABLE_POSITION_PROBE_XY
    assert (0.48, 0.00) in CHIPS_CAN_FAVORABLE_POSITION_PROBE_XY
    assert CHIPS_CAN_SHALLOW_FINAL_DESCEND_DEPTH_M == 0.005


def test_posture_search_start_log_contains_impl_rev() -> None:
    msg = format_chips_can_posture_search_start_log(
        {
            "impl_rev": CHIPS_CAN_POSTURE_SEARCH_IMPL_REV,
            "seeds": ["pick_workspace_ready", "home"],
            "yaw_candidates_count": 2,
            "depth_probe": 0.005,
        }
    )
    assert "[CHIPS_CAN_FINAL_DESCEND_POSTURE_SEARCH_START]" in msg
    assert CHIPS_CAN_POSTURE_SEARCH_IMPL_REV in msg
    assert "depth_probe=0.0050" in msg


def test_summarize_posture_search_exhausted() -> None:
    probes = [
        {"full_route_ok": False, "low_to_grasp_fraction": 0.63, "ik_seed_name": "home", "commanded_yaw_rad": 3.14},
        {"full_route_ok": False, "low_to_grasp_fraction": 0.49, "ik_seed_name": "pick_workspace_ready", "commanded_yaw_rad": 0.0},
    ]
    summary = summarize_chips_can_posture_search(probes, fraction_threshold=0.95)
    assert summary["total_variants"] == 2
    assert summary["ok_variants"] == 0
    assert summary["result"] == "EXHAUSTED"
    assert summary["best_fraction"] == 0.63
    assert summary["best_seed"] == "home"
