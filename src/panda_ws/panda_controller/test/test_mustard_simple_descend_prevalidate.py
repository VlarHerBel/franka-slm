"""Tests prevalidación descenso mustard_bottle."""

from panda_controller.mustard_simple_descend_prevalidate import (
    build_mustard_descend_candidate_specs,
    cartesian_accept,
    evaluate_mustard_contact_depth_policy,
    format_mustard_endpoint_ik_scene_log,
    format_mustard_grasp_endpoint_ik_validate_log,
    format_mustard_simple_descend_prevalidate_log,
    interpolate_vertical_tcp,
)


def test_build_descend_candidates_conservative_first() -> None:
    pre = (0.662, 0.084, 0.492)
    specs = build_mustard_descend_candidate_specs(
        pre_plan=pre,
        top_z_m=0.427,
        xy=(0.662, 0.084),
        min_grasp_z_m=0.400,
    )
    assert specs
    deltas = [float(s["descend_delta_m"]) for s in specs]
    assert deltas == sorted(deltas)
    assert float(specs[0]["descend_delta_m"]) <= 0.075


def test_interpolate_vertical_tcp() -> None:
    pre = (0.1, 0.2, 0.5)
    gr = (0.1, 0.2, 0.4)
    mid = interpolate_vertical_tcp(pre, gr, 0.5)
    assert mid == (0.1, 0.2, 0.45)


def test_contact_depth_policy() -> None:
    ok, _ = evaluate_mustard_contact_depth_policy(
        depth_from_top_m=0.015,
        insertion_depth_limit_m=0.030,
    )
    assert ok
    fail, reason = evaluate_mustard_contact_depth_policy(
        depth_from_top_m=0.035,
        insertion_depth_limit_m=0.030,
    )
    assert not fail
    assert "insertion_limit" in reason


def test_cartesian_accept() -> None:
    assert cartesian_accept(
        fraction=0.96,
        threshold=0.95,
        traj_points=4,
        start_state_honored=True,
    )
    assert not cartesian_accept(
        fraction=0.16667,
        threshold=0.95,
        traj_points=4,
        start_state_honored=True,
    )


def test_mustard_simple_descend_log_format() -> None:
    log = format_mustard_simple_descend_prevalidate_log(
        {
            "pregrasp_tcp": "(0.662, 0.084, 0.492)",
            "grasp_tcp": "(0.662, 0.084, 0.462)",
            "descend_delta": "0.030",
            "cartesian_fraction": "1.00000",
            "endpoint_ik_ok": "true",
            "selected_depth_from_top": "0.015",
            "result": "OK",
            "reason": "vertical_descend_prevalidated",
        }
    )
    assert "[MUSTARD_SIMPLE_DESCEND_PREVALIDATE]" in log
    assert "selected_depth_from_top=0.015" in log


def test_mustard_grasp_endpoint_ik_validate_log_format() -> None:
    log = format_mustard_grasp_endpoint_ik_validate_log(
        {
            "endpoint_seed_source": "pregrasp_plan_traj",
            "endpoint_seed_pregrasp_tcp_z": "0.4909",
            "endpoint_seed_joints": "j1=0.0",
            "target_tcp": "(0.659, 0.060, 0.4274)",
            "target_hand": "(0.659, 0.060, 0.5274)",
            "quaternion": "(0, 1, 0, 0)",
            "commanded_tcp_yaw_rad": "-3.073189",
            "target_collision_present": "false",
            "pregrasp_tcp": "(0.659, 0.060, 0.4909)",
            "grasp_tcp": "(0.659, 0.060, 0.4269)",
            "endpoint_ik_ok": "true",
            "plan_to_endpoint_ok": "true",
            "result": "OK",
        }
    )
    assert "[MUSTARD_GRASP_ENDPOINT_IK_VALIDATE]" in log
    assert "endpoint_seed_source=pregrasp_plan_traj" in log
    assert "endpoint_seed_pregrasp_tcp_z=0.4909" in log
    assert "target_collision_present=false" in log


def test_mustard_endpoint_ik_scene_log_format() -> None:
    log = format_mustard_endpoint_ik_scene_log(
        {
            "target_collision_present_before": "true",
            "target_collision_present_during_endpoint": "false",
            "obstacles_present": ["sugar_box"],
            "result": "OK",
        }
    )
    assert "[MUSTARD_ENDPOINT_IK_SCENE]" in log
    assert "target_collision_present_before=true" in log
    assert "target_collision_present_during_endpoint=false" in log
    assert "obstacles_present=[sugar_box]" in log
    assert "result=OK" in log
