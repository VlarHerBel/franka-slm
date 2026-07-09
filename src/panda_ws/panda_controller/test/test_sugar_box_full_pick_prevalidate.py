"""Tests: evaluación candidato pick completo sugar_box."""

from panda_controller.sugar_box_full_pick_prevalidate import (
    SUGAR_BOX_OFFLINE_CENTERING_TCP_ACCEPT_M,
    centering_xy_error_m,
    compute_sugar_box_full_pick_score,
    evaluate_sugar_box_offline_centering,
    format_full_pick_route_no_candidate_log,
    format_sugar_box_descend_scene_contract_violation_log,
    format_sugar_box_endpoint_orientation_contract_log,
    format_sugar_box_full_pick_candidate_eval_log,
    format_sugar_box_full_pick_descend_scene_log,
    format_sugar_box_offline_centering_diag_log,
    is_sugar_box_full_pick_candidate_complete,
    select_sugar_box_full_pick_candidate,
)
from panda_controller.sugar_box_depth_search import (
    SUGAR_BOX_DEMO_MAX_CANDIDATES,
    build_sugar_box_demo_depth_descend_tcp_specs,
    prioritize_sugar_box_demo_yaw_variants,
)


def test_complete_candidate_requires_all_flags() -> None:
    ok = {
        "pregrasp_plan_ok": True,
        "joint7_correction_virtual_ok": True,
        "gap_after_joint7_ok": True,
        "centering_after_joint7_ok": True,
        "descend_first_step_ok": True,
        "descend_full_or_guarded_ok": True,
        "result": "OK",
    }
    assert is_sugar_box_full_pick_candidate_complete(ok)
    incomplete = dict(ok)
    incomplete["descend_full_or_guarded_ok"] = False
    assert not is_sugar_box_full_pick_candidate_complete(incomplete)


def test_complete_candidate_accepts_micro_descend_fallback() -> None:
    micro_ok = {
        "pregrasp_plan_ok": True,
        "joint7_virtual_ok": True,
        "gap_after_joint7_ok": True,
        "centering_after_joint7_ok": True,
        "descend_first_step_ok": False,
        "descend_full_or_guarded_ok": False,
        "guarded_ik_first_step_ok": False,
        "micro_descend_ok": True,
        "selected_micro_descend_m": 0.047,
        "result": "OK",
    }
    assert is_sugar_box_full_pick_candidate_complete(micro_ok)


def test_select_prefers_deeper_and_higher_fraction() -> None:
    shallow = {
        "pregrasp_plan_ok": True,
        "joint7_correction_virtual_ok": True,
        "gap_after_joint7_ok": True,
        "centering_after_joint7_ok": True,
        "descend_first_step_ok": True,
        "descend_full_or_guarded_ok": True,
        "result": "OK",
        "depth_from_top_m": 0.010,
        "descend_cartesian_fraction": 0.96,
        "gap_after_joint7_deg": 2.0,
        "centering_after_joint7_xy": 0.001,
    }
    deep = dict(shallow)
    deep["depth_from_top_m"] = 0.022
    deep["descend_cartesian_fraction"] = 0.98
    picked = select_sugar_box_full_pick_candidate([shallow, deep])
    assert picked is not None
    assert float(picked["depth_from_top_m"]) == 0.022


def test_score_not_joint_dist() -> None:
    low_dist = compute_sugar_box_full_pick_score(
        {
            "depth_from_top_m": 0.012,
            "descend_cartesian_fraction": 0.99,
            "gap_after_joint7_deg": 1.0,
            "centering_after_joint7_xy": 0.001,
        }
    )
    high_quality = compute_sugar_box_full_pick_score(
        {
            "depth_from_top_m": 0.020,
            "descend_cartesian_fraction": 0.99,
            "gap_after_joint7_deg": 0.5,
            "centering_after_joint7_xy": 0.0005,
        }
    )
    assert high_quality > low_dist


def test_eval_log_format() -> None:
    log = format_sugar_box_full_pick_candidate_eval_log(
        {
            "candidate_id": 3,
            "yaw_variant": "top_down_yaw_pi",
            "pregrasp_tcp_z": 0.468,
            "grasp_tcp_z": 0.413,
            "pregrasp_plan_ok": True,
            "joint7_virtual_ok": True,
            "gap_after_joint7_deg": 1.2,
            "centering_after_joint7_xy": 0.002,
            "descend_cartesian_fraction": 0.97,
            "guarded_ik_first_step_ok": False,
            "guarded_ik_full_ok": False,
            "score": "(0.022, 0.97)",
            "result": "OK",
            "reject_reason": "",
        }
    )
    assert "[SUGAR_BOX_FULL_PICK_CANDIDATE_EVAL]" in log
    assert "yaw_variant=top_down_yaw_pi" in log


def test_no_candidate_log() -> None:
    log = format_full_pick_route_no_candidate_log()
    assert "no_full_pick_candidate" in log
    assert "ABORT_IN_HOME" in log


def test_centering_xy_error() -> None:
    from panda_controller.sugar_box_full_pick_prevalidate import centering_xy_error_m

    assert centering_xy_error_m((0.0, 0.0), (0.003, 0.004)) == 0.005


def test_offline_centering_accepts_tcp_within_3mm() -> None:
    ok, tcp_err, finger_err, source = evaluate_sugar_box_offline_centering(
        fk_tcp_xy=(0.630, -0.175),
        target_center_xy=(0.631, -0.175),
        finger_midpoint_xy=(0.645, -0.175),
        z_err_m=0.001,
    )
    assert ok
    assert tcp_err <= SUGAR_BOX_OFFLINE_CENTERING_TCP_ACCEPT_M
    assert source == "tcp"
    assert finger_err is not None and finger_err > tcp_err


def test_offline_centering_diag_log() -> None:
    log = format_sugar_box_offline_centering_diag_log(
        {
            "candidate_id": 2,
            "fk_tcp_x": 0.630,
            "fk_tcp_y": -0.175,
            "target_center_x": 0.630,
            "target_center_y": -0.175,
            "finger_midpoint_xy": (0.640, -0.175),
            "tcp_error_xy": 0.0005,
            "finger_midpoint_error_xy": 0.010,
            "centering_source": "tcp",
            "result": "OK",
        }
    )
    assert "[SUGAR_BOX_OFFLINE_CENTERING_DIAG]" in log
    assert "centering_source=tcp" in log


def test_descend_scene_log_and_contract_violation() -> None:
    orient_log = format_sugar_box_endpoint_orientation_contract_log(
        {
            "candidate_idx": 3,
            "yaw_variant": "yaw_pi",
            "desired_hand_quat": (0.0, 1.0, 0.0, 0.0),
            "fk_hand_quat": (0.05, 0.99, 0.0, 0.0),
            "hand_orientation_error_deg": 5.7,
            "expected_hand_to_tcp_base": (0.0, 0.0, -0.1),
            "actual_hand_to_tcp_base": (0.02, -0.01, -0.08),
            "tcp_error_xy": 0.019,
            "result": "FAIL",
            "reject_reason": "pregrasp_endpoint_orientation_contract_fail",
        }
    )
    assert "[SUGAR_BOX_ENDPOINT_ORIENTATION_CONTRACT]" in orient_log
    scene = format_sugar_box_full_pick_descend_scene_log(
        {
            "target_collision_present": False,
            "obstacles": ["mustard_bottle"],
            "result": "OK",
        }
    )
    assert "[SUGAR_BOX_FULL_PICK_DESCEND_SCENE]" in scene
    assert "target_collision_present=false" in scene
    viol = format_sugar_box_descend_scene_contract_violation_log()
    assert "[SUGAR_BOX_DESCEND_SCENE_CONTRACT_VIOLATION]" in viol


def test_demo_grid_max_72_candidates() -> None:
    specs = build_sugar_box_demo_depth_descend_tcp_specs(
        xy=(0.630, -0.175), top_z_m=0.435
    )
    assert len(specs) <= 12
    ranked = [
        ("top_down_yaw_zero", None, 0.0, None, 1.0),
        ("top_down_yaw_pi", None, 0.0, None, 0.5),
        ("commanded_yaw", None, 0.0, None, 0.2),
    ]
    ordered = prioritize_sugar_box_demo_yaw_variants(ranked)
    assert ordered[0][0] == "top_down_yaw_pi"
    assert ordered[-1][0] == "top_down_yaw_zero"
    assert SUGAR_BOX_DEMO_MAX_CANDIDATES == 72
