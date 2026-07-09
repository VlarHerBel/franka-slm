"""Tests microdescenso sugar_box demo_scene_02 (sin ROS)."""

from panda_controller.sugar_box_micro_descend import (
    SUGAR_BOX_MICRO_DESCEND_DEFAULT_BACKEND,
    build_sugar_box_dynamic_micro_descend_candidates_m,
    build_sugar_box_micro_descend_ik_step_targets,
    evaluate_sugar_box_micro_descend_fk,
    evaluate_sugar_box_micro_descend_ik_step_fk,
    evaluate_sugar_box_micro_descend_ik_success,
    format_sugar_box_micro_descend_execute_log,
    format_sugar_box_micro_descend_policy_log,
    normalize_sugar_box_micro_descend_backend,
    ordered_sugar_box_micro_descend_candidates_m,
    sugar_box_micro_descend_eligible,
)


def test_micro_descend_eligible_only_sugar_demo() -> None:
    assert sugar_box_micro_descend_eligible(
        label="sugar_box", scene_id="demo_scene_02", enabled=True
    )
    assert not sugar_box_micro_descend_eligible(
        label="cracker_box", scene_id="demo_scene_02", enabled=True
    )
    assert not sugar_box_micro_descend_eligible(
        label="sugar_box", scene_id="demo_scene_01", enabled=True
    )


def test_ordered_candidates_deepest_first() -> None:
    assert ordered_sugar_box_micro_descend_candidates_m(
        [0.010, 0.025, 0.015, 0.020]
    ) == (0.025, 0.020, 0.015, 0.010)


def test_micro_descend_fk_pass_and_fail() -> None:
    ok, xy_err, depth, preferred, reject = evaluate_sugar_box_micro_descend_fk(
        fk_tcp=(0.630, -0.175, 0.460),
        target_center_xy=(0.630, -0.175),
        top_z=0.472,
        table_top_z=0.40,
        min_table_clearance_m=0.015,
    )
    assert ok
    assert xy_err <= 0.004
    assert depth is not None and depth >= 0.010
    assert preferred
    assert reject == ""

    fail_center, _, _, _, reject_center = evaluate_sugar_box_micro_descend_fk(
        fk_tcp=(0.636, -0.192, 0.460),
        target_center_xy=(0.630, -0.175),
        top_z=0.472,
        table_top_z=0.40,
        min_table_clearance_m=0.015,
    )
    assert not fail_center
    assert reject_center == "micro_descend_centering_fail"


def test_micro_descend_policy_log_format() -> None:
    log = format_sugar_box_micro_descend_policy_log(
        {
            "candidate_id": 1,
            "trigger_reason": "guarded_ik_first_step_fail",
            "enabled": True,
            "gated_out_reason": "",
            "full_descend_m": 0.050,
            "selected_micro_descend_m": 0.047,
            "requested_micro_descend_m": 0.047,
            "pregrasp_tcp_z": 0.472,
            "final_tcp_z": 0.425,
            "top_z": 0.435,
            "depth_below_top": 0.010,
            "centering_xy": 0.0009,
            "cartesian_fraction": 0.98,
            "ik_ok": True,
            "fk_ok": True,
            "candidates_m": [0.047, 0.043, 0.039, 0.035],
            "result": "OK",
            "reject_reason": "",
        }
    )
    assert "[SUGAR_BOX_MICRO_DESCEND_POLICY]" in log
    assert "trigger_reason=guarded_ik_first_step_fail" in log
    assert "selected_micro_descend_m=0.0470" in log
    assert "result=OK" in log


def test_dynamic_micro_descend_candidates_from_pregrasp_and_top() -> None:
    candidates = build_sugar_box_dynamic_micro_descend_candidates_m(
        pregrasp_tcp_z=0.472,
        top_z=0.435,
    )
    assert len(candidates) == 4
    assert abs(candidates[0] - 0.059) < 1e-6
    assert abs(candidates[-1] - 0.047) < 1e-6
    assert candidates == ordered_sugar_box_micro_descend_candidates_m(candidates)


def test_normalize_backend_defaults_to_ik_stepwise() -> None:
    assert normalize_sugar_box_micro_descend_backend("ik_stepwise") == "ik_stepwise"
    assert normalize_sugar_box_micro_descend_backend("cartesian_path") == "cartesian_path"
    assert (
        normalize_sugar_box_micro_descend_backend("direct_joint_endpoint")
        == "direct_joint_endpoint"
    )
    assert (
        normalize_sugar_box_micro_descend_backend("garbage")
        == SUGAR_BOX_MICRO_DESCEND_DEFAULT_BACKEND
    )
    assert normalize_sugar_box_micro_descend_backend(None) == "ik_stepwise"


def test_ik_step_targets_accumulate_and_end_at_requested() -> None:
    targets = build_sugar_box_micro_descend_ik_step_targets(0.047, step_dz_m=0.005)
    assert targets[0] == 0.005
    assert abs(targets[-1] - 0.047) < 1e-9
    # estrictamente creciente
    assert all(b > a for a, b in zip(targets, targets[1:]))
    # ningún paso supera requested
    assert all(t <= 0.047 + 1e-9 for t in targets)
    # paso único cuando step >= requested (direct_joint_endpoint)
    assert build_sugar_box_micro_descend_ik_step_targets(0.047, step_dz_m=1.0) == (0.047,)
    assert build_sugar_box_micro_descend_ik_step_targets(0.0) == ()


def test_ik_step_fk_pass_and_reject_reasons() -> None:
    ok, errs, reject = evaluate_sugar_box_micro_descend_ik_step_fk(
        fk_tcp=(0.630, -0.175, 0.430),
        start_tcp_xy=(0.630, -0.175),
        target_center_xy=(0.630, -0.175),
        target_tcp_z=0.430,
        orientation_error_deg=1.0,
        table_top_z=0.40,
        min_table_clearance_m=0.015,
    )
    assert ok
    assert reject == ""
    assert errs["tcp_error_z"] < 1e-6

    _bad_z, _e, reject_z = evaluate_sugar_box_micro_descend_ik_step_fk(
        fk_tcp=(0.630, -0.175, 0.440),
        start_tcp_xy=(0.630, -0.175),
        target_center_xy=(0.630, -0.175),
        target_tcp_z=0.430,
        orientation_error_deg=1.0,
        table_top_z=0.40,
        min_table_clearance_m=0.015,
    )
    assert reject_z == "ik_step_z_error"

    _bad_o, _e2, reject_o = evaluate_sugar_box_micro_descend_ik_step_fk(
        fk_tcp=(0.630, -0.175, 0.430),
        start_tcp_xy=(0.630, -0.175),
        target_center_xy=(0.630, -0.175),
        target_tcp_z=0.430,
        orientation_error_deg=9.0,
        table_top_z=0.40,
        min_table_clearance_m=0.015,
    )
    assert reject_o == "ik_step_orientation_error"


def test_ik_success_criteria() -> None:
    ok, reject = evaluate_sugar_box_micro_descend_ik_success(
        depth_below_top=0.012,
        centering_xy=0.002,
        final_tcp_z=0.425,
        table_top_z=0.40,
        min_table_clearance_m=0.015,
        all_steps_ok=True,
        target_collision_present=False,
    )
    assert ok and reject == ""

    fail_depth, reject_depth = evaluate_sugar_box_micro_descend_ik_success(
        depth_below_top=0.005,
        centering_xy=0.002,
        final_tcp_z=0.425,
        table_top_z=0.40,
        min_table_clearance_m=0.015,
        all_steps_ok=True,
        target_collision_present=False,
    )
    assert not fail_depth
    assert reject_depth == "micro_descend_depth_below_top_fail"

    fail_steps, reject_steps = evaluate_sugar_box_micro_descend_ik_success(
        depth_below_top=0.012,
        centering_xy=0.002,
        final_tcp_z=0.425,
        table_top_z=0.40,
        min_table_clearance_m=0.015,
        all_steps_ok=False,
        target_collision_present=False,
    )
    assert not fail_steps
    assert reject_steps == "ik_step_fail"


def test_policy_log_includes_backend_and_fallback_fields() -> None:
    log = format_sugar_box_micro_descend_policy_log(
        {
            "candidate_id": 1,
            "trigger_reason": "guarded_ik_first_step_fail",
            "enabled": True,
            "backend": "ik_stepwise",
            "selected_micro_descend_m": 0.047,
            "requested_micro_descend_m": 0.047,
            "cartesian_fraction": 0.10526,
            "cartesian_fallback_used": True,
            "ik_steps_ok": 10,
            "final_tcp_z": 0.425,
            "depth_below_top": 0.010,
            "centering_xy": 0.0009,
            "result": "OK",
            "reject_reason": "",
        }
    )
    assert "backend=ik_stepwise" in log
    assert "cartesian_fallback_used=true" in log
    assert "ik_steps_ok=10" in log
    assert "result=OK" in log


def test_execute_log_format() -> None:
    log = format_sugar_box_micro_descend_execute_log(
        {
            "backend": "ik_stepwise",
            "steps": 10,
            "selected_micro_descend_m": 0.047,
            "final_tcp_z": 0.425,
            "result": "OK",
        }
    )
    assert "[SUGAR_BOX_MICRO_DESCEND_EXECUTE]" in log
    assert "backend=ik_stepwise" in log
    assert "steps=10" in log
    assert "result=OK" in log
