"""Tests política legacy chips_can."""

from panda_controller.chips_can_legacy_success_policy import (
    CHIPS_CAN_HIGH_TO_LOW_TF_TOLERANCE_MAX_M,
    CHIPS_CAN_LEGACY_GOLDEN_LAYOUT_VERSION,
    CHIPS_CAN_LEGACY_GOLDEN_SCENE_ID,
    CHIPS_CAN_LEGACY_GOLDEN_TARGET_LABEL,
    CHIPS_CAN_LEGACY_GRASP_DEPTH_FROM_TOP_M,
    CHIPS_CAN_LEGACY_HISTORICAL_SUCCESS_REFERENCE,
    CHIPS_CAN_LEGACY_PREGRASP_HEIGHT_ABOVE_TOP_M,
    OK_CHIPS_LEGACY_PENDING_ACTUAL_TF_DESCEND,
    chips_can_legacy_final_descend_m,
    chips_can_legacy_grasp_tcp_z,
    chips_can_legacy_pending_variant_passes,
    chips_can_legacy_pregrasp_tcp_z,
    format_chips_can_high_to_low_tf_tolerance_log,
    format_chips_can_legacy_high_to_low_execute_log,
    format_chips_can_legacy_low_pregrasp_state_refresh_log,
    format_chips_can_low_pregrasp_refresh_from_actual_tf_log,
    format_chips_can_legacy_target_collision_required_for_approach_log,
    format_chips_can_legacy_target_collision_restore_log,
    chips_can_legacy_high_to_low_borderline_ok,
    chips_can_legacy_high_to_low_cache_reusable,
    chips_can_legacy_low_pregrasp_state_refresh_ok,
    chips_can_legacy_pre_descend_clearance_ok,
    evaluate_chips_can_high_to_low_tf_tolerance,
    evaluate_chips_can_legacy_pre_descend_pose_gate,
    format_chips_can_legacy_high_to_low_accept_borderline_log,
    format_chips_can_legacy_high_to_low_prevalidate_reuse_log,
    format_chips_can_legacy_pre_descend_pose_gate_log,
    format_chips_can_micro_descend_inter_segment_verify_log,
    select_chips_can_legacy_success_policy_variant,
)


def test_legacy_pregrasp_and_grasp_z() -> None:
    assert CHIPS_CAN_LEGACY_GOLDEN_SCENE_ID == "demo_scene_02"
    assert CHIPS_CAN_LEGACY_GOLDEN_TARGET_LABEL == "chips_can"
    assert CHIPS_CAN_LEGACY_GOLDEN_LAYOUT_VERSION == "v3_clear_table_transport"
    top = 0.510
    pre_h = 0.025
    depth = 0.033
    assert chips_can_legacy_pregrasp_tcp_z(top_z_m=top, pregrasp_height_above_top_m=pre_h) == 0.535
    assert chips_can_legacy_grasp_tcp_z(top_z_m=top, depth_from_top_m=depth) == 0.477


def test_legacy_final_descend_short() -> None:
    assert chips_can_legacy_final_descend_m(
        pregrasp_height_above_top_m=0.025,
        depth_from_top_m=0.033,
    ) == 0.058


def test_pending_variant_passes_without_full_route() -> None:
    probe = {
        "object_high_plan_ok": True,
        "object_high_to_low_fraction": 1.0,
        "low_to_grasp_fraction": 0.16667,
        "pregrasp_height_above_top_m": 0.045,
        "full_route_ok": False,
    }
    assert chips_can_legacy_pending_variant_passes(probe, fraction_threshold=0.95)


def test_pending_variant_rejects_low_high_to_low_fraction() -> None:
    probe = {
        "object_high_plan_ok": True,
        "object_high_to_low_fraction": 0.60,
        "low_to_grasp_fraction": 0.99,
        "pregrasp_height_above_top_m": 0.035,
    }
    assert not chips_can_legacy_pending_variant_passes(probe, fraction_threshold=0.95)


def test_select_legacy_pending_prefers_lower_pregrasp() -> None:
    variants = [
        {
            "object_high_plan_ok": True,
            "object_high_to_low_fraction": 1.0,
            "low_to_grasp_fraction": 0.99,
            "pregrasp_height_above_top_m": 0.045,
            "depth_from_top_m": 0.035,
            "joint_dist": 0.5,
            "full_route_ok": True,
        },
        {
            "object_high_plan_ok": True,
            "object_high_to_low_fraction": 1.0,
            "low_to_grasp_fraction": 0.15,
            "pregrasp_height_above_top_m": 0.025,
            "depth_from_top_m": 0.030,
            "joint_dist": 1.0,
            "full_route_ok": False,
        },
    ]
    selected = select_chips_can_legacy_success_policy_variant(
        variants, fraction_threshold=0.95
    )
    assert selected is not None
    assert selected["pregrasp_height_above_top_m"] == 0.025
    assert float(selected["low_to_grasp_fraction"]) == 0.15


def test_historical_reference_matches_documented_success() -> None:
    assert CHIPS_CAN_LEGACY_HISTORICAL_SUCCESS_REFERENCE["pregrasp_height_above_top_m"] == 0.025
    assert CHIPS_CAN_LEGACY_HISTORICAL_SUCCESS_REFERENCE["recommended_grasp_depth_from_top_m"] == 0.033
    assert 0.025 in CHIPS_CAN_LEGACY_PREGRASP_HEIGHT_ABOVE_TOP_M
    assert 0.030 in CHIPS_CAN_LEGACY_GRASP_DEPTH_FROM_TOP_M
    assert OK_CHIPS_LEGACY_PENDING_ACTUAL_TF_DESCEND


def test_legacy_high_to_low_execute_log_contains_required_fields() -> None:
    log = format_chips_can_legacy_high_to_low_execute_log(
        {
            "high_tcp_z": 0.660,
            "low_tcp_z": 0.545,
            "expected_delta_z": 0.115,
            "plan_fraction": 1.0,
            "actual_tcp_z_before": 0.660,
            "actual_tcp_z_after": 0.545,
            "actual_tcp_error_m": 0.001,
            "result": "OK",
        }
    )
    assert "[CHIPS_CAN_LEGACY_HIGH_TO_LOW_EXECUTE]" in log
    assert "stage=object_high_to_legacy_low_pregrasp" in log
    assert "result=OK" in log


def test_legacy_low_pregrasp_state_refresh_log_contains_required_fields() -> None:
    log = format_chips_can_legacy_low_pregrasp_state_refresh_log(
        {
            "current_tcp": (0.520, -0.095, 0.545),
            "expected_tcp": (0.520, -0.095, 0.545),
            "tcp_error_m": 0.0,
            "current_js": "[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]",
            "result": "OK",
        }
    )
    assert "[CHIPS_CAN_LEGACY_LOW_PREGRASP_STATE_REFRESH]" in log
    assert "result=OK" in log


def test_legacy_target_collision_restore_log() -> None:
    log = format_chips_can_legacy_target_collision_restore_log(
        {
            "target_id": "target_runtime_ycb_chips_can_1",
            "before_present": False,
            "after_present": True,
            "result": "OK",
        }
    )
    assert "[LEGACY_TARGET_COLLISION_RESTORE_AFTER_DIAGNOSTIC_PROBE]" in log
    assert "after_present=true" in log


def test_legacy_target_collision_required_for_approach_log() -> None:
    log = format_chips_can_legacy_target_collision_required_for_approach_log(
        {
            "target_collision_present": True,
            "approach_guard_present": True,
            "result": "OK",
        }
    )
    assert "[LEGACY_TARGET_COLLISION_REQUIRED_FOR_APPROACH]" in log
    assert "target_collision_present=true" in log


def test_legacy_pre_descend_pose_gate_accepts_low_clearance() -> None:
    ok, fields = evaluate_chips_can_legacy_pre_descend_pose_gate(
        actual_tcp_z=0.545,
        top_z_m=0.510,
        legacy_low_pregrasp_tcp_z=0.545,
        centering_ok=True,
        gripper_open_ok=True,
        disturbance_ok=True,
        target_collision_removed_ok=True,
        descend_route_prepared_ok=True,
        tcp_error_tolerance_m=0.015,
    )
    assert ok is True
    assert fields["legacy_low_pregrasp_ok"] is True
    assert abs(float(fields["clearance_above_top"]) - 0.035) < 1e-6


def test_legacy_pre_descend_pose_gate_accepts_borderline_clearance() -> None:
    ok, fields = evaluate_chips_can_legacy_pre_descend_pose_gate(
        actual_tcp_z=0.5585,
        top_z_m=0.510,
        legacy_low_pregrasp_tcp_z=0.545,
        centering_ok=True,
        gripper_open_ok=True,
        disturbance_ok=True,
        target_collision_removed_ok=True,
        descend_route_prepared_ok=True,
        tcp_error_tolerance_m=0.015,
    )
    assert ok is True
    assert fields["legacy_low_pregrasp_ok"] is True
    assert abs(float(fields["clearance_above_top"]) - 0.0485) < 1e-6


def test_legacy_high_to_low_borderline_ok() -> None:
    ok, z_error, clearance, reason = chips_can_legacy_high_to_low_borderline_ok(
        requested_low_tcp_z=0.5450,
        actual_tcp_z_after=0.5585,
        top_z_m=0.5100,
    )
    assert ok is True
    assert abs(z_error - 0.0135) < 1e-6
    assert abs(clearance - 0.0485) < 1e-6
    assert reason == "tf_z_above_request_but_inside_legacy_low_range"


def test_legacy_high_to_low_borderline_rejects_at_target() -> None:
    ok, _, _, reason = chips_can_legacy_high_to_low_borderline_ok(
        requested_low_tcp_z=0.5450,
        actual_tcp_z_after=0.5450,
        top_z_m=0.5100,
    )
    assert ok is False
    assert reason == "not_above_requested_low"


def test_high_to_low_tf_tolerance_accepts_demo_scene_02_borderline() -> None:
    ok, z_error, reason = evaluate_chips_can_high_to_low_tf_tolerance(
        requested_tcp_z=0.5450,
        actual_tcp_z_after=0.5576,
    )
    assert ok is True
    assert abs(z_error - 0.0126) < 1e-6
    assert reason == "tf_z_above_request_within_tolerance"


def test_high_to_low_tf_tolerance_rejects_below_requested() -> None:
    ok, _, reason = evaluate_chips_can_high_to_low_tf_tolerance(
        requested_tcp_z=0.5450,
        actual_tcp_z_after=0.5400,
    )
    assert ok is False
    assert reason == "actual_below_requested_contact_risk"


def test_high_to_low_tf_tolerance_rejects_above_max() -> None:
    ok, _, reason = evaluate_chips_can_high_to_low_tf_tolerance(
        requested_tcp_z=0.5450,
        actual_tcp_z_after=0.5660,
    )
    assert ok is False
    assert reason == "actual_above_requested_plus_tolerance"


def test_high_to_low_tf_tolerance_log() -> None:
    log = format_chips_can_high_to_low_tf_tolerance_log(
        {
            "requested_tcp_z": 0.5450,
            "actual_tcp_z_after": 0.5576,
            "z_error_m": 0.0126,
            "reason": "tf_z_above_request_within_tolerance",
            "result": "OK",
        }
    )
    assert "[CHIPS_CAN_HIGH_TO_LOW_TF_TOLERANCE]" in log
    assert "actual_tcp_z_after=0.5576" in log
    assert "max_tolerance_m=%.4f" % CHIPS_CAN_HIGH_TO_LOW_TF_TOLERANCE_MAX_M in log
    assert "result=OK" in log


def test_low_pregrasp_refresh_from_actual_tf_log() -> None:
    log = format_chips_can_low_pregrasp_refresh_from_actual_tf_log(
        {
            "current_tcp": (0.52, -0.095, 0.5576),
            "expected_tcp": (0.52, -0.095, 0.5450),
            "tcp_error_m": 0.0126,
            "current_js": "[0.0000]",
            "borderline": True,
            "result": "OK",
        }
    )
    assert "[CHIPS_CAN_LOW_PREGRASP_REFRESH_FROM_ACTUAL_TF]" in log
    assert "borderline=true" in log
    assert "result=OK" in log


def test_legacy_low_pregrasp_state_refresh_legacy_clearance_borderline() -> None:
    ok, tcp_err = chips_can_legacy_low_pregrasp_state_refresh_ok(
        current_tcp=(0.0, 0.0, 0.5585),
        expected_tcp=(0.0, 0.0, 0.5450),
        tcp_error_threshold_m=0.012,
        accept_borderline_low_pregrasp=True,
        top_z_m=0.5100,
    )
    assert ok is True
    assert tcp_err is not None


def test_legacy_low_pregrasp_state_refresh_borderline() -> None:
    ok, tcp_err = chips_can_legacy_low_pregrasp_state_refresh_ok(
        current_tcp=(0.0, 0.0, 0.5585),
        expected_tcp=(0.0, 0.0, 0.5450),
        tcp_error_threshold_m=0.015,
        accept_borderline_low_pregrasp=True,
        top_z_m=0.5100,
    )
    assert ok is True
    assert tcp_err is not None
    assert abs(float(tcp_err) - 0.0135) < 1e-6


def test_legacy_low_pregrasp_state_refresh_borderline_without_top_z() -> None:
    ok, tcp_err = chips_can_legacy_low_pregrasp_state_refresh_ok(
        current_tcp=(0.0, 0.0, 0.5576),
        expected_tcp=(0.0, 0.0, 0.5450),
        tcp_error_threshold_m=0.012,
        accept_borderline_low_pregrasp=True,
        top_z_m=None,
    )
    assert ok is True
    assert tcp_err is not None
    assert abs(float(tcp_err) - 0.0126) < 1e-6


def test_legacy_high_to_low_accept_borderline_log() -> None:
    log = format_chips_can_legacy_high_to_low_accept_borderline_log(
        {
            "requested_low_tcp_z": 0.5450,
            "actual_tcp_z_after": 0.5585,
            "z_error_m": 0.0135,
            "top_z": 0.5100,
            "actual_clearance_above_top": 0.0485,
            "reason": "tf_z_above_request_but_inside_legacy_low_range",
        }
    )
    assert "[CHIPS_CAN_LEGACY_HIGH_TO_LOW_ACCEPT_BORDERLINE]" in log
    assert "actual_tcp_z_after=0.5585" in log
    assert "allowed_clearance_range=[0.020, 0.055]" in log
    assert "result=OK" in log


def test_legacy_pre_descend_pose_gate_rejects_high_route_clearance_requirement() -> None:
    ok, fields = evaluate_chips_can_legacy_pre_descend_pose_gate(
        actual_tcp_z=0.610,
        top_z_m=0.510,
        legacy_low_pregrasp_tcp_z=0.545,
        centering_ok=True,
        gripper_open_ok=True,
        disturbance_ok=True,
        target_collision_removed_ok=True,
        descend_route_prepared_ok=True,
        tcp_error_tolerance_m=0.015,
    )
    assert ok is False
    assert fields["legacy_low_pregrasp_ok"] is False


def test_legacy_pre_descend_pose_gate_rejects_too_low() -> None:
    ok, clearance, reason = chips_can_legacy_pre_descend_clearance_ok(
        actual_tcp_z=0.523,
        top_z_m=0.510,
    )
    assert ok is False
    assert clearance < 0.020
    assert reason == "legacy_pre_descend_too_low"


def test_legacy_pre_descend_pose_gate_log_mode() -> None:
    _, fields = evaluate_chips_can_legacy_pre_descend_pose_gate(
        actual_tcp_z=0.545,
        top_z_m=0.510,
        legacy_low_pregrasp_tcp_z=0.545,
        centering_ok=True,
        gripper_open_ok=True,
        disturbance_ok=True,
        target_collision_removed_ok=True,
        descend_route_prepared_ok=True,
        tcp_error_tolerance_m=0.015,
    )
    fields["ok"] = True
    log = format_chips_can_legacy_pre_descend_pose_gate_log(fields)
    assert "mode=legacy_successful_pick" in log
    assert "result=OK" in log
    assert "allowed_clearance_range=[0.020, 0.055]" in log


def test_legacy_high_to_low_cache_reusable() -> None:
    assert chips_can_legacy_high_to_low_cache_reusable(
        cached_fraction=1.0,
        cached_high_js=object(),
        cached_low_js=object(),
        fraction_threshold=0.95,
    )
    assert not chips_can_legacy_high_to_low_cache_reusable(
        cached_fraction=0.19,
        cached_high_js=object(),
        cached_low_js=object(),
        fraction_threshold=0.95,
    )
    assert not chips_can_legacy_high_to_low_cache_reusable(
        cached_fraction=1.0,
        cached_high_js=None,
        cached_low_js=object(),
        fraction_threshold=0.95,
    )


def test_legacy_high_to_low_prevalidate_reuse_log() -> None:
    log = format_chips_can_legacy_high_to_low_prevalidate_reuse_log(
        {
            "cached_fraction": 1.0,
            "cached_yaw": 3.14,
            "cached_js_available": True,
            "revalidate_fraction": 0.19,
            "result": "OK",
        }
    )
    assert "[CHIPS_CAN_LEGACY_HIGH_TO_LOW_PREVALIDATE_REUSE]" in log
    assert "cached_fraction=1.00000" in log
    assert "result=OK" in log


def test_micro_descend_inter_segment_verify_log() -> None:
    log = format_chips_can_micro_descend_inter_segment_verify_log(
        {
            "step_idx": 1,
            "actual_tcp_z": 0.5295,
            "previous_target_tcp_z": 0.5450,
            "next_target_tcp_z": 0.4983,
            "z_progress_ok": True,
            "xy_drift_ok": True,
            "centering_ok": True,
            "gripper_open_ok": True,
            "gap_ok": True,
            "segment_tcp_ok": True,
            "tcp_ok_mode": "segment_progress_not_pregrasp",
            "result": "OK",
            "reason": "inter_segment_verify_ok",
        }
    )
    assert "[CHIPS_CAN_MICRO_DESCEND_INTER_SEGMENT_VERIFY]" in log
    assert "tcp_ok_mode=segment_progress_not_pregrasp" in log
    assert "result=OK" in log
