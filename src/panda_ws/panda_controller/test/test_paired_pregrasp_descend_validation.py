"""Tests: validación emparejada pregrasp + descenso cartesiano (demo_scene_02)."""

from panda_controller.demo_pick_route_preflight import (
    demo_full_pick_route_prevalidation_required,
    pick_route_preflight_allows_motion,
)
from panda_controller.paired_pregrasp_descend_validation import (
    PAIRED_REJECT_REASON_LIFT_AFTER_VALID_DESCEND,
    PAIRED_REJECT_REASON_NO_LIFT,
    PAIRED_REJECT_REASON_NO_CARTESIAN,
    PAIRED_REJECT_REASON_TRANSPORT_AFTER_VALID_LIFT,
    PAIRED_REJECT_REASON_NO_TRANSPORT_EXIT,
    build_paired_candidate_result,
    cartesian_descend_prevalidation_acceptable,
    compute_wrapped_joint_diffs,
    evaluate_paired_pregrasp_fk_contract,
    evaluate_pregrasp_execution_state_verify,
    evaluate_validated_joint7_runtime_match,
    paired_pregrasp_descend_validation_required,
    paired_preflight_fail_reason,
    select_paired_pregrasp_candidate,
)


def _scene02_candidate(**extra) -> dict:
    base = {
        "label": "cracker_box",
        "scene_id": "demo_scene_02",
        "scene_obstacles": [{"label": "chips_can"}],
    }
    base.update(extra)
    return base


def test_paired_validation_required_for_demo_scene_02_authoritative() -> None:
    assert paired_pregrasp_descend_validation_required(
        candidate=_scene02_candidate(),
        demo_authoritative_scene=True,
        scene_id="demo_scene_02",
    )
    assert not paired_pregrasp_descend_validation_required(
        candidate=_scene02_candidate(),
        demo_authoritative_scene=False,
        scene_id="demo_scene_02",
    )
    assert not paired_pregrasp_descend_validation_required(
        candidate=_scene02_candidate(scene_id="demo_scene_01"),
        demo_authoritative_scene=True,
        scene_id="demo_scene_01",
    )


def test_full_route_prevalidation_required_when_paired() -> None:
    assert demo_full_pick_route_prevalidation_required(
        candidate=_scene02_candidate(),
        demo_fast_mode=False,
        demo_motion_profile_active=False,
        require_param=False,
        chips_can_candidate=True,
        demo_authoritative_scene=True,
        scene_id="demo_scene_02",
    )


def test_cartesian_acceptable_rejects_geometric_fallback_when_paired() -> None:
    assert not cartesian_descend_prevalidation_acceptable(
        cartesian_ok=True,
        cartesian_fraction=1.0,
        fraction_threshold=0.95,
        prevalidation_source="geometric_fallback",
        paired_validation_required=True,
    )
    assert cartesian_descend_prevalidation_acceptable(
        cartesian_ok=True,
        cartesian_fraction=0.96,
        fraction_threshold=0.95,
        prevalidation_source="moveit",
        paired_validation_required=True,
    )


def test_cartesian_acceptable_allows_mustard_geometric_fallback_simple_direct_paired() -> (
    None
):
    assert cartesian_descend_prevalidation_acceptable(
        cartesian_ok=True,
        cartesian_fraction=0.0,
        fraction_threshold=0.95,
        prevalidation_source="geometric_fallback",
        paired_validation_required=True,
        label="mustard_bottle",
        simple_direct_route=True,
    )
    assert not cartesian_descend_prevalidation_acceptable(
        cartesian_ok=True,
        cartesian_fraction=0.0,
        fraction_threshold=0.95,
        prevalidation_source="geometric_fallback",
        paired_validation_required=True,
        label="cracker_box",
        simple_direct_route=True,
    )


def test_preflight_rejects_geometric_fallback_when_paired() -> None:
    ok, reason = pick_route_preflight_allows_motion(
        plan_before_result="OK_FULL_PICK_ROUTE_PREVALIDATED",
        cartesian_descend_prevalidated=True,
        full_route_required=True,
        cartesian_fraction=1.0,
        fraction_threshold=0.95,
        cartesian_descend_prevalidation_source="geometric_fallback",
        paired_validation_required=True,
    )
    assert not ok
    assert reason == "geometric_fallback_not_allowed_in_paired_pregrasp_validation"


def test_preflight_accepts_mustard_geometric_fallback_simple_direct_paired() -> None:
    ok, reason = pick_route_preflight_allows_motion(
        plan_before_result="OK_FULL_PICK_ROUTE_PREVALIDATED",
        cartesian_descend_prevalidated=True,
        full_route_required=True,
        cartesian_fraction=0.0,
        fraction_threshold=0.95,
        cartesian_descend_prevalidation_source="geometric_fallback",
        paired_validation_required=True,
        label="mustard_bottle",
        simple_direct_route=True,
    )
    assert ok, reason


def test_build_paired_accepts_mustard_geometric_fallback_simple_direct() -> None:
    js = [0.0, -0.5, 0.0, -2.0, 0.0, 1.5, 0.0]
    result = build_paired_candidate_result(
        label="mustard_bottle",
        candidate_idx=0,
        yaw_variant=0.07,
        ik_pregrasp_ok=True,
        plan_to_pregrasp_ok=True,
        candidate_pregrasp_js=js,
        cartesian_descend_fraction=0.0,
        cartesian_descend_ok=True,
        lift_ok=True,
        post_lift_exit_ok=True,
        direct_action_to_hub_ok=True,
        prevalidation_source="geometric_fallback",
        simple_direct_route=True,
    )
    assert result["result"] == "ACCEPT"


def test_preflight_accepts_geometric_fallback_without_paired() -> None:
    ok, reason = pick_route_preflight_allows_motion(
        plan_before_result="OK_FULL_PICK_ROUTE_PREVALIDATED",
        cartesian_descend_prevalidated=True,
        full_route_required=True,
        cartesian_fraction=0.19355,
        fraction_threshold=0.95,
        cartesian_descend_prevalidation_source="geometric_fallback",
        paired_validation_required=False,
    )
    assert ok, reason


def test_select_paired_candidate_requires_transport_accept() -> None:
    js = [0.0, -0.5, 0.0, -2.0, 0.0, 1.5, 0.0]
    paired = [
        build_paired_candidate_result(
            label="cracker_box",
            candidate_idx=0,
            yaw_variant=0.0,
            ik_pregrasp_ok=True,
            plan_to_pregrasp_ok=True,
            candidate_pregrasp_js=js,
            cartesian_descend_fraction=0.98,
            cartesian_descend_ok=True,
            lift_ok=True,
            post_lift_exit_ok=True,
            direct_action_to_hub_ok=True,
        ),
        build_paired_candidate_result(
            label="cracker_box",
            candidate_idx=1,
            yaw_variant=0.1,
            ik_pregrasp_ok=True,
            plan_to_pregrasp_ok=True,
            candidate_pregrasp_js=js,
            cartesian_descend_fraction=0.99,
            cartesian_descend_ok=True,
            lift_ok=True,
            post_lift_exit_ok=False,
            direct_action_to_hub_ok=False,
        ),
    ]
    transport = [
        {
            "candidate_idx": 0,
            "result": "ACCEPT",
            "joint_distance_to_hub": 0.5,
            "wrist_twist_score": 0.1,
            "elbow_score": 0.2,
        }
    ]
    selected = select_paired_pregrasp_candidate(paired, transport)
    assert selected is not None
    assert int(selected["candidate_idx"]) == 0
    assert select_paired_pregrasp_candidate(paired, []) is None


def test_paired_preflight_fail_reason_no_cartesian() -> None:
    assert (
        paired_preflight_fail_reason([])
        == PAIRED_REJECT_REASON_NO_CARTESIAN
    )


def test_paired_preflight_fail_reason_lift_after_valid_descend() -> None:
    results = [
        {
            "cartesian_descend_ok": True,
            "lift_ok": False,
            "result": "REJECT",
        }
    ]
    assert (
        paired_preflight_fail_reason(results)
        == PAIRED_REJECT_REASON_NO_LIFT
    )


def test_paired_preflight_fail_reason_transport_after_valid_lift() -> None:
    results = [
        {
            "cartesian_descend_ok": True,
            "lift_ok": True,
            "local_escape_ok": False,
            "post_lift_exit_ok": False,
            "global_route_ok": True,
            "direct_action_to_hub_ok": True,
            "result": "REJECT",
        }
    ]
    assert (
        paired_preflight_fail_reason(results)
        == PAIRED_REJECT_REASON_TRANSPORT_AFTER_VALID_LIFT
    )


def test_paired_preflight_accepts_local_escape_without_global_route() -> None:
    results = [
        {
            "cartesian_descend_ok": True,
            "lift_ok": True,
            "local_escape_ok": True,
            "global_route_ok": False,
            "result": "ACCEPT",
        }
    ]
    assert (
        paired_preflight_fail_reason(results)
        == PAIRED_REJECT_REASON_NO_TRANSPORT_EXIT
    )


def test_paired_pregrasp_fk_contract_ok() -> None:
    expected_tcp = (0.455, 0.115, 0.555)
    expected_hand = (0.455, 0.115, 0.655)
    actual_tcp = (0.4551, 0.1151, 0.5551)
    actual_hand = (0.4551, 0.1151, 0.6551)
    result = evaluate_paired_pregrasp_fk_contract(
        expected_tcp=expected_tcp,
        expected_hand=expected_hand,
        actual_tcp=actual_tcp,
        actual_hand=actual_hand,
    )
    assert result["ok"]
    assert not result["frame_mismatch"]
    assert float(result["tcp_error_m"]) < 0.005
    assert float(result["hand_error_m"]) < 0.005


def test_paired_pregrasp_fk_contract_detects_hand_tcp_frame_mismatch() -> None:
    expected_tcp = (0.455, 0.115, 0.555)
    expected_hand = (0.455, 0.115, 0.655)
    # Bug: panda_hand planificado a coords TCP (sin conversión).
    actual_hand_at_tcp_height = (0.455, 0.115, 0.555)
    actual_tcp_too_low = (0.455, 0.115, 0.455)
    result = evaluate_paired_pregrasp_fk_contract(
        expected_tcp=expected_tcp,
        expected_hand=expected_hand,
        actual_tcp=actual_tcp_too_low,
        actual_hand=actual_hand_at_tcp_height,
    )
    assert not result["ok"]
    assert result["frame_mismatch"]
    assert float(result["hand_vs_tcp_error_m"]) < 0.005
    assert float(result["hand_error_m"]) > 0.05


def test_build_paired_candidate_rejects_failed_fk_contract() -> None:
    js = [0.0, -0.5, 0.0, -2.0, 0.0, 1.5, 0.0]
    result = build_paired_candidate_result(
        label="cracker_box",
        candidate_idx=0,
        yaw_variant=0.0,
        ik_pregrasp_ok=True,
        plan_to_pregrasp_ok=True,
        candidate_pregrasp_js=js,
        fk_contract_ok=False,
        cartesian_descend_fraction=0.99,
        cartesian_descend_ok=True,
        lift_ok=True,
        post_lift_exit_ok=True,
        direct_action_to_hub_ok=True,
    )
    assert result["result"] == "REJECT"


def test_pregrasp_execution_state_verify_accepts_physical_ok_after_joint7_shift() -> None:
    ok, reason = evaluate_pregrasp_execution_state_verify(
        js_distance=10.1055,
        js_threshold=0.08,
        tcp_error_m=0.0,
        tcp_threshold_m=0.015,
        gap_ok=True,
        centering_ok=True,
        joint_limits_ok=True,
    )
    assert ok is True
    assert reason == "tcp_centering_gap_ok_joint7_correction_expected"


def test_pregrasp_execution_state_verify_strict_js_and_tcp() -> None:
    ok, reason = evaluate_pregrasp_execution_state_verify(
        js_distance=0.01,
        js_threshold=0.08,
        tcp_error_m=0.001,
        tcp_threshold_m=0.015,
        gap_ok=True,
        centering_ok=True,
        joint_limits_ok=True,
    )
    assert ok is True
    assert reason == "js_and_tcp_ok"


def test_compute_wrapped_joint_diffs_uses_shortest_angle() -> None:
    class _Js:
        def __init__(self, positions: list[float]) -> None:
            self.name = [
                "panda_joint1",
                "panda_joint2",
                "panda_joint3",
                "panda_joint4",
                "panda_joint5",
                "panda_joint6",
                "panda_joint7",
            ]
            self.position = positions

    current = _Js([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3.0])
    reference = _Js([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -3.0])
    diffs, largest, largest_name = compute_wrapped_joint_diffs(current, reference)
    assert largest_name == "panda_joint7"
    assert abs(float(largest)) < 0.5
    assert abs(float(diffs[-1])) < 0.5


def test_validated_joint7_runtime_allows_physical_gap_after_axis_correction() -> None:
    allow, action, reason = evaluate_validated_joint7_runtime_match(
        expected_joint7=0.6029,
        actual_joint7=-0.3277,
        joint7_tol_rad=0.05,
        expected_gap_error_deg=0.01,
        actual_gap_error_deg=0.13,
        gap_extra_deg=1.0,
        target_gap_deg=5.0,
        hard_max_gap_deg=10.0,
        axis_correction_applied=True,
        gap_ok=True,
        centering_ok=True,
        joint_limits_ok=True,
        tcp_ok=True,
    )
    assert allow is True
    assert action == "ALLOW_PHYSICAL_GAP_OK"
    assert reason == "joint7_corrected_in_place"


def test_validated_joint7_runtime_aborts_when_gap_exceeds_hard_max() -> None:
    allow, action, reason = evaluate_validated_joint7_runtime_match(
        expected_joint7=0.6029,
        actual_joint7=-0.3277,
        joint7_tol_rad=0.05,
        expected_gap_error_deg=0.01,
        actual_gap_error_deg=12.0,
        gap_extra_deg=1.0,
        target_gap_deg=5.0,
        hard_max_gap_deg=10.0,
        axis_correction_applied=True,
        gap_ok=False,
        centering_ok=True,
        joint_limits_ok=True,
        tcp_ok=True,
    )
    assert allow is False
    assert action == "ABORT_BEFORE_DESCEND"
    assert reason == "joint7_or_gap_mismatch"
