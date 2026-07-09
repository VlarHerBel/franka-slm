"""Política de contacto post-cierre mustard_bottle tall_object_topdown."""

from panda_controller.mustard_contact_policy import (
    MUSTARD_CAP_TOP_CONTACT_WIDTH_M,
    MUSTARD_MAX_ALLOWED_WIDTH_PENETRATION_M,
    build_mustard_operational_close_joint_targets_m,
    resolve_mustard_operational_close_joint_targets_m,
    evaluate_mustard_friction_lift_verify,
    evaluate_mustard_operational_close_squeeze,
    mustard_operational_lift_relaxed_safety,
    mustard_operational_pick_scene_active,
    evaluate_mustard_topdown_grasp_contact,
    mustard_gazebo_physical_attach_required,
    mustard_requires_gazebo_lift_verification,
    resolve_mustard_expected_contact_width_m,
)


def test_resolve_contact_width_prefers_cap_over_collision_minor() -> None:
    width, reason = resolve_mustard_expected_contact_width_m(
        db_required_width_m=0.058,
        effective_required_grasp_width_m=0.0577,
        footprint_minor_m=0.0577,
        collision_xy_minor_m=0.0577,
    )
    assert width == MUSTARD_CAP_TOP_CONTACT_WIDTH_M
    assert reason == "cap_top_contact_width"


def test_runtime_case_passes_with_symmetric_width_match() -> None:
    width, _ = resolve_mustard_expected_contact_width_m(
        db_required_width_m=0.058,
        effective_required_grasp_width_m=0.0577,
        footprint_minor_m=0.0577,
        collision_xy_minor_m=0.0577,
    )
    actual_total = 0.0549
    width_error = abs(actual_total - width)
    width_match_ok = width_error <= 0.014 and 0.010 <= (0.080 - actual_total)
    ok, reason = evaluate_mustard_topdown_grasp_contact(
        actual_total=actual_total,
        expected_contact_width_m=width,
        width_match_ok=width_match_ok,
        finger_asymmetry_m=0.0,
        max_asymmetry_m=0.008,
        centering_ok=True,
        axis_ok=True,
        max_allowed_width_penetration_m=MUSTARD_MAX_ALLOWED_WIDTH_PENETRATION_M,
    )
    assert ok is True
    assert reason == "mustard_width_match_symmetric_contact"


def test_rejects_asymmetric_contact() -> None:
    ok, reason = evaluate_mustard_topdown_grasp_contact(
        actual_total=0.0549,
        expected_contact_width_m=0.055,
        width_match_ok=True,
        finger_asymmetry_m=0.010,
        max_asymmetry_m=0.008,
        centering_ok=True,
        axis_ok=True,
        max_allowed_width_penetration_m=MUSTARD_MAX_ALLOWED_WIDTH_PENETRATION_M,
    )
    assert ok is False
    assert reason == "mustard_contact_asymmetric"


def test_rejects_margin_only_without_width_match() -> None:
    ok, reason = evaluate_mustard_topdown_grasp_contact(
        actual_total=0.040,
        expected_contact_width_m=0.055,
        width_match_ok=False,
        finger_asymmetry_m=0.0,
        max_asymmetry_m=0.008,
        centering_ok=True,
        axis_ok=True,
        max_allowed_width_penetration_m=MUSTARD_MAX_ALLOWED_WIDTH_PENETRATION_M,
    )
    assert ok is False
    assert reason == "mustard_width_mismatch_reject"


def test_mustard_topdown_uses_friction_attach_like_other_ycb() -> None:
    topdown = {"label": "mustard_bottle", "grasp_strategy": "tall_object_topdown"}
    assert not mustard_gazebo_physical_attach_required(topdown)
    assert not mustard_requires_gazebo_lift_verification(topdown)
    assert not mustard_gazebo_physical_attach_required(
        {"label": "mustard_bottle", "grasp_strategy": "short_axis"}
    )
    assert not mustard_gazebo_physical_attach_required(
        {"label": "cracker_box", "grasp_strategy": "tall_object_topdown"}
    )


def test_mustard_operational_chips_mustard_friction_no_gazebo_lift_abort() -> None:
    cand = {
        "label": "mustard_bottle",
        "grasp_strategy": "tall_object_topdown",
        "scene_id": "chips_mustard_01",
    }
    assert not mustard_gazebo_physical_attach_required(cand)
    assert not mustard_requires_gazebo_lift_verification(cand)
    assert mustard_operational_lift_relaxed_safety(cand)


def test_mustard_demo_scene_02_lift_relaxed_in_deposit_debug_scene() -> None:
    cand = {
        "label": "mustard_bottle",
        "grasp_strategy": "tall_object_topdown",
        "scene_id": "deposit_02_cracker_chips",
    }
    assert mustard_operational_lift_relaxed_safety(cand)


def test_mustard_operational_pick_active_in_demo_scene_02_family() -> None:
    cand = {
        "label": "mustard_bottle",
        "scene_id": "demo_scene_02",
    }
    assert mustard_operational_pick_scene_active(cand)
    assert mustard_operational_pick_scene_active(
        {"label": "mustard_bottle", "scene_id": "deposit_02_cracker_chips"}
    )
    assert mustard_operational_pick_scene_active(
        {"label": "mustard_bottle", "scene_id": "deposit_03_mustard_only"}
    )
    assert not mustard_operational_pick_scene_active(
        {"label": "mustard_bottle", "scene_id": "two_boxes_01"}
    )


def test_operational_close_targets_escalate() -> None:
    targets = build_mustard_operational_close_joint_targets_m(0.022)
    assert targets[0] == 0.022
    assert len(targets) >= 4
    assert targets[-1] < targets[0]
    assert targets[-1] >= 0.010


def test_operational_close_targets_single_attempt_default() -> None:
    single = resolve_mustard_operational_close_joint_targets_m(0.020)
    assert single == (0.020,)
    ladder = resolve_mustard_operational_close_joint_targets_m(
        0.020, single_attempt=False
    )
    assert len(ladder) >= 4
    assert ladder[0] == 0.020


def test_operational_squeeze_requires_real_closure() -> None:
    ok, reason = evaluate_mustard_operational_close_squeeze(
        actual_total=0.072,
        target_total=0.064,
        open_total=0.084,
    )
    assert ok is False
    assert reason == "mustard_operational_insufficient_squeeze"
    ok2, reason2 = evaluate_mustard_operational_close_squeeze(
        actual_total=0.063,
        target_total=0.064,
        open_total=0.080,
    )
    assert ok2 is True
    assert reason2 == "mustard_operational_squeeze_ok"


def test_operational_squeeze_passes_when_blocked_at_object_width() -> None:
    """Agarre real: dedos en ~55 mm aunque el cierre comandado sea ~36 mm."""
    ok, reason = evaluate_mustard_operational_close_squeeze(
        actual_total=0.0528,
        target_total=0.036,
        open_total=0.080,
        expected_contact_width_m=0.055,
    )
    assert ok is True
    assert reason == "mustard_operational_squeeze_ok"


def test_operational_squeeze_demo_lower_closure_delta() -> None:
    ok, reason = evaluate_mustard_operational_close_squeeze(
        actual_total=0.058,
        target_total=0.044,
        open_total=0.080,
        min_closure_delta_m=0.006,
    )
    assert ok is True
    assert reason == "mustard_operational_squeeze_ok"
    ok, reason = evaluate_mustard_friction_lift_verify(
        hand_lift_ok=True,
        attached_flag=True,
        contact_strict_pass=True,
        orient_ok=True,
        object_delta_z=float("nan"),
        hand_delta_z=0.055,
        min_delta=0.040,
        readback_available=False,
    )
    assert ok is True
    assert reason == "mustard_friction_lift_hand_proxy_ok"


def test_friction_lift_rejects_object_staying_on_table() -> None:
    ok, reason = evaluate_mustard_friction_lift_verify(
        hand_lift_ok=True,
        attached_flag=True,
        contact_strict_pass=True,
        orient_ok=True,
        object_delta_z=0.002,
        hand_delta_z=0.050,
        min_delta=0.040,
        readback_available=True,
    )
    assert ok is False
    assert reason == "mustard_gazebo_object_stayed_on_table"


def test_friction_lift_rejects_weak_partial_lift_with_readback() -> None:
    ok, reason = evaluate_mustard_friction_lift_verify(
        hand_lift_ok=True,
        attached_flag=True,
        contact_strict_pass=True,
        orient_ok=True,
        object_delta_z=0.0067,
        hand_delta_z=0.0527,
        min_delta=0.040,
        readback_available=True,
    )
    assert ok is False
    assert reason == "mustard_physical_lift_not_verified"


def test_friction_lift_partial_ok_at_18mm_object_delta() -> None:
    ok, reason = evaluate_mustard_friction_lift_verify(
        hand_lift_ok=True,
        attached_flag=True,
        contact_strict_pass=True,
        orient_ok=True,
        object_delta_z=0.0200,
        hand_delta_z=0.0527,
        min_delta=0.040,
        readback_available=True,
    )
    assert ok is True
    assert reason == "mustard_gazebo_object_lift_partial_ok"
