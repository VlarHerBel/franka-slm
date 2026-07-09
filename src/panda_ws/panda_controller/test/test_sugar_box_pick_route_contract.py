"""Tests: contrato ruta pick sugar_box direct_pregrasp vs object_safe_above."""

from panda_controller.sugar_box_pick_route_contract import (
    SUGAR_BOX_PLAN_TOP_DOWN_DOT_MIN,
    SUGAR_BOX_STRICT_TOP_DOWN_DOT_DEFAULT,
    sugar_box_descend_wrist_ok_with_gap_lock,
    sugar_box_direct_pregrasp_route_contract_violation,
    sugar_box_direct_pregrasp_yaw_locked,
    sugar_box_lift_centering_prerequisite_required,
    sugar_box_lift_gap_prerequisite_required,
    sugar_box_pregrasp_plan_ik_fallback_eligible,
    sugar_box_resolve_selected_route,
    sugar_box_skip_gripper_centering_verify,
    sugar_box_skip_gripper_gap_alignment,
    sugar_box_skip_joint7_axis_in_place_correction,
    sugar_box_skip_object_safe_above_stage,
    sugar_box_use_gap_aligned_descend_orientation_lock,
)


def test_skip_object_safe_above_when_direct_route_explicit() -> None:
    cand = {"label": "sugar_box", "_sugar_box_selected_route": "direct_pregrasp"}
    assert sugar_box_skip_object_safe_above_stage(cand, {}) is True


def test_skip_object_safe_above_when_preplanned_route_direct() -> None:
    cand = {"label": "sugar_box"}
    pre = {"route": "direct_pregrasp", "selected_entry_target": "pregrasp_tcp"}
    assert sugar_box_skip_object_safe_above_stage(cand, pre) is True


def test_no_skip_when_object_safe_above_selected() -> None:
    cand = {
        "label": "sugar_box",
        "_sugar_box_selected_route": "object_safe_above",
        "selected_entry_target": "object_safe_above_tcp",
    }
    pre = {
        "route": "object_safe_above",
        "selected_entry_target": "object_safe_above_tcp",
    }
    assert sugar_box_skip_object_safe_above_stage(cand, pre) is False


def test_no_skip_for_non_sugar_box() -> None:
    cand = {"label": "cracker_box", "_sugar_box_selected_route": "direct_pregrasp"}
    assert sugar_box_skip_object_safe_above_stage(cand, {}) is False


def test_contract_violation_detected() -> None:
    cand = {"label": "sugar_box", "_sugar_box_selected_route": "direct_pregrasp"}
    assert sugar_box_direct_pregrasp_route_contract_violation(
        cand, {}, attempted_stage="object_safe_above_to_pregrasp"
    )


def test_resolve_route_from_source() -> None:
    cand = {"label": "sugar_box"}
    pre = {"source": "sugar_box_direct_pregrasp_from_pick_workspace_ready"}
    assert sugar_box_resolve_selected_route(cand, pre) == "direct_pregrasp"


def test_yaw_locked_when_plan_before_direct_pregrasp() -> None:
    cand = {
        "label": "sugar_box",
        "active_yaw_rad": 1.5708,
        "_full_pick_route_prevalidated": True,
        "_plan_before_motion_validated": {
            "ok": True,
            "mode": "direct_pregrasp",
            "variant_name": "top_down_yaw",
        },
    }
    assert sugar_box_direct_pregrasp_yaw_locked(cand) is True
    assert sugar_box_skip_joint7_axis_in_place_correction(cand) is True


def test_oblique_yaw_requires_joint7_gap_alignment() -> None:
    cand = {
        "label": "sugar_box",
        "active_yaw_rad": -1.2,
        "_full_pick_route_prevalidated": True,
        "_plan_before_motion_validated": {"ok": True, "mode": "direct_pregrasp"},
    }
    assert sugar_box_direct_pregrasp_yaw_locked(cand) is True
    assert sugar_box_skip_joint7_axis_in_place_correction(cand) is False
    assert sugar_box_skip_gripper_gap_alignment(cand) is False


def test_skip_gripper_gap_when_yaw_locked_cardinal() -> None:
    cand = {
        "label": "sugar_box",
        "active_yaw_rad": 0.0,
        "_full_pick_route_prevalidated": True,
        "_plan_before_motion_validated": {"ok": True, "mode": "direct_pregrasp"},
    }
    assert sugar_box_skip_gripper_gap_alignment(cand) is True
    assert sugar_box_skip_joint7_axis_in_place_correction(cand) is True
    assert sugar_box_skip_gripper_centering_verify(cand) is True
    assert sugar_box_lift_gap_prerequisite_required(cand) is False
    assert sugar_box_lift_centering_prerequisite_required(cand) is False


def test_strict_top_down_constants() -> None:
    assert SUGAR_BOX_STRICT_TOP_DOWN_DOT_DEFAULT >= SUGAR_BOX_PLAN_TOP_DOWN_DOT_MIN


def test_pregrasp_plan_ik_fallback_eligible_for_sugar_only() -> None:
    assert sugar_box_pregrasp_plan_ik_fallback_eligible(
        {"label": "sugar_box"}
    )
    assert not sugar_box_pregrasp_plan_ik_fallback_eligible(
        {"label": "cracker_box"}
    )


def test_yaw_not_locked_for_safe_pregrasp_mode() -> None:
    cand = {
        "label": "sugar_box",
        "_full_pick_route_prevalidated": True,
        "_plan_before_motion_validated": {"ok": True, "mode": "safe_pregrasp"},
    }
    assert sugar_box_direct_pregrasp_yaw_locked(cand) is False


def test_gap_aligned_descend_lock_after_joint7() -> None:
    cand = {
        "label": "sugar_box",
        "_axis_correction_applied": True,
        "_full_pick_route_prevalidated": True,
        "_plan_before_motion_validated": {"ok": True, "mode": "direct_pregrasp"},
    }
    assert sugar_box_use_gap_aligned_descend_orientation_lock(cand) is True
    assert sugar_box_descend_wrist_ok_with_gap_lock(
        top_down_dot=1.0,
        gap_angle_error_deg=0.37,
        strict_top_down_dot=SUGAR_BOX_STRICT_TOP_DOWN_DOT_DEFAULT,
        gap_target_angle_deg=5.0,
    )
    assert not sugar_box_descend_wrist_ok_with_gap_lock(
        top_down_dot=1.0,
        gap_angle_error_deg=42.0,
        strict_top_down_dot=SUGAR_BOX_STRICT_TOP_DOWN_DOT_DEFAULT,
        gap_target_angle_deg=5.0,
    )
