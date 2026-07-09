"""Regresión: candidatos Z de entrada safe sugar_box."""

from panda_controller.sugar_box_safe_entry import (
    apply_sugar_box_demo_scene_02_remaining_equivalent_fields,
    apply_sugar_box_multiobject_safe_route_fields,
    build_sugar_box_object_safe_above_tcp,
    format_sugar_box_demo_scene_02_remaining_equivalent_log,
    format_sugar_box_object_safe_above_missing_log,
    format_sugar_box_object_safe_above_resolved_log,
    resolve_sugar_box_multiobject_object_safe_above,
    resolve_sugar_box_operative_center_xy,
    sanitize_sugar_box_direct_plan_targets,
    sugar_box_demo_golden_fast_execute_allowed,
    sugar_box_demo_scene_02_remaining_pick_equivalent,
    sugar_box_hold_gripper_frozen_after_grasp,
    sugar_box_multiobject_final_descend_target_removal_required,
    sugar_box_multiobject_full_pick_prevalidate_required,
    sugar_box_multiobject_use_object_safe_above_stage,
    sugar_box_nominal_safe_tcp_z,
    sugar_box_object_safe_above_tcp_resolved,
    sugar_box_safe_entry_tcp_z_candidates,
)
from panda_controller.simple_direct_pick_route import (
    simple_direct_pick_route_eligible,
    simple_direct_pick_route_prevalidate_required,
)


def test_nominal_safe_z_top_plus_clearance() -> None:
    assert sugar_box_nominal_safe_tcp_z(0.435, 0.10) == 0.535


def test_safe_entry_candidates_ordered_high_to_low() -> None:
    zs = sugar_box_safe_entry_tcp_z_candidates(0.435, min_clearance_above_top_m=0.055)
    assert zs[0] == 0.535
    assert zs[-1] == 0.495
    assert len(zs) == 5
    assert zs == sorted(zs, reverse=True)


def test_resolve_sugar_box_operative_center_xy_prefers_grasp_center() -> None:
    candidate = {
        "grasp_center_base": [0.630, -0.175, 0.435],
        "chosen_target_center_base": [0.620, -0.170, 0.435],
        "position": [0.625, -0.180, 0.435],
    }
    x, y, source = resolve_sugar_box_operative_center_xy(
        candidate, fallback_xy=(0.0, 0.0)
    )
    assert abs(x - 0.630) < 1e-6
    assert abs(y + 0.175) < 1e-6
    assert source == "grasp_center_base"


def test_build_sugar_box_object_safe_above_tcp_demo_scene_02() -> None:
    candidate = {"grasp_center_base": [0.630, -0.175, 0.435]}
    tcp, source = build_sugar_box_object_safe_above_tcp(
        candidate,
        safe_above_tcp_z=0.535,
        fallback_xy=(0.630, -0.175),
    )
    assert source == "grasp_center_base"
    assert abs(tcp[0] - 0.630) < 1e-6
    assert abs(tcp[1] + 0.175) < 1e-6
    assert abs(tcp[2] - 0.535) < 1e-6


def test_sugar_box_object_safe_above_logs() -> None:
    log = format_sugar_box_object_safe_above_resolved_log(
        {
            "center_source": "grasp_center_base",
            "object_safe_above_tcp": (0.630, -0.175, 0.535),
            "pregrasp_tcp": (0.630, -0.175, 0.462),
            "grasp_tcp": (0.630, -0.175, 0.407),
        }
    )
    assert "[SUGAR_BOX_OBJECT_SAFE_ABOVE_RESOLVED]" in log
    assert "center_source=grasp_center_base" in log
    assert "result=OK" in log
    assert "[SUGAR_BOX_OBJECT_SAFE_ABOVE_MISSING]" in format_sugar_box_object_safe_above_missing_log()


def test_resolve_sugar_box_operative_center_xy_known_box_fallback() -> None:
    candidate = {
        "known_box_center_base": [0.630, -0.175, 0.435],
    }
    x, y, source = resolve_sugar_box_operative_center_xy(
        candidate, fallback_xy=(0.0, 0.0)
    )
    assert abs(x - 0.630) < 1e-6
    assert abs(y + 0.175) < 1e-6
    assert source == "known_box_center_base"


def test_sugar_box_multiobject_safe_route_resolves_object_safe_above() -> None:
    """Regresión: ruta multiobjeto no debe abortar con OBJECT_SAFE_ABOVE_MISSING."""
    candidate: dict = {
        "label": "sugar_box",
        "grasp_center_base": [0.630, -0.175, 0.435],
        "top_z_m": 0.435,
    }
    seq = {
        "pregrasp_tcp": (0.630, -0.175, 0.462),
        "grasp_tcp": (0.630, -0.175, 0.407),
    }
    tcp, source, log = resolve_sugar_box_multiobject_object_safe_above(
        candidate,
        safe_above_tcp_z=0.535,
        pregrasp_tcp=seq["pregrasp_tcp"],
        grasp_tcp=seq["grasp_tcp"],
    )
    assert tcp is not None
    assert source == "grasp_center_base"
    assert "[SUGAR_BOX_OBJECT_SAFE_ABOVE_RESOLVED]" in log
    assert "result=OK" in log
    assert "[SUGAR_BOX_OBJECT_SAFE_ABOVE_MISSING]" not in log
    assert abs(tcp[2] - 0.535) < 1e-6

    ok, apply_log = apply_sugar_box_multiobject_safe_route_fields(
        candidate,
        seq,
        safe_above_tcp_z=0.535,
        clearance_above_top_m=0.100,
    )
    assert ok
    assert "[SUGAR_BOX_OBJECT_SAFE_ABOVE_RESOLVED]" in apply_log
    assert candidate["selected_entry_target"] == "object_safe_above_tcp"
    assert candidate["object_safe_above_tcp"] == [0.630, -0.175, 0.535]
    assert seq["object_safe_above_tcp"] == (0.630, -0.175, 0.535)
    assert sugar_box_multiobject_use_object_safe_above_stage(candidate)
    assert simple_direct_pick_route_eligible("sugar_box")


def test_sugar_box_multiobject_final_descend_target_removal_required() -> None:
    assert sugar_box_multiobject_final_descend_target_removal_required(
        {"label": "sugar_box", "sugar_box_multiobject_safe_route": True}
    )
    assert not sugar_box_multiobject_final_descend_target_removal_required(
        {"label": "sugar_box", "sugar_box_multiobject_safe_route": False}
    )
    assert not sugar_box_multiobject_final_descend_target_removal_required(
        {"label": "cracker_box", "sugar_box_multiobject_safe_route": True}
    )


def test_sugar_box_multiobject_full_pick_blocks_simple_direct_prevalidate() -> None:
    candidate = {
        "label": "sugar_box",
        "sugar_box_multiobject_safe_route": True,
        "object_safe_above_tcp": [0.630, -0.175, 0.535],
    }
    assert sugar_box_object_safe_above_tcp_resolved(candidate)
    assert sugar_box_multiobject_full_pick_prevalidate_required(candidate)
    assert simple_direct_pick_route_eligible("sugar_box")
    assert not simple_direct_pick_route_prevalidate_required(
        candidate,
        enable_pick_workspace_prelude=True,
        plan_before_prelude_skip_workspace_prelude=False,
    )


def test_remaining_equivalent_enables_simple_direct_prevalidate() -> None:
    candidate = {
        "label": "sugar_box",
        "sugar_box_multiobject_safe_route": True,
        "object_safe_above_tcp": [0.630, -0.175, 0.535],
        "_sugar_box_demo_scene_02_remaining_equivalent": True,
    }
    apply_sugar_box_demo_scene_02_remaining_equivalent_fields(candidate)
    assert not sugar_box_multiobject_full_pick_prevalidate_required(candidate)
    assert simple_direct_pick_route_prevalidate_required(
        candidate,
        enable_pick_workspace_prelude=True,
        plan_before_prelude_skip_workspace_prelude=False,
    )


def test_sugar_box_multiobject_full_pick_requires_object_safe_above_tcp() -> None:
    candidate = {
        "label": "sugar_box",
        "sugar_box_multiobject_safe_route": True,
    }
    assert not sugar_box_multiobject_full_pick_prevalidate_required(candidate)


def test_sugar_box_multiobject_use_object_safe_above_stage() -> None:
    candidate = {"label": "sugar_box", "sugar_box_multiobject_safe_route": True}
    assert sugar_box_multiobject_use_object_safe_above_stage(candidate)


def test_demo_scene_02_remaining_equivalent_when_only_sugar_mustard_on_table() -> None:
    assert sugar_box_demo_scene_02_remaining_pick_equivalent(
        label="sugar_box",
        scene_id="demo_scene_02",
        completed_labels=set(),
        present_table_labels={"sugar_box", "mustard_bottle"},
    )


def test_sugar_box_hold_gripper_frozen_in_demo_scene_02_family() -> None:
    assert sugar_box_hold_gripper_frozen_after_grasp(
        label="sugar_box", scene_id="deposit_02_cracker_chips"
    )
    assert sugar_box_hold_gripper_frozen_after_grasp(
        label="sugar_box", scene_id="demo_scene_02"
    )
    assert not sugar_box_hold_gripper_frozen_after_grasp(
        label="mustard_bottle", scene_id="demo_scene_02"
    )
    assert not sugar_box_hold_gripper_frozen_after_grasp(
        label="sugar_box", scene_id="two_boxes_01"
    )


def test_deposit_02_cracker_chips_sugar_golden_allowed() -> None:
    assert sugar_box_demo_scene_02_remaining_pick_equivalent(
        label="sugar_box",
        scene_id="deposit_02_cracker_chips",
        completed_labels=set(),
    )
    assert sugar_box_demo_golden_fast_execute_allowed(
        {
            "label": "sugar_box",
            "scene_id": "deposit_02_cracker_chips",
            "_sugar_box_demo_scene_02_remaining_equivalent": True,
        }
    )
    assert not sugar_box_demo_golden_fast_execute_allowed(
        {
            "label": "sugar_box",
            "scene_id": "demo_scene_02",
        }
    )


def test_demo_scene_02_remaining_equivalent_after_cracker_chips() -> None:
    assert sugar_box_demo_scene_02_remaining_pick_equivalent(
        label="sugar_box",
        scene_id="demo_scene_02",
        completed_labels={"cracker_box", "chips_can"},
    )
    assert sugar_box_demo_scene_02_remaining_pick_equivalent(
        label="sugar_box",
        scene_id="demo_scene_02",
        completed_labels={"cracker_box", "chips_can"},
        active_obstacle_labels={"mustard_bottle"},
    )
    assert sugar_box_demo_scene_02_remaining_pick_equivalent(
        label="sugar_box",
        scene_id="demo_scene_02_3obj",
        completed_labels={"cracker_box", "chips_can"},
    )
    assert not sugar_box_demo_scene_02_remaining_pick_equivalent(
        label="sugar_box",
        scene_id="demo_scene_02",
        completed_labels={"cracker_box"},
    )


def test_demo_scene_02_3obj_remaining_when_only_sugar_on_table() -> None:
    assert sugar_box_demo_scene_02_remaining_pick_equivalent(
        label="sugar_box",
        scene_id="demo_scene_02_3obj",
        completed_labels={"chips_can"},
        present_table_labels={"sugar_box"},
    )
    assert not sugar_box_demo_scene_02_remaining_pick_equivalent(
        label="sugar_box",
        scene_id="demo_scene_02_3obj",
        completed_labels=set(),
        present_table_labels={"cracker_box", "sugar_box"},
    )


def test_golden_fast_execute_blocked_for_sugar_with_obstacles_on_table() -> None:
    candidate = {
        "label": "sugar_box",
        "scene_id": "demo_scene_02",
        "sugar_box_multiobject_safe_route": True,
        "object_safe_above_tcp": [0.630, -0.175, 0.572],
    }
    assert not sugar_box_demo_golden_fast_execute_allowed(candidate)
    remaining = {
        "label": "sugar_box",
        "scene_id": "demo_scene_02",
        "_sugar_box_demo_scene_02_remaining_equivalent": True,
    }
    apply_sugar_box_demo_scene_02_remaining_equivalent_fields(remaining)
    assert sugar_box_demo_golden_fast_execute_allowed(remaining)


def test_golden_fast_execute_allowed_for_other_labels() -> None:
    assert sugar_box_demo_golden_fast_execute_allowed({"label": "chips_can"})


def test_sanitize_sugar_box_direct_plan_targets() -> None:
    candidate = {
        "label": "sugar_box",
        "sugar_box_multiobject_safe_route": True,
        "object_safe_above_tcp": [0.630, -0.175, 0.572],
    }
    plan_targets = {
        "pregrasp_tcp": (0.630, -0.175, 0.472),
        "safe_pregrasp_tcp": (0.630, -0.175, 0.572),
        "object_safe_above_tcp": (0.630, -0.175, 0.572),
    }
    sanitize_sugar_box_direct_plan_targets(candidate, plan_targets)
    assert plan_targets["safe_pregrasp_tcp"] == (0.630, -0.175, 0.472)
    assert "object_safe_above_tcp" not in plan_targets
    assert "object_safe_above_tcp" not in candidate
    assert candidate["selected_entry_target"] == "pregrasp_tcp"


def test_remaining_equivalent_disables_multiobject_full_pick() -> None:
    candidate = {
        "label": "sugar_box",
        "sugar_box_multiobject_safe_route": True,
        "object_safe_above_tcp": [0.630, -0.175, 0.535],
        "_sugar_box_demo_scene_02_remaining_equivalent": True,
    }
    apply_sugar_box_demo_scene_02_remaining_equivalent_fields(candidate)
    assert not sugar_box_multiobject_use_object_safe_above_stage(candidate)
    assert not sugar_box_multiobject_full_pick_prevalidate_required(candidate)
    assert candidate["_simple_direct_pick_route"] is True
    assert candidate["selected_entry_target"] == "pregrasp_tcp"
    assert "object_safe_above_tcp" not in candidate


def test_remaining_equivalent_log() -> None:
    log = format_sugar_box_demo_scene_02_remaining_equivalent_log(
        scene_id="demo_scene_02_3obj",
        completed_labels={"cracker_box", "chips_can"},
    )
    assert "[SUGAR_BOX_DEMO_SCENE_02_REMAINING_EQUIVALENT]" in log
    assert "policy_source=demo_scene_02_remaining_sugar_mustard" in log
    log_only_sugar = format_sugar_box_demo_scene_02_remaining_equivalent_log(
        scene_id="demo_scene_02_3obj",
        present_table_labels={"sugar_box"},
    )
    assert "policy_source=only_sugar_on_table_3obj" in log_only_sugar
