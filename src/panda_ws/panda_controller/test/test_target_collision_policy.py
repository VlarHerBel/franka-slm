"""Tests: colisión del target, object_safe_above y guardas de aproximación."""

from __future__ import annotations

from panda_controller.demo_pick_route_preflight import pick_route_preflight_allows_motion
from panda_controller.target_collision_policy import (
    OK_FULL_PICK_ROUTE_PREVALIDATED,
    REJECTED_DEFERRED_PICK_ROUTE_RESULTS,
    approach_requires_vertical_from_safe_above,
    build_target_collision_obstacle,
    format_mustard_target_collision_ready_log,
    include_target_collision,
    invalidate_stale_demo_completed_entities,
    object_safe_above_stage_required,
    pick_prevalidate_planning_scene_update_required,
    pick_route_result_allows_demo_motion,
    resolve_selected_entry_target,
    scene_has_target_collision,
    target_collision_required_for_approach,
)


def _cracker_candidate() -> dict:
    return {
        "label": "cracker_box",
        "gt_entity_name": "runtime_ycb_cracker_box_0_455633",
        "grasp_center_base": [0.475, 0.115, 0.437],
        "top_z_m": 0.470,
        "collision_dims": {
            "shape": "box",
            "box": [0.162, 0.224, 0.084],
        },
        "known_box_yaw_rad": 0.0,
        "use_target_collision_until_pregrasp": True,
        "object_safe_above_tcp": [0.475, 0.115, 0.620],
        "object_safe_above_clearance_m": 0.150,
    }


def _obstacles_only() -> list:
    return [
        {
            "entity_name": "runtime_ycb_mustard_bottle_2_455633",
            "label": "mustard_bottle",
            "is_target": False,
        },
        {
            "entity_name": "runtime_ycb_sugar_box_1_455633",
            "label": "sugar_box",
            "is_target": False,
        },
        {
            "entity_name": "runtime_ycb_chips_can_3_455633",
            "label": "chips_can",
            "is_target": False,
        },
    ]


def test_cracker_box_target_collision_obstacle_build() -> None:
    cand = _cracker_candidate()
    obs = build_target_collision_obstacle(cand, table_z_m=0.428, padding_m=0.0)
    assert obs is not None
    assert obs["is_target"] is True
    assert obs["collision_id"] == "target_runtime_ycb_cracker_box_0_455633"
    assert obs["role"] == "target"
    assert not scene_has_target_collision(_obstacles_only(), cand)


def test_planning_scene_needs_target_from_candidate_when_filtered() -> None:
    """Tras filter_scene_obstacles, el target no está en scene_obstacles."""
    cand = _cracker_candidate()
    obstacles = _obstacles_only()
    assert len(obstacles) == 3
    assert not scene_has_target_collision(obstacles, cand)
    built = build_target_collision_obstacle(cand, table_z_m=0.428)
    assert built is not None
    assert built["entity_name"] == "runtime_ycb_cracker_box_0_455633"


def test_pick_prevalidate_requires_scene_update_with_target_only() -> None:
    assert pick_prevalidate_planning_scene_update_required(
        include_target=True,
        scene_obstacles=[],
    )
    assert not pick_prevalidate_planning_scene_update_required(
        include_target=False,
        scene_obstacles=[],
    )


def test_mustard_target_collision_ready_log_only_for_mustard() -> None:
    log = format_mustard_target_collision_ready_log(
        label="mustard_bottle",
        target_collision_present=True,
        object_id="target_runtime_ycb_mustard_bottle_2_455633",
    )
    assert "MUSTARD_TARGET_COLLISION_READY" in log
    assert format_mustard_target_collision_ready_log(
        label="cracker_box",
        target_collision_present=True,
        object_id="target_cracker",
    ) == ""


def test_remove_target_collision_timing_contract() -> None:
    """Target collision activo hasta pregrasp; remove solo antes del descend."""
    cand = _cracker_candidate()
    assert include_target_collision(cand) is True
    assert bool(cand.get("remove_target_collision_before_descend", True)) is True


def test_object_safe_above_mandatory_for_cracker_multiobject() -> None:
    cand = _cracker_candidate()
    plan_targets = {"pregrasp_tcp": (0.475, 0.115, 0.512)}
    required = object_safe_above_stage_required(
        cand,
        plan_targets,
        include_target_collision_flag=True,
        obstacle_count=3,
    )
    assert required is True


def test_object_safe_above_not_blocked_by_safe_pregrasp_disabled() -> None:
    """safe_pregrasp_stage=false no debe desactivar object_safe_above."""
    cand = _cracker_candidate()
    plan_targets = {"pregrasp_tcp": (0.475, 0.115, 0.512)}
    assert object_safe_above_stage_required(
        cand,
        plan_targets,
        include_target_collision_flag=True,
        obstacle_count=3,
    )


def test_regression_abort_when_target_collision_missing_near_object() -> None:
    cand = _cracker_candidate()
    abort = target_collision_required_for_approach(
        target_collision_present=False,
        goal_tcp_xy=(0.475, 0.115),
        goal_tcp_z=0.512,
        object_center_xy=(0.475, 0.115),
        top_z=0.470,
        object_radius_m=0.11,
    )
    assert abort is True


def test_approach_vertical_from_safe_above_required() -> None:
    assert approach_requires_vertical_from_safe_above(
        from_tcp_z=0.512,
        to_tcp_z=0.512,
        object_safe_above_tcp_z=0.620,
    )
    assert not approach_requires_vertical_from_safe_above(
        from_tcp_z=0.620,
        to_tcp_z=0.512,
        object_safe_above_tcp_z=0.620,
    )


def test_stale_demo_completed_entities_cleared() -> None:
    completed = {"runtime_ycb_cracker_box_0_885932"}
    runtime = {"runtime_ycb_cracker_box_0_455633", "runtime_ycb_sugar_box_1_455633"}
    kept, stale = invalidate_stale_demo_completed_entities(completed, runtime)
    assert stale == ["runtime_ycb_cracker_box_0_885932"]
    assert kept == set()


def test_stale_demo_completed_entities_keep_deposited_off_table() -> None:
    completed = {
        "runtime_ycb_cracker_box_0_885932",
        "runtime_ycb_chips_can_1_885932",
    }
    runtime = {"runtime_ycb_sugar_box_2_957867"}
    deposited = {
        "runtime_ycb_cracker_box_0_885932",
        "runtime_ycb_chips_can_1_885932",
    }
    kept, stale = invalidate_stale_demo_completed_entities(
        completed, runtime, deposited_entities=deposited
    )
    assert stale == []
    assert kept == completed


def test_deferred_object_safe_above_results_rejected_by_preflight() -> None:
    for result in REJECTED_DEFERRED_PICK_ROUTE_RESULTS:
        ok, reason = pick_route_preflight_allows_motion(
            plan_before_result=result,
            cartesian_descend_prevalidated=False,
            full_route_required=True,
            cartesian_fraction=None,
            fraction_threshold=0.95,
        )
        assert ok is False
        assert reason == "deferred_or_incomplete_pick_route"


def test_ok_object_safe_above_prelude_phase1_never_allows_motion() -> None:
    ok, reason = pick_route_preflight_allows_motion(
        plan_before_result="OK_OBJECT_SAFE_ABOVE_PRELUDE_PHASE1",
        cartesian_descend_prevalidated=False,
        full_route_required=True,
        cartesian_fraction=None,
        fraction_threshold=0.95,
        object_safe_above_deferred=True,
    )
    assert ok is False
    assert reason in (
        "deferred_or_incomplete_pick_route",
        "object_safe_above_route_not_fully_prevalidated",
    )


def test_full_pick_route_prevalidated_allows_motion() -> None:
    ok, reason = pick_route_preflight_allows_motion(
        plan_before_result=OK_FULL_PICK_ROUTE_PREVALIDATED,
        cartesian_descend_prevalidated=True,
        full_route_required=True,
        cartesian_fraction=0.99,
        fraction_threshold=0.95,
    )
    assert ok is True
    assert reason == "full_pick_route_prevalidated"


def test_pick_route_result_allows_demo_motion_helper() -> None:
    ok, reason = pick_route_result_allows_demo_motion(
        plan_before_result=OK_FULL_PICK_ROUTE_PREVALIDATED,
        cartesian_descend_prevalidated=True,
        full_route_required=True,
    )
    assert ok is True
    assert reason == "ok"


def test_selected_entry_target_object_safe_above_when_prevalidated() -> None:
    cand = {"_object_safe_above_route_prevalidated": True}
    assert (
        resolve_selected_entry_target(cand, OK_FULL_PICK_ROUTE_PREVALIDATED)
        == "object_safe_above_tcp"
    )
