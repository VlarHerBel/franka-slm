"""Regresión: preflight ruta pick demo multiobjeto."""

from panda_controller.demo_pick_route_preflight import (
    demo_full_pick_route_prevalidation_required,
    executor_object_excluded_from_table_obstacles,
    pick_route_preflight_allows_motion,
    sugar_box_force_safe_pregrasp_when_obstacles,
)


def test_full_route_required_demo_fast_with_obstacles() -> None:
    c = {
        "label": "sugar_box",
        "scene_obstacles": [{"label": "cracker_box", "is_target": False}],
    }
    assert (
        demo_full_pick_route_prevalidation_required(
            candidate=c,
            demo_fast_mode=True,
            demo_motion_profile_active=False,
            require_param=False,
            chips_can_candidate=False,
        )
        is True
    )


def test_full_route_not_required_chips_can() -> None:
    c = {
        "label": "chips_can",
        "scene_obstacles": [{"label": "cracker_box", "is_target": False}],
    }
    assert (
        demo_full_pick_route_prevalidation_required(
            candidate=c,
            demo_fast_mode=True,
            demo_motion_profile_active=False,
            require_param=True,
            chips_can_candidate=True,
        )
        is False
    )


def test_preflight_rejects_ok_pregrasp_only_in_demo() -> None:
    ok, reason = pick_route_preflight_allows_motion(
        plan_before_result="OK_PREGRASP_ONLY",
        cartesian_descend_prevalidated=False,
        full_route_required=True,
        cartesian_fraction=None,
        fraction_threshold=0.95,
    )
    assert ok is False
    assert reason == "cartesian_descend_not_prevalidated"


def test_preflight_accepts_full_route() -> None:
    ok, reason = pick_route_preflight_allows_motion(
        plan_before_result="OK_FULL_PICK_ROUTE",
        cartesian_descend_prevalidated=True,
        full_route_required=True,
        cartesian_fraction=0.98,
        fraction_threshold=0.95,
    )
    assert ok is True
    assert reason == "ok"


def test_preflight_pending_descend_at_pregrasp() -> None:
    ok, reason = pick_route_preflight_allows_motion(
        plan_before_result="OK_PREGRASP_PENDING_DESCEND_VALIDATE",
        cartesian_descend_prevalidated=False,
        full_route_required=True,
        cartesian_fraction=None,
        fraction_threshold=0.95,
        cartesian_descend_pending_at_pregrasp=True,
    )
    assert ok is True
    assert reason == "descend_validate_at_pregrasp"


def test_preflight_deferred_sugar_phase1() -> None:
    ok, reason = pick_route_preflight_allows_motion(
        plan_before_result="OK_PRELUDE_PHASE1",
        cartesian_descend_prevalidated=False,
        full_route_required=True,
        cartesian_fraction=None,
        fraction_threshold=0.95,
        sugar_two_phase_deferred=True,
    )
    assert ok is True
    assert reason == "deferred_full_route_after_pick_workspace_ready"


def test_preflight_rejects_object_safe_above_deferred_phase1() -> None:
    ok, reason = pick_route_preflight_allows_motion(
        plan_before_result="OK_OBJECT_SAFE_ABOVE_PRELUDE_PHASE1",
        cartesian_descend_prevalidated=False,
        full_route_required=True,
        cartesian_fraction=None,
        fraction_threshold=0.95,
    )
    assert ok is False
    assert reason == "deferred_or_incomplete_pick_route"


def test_preflight_accepts_full_route_prevalidated() -> None:
    ok, reason = pick_route_preflight_allows_motion(
        plan_before_result="OK_FULL_PICK_ROUTE_PREVALIDATED",
        cartesian_descend_prevalidated=True,
        full_route_required=True,
        cartesian_fraction=0.98,
        fraction_threshold=0.95,
    )
    assert ok is True
    assert reason == "full_pick_route_prevalidated"


def test_preflight_accepts_collision_off_descend_with_paired_validation() -> None:
    from panda_controller.demo_cracker_collision_off_final_descend import (
        DEMO_COLLISION_OFF_FINAL_DESCEND_SOURCE,
    )

    ok, reason = pick_route_preflight_allows_motion(
        plan_before_result="OK_FULL_PICK_ROUTE",
        cartesian_descend_prevalidated=True,
        full_route_required=True,
        cartesian_fraction=1.0,
        fraction_threshold=0.95,
        cartesian_descend_prevalidation_source=DEMO_COLLISION_OFF_FINAL_DESCEND_SOURCE,
        paired_validation_required=True,
    )
    assert ok is True
    assert reason == "ok"


def test_preflight_accepts_full_route_prevalidated_with_collision_off_paired() -> None:
    from panda_controller.demo_cracker_collision_off_final_descend import (
        DEMO_COLLISION_OFF_FINAL_DESCEND_SOURCE,
    )

    ok, reason = pick_route_preflight_allows_motion(
        plan_before_result="OK_FULL_PICK_ROUTE_PREVALIDATED",
        cartesian_descend_prevalidated=True,
        full_route_required=True,
        cartesian_fraction=1.0,
        fraction_threshold=0.95,
        cartesian_descend_prevalidation_source=DEMO_COLLISION_OFF_FINAL_DESCEND_SOURCE,
        paired_validation_required=True,
    )
    assert ok is True
    assert reason == "full_pick_route_prevalidated"


def test_sugar_box_safe_when_obstacles() -> None:
    c = {
        "label": "sugar_box",
        "scene_obstacles": [{"label": "mustard_bottle", "is_target": False}],
    }
    assert (
        sugar_box_force_safe_pregrasp_when_obstacles(
            candidate=c, param_enabled=True, min_obstacles=1
        )
        is True
    )


def test_placed_executor_object_excluded() -> None:
    obj = {"label": "cracker_box", "entity_name": "runtime_ycb_cracker_1", "placed": True}
    assert (
        executor_object_excluded_from_table_obstacles(
            obj,
            completed_entities=set(),
            completed_labels=set(),
        )
        is True
    )
    assert (
        executor_object_excluded_from_table_obstacles(
            {"label": "cracker_box", "entity_name": "runtime_ycb_cracker_1"},
            completed_entities={"runtime_ycb_cracker_1"},
            completed_labels=set(),
        )
        is True
    )
