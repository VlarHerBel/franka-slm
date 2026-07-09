"""Tests ruta pick simplificada sugar_box / mustard_bottle."""

from panda_controller.simple_direct_pick_route import (
    evaluate_simple_direct_pregrasp_start_fk,
    evaluate_simple_direct_vertical_descend_geometry,
    format_simple_direct_pregrasp_start_fk_validate_log,
    format_sugar_box_simple_descend_prevalidate_log,
    mustard_pregrasp_plan_ik_fallback_eligible,
    simple_direct_mustard_bottle_route_active,
    simple_direct_pick_route_eligible,
    simple_direct_pick_route_prevalidate_required,
    SIMPLE_DIRECT_PREGRASP_START_FK_TCP_ERROR_THRESHOLD_M,
)
from panda_controller.target_collision_policy import object_safe_above_stage_required


def test_simple_direct_labels() -> None:
    assert simple_direct_pick_route_eligible("sugar_box")
    assert simple_direct_pick_route_eligible("mustard_bottle")
    assert not simple_direct_pick_route_eligible("cracker_box")


def test_mustard_pregrasp_plan_ik_fallback_eligible() -> None:
    cand = {
        "label": "mustard_bottle",
        "_simple_direct_pick_route": True,
        "scene_id": "chips_mustard_01",
    }
    assert mustard_pregrasp_plan_ik_fallback_eligible(cand)
    assert mustard_pregrasp_plan_ik_fallback_eligible(
        cand, mustard_pregrasp_ik_joint_goal=True
    )
    assert not mustard_pregrasp_plan_ik_fallback_eligible(
        {"label": "mustard_bottle", "_simple_direct_pick_route": True, "scene_id": "demo_scene_01"}
    )
    assert not mustard_pregrasp_plan_ik_fallback_eligible(
        {"label": "sugar_box", "_simple_direct_pick_route": True}
    )


    assert simple_direct_mustard_bottle_route_active(
        {"label": "mustard_bottle", "_simple_direct_pick_route": True}
    )
    assert not simple_direct_mustard_bottle_route_active(
        {"label": "mustard_bottle", "_simple_direct_pick_route": False}
    )
    assert not simple_direct_mustard_bottle_route_active({"label": "sugar_box"})


def test_pregrasp_start_fk_validate_threshold() -> None:
    pre_tcp = (0.662, 0.084, 0.462)
    pre_hand = (0.662, 0.084, 0.562)
    ok_eval = evaluate_simple_direct_pregrasp_start_fk(
        pregrasp_tcp_desired=pre_tcp,
        pre_hand_plan=pre_hand,
        fk_tcp=(0.662, 0.084, 0.461),
        fk_hand=(0.662, 0.084, 0.562),
    )
    assert ok_eval["ok"]
    assert float(ok_eval["start_tcp_error_m"]) < SIMPLE_DIRECT_PREGRASP_START_FK_TCP_ERROR_THRESHOLD_M

    fail_eval = evaluate_simple_direct_pregrasp_start_fk(
        pregrasp_tcp_desired=pre_tcp,
        pre_hand_plan=pre_hand,
        fk_tcp=(0.662, 0.084, 0.432),
        fk_hand=(0.662, 0.084, 0.562),
    )
    assert not fail_eval["ok"]
    assert float(fail_eval["start_tcp_error_m"]) > SIMPLE_DIRECT_PREGRASP_START_FK_TCP_ERROR_THRESHOLD_M


def test_pregrasp_start_fk_validate_log_format() -> None:
    fk_eval = evaluate_simple_direct_pregrasp_start_fk(
        pregrasp_tcp_desired=(0.1, 0.2, 0.3),
        pre_hand_plan=(0.1, 0.2, 0.4),
        fk_tcp=(0.1, 0.2, 0.3),
        fk_hand=(0.1, 0.2, 0.4),
    )
    log = format_simple_direct_pregrasp_start_fk_validate_log(
        label="mustard_bottle",
        fk_eval=fk_eval,
        result="OK",
    )
    assert "[SIMPLE_DIRECT_PREGRASP_START_FK_VALIDATE]" in log
    assert "label=mustard_bottle" in log
    assert "pregrasp_tcp_desired=" in log
    assert "fk_grasp_tcp=" in log


def test_vertical_descend_geometry() -> None:
    ok, reason = evaluate_simple_direct_vertical_descend_geometry(
        (0.662, 0.084, 0.462),
        (0.662, 0.084, 0.407),
    )
    assert ok
    assert reason == "vertical_ok"

    fail_xy, reason_xy = evaluate_simple_direct_vertical_descend_geometry(
        (0.662, 0.084, 0.462),
        (0.670, 0.084, 0.407),
    )
    assert not fail_xy
    assert "xy_mismatch" in reason_xy


def test_object_safe_above_not_required_for_sugar_and_mustard() -> None:
    plan_targets = {"pregrasp_tcp": (0.662, 0.084, 0.462)}
    for label in ("sugar_box", "mustard_bottle"):
        cand = {
            "label": label,
            "top_z_m": 0.437,
            "object_safe_above_tcp": [0.662, 0.084, 0.587],
        }
        assert not object_safe_above_stage_required(
            cand,
            plan_targets,
            include_target_collision_flag=True,
            obstacle_count=1,
        )


def test_prevalidate_required_with_workspace_prelude() -> None:
    cand = {"label": "sugar_box"}
    assert simple_direct_pick_route_prevalidate_required(
        cand,
        enable_pick_workspace_prelude=True,
        plan_before_prelude_skip_workspace_prelude=False,
    )
    assert not simple_direct_pick_route_prevalidate_required(
        cand,
        enable_pick_workspace_prelude=False,
        plan_before_prelude_skip_workspace_prelude=False,
    )


def test_sugar_simple_descend_log_format() -> None:
    log = format_sugar_box_simple_descend_prevalidate_log(
        {
            "pregrasp_tcp": "(0.662, 0.084, 0.462)",
            "grasp_tcp": "(0.662, 0.084, 0.407)",
            "descend_delta": "0.055",
            "cartesian_fraction": "1.000",
            "endpoint_ik_ok": "true",
            "result": "OK",
            "reason": "vertical_descend_prevalidated",
        }
    )
    assert "[SUGAR_BOX_SIMPLE_DESCEND_PREVALIDATE]" in log
    assert "descend_delta=0.055" in log
    assert "target_collision_present=false" in log
