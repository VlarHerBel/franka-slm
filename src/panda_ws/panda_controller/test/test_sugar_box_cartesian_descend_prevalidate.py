"""Tests fallback geométrico prevalidate descend sugar_box demo_scene_02."""

import pytest

from panda_controller.sugar_box_cartesian_descend_prevalidate import (
    evaluate_sugar_box_demo_geometric_vertical_descend_fallback,
    format_sugar_box_cartesian_descend_prevalidate_fallback_log,
    sugar_box_demo_scene_02_policy_active,
    sugar_box_geometric_fallback_runtime_descend_eligible,
)


def test_sugar_box_demo_scene_02_policy_active() -> None:
    assert sugar_box_demo_scene_02_policy_active(
        label="sugar_box", scene_id="demo_scene_02"
    )
    assert not sugar_box_demo_scene_02_policy_active(
        label="cracker_box", scene_id="demo_scene_02"
    )


def test_final_descend_avoid_collisions_geometric_fallback() -> None:
    from panda_controller.sugar_box_cartesian_descend_prevalidate import (
        sugar_box_final_descend_avoid_collisions_effective,
        sugar_box_final_descend_fraction_threshold_effective,
    )

    candidate = {
        "label": "sugar_box",
        "_cartesian_descend_prevalidation_source": "geometric_fallback",
    }
    assert (
        sugar_box_final_descend_avoid_collisions_effective(
            candidate,
            policy_in_contact_zone=True,
            target_collision_removed=False,
        )
        is False
    )
    assert (
        sugar_box_final_descend_avoid_collisions_effective(
            candidate,
            policy_in_contact_zone=True,
            target_collision_removed=True,
        )
        is False
    )
    assert sugar_box_final_descend_fraction_threshold_effective(
        candidate, default_threshold=0.95, geometric_fallback_threshold=0.80
    ) == pytest.approx(0.80)


def test_geometric_fallback_runtime_descend_eligible() -> None:
    from panda_controller.sugar_box_cartesian_descend_prevalidate import (
        sugar_box_geometric_fallback_runtime_descend_eligible,
    )

    base = {
        "label": "sugar_box",
        "_cartesian_descend_prevalidation_source": "geometric_fallback",
    }
    assert sugar_box_geometric_fallback_runtime_descend_eligible(base)
    assert not sugar_box_geometric_fallback_runtime_descend_eligible(
        {**base, "_cartesian_descend_prevalidation_source": "moveit"}
    )
    assert not sugar_box_geometric_fallback_runtime_descend_eligible(
        {
            **base,
            "_sugar_box_micro_descend_from_pregrasp_used": True,
        }
    )
    assert not sugar_box_geometric_fallback_runtime_descend_eligible(
        {
            **base,
            "_sugar_box_geometric_fallback_runtime_prepared": True,
        }
    )


def test_geometric_fallback_ok_two_boxes_simple_direct() -> None:
    candidate = {
        "label": "sugar_box",
        "_simple_direct_pick_route": True,
        "top_z_m": 0.435,
        "max_cartesian_descend_m": 0.065,
        "recommended_grasp_depth_from_top_m": 0.028,
    }
    ok, reason = evaluate_sugar_box_demo_geometric_vertical_descend_fallback(
        label="sugar_box",
        scene_id="two_boxes_01",
        stage_label="pregrasp_to_grasp_cartesian",
        target_collision_removed=True,
        object_safe_above_to_pregrasp_ok=True,
        pre_plan=(0.600, 0.100, 0.472),
        gr_plan=(0.600, 0.100, 0.407),
        candidate=candidate,
        scene_obstacles=[],
        moveit_fraction=0.148,
        table_z_m=0.270,
        endpoint_ik_ok=True,
    )
    assert ok, reason
    assert "known_box_geometric" in reason


def test_geometric_fallback_ok_scanner_like_pose() -> None:
    candidate = {
        "top_z_m": 0.435,
        "max_cartesian_descend_m": 0.120,
        "recommended_grasp_depth_from_top_m": 0.036,
    }
    ok, reason = evaluate_sugar_box_demo_geometric_vertical_descend_fallback(
        label="sugar_box",
        scene_id="demo_scene_02",
        stage_label="pregrasp_to_grasp_cartesian",
        target_collision_removed=True,
        object_safe_above_to_pregrasp_ok=True,
        pre_plan=(0.630, -0.175, 0.500),
        gr_plan=(0.630, -0.175, 0.420),
        candidate=candidate,
        scene_obstacles=[
            {
                "label": "mustard_bottle",
                "position": (0.660, 0.060, 0.47),
                "collision_dims": {"shape": "cylinder", "cylinder": [0.053, 0.19]},
            }
        ],
        moveit_fraction=0.133,
        table_z_m=0.270,
        endpoint_ik_ok=True,
    )
    assert ok, reason
    log = format_sugar_box_cartesian_descend_prevalidate_fallback_log(
        result="OK", reason=reason, moveit_fraction=0.133
    )
    assert "[SUGAR_BOX_CARTESIAN_DESCEND_PREVALIDATE_FALLBACK]" in log
    assert "result=OK" in log


def test_geometric_fallback_ok_when_endpoint_ik_fails_but_volume_clear() -> None:
    """MoveIt IK puede fallar con mostaza lejana aunque el descenso vertical esté libre."""
    candidate = {
        "top_z_m": 0.435,
        "max_cartesian_descend_m": 0.120,
        "recommended_grasp_depth_from_top_m": 0.036,
    }
    ok, reason = evaluate_sugar_box_demo_geometric_vertical_descend_fallback(
        label="sugar_box",
        scene_id="demo_scene_02",
        stage_label="pregrasp_to_grasp_cartesian",
        target_collision_removed=True,
        object_safe_above_to_pregrasp_ok=True,
        pre_plan=(0.630, -0.175, 0.468),
        gr_plan=(0.630, -0.175, 0.413),
        candidate=candidate,
        scene_obstacles=[
            {
                "label": "mustard_bottle",
                "position": (0.660, 0.060, 0.368),
                "collision_dims": {
                    "shape": "box",
                    "box": [0.1003, 0.0627, 0.1959],
                },
            }
        ],
        moveit_fraction=0.0,
        table_z_m=0.270,
        endpoint_ik_ok=False,
    )
    assert ok, reason


def test_geometric_lift_pregrasp_proxy_eligible_with_distant_mustard() -> None:
    from panda_controller.sugar_box_cartesian_descend_prevalidate import (
        sugar_box_geometric_lift_pregrasp_proxy_eligible,
    )

    candidate = {
        "label": "sugar_box",
        "_cartesian_descend_prevalidation_source": "geometric_fallback",
        "_simple_direct_pick_route": True,
        "pregrasp_tcp": [0.630, -0.175, 0.472],
        "scene_obstacles": [
            {
                "label": "mustard_bottle",
                "position": (0.660, 0.060, 0.368),
                "collision_dims": {
                    "shape": "box",
                    "box": [0.1003, 0.0627, 0.1959],
                },
            }
        ],
    }
    assert sugar_box_geometric_lift_pregrasp_proxy_eligible(
        candidate,
        gr_plan=(0.630, -0.175, 0.407),
        pregrasp_js=[0.0] * 7,
    )


def test_geometric_fallback_rejects_when_target_collision_present() -> None:
    ok, reason = evaluate_sugar_box_demo_geometric_vertical_descend_fallback(
        label="sugar_box",
        scene_id="demo_scene_02",
        stage_label="pregrasp_to_grasp_cartesian",
        target_collision_removed=False,
        object_safe_above_to_pregrasp_ok=True,
        pre_plan=(0.630, -0.175, 0.500),
        gr_plan=(0.630, -0.175, 0.420),
        candidate={"top_z_m": 0.435},
        scene_obstacles=[],
        moveit_fraction=0.10,
        table_z_m=0.270,
    )
    assert not ok
    assert reason == "target_collision_not_removed"


def test_simple_direct_geometric_fallback_eligible() -> None:
    from panda_controller.sugar_box_cartesian_descend_prevalidate import (
        sugar_box_cartesian_geometric_fallback_eligible,
        sync_sugar_box_max_cartesian_descend_after_pregrasp_raise,
    )

    assert sugar_box_cartesian_geometric_fallback_eligible(
        {"label": "sugar_box", "_simple_direct_pick_route": True}
    )
    assert not sugar_box_cartesian_geometric_fallback_eligible(
        {"label": "sugar_box", "_simple_direct_pick_route": False}
    )
    candidate = {"label": "sugar_box", "max_cartesian_descend_m": 0.055}
    assert sync_sugar_box_max_cartesian_descend_after_pregrasp_raise(
        candidate, selected_pregrasp_tcp_z=0.482, grasp_tcp_z=0.407
    )
    assert abs(float(candidate["max_cartesian_descend_m"]) - 0.075) < 1e-6


def test_geometric_fallback_ok_solo_table_raised_pregrasp() -> None:
    candidate = {
        "top_z_m": 0.435,
        "max_cartesian_descend_m": 0.075,
        "recommended_grasp_depth_from_top_m": 0.028,
        "insertion_depth_limit_m": 0.032,
    }
    ok, reason = evaluate_sugar_box_demo_geometric_vertical_descend_fallback(
        label="sugar_box",
        scene_id="demo_scene_02",
        stage_label="pregrasp_to_grasp_cartesian",
        target_collision_removed=True,
        object_safe_above_to_pregrasp_ok=True,
        pre_plan=(0.630, -0.175, 0.482),
        gr_plan=(0.630, -0.175, 0.407),
        candidate=candidate,
        scene_obstacles=[],
        moveit_fraction=0.233,
        table_z_m=0.270,
        endpoint_ik_ok=True,
    )
    assert ok, reason
