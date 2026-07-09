"""Tests fallback geométrico descend mustard_bottle."""

from __future__ import annotations

from panda_controller.mustard_cartesian_descend_prevalidate import (
    MUSTARD_GEOMETRIC_FALLBACK_DESCEND_FRACTION_THRESHOLD,
    evaluate_mustard_demo_geometric_vertical_descend_fallback,
    format_mustard_cartesian_descend_prevalidate_fallback_log,
    mustard_demo_scene_02_policy_active,
    mustard_final_descend_avoid_collisions_effective,
    mustard_final_descend_fraction_threshold_effective,
    mustard_geometric_fallback_runtime_descend_eligible,
    mustard_geometric_lift_pregrasp_proxy_eligible,
)
from panda_controller.mustard_xy_reachability_search import (
    build_mustard_demo_xy_reachability_specs,
)


def test_mustard_demo_scene_02_policy_active() -> None:
    assert mustard_demo_scene_02_policy_active(
        label="mustard_bottle", scene_id="demo_scene_02"
    )
    assert mustard_demo_scene_02_policy_active(
        label="mustard_bottle", scene_id="chips_mustard_01"
    )
    assert mustard_demo_scene_02_policy_active(
        label="mustard_bottle", scene_id="deposit_02_cracker_chips"
    )
    assert not mustard_demo_scene_02_policy_active(
        label="sugar_box", scene_id="demo_scene_02"
    )
    assert not mustard_demo_scene_02_policy_active(
        label="mustard_bottle", scene_id="two_boxes_01"
    )


def test_geometric_fallback_ok_runtime_top_controller_grasp_z() -> None:
    candidate = {
        "top_z_m": 0.470,
        "recommended_grasp_depth_from_top_m": 0.0340,
        "mustard_scanner_aligned_pregrasp_locked": True,
        "max_cartesian_descend_m": 0.090,
    }
    ok, reason = evaluate_mustard_demo_geometric_vertical_descend_fallback(
        label="mustard_bottle",
        scene_id="demo_scene_02",
        stage_label="pregrasp_to_grasp_cartesian",
        target_collision_removed=True,
        object_safe_above_to_pregrasp_ok=True,
        pre_plan=(0.6623, 0.0843, 0.4909),
        gr_plan=(0.6623, 0.0843, 0.4360),
        candidate=candidate,
        scene_obstacles=[],
        moveit_fraction=0.0,
        table_z_m=0.270,
        endpoint_ik_ok=False,
    )
    assert ok, reason


def test_geometric_fallback_ok_when_endpoint_ik_fails_but_volume_clear() -> None:
    candidate = {
        "top_z_m": 0.470,
        "max_cartesian_descend_m": 0.120,
        "recommended_grasp_depth_from_top_m": 0.040,
    }
    ok, reason = evaluate_mustard_demo_geometric_vertical_descend_fallback(
        label="mustard_bottle",
        scene_id="demo_scene_02",
        stage_label="pregrasp_to_grasp_cartesian",
        target_collision_removed=True,
        object_safe_above_to_pregrasp_ok=True,
        pre_plan=(0.6623, 0.0843, 0.4909),
        gr_plan=(0.6623, 0.0843, 0.4360),
        candidate=candidate,
        scene_obstacles=[],
        moveit_fraction=0.0,
        table_z_m=0.270,
        endpoint_ik_ok=False,
    )
    assert ok, reason


def test_geometric_fallback_rejects_when_target_collision_present() -> None:
    ok, _ = evaluate_mustard_demo_geometric_vertical_descend_fallback(
        label="mustard_bottle",
        scene_id="demo_scene_02",
        stage_label="pregrasp_to_grasp_cartesian",
        target_collision_removed=False,
        object_safe_above_to_pregrasp_ok=True,
        pre_plan=(0.6623, 0.0843, 0.4909),
        gr_plan=(0.6623, 0.0843, 0.4360),
        candidate={"top_z_m": 0.470},
        scene_obstacles=[],
        moveit_fraction=0.0,
        table_z_m=0.270,
    )
    assert not ok


def test_runtime_geometric_fallback_eligible() -> None:
    base = {
        "label": "mustard_bottle",
        "_cartesian_descend_prevalidation_source": "geometric_fallback",
        "_simple_direct_pick_route": True,
    }
    assert mustard_geometric_fallback_runtime_descend_eligible(base)
    assert not mustard_geometric_fallback_runtime_descend_eligible(
        {**base, "_mustard_geometric_fallback_runtime_prepared": True}
    )


def test_xy_specs_include_runtime_relational_grasp_z() -> None:
    candidate = {
        "known_box_center_base": [0.660, 0.060, 0.43],
        "grasp_center_base": [0.6623, 0.0843, 0.43],
        "grasp_tcp_z": 0.4360,
        "major_axis_xy": [1.0, 0.0],
        "minor_axis_xy": [0.0, 1.0],
    }
    specs = build_mustard_demo_xy_reachability_specs(
        candidate,
        controller_grasp_xy=(0.6623, 0.0843),
        controller_yaw_rad=0.0684,
        effective_top_z_m=0.470,
        min_grasp_z_m=0.40,
    )
    grasp_zs = {round(float(s["grasp_tcp"][2]), 4) for s in specs}
    assert 0.4365 in grasp_zs or 0.4360 in grasp_zs


def test_mustard_geometric_lift_pregrasp_proxy_eligible() -> None:
    candidate = {
        "label": "mustard_bottle",
        "_cartesian_descend_prevalidation_source": "geometric_fallback",
        "_simple_direct_pick_route": True,
        "_mustard_selected_pregrasp_tcp": [0.6623, 0.0843, 0.4909],
        "scene_obstacles": [],
    }
    assert mustard_geometric_lift_pregrasp_proxy_eligible(
        candidate,
        gr_plan=(0.6623, 0.0843, 0.4269),
        pregrasp_js=object(),
    )
    assert not mustard_geometric_lift_pregrasp_proxy_eligible(
        {**candidate, "_cartesian_descend_prevalidation_source": "moveit"},
        gr_plan=(0.6623, 0.0843, 0.4269),
        pregrasp_js=object(),
    )


def test_fallback_log_format() -> None:
    log = format_mustard_cartesian_descend_prevalidate_fallback_log(
        result="OK",
        reason="mustard_demo_vertical_descend_volume_clear",
        moveit_fraction=0.0,
    )
    assert "[MUSTARD_CARTESIAN_DESCEND_PREVALIDATE_FALLBACK]" in log
    assert "result=OK" in log


def test_mustard_final_descend_relaxed_cartesian_policy() -> None:
    base = {
        "label": "mustard_bottle",
        "_cartesian_descend_prevalidation_source": "geometric_fallback",
    }
    assert mustard_final_descend_avoid_collisions_effective(
        base, target_collision_removed=True
    ) is False
    thr = mustard_final_descend_fraction_threshold_effective(
        base, default_threshold=0.95
    )
    assert thr == MUSTARD_GEOMETRIC_FALLBACK_DESCEND_FRACTION_THRESHOLD
    assert (
        mustard_final_descend_avoid_collisions_effective(
            {"label": "sugar_box"}, target_collision_removed=True
        )
        is None
    )
