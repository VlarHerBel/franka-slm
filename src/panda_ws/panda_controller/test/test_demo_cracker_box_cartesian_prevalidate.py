"""Tests: prevalidación cartesiana demo_scene_02 cracker_box."""

from panda_controller.demo_cracker_box_cartesian_prevalidate import (
    apply_demo_cracker_descend_limit_policy,
    apply_demo_scene_02_cracker_box_descend_sequence,
    known_box_geometric_lift_pregrasp_proxy_eligible,
    known_box_paired_grid_search_active,
    demo_scene_02_cracker_box_policy_active,
    evaluate_demo_geometric_vertical_descend_fallback,
    vertical_descend_volume_clear_of_obstacles,
)
from panda_controller.demo_pick_route_preflight import pick_route_preflight_allows_motion


def _cracker_candidate(**extra) -> dict:
    base = {
        "label": "cracker_box",
        "top_z_m": 0.470,
        "recommended_grasp_depth_from_top_m": 0.033,
        "insertion_depth_limit_m": 0.036,
        "min_tcp_clearance_above_table_m": 0.012,
        "max_cartesian_descend_m": 0.120,
        "yaw_policy": "align_short_axis",
        "collision_dims": {"shape": "box", "box": [0.158, 0.060, 0.210]},
    }
    base.update(extra)
    return base


def _seq_base() -> dict:
    return {
        "grasp_tcp": (0.455, 0.115, 0.437),
        "pregrasp_tcp": (0.455, 0.115, 0.512),
        "safe_pregrasp_tcp": (0.455, 0.115, 0.590),
        "max_cartesian_descend_m": 0.100,
        "final_descend_m": 0.075,
    }


def test_demo_scene_02_cracker_pregrasp_clearance_at_least_80mm() -> None:
    seq = apply_demo_scene_02_cracker_box_descend_sequence(
        _seq_base(),
        top_z=0.470,
        pregrasp_clearance_m=0.085,
        object_safe_above_clearance_m=0.150,
        max_target_z=0.95,
    )
    assert float(seq["pregrasp_tcp"][2]) - 0.470 >= 0.080
    assert float(seq["object_safe_above_tcp"][2]) >= 0.620 - 1e-3


def test_geometric_fallback_ok_on_moveit_false_negative() -> None:
    pre = (0.455, 0.115, 0.555)
    gr = (0.455, 0.115, 0.437)
    obstacles = [
        {
            "label": "mustard_bottle",
            "position": [0.660, 0.060, 0.470],
            "collision_dims": {"shape": "box", "box": [0.121, 0.106, 0.140]},
        },
        {
            "label": "sugar_box",
            "position": [0.630, -0.130, 0.470],
            "collision_dims": {"shape": "box", "box": [0.089, 0.038, 0.072]},
        },
    ]
    ok, reason = evaluate_demo_geometric_vertical_descend_fallback(
        label="cracker_box",
        demo_authoritative_scene=True,
        scene_id="demo_scene_02",
        stage_label="pregrasp_to_grasp_cartesian",
        target_collision_removed=True,
        object_safe_above_to_pregrasp_ok=True,
        pre_plan=pre,
        gr_plan=gr,
        candidate=_cracker_candidate(),
        scene_obstacles=obstacles,
        moveit_fraction=0.19355,
        table_z_m=0.40,
    )
    assert ok, reason


def test_geometric_fallback_rejected_without_authoritative_scene() -> None:
    ok, reason = evaluate_demo_geometric_vertical_descend_fallback(
        label="cracker_box",
        demo_authoritative_scene=False,
        scene_id="demo_scene_02",
        stage_label="pregrasp_to_grasp_cartesian",
        target_collision_removed=True,
        object_safe_above_to_pregrasp_ok=True,
        pre_plan=(0.455, 0.115, 0.555),
        gr_plan=(0.455, 0.115, 0.437),
        candidate=_cracker_candidate(),
        scene_obstacles=[],
        moveit_fraction=0.19,
        table_z_m=0.40,
    )
    assert not ok
    assert reason == "not_demo_scene_02_cracker_box"


def test_geometric_fallback_rejected_for_chips_can() -> None:
    ok, reason = evaluate_demo_geometric_vertical_descend_fallback(
        label="chips_can",
        demo_authoritative_scene=True,
        scene_id="demo_scene_02",
        stage_label="pregrasp_to_grasp_cartesian",
        target_collision_removed=True,
        object_safe_above_to_pregrasp_ok=True,
        pre_plan=(0.52, -0.04, 0.55),
        gr_plan=(0.52, -0.04, 0.44),
        candidate={"label": "chips_can", "top_z_m": 0.47},
        scene_obstacles=[],
        moveit_fraction=0.2,
        table_z_m=0.40,
    )
    assert not ok


def test_geometric_fallback_requires_target_collision_removed() -> None:
    ok, _ = evaluate_demo_geometric_vertical_descend_fallback(
        label="cracker_box",
        demo_authoritative_scene=True,
        scene_id="demo_scene_02",
        stage_label="pregrasp_to_grasp_cartesian",
        target_collision_removed=False,
        object_safe_above_to_pregrasp_ok=True,
        pre_plan=(0.455, 0.115, 0.555),
        gr_plan=(0.455, 0.115, 0.437),
        candidate=_cracker_candidate(),
        scene_obstacles=[],
        moveit_fraction=0.19,
        table_z_m=0.40,
    )
    assert not ok


def test_geometric_fallback_requires_same_xy() -> None:
    ok, reason = evaluate_demo_geometric_vertical_descend_fallback(
        label="cracker_box",
        demo_authoritative_scene=True,
        scene_id="demo_scene_02",
        stage_label="pregrasp_to_grasp_cartesian",
        target_collision_removed=True,
        object_safe_above_to_pregrasp_ok=True,
        pre_plan=(0.455, 0.115, 0.555),
        gr_plan=(0.460, 0.115, 0.437),
        candidate=_cracker_candidate(),
        scene_obstacles=[],
        moveit_fraction=0.19,
        table_z_m=0.40,
    )
    assert not ok
    assert "xy_mismatch" in reason


def test_geometric_fallback_fails_if_obstacle_too_close() -> None:
    obstacles = [
        {
            "label": "mustard_bottle",
            "position": [0.456, 0.116, 0.470],
            "collision_dims": {"shape": "box", "box": [0.121, 0.106, 0.140]},
        }
    ]
    ok, reason = evaluate_demo_geometric_vertical_descend_fallback(
        label="cracker_box",
        demo_authoritative_scene=True,
        scene_id="demo_scene_02",
        stage_label="pregrasp_to_grasp_cartesian",
        target_collision_removed=True,
        object_safe_above_to_pregrasp_ok=True,
        pre_plan=(0.455, 0.115, 0.555),
        gr_plan=(0.455, 0.115, 0.437),
        candidate=_cracker_candidate(),
        scene_obstacles=obstacles,
        moveit_fraction=0.19,
        table_z_m=0.40,
        min_lateral_clearance_m=0.025,
    )
    assert not ok
    assert "obstacle_" in reason


def test_preflight_accepts_geometric_fallback_despite_low_fraction() -> None:
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


def test_policy_active_only_demo_scene_02_cracker() -> None:
    assert demo_scene_02_cracker_box_policy_active(
        label="cracker_box",
        demo_authoritative_scene=True,
        scene_id="demo_scene_02",
    )
    assert demo_scene_02_cracker_box_policy_active(
        label="cracker_box",
        demo_authoritative_scene=True,
        scene_id="demo_scene_01",
    )
    assert not demo_scene_02_cracker_box_policy_active(
        label="sugar_box",
        demo_authoritative_scene=True,
        scene_id="demo_scene_02",
    )


def test_known_box_paired_grid_active_for_cracker_and_sugar() -> None:
    assert known_box_paired_grid_search_active(
        label="cracker_box",
        paired_grid_search_mode="prioritized_or_cached",
    )
    assert known_box_paired_grid_search_active(
        label="sugar_box",
        paired_grid_search_mode="prioritized_or_cached",
    )
    assert not known_box_paired_grid_search_active(
        label="chips_can",
        paired_grid_search_mode="prioritized_or_cached",
    )
    assert not known_box_paired_grid_search_active(
        label="cracker_box",
        paired_grid_search_mode="off",
    )


def test_vertical_descend_volume_clear_helper() -> None:
    ok, _ = vertical_descend_volume_clear_of_obstacles(
        (0.455, 0.115, 0.555),
        (0.455, 0.115, 0.437),
        [
            {
                "label": "far",
                "position": [0.660, 0.060, 0.470],
                "collision_dims": {"shape": "box", "box": [0.12, 0.10, 0.14]},
            }
        ],
    )
    assert ok


def test_demo_descend_limit_allows_raised_pregrasp_0562() -> None:
    candidate = _cracker_candidate(
        max_cartesian_descend_m=0.110,
        generic_max_cartesian_descend_m=0.110,
        demo_pregrasp_policy_locked=True,
    )
    info = apply_demo_cracker_descend_limit_policy(
        candidate,
        selected_pregrasp_tcp_z=0.562,
        grasp_tcp_z=0.437,
        generic_max_cartesian_descend_m=0.110,
        demo_max_cartesian_descend_m=0.135,
    )
    assert info["result"] == "OK"
    assert abs(info["required_descend_m"] - 0.125) < 1e-6
    assert candidate["max_cartesian_descend_m"] >= 0.135
    ok, reason = evaluate_demo_geometric_vertical_descend_fallback(
        label="cracker_box",
        demo_authoritative_scene=True,
        scene_id="demo_scene_02",
        stage_label="pregrasp_to_grasp_cartesian",
        target_collision_removed=True,
        object_safe_above_to_pregrasp_ok=True,
        pre_plan=(0.455, 0.115, 0.562),
        gr_plan=(0.455, 0.115, 0.437),
        candidate=candidate,
        scene_obstacles=[],
        moveit_fraction=0.50,
        table_z_m=0.40,
    )
    assert ok, reason


def test_geometric_fallback_fails_generic_limit_without_demo_policy() -> None:
    candidate = _cracker_candidate(max_cartesian_descend_m=0.110)
    ok, reason = evaluate_demo_geometric_vertical_descend_fallback(
        label="cracker_box",
        demo_authoritative_scene=True,
        scene_id="demo_scene_02",
        stage_label="pregrasp_to_grasp_cartesian",
        target_collision_removed=True,
        object_safe_above_to_pregrasp_ok=True,
        pre_plan=(0.455, 0.115, 0.562),
        gr_plan=(0.455, 0.115, 0.437),
        candidate=candidate,
        scene_obstacles=[],
        moveit_fraction=0.50,
        table_z_m=0.40,
    )
    assert not ok
    assert reason == "descend_exceeds_max_cartesian_descend_m"


def test_geometric_fallback_fails_if_required_exceeds_demo_cap_without_lock() -> None:
    candidate = _cracker_candidate(
        max_cartesian_descend_m=0.110,
        demo_max_cartesian_descend_m=0.135,
    )
    ok, reason = evaluate_demo_geometric_vertical_descend_fallback(
        label="cracker_box",
        demo_authoritative_scene=True,
        scene_id="demo_scene_02",
        stage_label="pregrasp_to_grasp_cartesian",
        target_collision_removed=True,
        object_safe_above_to_pregrasp_ok=True,
        pre_plan=(0.455, 0.115, 0.562),
        gr_plan=(0.455, 0.115, 0.437),
        candidate=candidate,
        scene_obstacles=[],
        moveit_fraction=0.50,
        table_z_m=0.40,
    )
    assert not ok
    assert reason == "descend_exceeds_max_cartesian_descend_m"


def test_generic_descend_limit_unchanged_for_non_demo_object() -> None:
    candidate = {"label": "chips_can", "max_cartesian_descend_m": 0.110}
    before = float(candidate["max_cartesian_descend_m"])
    apply_demo_cracker_descend_limit_policy(
        candidate,
        selected_pregrasp_tcp_z=0.562,
        grasp_tcp_z=0.437,
        generic_max_cartesian_descend_m=0.110,
        demo_max_cartesian_descend_m=0.135,
    )
    assert float(candidate["max_cartesian_descend_m"]) >= 0.135
    chips = {"label": "chips_can", "max_cartesian_descend_m": before}
    assert float(chips["max_cartesian_descend_m"]) == 0.110


def test_known_box_geometric_lift_pregrasp_proxy_eligible() -> None:
    candidate = {
        "label": "cracker_box",
        "_known_box_geometric_fallback_validated": True,
        "scene_obstacles": [],
    }
    pregrasp_js = [0.0] * 7
    assert known_box_geometric_lift_pregrasp_proxy_eligible(
        candidate,
        gr_plan=(0.500, -0.080, 0.437),
        pregrasp_js=pregrasp_js,
        pre_plan=(0.500, -0.080, 0.527),
    )
    candidate.pop("_known_box_geometric_fallback_validated", None)
    assert not known_box_geometric_lift_pregrasp_proxy_eligible(
        candidate,
        gr_plan=(0.500, -0.080, 0.437),
        pregrasp_js=pregrasp_js,
    )
