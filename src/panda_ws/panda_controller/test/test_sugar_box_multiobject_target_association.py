"""Regresión: target_label=sugar_box en escena demo multiobjeto."""

from __future__ import annotations

from panda_controller.demo_multiobject_target import (
    DEMO_KNOWN_BOX_LABELS,
    demo_grasp_policy_sources_ok,
    filter_scene_obstacles_for_target,
    normalize_demo_grasp_sources,
    resolve_target_entity_for_candidate,
)


def _scene_objects_four_demo() -> list:
    return [
        {
            "label": "cracker_box",
            "entity_name": "runtime_ycb_cracker_box_0_493499",
            "role": "target",
            "semantic_center_base": [0.50, 0.12, 0.35],
        },
        {
            "label": "sugar_box",
            "entity_name": "runtime_ycb_sugar_box_1_493499",
            "role": "obstacle",
            "semantic_center_base": [0.709, -0.084, 0.435],
        },
        {
            "label": "mustard_bottle",
            "entity_name": "runtime_ycb_mustard_bottle_2_493499",
            "role": "obstacle",
            "semantic_center_base": [0.71, -0.11, 0.36],
        },
        {
            "label": "chips_can",
            "entity_name": "runtime_ycb_chips_can_3_493499",
            "role": "obstacle",
            "semantic_center_base": [0.58, -0.01, 0.39],
        },
    ]


def _vision_sugar_candidate() -> dict:
    return {
        "label": "sugar_box",
        "score": 0.924,
        "position": (0.709, -0.084, 0.435),
        "grasp_center_base": [0.709, -0.084, 0.435],
        "top_face_source": "runtime_gt_known_box",
        "grasp_center_source": "runtime_gt_box_center",
        "yaw_source": "runtime_gt_spawn_yaw",
        "closing_yaw_rad": -1.615,
        "closing_yaw_source": "",
    }


def _obstacles_with_sugar_wrongly_included() -> list:
    return [
        {
            "entity_name": "runtime_ycb_mustard_bottle_2_493499",
            "label": "mustard_bottle",
            "is_target": False,
        },
        {
            "entity_name": "runtime_ycb_sugar_box_1_493499",
            "label": "sugar_box",
            "is_target": False,
        },
        {
            "entity_name": "runtime_ycb_cracker_box_0_493499",
            "label": "cracker_box",
            "is_target": False,
        },
        {
            "entity_name": "runtime_ycb_chips_can_3_493499",
            "label": "chips_can",
            "is_target": False,
        },
    ]


def test_sugar_box_target_entity_resolve_and_obstacle_filter() -> None:
    scene_objects = _scene_objects_four_demo()
    candidate = _vision_sugar_candidate()
    assert "sugar_box" in DEMO_KNOWN_BOX_LABELS

    ok, method, resolved = resolve_target_entity_for_candidate(
        candidate, scene_objects, scene_objects
    )
    assert ok, method
    assert resolved == "runtime_ycb_sugar_box_1_493499"
    assert method in (
        "candidate_entity",
        "unique_label_executor",
        "unique_label_scene",
        "nearest_xy_scene",
        "nearest_xy_executor",
        "gt_entity",
    )

    filtered, removed = filter_scene_obstacles_for_target(
        candidate, _obstacles_with_sugar_wrongly_included()
    )
    assert "runtime_ycb_sugar_box_1_493499" in removed
    labels_after = {str(o["label"]) for o in filtered}
    assert labels_after == {"cracker_box", "mustard_bottle", "chips_can"}
    assert "sugar_box" not in labels_after


def test_sugar_box_closing_yaw_source_normalize_and_grasp_policy_ok() -> None:
    candidate = _vision_sugar_candidate()
    assert candidate.get("closing_yaw_source") == ""

    applied, new_src = normalize_demo_grasp_sources(candidate)
    assert applied
    assert new_src == "runtime_gt_short_axis"
    assert candidate["closing_yaw_source"] == "runtime_gt_short_axis"
    assert demo_grasp_policy_sources_ok(candidate)


def test_sugar_box_preflight_sources_after_normalize() -> None:
    candidate = _vision_sugar_candidate()
    normalize_demo_grasp_sources(candidate)
    assert demo_grasp_policy_sources_ok(candidate)
    for fld in (
        "top_face_source",
        "grasp_center_source",
        "yaw_source",
        "closing_yaw_source",
    ):
        assert candidate.get(fld)
