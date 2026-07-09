"""Regresión preflight multiobjeto demo (cracker/sugar/chips)."""

from __future__ import annotations

from panda_controller.demo_multiobject_target import (
    _det_is_target,
    apply_demo_multiobject_target_pipeline,
    demo_grasp_policy_sources_ok,
    filter_scene_obstacles_for_target,
    merge_grasp_policy_preserving_runtime_gt,
    normalize_demo_grasp_sources,
)


def _executor_four_demo() -> list:
    return [
        {
            "label": "cracker_box",
            "entity_name": "runtime_ycb_cracker_box_0_493499",
            "score": 0.956,
            "position": [0.498, 0.118, 0.470],
            "grasp_center_base": [0.498, 0.118, 0.470],
            "top_face_source": "runtime_gt_known_box",
            "grasp_center_source": "runtime_gt_box_center",
            "yaw_source": "runtime_gt_spawn_yaw",
            "closing_yaw_rad": 2.915,
            "closing_yaw_source": "",
            "collision_dims": {"shape": "box", "box": [0.06, 0.158, 0.21]},
        },
        {
            "label": "sugar_box",
            "entity_name": "runtime_ycb_sugar_box_1_493499",
            "score": 0.924,
            "position": [0.709, -0.084, 0.435],
            "grasp_center_base": [0.709, -0.084, 0.435],
            "top_face_source": "runtime_gt_known_box",
            "grasp_center_source": "runtime_gt_box_center",
            "yaw_source": "runtime_gt_spawn_yaw",
            "closing_yaw_rad": -1.615,
            "closing_yaw_source": "",
        },
        {
            "label": "mustard_bottle",
            "entity_name": "runtime_ycb_mustard_bottle_2_493499",
            "score": 0.965,
            "position": [0.71, -0.11, 0.36],
            "grasp_center_base": [0.71, -0.11, 0.36],
            "top_face_source": "runtime_gt_tall_object",
            "grasp_center_source": "runtime_gt_mustard_mesh_local_cap_center",
            "yaw_source": "runtime_gt_spawn_yaw",
            "closing_yaw_rad": 0.5,
            "closing_yaw_source": "runtime_gt_short_axis",
        },
        {
            "label": "chips_can",
            "entity_name": "runtime_ycb_chips_can_3_493499",
            "score": 0.895,
            "position": [0.571, -0.057, 0.385],
            "grasp_center_base": [0.571, -0.057, 0.385],
            "top_face_source": "runtime_gt_known_cylinder",
            "grasp_center_source": "runtime_gt_cylinder_center",
            "yaw_source": "runtime_gt_spawn_yaw",
            "closing_yaw_rad": 1.395,
            "closing_yaw_source": "",
            "collision_dims": {"shape": "cylinder", "cylinder": [0.0375, 0.25]},
        },
    ]


def _scene_objects_four() -> list:
    objs = _executor_four_demo()
    return [
        {
            "label": o["label"],
            "entity_name": o["entity_name"],
            "role": "target" if o["label"] == "cracker_box" else "obstacle",
            "semantic_center_base": o["position"],
            "collision_dims": o.get("collision_dims"),
            "yaw_source": o.get("yaw_source"),
        }
        for o in objs
    ]


def _candidate_from_executor(label: str) -> dict:
    for o in _executor_four_demo():
        if o["label"] == label:
            c = dict(o)
            c["position"] = tuple(o["position"])
            return c
    raise KeyError(label)


def _obstacles_all_non_target_wrongly() -> list:
    return [
        {
            "entity_name": o["entity_name"],
            "label": o["label"],
            "is_target": False,
            "position": tuple(o["position"]),
        }
        for o in _executor_four_demo()
    ]


def _run_pipeline(label: str) -> dict:
    candidate = _candidate_from_executor(label)
    exec_objs = _executor_four_demo()
    scene_objs = _scene_objects_four()
    apply_demo_multiobject_target_pipeline(
        candidate,
        payload={"objects": exec_objs, "scene_objects": scene_objs},
        scene_objects=scene_objs,
        executor_objects=exec_objs,
        logger=None,
    )
    obstacles = [
        o
        for o in exec_objs
        if not _det_is_target(o, candidate)
    ]
    filtered, removed = filter_scene_obstacles_for_target(
        candidate,
        [
            {
                "entity_name": o["entity_name"],
                "label": o["label"],
                "is_target": False,
                "position": tuple(o["position"]),
            }
            for o in obstacles
        ],
    )
    candidate["scene_obstacles"] = filtered
    return candidate


def test_cracker_box_preflight_sources() -> None:
    c = _run_pipeline("cracker_box")
    assert c["entity_name"] == "runtime_ycb_cracker_box_0_493499"
    assert c["closing_yaw_source"] == "runtime_gt_short_axis"
    assert demo_grasp_policy_sources_ok(c)
    labels = {o["label"] for o in c["scene_obstacles"]}
    assert labels == {"sugar_box", "mustard_bottle", "chips_can"}
    assert len(c["scene_obstacles"]) == 3


def test_sugar_box_preflight_sources() -> None:
    c = _run_pipeline("sugar_box")
    assert c["entity_name"] == "runtime_ycb_sugar_box_1_493499"
    assert c["closing_yaw_source"] == "runtime_gt_short_axis"
    assert demo_grasp_policy_sources_ok(c)
    labels = {o["label"] for o in c["scene_obstacles"]}
    assert "sugar_box" not in labels
    assert len(c["scene_obstacles"]) == 3


def test_chips_can_preflight_sources() -> None:
    c = _run_pipeline("chips_can")
    assert c["entity_name"] == "runtime_ycb_chips_can_3_493499"
    assert c["closing_yaw_source"] == "runtime_gt_cylinder_axis"
    assert demo_grasp_policy_sources_ok(c)
    labels = {o["label"] for o in c["scene_obstacles"]}
    assert "chips_can" not in labels
    assert len(c["scene_obstacles"]) == 3


def test_obstacle_filter_removes_target_only() -> None:
    candidate = _candidate_from_executor("sugar_box")
    candidate["entity_name"] = "runtime_ycb_sugar_box_1_493499"
    candidate["gt_entity_name"] = candidate["entity_name"]
    filtered, removed = filter_scene_obstacles_for_target(
        candidate, _obstacles_all_non_target_wrongly()
    )
    assert "runtime_ycb_sugar_box_1_493499" in removed
    assert len(filtered) == 3


def test_merge_policy_preserves_runtime_gt_sources() -> None:
    entry = {
        "label": "cracker_box",
        "top_face_source": "runtime_gt_known_box",
        "grasp_center_source": "runtime_gt_box_center",
        "yaw_source": "runtime_gt_spawn_yaw",
        "closing_yaw_rad": 1.0,
        "closing_yaw_source": "",
        "entity_name": "runtime_ycb_cracker_box_0",
    }
    policy = {
        "grasp_strategy": "top_down_short_axis",
        "top_face_source": "should_not_overwrite",
        "closing_yaw_source": "bad",
        "required_grasp_width_m": 0.06,
    }
    merge_grasp_policy_preserving_runtime_gt(entry, policy)
    normalize_demo_grasp_sources(entry)
    assert entry["top_face_source"] == "runtime_gt_known_box"
    assert entry["closing_yaw_source"] == "runtime_gt_short_axis"
    assert entry["required_grasp_width_m"] == 0.06
