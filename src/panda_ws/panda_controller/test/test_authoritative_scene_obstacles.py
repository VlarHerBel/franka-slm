"""Tests obstáculos autoritativos RuntimeScene."""

from __future__ import annotations

from panda_controller.authoritative_scene_obstacles import (
    build_authoritative_scene_obstacles,
    expected_authoritative_obstacle_labels,
)


def _scene_objects_demo02() -> list:
    labels = ["cracker_box", "chips_can", "sugar_box", "mustard_bottle"]
    return [
        {
            "label": lb,
            "entity_name": "runtime_ycb_%s_%d_909078" % (lb, i),
            "role": "target" if lb == "cracker_box" else "obstacle",
            "position": [0.45 + 0.05 * i, 0.1 - 0.05 * i, 0.47],
            "collision_dims": {"shape": "box", "box": [0.1, 0.1, 0.1]},
        }
        for i, lb in enumerate(labels)
    ]


def test_cracker_target_includes_chips_can_as_obstacle() -> None:
    scene = _scene_objects_demo02()
    # chips_can sin semantic_center_base pero con position (regresión payload parcial)
    for so in scene:
        if so["label"] == "chips_can":
            so.pop("position", None)
            so["grasp_center_base"] = [0.520, -0.040, 0.385]

    def _build(idx: int, so: dict, is_target: bool) -> dict:
        pos = so.get("grasp_center_base") or so.get("position")
        return {
            "idx": idx,
            "entity_name": so["entity_name"],
            "label": so["label"],
            "position": tuple(pos),
            "is_target": is_target,
            "collision_dims": so.get("collision_dims"),
        }

    obstacles, log = build_authoritative_scene_obstacles(
        scene,
        target_label="cracker_box",
        target_entity="runtime_ycb_cracker_box_0_909078",
        completed_entities=set(),
        completed_labels=set(),
        build_obstacle_fn=_build,
    )
    labels = {o["label"] for o in obstacles if not o.get("is_target")}
    assert labels == {"chips_can", "sugar_box", "mustard_bottle"}
    assert log["result"] == "OK"
    assert "chips_can" in log["obstacles"]


def test_stale_target_role_does_not_drop_chips_can() -> None:
    scene = _scene_objects_demo02()
    for so in scene:
        if so["label"] == "chips_can":
            so["role"] = "target"
            so["semantic_center_base"] = [0.520, -0.040, 0.470]

    def _build(idx: int, so: dict, is_target: bool) -> dict:
        pos = so.get("semantic_center_base") or so.get("position")
        return {
            "entity_name": so["entity_name"],
            "label": so["label"],
            "is_target": is_target,
            "position": tuple(pos),
        }

    obstacles, log = build_authoritative_scene_obstacles(
        scene,
        target_label="cracker_box",
        target_entity="runtime_ycb_cracker_box_0_909078",
        completed_entities=set(),
        completed_labels=set(),
        build_obstacle_fn=_build,
    )
    labels = {o["label"] for o in obstacles if not o.get("is_target")}
    assert "chips_can" in labels
    assert log["result"] == "OK"


def test_completed_object_excluded() -> None:
    scene = _scene_objects_demo02()
    expected = expected_authoritative_obstacle_labels(
        scene,
        target_label="cracker_box",
        target_entity="runtime_ycb_cracker_box_0_909078",
        completed_entities={"runtime_ycb_chips_can_1_909078"},
        completed_labels=set(),
    )
    assert "chips_can" not in expected
    assert "sugar_box" in expected


def test_chips_can_target_excludes_completed_cracker_box() -> None:
    scene = _scene_objects_demo02()

    def _build(idx: int, so: dict, is_target: bool) -> dict:
        pos = so.get("position")
        return {
            "idx": idx,
            "entity_name": so["entity_name"],
            "label": so["label"],
            "position": tuple(pos),
            "is_target": is_target,
            "collision_dims": so.get("collision_dims"),
        }

    obstacles, log = build_authoritative_scene_obstacles(
        scene,
        target_label="chips_can",
        target_entity="runtime_ycb_chips_can_1_909078",
        completed_entities={"runtime_ycb_cracker_box_0_909078"},
        completed_labels={"cracker_box"},
        build_obstacle_fn=_build,
    )
    labels = {o["label"] for o in obstacles if not o.get("is_target")}
    assert labels == {"mustard_bottle", "sugar_box"}
    assert log["result"] == "OK"
    assert "cracker_box" not in log["obstacles"]
    assert log["obstacles"] == ["mustard_bottle", "sugar_box"]


def test_sugar_target_excludes_static_objects_not_in_live_perception() -> None:
    scene = _scene_objects_demo02()

    def _build(idx: int, so: dict, is_target: bool) -> dict:
        pos = so.get("position")
        return {
            "entity_name": so["entity_name"],
            "label": so["label"],
            "is_target": is_target,
            "position": tuple(pos),
        }

    obstacles, log = build_authoritative_scene_obstacles(
        scene,
        target_label="sugar_box",
        target_entity="runtime_ycb_sugar_box_2_909078",
        completed_entities=set(),
        completed_labels={"chips_can"},
        build_obstacle_fn=_build,
        live_table_labels={"sugar_box"},
    )
    labels = {o["label"] for o in obstacles if not o.get("is_target")}
    assert labels == set()
    assert log["result"] == "OK"
    assert log["obstacles"] == []
