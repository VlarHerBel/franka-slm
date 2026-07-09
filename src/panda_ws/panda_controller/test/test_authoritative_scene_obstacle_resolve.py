"""Tests resolución obstáculos autoritativos con payload merge."""

from __future__ import annotations

from panda_controller.authoritative_scene_obstacle_resolve import (
    resolve_authoritative_scene_object,
)
from panda_controller.authoritative_scene_obstacles import (
    build_authoritative_scene_obstacles,
)


def test_chips_can_without_semantic_center_uses_payload_position() -> None:
    scene_obj = {
        "label": "chips_can",
        "entity_name": "runtime_ycb_chips_can_3_771509",
        "role": "obstacle",
    }
    payload_obj = {
        "label": "chips_can",
        "entity_name": "runtime_ycb_chips_can_3_771509",
        "position": [0.520, -0.040, 0.385],
    }

    def _fake_collision_dims(label: str, padding_m: float):
        assert label == "chips_can"
        return {"shape": "cylinder", "cylinder": [0.0375, 0.25]}

    merged, meta = resolve_authoritative_scene_object(
        scene_obj,
        executor_objects=[payload_obj],
        collision_dims_fn=_fake_collision_dims,
    )
    assert meta["result"] == "OK"
    assert meta["position_source"].startswith("payload_")
    assert merged["position"] == [0.520, -0.040, 0.385]
    assert merged["collision_dims"]["shape"] == "cylinder"


def test_authoritative_set_includes_chips_can_for_cracker_target() -> None:
    labels = ["cracker_box", "chips_can", "sugar_box", "mustard_bottle"]
    scene = [
        {
            "label": lb,
            "entity_name": "runtime_ycb_%s_%d" % (lb, i),
            "role": "target" if lb == "cracker_box" else "obstacle",
            **(
                {}
                if lb == "chips_can"
                else {"position": [0.45 + 0.05 * i, 0.1 - 0.05 * i, 0.385]}
            ),
        }
        for i, lb in enumerate(labels)
    ]
    payload = [
        {
            "label": "chips_can",
            "entity_name": "runtime_ycb_chips_can_1",
            "position": [0.520, -0.040, 0.385],
        }
    ]

    def _build(idx: int, so: dict, is_target: bool):
        merged, meta = resolve_authoritative_scene_object(
            so,
            executor_objects=payload,
            collision_dims_fn=lambda lb, _p: {
                "shape": "box",
                "box": [0.1, 0.1, 0.1],
            },
        )
        if meta["result"] != "OK":
            return None
        pos = merged.get("position")
        return {
            "label": merged["label"],
            "entity_name": merged["entity_name"],
            "position": tuple(pos),
            "is_target": is_target,
            "collision_dims": merged.get("collision_dims"),
        }

    obstacles, log = build_authoritative_scene_obstacles(
        scene,
        target_label="cracker_box",
        target_entity="runtime_ycb_cracker_box_0",
        completed_entities=set(),
        completed_labels=set(),
        build_obstacle_fn=_build,
    )
    obs_labels = {o["label"] for o in obstacles if not o.get("is_target")}
    assert obs_labels == {"chips_can", "sugar_box", "mustard_bottle"}
    assert log["result"] == "OK"
