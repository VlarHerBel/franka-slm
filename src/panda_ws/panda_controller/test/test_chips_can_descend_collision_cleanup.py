"""Regresión: retirar target + approach guard antes del descenso chips_can."""

import logging

from panda_controller.perception_to_pregrasp_test import PerceptionToPregraspTest


class _CollisionCleanupStub(PerceptionToPregraspTest):
    def __init__(self) -> None:
        self._planning_scene_object_ids = set()
        self._planning_scene_pub = object()
        self._current_target_collision_id = "target_chips_can"
        self._current_target_collision_spec = {
            "id": "target_chips_can",
            "shape": "cylinder",
            "dims": [0.25, 0.0375],
        }
        self._table_collision_applied = True
        self._cartesian_avoid_collisions = True
        self._planning_scene_settle_after_remove_s = 0.0
        self._planning_frame = "world"
        self._chips_can_approach_collision_inflate_m = 0.015
        self._table_z_m = 0.0
        self._removed: list[str] = []

    def get_logger(self) -> logging.Logger:
        return logging.getLogger("test_chips_can_descend_collision_cleanup")

    def _is_chips_can_cylinder_topdown_candidate(self, candidate):  # type: ignore[override]
        return str(candidate.get("label", "")).lower() == "chips_can"

    def _remove_collision_object(self, object_id: str) -> None:  # type: ignore[override]
        self._removed.append(str(object_id))
        self._planning_scene_object_ids.discard(str(object_id))

    def _publish_collision_objects(self, objects_msg):  # type: ignore[no-untyped-def]
        for col_id, *_rest in objects_msg:
            self._planning_scene_object_ids.add(str(col_id))


def test_approach_guard_uses_separate_collision_id() -> None:
    stub = _CollisionCleanupStub()
    candidate = {
        "label": "chips_can",
        "grasp_strategy": "cylinder_topdown",
        "collision_dims": {"cylinder": [0.0375, 0.25]},
        "top_z_m": 0.51,
        "position": [0.55, 0.19, 0.385],
    }
    stub._planning_scene_object_ids.add("target_chips_can")
    ok = stub._apply_chips_can_approach_collision_guard(
        candidate, "target_chips_can"
    )
    assert ok is True
    guard_id = candidate["_chips_can_approach_collision_guard_id"]
    assert guard_id == "target_chips_can_approach_guard"
    assert "target_chips_can" in stub._planning_scene_object_ids
    assert guard_id in stub._planning_scene_object_ids


def test_cleanup_before_descend_removes_target_and_guard() -> None:
    stub = _CollisionCleanupStub()
    candidate = {
        "label": "chips_can",
        "grasp_strategy": "cylinder_topdown",
        "_chips_can_approach_collision_guard_applied": True,
        "_chips_can_approach_collision_guard_id": "target_chips_can_approach_guard",
    }
    stub._planning_scene_object_ids.update(
        {"target_chips_can", "target_chips_can_approach_guard"}
    )
    assert stub._cleanup_chips_can_collision_before_descend(
        candidate, "target_chips_can"
    )
    assert set(stub._removed) == {
        "target_chips_can",
        "target_chips_can_approach_guard",
    }
    assert candidate["_chips_can_collision_removed_before_descend"] is True
    assert not stub._chips_can_planning_scene_has_collision_id("target_chips_can")
    assert not stub._chips_can_planning_scene_has_collision_id(
        "target_chips_can_approach_guard"
    )
    assert stub._table_collision_exists_in_scene() is True


def test_descend_collision_state_gate_fails_if_target_remains() -> None:
    stub = _CollisionCleanupStub()
    candidate = {"label": "chips_can", "grasp_strategy": "cylinder_topdown"}
    stub._planning_scene_object_ids.add("target_chips_can")
    ok = stub._log_chips_can_descend_collision_state(candidate, "target_chips_can")
    assert ok is False


def test_descend_collision_state_ok_when_cleared() -> None:
    stub = _CollisionCleanupStub()
    candidate = {"label": "chips_can", "grasp_strategy": "cylinder_topdown"}
    ok = stub._log_chips_can_descend_collision_state(candidate, "target_chips_can")
    assert ok is True
