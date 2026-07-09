"""Regresión: target collision retirado según planning scene real (no spec cache)."""

import logging

from panda_controller.demo_cracker_collision_off_final_descend import (
    compute_final_descend_safety_metrics,
)
from panda_controller.perception_to_pregrasp_test import PerceptionToPregraspTest


class _FinalDescendTargetRemoveStub(PerceptionToPregraspTest):
    def __init__(self) -> None:
        self._planning_scene_object_ids = set()
        self._planning_scene_pub = object()
        self._current_target_collision_id = "target_cracker_box"
        self._current_target_collision_spec = {
            "id": "target_cracker_box",
            "label": "cracker_box",
            "shape": "box",
            "size": [0.08, 0.16, 0.04],
        }
        self._planning_scene_settle_after_remove_s = 0.0
        self._planning_frame = "world"
        self._table_z_m = 0.40
        self._removed: list[str] = []

    def get_logger(self) -> logging.Logger:
        return logging.getLogger("test_demo_cracker_final_descend_target_remove")

    def _remove_collision_object(self, object_id: str) -> None:  # type: ignore[override]
        self._removed.append(str(object_id))
        self._planning_scene_object_ids.discard(str(object_id))


def _cracker_candidate() -> dict:
    return {
        "label": "cracker_box",
        "gt_entity_name": "cracker_box",
        "grasp_center_base": [0.455, 0.115, 0.437],
    }


def test_known_planning_objects_ignores_stored_spec_when_id_absent() -> None:
    stub = _FinalDescendTargetRemoveStub()
    candidate = _cracker_candidate()
    ids = stub._collect_target_collision_ids_for_removal(candidate, "target_cracker_box")
    assert "target_cracker_box" in ids
    present, _, source = stub._target_collision_present_in_known_planning_objects(
        ids
    )
    assert not present
    assert source == "planning_scene_known_objects"
    assert stub._target_collision_present_in_scene("target_cracker_box") is False


def test_ensure_remove_clears_stale_target_id_from_known_objects() -> None:
    stub = _FinalDescendTargetRemoveStub()
    candidate = _cracker_candidate()
    stub._planning_scene_object_ids.add("target_cracker_box")
    ok = stub._ensure_target_collision_removed_for_final_descend(
        "target_cracker_box",
        candidate,
    )
    assert ok is True
    assert "target_cracker_box" in stub._removed
    present, _, _ = stub._target_collision_present_in_known_planning_objects(
        stub._collect_target_collision_ids_for_removal(candidate, "target_cracker_box")
    )
    assert not present


def test_safety_metrics_target_removed_ok_from_planning_scene_not_spec() -> None:
    """Summary target=none + spec interno: métricas usan target_removed_ok=true."""
    stub = _FinalDescendTargetRemoveStub()
    candidate = _cracker_candidate()
    stub._planning_scene_object_ids.add("target_cracker_box")
    stub._ensure_target_collision_removed_for_final_descend(
        "target_cracker_box",
        candidate,
    )
    ids = stub._collect_target_collision_ids_for_removal(candidate, "target_cracker_box")
    present, _, _ = stub._target_collision_present_in_known_planning_objects(ids)
    target_removed_ok = not bool(present)
    assert target_removed_ok is True
    assert stub._current_target_collision_spec is not None
    metrics = compute_final_descend_safety_metrics(
        pre_plan=(0.455, 0.115, 0.575),
        gr_plan=(0.455, 0.115, 0.437),
        scene_obstacles=[
            {
                "label": "chips_can",
                "position": [0.52, -0.04, 0.45],
                "shape": "box",
                "size": [0.08, 0.08, 0.10],
            }
        ],
        table_z_m=0.40,
        joint_values_7=[0.0] * 7,
        target_removed_ok=target_removed_ok,
        min_lateral_clearance_m=0.025,
    )
    assert metrics["target_removed_ok"] is True
    assert metrics["obstacle_clearance_ok"] is True
