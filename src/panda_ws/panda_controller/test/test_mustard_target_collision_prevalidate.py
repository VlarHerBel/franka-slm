"""Regresión: target collision mustard_bottle con obstacles_count=0."""

from __future__ import annotations

import logging
from typing import Any, List, Tuple

import pytest

from panda_controller.target_collision_policy import (
    build_target_collision_obstacle,
    format_mustard_target_collision_ready_log,
    pick_prevalidate_planning_scene_update_required,
)


def _mustard_candidate_no_obstacles() -> dict[str, Any]:
    return {
        "label": "mustard_bottle",
        "gt_entity_name": "runtime_ycb_mustard_bottle_2_455633",
        "grasp_center_base": [0.660, 0.060, 0.427],
        "top_z_m": 0.427,
        "collision_dims": {
            "shape": "cylinder",
            "cylinder": [0.035, 0.140],
        },
        "db_height_m": 0.140,
        "known_box_yaw_rad": 1.6392,
        "use_target_collision_until_pregrasp": True,
        "remove_target_collision_before_descend": True,
        "scene_obstacles": [],
    }


def test_pick_prevalidate_planning_scene_required_with_zero_obstacles() -> None:
    assert pick_prevalidate_planning_scene_update_required(
        include_target=True,
        scene_obstacles=[],
    )
    assert not pick_prevalidate_planning_scene_update_required(
        include_target=False,
        scene_obstacles=[],
    )


def test_mustard_target_collision_ready_log_format() -> None:
    log = format_mustard_target_collision_ready_log(
        label="mustard_bottle",
        target_collision_present=True,
        object_id="target_runtime_ycb_mustard_bottle_2_455633",
    )
    assert "[MUSTARD_TARGET_COLLISION_READY]" in log
    assert "target_collision_present=true" in log
    assert "object_id=target_runtime_ycb_mustard_bottle_2_455633" in log
    assert "result=OK" in log
    assert format_mustard_target_collision_ready_log(
        label="sugar_box",
        target_collision_present=True,
        object_id="target_sugar",
    ) == ""


def test_mustard_builds_target_collision_obstacle_without_obstacles() -> None:
    candidate = _mustard_candidate_no_obstacles()
    assert pick_prevalidate_planning_scene_update_required(
        include_target=True,
        scene_obstacles=candidate["scene_obstacles"],
    )
    built = build_target_collision_obstacle(candidate, table_z_m=0.428)
    assert built is not None
    assert built["is_target"] is True
    assert built["collision_id"] == "target_runtime_ycb_mustard_bottle_2_455633"


def _make_mustard_planning_scene_stub():
    pytest.importorskip("rclpy")
    from panda_controller.perception_to_pregrasp_test import PerceptionToPregraspTest

    class _MustardPlanningSceneStub(PerceptionToPregraspTest):
        def __init__(self) -> None:
            self._add_detected_objects_to_scene = True
            self._table_z_m = 0.428
            self._collision_object_padding_m = 0.0
            self._planning_frame = "world"
            self._planning_scene_object_ids: set[str] = set()
            self._current_target_collision_id: str | None = None
            self._current_target_collision_spec: dict | None = None
            self._current_target_entity_name: str | None = None
            self._planning_scene_pub = object()
            self._table_collision_applied = False
            self._published: list[
                Tuple[str, str, List[float], Tuple[float, float, float], float]
            ] = []
            self._demo_completed_entities: set[str] = {
                "runtime_ycb_sugar_box_1_455633"
            }

        def get_logger(self) -> logging.Logger:
            return logging.getLogger("test_mustard_target_collision_prevalidate")

        def _apply_table_collision_if_needed(self) -> None:
            self._table_collision_applied = True

        def _demo_object_marked_completed(self, *, entity: str, label: str) -> bool:
            ent = str(entity or "").strip()
            return ent in self._demo_completed_entities

        def _publish_collision_objects(
            self,
            objects: List[
                Tuple[str, str, List[float], Tuple[float, float, float], float]
            ],
        ) -> None:
            for item in objects:
                self._planning_scene_object_ids.add(str(item[0]))
            self._published.extend(objects)

    return _MustardPlanningSceneStub()


def test_mustard_adds_target_collision_when_obstacles_count_zero() -> None:
    stub = _make_mustard_planning_scene_stub()
    candidate = _mustard_candidate_no_obstacles()
    assert candidate["scene_obstacles"] == []

    stub._ensure_pick_prevalidate_planning_scene(candidate)

    target_id = "target_runtime_ycb_mustard_bottle_2_455633"
    assert stub._table_collision_applied is True
    assert stub._current_target_collision_id == target_id
    assert stub._target_collision_present_in_scene(target_id) is True
    assert any(item[0] == target_id for item in stub._published)


def test_mustard_prevalidate_does_not_abort_missing_target_collision() -> None:
    stub = _make_mustard_planning_scene_stub()
    candidate = _mustard_candidate_no_obstacles()
    stub._ensure_pick_prevalidate_planning_scene(candidate)

    target_id = str(stub._current_target_collision_id or "").strip()
    assert stub._include_target_collision(candidate) is True
    assert stub._target_collision_present_in_scene(target_id) is True
    assert candidate.get("_plan_before_motion_fail_reason") != (
        "target_collision_missing_in_planning_scene"
    )
