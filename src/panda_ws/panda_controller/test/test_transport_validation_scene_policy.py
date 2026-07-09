"""Transport exit debe validarse con target retirado del mundo."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple

from panda_controller.paired_joint7_offline_sim import joint_state_from_positions
from panda_controller.perception_to_pregrasp_test import PerceptionToPregraspTest


class _TransportSceneStub(PerceptionToPregraspTest):
    def __init__(self) -> None:
        self._table_z_m = 0.270
        self._current_target_collision_id = "target_runtime_ycb_cracker_box_0"
        self._current_target_collision_spec: Optional[Dict[str, Any]] = None
        self._current_target_entity_name = "runtime::ycb_cracker_box_0"
        self._planning_scene_settle_after_remove_s = 0.0
        self._planning_scene_object_ids: List[str] = [
            "target_runtime_ycb_cracker_box_0",
            "obstacle_runtime_ycb_chips_can_1",
        ]
        self._motion_waypoints_data = {"carry_mid_high": {"joints": {}}}
        self._demo_scene_policy = {}
        self._add_detected_objects_to_scene = True
        self._restore_calls = 0
        self._apply_calls: List[bool] = []

    def get_logger(self) -> logging.Logger:
        return logging.getLogger("test_transport_validation_scene_policy")

    def _apply_paired_grid_planning_scene(
        self,
        candidate: Dict[str, Any],
        scene_obstacles: Sequence[Dict[str, Any]],
        *,
        include_target: bool,
    ) -> None:
        self._apply_calls.append(bool(include_target))

    def _apply_table_collision_if_needed(self) -> None:
        return None

    def _ensure_target_collision_removed_for_final_descend(
        self,
        target_collision_id: Optional[str],
        candidate: Dict[str, Any],
        *,
        timeout_sec: float = 1.0,
    ) -> bool:
        for tid in self._collect_target_collision_ids_for_removal(
            candidate, target_collision_id
        ):
            if tid in self._planning_scene_object_ids:
                self._planning_scene_object_ids.remove(tid)
        return True

    def _remove_collision_object(self, obj_id: str) -> None:
        if obj_id in self._planning_scene_object_ids:
            self._planning_scene_object_ids.remove(obj_id)

    def _restore_planning_scene_after_lift_prevalidation(
        self,
        candidate: Dict[str, Any],
        scene_obstacles: Optional[Sequence[Dict[str, Any]]] = None,
    ) -> None:
        self._restore_calls += 1
        self._planning_scene_object_ids.append("target_runtime_ycb_cracker_box_0")

    def _joint_values_7d_from_any(self, js, context=""):  # type: ignore[no-untyped-def]
        if hasattr(js, "position"):
            return list(js.position)[:7]
        return list(js)[:7]

    def _fk_hand_position_from_joint_state(self, js):  # type: ignore[no-untyped-def]
        return (0.456, 0.115, 0.587)

    def _fk_tcp_position_from_joint_state(self, js):  # type: ignore[no-untyped-def]
        return (0.456, 0.115, 0.487)


def test_transport_scene_prepare_removes_target_from_world() -> None:
    stub = _TransportSceneStub()
    candidate = {
        "label": "cracker_box",
        "scene_obstacles": [
            {"label": "chips_can", "is_target": False},
            {"label": "cracker_box", "is_target": True},
        ],
        "prevalidated_lift_endpoint_js": joint_state_from_positions([0.1] * 7),
        "prevalidated_lift_hand_from_endpoint": (0.456, 0.115, 0.587),
        "prevalidated_lift_tcp_from_endpoint": (0.456, 0.115, 0.487),
    }
    ok, after_present = stub._prepare_planning_scene_for_attached_transport_validation(
        candidate,
        "target_runtime_ycb_cracker_box_0",
        candidate["scene_obstacles"],
        candidate_idx=13,
    )
    assert ok is True
    assert after_present is False
    assert "target_runtime_ycb_cracker_box_0" not in stub._planning_scene_object_ids
    assert stub._apply_calls == [False]
    assert candidate.get("transport_scene_policy") == (
        "target_removed_for_attached_transport_validation"
    )


def test_compute_transport_score_aborts_when_target_still_in_world() -> None:
    stub = _TransportSceneStub()

    def _prepare_fail(*args, **kwargs):  # type: ignore[no-untyped-def]
        return False, True

    stub._prepare_planning_scene_for_attached_transport_validation = _prepare_fail
    candidate = {
        "label": "cracker_box",
        "_scene_policy": {"transport_policy": {}},
        "scene_obstacles": [{"label": "chips_can", "is_target": False}],
        "prevalidated_lift_endpoint_js": joint_state_from_positions([0.1] * 7),
        "prevalidated_lift_hand_from_endpoint": (0.456, 0.115, 0.587),
    }
    score = stub._compute_transport_aware_pick_score(
        candidate,
        plan_targets={"grasp_tcp": (0.456, 0.115, 0.437)},
        grasp_joints=joint_state_from_positions([0.0] * 7),
        yaw_variant=0.0,
        candidate_idx=13,
        lift_ok=True,
    )
    assert score is not None
    assert score.get("result") == "REJECT"
    assert (
        score.get("reject_reason") == "transport_scene_target_still_world_collision"
    )
