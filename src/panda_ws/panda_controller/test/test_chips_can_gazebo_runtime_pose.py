"""Regresión: chips_can usa centro semántico Gazebo, no solo percepción/golden."""

import logging

from panda_controller.demo_multiobject_target import demo_grasp_policy_sources_ok
from panda_controller.perception_to_pregrasp_test import PerceptionToPregraspTest


class _RuntimePoseStub(PerceptionToPregraspTest):
    def __init__(self) -> None:
        self._dry_run = False

    def get_logger(self) -> logging.Logger:
        return logging.getLogger("test_chips_can_gazebo_runtime_pose")

    def _is_chips_can_cylinder_topdown_candidate(self, candidate):  # type: ignore[override]
        return str(candidate.get("label", "")).lower() == "chips_can"

    def _resolve_target_entity_name(self, candidate):  # type: ignore[override]
        return str(candidate.get("entity_name") or "runtime_ycb_chips_can_1")

    def _sample_gazebo_entity_pose(self, entity_name, collect_sec=0.15):  # type: ignore[override]
        return (
            0.548,
            0.182,
            0.385,
            0.0,
            0.0,
            0.7071,
            0.7071,
        )


def test_enrich_replaces_perception_xy_with_gazebo_semantic_center() -> None:
    stub = _RuntimePoseStub()
    candidate = {
        "label": "chips_can",
        "grasp_strategy": "cylinder_topdown",
        "entity_name": "runtime_ycb_chips_can_1",
        "position": (0.520, -0.095, 0.385),
        "grasp_center_base": [0.520, -0.095, 0.385],
        "top_z_m": 0.511,
        "yaw_source": "runtime_gt_spawn_yaw",
        "closing_yaw_source": "runtime_gt_cylinder_axis",
        "closing_yaw_rad": 1.395,
    }
    assert stub._enrich_chips_can_runtime_pose_from_gazebo(candidate)
    assert candidate["grasp_center_source"] == "runtime_gt_cylinder_center"
    assert candidate["top_face_source"] == "runtime_gt_known_cylinder"
    assert demo_grasp_policy_sources_ok(candidate)
    gx, gy, _ = candidate["grasp_center_base"]
    assert abs(gx - 0.520) > 0.01 or abs(gy - (-0.095)) > 0.01
    assert abs(gx - 0.548) < 0.02
    assert abs(gy - 0.182) < 0.02
    assert candidate["_chips_can_runtime_pose_gazebo_applied"] is True


def test_enrich_skipped_in_dry_run() -> None:
    stub = _RuntimePoseStub()
    stub._dry_run = True
    candidate = {
        "label": "chips_can",
        "grasp_strategy": "cylinder_topdown",
        "position": (0.520, -0.095, 0.385),
    }
    assert not stub._enrich_chips_can_runtime_pose_from_gazebo(candidate)
