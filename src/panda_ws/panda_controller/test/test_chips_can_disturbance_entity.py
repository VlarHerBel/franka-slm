"""Regresión: chips_can disturbance check usa entity name resuelto."""

import logging

from panda_controller.perception_to_pregrasp_test import PerceptionToPregraspTest


class _DisturbanceStub(PerceptionToPregraspTest):
    def __init__(self) -> None:
        self._chips_can_check_object_disturbance_before_close = True
        self._chips_can_max_object_xy_shift_before_close_m = 0.006
        self._sample_calls: list[str] = []

    def get_logger(self) -> logging.Logger:
        return logging.getLogger("test_chips_can_disturbance")

    def _is_chips_can_cylinder_topdown_candidate(self, candidate):  # type: ignore[override]
        return True

    def _resolve_target_entity_name(self, candidate):  # type: ignore[override]
        return str(
            candidate.get("gt_entity_name")
            or candidate.get("entity_name")
            or "runtime_ycb_chips_can_seed1004"
        )

    def _sample_gazebo_entity_pose(self, entity_name, collect_sec=0.12):  # type: ignore[override]
        self._sample_calls.append(str(entity_name))
        return (
            0.555,
            0.190,
            0.511,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
        )


def test_disturbance_check_uses_resolve_target_entity_name() -> None:
    stub = _DisturbanceStub()
    candidate = {
        "label": "chips_can",
        "grasp_strategy": "cylinder_topdown",
        "gt_entity_name": "runtime_ycb_chips_can_seed1004",
        "_chips_can_snapshot_object_xy": [0.555, 0.190],
        "_chips_can_snapshot_source": "after_object_high_pregrasp",
    }
    ok = stub._chips_can_object_disturbance_check(candidate, stage="before_close")
    assert ok is True
    assert stub._sample_calls == ["runtime_ycb_chips_can_seed1004"]


def test_disturbance_snapshot_refresh_updates_baseline() -> None:
    stub = _DisturbanceStub()
    candidate = {
        "label": "chips_can",
        "grasp_strategy": "cylinder_topdown",
        "grasp_center_source": "runtime_gt_cylinder_center",
        "top_face_source": "runtime_gt_known_cylinder",
        "gt_entity_name": "runtime_ycb_chips_can_seed1002",
        "_chips_can_snapshot_object_xy": [0.580, -0.032],
    }
    ok = stub._refresh_chips_can_disturbance_snapshot_from_gazebo(
        candidate, stage="after_object_high_pregrasp"
    )
    assert ok is True
    assert candidate["_chips_can_snapshot_object_xy"] == [0.555, 0.190]
    assert candidate["_chips_can_snapshot_source"] == "after_object_high_pregrasp"
