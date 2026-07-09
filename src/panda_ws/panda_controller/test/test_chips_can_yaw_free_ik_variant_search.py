"""Regresión: barrido yaw_free en IK/plan pregrasp chips_can."""

import logging
import math

from panda_controller.perception_to_pregrasp_test import PerceptionToPregraspTest


class _YawFreeStub(PerceptionToPregraspTest):
    def __init__(self) -> None:
        self._gripper_physical_yaw_correction_rad = 0.0
        self._moveit2 = None
        self._moveit_target_link = "panda_hand"
        self._planning_frame = "world"
        self._plan_before_prelude_orientation_tolerance = 0.05
        self._tf_buffer = None

    def get_logger(self) -> logging.Logger:
        return logging.getLogger("test_chips_can_yaw_free_ik")

    def _is_chips_can_cylinder_topdown_candidate(self, candidate):  # type: ignore[override]
        return str(candidate.get("label", "")).lower() == "chips_can"


def test_build_chips_can_yaw_free_candidate_yaws_includes_absolute_angles() -> None:
    stub = _YawFreeStub()
    stub._chips_can_yaw_nearest_to_current_tcp = lambda _phys: None  # type: ignore[method-assign]
    js = type(
        "JS",
        (),
        {
            "name": ["panda_joint7"],
            "position": [0.25],
        },
    )()
    yaws = stub._build_chips_can_yaw_free_candidate_yaws(1.6351, js)
    names = [n for n, _ in yaws]
    values = [round(v, 3) for _, v in yaws]
    assert "commanded_yaw" in names
    assert "commanded_yaw_pi" in names
    assert "top_down_yaw_zero" in names
    assert "top_down_yaw_pi_over_2" in names
    assert "top_down_yaw_neg_pi_over_2" in names
    assert "top_down_yaw_pi" in names
    assert "yaw_from_current_joint7_min_travel" in names
    assert 0.0 in values
    assert round(math.pi / 2.0, 3) in values
    assert round(-math.pi / 2.0, 3) in values


def test_ranked_pregrasp_yaw_variants_uses_expanded_search_for_chips_can() -> None:
    stub = _YawFreeStub()
    candidate = {"label": "chips_can", "grasp_strategy": "cylinder_topdown"}
    calls: list[float] = []

    def _fake_ranked(base_yaw, tcp_pose, start_js, cand):  # type: ignore[no-untyped-def]
        calls.append(float(base_yaw))
        return [("fake", (0.0, 1.0, 0.0, 0.0), 0.0, object(), 0.1)]

    stub._ranked_chips_can_yaw_free_variants_for_pose = _fake_ranked  # type: ignore[method-assign]
    ranked = stub._ranked_pregrasp_yaw_variants_for_pose(
        candidate, 1.2, (0.6, -0.2, 0.58), None
    )
    assert len(ranked) == 1
    assert calls == [1.2]


def test_ranked_pregrasp_yaw_variants_keeps_legacy_for_other_labels() -> None:
    stub = _YawFreeStub()
    candidate = {"label": "mustard_bottle", "grasp_strategy": "short_axis"}
    calls: list[float] = []

    def _legacy(base_yaw, tcp_pose, start_js):  # type: ignore[no-untyped-def]
        calls.append(float(base_yaw))
        return []

    stub._ranked_downward_yaw_variants_for_pose = _legacy  # type: ignore[method-assign]
    stub._ranked_pregrasp_yaw_variants_for_pose(
        candidate, 0.5, (0.5, 0.0, 0.6), None
    )
    assert calls == [0.5]
