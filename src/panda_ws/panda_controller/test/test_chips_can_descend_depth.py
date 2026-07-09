"""Regresión: descenso chips_can desde TF real y depth verify post-descenso."""

import logging
from typing import Any, Dict, Optional, Tuple

import numpy as np

from panda_controller.chips_can_descend_depth import (
    chips_can_depth_gate_ok,
    chips_can_micro_descend_target_tcp_z,
)
from panda_controller.perception_to_pregrasp_test import PerceptionToPregraspTest


class _DescendDepthStub(PerceptionToPregraspTest):
    def __init__(self) -> None:
        self._chips_can_max_cartesian_descend_m = 0.120
        self._chips_can_max_cartesian_descend_from_actual_tf_m = 0.180
        self._chips_can_min_actual_depth_below_top_m = 0.015
        self._chips_can_target_actual_depth_below_top_m = 0.025
        self._chips_can_two_phase_descend_clearance_above_top_m = 0.015
        self._chips_can_enable_post_descend_depth_corrective_descend = False
        self._chips_can_post_descend_depth_corrective_max_m = 0.040
        self._chips_can_descend_tcp_z_tolerance_m = 0.012
        self._chips_can_descend_min_progress_m = 0.010
        self._chips_can_micro_descend_step_m = 0.025
        self._chips_can_micro_descend_max_iterations = 6
        self._chips_can_micro_descend_max_total_m = 0.120
        self._use_grasp_tcp = True
        self._moveit_target_link = "panda_hand"
        self._grasp_tcp_frame = "panda_grasp_tcp"
        self._descend_vel = 0.02
        self._final_tcp: Optional[Tuple[float, float, float]] = None

    def get_logger(self) -> logging.Logger:
        return logging.getLogger("test_chips_can_descend_depth")

    def _is_chips_can_cylinder_topdown_candidate(self, candidate):  # type: ignore[no-untyped-def]
        return True

    def _shifted_tcp_from_plan_targets(self, plan_targets, key, dx, dy):  # type: ignore[no-untyped-def]
        tcp = plan_targets.get(key)
        if tcp is None:
            return None
        return (float(tcp[0]) + dx, float(tcp[1]) + dy, float(tcp[2]))

    def _lookup_tcp_transform(self):  # type: ignore[no-untyped-def]
        return (0.0, 0.0, 0.1)

    def _tcp_position_in_planning_frame(self):  # type: ignore[no-untyped-def]
        return self._final_tcp


def _candidate() -> Dict[str, Any]:
    return {
        "label": "chips_can",
        "grasp_strategy": "cylinder_topdown",
        "top_z_m": 0.511,
        "recommended_grasp_depth_from_top_m": 0.035,
    }


def test_plan_uses_actual_tcp_delta_not_planned_pregrasp_grasp_diff() -> None:
    stub = _DescendDepthStub()
    orientation_lock = {
        "current_tcp_position": (0.555, 0.190, 0.537),
        "current_hand_position": (0.555, 0.190, 0.637),
        "locked_hand_quat": (0.0, 1.0, 0.0, 0.0),
    }
    plan_targets = {"pregrasp_tcp": (0.555, 0.190, 0.586), "grasp_tcp": (0.555, 0.190, 0.476)}
    plan = stub._plan_chips_can_descend_depth(
        _candidate(), orientation_lock, (0.0, 1.0, 0.0, 0.0), plan_targets, 0.0, 0.0
    )
    assert plan is not None
    assert abs(float(plan["desired_grasp_tcp_z"]) - 0.476) < 1e-6
    assert abs(float(plan["required_delta_z"]) - 0.061) < 1e-3
    assert len(plan["segments"]) == 1


def test_post_descend_depth_fail_when_tcp_above_top() -> None:
    stub = _DescendDepthStub()
    stub._final_tcp = (0.5553, 0.1895, 0.5315)
    ok = stub._verify_chips_can_post_descend_depth(_candidate(), allow_corrective=False)
    assert ok is False


def test_post_descend_depth_ok_when_tcp_below_top_enough() -> None:
    stub = _DescendDepthStub()
    stub._final_tcp = (0.555, 0.190, 0.493)
    ok = stub._verify_chips_can_post_descend_depth(_candidate(), allow_corrective=False)
    assert ok is True


def test_micro_descend_target_clamped_to_desired_grasp_z() -> None:
    actual_z = 0.5404
    step = 0.025
    desired = 0.4750
    target = chips_can_micro_descend_target_tcp_z(actual_z, step, desired)
    assert abs(target - 0.5154) < 1e-4
    target2 = chips_can_micro_descend_target_tcp_z(target, step, desired)
    assert target2 > desired


def test_depth_gate_satisfied_at_495_with_top_510() -> None:
    assert chips_can_depth_gate_ok(0.510, 0.495, 0.015) is True
    assert chips_can_depth_gate_ok(0.510, 0.5404, 0.015) is False


def test_two_phase_plan_when_required_delta_exceeds_max_segment() -> None:
    stub = _DescendDepthStub()
    orientation_lock = {
        "current_tcp_position": (0.555, 0.190, 0.650),
        "current_hand_position": (0.555, 0.190, 0.750),
        "locked_hand_quat": (0.0, 1.0, 0.0, 0.0),
    }
    plan_targets = {"pregrasp_tcp": (0.555, 0.190, 0.586), "grasp_tcp": (0.555, 0.190, 0.476)}
    plan = stub._plan_chips_can_descend_depth(
        _candidate(), orientation_lock, (0.0, 1.0, 0.0, 0.0), plan_targets, 0.0, 0.0
    )
    assert plan is not None
    assert plan["single_segment_allowed"] is False
    assert len(plan["segments"]) == 2
