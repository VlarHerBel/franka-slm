"""Regresión: métricas de centrado GT chips_can sin KeyError."""

import logging

import numpy as np

from panda_controller.perception_to_pregrasp_test import PerceptionToPregraspTest


class _MetricsStub(PerceptionToPregraspTest):
    def __init__(self) -> None:
        self._chips_can_gt_cylinder_centering_max_error_xy_m = 0.006
        self._chips_can_skip_gripper_gap_alignment_when_gt = True
        self._gripper_centering_max_error_xy_m = 0.012
        self._gripper_centering_max_error_closing_axis_m = 0.005
        self._gripper_centering_max_error_long_axis_m = 0.015

    def get_logger(self) -> logging.Logger:
        return logging.getLogger("test_chips_can_gt_centering_metrics")

    def _is_chips_can_gt_cylinder_candidate(self, candidate):  # type: ignore[no-untyped-def]
        return True

    def _chips_can_skip_gripper_gap_alignment(self, candidate):  # type: ignore[no-untyped-def]
        return True

    def _compute_chips_can_gt_cylinder_centering_metrics(self, candidate):  # type: ignore[no-untyped-def]
        return None

    def _compute_gripper_centering_metrics(self, candidate):  # type: ignore[no-untyped-def]
        return {
            "target_center": np.array([0.55, 0.19]),
            "actual_tcp_center": np.array([0.5505, 0.1905]),
            "actual_source": "finger_midpoint",
            "error_xy_m": 0.0007,
            "error_long_axis_m": 0.0,
            "error_closing_axis_m": 0.0,
        }

    def _verify_gripper_gap_axis_after_pregrasp_pose(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        return True


def _gt_candidate() -> dict:
    return {
        "label": "chips_can",
        "grasp_strategy": "cylinder_topdown",
        "grasp_center_source": "runtime_gt_cylinder_center",
        "top_face_source": "runtime_gt_known_cylinder",
        "_chips_can_gt_centering_verified_at_pregrasp": True,
    }


def test_log_ok_with_actual_finger_midpoint_xy() -> None:
    stub = _MetricsStub()
    metrics = {
        "target_center": np.array([0.55, 0.19]),
        "actual_finger_midpoint_xy": np.array([0.55, 0.19]),
        "error_xy_m": 0.001,
        "centering_target_source": "runtime_gt_cylinder_axis",
    }
    assert stub._log_chips_can_gt_cylinder_centering_verify(metrics, "OK") is True


def test_log_ok_with_actual_tcp_center_fallback() -> None:
    stub = _MetricsStub()
    metrics = {
        "target_center": np.array([0.55, 0.19]),
        "actual_tcp_center": np.array([0.5505, 0.1905]),
        "actual_source": "finger_midpoint",
        "error_xy_m": 0.0007,
    }
    assert stub._log_chips_can_gt_cylinder_centering_verify(metrics, "OK") is True
    normalized = stub._normalize_centering_metrics_dict(metrics)
    assert "actual_finger_midpoint_xy" in normalized


def test_log_unavailable_without_actual_xy() -> None:
    stub = _MetricsStub()
    metrics = {
        "target_center": np.array([0.55, 0.19]),
        "error_xy_m": 0.001,
    }
    assert stub._log_chips_can_gt_cylinder_centering_verify(metrics, "OK") is False


def test_pre_descend_reverify_no_crash_on_tcp_center_metrics() -> None:
    stub = _MetricsStub()
    assert stub._pre_descend_gripper_reverify(_gt_candidate()) is True


def test_normalize_adds_finger_midpoint_from_tcp_center() -> None:
    stub = _MetricsStub()
    metrics = {
        "actual_tcp_center": np.array([1.0, 2.0]),
        "actual_source": "finger_midpoint",
    }
    out = stub._normalize_centering_metrics_dict(metrics)
    assert np.allclose(out["actual_finger_midpoint_xy"], [1.0, 2.0])
