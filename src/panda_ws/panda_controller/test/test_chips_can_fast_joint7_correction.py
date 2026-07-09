"""Regresión: elegibilidad de corrección rápida joint7 para chips_can cylinder_topdown."""

import logging
from typing import Any, Dict
from unittest.mock import patch

from panda_controller.perception_to_pregrasp_test import PerceptionToPregraspTest


class _FastJ7Stub(PerceptionToPregraspTest):
    def __init__(self) -> None:
        self._chips_can_fast_joint7_axis_correction_enabled = True
        self._chips_can_fast_joint7_error_threshold_deg = 20.0
        self._enable_gripper_axis_in_place_correction = True
        self._gripper_open_width_m = 0.0399

    def _effective_open_joint(self) -> float:
        return 0.0399

    def get_logger(self) -> logging.Logger:
        return logging.getLogger("test_chips_can_fast_joint7")


def _candidate(**overrides: Any) -> Dict[str, Any]:
    base: Dict[str, Any] = {
        "label": "chips_can",
        "grasp_strategy": "cylinder_topdown",
    }
    base.update(overrides)
    return base


def test_eligible_chips_can_large_error_joint7_direct() -> None:
    stub = _FastJ7Stub()
    with patch.object(stub, "_gripper_fingers_near_width", return_value=True):
        assert stub._chips_can_fast_joint7_correction_eligible(
            _candidate(), 81.62, "joint7_direct"
        )


def test_not_eligible_small_error() -> None:
    stub = _FastJ7Stub()
    with patch.object(stub, "_gripper_fingers_near_width", return_value=True):
        assert not stub._chips_can_fast_joint7_correction_eligible(
            _candidate(), 15.0, "joint7_direct"
        )


def test_not_eligible_wrong_label() -> None:
    stub = _FastJ7Stub()
    with patch.object(stub, "_gripper_fingers_near_width", return_value=True):
        assert not stub._chips_can_fast_joint7_correction_eligible(
            _candidate(label="cracker_box", grasp_strategy="top_down_short_axis"),
            81.62,
            "joint7_direct",
        )


def test_not_eligible_gripper_not_open() -> None:
    stub = _FastJ7Stub()
    with patch.object(stub, "_gripper_fingers_near_width", return_value=False):
        assert not stub._chips_can_fast_joint7_correction_eligible(
            _candidate(), 81.62, "joint7_direct"
        )
