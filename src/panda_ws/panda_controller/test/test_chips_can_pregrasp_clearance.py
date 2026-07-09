"""Regresión: pregrasp directo alto para chips_can cylinder_topdown."""

import logging

from panda_controller.perception_to_pregrasp_test import PerceptionToPregraspTest


class _ClearanceStub(PerceptionToPregraspTest):
    def __init__(self) -> None:
        self._chips_can_force_high_pregrasp_stage = False
        self._chips_can_min_pregrasp_clearance_above_top_m = 0.100
        self._chips_can_low_pregrasp_clearance_above_top_m = 0.100
        self._chips_can_max_cartesian_descend_m = 0.120
        self._max_target_z = 1.20

    def get_logger(self) -> logging.Logger:
        return logging.getLogger("test_chips_can_pregrasp_clearance")


class _HighClearanceStub(_ClearanceStub):
    def __init__(self) -> None:
        super().__init__()
        self._chips_can_force_high_pregrasp_stage = True
        self._chips_can_high_pregrasp_clearance_above_top_m = 0.150
        self._chips_can_min_non_contact_clearance_above_top_m = 0.120
        self._chips_can_top_clearance_epsilon_m = 0.003


def test_pregrasp_clearance_raises_low_direct_pregrasp() -> None:
    stub = _ClearanceStub()
    candidate = {"label": "chips_can", "grasp_strategy": "cylinder_topdown"}
    top_z = 0.511
    seq = {
        "grasp_tcp": (0.649, -0.177, top_z - 0.035),
        "pregrasp_tcp": (0.649, -0.177, top_z + 0.025),
        "final_descend_m": 0.060,
    }
    stub._apply_chips_can_pregrasp_clearance_policy(candidate, seq, top_z)
    assert seq["pregrasp_tcp"][2] == top_z + 0.100
    assert abs(seq["final_descend_m"] - 0.135) < 1e-6


def test_pregrasp_clearance_unchanged_if_already_high_enough() -> None:
    stub = _ClearanceStub()
    candidate = {"label": "chips_can", "grasp_strategy": "cylinder_topdown"}
    top_z = 0.511
    seq = {
        "grasp_tcp": (0.649, -0.177, top_z - 0.035),
        "pregrasp_tcp": (0.649, -0.177, top_z + 0.110),
        "final_descend_m": 0.115,
    }
    stub._apply_chips_can_pregrasp_clearance_policy(candidate, seq, top_z)
    assert seq["pregrasp_tcp"][2] == top_z + 0.110


def test_pregrasp_clearance_skipped_for_other_labels() -> None:
    stub = _ClearanceStub()
    candidate = {"label": "cracker_box", "grasp_strategy": "top_down_short_axis"}
    top_z = 0.511
    seq = {
        "grasp_tcp": (0.649, -0.177, top_z - 0.035),
        "pregrasp_tcp": (0.649, -0.177, top_z + 0.025),
        "final_descend_m": 0.060,
    }
    stub._apply_chips_can_pregrasp_clearance_policy(candidate, seq, top_z)
    assert seq["pregrasp_tcp"][2] == top_z + 0.025


def test_high_policy_uses_low_clearance_not_legacy_075() -> None:
    stub = _HighClearanceStub()
    candidate = {"label": "chips_can", "grasp_strategy": "cylinder_topdown"}
    top_z = 0.510
    seq = {
        "grasp_tcp": (0.580, -0.032, top_z - 0.035),
        "pregrasp_tcp": (0.580, -0.032, top_z + 0.075),
        "final_descend_m": 0.110,
    }
    stub._apply_chips_can_pregrasp_clearance_policy(candidate, seq, top_z)
    stub._apply_chips_can_high_pregrasp_policy(candidate, seq, top_z)
    assert seq["pregrasp_tcp"][2] == top_z + 0.100
    assert seq["object_high_pregrasp_tcp"][2] == top_z + 0.150
