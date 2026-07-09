"""Regresión: política high→low pregrasp solo chips_can cylinder_topdown."""

import logging

from panda_controller.perception_to_pregrasp_test import PerceptionToPregraspTest


class _ChipsHighPolicyStub(PerceptionToPregraspTest):
    def __init__(self) -> None:
        self._chips_can_force_high_pregrasp_stage = True
        self._chips_can_high_pregrasp_clearance_above_top_m = 0.150
        self._chips_can_low_pregrasp_clearance_above_top_m = 0.100
        self._chips_can_min_non_contact_clearance_above_top_m = 0.120
        self._chips_can_top_clearance_epsilon_m = 0.003
        self._chips_can_min_pregrasp_clearance_above_top_m = 0.100
        self._chips_can_max_cartesian_descend_m = 0.120
        self._max_target_z = 1.20

    def get_logger(self) -> logging.Logger:
        return logging.getLogger("test_chips_can_high_pregrasp_policy")


def _chips_candidate() -> dict:
    return {"label": "chips_can", "grasp_strategy": "cylinder_topdown"}


def test_high_pregrasp_policy_sets_staging_targets() -> None:
    stub = _ChipsHighPolicyStub()
    candidate = _chips_candidate()
    top_z = 0.510
    seq = {
        "grasp_tcp": (0.580, -0.032, top_z - 0.035),
        "pregrasp_tcp": (0.580, -0.032, top_z + 0.075),
        "final_descend_m": 0.110,
    }
    stub._apply_chips_can_high_pregrasp_policy(candidate, seq, top_z)
    assert bool(seq["uses_low_object_high_approach_stage"])
    assert seq["desired_high_pregrasp_tcp"][2] == top_z + 0.150
    assert seq["object_high_pregrasp_tcp"][2] == top_z + 0.150
    assert seq["pregrasp_tcp"][2] == top_z + 0.100
    assert bool(candidate["uses_low_object_high_approach_stage"])


def test_high_pregrasp_policy_inactive_for_cracker_box() -> None:
    stub = _ChipsHighPolicyStub()
    candidate = {"label": "cracker_box", "grasp_strategy": "top_down_short_axis"}
    top_z = 0.385
    seq = {
        "grasp_tcp": (0.580, -0.032, top_z - 0.035),
        "pregrasp_tcp": (0.580, -0.032, top_z + 0.075),
    }
    stub._apply_chips_can_high_pregrasp_policy(candidate, seq, top_z)
    assert "object_high_pregrasp_tcp" not in seq or seq.get(
        "uses_low_object_high_approach_stage"
    ) is not True


def test_z_safety_guard_clamps_low_request() -> None:
    stub = _ChipsHighPolicyStub()
    candidate = _chips_candidate()
    candidate["chips_can_low_pregrasp_tcp_z"] = 0.485
    top_z = 0.385
    guarded, result = stub._chips_can_guard_tcp_z_non_contact(
        candidate,
        requested_tcp_z=top_z + 0.050,
        phase="pregrasp_motion",
        top_z=top_z,
    )
    assert result == "CLAMPED"
    assert guarded == top_z + 0.120


def test_z_safety_guard_allows_low_pregrasp_in_controlled_descend() -> None:
    stub = _ChipsHighPolicyStub()
    candidate = _chips_candidate()
    candidate["chips_can_low_pregrasp_tcp_z"] = 0.485
    top_z = 0.385
    guarded, result = stub._chips_can_guard_tcp_z_non_contact(
        candidate,
        requested_tcp_z=0.485,
        phase="high_to_low_pregrasp",
        top_z=top_z,
    )
    assert result == "OK"
    assert abs(guarded - 0.485) < 1e-6


def test_z_safety_guard_blocks_below_top_epsilon() -> None:
    stub = _ChipsHighPolicyStub()
    candidate = _chips_candidate()
    top_z = 0.385
    guarded, result = stub._chips_can_guard_tcp_z_non_contact(
        candidate,
        requested_tcp_z=top_z + 0.001,
        phase="yaw_refresh",
        top_z=top_z,
    )
    assert result == "BLOCKED"
    assert guarded == top_z + 0.120
