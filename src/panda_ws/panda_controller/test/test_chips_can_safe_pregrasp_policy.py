"""Regresión: chips_can no fuerza safe_pregrasp si global está desactivado."""

import logging

from panda_controller.perception_to_pregrasp_test import PerceptionToPregraspTest


class _SafePregraspPolicyStub(PerceptionToPregraspTest):
    def __init__(self) -> None:
        self._enable_safe_pregrasp_stage = False
        self._chips_can_force_safe_pregrasp_stage = False
        self._chips_can_use_safe_above_stage = True

    def get_logger(self) -> logging.Logger:
        return logging.getLogger("test_chips_can_safe_pregrasp_policy")


def _chips_candidate() -> dict:
    return {"label": "chips_can", "grasp_strategy": "cylinder_topdown"}


def test_chips_can_does_not_force_safe_when_global_disabled() -> None:
    stub = _SafePregraspPolicyStub()
    assert stub._safe_pregrasp_stage_enabled_for_candidate(_chips_candidate()) is False
    assert stub._chips_can_force_safe_pregrasp_stage_active(_chips_candidate()) is False


def test_chips_can_force_requires_global_enable() -> None:
    stub = _SafePregraspPolicyStub()
    stub._chips_can_force_safe_pregrasp_stage = True
    assert stub._safe_pregrasp_stage_enabled_for_candidate(_chips_candidate()) is False
    assert stub._chips_can_force_safe_pregrasp_stage_active(_chips_candidate()) is False
    stub._enable_safe_pregrasp_stage = True
    assert stub._safe_pregrasp_stage_enabled_for_candidate(_chips_candidate()) is True
    assert stub._chips_can_force_safe_pregrasp_stage_active(_chips_candidate()) is True


def test_non_chips_unaffected() -> None:
    stub = _SafePregraspPolicyStub()
    stub._enable_safe_pregrasp_stage = True
    cand = {"label": "cracker_box", "grasp_strategy": "top_down_short_axis"}
    assert stub._chips_can_force_safe_pregrasp_stage_active(cand) is False
    assert stub._safe_pregrasp_stage_enabled_for_candidate(cand) is True


class _HighPregraspPolicyStub(PerceptionToPregraspTest):
    def __init__(self) -> None:
        self._chips_can_force_high_pregrasp_stage = True

    def get_logger(self) -> logging.Logger:
        return logging.getLogger("test_chips_can_high_pregrasp_active")


def test_high_pregrasp_policy_only_chips_cylinder() -> None:
    stub = _HighPregraspPolicyStub()
    assert stub._chips_can_high_pregrasp_policy_active(_chips_candidate()) is True
    mustard = {"label": "mustard_bottle", "grasp_strategy": "tall_object_topdown"}
    assert stub._chips_can_high_pregrasp_policy_active(mustard) is False
    stub._chips_can_force_high_pregrasp_stage = False
    assert stub._chips_can_high_pregrasp_policy_active(_chips_candidate()) is False
