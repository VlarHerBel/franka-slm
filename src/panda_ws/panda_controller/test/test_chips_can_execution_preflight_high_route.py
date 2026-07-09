"""Regresión: EXECUTION_PREFLIGHT acepta entrada object_high para chips_can."""

import logging

from panda_controller.perception_to_pregrasp_test import PerceptionToPregraspTest


class _PreflightStub(PerceptionToPregraspTest):
    def __init__(self) -> None:
        self._chips_can_force_high_pregrasp_stage = True
        self._plan_before_prelude = True
        self._dry_run = False
        self._moveit2 = object()
        self._cartesian_fraction_threshold = 0.95

    def get_logger(self) -> logging.Logger:
        return logging.getLogger("test_chips_can_execution_preflight_high_route")

    def _grasp_ik_required_before_pregrasp_from_home(self) -> bool:
        return False

    def _demo_full_pick_route_prevalidation_required(self, candidate: dict) -> bool:
        return False

    def _paired_pregrasp_validation_required(self, candidate: dict) -> bool:
        return False


def _chips_high_candidate() -> dict:
    return {
        "label": "chips_can",
        "grasp_strategy": "cylinder_topdown",
        "uses_low_object_high_approach_stage": True,
        "_plan_before_motion_validated": {
            "ok": True,
            "mode": "high_pregrasp",
        },
        "_pregrasp_plan_preflight": {
            "ik_pregrasp": "SKIP_HIGH_ENTRY",
            "plan_pregrasp": "OK_OBJECT_HIGH",
            "ik_grasp": "FAIL",
            "plan_before_result": "OK_CHIPS_HIGH_PREGRASP",
            "grasp_ik_missing_allowed": True,
        },
    }


def test_adaptive_entry_preflight_ok() -> None:
    stub = _PreflightStub()
    cand = _chips_high_candidate()
    cand["_pregrasp_plan_preflight"]["plan_before_result"] = (
        "OK_CHIPS_ADAPTIVE_ENTRY_PREGRASP"
    )
    status = stub._resolve_pick_route_preflight_plan_status(cand)
    assert status["object_high_plan_ok"] is True
    assert status["route_entry_plan_ok"] is True


def test_high_route_preflight_ok_with_skip_high_entry() -> None:
    stub = _PreflightStub()
    status = stub._resolve_pick_route_preflight_plan_status(_chips_high_candidate())
    assert status["object_high_plan_ok"] is True
    assert status["route_entry_plan_ok"] is True
    assert status["pregrasp_plan_ok"] is True
    assert status["selected_entry_target"] == "object_high_pregrasp"
    assert status["ik_grasp"] == "FAIL"
    assert status["grasp_ik_required_before_pregrasp"] is False
    assert status["route_preflight_ok"] is True


def test_high_route_blocked_when_full_route_required_without_cartesian() -> None:
    stub = _PreflightStub()
    stub._demo_full_pick_route_prevalidation_required = lambda _c: True  # type: ignore[method-assign]
    status = stub._resolve_pick_route_preflight_plan_status(_chips_high_candidate())
    assert status["object_high_plan_ok"] is True
    assert status["pregrasp_plan_ok"] is False
    assert status["route_preflight_ok"] is False


def test_legacy_low_pregrasp_preflight_fails_without_object_high() -> None:
    stub = _PreflightStub()
    cand = _chips_high_candidate()
    cand["_pregrasp_plan_preflight"] = {
        "ik_pregrasp": "SKIP_HIGH_ENTRY",
        "plan_pregrasp": "OK_OBJECT_HIGH",
        "ik_grasp": "FAIL",
        "plan_before_result": "OK_CHIPS_HIGH_PREGRASP",
    }
    status = stub._resolve_pick_route_preflight_plan_status(cand)
    assert status["pregrasp_plan_ok"] is True
    assert status["route_preflight_ok"] is True

    stub_strict = _PreflightStub()
    stub_strict._demo_full_pick_route_prevalidation_required = lambda _c: True  # type: ignore[method-assign]
    status_strict = stub_strict._resolve_pick_route_preflight_plan_status(cand)
    assert status_strict["pregrasp_plan_ok"] is False
    assert status_strict["route_preflight_ok"] is False

    cand_fail = _chips_high_candidate()
    cand_fail["_pregrasp_plan_preflight"] = {
        "ik_pregrasp": "FAIL",
        "plan_pregrasp": "FAIL",
        "ik_grasp": "FAIL",
        "plan_before_result": "FAIL",
    }
    status_fail = stub._resolve_pick_route_preflight_plan_status(cand_fail)
    assert status_fail["pregrasp_plan_ok"] is False
    assert status_fail["route_entry_plan_ok"] is False
    assert status_fail["route_preflight_ok"] is False


def test_cracker_box_uses_legacy_pregrasp_plan_check() -> None:
    stub = _PreflightStub()
    stub._chips_can_force_high_pregrasp_stage = True
    cand = {
        "label": "cracker_box",
        "grasp_strategy": "top_down_short_axis",
        "_plan_before_motion_validated": {"ok": True},
        "_pregrasp_plan_preflight": {
            "ik_pregrasp": "OK",
            "plan_pregrasp": "OK",
            "ik_grasp": "OK",
            "plan_before_result": "OK",
        },
    }
    status = stub._resolve_pick_route_preflight_plan_status(cand)
    assert status["pregrasp_plan_ok"] is True
    assert status["selected_entry_target"] == "pregrasp_tcp"
    assert status["object_high_plan_ok"] is False
