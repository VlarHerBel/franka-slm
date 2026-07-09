"""Regresión: EXECUTION_PREFLIGHT autoriza sugar_box defer final descend."""

import logging

from panda_controller.perception_to_pregrasp_test import PerceptionToPregraspTest
from panda_controller.sugar_box_depth_search import (
    SUGAR_BOX_DEFER_FINAL_DESCEND_PREFLIGHT_SOURCE,
)


class _PreflightStub(PerceptionToPregraspTest):
    def __init__(self) -> None:
        self._plan_before_prelude = True
        self._dry_run = False
        self._moveit2 = object()
        self._cartesian_fraction_threshold = 0.95

    def get_logger(self) -> logging.Logger:
        return logging.getLogger("test_sugar_box_deferred_execution_preflight")

    def _grasp_ik_required_before_pregrasp_from_home(self) -> bool:
        return False

    def _demo_full_pick_route_prevalidation_required(self, candidate: dict) -> bool:
        return True

    def _paired_pregrasp_validation_required(self, candidate: dict) -> bool:
        return False

    def _chips_can_high_pregrasp_policy_active(self, candidate: dict) -> bool:
        return False


def _sugar_defer_candidate() -> dict:
    return {
        "label": "sugar_box",
        "grasp_strategy": "top_down_short_axis",
        "sugar_box_multiobject_safe_route": True,
        "sugar_box_multiobject_safe_route_enabled": True,
        "_object_safe_above_route_prevalidated": True,
        "_cartesian_descend_pending_at_pregrasp": True,
        "_cartesian_descend_prevalidated": False,
        "selected_entry_target": "object_safe_above_tcp",
        "object_safe_above_plan_ok": True,
        "pregrasp_plan_ok": True,
        "route_entry_plan_ok": True,
        "actual_tf_descend_required": True,
        "_plan_before_motion_validated": {
            "ok": True,
            "mode": "direct_pregrasp",
        },
        "_pregrasp_plan_preflight": {
            "ik_pregrasp": "OK",
            "plan_pregrasp": "OK",
            "ik_grasp": "SKIP",
            "plan_before_result": "OK_PREGRASP_PENDING_DESCEND_VALIDATE",
        },
    }


def test_sugar_defer_preflight_authorizes_motion() -> None:
    stub = _PreflightStub()
    status = stub._resolve_pick_route_preflight_plan_status(_sugar_defer_candidate())
    assert status["pregrasp_plan_ok"] is True
    assert status["route_entry_plan_ok"] is True
    assert status["object_safe_above_plan_ok"] is True
    assert status["selected_entry_target"] == "object_safe_above_tcp"
    assert status["cartesian_prevalidated"] is False
    assert status["cartesian_descend_pending_at_pregrasp"] is True
    assert status["actual_tf_descend_required"] is True
    assert status["preflight_source"] == SUGAR_BOX_DEFER_FINAL_DESCEND_PREFLIGHT_SOURCE
    assert status["route_preflight_ok"] is True


def test_sugar_defer_preflight_not_confused_with_full_route() -> None:
    stub = _PreflightStub()
    cand = _sugar_defer_candidate()
    cand["_object_safe_above_route_prevalidated"] = True
    cand["_cartesian_descend_prevalidated"] = False
    status = stub._resolve_pick_route_preflight_plan_status(cand)
    assert status["pregrasp_plan_ok"] is True
    assert status["route_preflight_ok"] is True


def test_cracker_box_unaffected() -> None:
    stub = _PreflightStub()
    cand = {
        "label": "cracker_box",
        "grasp_strategy": "top_down_short_axis",
        "_plan_before_motion_validated": {"ok": True},
        "_pregrasp_plan_preflight": {
            "ik_pregrasp": "OK",
            "plan_pregrasp": "OK",
            "ik_grasp": "OK",
            "plan_before_result": "OK_FULL_PICK_ROUTE_PREVALIDATED",
        },
        "_cartesian_descend_prevalidated": True,
    }
    status = stub._resolve_pick_route_preflight_plan_status(cand)
    assert status["selected_entry_target"] == "pregrasp_tcp"
    assert status["preflight_source"] != SUGAR_BOX_DEFER_FINAL_DESCEND_PREFLIGHT_SOURCE
