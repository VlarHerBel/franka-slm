"""Regresión: búsqueda adaptativa de Z de entrada chips_can."""

import logging

from panda_controller.perception_to_pregrasp_test import PerceptionToPregraspTest


class _AdaptiveEntryStub(PerceptionToPregraspTest):
    def __init__(self) -> None:
        self._chips_can_force_high_pregrasp_stage = True
        self._chips_can_high_pregrasp_clearance_above_top_m = 0.150
        self._chips_can_low_pregrasp_clearance_above_top_m = 0.100
        self._chips_can_min_non_contact_clearance_above_top_m = 0.120
        self._max_target_z = 1.20

    def get_logger(self) -> logging.Logger:
        return logging.getLogger("test_chips_can_adaptive_entry_z")


def test_entry_clearance_search_order_descends_by_01() -> None:
    stub = _AdaptiveEntryStub()
    order = stub._chips_can_entry_clearance_search_order()
    assert order[0] == 0.150
    assert order[-1] == 0.100
    assert 0.120 in order
    assert 0.130 in order
    assert order.index(0.150) < order.index(0.120) < order.index(0.100)


def test_high_policy_sets_desired_and_low_separate_from_selected() -> None:
    stub = _AdaptiveEntryStub()
    candidate = {"label": "chips_can", "grasp_strategy": "cylinder_topdown"}
    top_z = 0.510
    seq = {
        "grasp_tcp": (0.580, -0.032, top_z - 0.035),
        "pregrasp_tcp": (0.580, -0.032, top_z + 0.075),
    }
    stub._apply_chips_can_high_pregrasp_policy(candidate, seq, top_z)
    assert seq["desired_high_pregrasp_tcp"][2] == top_z + 0.150
    assert seq["pregrasp_tcp"][2] == top_z + 0.100
    assert candidate["chips_can_desired_high_pregrasp_tcp_z"] == top_z + 0.150
    assert candidate["chips_can_low_pregrasp_tcp_z"] == top_z + 0.100
    assert candidate["chips_can_selected_entry_pregrasp_tcp_z"] is None


def test_apply_selected_entry_updates_object_high_tcp() -> None:
    stub = _AdaptiveEntryStub()
    candidate = {"label": "chips_can", "grasp_strategy": "cylinder_topdown"}
    targets = {
        "object_high_pregrasp_tcp": (0.58, -0.03, 0.66),
        "pregrasp_tcp": (0.58, -0.03, 0.61),
    }
    pick = {
        "entry_tcp": (0.58, -0.03, 0.63),
        "entry_plan": (0.58, -0.03, 0.63),
        "clearance_above_top": 0.120,
    }
    entry_tcp, _ = stub._apply_chips_can_selected_entry_pregrasp(
        candidate, targets, pick
    )
    assert entry_tcp[2] == 0.63
    assert targets["object_high_pregrasp_tcp"][2] == 0.63
    assert candidate["chips_can_selected_entry_pregrasp_tcp_z"] == 0.63
