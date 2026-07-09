"""Regresión: top_z de chips_can = centro geométrico + H/2, no el centro como tapa."""

import logging

from panda_controller.perception_to_pregrasp_test import PerceptionToPregraspTest


class _TopZStub(PerceptionToPregraspTest):
    def __init__(self) -> None:
        pass

    def get_logger(self) -> logging.Logger:
        return logging.getLogger("test_chips_can_top_z_resolve")


def test_top_z_corrected_when_payload_uses_geometry_center() -> None:
    stub = _TopZStub()
    candidate = {
        "label": "chips_can",
        "grasp_strategy": "cylinder_topdown",
        "position": (0.580, -0.032, 0.385),
        "top_z_m": 0.385,
        "object_height_m": 0.250,
        "collision_dims": {"shape": "cylinder", "cylinder": [0.0375, 0.250]},
    }
    top_z = stub._chips_can_resolve_top_z_m(candidate, 0.385)
    assert abs(top_z - 0.510) < 1e-4


def test_top_z_kept_when_cap_center_without_top_z_m() -> None:
    """Sin top_z_m: center≈0.511 es tapa; no inflar a center+H/2."""
    stub = _TopZStub()
    candidate = {
        "label": "chips_can",
        "grasp_strategy": "cylinder_topdown",
        "position": (0.520, -0.095, 0.511),
        "object_height_m": 0.250,
        "collision_dims": {"shape": "cylinder", "cylinder": [0.0375, 0.250]},
    }
    top_z = stub._chips_can_resolve_top_z_m(candidate, 0.511)
    assert abs(top_z - 0.511) < 1e-4


def test_top_z_kept_when_vision_reports_cap_center() -> None:
    """demo_scene_02: center≈0.511 ya es tapa; no inflar a center+H/2."""
    stub = _TopZStub()
    candidate = {
        "label": "chips_can",
        "grasp_strategy": "cylinder_topdown",
        "position": (0.520, -0.095, 0.511),
        "top_z_m": 0.511,
        "object_height_m": 0.250,
        "collision_dims": {"shape": "cylinder", "cylinder": [0.0375, 0.250]},
    }
    top_z = stub._chips_can_resolve_top_z_m(candidate, 0.511)
    assert abs(top_z - 0.511) < 1e-4


def test_top_z_kept_when_already_at_cap() -> None:
    stub = _TopZStub()
    candidate = {
        "label": "chips_can",
        "grasp_strategy": "cylinder_topdown",
        "position": (0.580, -0.032, 0.385),
        "top_z_m": 0.510,
        "object_height_m": 0.250,
    }
    top_z = stub._chips_can_resolve_top_z_m(candidate, 0.385)
    assert abs(top_z - 0.510) < 1e-4


def test_apply_correction_updates_candidate() -> None:
    stub = _TopZStub()
    candidate = {
        "label": "chips_can",
        "grasp_strategy": "cylinder_topdown",
        "position": (0.580, -0.032, 0.385),
        "top_z_m": 0.385,
        "db_height_m": 0.250,
    }
    corrected = stub._apply_chips_can_top_z_correction(candidate)
    assert abs(corrected - 0.510) < 1e-4
    assert abs(float(candidate["top_z_m"]) - 0.510) < 1e-4
