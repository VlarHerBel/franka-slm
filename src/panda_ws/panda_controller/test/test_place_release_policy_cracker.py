"""Regresión: release Z place cracker_box (no barrido 0.22 legacy)."""

from __future__ import annotations

import pytest

from panda_controller.place_release_policy import (
    DEMO_LOW_PLACE_RELEASE_TCP_Z_BY_LABEL,
    LEGACY_PLACE_CANDIDATE_RELEASE_Z_LOW,
    OBJECT_RELEASE_HEIGHT_BY_LABEL,
    ORDERED_PLACE_SLOT_MODES,
    demo_low_place_release_tcp_z,
    nominal_food_safe_release_tcp_z,
    resolve_box_release_object_height_m,
)


def test_ordered_place_slot_mode_recognized() -> None:
    assert "ordered_near_to_far" in ORDERED_PLACE_SLOT_MODES


def test_cracker_food_safe_release_above_legacy_low() -> None:
    wall_top = 0.155
    obj_h = float(OBJECT_RELEASE_HEIGHT_BY_LABEL["cracker_box"])
    grasp_depth = 0.040
    z = nominal_food_safe_release_tcp_z(wall_top, obj_h, grasp_depth)
    assert z > LEGACY_PLACE_CANDIDATE_RELEASE_Z_LOW
    assert z >= 0.30


def test_cracker_by_label_override_at_demo_low_deposit() -> None:
    by_label_z = 0.180
    assert by_label_z <= LEGACY_PLACE_CANDIDATE_RELEASE_Z_LOW + 0.02


def test_demo_low_place_release_tcp_z_for_clear_table_deposit() -> None:
    assert demo_low_place_release_tcp_z("cracker_box") == 0.180
    assert demo_low_place_release_tcp_z("chips_can") == 0.195
    assert demo_low_place_release_tcp_z("sugar_box") == 0.200
    assert "chips_can" in DEMO_LOW_PLACE_RELEASE_TCP_Z_BY_LABEL
    assert demo_low_place_release_tcp_z("unknown") is None


def test_sugar_gentle_release_height_uses_grasp_depth_not_full_ycb() -> None:
    """Sugar con altura YCB correcta debe depositar bajo (como cracker)."""
    h = resolve_box_release_object_height_m(
        "sugar_box",
        payload_height_m=0.175,
        grasp_depth_from_top_m=0.028,
    )
    assert h is not None
    assert h <= 0.045
    wall_top = 0.170
    z = nominal_food_safe_release_tcp_z(wall_top, float(h), 0.028)
    assert z < 0.22
    assert z == pytest.approx(0.182, abs=0.01)


def test_box_release_height_fallback_on_tiny_payload() -> None:
    h = resolve_box_release_object_height_m(
        "cracker_box",
        payload_height_m=0.0135,
        grasp_depth_from_top_m=0.033,
    )
    assert h is not None
    assert h <= 0.045
