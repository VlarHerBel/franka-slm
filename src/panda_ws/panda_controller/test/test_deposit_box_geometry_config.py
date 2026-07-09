"""Tests: geometría nominal bandeja compacta."""

from panda_controller.deposit_layout_policy import (
    DEFAULT_PLACE_SLOTS_ORDERED,
    DEPOSIT_BOX_FLOOR_TOP_Z_M,
    DEPOSIT_BOX_INTERIOR_X_M,
    DEPOSIT_BOX_INTERIOR_Y_M,
    DEPOSIT_BOX_WALL_TOP_Z_M,
    DEPOSIT_BOX_X_MAX_M,
    DEPOSIT_BOX_X_MIN_M,
    DEPOSIT_BOX_Y_MAX_M,
    DEPOSIT_BOX_Y_MIN_M,
)


def test_tray_covers_slot_extents() -> None:
    xs = [float(s["x"]) for s in DEFAULT_PLACE_SLOTS_ORDERED]
    ys = [float(s["y"]) for s in DEFAULT_PLACE_SLOTS_ORDERED]
    margin = 0.08
    assert min(xs) - margin >= DEPOSIT_BOX_X_MIN_M - 0.02
    assert max(xs) + margin <= DEPOSIT_BOX_X_MAX_M + 0.02
    assert min(ys) - margin >= DEPOSIT_BOX_Y_MIN_M - 0.02
    assert max(ys) + margin <= DEPOSIT_BOX_Y_MAX_M + 0.02


def test_compact_interior_size() -> None:
    assert 0.35 <= DEPOSIT_BOX_INTERIOR_X_M <= 0.40
    assert 0.35 <= DEPOSIT_BOX_INTERIOR_Y_M <= 0.40


def test_wall_top_z_demo_value() -> None:
    assert DEPOSIT_BOX_WALL_TOP_Z_M == 0.17
    assert DEPOSIT_BOX_WALL_TOP_Z_M <= 0.20
    assert DEPOSIT_BOX_FLOOR_TOP_Z_M == 0.01
