"""Regresión pura: microdescenso y depth gate chips_can."""

from panda_controller.chips_can_descend_depth import (
    chips_can_depth_gate_ok,
    chips_can_micro_descend_target_tcp_z,
)


def test_micro_descend_target_clamped_to_desired_grasp_z() -> None:
    target = chips_can_micro_descend_target_tcp_z(0.5404, 0.025, 0.4750)
    assert abs(target - 0.5154) < 1e-4


def test_depth_gate_at_495_with_top_510() -> None:
    assert chips_can_depth_gate_ok(0.510, 0.495, 0.015) is True
    assert chips_can_depth_gate_ok(0.510, 0.5404, 0.015) is False
