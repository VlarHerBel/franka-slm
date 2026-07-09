"""Regresión: microlift cartesiano Z chips_can."""

from panda_controller.chips_can_lift_cartesian import (
    build_chips_can_lift_hand_z_waypoints,
)


def test_micro_lift_waypoints_150mm() -> None:
    wps = build_chips_can_lift_hand_z_waypoints(0.476, 0.626, max_step_m=0.025)
    assert len(wps) >= 5
    assert abs(wps[0] - 0.501) < 1e-3 or wps[0] > 0.476
    assert abs(wps[-1] - 0.626) < 1e-3
    for a, b in zip(wps, wps[1:]):
        assert b > a
        assert b - a <= 0.025 + 1e-3
