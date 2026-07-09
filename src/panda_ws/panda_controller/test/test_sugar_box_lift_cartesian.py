"""Regresión: gate micro-lift sugar_box con tolerancia."""

from panda_controller.chips_can_lift_cartesian import build_chips_can_lift_hand_z_waypoints
from panda_controller.sugar_box_lift_cartesian import (
    sugar_box_lift_first_micro_rise_m,
    sugar_box_lift_min_micro_progress_m,
    sugar_box_lift_z_progress_tolerance_m,
    sugar_box_lift_xy_drift_tolerance_m,
    sugar_box_z_progress_ok,
)


def test_sugar_micro_lift_constants() -> None:
    assert sugar_box_lift_first_micro_rise_m() == 0.040
    assert sugar_box_lift_min_micro_progress_m() == 0.030
    assert sugar_box_lift_z_progress_tolerance_m() == 0.005
    assert sugar_box_lift_xy_drift_tolerance_m() == 0.010


def test_z_progress_ok_with_demo_tolerance() -> None:
    """Regresión log: 34.6 mm con umbral 30 mm + 5 mm tolerancia."""
    assert sugar_box_z_progress_ok(0.0346, 0.030, tolerance_m=0.005)
    assert not sugar_box_z_progress_ok(0.024, 0.030, tolerance_m=0.005)


def test_lift_waypoints_after_initial_micro() -> None:
    z_start = 0.5174
    z_actual = 0.5520
    z_final = z_start + 0.15
    wps = build_chips_can_lift_hand_z_waypoints(
        z_actual, z_final, max_step_m=0.04, min_step_m=0.01
    )
    assert wps[-1] == z_final
