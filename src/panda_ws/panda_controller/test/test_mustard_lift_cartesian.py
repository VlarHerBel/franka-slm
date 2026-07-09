"""Tests política micro-lift mustard."""

from panda_controller.mustard_lift_cartesian import (
    mustard_attached_transport_cartesian_fraction_threshold,
    mustard_lift_first_micro_rise_m,
    mustard_lift_min_micro_progress_m,
)

from panda_controller.chips_can_lift_cartesian import build_chips_can_lift_hand_z_waypoints


def test_mustard_lift_defaults() -> None:
    from panda_controller.mustard_lift_cartesian import (
        mustard_lift_cartesian_fraction_threshold,
        mustard_lift_cartesian_retry_fraction_threshold,
    )

    assert mustard_lift_first_micro_rise_m() == 0.040
    assert mustard_lift_min_micro_progress_m() == 0.030
    assert mustard_attached_transport_cartesian_fraction_threshold() == 0.25
    assert mustard_lift_cartesian_fraction_threshold() == 0.20
    assert mustard_lift_cartesian_retry_fraction_threshold() == 0.15


def test_mustard_lift_waypoints_from_grasp_height() -> None:
    waypoints = build_chips_can_lift_hand_z_waypoints(
        0.5360, 0.5960, max_step_m=0.040, min_step_m=0.010
    )
    assert len(waypoints) >= 1
    assert waypoints[-1] == 0.5960
