"""Política de micro-lift cartesiano Z para mustard_bottle tras attach (sin ROS)."""

from __future__ import annotations

from panda_controller.chips_can_lift_cartesian import build_chips_can_lift_hand_z_waypoints

MUSTARD_ATTACHED_TRANSPORT_CARTESIAN_FRACTION_THRESHOLD = 0.25
MUSTARD_LIFT_CARTESIAN_FRACTION_THRESHOLD = 0.20
MUSTARD_LIFT_CARTESIAN_RETRY_FRACTION_THRESHOLD = 0.15

__all__ = [
    "MUSTARD_ATTACHED_TRANSPORT_CARTESIAN_FRACTION_THRESHOLD",
    "MUSTARD_LIFT_CARTESIAN_FRACTION_THRESHOLD",
    "MUSTARD_LIFT_CARTESIAN_RETRY_FRACTION_THRESHOLD",
    "build_chips_can_lift_hand_z_waypoints",
    "mustard_attached_transport_cartesian_fraction_threshold",
    "mustard_lift_cartesian_fraction_threshold",
    "mustard_lift_cartesian_retry_fraction_threshold",
    "mustard_lift_first_micro_rise_m",
    "mustard_lift_min_micro_progress_m",
    "mustard_lift_z_progress_tolerance_m",
    "mustard_lift_xy_drift_tolerance_m",
]


def mustard_lift_first_micro_rise_m() -> float:
    return 0.040


def mustard_lift_min_micro_progress_m() -> float:
    return 0.030


def mustard_lift_z_progress_tolerance_m() -> float:
    return 0.005


def mustard_lift_xy_drift_tolerance_m() -> float:
    return 0.010


def mustard_attached_transport_cartesian_fraction_threshold() -> float:
    return float(MUSTARD_ATTACHED_TRANSPORT_CARTESIAN_FRACTION_THRESHOLD)


def mustard_lift_cartesian_fraction_threshold() -> float:
    return float(MUSTARD_LIFT_CARTESIAN_FRACTION_THRESHOLD)


def mustard_lift_cartesian_retry_fraction_threshold() -> float:
    return float(MUSTARD_LIFT_CARTESIAN_RETRY_FRACTION_THRESHOLD)
