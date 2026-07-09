"""Política de micro-lift cartesiano Z para sugar_box tras attach (sin ROS)."""

from __future__ import annotations

from panda_controller.chips_can_lift_cartesian import build_chips_can_lift_hand_z_waypoints

__all__ = [
    "build_chips_can_lift_hand_z_waypoints",
    "sugar_box_lift_first_micro_rise_m",
    "sugar_box_lift_min_micro_progress_m",
    "sugar_box_lift_z_progress_tolerance_m",
    "sugar_box_lift_xy_drift_tolerance_m",
    "sugar_box_z_progress_ok",
]


def sugar_box_lift_first_micro_rise_m() -> float:
    return 0.040


def sugar_box_lift_min_micro_progress_m() -> float:
    return 0.030


def sugar_box_lift_z_progress_tolerance_m() -> float:
    return 0.005


def sugar_box_lift_xy_drift_tolerance_m() -> float:
    return 0.010


def sugar_box_z_progress_ok(
    actual_progress_m: float,
    min_progress_m: float,
    *,
    tolerance_m: float = 0.005,
) -> bool:
    return float(actual_progress_m) + 1e-6 >= float(min_progress_m) - float(
        tolerance_m
    )
