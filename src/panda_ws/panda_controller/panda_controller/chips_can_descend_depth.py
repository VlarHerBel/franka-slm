"""Utilidades de profundidad de descenso chips_can (sin ROS)."""

from __future__ import annotations


def chips_can_depth_below_top_m(top_z_m: float, actual_tcp_z: float) -> float:
    return float(top_z_m) - float(actual_tcp_z)


def chips_can_depth_gate_ok(
    top_z_m: float,
    actual_tcp_z: float,
    min_depth_below_top_m: float,
) -> bool:
    return (
        chips_can_depth_below_top_m(top_z_m, actual_tcp_z) + 1e-6
        >= float(min_depth_below_top_m)
    )


def chips_can_micro_descend_target_tcp_z(
    actual_tcp_z: float,
    step_m: float,
    desired_grasp_tcp_z: float,
) -> float:
    return max(float(actual_tcp_z) - float(step_m), float(desired_grasp_tcp_z))
