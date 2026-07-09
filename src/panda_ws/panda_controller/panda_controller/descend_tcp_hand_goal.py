"""Conversión TCP -> hand para descenso cartesiano (sin ROS)."""

from __future__ import annotations

from typing import Callable, Optional, Tuple

TcpToMoveItFn = Callable[
    [Tuple[float, float, float], Tuple[float, float, float, float]],
    Tuple[float, float, float],
]


def hand_z_from_tcp_target_delta(
    current_tcp_z: float,
    target_tcp_z: float,
    current_hand_z: float,
) -> float:
    """Mantiene el offset TCP-hand del estado actual (descenso vertical)."""
    return float(current_hand_z) + (float(target_tcp_z) - float(current_tcp_z))


def hand_goal_from_tcp_target(
    orientation_lock: dict,
    target_tcp: Tuple[float, float, float],
    quat: Tuple[float, float, float, float],
    *,
    use_grasp_tcp: bool,
    tcp_to_moveit: Optional[TcpToMoveItFn] = None,
) -> Tuple[float, float, float]:
    hand_pos = orientation_lock["current_hand_position"]
    tcp_pos = orientation_lock.get("current_tcp_position")
    if use_grasp_tcp and tcp_to_moveit is not None:
        return tcp_to_moveit(
            (float(target_tcp[0]), float(target_tcp[1]), float(target_tcp[2])),
            quat,
        )
    if use_grasp_tcp and isinstance(tcp_pos, (list, tuple)) and len(tcp_pos) >= 3:
        hz = hand_z_from_tcp_target_delta(
            float(tcp_pos[2]), float(target_tcp[2]), float(hand_pos[2])
        )
        return (float(target_tcp[0]), float(target_tcp[1]), float(hz))
    return (float(target_tcp[0]), float(target_tcp[1]), float(target_tcp[2]))


def descend_tcp_hand_delta_mismatch(
    current_tcp_z: float,
    target_tcp_z: float,
    current_hand_z: float,
    descend_hand_goal_z: float,
    *,
    tolerance_m: float = 0.01,
) -> Tuple[bool, float, float]:
    expected_hand_dz = float(target_tcp_z) - float(current_tcp_z)
    actual_hand_dz = float(descend_hand_goal_z) - float(current_hand_z)
    ok = abs(expected_hand_dz - actual_hand_dz) <= float(tolerance_m) + 1e-9
    return ok, expected_hand_dz, actual_hand_dz


def descend_goal_semantic_check(
    current_tcp_z: float,
    target_tcp_z: float,
    expected_grasp_tcp_z: float,
    current_hand_z: float,
    target_hand_z: float,
    *,
    min_descend_tcp_m: float = 0.02,
    delta_tolerance_m: float = 0.01,
    grasp_tcp_tolerance_m: float = 0.002,
) -> Tuple[bool, str, float, float]:
    """
    Valida descenso top-down: TCP baja, target_tcp == grasp_tcp, Δhand ≈ Δtcp.
    """
    tcp_dz = float(target_tcp_z) - float(current_tcp_z)
    hand_dz = float(target_hand_z) - float(current_hand_z)
    if float(target_tcp_z) > float(current_tcp_z) - float(min_descend_tcp_m) + 1e-9:
        return (
            False,
            "target_tcp_not_below_current_tcp",
            tcp_dz,
            hand_dz,
        )
    if abs(float(target_tcp_z) - float(expected_grasp_tcp_z)) > float(
        grasp_tcp_tolerance_m
    ):
        return (
            False,
            "target_tcp_not_grasp_tcp",
            tcp_dz,
            hand_dz,
        )
    if abs(hand_dz - tcp_dz) > float(delta_tolerance_m) + 1e-9:
        return (
            False,
            "hand_delta_mismatch_tcp_delta",
            tcp_dz,
            hand_dz,
        )
    return True, "ok", tcp_dz, hand_dz
