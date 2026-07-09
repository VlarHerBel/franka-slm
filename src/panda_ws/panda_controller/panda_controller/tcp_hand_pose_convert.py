"""Conversión SE(3) TCP deseado (panda_grasp_tcp) -> pose MoveIt (panda_hand)."""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple

import numpy as np

from panda_controller.math_tf_utils import quaternion_matrix

PREGRASP_ENDPOINT_FK_XY_THRESHOLD_M = 0.003
PREGRASP_ENDPOINT_FK_Z_THRESHOLD_M = 0.003
DEFAULT_SUGAR_BOX_PREGRASP_ENDPOINT_HAND_ORIENTATION_TOL_DEG = 2.0


def quaternion_angle_deg(
    q_a: Tuple[float, float, float, float],
    q_b: Tuple[float, float, float, float],
) -> float:
    a = np.array([float(q_a[0]), float(q_a[1]), float(q_a[2]), float(q_a[3])])
    b = np.array([float(q_b[0]), float(q_b[1]), float(q_b[2]), float(q_b[3])])
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < 1e-9 or nb < 1e-9:
        return 180.0
    a = a / na
    b = b / nb
    dot = float(abs(np.dot(a, b)))
    dot = min(1.0, dot)
    return math.degrees(2.0 * math.acos(dot))


def invert_transform(T: np.ndarray) -> np.ndarray:
    R = T[:3, :3]
    t = T[:3, 3]
    inv = np.eye(4, dtype=float)
    inv[:3, :3] = R.T
    inv[:3, 3] = -R.T @ t
    return inv


def pose_to_matrix(
    position: Tuple[float, float, float],
    quat_xyzw: Tuple[float, float, float, float],
) -> np.ndarray:
    T = quaternion_matrix(quat_xyzw)
    T[0, 3] = float(position[0])
    T[1, 3] = float(position[1])
    T[2, 3] = float(position[2])
    return T


def matrix_to_position_quat(T: np.ndarray) -> Tuple[Tuple[float, float, float], Tuple[float, float, float, float]]:
    R = T[:3, :3]
    pos = (float(T[0, 3]), float(T[1, 3]), float(T[2, 3]))
    trace = float(R[0, 0] + R[1, 1] + R[2, 2])
    if trace > 0.0:
        s = (trace + 1.0) ** 0.5 * 2.0
        qw = 0.25 * s
        qx = (R[2, 1] - R[1, 2]) / s
        qy = (R[0, 2] - R[2, 0]) / s
        qz = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = (1.0 + R[0, 0] - R[1, 1] - R[2, 2]) ** 0.5 * 2.0
        qw = (R[2, 1] - R[1, 2]) / s
        qx = 0.25 * s
        qy = (R[0, 1] + R[1, 0]) / s
        qz = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = (1.0 + R[1, 1] - R[0, 0] - R[2, 2]) ** 0.5 * 2.0
        qw = (R[0, 2] - R[2, 0]) / s
        qx = (R[0, 1] + R[1, 0]) / s
        qy = 0.25 * s
        qz = (R[1, 2] + R[2, 1]) / s
    else:
        s = (1.0 + R[2, 2] - R[0, 0] - R[1, 1]) ** 0.5 * 2.0
        qw = (R[1, 0] - R[0, 1]) / s
        qx = (R[0, 2] + R[2, 0]) / s
        qy = (R[1, 2] + R[2, 1]) / s
        qz = 0.25 * s
    quat = np.array([qx, qy, qz, qw], dtype=float)
    quat = quat / max(float(np.linalg.norm(quat)), 1e-12)
    return pos, (float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3]))


def compose_world_tcp_from_hand(
    hand_position: Tuple[float, float, float],
    hand_quat_xyzw: Tuple[float, float, float, float],
    hand_to_tcp_translation: Tuple[float, float, float],
    hand_to_tcp_quat_xyzw: Tuple[float, float, float, float],
) -> Tuple[Tuple[float, float, float], Tuple[float, float, float, float]]:
    """T_world_tcp = T_world_hand @ T_hand_tcp (TF lookup panda_hand <- panda_grasp_tcp)."""
    T_w_h = pose_to_matrix(hand_position, hand_quat_xyzw)
    T_h_t = pose_to_matrix(hand_to_tcp_translation, hand_to_tcp_quat_xyzw)
    return matrix_to_position_quat(T_w_h @ T_h_t)


def hand_pose_from_desired_tcp(
    desired_tcp_position: Tuple[float, float, float],
    desired_tcp_quat_xyzw: Tuple[float, float, float, float],
    hand_to_tcp_translation: Tuple[float, float, float],
    hand_to_tcp_quat_xyzw: Tuple[float, float, float, float],
) -> Tuple[Tuple[float, float, float], Tuple[float, float, float, float]]:
    """
    T_world_hand = T_world_tcp_desired @ inverse(T_hand_tcp).

    hand_to_tcp_*: transform de panda_grasp_tcp -> panda_hand (lookup TF parent=hand child=tcp).
    """
    T_w_tcp = pose_to_matrix(desired_tcp_position, desired_tcp_quat_xyzw)
    T_h_t = pose_to_matrix(hand_to_tcp_translation, hand_to_tcp_quat_xyzw)
    T_w_h = T_w_tcp @ invert_transform(T_h_t)
    return matrix_to_position_quat(T_w_h)


def hand_position_from_desired_tcp(
    desired_tcp_position: Tuple[float, float, float],
    desired_tcp_quat_xyzw: Tuple[float, float, float, float],
    hand_to_tcp_translation: Tuple[float, float, float],
    hand_to_tcp_quat_xyzw: Tuple[float, float, float, float],
) -> Tuple[float, float, float]:
    hand_pos, _ = hand_pose_from_desired_tcp(
        desired_tcp_position,
        desired_tcp_quat_xyzw,
        hand_to_tcp_translation,
        hand_to_tcp_quat_xyzw,
    )
    return hand_pos


def hand_pose_from_tcp_with_hand_orientation(
    desired_tcp_position: Tuple[float, float, float],
    desired_hand_quat_xyzw: Tuple[float, float, float, float],
    hand_tcp_translation_in_hand_frame: Tuple[float, float, float],
) -> Tuple[Tuple[float, float, float], Tuple[float, float, float, float]]:
    """
    T_base_hand posición: tcp - R_base_hand @ t_hand_tcp (solo traslación local rotada).

    desired_hand_quat_xyzw es la orientación MoveIt del link panda_hand (no del TCP).
    hand_tcp_translation_in_hand_frame: origen TCP expresado en frame hand (TF lookup).
    """
    R = quaternion_matrix(desired_hand_quat_xyzw)[:3, :3]
    t_local = np.array(hand_tcp_translation_in_hand_frame, dtype=float)
    offset_base = R @ t_local
    tcp = np.array(desired_tcp_position, dtype=float)
    hand_pos = (
        float(tcp[0] - offset_base[0]),
        float(tcp[1] - offset_base[1]),
        float(tcp[2] - offset_base[2]),
    )
    return hand_pos, desired_hand_quat_xyzw


def naive_world_z_hand_goal(
    tcp_position: Tuple[float, float, float],
    z_offset_m: float = 0.100,
) -> Tuple[float, float, float]:
    """Regresión: offset fijo +Z mundo (incorrecto con orientación inclinada)."""
    return (
        float(tcp_position[0]),
        float(tcp_position[1]),
        float(tcp_position[2]) + float(z_offset_m),
    )


def xy_compensation_m(
    hand_goal: Tuple[float, float, float],
    naive_hand_goal: Tuple[float, float, float],
) -> float:
    return float(
        math.hypot(
            float(hand_goal[0]) - float(naive_hand_goal[0]),
            float(hand_goal[1]) - float(naive_hand_goal[1]),
        )
    )


def evaluate_pregrasp_endpoint_fk_contract(
    *,
    desired_tcp: Tuple[float, float, float],
    desired_hand_pos: Tuple[float, float, float],
    desired_hand_quat: Tuple[float, float, float, float],
    fk_tcp: Optional[Tuple[float, float, float]],
    fk_hand_pos: Optional[Tuple[float, float, float]] = None,
    fk_hand_quat: Optional[Tuple[float, float, float, float]] = None,
    hand_to_tcp_local: Tuple[float, float, float] = (0.0, 0.0, 0.10),
    hand_orientation_tol_deg: float = DEFAULT_SUGAR_BOX_PREGRASP_ENDPOINT_HAND_ORIENTATION_TOL_DEG,
    xy_threshold_m: float = PREGRASP_ENDPOINT_FK_XY_THRESHOLD_M,
    z_threshold_m: float = PREGRASP_ENDPOINT_FK_Z_THRESHOLD_M,
) -> Dict[str, Any]:
    tcp_err_xy = None
    tcp_err_z = None
    if fk_tcp is not None:
        tcp_err_xy = float(
            math.hypot(
                float(fk_tcp[0]) - float(desired_tcp[0]),
                float(fk_tcp[1]) - float(desired_tcp[1]),
            )
        )
        tcp_err_z = abs(float(fk_tcp[2]) - float(desired_tcp[2]))
    hand_orientation_error_deg = None
    if fk_hand_quat is not None:
        hand_orientation_error_deg = float(
            quaternion_angle_deg(desired_hand_quat, fk_hand_quat)
        )
    R = quaternion_matrix(desired_hand_quat)[:3, :3]
    expected_offset = R @ np.array(hand_to_tcp_local, dtype=float)
    expected_hand_to_tcp_base = (
        float(expected_offset[0]),
        float(expected_offset[1]),
        float(expected_offset[2]),
    )
    actual_hand_to_tcp_base: Optional[Tuple[float, float, float]] = None
    hand_to_tcp_vector_error_xy = None
    if fk_tcp is not None and fk_hand_pos is not None:
        actual_hand_to_tcp_base = (
            float(fk_tcp[0]) - float(fk_hand_pos[0]),
            float(fk_tcp[1]) - float(fk_hand_pos[1]),
            float(fk_tcp[2]) - float(fk_hand_pos[2]),
        )
        hand_to_tcp_vector_error_xy = float(
            math.hypot(
                float(actual_hand_to_tcp_base[0]) - float(expected_hand_to_tcp_base[0]),
                float(actual_hand_to_tcp_base[1]) - float(expected_hand_to_tcp_base[1]),
            )
        )
    orientation_ok = bool(
        hand_orientation_error_deg is not None
        and float(hand_orientation_error_deg) + 1e-9
        <= float(hand_orientation_tol_deg)
    )
    tcp_ok = bool(
        tcp_err_xy is not None
        and tcp_err_z is not None
        and float(tcp_err_xy) + 1e-9 <= float(xy_threshold_m)
        and float(tcp_err_z) + 1e-9 <= float(z_threshold_m)
    )
    ok = bool(orientation_ok and tcp_ok)
    reject_reason = ""
    if not ok:
        if not orientation_ok:
            reject_reason = "pregrasp_endpoint_orientation_contract_fail"
        elif not tcp_ok:
            reject_reason = "pregrasp_endpoint_fk_contract_fail"
    return {
        "desired_tcp": desired_tcp,
        "desired_hand_pos": desired_hand_pos,
        "desired_hand_quat": desired_hand_quat,
        "fk_hand_pos": fk_hand_pos,
        "fk_hand_quat": fk_hand_quat,
        "fk_tcp": fk_tcp,
        "hand_orientation_error_deg": hand_orientation_error_deg,
        "expected_hand_to_tcp_base": expected_hand_to_tcp_base,
        "actual_hand_to_tcp_base": actual_hand_to_tcp_base,
        "hand_to_tcp_vector_error_xy": hand_to_tcp_vector_error_xy,
        "tcp_error_xy": tcp_err_xy,
        "tcp_error_z": tcp_err_z,
        "orientation_ok": orientation_ok,
        "tcp_ok": tcp_ok,
        "ok": ok,
        "reject_reason": reject_reason,
    }


def evaluate_pregrasp_endpoint_orientation_only(
    *,
    desired_hand_quat: Tuple[float, float, float, float],
    fk_hand_quat: Optional[Tuple[float, float, float, float]],
    hand_orientation_tol_deg: float = DEFAULT_SUGAR_BOX_PREGRASP_ENDPOINT_HAND_ORIENTATION_TOL_DEG,
) -> Tuple[bool, Optional[float]]:
    if fk_hand_quat is None:
        return False, None
    err = float(quaternion_angle_deg(desired_hand_quat, fk_hand_quat))
    return bool(err + 1e-9 <= float(hand_orientation_tol_deg)), err


def format_tcp_to_hand_full_transform_log(
    *,
    target_link: str,
    tcp_goal: Tuple[float, float, float],
    hand_goal: Tuple[float, float, float],
    hand_to_tcp_local: Tuple[float, float, float],
    hand_to_tcp_base: Tuple[float, float, float],
    old_world_z_hand_goal: Tuple[float, float, float],
    xy_compensation_m: float,
    result: str = "OK",
) -> str:
    return (
        "[TCP_TO_HAND_FULL_TRANSFORM]\n"
        "target_link=%s\n"
        "tcp_goal=(%.4f, %.4f, %.4f)\n"
        "hand_goal=(%.4f, %.4f, %.4f)\n"
        "hand_to_tcp_local=(%.4f, %.4f, %.4f)\n"
        "hand_to_tcp_base=(%.4f, %.4f, %.4f)\n"
        "old_world_z_hand_goal=(%.4f, %.4f, %.4f)\n"
        "xy_compensation_m=%.4f\n"
        "result=%s"
        % (
            str(target_link),
            float(tcp_goal[0]),
            float(tcp_goal[1]),
            float(tcp_goal[2]),
            float(hand_goal[0]),
            float(hand_goal[1]),
            float(hand_goal[2]),
            float(hand_to_tcp_local[0]),
            float(hand_to_tcp_local[1]),
            float(hand_to_tcp_local[2]),
            float(hand_to_tcp_base[0]),
            float(hand_to_tcp_base[1]),
            float(hand_to_tcp_base[2]),
            float(old_world_z_hand_goal[0]),
            float(old_world_z_hand_goal[1]),
            float(old_world_z_hand_goal[2]),
            float(xy_compensation_m),
            str(result),
        )
    )


def format_pregrasp_endpoint_fk_contract_log(
    fk_result: Dict[str, Any],
    *,
    result: str,
) -> str:
    desired = fk_result.get("desired_tcp") or (0.0, 0.0, 0.0)
    desired_hand = fk_result.get("desired_hand_pos") or (0.0, 0.0, 0.0)
    desired_hand_q = fk_result.get("desired_hand_quat") or (0.0, 0.0, 0.0, 1.0)
    fk_hand_pos = fk_result.get("fk_hand_pos")
    fk_hand_q = fk_result.get("fk_hand_quat")
    fk_tcp = fk_result.get("fk_tcp")
    expected_vec = fk_result.get("expected_hand_to_tcp_base")
    actual_vec = fk_result.get("actual_hand_to_tcp_base")
    orient_err = fk_result.get("hand_orientation_error_deg")
    return (
        "[PREGRASP_ENDPOINT_FK_CONTRACT]\n"
        "desired_tcp=(%.4f, %.4f, %.4f)\n"
        "desired_hand_pos=(%.4f, %.4f, %.4f)\n"
        "desired_hand_quat=(%.4f, %.4f, %.4f, %.4f)\n"
        "fk_hand_pos=%s\n"
        "fk_hand_quat=%s\n"
        "hand_orientation_error_deg=%s\n"
        "expected_hand_to_tcp_base=%s\n"
        "actual_hand_to_tcp_base=%s\n"
        "hand_to_tcp_vector_error_xy=%s\n"
        "fk_tcp=%s\n"
        "tcp_error_xy=%s\n"
        "tcp_error_z=%s\n"
        "result=%s"
        % (
            float(desired[0]),
            float(desired[1]),
            float(desired[2]),
            float(desired_hand[0]),
            float(desired_hand[1]),
            float(desired_hand[2]),
            float(desired_hand_q[0]),
            float(desired_hand_q[1]),
            float(desired_hand_q[2]),
            float(desired_hand_q[3]),
            "n/a"
            if fk_hand_pos is None
            else "(%.4f, %.4f, %.4f)"
            % (float(fk_hand_pos[0]), float(fk_hand_pos[1]), float(fk_hand_pos[2])),
            "n/a"
            if fk_hand_q is None
            else "(%.4f, %.4f, %.4f, %.4f)"
            % (
                float(fk_hand_q[0]),
                float(fk_hand_q[1]),
                float(fk_hand_q[2]),
                float(fk_hand_q[3]),
            ),
            "n/a" if orient_err is None else "%.4f" % float(orient_err),
            "n/a"
            if expected_vec is None
            else "(%.4f, %.4f, %.4f)"
            % (
                float(expected_vec[0]),
                float(expected_vec[1]),
                float(expected_vec[2]),
            ),
            "n/a"
            if actual_vec is None
            else "(%.4f, %.4f, %.4f)"
            % (float(actual_vec[0]), float(actual_vec[1]), float(actual_vec[2])),
            "n/a"
            if fk_result.get("hand_to_tcp_vector_error_xy") is None
            else "%.4f" % float(fk_result["hand_to_tcp_vector_error_xy"]),
            "n/a"
            if fk_tcp is None
            else "(%.4f, %.4f, %.4f)"
            % (float(fk_tcp[0]), float(fk_tcp[1]), float(fk_tcp[2])),
            "n/a"
            if fk_result.get("tcp_error_xy") is None
            else "%.4f" % float(fk_result["tcp_error_xy"]),
            "n/a"
            if fk_result.get("tcp_error_z") is None
            else "%.4f" % float(fk_result["tcp_error_z"]),
            str(result),
        )
    )
