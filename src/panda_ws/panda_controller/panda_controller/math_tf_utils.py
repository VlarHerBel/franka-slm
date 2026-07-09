"""Utilidades mínimas de quaternion/Euler sin tf_transformations."""

from __future__ import annotations

import math
from typing import Iterable, Tuple

import numpy as np


def normalize_angle_rad(angle: float) -> float:
    return float((float(angle) + math.pi) % (2.0 * math.pi) - math.pi)


def quaternion_from_euler(roll: float, pitch: float, yaw: float) -> Tuple[float, float, float, float]:
    cr = math.cos(float(roll) * 0.5)
    sr = math.sin(float(roll) * 0.5)
    cp = math.cos(float(pitch) * 0.5)
    sp = math.sin(float(pitch) * 0.5)
    cy = math.cos(float(yaw) * 0.5)
    sy = math.sin(float(yaw) * 0.5)
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy
    qw = cr * cp * cy + sr * sp * sy
    return (float(qx), float(qy), float(qz), float(qw))


def euler_from_quaternion(
    qx: float, qy: float, qz: float, qw: float
) -> Tuple[float, float, float]:
    x = float(qx)
    y = float(qy)
    z = float(qz)
    w = float(qw)

    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (w * y - z * x)
    if abs(sinp) >= 1.0:
        pitch = math.copysign(math.pi / 2.0, sinp)
    else:
        pitch = math.asin(sinp)

    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return (float(roll), float(pitch), float(yaw))


def yaw_from_quaternion(quat: Iterable[float]) -> float:
    q = list(quat)
    if len(q) < 4:
        raise ValueError("quat must have 4 components")
    _, _, yaw = euler_from_quaternion(float(q[0]), float(q[1]), float(q[2]), float(q[3]))
    return float(yaw)


def quaternion_multiply(
    q1: Iterable[float], q2: Iterable[float]
) -> Tuple[float, float, float, float]:
    a = list(q1)
    b = list(q2)
    if len(a) < 4 or len(b) < 4:
        raise ValueError("q1 and q2 must have 4 components")
    x1, y1, z1, w1 = float(a[0]), float(a[1]), float(a[2]), float(a[3])
    x2, y2, z2, w2 = float(b[0]), float(b[1]), float(b[2]), float(b[3])
    return (
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
    )


def quaternion_matrix(quat: Iterable[float]) -> np.ndarray:
    q = np.asarray(list(quat), dtype=float).reshape(4)
    x, y, z, w = q
    n = float(np.dot(q, q))
    if n < 1e-16:
        return np.eye(4, dtype=float)
    q = q / math.sqrt(n)
    x, y, z, w = q
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w), 0.0],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w), 0.0],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y), 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=float,
    )
