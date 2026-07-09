"""Pinhole projection: depth + mask to 3D points in the camera optical frame."""

from __future__ import annotations

from typing import Tuple

import numpy as np
from numpy.typing import NDArray
from sensor_msgs.msg import CameraInfo


def scaled_intrinsics_from_camera_info(
    camera_info: CameraInfo, image_height: int, image_width: int
) -> Tuple[float, float, float, float]:
    """Return (fx, fy, cx, cy) scaled to match the actual image resolution.

    Some simulators publish ``K`` for a smaller internal resolution than the
    image topic; detect inconsistency via principal point and scale.
    """
    fx = float(camera_info.k[0])
    fy = float(camera_info.k[4])
    cx = float(camera_info.k[2])
    cy = float(camera_info.k[5])

    expected_cx = image_width / 2.0
    expected_cy = image_height / 2.0

    if cx > 0 and (expected_cx / cx) > 1.5:
        sx = expected_cx / cx
        sy = expected_cy / max(1.0, cy)
        fx *= sx
        fy *= sy
        cx *= sx
        cy *= sy

    return fx, fy, cx, cy


def depth_mask_to_points_camera(
    mask: NDArray[np.bool_],
    depth_m: NDArray[np.floating],
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    z_min: float = 0.05,
) -> NDArray[np.float64]:
    """Project masked pixels to Nx3 points (x, y, z) in the optical frame (+Z forward)."""
    valid = mask & np.isfinite(depth_m) & (depth_m > z_min)
    vs, us = np.where(valid)
    if us.size == 0:
        return np.empty((0, 3), dtype=np.float64)

    zs = depth_m[vs, us].astype(np.float64)
    xs = (us.astype(np.float64) - cx) * zs / fx
    ys = (vs.astype(np.float64) - cy) * zs / fy
    return np.column_stack((xs, ys, zs))
