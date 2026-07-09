"""Validación TCP yaw mustard: gap axis vs finger pad axis (sin ROS)."""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple

import numpy as np


def _unit_xy(xy: Any) -> Optional[np.ndarray]:
    if not isinstance(xy, (list, tuple)) or len(xy) < 2:
        return None
    v = np.array([float(xy[0]), float(xy[1])], dtype=np.float64)
    n = float(np.linalg.norm(v))
    if n < 1e-9:
        return None
    return v / n


def commanded_gap_axis_xy_from_tcp_yaw(
    tcp_yaw_rad: float, local_gap_xy: np.ndarray
) -> np.ndarray:
    cf = math.cos(float(tcp_yaw_rad))
    sf = math.sin(float(tcp_yaw_rad))
    lx = float(local_gap_xy[0])
    ly = float(local_gap_xy[1])
    out = np.array([cf * lx - sf * ly, sf * lx + cf * ly], dtype=np.float64)
    n = float(np.linalg.norm(out))
    if n > 1e-9:
        out = out / n
    return out


def mustard_tcp_local_axes_for_gap_param(
    commanded_tcp_yaw_rad: float, selected_gap_axis: str
) -> Tuple[np.ndarray, np.ndarray, str, str]:
    yaw = float(commanded_tcp_yaw_rad)
    tcp_x = np.array([math.cos(yaw), math.sin(yaw)], dtype=np.float64)
    tcp_y = np.array([-math.sin(yaw), math.cos(yaw)], dtype=np.float64)
    axis = str(selected_gap_axis or "x").strip().lower()
    if axis in ("y", "-y"):
        return tcp_y, tcp_x, "tcp_y", "tcp_x"
    return tcp_x, tcp_y, "tcp_x", "tcp_y"


def evaluate_mustard_tcp_yaw_gap_alignment(
    *,
    commanded_tcp_yaw_rad: float,
    local_gap_xy: Tuple[float, float],
    selected_gap_axis: str,
    grasp_gap_axis_xy: Tuple[float, float],
    finger_pad_axis_xy: Tuple[float, float],
    threshold: float = 0.98,
) -> Dict[str, Any]:
    gap_u = _unit_xy(grasp_gap_axis_xy)
    pad_u = _unit_xy(finger_pad_axis_xy)
    local_gap = np.array(local_gap_xy, dtype=np.float64)
    pred_gap = commanded_gap_axis_xy_from_tcp_yaw(
        float(commanded_tcp_yaw_rad), local_gap
    )
    _gap_vec, tcp_pad_vec, local_gap_used, local_pad_used = (
        mustard_tcp_local_axes_for_gap_param(
            float(commanded_tcp_yaw_rad), selected_gap_axis
        )
    )
    dot_gap = abs(float(np.dot(pred_gap, gap_u))) if gap_u is not None else 0.0
    dot_pad = abs(float(np.dot(tcp_pad_vec, pad_u))) if pad_u is not None else 0.0
    dot_gap_ok = dot_gap >= float(threshold)
    dot_pad_ok = dot_pad >= float(threshold)
    return {
        "selected_gap_axis": selected_gap_axis,
        "local_gap_axis_used": local_gap_used,
        "local_finger_pad_axis_used": local_pad_used,
        "dot_tcp_gap_vs_grasp_gap": dot_gap,
        "dot_tcp_pad_vs_finger_pad": dot_pad,
        "dot_gap_ok": dot_gap_ok,
        "dot_pad_ok": dot_pad_ok,
        "result": "OK" if (dot_gap_ok and dot_pad_ok) else "FAIL",
        "allow_continue": dot_gap_ok,
    }
