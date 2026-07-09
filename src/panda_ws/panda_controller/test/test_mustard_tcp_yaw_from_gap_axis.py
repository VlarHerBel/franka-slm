"""Regresión: finger pad en tcp_y cuando selected_gap_axis=x (mustard)."""

from __future__ import annotations

import math

import numpy as np

from panda_controller.mustard_tcp_yaw_check import evaluate_mustard_tcp_yaw_gap_alignment


def test_gap_axis_x_validates_tcp_y_for_finger_pad() -> None:
    gap_yaw = 0.75
    grasp_gap = (math.cos(gap_yaw), math.sin(gap_yaw))
    finger_pad = (-math.sin(gap_yaw), math.cos(gap_yaw))
    out = evaluate_mustard_tcp_yaw_gap_alignment(
        commanded_tcp_yaw_rad=gap_yaw,
        local_gap_xy=(1.0, 0.0),
        selected_gap_axis="x",
        grasp_gap_axis_xy=grasp_gap,
        finger_pad_axis_xy=finger_pad,
    )
    assert out["dot_tcp_gap_vs_grasp_gap"] >= 0.98
    assert out["dot_tcp_pad_vs_finger_pad"] >= 0.98
    assert out["local_gap_axis_used"] == "tcp_x"
    assert out["local_finger_pad_axis_used"] == "tcp_y"
    assert out["result"] == "OK"
    assert out["allow_continue"] is True


def test_gap_axis_x_old_tcp_x_pad_check_would_fail() -> None:
    gap_yaw = 0.75
    grasp_gap = (math.cos(gap_yaw), math.sin(gap_yaw))
    finger_pad = (-math.sin(gap_yaw), math.cos(gap_yaw))
    tcp_x = np.array([math.cos(gap_yaw), math.sin(gap_yaw)])
    dot_wrong = abs(float(np.dot(tcp_x, finger_pad)))
    assert dot_wrong < 0.1
