"""Regresión: centering edge_grasp usa target desplazado, no centro geométrico."""

import numpy as np

from panda_controller.grasp_centering_target import (
    centering_error_xy_m,
    resolve_gripper_centering_target_xy,
)


def test_edge_grasp_centering_uses_commanded_hand_not_object_center() -> None:
    obj = {
        "label": "pudding_box",
        "grasp_strategy": "edge_grasp",
        "edge_grasp_requested": True,
        "edge_offset_m": 0.015,
        "known_box_center_base": [0.5651, -0.0251, 0.285],
        "grasp_center_base": [0.5651, -0.0251, 0.285],
        "_runtime_pregrasp_tcp_xy": [0.562, -0.010],
    }
    commanded = (0.562, -0.010)
    target, source = resolve_gripper_centering_target_xy(
        obj, commanded_hand_xy=commanded
    )
    assert source == "commanded_pregrasp_hand_xy"
    assert target is not None
    assert abs(float(target[0]) - 0.562) < 1e-6
    assert abs(float(target[1]) - (-0.010)) < 1e-6

    actual = np.array([0.5614, -0.0103], dtype=np.float64)
    err = centering_error_xy_m(target, actual)
    assert err < 0.012
    assert err < 0.002
