"""Ejes locales L/W y yaw para top face sintética (sin offsets)."""

import math

import pytest

from panda_vision.spawn.runtime_scene_gt_geometry import (
    compute_synthetic_operational_top_face_base,
    get_known_box_gt_spec,
)


def test_cracker_yaw0_long_axis_along_y():
    out = compute_synthetic_operational_top_face_base(
        (0.5, 0.0, 0.366),
        0.0,
        label="cracker_box",
        apply_yaw_offset=False,
    )
    lx, ly = out["long_axis_xy"]
    assert out["dims_used_lwh"][0] == pytest.approx(0.158, abs=1e-3)
    assert out["dims_used_lwh"][1] == pytest.approx(0.060, abs=1e-3)
    assert abs(ly) > abs(lx)


def test_cracker_yaw0_closing_axis_along_x():
    out = compute_synthetic_operational_top_face_base(
        (0.5, 0.0, 0.366),
        0.0,
        label="cracker_box",
        apply_yaw_offset=False,
    )
    sx, sy = out["short_axis_xy"]
    assert abs(sx) > abs(sy)


def test_cracker_yaw90_long_axis_along_x():
    out = compute_synthetic_operational_top_face_base(
        (0.5, 0.0, 0.366),
        math.pi / 2.0,
        label="cracker_box",
        apply_yaw_offset=False,
    )
    lx, ly = out["long_axis_xy"]
    assert abs(lx) > abs(ly)


def test_cracker_yaw45_edge_lengths():
    out = compute_synthetic_operational_top_face_base(
        (0.5, 0.0, 0.366),
        math.pi / 4.0,
        label="cracker_box",
        apply_yaw_offset=False,
    )
    corners = out["top_face_corners_base"]
    assert len(corners) == 4
    c0 = corners[0]
    c1 = corners[1]
    edge_len = math.hypot(c1[0] - c0[0], c1[1] - c0[1])
    c2 = corners[2]
    edge_wid = math.hypot(c2[0] - c1[0], c2[1] - c1[1])
    long_m, short_m = sorted([edge_len, edge_wid], reverse=True)
    assert long_m == pytest.approx(0.158, abs=1e-3)
    assert short_m == pytest.approx(0.060, abs=1e-3)


def test_overlay_inset_length_only():
    sem = (0.5, 0.0, 0.366)
    op = compute_synthetic_operational_top_face_base(
        sem, 0.0, label="cracker_box", apply_yaw_offset=False
    )
    ov = compute_synthetic_operational_top_face_base(
        sem, 0.0, label="cracker_box", apply_yaw_offset=False, for_overlay=True
    )
    assert ov["dims_used_lwh"][0] < op["dims_used_lwh"][0]
    assert ov["dims_used_lwh"][1] == pytest.approx(op["dims_used_lwh"][1])
    assert ov["for_overlay"] is True


def test_pudding_closing_on_short_axis():
    out = compute_synthetic_operational_top_face_base(
        (0.5, 0.0, 0.28),
        0.0,
        label="pudding_box",
        apply_yaw_offset=False,
    )
    spec = get_known_box_gt_spec("pudding_box")
    assert spec is not None
    assert out["dims_used_lwh"][0] == pytest.approx(0.110, abs=1e-3)
    assert out["dims_used_lwh"][1] == pytest.approx(0.089, abs=1e-3)
    assert spec.local_width_axis == "y"
