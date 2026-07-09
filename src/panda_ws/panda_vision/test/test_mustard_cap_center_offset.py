"""Tests de centro del tapón mustard (offset sobre ejes pad/gap)."""

from panda_vision.grasp.mustard_cap_center import (
    CAP_CENTER_SOURCE,
    apply_mustard_cap_center_calibration,
    compute_mustard_cap_center_xy,
    resolve_mustard_cap_center_offsets,
)


def test_compute_mustard_cap_center_xy_along_axes() -> None:
    body = (1.0, 2.0)
    long_ax = (0.0, 1.0)
    short_ax = (1.0, 0.0)
    cap = compute_mustard_cap_center_xy(body, long_ax, short_ax, 0.01, -0.02)
    assert abs(cap[0] - 0.98) < 1e-6
    assert abs(cap[1] - 2.01) < 1e-6


def test_resolve_offsets_from_candidate_index() -> None:
    off_l, off_s, src = resolve_mustard_cap_center_offsets(candidate_index=1)
    assert abs(off_l - 0.005) < 1e-9
    assert abs(off_s) < 1e-9
    assert "candidates[1]" in src


def test_apply_mustard_cap_center_updates_grasp_center() -> None:
    pose_meta = {
        "label": "mustard_bottle",
        "tall_object_body_center_base": [0.5, 0.6, 0.45],
        "finger_pad_axis_xy": [0.0, 1.0],
        "grasp_gap_axis_xy": [1.0, 0.0],
        "grasp_center_base": [0.5, 0.6, 0.48],
        "top_z_m": 0.48,
    }
    out = apply_mustard_cap_center_calibration(
        pose_meta=pose_meta,
        candidate_index=0,
    )
    assert out["applied"] is True
    assert out["result"] == "OK"
    assert pose_meta["grasp_center_source"] == CAP_CENTER_SOURCE
    assert len(pose_meta["grasp_center_base"]) == 3
    assert abs(float(pose_meta["grasp_center_base"][2]) - 0.48) < 1e-6
