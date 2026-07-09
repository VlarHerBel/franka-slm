"""Tests de corrección SDF (origen modelo → centro tapón real) para mustard_bottle."""

import math

from panda_vision.grasp.object_grasp_policy import resolve_tall_object_top_z_m
from panda_vision.spawn.known_object_geometry import (
    MUSTARD_CAP_CENTER_MODE_SDF_OFFSET,
    MUSTARD_TOP_CAP_CENTER_GEOMETRY_SOURCE,
    apply_tall_object_sdf_geometry_correction,
    get_known_tall_object_sdf_spec,
)


def test_mustard_sdf_spec_mesh_measured_offsets() -> None:
    spec = get_known_tall_object_sdf_spec("mustard_bottle")
    assert spec is not None
    assert spec.model_origin_to_geometry_center_offset_xyz == (
        0.0236,
        -0.0047,
        0.0818,
    )
    assert spec.model_origin_to_top_cap_center_offset_xyz == (
        0.0217,
        0.0311,
        0.0616,
    )
    assert spec.geometry_center_to_cap_center_offset_local_xyz == (
        -0.0019,
        0.0358,
        -0.0202,
    )


def test_sdf_correction_cap_xy_differs_from_geometry_center_at_yaw0() -> None:
    entry = {
        "label": "mustard_bottle",
        "source_pose_semantics": "model_link_origin",
        "pose_world": {"x": 0.6492, "y": -0.1765, "z": 0.2601, "yaw": 0.0},
        "yaw_rad": 0.0,
        "grasp_policy": "tall_object_topdown",
        "height_m": 0.1909,
        "mustard_cap_center_mode": MUSTARD_CAP_CENTER_MODE_SDF_OFFSET,
    }
    out = apply_tall_object_sdf_geometry_correction(entry)
    geom = out["semantic_box_center_world"]
    cap = out["gt_top_face_center_world"]
    assert out.get("grasp_center_source") == MUSTARD_TOP_CAP_CENTER_GEOMETRY_SOURCE
    assert abs(geom[0] - (0.6492 + 0.0236)) < 1e-4
    assert abs(geom[1] - (-0.1765 - 0.0047)) < 1e-4
    assert abs(cap[0] - (0.6492 + 0.0217)) < 1e-4
    assert abs(cap[1] - (-0.1765 + 0.0311)) < 1e-4
    assert math.hypot(cap[0] - geom[0], cap[1] - geom[1]) > 0.03


def test_sdf_correction_preserves_top_z_when_geometry_center_input() -> None:
    entry = {
        "label": "mustard_bottle",
        "source_pose_semantics": "geometry_center",
        "pose_world": {"x": 0.6492, "y": -0.1765, "z": 0.3560, "yaw": 0.0},
        "yaw_rad": 0.0,
        "grasp_policy": "tall_object_topdown",
        "height_m": 0.1909,
        "top_z_m": 0.45145,
        "mustard_cap_center_mode": MUSTARD_CAP_CENTER_MODE_SDF_OFFSET,
    }
    out = apply_tall_object_sdf_geometry_correction(entry)
    assert abs(float(out["top_z_m"]) - 0.45145) < 1e-4
    cap = out["gt_top_face_center_world"]
    assert abs(cap[2] - 0.45145) < 1e-4
    assert abs(cap[0] - (0.6492 - 0.0019)) < 1e-4
    assert abs(cap[1] - (-0.1765 + 0.0358)) < 1e-4


def test_sdf_correction_xy_only_when_top_z_present() -> None:
    """Z≈mesa+H/2 → pose tratada como centro geométrico; tapón = centro + offset cap−geom."""
    entry = {
        "label": "mustard_bottle",
        "source_pose_semantics": "model_link_origin",
        "pose_world": {"x": 0.6492, "y": -0.1765, "z": 0.3560, "yaw": 0.0},
        "yaw_rad": 0.0,
        "grasp_policy": "tall_object_topdown",
        "height_m": 0.1909,
        "top_z_m": 0.45145,
        "mustard_cap_center_mode": MUSTARD_CAP_CENTER_MODE_SDF_OFFSET,
    }
    out = apply_tall_object_sdf_geometry_correction(entry)
    assert abs(float(out["top_z_m"]) - 0.45145) < 1e-4
    cap = out["gt_top_face_center_world"]
    d = get_known_tall_object_sdf_spec("mustard_bottle")
    assert d is not None
    dg = d.geometry_center_to_cap_center_offset_local_xyz
    assert abs(cap[0] - (0.6492 + dg[0])) < 1e-4
    assert abs(cap[1] - (-0.1765 + dg[1])) < 1e-4


def test_sdf_correction_idempotent() -> None:
    entry = {
        "label": "mustard_bottle",
        "pose_world": {"x": 0.6492, "y": -0.1765, "z": 0.2601, "yaw": 0.0},
        "yaw_rad": 0.0,
        "grasp_policy": "tall_object_topdown",
        "height_m": 0.1909,
        "mustard_cap_center_mode": MUSTARD_CAP_CENTER_MODE_SDF_OFFSET,
    }
    once = apply_tall_object_sdf_geometry_correction(dict(entry))
    twice = apply_tall_object_sdf_geometry_correction(dict(once))
    assert twice.get("top_z_m") == once.get("top_z_m")
    assert twice.get("grasp_center_base") == once.get("grasp_center_base")


def test_sdf_correction_cap_xy_shift_with_yaw() -> None:
    yaw = math.pi / 4.0
    entry = {
        "label": "mustard_bottle",
        "source_pose_semantics": "model_link_origin",
        "pose_world": {"x": 1.0, "y": 2.0, "z": 0.26, "yaw": yaw},
        "yaw_rad": yaw,
        "grasp_policy": "tall_object_topdown",
        "height_m": 0.1909,
        "mustard_cap_center_mode": MUSTARD_CAP_CENTER_MODE_SDF_OFFSET,
    }
    out = apply_tall_object_sdf_geometry_correction(entry)
    cap = out["gt_top_face_center_world"]
    c, s = math.cos(yaw), math.sin(yaw)
    cx, cy, cz = 0.0217, 0.0311, 0.0616
    exp_x = 1.0 + c * cx - s * cy
    exp_y = 2.0 + s * cx + c * cy
    assert abs(cap[0] - exp_x) < 1e-5
    assert abs(cap[1] - exp_y) < 1e-5
    top_z, _ = resolve_tall_object_top_z_m(
        "mustard_bottle", float(out["semantic_box_center_world"][2]), height_m=0.1909
    )
    assert abs(float(out["top_z_m"]) - top_z) < 1e-4
