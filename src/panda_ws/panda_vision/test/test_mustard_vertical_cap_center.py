"""Centro de tapón mustard: eje vertical desde footprint (sin offset lateral SDF)."""

import math

from panda_vision.spawn.known_object_geometry import (
    MUSTARD_CAP_CENTER_MODE_VERTICAL_AXIS,
    MUSTARD_VERTICAL_AXIS_CAP_CENTER_SOURCE,
    apply_tall_object_sdf_geometry_correction,
)


def test_vertical_axis_cap_center_uses_footprint_xy_not_lateral_offset() -> None:
    entry = {
        "label": "mustard_bottle",
        "mustard_cap_center_mode": MUSTARD_CAP_CENTER_MODE_VERTICAL_AXIS,
        "source_pose_semantics": "model_link_origin",
        "pose_world": {"x": 0.6492, "y": -0.1765, "z": 0.2601, "yaw": 0.0},
        "yaw_rad": 0.0,
        "grasp_policy": "tall_object_topdown",
        "height_m": 0.1909,
    }
    out = apply_tall_object_sdf_geometry_correction(entry)
    assert (
        out.get("grasp_center_source")
        == MUSTARD_VERTICAL_AXIS_CAP_CENTER_SOURCE
    )
    cap = out["grasp_center_base"]
    geom = out["semantic_box_center_world"]
    old = out["mustard_old_offset_cap_center_world"]
    assert abs(cap[0] - geom[0]) < 1e-4
    assert abs(cap[1] - geom[1]) < 1e-4
    assert abs(cap[2] - out["top_z_m"]) < 1e-4
    assert math.hypot(cap[0] - old[0], cap[1] - old[1]) > 0.01
    assert out.get("mustard_footprint_center_source") == "collision_box_pose"
