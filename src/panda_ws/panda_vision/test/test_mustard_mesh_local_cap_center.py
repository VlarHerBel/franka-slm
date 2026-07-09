"""Centro de tapón mustard: punto local del mesh transformado con la pose del modelo."""

import math
import random

from panda_vision.spawn.known_object_geometry import (
    MUSTARD_CAP_CENTER_MODE_MESH_LOCAL,
    MUSTARD_MESH_LOCAL_CAP_CENTER_SOURCE,
    RUNTIME_GT_TALL_CAP_CENTER_SOURCES,
    apply_tall_object_sdf_geometry_correction,
    compute_mustard_mesh_local_cap_center_world,
    get_known_tall_object_sdf_spec,
    is_runtime_gt_tall_cap_center_source,
)


def _mustard_entry(*, x: float, y: float, z: float, yaw: float) -> dict:
    return {
        "label": "mustard_bottle",
        "mustard_cap_center_mode": MUSTARD_CAP_CENTER_MODE_MESH_LOCAL,
        "source_pose_semantics": "model_link_origin",
        "pose_world": {"x": x, "y": y, "z": z, "yaw": yaw},
        "yaw_rad": yaw,
        "grasp_policy": "tall_object_topdown",
        "height_m": 0.1909,
    }


def test_mesh_local_cap_center_yaw0_from_model_origin() -> None:
    x, y, z = 0.6492, -0.1765, 0.2601
    out = apply_tall_object_sdf_geometry_correction(_mustard_entry(x=x, y=y, z=z, yaw=0.0))
    assert out.get("grasp_center_source") == MUSTARD_MESH_LOCAL_CAP_CENTER_SOURCE
    cap = out["grasp_center_base"]
    assert abs(cap[0] - (x + 0.0240)) < 1e-4
    assert abs(cap[1] - (y - 0.0049)) < 1e-4
    assert abs(cap[2] - (z + 0.1914)) < 1e-4


def test_mesh_local_cap_center_yaw90_rotates_xy() -> None:
    x, y, z = 1.0, 2.0, 0.26
    yaw = math.pi / 2.0
    out = apply_tall_object_sdf_geometry_correction(
        _mustard_entry(x=x, y=y, z=z, yaw=yaw)
    )
    cap = out["grasp_center_base"]
    x_local, y_local = 0.0240, -0.0049
    assert abs(cap[0] - (x - y_local)) < 1e-4
    assert abs(cap[1] - (y + x_local)) < 1e-4
    assert abs(cap[2] - (z + 0.1914)) < 1e-4


def test_mesh_local_cap_distance_invariant_under_yaw() -> None:
    spec = get_known_tall_object_sdf_spec("mustard_bottle")
    assert spec is not None
    cap_local = spec.cap_center_local_m
    local_r = math.sqrt(sum(v * v for v in cap_local))
    rng = random.Random(42)
    for _ in range(12):
        ox, oy = rng.uniform(0.4, 0.8), rng.uniform(-0.3, 0.3)
        oz = 0.26
        yaw = rng.uniform(-math.pi, math.pi)
        mesh = compute_mustard_mesh_local_cap_center_world(
            (ox, oy, oz), cap_local, yaw_rad=yaw
        )
        world_r = math.hypot(mesh[0] - ox, mesh[1] - oy, mesh[2] - oz)
        assert abs(world_r - local_r) < 1e-4


def test_mesh_local_cap_center_source_in_perception_runtime_gt_set() -> None:
    src = MUSTARD_MESH_LOCAL_CAP_CENTER_SOURCE
    assert src in RUNTIME_GT_TALL_CAP_CENTER_SOURCES
    assert is_runtime_gt_tall_cap_center_source(src)
