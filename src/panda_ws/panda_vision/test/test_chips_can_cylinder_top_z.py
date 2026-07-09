"""Regresión: chips_can runtime GT usa tapa = centro + H/2."""

from panda_vision.grasp.object_grasp_policy import resolve_tall_object_top_z_m
from panda_vision.spawn.runtime_scene_gt_geometry import (
    get_known_box_gt_spec,
    semantic_center_from_gazebo_model_origin,
    top_face_center_from_semantic_center,
)


def test_resolve_top_z_from_geometry_center() -> None:
    top_z, dbg = resolve_tall_object_top_z_m(
        "chips_can",
        0.385,
        height_m=0.250,
        payload_top_z_before=0.385,
    )
    assert abs(top_z - 0.510) < 1e-4
    assert dbg["source"] == "known_geometry_height"


def test_gazebo_origin_to_semantic_center_and_top() -> None:
    spec = get_known_box_gt_spec("chips_can")
    assert spec is not None
    assert spec.model_origin_to_geometry_center_offset_xyz[2] == 0.125
    origin = (0.580, -0.032, 0.265)
    quat = (0.0, 0.0, 0.0, 1.0)
    sem = semantic_center_from_gazebo_model_origin(origin, quat, "chips_can")
    assert abs(sem[2] - 0.390) < 1e-3
    top = top_face_center_from_semantic_center(sem, quat, "chips_can")
    assert abs(top[2] - 0.515) < 1e-3
