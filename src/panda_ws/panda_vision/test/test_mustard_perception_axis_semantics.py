"""Ejes operativos mustard_bottle: gap en ancho corto (link X), pads en largo (link Y)."""

import math

from panda_vision.grasp.mustard_bottle_axis_semantics import (
    MUSTARD_AXIS_DOT_LONG_MAX,
    MUSTARD_AXIS_DOT_SHORT_MIN,
    apply_mustard_bottle_axis_semantics,
    log_mustard_overlay_axis_debug,
)
from panda_vision.spawn.known_object_geometry import (
    MUSTARD_TOP_CAP_CENTER_GEOMETRY_SOURCE,
)
from panda_vision.spawn.known_object_geometry import get_known_object_geometry
from panda_vision.spawn.runtime_scene_gt_geometry import resolve_runtime_gt_spawn_axes


def test_mustard_geometry_spec_uses_link_y_as_length() -> None:
    geo = get_known_object_geometry("mustard_bottle")
    assert geo is not None
    assert geo.local_length_axis == "y"
    assert geo.local_width_axis == "x"
    assert abs(geo.dims_lwh[0] - 0.0953) < 1e-3
    assert abs(geo.dims_lwh[1] - 0.0577) < 1e-3


def test_mustard_spawn_axes_closing_on_short_at_zero_yaw() -> None:
    axes = resolve_runtime_gt_spawn_axes(
        0.0,
        local_length_axis="y",
        local_width_axis="x",
    )
    closing = float(axes["closing_yaw_rad"])
    lx, ly = axes["long_axis_xy"]
    sx, sy = axes["short_axis_xy"]
    assert abs(math.atan2(sy, sx) - closing) < 1e-6
    assert abs(lx) < 1e-6 and abs(ly - 1.0) < 1e-6
    assert abs(sx - 1.0) < 1e-6 and abs(sy) < 1e-6


def test_mustard_perception_axis_semantics_ok_normal_mapping() -> None:
    axes = resolve_runtime_gt_spawn_axes(
        math.radians(30.0),
        local_length_axis="y",
        local_width_axis="x",
    )
    pose_meta = {
        "long_axis_xy": list(axes["long_axis_xy"]),
        "short_axis_xy": list(axes["short_axis_xy"]),
        "major_axis_xy": list(axes["long_axis_xy"]),
        "minor_axis_xy": list(axes["short_axis_xy"]),
        "object_yaw_rad": math.radians(30.0),
    }
    grasp_fields: dict = {}
    out = apply_mustard_bottle_axis_semantics(
        label="mustard_bottle",
        mapping="normal",
        pose_meta=pose_meta,
        grasp_fields=grasp_fields,
        logger=None,
    )
    assert out["axis_debug_result"] == "OK"
    assert out["width_sanity_result"] == "OK"
    assert out["publish_allowed"] is True
    assert out["closing_axis_dot_short"] >= MUSTARD_AXIS_DOT_SHORT_MIN
    assert out["closing_axis_dot_long"] <= MUSTARD_AXIS_DOT_LONG_MAX
    gap = grasp_fields["grasp_gap_axis_xy"]
    pad = grasp_fields["finger_pad_axis_xy"]
    assert gap is not None and pad is not None
    assert abs(gap[0] * pad[0] + gap[1] * pad[1]) < 0.05
    assert abs(float(grasp_fields["required_grasp_width_m"]) - 0.058) < 1e-3
    assert abs(float(grasp_fields["closing_yaw_rad"]) - float(out["grasp_gap_yaw_rad"])) < 1e-6
    assert abs(float(grasp_fields["object_yaw_rad"]) - float(out["finger_pad_yaw_rad"])) < 1e-6
    assert abs(float(grasp_fields["closing_yaw_rad"]) - float(grasp_fields["object_yaw_rad"])) > 0.5
    assert str(grasp_fields.get("closing_yaw_semantics")) == "gap_axis"


def test_overlay_axis_debug_ok_when_orange_pad_cyan_gap() -> None:
    axes = resolve_runtime_gt_spawn_axes(
        0.0,
        local_length_axis="y",
        local_width_axis="x",
    )
    pose_meta = {
        "finger_pad_axis_xy": list(axes["long_axis_xy"]),
        "grasp_gap_axis_xy": list(axes["short_axis_xy"]),
        "finger_pad_yaw_rad": math.atan2(
            float(axes["long_axis_xy"][1]), float(axes["long_axis_xy"][0])
        ),
        "grasp_gap_yaw_rad": math.atan2(
            float(axes["short_axis_xy"][1]), float(axes["short_axis_xy"][0])
        ),
    }
    out = log_mustard_overlay_axis_debug(
        pose_meta,
        orange_axis_xy=list(axes["long_axis_xy"]),
        cyan_axis_xy=list(axes["short_axis_xy"]),
        orange_source="finger_pad_axis_xy",
        cyan_source="grasp_gap_axis_xy",
        logger=None,
    )
    assert out["result"] == "OK"
    assert out["dot_orange_vs_finger_pad"] >= MUSTARD_AXIS_DOT_SHORT_MIN
    assert out["dot_cyan_vs_gap"] >= MUSTARD_AXIS_DOT_SHORT_MIN


def test_runtime_gt_cap_center_allows_publish_with_misaligned_pca_axes() -> None:
    """PCA de máscara puede no alinear con GT; no bloquear publicación operativa."""
    pose_meta = {
        "runtime_gt_geometry_applied": True,
        "top_face_source": "runtime_gt_tall_object",
        "grasp_center_source": MUSTARD_TOP_CAP_CENTER_GEOMETRY_SOURCE,
        "long_axis_xy": [0.0, 1.0],
        "short_axis_xy": [1.0, 0.0],
        "major_axis_xy": [1.0, 0.0],
        "minor_axis_xy": [0.0, 1.0],
    }
    out = apply_mustard_bottle_axis_semantics(
        label="mustard_bottle",
        mapping="normal",
        pose_meta=pose_meta,
        grasp_fields={},
        logger=None,
    )
    assert out["publish_allowed"] is True
    assert out["axis_debug_result"] == "OK"


def test_swap_major_minor_fails_axis_debug_with_correct_geometry() -> None:
    axes = resolve_runtime_gt_spawn_axes(
        0.0,
        local_length_axis="y",
        local_width_axis="x",
    )
    pose_meta = {
        "long_axis_xy": list(axes["long_axis_xy"]),
        "short_axis_xy": list(axes["short_axis_xy"]),
        "major_axis_xy": list(axes["long_axis_xy"]),
        "minor_axis_xy": list(axes["short_axis_xy"]),
    }
    out = apply_mustard_bottle_axis_semantics(
        label="mustard_bottle",
        mapping="swap_major_minor",
        pose_meta=pose_meta,
        grasp_fields={},
        logger=None,
    )
    assert out["axis_debug_result"] == "FAIL"
    assert out["publish_allowed"] is False
