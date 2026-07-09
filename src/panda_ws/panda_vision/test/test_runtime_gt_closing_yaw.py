"""Yaw de cierre runtime GT: gap en eje corto (width), no en yaw de spawn (long)."""
import math

from panda_vision.spawn.runtime_scene_gt_geometry import resolve_runtime_gt_spawn_axes


def test_closing_yaw_perpendicular_to_spawn_yaw() -> None:
    gt_yaw = math.radians(-100.5)
    axes = resolve_runtime_gt_spawn_axes(gt_yaw)
    closing = float(axes["closing_yaw_rad"])
    assert abs(math.cos(closing) - math.cos(gt_yaw + math.pi / 2.0)) < 1e-6
    assert abs(math.sin(closing) - math.sin(gt_yaw + math.pi / 2.0)) < 1e-6


def test_short_axis_matches_closing_direction() -> None:
    gt_yaw = 0.3
    axes = resolve_runtime_gt_spawn_axes(gt_yaw)
    sx, sy = axes["short_axis_xy"]
    closing = float(axes["closing_yaw_rad"])
    assert abs(math.atan2(sy, sx) - closing) < 1e-6


def test_long_and_short_axes_orthogonal() -> None:
    axes = resolve_runtime_gt_spawn_axes(1.1)
    lx, ly = axes["long_axis_xy"]
    sx, sy = axes["short_axis_xy"]
    dot = abs(lx * sx + ly * sy)
    assert dot < 1e-6
