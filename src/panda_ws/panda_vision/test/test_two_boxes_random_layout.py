"""Layout aleatorio two_boxes_03."""

from __future__ import annotations

import math
import random

import pytest

from panda_vision.spawn.demo_scene_presets import (
    DEMO_SCENE_LAYOUT_FOOTPRINT_SCALE,
    get_demo_scene_preset,
    validate_demo_scene_layout,
    validate_demo_scene_preset,
)
from panda_vision.spawn.two_boxes_random_layout import (
    TWO_BOXES_03_REFERENCE_SEED,
    default_two_boxes_table_region,
    is_two_boxes_random_scene,
    sample_two_boxes_random_layout,
    two_boxes_03_reference_preset,
)


def test_two_boxes_03_is_random_spawn_mode() -> None:
    assert is_two_boxes_random_scene("two_boxes_03") is True
    assert is_two_boxes_random_scene("two_boxes_02") is False


def test_reference_preset_validates() -> None:
    preset = two_boxes_03_reference_preset()
    ok, reason = validate_demo_scene_preset(
        preset, footprint_safety_scale=DEMO_SCENE_LAYOUT_FOOTPRINT_SCALE
    )
    assert ok, reason


def test_builtin_two_boxes_03_preset_registered() -> None:
    preset = get_demo_scene_preset("two_boxes_03")
    assert len(preset.objects) == 2
    labels = {o.label for o in preset.objects}
    assert labels == {"cracker_box", "sugar_box"}


def test_random_layout_reproducible_with_seed() -> None:
    region = default_two_boxes_table_region()
    a = sample_two_boxes_random_layout(
        random.Random(4242), region=region, max_attempts=3000
    )
    b = sample_two_boxes_random_layout(
        random.Random(4242), region=region, max_attempts=3000
    )
    assert len(a) == 2
    for left, right in zip(a, b):
        assert math.isclose(left.x, right.x, abs_tol=1e-6)
        assert math.isclose(left.y, right.y, abs_tol=1e-6)
        assert math.isclose(left.yaw, right.yaw, abs_tol=1e-6)


def test_random_layout_varies_yaw_across_seeds() -> None:
    region = default_two_boxes_table_region()
    yaws = []
    for seed in (11, 22, 33, 44, 55):
        poses = sample_two_boxes_random_layout(
            random.Random(seed), region=region, max_attempts=3000
        )
        yaws.extend([p.yaw for p in poses])
    spread = max(yaws) - min(yaws)
    assert spread > 1.0


def test_random_layout_pairwise_clearance() -> None:
    poses = sample_two_boxes_random_layout(
        random.Random(TWO_BOXES_03_REFERENCE_SEED), max_attempts=5000
    )
    ok, pairs = validate_demo_scene_layout(
        poses, footprint_safety_scale=DEMO_SCENE_LAYOUT_FOOTPRINT_SCALE
    )
    assert ok, pairs


def test_random_layout_sampler_eventually_succeeds() -> None:
    sample_two_boxes_random_layout(random.Random(99), max_attempts=5000)
