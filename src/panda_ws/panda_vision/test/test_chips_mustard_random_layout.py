"""Layout aleatorio chips_mustard_01."""

from __future__ import annotations

import math
import random

from panda_vision.spawn.chips_mustard_random_layout import (
    CHIPS_MUSTARD_01_REFERENCE_SEED,
    CHIPS_MUSTARD_LABELS,
    CHIPS_MUSTARD_MUSTARD_YAW_MAX_RAD,
    CHIPS_MUSTARD_MUSTARD_YAW_MIN_RAD,
    chips_mustard_01_reference_preset,
    is_chips_mustard_random_scene,
    mustard_spawn_yaw_in_operational_band,
    sample_chips_mustard_random_spawn_entries,
)
from panda_vision.spawn.demo_scene_presets import (
    DEMO_SCENE_LAYOUT_FOOTPRINT_SCALE,
    DemoSceneObjectPose,
    DemoScenePreset,
    get_demo_scene_preset,
    validate_demo_scene_layout,
    validate_demo_scene_preset,
)
from panda_vision.spawn.pair_scene_random_layout import (
    default_pair_table_region,
    is_random_spawn_scene,
    sample_pair_random_layout,
    sample_pair_random_spawn_entries,
)


def test_chips_mustard_is_random_spawn_mode() -> None:
    assert is_chips_mustard_random_scene("chips_mustard_01") is True
    assert is_random_spawn_scene("chips_mustard_01") is True
    assert is_chips_mustard_random_scene("two_boxes_03") is False


def test_reference_preset_validates() -> None:
    preset = chips_mustard_01_reference_preset()
    ok, reason = validate_demo_scene_preset(
        preset, footprint_safety_scale=DEMO_SCENE_LAYOUT_FOOTPRINT_SCALE
    )
    assert ok, reason


def test_builtin_preset_registered() -> None:
    preset = get_demo_scene_preset("chips_mustard_01")
    labels = {o.label for o in preset.objects}
    assert labels == set(CHIPS_MUSTARD_LABELS)


def test_random_layout_reproducible_with_seed() -> None:
    region = default_pair_table_region()
    a = sample_chips_mustard_random_spawn_entries(
        random.Random(5151), region=region, max_attempts=5000
    )
    b = sample_chips_mustard_random_spawn_entries(
        random.Random(5151), region=region, max_attempts=5000
    )
    for left, right in zip(a, b):
        assert math.isclose(left["x"], right["x"], abs_tol=1e-6)
        assert math.isclose(left["y"], right["y"], abs_tol=1e-6)
        assert math.isclose(left["yaw"], right["yaw"], abs_tol=1e-6)


def test_mustard_yaw_restricted_in_chips_mustard_layout() -> None:
    entries = sample_chips_mustard_random_spawn_entries(
        random.Random(5151), max_attempts=5000
    )
    mustard = next(e for e in entries if e["label"] == "mustard_bottle")
    assert mustard_spawn_yaw_in_operational_band(float(mustard["yaw"]))
    assert float(mustard["yaw"]) >= float(CHIPS_MUSTARD_MUSTARD_YAW_MIN_RAD)
    assert float(mustard["yaw"]) <= float(CHIPS_MUSTARD_MUSTARD_YAW_MAX_RAD)


def test_random_layout_validates_with_scene_seed_401() -> None:
    """Regresión: scene_random_seed no debe copiarse al campo seed de chips_can."""
    entries = sample_chips_mustard_random_spawn_entries(
        random.Random(401),
        random_seed=401,
        max_attempts=5000,
    )
    assert "seed" not in entries[0]
    poses = tuple(
        DemoSceneObjectPose(
            e["label"], e["x"], e["y"], e["yaw"], order_index=e["order_index"]
        )
        for e in entries
    )
    ok, reason = validate_demo_scene_preset(
        DemoScenePreset("chips_mustard_01", objects=poses),
        footprint_safety_scale=DEMO_SCENE_LAYOUT_FOOTPRINT_SCALE,
    )
    assert ok, reason


def test_random_layout_pairwise_clearance() -> None:
    poses = sample_pair_random_layout(
        random.Random(CHIPS_MUSTARD_01_REFERENCE_SEED),
        CHIPS_MUSTARD_LABELS,
        max_attempts=5000,
    )
    ok, pairs = validate_demo_scene_layout(
        poses, footprint_safety_scale=DEMO_SCENE_LAYOUT_FOOTPRINT_SCALE
    )
    assert ok, pairs
