"""Layout aleatorio reproducible: cracker_box + sugar_box (escena two_boxes_03)."""

from __future__ import annotations

from typing import Tuple

from panda_vision.spawn.demo_scene_presets import DemoScenePreset
from panda_vision.spawn.pair_scene_random_layout import (
    default_pair_table_region,
    is_random_spawn_scene,
    pair_reference_preset,
    sample_pair_random_layout,
    sample_pair_random_spawn_entries,
    scene_random_seed_from_yaml,
)

TWO_BOXES_RANDOM_LABELS: Tuple[str, ...] = ("cracker_box", "sugar_box")
TWO_BOXES_03_REFERENCE_SEED = 303

default_two_boxes_table_region = default_pair_table_region


def is_two_boxes_random_scene(scene_id: str) -> bool:
    sid = str(scene_id or "").strip().lower()
    return sid == "two_boxes_03" and is_random_spawn_scene(sid)


def sample_two_boxes_random_layout(*args, **kwargs):
    return sample_pair_random_layout(
        *args,
        labels=TWO_BOXES_RANDOM_LABELS,
        log_tag="TWO_BOXES_RANDOM_LAYOUT",
        **kwargs,
    )


def sample_two_boxes_random_spawn_entries(*args, **kwargs):
    return sample_pair_random_spawn_entries(
        *args,
        labels=TWO_BOXES_RANDOM_LABELS,
        log_tag="TWO_BOXES_RANDOM_LAYOUT",
        **kwargs,
    )


def two_boxes_03_reference_preset() -> DemoScenePreset:
    return pair_reference_preset(
        "two_boxes_03",
        TWO_BOXES_RANDOM_LABELS,
        TWO_BOXES_03_REFERENCE_SEED,
    )
