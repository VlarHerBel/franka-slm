"""Layout aleatorio: chips_can + mustard_bottle (escena chips_mustard_01)."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

from panda_vision.spawn.demo_scene_presets import DemoScenePreset
from panda_vision.spawn.pair_scene_random_layout import (
    is_random_spawn_scene,
    pair_reference_preset,
    sample_pair_random_spawn_entries,
    scene_random_seed_from_yaml,
)

CHIPS_MUSTARD_SCENE_ID = "chips_mustard_01"
CHIPS_MUSTARD_LABELS: Tuple[str, ...] = ("chips_can", "mustard_bottle")
CHIPS_MUSTARD_01_REFERENCE_SEED = 401

# Banda yaw mostaza: upright top-down (excluye yaw casi ±π que tumba el eje corto en XY).
CHIPS_MUSTARD_MUSTARD_YAW_MIN_RAD = -1.5
CHIPS_MUSTARD_MUSTARD_YAW_MAX_RAD = 2.6


def chips_mustard_label_yaw_limits() -> Dict[str, Tuple[float, float]]:
    return {
        "mustard_bottle": (
            float(CHIPS_MUSTARD_MUSTARD_YAW_MIN_RAD),
            float(CHIPS_MUSTARD_MUSTARD_YAW_MAX_RAD),
        )
    }


def is_chips_mustard_random_scene(scene_id: str) -> bool:
    sid = str(scene_id or "").strip().lower()
    return sid == CHIPS_MUSTARD_SCENE_ID and is_random_spawn_scene(sid)


def chips_mustard_01_reference_preset() -> DemoScenePreset:
    return pair_reference_preset(
        CHIPS_MUSTARD_SCENE_ID,
        CHIPS_MUSTARD_LABELS,
        CHIPS_MUSTARD_01_REFERENCE_SEED,
        label_yaw_limits=chips_mustard_label_yaw_limits(),
    )


def sample_chips_mustard_random_spawn_entries(
    rng,
    labels: Tuple[str, ...] = CHIPS_MUSTARD_LABELS,
    *,
    region=None,
    min_clearance_m: float = 0.03,
    random_seed: int = 0,
    max_attempts: int = 2000,
    logger: Any = None,
    log_tag: str = "CHIPS_MUSTARD_RANDOM_LAYOUT",
) -> List[Dict[str, Any]]:
    return sample_pair_random_spawn_entries(
        rng,
        labels,
        region=region,
        min_clearance_m=min_clearance_m,
        random_seed=random_seed,
        max_attempts=max_attempts,
        logger=logger,
        log_tag=log_tag,
        label_yaw_limits=chips_mustard_label_yaw_limits(),
    )


def mustard_spawn_yaw_in_operational_band(yaw_rad: float) -> bool:
    y = float(yaw_rad)
    lo = float(CHIPS_MUSTARD_MUSTARD_YAW_MIN_RAD)
    hi = float(CHIPS_MUSTARD_MUSTARD_YAW_MAX_RAD)
    return lo - 1e-6 <= y <= hi + 1e-6
