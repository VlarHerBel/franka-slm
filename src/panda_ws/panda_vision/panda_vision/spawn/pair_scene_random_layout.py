"""Layout aleatorio reproducible para escenas de 2 objetos (YAML spawn_mode: random)."""

from __future__ import annotations

import math
import random
from typing import Any, Dict, List, Mapping, Optional, Tuple

import yaml

from panda_vision.spawn.demo_scene_presets import (
    DEFAULT_DEMO_SCENE_MIN_CLEARANCE_M,
    DEMO_SCENE_LAYOUT_FOOTPRINT_SCALE,
    TABLE_CENTER_XY,
    TABLE_EDGE_MARGIN_M,
    TABLE_HALF_EXTENT_XY,
    DemoSceneObjectPose,
    DemoScenePreset,
    required_pairwise_min_distance_m,
    validate_demo_scene_layout,
    validate_demo_scene_object,
)
from panda_vision.spawn.demo_scene_yaml_spawn import demo_scene_yaml_path
from panda_vision.spawn.semantic_spawn_sampling import (
    TableSpawnRegion,
    sample_semantic_box_pose_xyyaw,
)


def default_pair_table_region() -> TableSpawnRegion:
    """Región de muestreo alineada con vision_test_table y alcance Panda."""
    cx, cy = TABLE_CENTER_XY
    hx, hy = TABLE_HALF_EXTENT_XY
    margin = float(TABLE_EDGE_MARGIN_M)
    return TableSpawnRegion(
        x_min=float(cx) - float(hx) + margin,
        x_max=float(cx) + float(hx) - margin,
        y_min=float(cy) - float(hy) + margin,
        y_max=float(cy) + float(hy) - margin,
        margin_m=margin,
    )


def _load_scene_yaml_raw(scene_id: str) -> Dict[str, Any]:
    path = demo_scene_yaml_path(scene_id)
    if not path:
        return {}
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def is_random_spawn_scene(scene_id: str) -> bool:
    raw = _load_scene_yaml_raw(str(scene_id or "").strip().lower())
    return str(raw.get("spawn_mode", "")).strip().lower() == "random"


def pair_labels_from_scene_yaml(scene_id: str) -> Tuple[str, ...]:
    raw = _load_scene_yaml_raw(str(scene_id or "").strip().lower())
    labels = tuple(
        str(x).strip().lower()
        for x in (raw.get("pick_order") or [])
        if str(x).strip()
    )
    return labels


def scene_random_seed_from_yaml(scene_id: str) -> int:
    raw = _load_scene_yaml_raw(str(scene_id or "").strip().lower())
    try:
        return int(raw.get("random_seed", 0))
    except (TypeError, ValueError):
        return 0


def _yaw_limits_for_label(
    label: str,
    label_yaw_limits: Optional[Mapping[str, Tuple[float, float]]],
) -> Tuple[float, float]:
    if label_yaw_limits is not None:
        band = label_yaw_limits.get(str(label).strip().lower())
        if band is not None and len(band) >= 2:
            return float(band[0]), float(band[1])
    return -math.pi, math.pi


def sample_pair_random_layout(
    rng: random.Random,
    labels: Tuple[str, ...],
    *,
    region: Optional[TableSpawnRegion] = None,
    min_clearance_m: float = DEFAULT_DEMO_SCENE_MIN_CLEARANCE_M,
    max_attempts: int = 2000,
    logger: Any = None,
    log_tag: str = "PAIR_RANDOM_LAYOUT",
    label_yaw_limits: Optional[Mapping[str, Tuple[float, float]]] = None,
) -> Tuple[DemoSceneObjectPose, ...]:
    """Dos objetos con yaw y centro semántico aleatorios, sin solapamiento."""
    if len(labels) != 2:
        raise ValueError("sample_pair_random_layout requiere exactamente 2 labels")
    table = region or default_pair_table_region()
    last_reason = "no_attempts"
    for attempt in range(1, max(1, int(max_attempts)) + 1):
        poses: List[DemoSceneObjectPose] = []
        failed = False
        for idx, label in enumerate(labels):
            yaw_min, yaw_max = _yaw_limits_for_label(label, label_yaw_limits)
            try:
                x, y, yaw = sample_semantic_box_pose_xyyaw(
                    rng,
                    label,
                    table,
                    logger=logger,
                    yaw_min_rad=yaw_min,
                    yaw_max_rad=yaw_max,
                )
            except RuntimeError as exc:
                failed = True
                last_reason = str(exc)
                break
            for placed in poses:
                req = required_pairwise_min_distance_m(
                    label,
                    placed.label,
                    min_clearance_m=min_clearance_m,
                    footprint_safety_scale=DEMO_SCENE_LAYOUT_FOOTPRINT_SCALE,
                )
                if math.hypot(x - placed.x, y - placed.y) < req - 1e-3:
                    failed = True
                    last_reason = (
                        "pairwise_too_close %s vs %s dist=%.4f req=%.4f"
                        % (label, placed.label, math.hypot(x - placed.x, y - placed.y), req)
                    )
                    break
            if failed:
                break
            poses.append(
                DemoSceneObjectPose(
                    label,
                    float(x),
                    float(y),
                    float(yaw),
                    order_index=idx,
                )
            )
        if failed or len(poses) != 2:
            continue
        layout_ok, _pairs = validate_demo_scene_layout(
            poses,
            min_clearance_m=min_clearance_m,
            footprint_safety_scale=DEMO_SCENE_LAYOUT_FOOTPRINT_SCALE,
        )
        if not layout_ok:
            last_reason = "layout_collision"
            continue
        for obj in poses:
            ok_obj, reason = validate_demo_scene_object(obj)
            if not ok_obj:
                failed = True
                last_reason = "%s: %s" % (obj.label, reason)
                break
        if failed:
            continue
        if logger is not None:
            try:
                logger.info(
                    "[%s]\n"
                    "attempt=%d\n"
                    "%s=(%.4f, %.4f, %.4f)\n"
                    "%s=(%.4f, %.4f, %.4f)\n"
                    "result=OK"
                    % (
                        log_tag,
                        attempt,
                        poses[0].label,
                        poses[0].x,
                        poses[0].y,
                        poses[0].yaw,
                        poses[1].label,
                        poses[1].x,
                        poses[1].y,
                        poses[1].yaw,
                    )
                )
            except Exception:
                pass
        return tuple(poses)
    raise RuntimeError(
        "No se encontró layout aleatorio para %s tras %d intentos (%s)"
        % (labels, max_attempts, last_reason)
    )


def sample_pair_random_spawn_entries(
    rng: random.Random,
    labels: Tuple[str, ...],
    *,
    region: Optional[TableSpawnRegion] = None,
    min_clearance_m: float = DEFAULT_DEMO_SCENE_MIN_CLEARANCE_M,
    random_seed: int = 0,
    max_attempts: int = 2000,
    logger: Any = None,
    log_tag: str = "PAIR_RANDOM_LAYOUT",
    label_yaw_limits: Optional[Mapping[str, Tuple[float, float]]] = None,
) -> List[Dict[str, Any]]:
    poses = sample_pair_random_layout(
        rng,
        labels,
        region=region,
        min_clearance_m=min_clearance_m,
        max_attempts=max_attempts,
        logger=logger,
        log_tag=log_tag,
        label_yaw_limits=label_yaw_limits,
    )
    entries: List[Dict[str, Any]] = []
    for obj in poses:
        entries.append(
            {
                "label": obj.label,
                "x": float(obj.x),
                "y": float(obj.y),
                "yaw": float(obj.yaw),
                "order_index": int(obj.order_index or 0),
            }
        )
    return entries


def pair_reference_preset(
    scene_id: str,
    labels: Tuple[str, ...],
    reference_seed: int,
    *,
    label_yaw_limits: Optional[Mapping[str, Tuple[float, float]]] = None,
) -> DemoScenePreset:
    rng = random.Random(int(reference_seed))
    poses = sample_pair_random_layout(
        rng,
        labels,
        max_attempts=5000,
        label_yaw_limits=label_yaw_limits,
    )
    return DemoScenePreset(scene_id=str(scene_id), objects=poses)
