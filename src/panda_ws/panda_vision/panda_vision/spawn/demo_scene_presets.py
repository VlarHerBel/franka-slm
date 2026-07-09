"""Presets de escena demo reproducibles (4 objetos consolidados en mesa)."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

# Posición semántica de referencia de chips_can seed1001 (IK/reachability no válida).
CHIPS_CAN_BANNED_SEED = 1001
CHIPS_CAN_ALLOWED_SEEDS: Tuple[int, ...] = (1002, 1003, 1004)
CHIPS_CAN_BANNED_REFERENCE_XY = (0.6492, -0.1765)
CHIPS_CAN_BANNED_XY_TOLERANCE_M = 0.04

DEMO_SCENE_OBJECT_LABELS: Tuple[str, ...] = (
    "cracker_box",
    "sugar_box",
    "mustard_bottle",
    "chips_can",
)

TWO_BOXES_OBJECT_LABELS: Tuple[str, ...] = ("cracker_box", "sugar_box")
TWO_BOXES_SCENE_IDS: Tuple[str, ...] = (
    "two_boxes_01",
    "two_boxes_02",
    "two_boxes_03",
)
CHIPS_MUSTARD_OBJECT_LABELS: Tuple[str, ...] = ("chips_can", "mustard_bottle")
CHIPS_MUSTARD_SCENE_IDS: Tuple[str, ...] = ("chips_mustard_01", "chips_mustard_02")

DEFAULT_DEMO_SCENE_MIN_CLEARANCE_M = 0.03
# Poses demo fijas: validadas con escala 1.0 (no usar footprint_safety_scale del spawner aleatorio).
DEMO_SCENE_LAYOUT_FOOTPRINT_SCALE = 1.0

# demo_scene_02 v3: más aire lateral para transport exit con cracker attached.
DEMO_SCENE_02_LAYOUT_VERSION = "v3_clear_table_transport"
DEMO_SCENE_02_PICK_ORDER: Tuple[str, ...] = (
    "cracker_box",
    "chips_can",
    "sugar_box",
    "mustard_bottle",
)
DEMO_SCENE_02_REMAINING_SUGAR_MUSTARD_SCENE_ID = (
    "demo_scene_02_remaining_sugar_mustard"
)
# Escenas deposit_* con poses demo_scene_02 (cracker/chips precargados en cajón).
DEMO_SCENE_02_DEPOSIT_POLICY_SCENE_IDS: frozenset = frozenset(
    {"deposit_02_cracker_chips", "deposit_03_mustard_only"}
)
# Validación temporal: sugar/mustard con poses de demo_scene_02 (cracker/chips ya depositados).
DEMO_SCENE_02_REMAINING_SUGAR_MUSTARD_PICK_ORDER: Tuple[str, ...] = (
    "sugar_box",
    "mustard_bottle",
)
# Zona cómoda Panda (centro XY); cracker_box puede usar x ligeramente menor.
DEMO_SCENE_02_REACH_X_RANGE = (0.48, 0.64)
DEMO_SCENE_02_REACH_Y_RANGE = (-0.18, 0.12)
DEMO_SCENE_02_CRACKER_REACH_X_RANGE = (0.45, 0.64)
# mustard_bottle: x hasta 0.66 (mínimo sin colisión con cracker_box en layout 4 objetos).
DEMO_SCENE_02_REACH_X_OVERRIDES: Dict[str, Tuple[float, float]] = {
    "mustard_bottle": (0.48, 0.66),
}

# Mesa vision_test_table (centro 0.60,0.0; tamaño 0.72×0.48 m en Gazebo).
TABLE_CENTER_XY = (0.60, 0.0)
TABLE_HALF_EXTENT_XY = (0.36, 0.24)
TABLE_EDGE_MARGIN_M = 0.02


@dataclass(frozen=True)
class DemoSceneObjectPose:
    label: str
    x: float
    y: float
    yaw: float
    seed: Optional[int] = None
    order_index: Optional[int] = None


@dataclass(frozen=True)
class DemoScenePreset:
    scene_id: str
    objects: Tuple[DemoSceneObjectPose, ...]


# Poses explícitas: huella completa dentro de mesa, yaw variado, chips_can seeds 1002–1004.
DEMO_SCENE_PRESETS: Dict[str, DemoScenePreset] = {
    "demo_scene_01": DemoScenePreset(
        scene_id="demo_scene_01",
        objects=(
            DemoSceneObjectPose("cracker_box", 0.4700, 0.1317, -0.6275),
            DemoSceneObjectPose("sugar_box", 0.4300, -0.1450, 0.8500),
            DemoSceneObjectPose("mustard_bottle", 0.7148, -0.1066, -2.9700),
            DemoSceneObjectPose(
                "chips_can",
                0.5757,
                -0.0140,
                -1.7533,
                seed=1002,
            ),
        ),
    ),
    "demo_scene_02": DemoScenePreset(
        scene_id="demo_scene_02",
        objects=(
            DemoSceneObjectPose(
                "cracker_box", 0.4550, 0.1150, 2.9155, order_index=0
            ),
            DemoSceneObjectPose(
                "chips_can",
                0.5200,
                -0.0950,
                1.3952551671329498,
                seed=1003,
                order_index=1,
            ),
            DemoSceneObjectPose(
                "sugar_box", 0.6300, -0.1750, -3.0159, order_index=2
            ),
            DemoSceneObjectPose(
                "mustard_bottle", 0.6600, 0.0600, 1.6392, order_index=3
            ),
        ),
    ),
    DEMO_SCENE_02_REMAINING_SUGAR_MUSTARD_SCENE_ID: DemoScenePreset(
        scene_id=DEMO_SCENE_02_REMAINING_SUGAR_MUSTARD_SCENE_ID,
        objects=(
            DemoSceneObjectPose(
                "sugar_box", 0.6300, -0.1750, -3.0159, order_index=0
            ),
            # Spawn compensa offset SDF interno (cap center ≈ spawn + (0.0033, 0.0238) @ yaw=1.6392).
            DemoSceneObjectPose(
                "mustard_bottle", 0.6570, 0.0360, 1.6392, order_index=1
            ),
        ),
    ),
    "demo_scene_03": DemoScenePreset(
        scene_id="demo_scene_03",
        objects=(
            DemoSceneObjectPose("cracker_box", 0.5181, -0.1773, -2.9986),
            DemoSceneObjectPose("sugar_box", 0.6554, 0.0544, -1.9860),
            DemoSceneObjectPose("mustard_bottle", 0.7188, -0.1252, -0.4088),
            DemoSceneObjectPose(
                "chips_can",
                0.5714,
                0.1667,
                0.7116,
                seed=1004,
            ),
        ),
    ),
    # Escenas mínimas SLM: solo cracker_box + sugar_box (validación pick_place individual).
    "two_boxes_01": DemoScenePreset(
        scene_id="two_boxes_01",
        objects=(
            DemoSceneObjectPose("cracker_box", 0.5000, -0.0800, 0.0000, order_index=0),
            DemoSceneObjectPose("sugar_box", 0.6000, 0.1000, 0.0000, order_index=1),
        ),
    ),
    "two_boxes_02": DemoScenePreset(
        scene_id="two_boxes_02",
        objects=(
            DemoSceneObjectPose("cracker_box", 0.4800, 0.0600, 1.5708, order_index=0),
            DemoSceneObjectPose("sugar_box", 0.6200, -0.1000, -1.2000, order_index=1),
        ),
    ),
    "chips_mustard_02": DemoScenePreset(
        scene_id="chips_mustard_02",
        objects=(
            DemoSceneObjectPose(
                "chips_can",
                0.5000,
                -0.0800,
                0.0000,
                order_index=0,
            ),
            DemoSceneObjectPose(
                "mustard_bottle", 0.6000, 0.1000, 0.0000, order_index=1
            ),
        ),
    ),
}

# Variantes 3 objetos: mismas poses asentadas que 01/02/03, sin mustard_bottle.
DEMO_SCENE_3OBJ_OMITTED_LABEL = "mustard_bottle"
DEMO_SCENE_3OBJ_PICK_ORDER: Tuple[str, ...] = (
    "cracker_box",
    "chips_can",
    "sugar_box",
)
DEMO_SCENE_3OBJ_PARENT_SCENE_IDS: Tuple[str, ...] = (
    "demo_scene_01",
    "demo_scene_02",
    "demo_scene_03",
)


def three_object_demo_scene_id(parent_scene_id: str) -> str:
    return "%s_3obj" % str(parent_scene_id).strip()


def normalize_demo_scene_policy_key(preset_name: str) -> str:
    """Quita sufijos de variante (_nogolden) antes de resolver políticas."""
    key = str(preset_name or "").strip()
    if key.endswith("_nogolden"):
        key = key.removesuffix("_nogolden")
    return key


def is_demo_scene_3obj_scene_id(scene_id: str) -> bool:
    sid = normalize_demo_scene_policy_key(str(scene_id or "").strip().lower())
    if not sid.endswith("_3obj"):
        return False
    return sid.removesuffix("_3obj") in DEMO_SCENE_3OBJ_PARENT_SCENE_IDS


def build_three_object_demo_preset(parent_scene_id: str) -> DemoScenePreset:
    parent_id = str(parent_scene_id).strip()
    parent = DEMO_SCENE_PRESETS.get(parent_id)
    if parent is None:
        raise ValueError("parent scene desconocida: %r" % parent_id)
    by_label = {o.label: o for o in parent.objects}
    missing = [lb for lb in DEMO_SCENE_3OBJ_PICK_ORDER if lb not in by_label]
    if missing:
        raise ValueError("%s: faltan labels para 3obj: %s" % (parent_id, missing))
    objects: List[DemoSceneObjectPose] = []
    for idx, label in enumerate(DEMO_SCENE_3OBJ_PICK_ORDER):
        src = by_label[label]
        objects.append(
            DemoSceneObjectPose(
                src.label,
                src.x,
                src.y,
                src.yaw,
                seed=src.seed,
                order_index=idx,
            )
        )
    return DemoScenePreset(
        scene_id=three_object_demo_scene_id(parent_id),
        objects=tuple(objects),
    )


for _parent_scene_id in DEMO_SCENE_3OBJ_PARENT_SCENE_IDS:
    _three_obj_preset = build_three_object_demo_preset(_parent_scene_id)
    DEMO_SCENE_PRESETS[_three_obj_preset.scene_id] = _three_obj_preset

# Huella XY conservadora (m) alineada con ycb_obb_dataset / collision_dims.
_FOOTPRINT_LW_M: Dict[str, Tuple[float, float]] = {
    "cracker_box": (0.158, 0.060),
    "sugar_box": (0.089, 0.038),
    "mustard_bottle": (0.121, 0.106),
    "chips_can": (0.075, 0.075),
}


def is_demo_scene_preset(name: str) -> bool:
    key = str(name).strip()
    return (
        key in DEMO_SCENE_PRESETS
        or key == "two_boxes_03"
        or key == "chips_mustard_01"
    )


def is_two_boxes_scene_preset(name: str) -> bool:
    return str(name).strip() in TWO_BOXES_SCENE_IDS


def is_chips_mustard_scene_preset(name: str) -> bool:
    return str(name).strip() in CHIPS_MUSTARD_SCENE_IDS


def is_pair_validation_scene_preset(name: str) -> bool:
    key = str(name).strip()
    return key in TWO_BOXES_SCENE_IDS or key in CHIPS_MUSTARD_SCENE_IDS


def is_demo_scene_02_remaining_sugar_mustard_scene_id(scene_id: str) -> bool:
    return (
        str(scene_id or "").strip().lower()
        == DEMO_SCENE_02_REMAINING_SUGAR_MUSTARD_SCENE_ID
    )


def demo_scene_policy_scene_id_for_preset(preset_name: str) -> str:
    """scene_id de políticas pick/place (demo_scene_02 para presets derivados)."""
    key = normalize_demo_scene_policy_key(preset_name)
    if key in DEMO_SCENE_02_DEPOSIT_POLICY_SCENE_IDS:
        return "demo_scene_02"
    if key == DEMO_SCENE_02_REMAINING_SUGAR_MUSTARD_SCENE_ID:
        return "demo_scene_02"
    if is_demo_scene_3obj_scene_id(key):
        return key.removesuffix("_3obj")
    return key


def get_demo_scene_preset(name: str) -> DemoScenePreset:
    key = str(name).strip()
    if key == "two_boxes_03":
        from panda_vision.spawn.two_boxes_random_layout import two_boxes_03_reference_preset

        return two_boxes_03_reference_preset()
    if key == "chips_mustard_01":
        from panda_vision.spawn.chips_mustard_random_layout import (
            chips_mustard_01_reference_preset,
        )

        return chips_mustard_01_reference_preset()
    preset = DEMO_SCENE_PRESETS.get(key)
    if preset is None:
        raise ValueError(f"demo scene preset desconocido: '{key}'")
    return preset


def runtime_labels_from_scene_objects(
    scene_objects: Sequence[Dict[str, Any]],
) -> List[str]:
    """Labels únicos en scene_objects (orden canónico demo si aplica)."""
    found: List[str] = []
    seen: set = set()
    for obj in scene_objects:
        if not isinstance(obj, dict):
            continue
        lb = str(obj.get("label", "")).strip().lower()
        if not lb or lb in seen:
            continue
        seen.add(lb)
        found.append(lb)
    canonical = [lb for lb in DEMO_SCENE_OBJECT_LABELS if lb in seen]
    if set(canonical) == seen and len(canonical) == len(found):
        return list(canonical)
    return sorted(found)


def is_consolidated_demo_scene_objects(
    scene_objects: Sequence[Dict[str, Any]],
) -> bool:
    labels = {
        str(o.get("label", "")).strip().lower()
        for o in scene_objects
        if isinstance(o, dict) and str(o.get("label", "")).strip()
    }
    return labels == set(DEMO_SCENE_OBJECT_LABELS)


def log_demo_scene_vision_labels(
    logger: Any,
    *,
    scene_preset: str,
    runtime_labels: Sequence[str],
    vision_labels: Sequence[str],
) -> Tuple[bool, List[str]]:
    """Compara labels RuntimeScene vs objetos operativos en /vision_to_executor."""
    expected = list(DEMO_SCENE_OBJECT_LABELS)
    runtime_sorted = [
        lb for lb in expected if lb in {str(x).strip().lower() for x in runtime_labels}
    ]
    vision_set = {str(x).strip().lower() for x in vision_labels if str(x).strip()}
    vision_sorted = [lb for lb in expected if lb in vision_set]
    missing = [lb for lb in expected if lb not in vision_set]
    ok = len(missing) == 0 and set(runtime_sorted) == set(expected)
    preset_s = str(scene_preset).strip() or "(no indicado)"
    if logger is not None:
        try:
            logger.info(
                "[DEMO_SCENE_VISION_LABELS]\n"
                "scene_preset=%s\n"
                "runtime_labels=%s\n"
                "vision_labels=%s\n"
                "missing_from_vision=%s\n"
                "result=%s"
                % (
                    preset_s,
                    runtime_sorted,
                    vision_sorted,
                    missing,
                    "OK" if ok else "FAIL",
                )
            )
        except Exception:
            pass
    return ok, missing


def chips_can_seed_pose_uniform(random_seed: int) -> Tuple[float, float, float]:
    """Réplica de spawn_ycb_object (chips_can, random_pose, región 0.45–0.70 × ±0.20)."""
    import random

    rng = random.Random(int(random_seed))
    x = float(rng.uniform(0.45, 0.70))
    y = float(rng.uniform(-0.20, 0.20))
    yaw = float(rng.uniform(-math.pi, math.pi))
    return x, y, yaw


def is_chips_can_banned_seed(seed: Optional[int]) -> bool:
    return seed is not None and int(seed) == CHIPS_CAN_BANNED_SEED


def _resolve_footprint_lwh(label: str) -> Tuple[float, float]:
    """(length, width) con length >= width, alineado con known_box_gt / YAML."""
    try:
        from panda_vision.spawn.semantic_spawn_sampling import resolve_box_dims_lwh

        return resolve_box_dims_lwh(label)
    except Exception:
        fl, fw = _FOOTPRINT_LW_M.get(str(label).strip().lower(), (0.08, 0.08))
        return (max(fl, fw), min(fl, fw))


def _semantic_footprint_corners_xy(
    center_xy: Tuple[float, float],
    yaw_rad: float,
    length_m: float,
    width_m: float,
) -> List[Tuple[float, float]]:
    cx, cy = float(center_xy[0]), float(center_xy[1])
    half_l = 0.5 * float(length_m)
    half_w = 0.5 * float(width_m)
    c = math.cos(float(yaw_rad))
    s = math.sin(float(yaw_rad))
    e_len = (c, s)
    e_wid = (-s, c)
    return [
        (
            cx + half_l * e_len[0] + half_w * e_wid[0],
            cy + half_l * e_len[1] + half_w * e_wid[1],
        ),
        (
            cx + half_l * e_len[0] - half_w * e_wid[0],
            cy + half_l * e_len[1] - half_w * e_wid[1],
        ),
        (
            cx - half_l * e_len[0] - half_w * e_wid[0],
            cy - half_l * e_len[1] - half_w * e_wid[1],
        ),
        (
            cx - half_l * e_len[0] + half_w * e_wid[0],
            cy - half_l * e_len[1] + half_w * e_wid[1],
        ),
    ]


def object_footprint_fits_working_table(
    obj: DemoSceneObjectPose,
    *,
    table_margin_m: float = TABLE_EDGE_MARGIN_M,
) -> Tuple[bool, str]:
    """Todas las esquinas del footprint dentro del tablero físico vision_test_table."""
    cx, cy = TABLE_CENTER_XY
    hx, hy = TABLE_HALF_EXTENT_XY
    m = float(table_margin_m)
    x_lo = float(cx) - float(hx) + m
    x_hi = float(cx) + float(hx) - m
    y_lo = float(cy) - float(hy) + m
    y_hi = float(cy) + float(hy) - m
    length_m, width_m = _resolve_footprint_lwh(obj.label)
    for i, (px, py) in enumerate(
        _semantic_footprint_corners_xy((obj.x, obj.y), obj.yaw, length_m, width_m)
    ):
        if px < x_lo or px > x_hi:
            return (
                False,
                "corner[%d]_x=%.4f not in [%.4f,%.4f]"
                % (i, px, x_lo, x_hi),
            )
        if py < y_lo or py > y_hi:
            return (
                False,
                "corner[%d]_y=%.4f not in [%.4f,%.4f]"
                % (i, py, y_lo, y_hi),
            )
    return True, "ok"


def is_chips_can_demo_xy_allowed(
    x: float,
    y: float,
    *,
    reference_xy: Tuple[float, float] = CHIPS_CAN_BANNED_REFERENCE_XY,
    tolerance_m: float = CHIPS_CAN_BANNED_XY_TOLERANCE_M,
) -> Tuple[bool, str]:
    bx, by = float(reference_xy[0]), float(reference_xy[1])
    dist = math.hypot(float(x) - bx, float(y) - by)
    if dist < float(tolerance_m):
        return (
            False,
            "near_banned_seed1001_region dist_xy=%.4f tolerance=%.4f ref=(%.4f,%.4f)"
            % (dist, tolerance_m, bx, by),
        )
    return True, "ok"


def _collision_xy_radius_m(
    label: str,
    *,
    footprint_safety_scale: float = 1.1,
) -> float:
    """Radio XY aproximado desde collision_dims (o huella YAML)."""
    key = str(label).strip().lower()
    try:
        from panda_vision.grasp.object_grasp_policy import get_collision_dimensions

        col = get_collision_dimensions(key)
        if col is not None:
            shape = str(col.get("shape", ""))
            if shape == "cylinder" and col.get("cylinder"):
                radius, _h = col["cylinder"]
                return float(radius) * max(1.0, footprint_safety_scale)
            if shape == "box" and col.get("box"):
                sx, sy, _sz = col["box"]
                half_diag = 0.5 * math.hypot(float(sx), float(sy))
                return half_diag * max(1.0, footprint_safety_scale)
    except Exception:
        pass
    length_m, width_m = _FOOTPRINT_LW_M.get(key, (0.08, 0.08))
    half_diag = 0.5 * math.hypot(length_m, width_m)
    return half_diag * max(1.0, footprint_safety_scale)


def _footprint_xy_radius_m(
    label: str,
    *,
    footprint_safety_scale: float = 1.1,
) -> float:
    length_m, width_m = _resolve_footprint_lwh(label)
    half_diag = 0.5 * math.hypot(length_m, width_m)
    return half_diag * max(1.0, float(footprint_safety_scale))


def required_pairwise_min_distance_m(
    label_a: str,
    label_b: str,
    *,
    min_clearance_m: float,
    footprint_safety_scale: float = 1.1,
) -> float:
    return (
        _footprint_xy_radius_m(label_a, footprint_safety_scale=footprint_safety_scale)
        + _footprint_xy_radius_m(label_b, footprint_safety_scale=footprint_safety_scale)
        + float(min_clearance_m)
    )


@dataclass(frozen=True)
class DemoSceneCollisionPairResult:
    obj_a: str
    obj_b: str
    distance_xy: float
    required_min_distance: float
    ok: bool


def validate_demo_scene_layout(
    objects: Sequence[DemoSceneObjectPose],
    *,
    min_clearance_m: float = DEFAULT_DEMO_SCENE_MIN_CLEARANCE_M,
    footprint_safety_scale: float = 1.1,
) -> Tuple[bool, List[DemoSceneCollisionPairResult]]:
    pairs: List[DemoSceneCollisionPairResult] = []
    all_ok = True
    for i, a in enumerate(objects):
        for b in objects[i + 1 :]:
            dist = math.hypot(a.x - b.x, a.y - b.y)
            req = required_pairwise_min_distance_m(
                a.label,
                b.label,
                min_clearance_m=min_clearance_m,
                footprint_safety_scale=footprint_safety_scale,
            )
            ok = dist >= (req - 1e-3)
            if not ok:
                all_ok = False
            pairs.append(
                DemoSceneCollisionPairResult(
                    obj_a=a.label,
                    obj_b=b.label,
                    distance_xy=float(dist),
                    required_min_distance=float(req),
                    ok=ok,
                )
            )
    return all_ok, pairs


def validate_demo_scene_object(
    obj: DemoSceneObjectPose,
) -> Tuple[bool, str]:
    table_ok, table_reason = object_footprint_fits_working_table(obj)
    if not table_ok:
        return False, "off_table: %s" % table_reason
    if obj.label == "chips_can":
        if is_chips_can_banned_seed(obj.seed):
            return False, "banned_seed=%d" % int(obj.seed)
        if obj.seed is not None and int(obj.seed) not in CHIPS_CAN_ALLOWED_SEEDS:
            return (
                False,
                "seed_not_in_allowed_list seed=%d allowed=%s"
                % (int(obj.seed), list(CHIPS_CAN_ALLOWED_SEEDS)),
            )
        allowed, reason = is_chips_can_demo_xy_allowed(obj.x, obj.y)
        return allowed, reason
    return True, "ok"


def validate_demo_scene_preset(
    preset: DemoScenePreset,
    *,
    min_clearance_m: float = DEFAULT_DEMO_SCENE_MIN_CLEARANCE_M,
    footprint_safety_scale: float = 1.1,
) -> Tuple[bool, str]:
    for obj in preset.objects:
        ok_obj, reason = validate_demo_scene_object(obj)
        if not ok_obj:
            return False, "object %s: %s" % (obj.label, reason)
    layout_ok, pairs = validate_demo_scene_layout(
        preset.objects,
        min_clearance_m=min_clearance_m,
        footprint_safety_scale=footprint_safety_scale,
    )
    if not layout_ok:
        bad = next(p for p in pairs if not p.ok)
        return (
            False,
            "collision %s vs %s dist=%.4f required=%.4f"
            % (bad.obj_a, bad.obj_b, bad.distance_xy, bad.required_min_distance),
        )
    return True, "ok"


def demo_scene_to_legacy_spec(preset: DemoScenePreset) -> List[Dict[str, object]]:
    """Formato compatible con _SCENE_PRESET_SPECS del runtime_scene_spawner."""
    out: List[Dict[str, object]] = []
    for obj in preset.objects:
        entry: Dict[str, object] = {
            "label": obj.label,
            "x": float(obj.x),
            "y": float(obj.y),
            "yaw": float(obj.yaw),
        }
        if obj.seed is not None:
            entry["seed"] = int(obj.seed)
        out.append(entry)
    return out


def log_chips_can_reachability_region(
    logger: Any,
    *,
    seed: Optional[int],
    x: float,
    y: float,
    allowed: bool,
    reason: str,
) -> None:
    if logger is None:
        return
    try:
        logger.info(
            "[CHIPS_CAN_DEMO_REACHABILITY_REGION]\n"
            "seed=%s\n"
            "sampled_xy=(%.4f, %.4f)\n"
            "allowed=%s\n"
            "reason=%s"
            % (
                str(seed) if seed is not None else "none",
                float(x),
                float(y),
                str(bool(allowed)).lower(),
                reason,
            )
        )
    except Exception:
        pass


def log_demo_scene_build(
    logger: Any,
    *,
    scene_id: str,
    objects: Sequence[DemoSceneObjectPose],
    result: str,
) -> None:
    if logger is None:
        return
    labels = [o.label for o in objects]
    try:
        logger.info(
            "[DEMO_SCENE_BUILD]\n"
            "scene_id=%s\n"
            "objects=%s\n"
            "result=%s" % (scene_id, labels, result)
        )
    except Exception:
        pass


def log_demo_scene_object(
    logger: Any,
    obj: DemoSceneObjectPose,
    *,
    allowed: bool,
    reason: str,
) -> None:
    if logger is None:
        return
    try:
        logger.info(
            "[DEMO_SCENE_OBJECT]\n"
            "label=%s\n"
            "seed=%s\n"
            "x=%.4f\n"
            "y=%.4f\n"
            "yaw=%.4f\n"
            "allowed=%s\n"
            "reason=%s"
            % (
                obj.label,
                str(obj.seed) if obj.seed is not None else "none",
                float(obj.x),
                float(obj.y),
                float(obj.yaw),
                str(bool(allowed)).lower(),
                reason,
            )
        )
    except Exception:
        pass


def log_demo_scene_table_check(
    logger: Any,
    obj: DemoSceneObjectPose,
    *,
    allowed: bool,
    reason: str,
) -> None:
    if logger is None:
        return
    try:
        logger.info(
            "[DEMO_SCENE_TABLE_CHECK]\n"
            "label=%s\n"
            "x=%.4f\n"
            "y=%.4f\n"
            "yaw=%.4f\n"
            "allowed=%s\n"
            "reason=%s"
            % (
                obj.label,
                float(obj.x),
                float(obj.y),
                float(obj.yaw),
                str(bool(allowed)).lower(),
                reason,
            )
        )
    except Exception:
        pass


def log_demo_scene_collision_check(
    logger: Any,
    pair: DemoSceneCollisionPairResult,
) -> None:
    if logger is None:
        return
    try:
        logger.info(
            "[DEMO_SCENE_COLLISION_CHECK]\n"
            "obj_a=%s\n"
            "obj_b=%s\n"
            "distance_xy=%.4f\n"
            "required_min_distance=%.4f\n"
            "result=%s"
            % (
                pair.obj_a,
                pair.obj_b,
                pair.distance_xy,
                pair.required_min_distance,
                "ok" if pair.ok else "reject",
            )
        )
    except Exception:
        pass


def object_in_demo_scene_02_reach_zone(obj: DemoSceneObjectPose) -> Tuple[bool, str]:
    """Centro XY dentro de la zona cómoda de alcance para demo_scene_02 v2."""
    if obj.label == "cracker_box":
        x_lo, x_hi = DEMO_SCENE_02_CRACKER_REACH_X_RANGE
    elif obj.label in DEMO_SCENE_02_REACH_X_OVERRIDES:
        x_lo, x_hi = DEMO_SCENE_02_REACH_X_OVERRIDES[obj.label]
    else:
        x_lo, x_hi = DEMO_SCENE_02_REACH_X_RANGE
    y_lo, y_hi = DEMO_SCENE_02_REACH_Y_RANGE
    if obj.x < x_lo or obj.x > x_hi:
        return False, "x=%.4f not in [%.4f,%.4f]" % (obj.x, x_lo, x_hi)
    if obj.y < y_lo or obj.y > y_hi:
        return False, "y=%.4f not in [%.4f,%.4f]" % (obj.y, y_lo, y_hi)
    return True, "ok"


def validate_demo_scene_02_pick_order_labels(
    preset: DemoScenePreset,
    *,
    pick_order: Sequence[str] = DEMO_SCENE_02_PICK_ORDER,
) -> Tuple[bool, str]:
    labels = {o.label for o in preset.objects}
    expected = {_label_lower(lb) for lb in pick_order if _label_lower(lb)}
    if labels != expected:
        missing = sorted(expected - labels)
        extra = sorted(labels - expected)
        return False, "pick_order_mismatch missing=%s extra=%s" % (missing, extra)
    return True, "ok"


def _label_lower(value: Any) -> str:
    return str(value or "").strip().lower()


def validate_demo_scene_02_remaining_sugar_mustard_layout(
    preset: DemoScenePreset,
    *,
    min_clearance_m: float = DEFAULT_DEMO_SCENE_MIN_CLEARANCE_M,
    footprint_safety_scale: float = DEMO_SCENE_LAYOUT_FOOTPRINT_SCALE,
    pick_order: Sequence[str] = DEMO_SCENE_02_REMAINING_SUGAR_MUSTARD_PICK_ORDER,
) -> Tuple[bool, str]:
    if preset.scene_id != DEMO_SCENE_02_REMAINING_SUGAR_MUSTARD_SCENE_ID:
        return False, "wrong_scene_id=%s" % preset.scene_id
    ok_po, reason_po = validate_demo_scene_02_pick_order_labels(
        preset, pick_order=pick_order
    )
    if not ok_po:
        return False, reason_po
    for obj in preset.objects:
        reach_ok, reach_reason = object_in_demo_scene_02_reach_zone(obj)
        if not reach_ok:
            return False, "reach_zone %s: %s" % (obj.label, reach_reason)
    return validate_demo_scene_preset(
        preset,
        min_clearance_m=min_clearance_m,
        footprint_safety_scale=footprint_safety_scale,
    )


def _is_demo_scene_02_clear_table_family(scene_id: str) -> bool:
    sid = str(scene_id or "").strip()
    return sid in ("demo_scene_02", "demo_scene_02_3obj")


def validate_demo_scene_02_clear_table_layout(
    preset: DemoScenePreset,
    *,
    min_clearance_m: float = DEFAULT_DEMO_SCENE_MIN_CLEARANCE_M,
    footprint_safety_scale: float = DEMO_SCENE_LAYOUT_FOOTPRINT_SCALE,
    pick_order: Optional[Sequence[str]] = None,
) -> Tuple[bool, str]:
    if preset.scene_id == DEMO_SCENE_02_REMAINING_SUGAR_MUSTARD_SCENE_ID:
        return validate_demo_scene_02_remaining_sugar_mustard_layout(
            preset,
            min_clearance_m=min_clearance_m,
            footprint_safety_scale=footprint_safety_scale,
        )
    if not _is_demo_scene_02_clear_table_family(preset.scene_id):
        return validate_demo_scene_preset(
            preset,
            min_clearance_m=min_clearance_m,
            footprint_safety_scale=footprint_safety_scale,
        )
    effective_pick_order = (
        list(pick_order)
        if pick_order is not None
        else (
            list(DEMO_SCENE_3OBJ_PICK_ORDER)
            if preset.scene_id == "demo_scene_02_3obj"
            else list(DEMO_SCENE_02_PICK_ORDER)
        )
    )
    ok_po, reason_po = validate_demo_scene_02_pick_order_labels(
        preset, pick_order=effective_pick_order
    )
    if not ok_po:
        return False, reason_po
    for obj in preset.objects:
        reach_ok, reach_reason = object_in_demo_scene_02_reach_zone(obj)
        if not reach_ok:
            return False, "reach_zone %s: %s" % (obj.label, reach_reason)
    return validate_demo_scene_preset(
        preset,
        min_clearance_m=min_clearance_m,
        footprint_safety_scale=footprint_safety_scale,
    )


def log_demo_scene_02_clear_table_layout(
    logger: Any,
    preset: DemoScenePreset,
    *,
    pick_order: Sequence[str] = DEMO_SCENE_02_PICK_ORDER,
) -> None:
    if logger is None or preset.scene_id not in (
        "demo_scene_02",
        "demo_scene_02_3obj",
        DEMO_SCENE_02_REMAINING_SUGAR_MUSTARD_SCENE_ID,
    ):
        return
    try:
        logger.info(
            "[DEMO_SCENE_02_LAYOUT]\n"
            "version=%s\n"
            "object_count=%d\n"
            "pick_order=%s"
            % (
                DEMO_SCENE_02_LAYOUT_VERSION,
                len(preset.objects),
                list(pick_order),
            )
        )
        for idx, lb in enumerate(pick_order):
            obj = next((o for o in preset.objects if o.label == lb), None)
            if obj is None:
                continue
            logger.info(
                "[DEMO_SCENE_OBJECT_POSE]\n"
                "label=%s\n"
                "order_index=%d\n"
                "x=%.4f\n"
                "y=%.4f\n"
                "yaw=%.4f\n"
                "seed=%s"
                % (
                    obj.label,
                    int(obj.order_index if obj.order_index is not None else idx),
                    float(obj.x),
                    float(obj.y),
                    float(obj.yaw),
                    str(obj.seed) if obj.seed is not None else "none",
                )
            )
    except Exception:
        pass


def log_demo_scene_layout_validate(
    logger: Any,
    *,
    scene_id: str,
    ok: bool,
    reason: str = "ok",
) -> None:
    if logger is None:
        return
    try:
        logger.info(
            "[DEMO_SCENE_LAYOUT_VALIDATE]\n"
            "scene_id=%s\n"
            "result=%s\n"
            "reason=%s"
            % (scene_id, "OK" if ok else "FAIL", reason if ok else reason)
        )
    except Exception:
        pass


def log_demo_scene_preset_validation(
    logger: Any,
    preset: DemoScenePreset,
    *,
    min_clearance_m: float,
    footprint_safety_scale: float,
) -> Tuple[bool, str]:
    """Valida preset, emite logs estructurados y devuelve (ok, reason)."""
    if preset.scene_id in ("demo_scene_02", "demo_scene_02_3obj", DEMO_SCENE_02_REMAINING_SUGAR_MUSTARD_SCENE_ID):
        log_demo_scene_02_clear_table_layout(
            logger,
            preset,
            pick_order=(
                DEMO_SCENE_02_REMAINING_SUGAR_MUSTARD_PICK_ORDER
                if preset.scene_id == DEMO_SCENE_02_REMAINING_SUGAR_MUSTARD_SCENE_ID
                else (
                    DEMO_SCENE_3OBJ_PICK_ORDER
                    if preset.scene_id == "demo_scene_02_3obj"
                    else DEMO_SCENE_02_PICK_ORDER
                )
            ),
        )
    for obj in preset.objects:
        table_ok, table_reason = object_footprint_fits_working_table(obj)
        log_demo_scene_table_check(
            logger, obj, allowed=table_ok, reason=table_reason
        )
        ok_obj, reason = validate_demo_scene_object(obj)
        log_demo_scene_object(logger, obj, allowed=ok_obj, reason=reason)
        if obj.label == "chips_can":
            log_chips_can_reachability_region(
                logger,
                seed=obj.seed,
                x=obj.x,
                y=obj.y,
                allowed=ok_obj,
                reason=reason,
            )
    _, pairs = validate_demo_scene_layout(
        preset.objects,
        min_clearance_m=min_clearance_m,
        footprint_safety_scale=footprint_safety_scale,
    )
    for pair in pairs:
        log_demo_scene_collision_check(logger, pair)
    if preset.scene_id in (
        "demo_scene_02",
        "demo_scene_02_3obj",
        DEMO_SCENE_02_REMAINING_SUGAR_MUSTARD_SCENE_ID,
    ):
        ok, reason = validate_demo_scene_02_clear_table_layout(
            preset,
            min_clearance_m=min_clearance_m,
            footprint_safety_scale=footprint_safety_scale,
        )
    else:
        ok, reason = validate_demo_scene_preset(
            preset,
            min_clearance_m=min_clearance_m,
            footprint_safety_scale=footprint_safety_scale,
        )
    log_demo_scene_layout_validate(
        logger, scene_id=preset.scene_id, ok=ok, reason=reason
    )
    log_demo_scene_build(
        logger,
        scene_id=preset.scene_id,
        objects=preset.objects,
        result="accept" if ok else "reject",
    )
    if (
        not ok
        and "collision" in reason
        and float(min_clearance_m) > DEFAULT_DEMO_SCENE_MIN_CLEARANCE_M + 1e-6
        and logger is not None
    ):
        ok_loose, _ = validate_demo_scene_preset(
            preset,
            min_clearance_m=DEFAULT_DEMO_SCENE_MIN_CLEARANCE_M,
            footprint_safety_scale=footprint_safety_scale,
        )
        if ok_loose:
            try:
                logger.warning(
                    "[DEMO_SCENE_VALIDATION] preset=%s falla con clearance=%.3f; "
                    "prueba demo_scene_min_clearance_m:=%.3f (default del launch)."
                    % (
                        preset.scene_id,
                        float(min_clearance_m),
                        DEFAULT_DEMO_SCENE_MIN_CLEARANCE_M,
                    )
                )
            except Exception:
                pass
    return ok, reason


def assert_builtin_presets_valid() -> None:
    """Llamar desde tests; no en import del módulo (evita tumbar runtime_scene_spawner)."""
    for scene_id, preset in DEMO_SCENE_PRESETS.items():
        if scene_id in (
            "demo_scene_02",
            "demo_scene_02_3obj",
            DEMO_SCENE_02_REMAINING_SUGAR_MUSTARD_SCENE_ID,
        ):
            ok, reason = validate_demo_scene_02_clear_table_layout(
                preset,
                min_clearance_m=DEFAULT_DEMO_SCENE_MIN_CLEARANCE_M,
                footprint_safety_scale=DEMO_SCENE_LAYOUT_FOOTPRINT_SCALE,
            )
        else:
            ok, reason = validate_demo_scene_preset(
                preset,
                min_clearance_m=DEFAULT_DEMO_SCENE_MIN_CLEARANCE_M,
                footprint_safety_scale=DEMO_SCENE_LAYOUT_FOOTPRINT_SCALE,
            )
        if not ok:
            raise RuntimeError("Preset demo inválido: %s (%s)" % (scene_id, reason))
    ok_03, reason_03 = validate_demo_scene_preset(
        get_demo_scene_preset("two_boxes_03"),
        min_clearance_m=DEFAULT_DEMO_SCENE_MIN_CLEARANCE_M,
        footprint_safety_scale=DEMO_SCENE_LAYOUT_FOOTPRINT_SCALE,
    )
    if not ok_03:
        raise RuntimeError("Preset demo inválido: two_boxes_03 (%s)" % reason_03)
    ok_cm, reason_cm = validate_demo_scene_preset(
        get_demo_scene_preset("chips_mustard_01"),
        min_clearance_m=DEFAULT_DEMO_SCENE_MIN_CLEARANCE_M,
        footprint_safety_scale=DEMO_SCENE_LAYOUT_FOOTPRINT_SCALE,
    )
    if not ok_cm:
        raise RuntimeError("Preset demo inválido: chips_mustard_01 (%s)" % reason_cm)
