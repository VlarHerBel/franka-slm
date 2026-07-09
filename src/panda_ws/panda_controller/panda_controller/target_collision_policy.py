"""Política de colisión del target y ruta object_safe_above (demo multiobjeto)."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Set, Tuple

KNOWN_BOX_LABELS = frozenset({"cracker_box", "sugar_box", "gelatin_box"})

OK_FULL_PICK_ROUTE_PREVALIDATED = "OK_FULL_PICK_ROUTE_PREVALIDATED"

# Resultados de fase-1 / diferidos: nunca autorizan movimiento en demo multiobjeto.
REJECTED_DEFERRED_PICK_ROUTE_RESULTS = frozenset(
    {
        "OK_OBJECT_SAFE_ABOVE_PRELUDE_PHASE1",
        "OK_OBJECT_SAFE_ABOVE_DEFERRED",
        "OK_PRELUDE_PHASE1",
    }
)


def pick_route_result_allows_demo_motion(
    *,
    plan_before_result: str,
    cartesian_descend_prevalidated: bool,
    full_route_required: bool,
) -> Tuple[bool, str]:
    """True solo si la ruta completa está prevalidada antes de mover desde HOME."""
    result = str(plan_before_result or "").strip()
    if not full_route_required:
        return True, "full_route_not_required"
    if result in REJECTED_DEFERRED_PICK_ROUTE_RESULTS:
        return False, "deferred_or_incomplete_pick_route"
    if result == OK_FULL_PICK_ROUTE_PREVALIDATED:
        if not cartesian_descend_prevalidated:
            return False, "cartesian_descend_not_prevalidated"
        return True, "ok"
    if result in ("OK_FULL_PICK_ROUTE", "OK"):
        if not cartesian_descend_prevalidated:
            return False, "cartesian_descend_not_prevalidated"
        return True, "ok"
    return False, "plan_before_result_not_full_route"


def resolve_selected_entry_target(
    candidate: Dict[str, Any],
    plan_before_result: str,
) -> str:
    if str(plan_before_result) == OK_FULL_PICK_ROUTE_PREVALIDATED:
        return "object_safe_above_tcp"
    if bool(candidate.get("_object_safe_above_route_prevalidated")):
        return "object_safe_above_tcp"
    return "pregrasp_tcp"


def _entity_short(name: str) -> str:
    return str(name or "").strip().split("::")[-1]


def _to_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _xyz_from_candidate(candidate: Dict[str, Any]) -> Optional[Tuple[float, float, float]]:
    for key in ("grasp_center_base", "known_box_center_base", "position"):
        raw = candidate.get(key)
        if isinstance(raw, (list, tuple)) and len(raw) >= 2:
            z = float(raw[2]) if len(raw) >= 3 else 0.0
            return (float(raw[0]), float(raw[1]), z)
    return None


def include_target_collision(
    candidate: Dict[str, Any],
    *,
    override: Optional[bool] = None,
) -> bool:
    if override is not None:
        return bool(override)
    return bool(candidate.get("use_target_collision_until_pregrasp", True))


def shape_dims_from_collision_dims(
    col_dims: Optional[Dict[str, Any]],
    height_fallback_m: Optional[float],
    *,
    padding_m: float = 0.0,
) -> Tuple[str, List[float]]:
    if not isinstance(col_dims, dict):
        h = float(height_fallback_m) if height_fallback_m else 0.080
        return "box", [0.060, 0.060, h]
    shape = str(col_dims.get("shape", "box")).lower()
    pad = float(padding_m)
    if shape == "cylinder":
        cyl = col_dims.get("cylinder")
        if isinstance(cyl, (list, tuple)) and len(cyl) >= 2:
            try:
                r = float(cyl[0]) + pad / 2.0
                h = float(cyl[1]) + pad
                return "cylinder", [max(h, 1e-4), max(r, 1e-4)]
            except (TypeError, ValueError):
                pass
    if shape == "sphere":
        cyl = col_dims.get("cylinder")
        if isinstance(cyl, (list, tuple)) and len(cyl) >= 2:
            try:
                r = float(cyl[0]) + pad / 2.0
                h = float(cyl[1]) + pad
                return "cylinder", [max(h, 1e-4), max(r, 1e-4)]
            except (TypeError, ValueError):
                pass
    box = col_dims.get("box") or col_dims.get("box_fallback")
    if isinstance(box, (list, tuple)) and len(box) >= 3:
        try:
            sx = float(box[0]) + pad
            sy = float(box[1]) + pad
            sz = float(box[2]) + pad
            return "box", [max(sx, 1e-4), max(sy, 1e-4), max(sz, 1e-4)]
        except (TypeError, ValueError):
            pass
    h = float(height_fallback_m) if height_fallback_m else 0.080
    return "box", [0.060 + pad, 0.060 + pad, h + pad]


def build_target_collision_obstacle(
    candidate: Dict[str, Any],
    *,
    table_z_m: float,
    padding_m: float = 0.0,
) -> Optional[Dict[str, Any]]:
    """Construye entrada de obstáculo MoveIt para el target (role=target)."""
    label = str(candidate.get("label", "")).strip().lower()
    if not label:
        return None
    pos = _xyz_from_candidate(candidate)
    if pos is None:
        return None
    ent = _entity_short(
        str(candidate.get("gt_entity_name") or candidate.get("entity_name") or "")
    )
    col_dims = candidate.get("collision_dims")
    db_h = _to_float(candidate.get("db_height_m")) or _to_float(
        candidate.get("effective_height_m")
    )
    shape, dims = shape_dims_from_collision_dims(
        col_dims if isinstance(col_dims, dict) else None,
        db_h,
        padding_m=padding_m,
    )
    height_for_center = dims[0] if shape == "cylinder" else dims[2]
    center_z = float(table_z_m) + float(height_for_center) / 2.0
    yaw = _to_float(candidate.get("known_box_yaw_rad"))
    if yaw is None:
        yaw = _to_float(candidate.get("object_yaw_rad"))
    if yaw is None:
        yaw = 0.0
    col_id = f"target_{ent}" if ent else f"target_{label}"
    return {
        "idx": 0,
        "label": label,
        "entity_name": ent,
        "position": (float(pos[0]), float(pos[1]), float(center_z)),
        "collision_dims": col_dims,
        "db_height_m": db_h,
        "is_target": True,
        "object_yaw_rad": float(yaw),
        "collision_id": col_id,
        "role": "target",
        "_collision_shape": shape,
        "_collision_dims_resolved": [float(v) for v in dims],
    }


def pick_prevalidate_planning_scene_update_required(
    *,
    include_target: bool,
    scene_obstacles: List[Dict[str, Any]],
) -> bool:
    """True cuando hay que poblar la planning scene antes de prevalidar la ruta pick."""
    return bool(scene_obstacles) or bool(include_target)


def format_mustard_target_collision_ready_log(
    *,
    label: str,
    target_collision_present: bool,
    object_id: str,
) -> str:
    if str(label or "").strip().lower() != "mustard_bottle":
        return ""
    return (
        "[MUSTARD_TARGET_COLLISION_READY]\n"
        "target_collision_present=%s\n"
        "object_id=%s\n"
        "result=%s"
        % (
            str(bool(target_collision_present)).lower(),
            object_id or "n/a",
            "OK" if target_collision_present else "FAIL",
        )
    )


def scene_has_target_collision(
    scene_obstacles: List[Dict[str, Any]],
    candidate: Dict[str, Any],
) -> bool:
    for obs in scene_obstacles:
        if not isinstance(obs, dict):
            continue
        if bool(obs.get("is_target", False)):
            return True
        ent = _entity_short(str(obs.get("entity_name") or ""))
        target_ent = _entity_short(
            str(candidate.get("gt_entity_name") or candidate.get("entity_name") or "")
        )
        if target_ent and ent == target_ent:
            return True
    return False


def count_non_target_obstacles(scene_obstacles: List[Dict[str, Any]]) -> int:
    n = 0
    for obs in scene_obstacles:
        if isinstance(obs, dict) and not bool(obs.get("is_target", False)):
            n += 1
    return n


def object_safe_above_stage_required(
    candidate: Dict[str, Any],
    plan_targets: Dict[str, Any],
    *,
    include_target_collision_flag: bool,
    obstacle_count: int,
    close_pregrasp_clearance_m: float = 0.15,
    min_vertical_segment_m: float = 0.05,
) -> bool:
    """True cuando la ruta debe pasar por object_safe_above antes de pregrasp."""
    if not include_target_collision_flag:
        return False
    label = str(candidate.get("label", "")).strip().lower()
    if label in ("chips_can", "sugar_box", "mustard_bottle"):
        return False
    if obstacle_count < 1:
        return False
    top_z = _to_float(candidate.get("top_z_m"))
    if top_z is None:
        top_z = _to_float(candidate.get("top_z_estimated"))
    obj_safe = candidate.get("object_safe_above_tcp")
    pre_tcp = plan_targets.get("pregrasp_tcp")
    if top_z is None or not isinstance(obj_safe, (list, tuple)) or len(obj_safe) < 3:
        return False
    if not isinstance(pre_tcp, (list, tuple)) or len(pre_tcp) < 3:
        return False
    pregrasp_z = float(pre_tcp[2])
    safe_z = float(obj_safe[2])
    clearance_above_top = pregrasp_z - float(top_z)
    if clearance_above_top <= float(close_pregrasp_clearance_m):
        return True
    return (safe_z - pregrasp_z) >= float(min_vertical_segment_m)


def target_collision_required_for_approach(
    *,
    target_collision_present: bool,
    goal_tcp_xy: Tuple[float, float],
    goal_tcp_z: float,
    object_center_xy: Tuple[float, float],
    top_z: float,
    object_radius_m: float,
    xy_margin_m: float = 0.05,
    z_clearance_m: float = 0.15,
) -> bool:
    """True si falta colisión del target y el movimiento acerca el TCP al objeto."""
    if target_collision_present:
        return False
    dist_xy = math.hypot(
        float(goal_tcp_xy[0]) - float(object_center_xy[0]),
        float(goal_tcp_xy[1]) - float(object_center_xy[1]),
    )
    near_xy = dist_xy < (float(object_radius_m) + float(xy_margin_m))
    low_z = float(goal_tcp_z) < (float(top_z) + float(z_clearance_m))
    return bool(near_xy and low_z)


def object_collision_radius_xy(candidate: Dict[str, Any]) -> float:
    col_dims = candidate.get("collision_dims")
    if isinstance(col_dims, dict):
        box = col_dims.get("box") or col_dims.get("box_fallback")
        if isinstance(box, (list, tuple)) and len(box) >= 2:
            try:
                return 0.5 * math.hypot(float(box[0]), float(box[1]))
            except (TypeError, ValueError):
                pass
        cyl = col_dims.get("cylinder")
        if isinstance(cyl, (list, tuple)) and len(cyl) >= 1:
            try:
                return float(cyl[0])
            except (TypeError, ValueError):
                pass
    fp_major = _to_float(candidate.get("footprint_major_m"))
    fp_minor = _to_float(candidate.get("footprint_minor_m"))
    if fp_major is not None and fp_minor is not None:
        return 0.5 * max(float(fp_major), float(fp_minor))
    return 0.06


def approach_requires_vertical_from_safe_above(
    *,
    from_tcp_z: float,
    to_tcp_z: float,
    object_safe_above_tcp_z: float,
    tolerance_m: float = 0.03,
) -> bool:
    """True si el destino no es un descenso vertical desde object_safe_above."""
    at_safe = abs(float(from_tcp_z) - float(object_safe_above_tcp_z)) <= tolerance_m
    vertical = abs(float(to_tcp_z) - float(from_tcp_z)) > 1e-4
    return not (at_safe and vertical and float(to_tcp_z) < float(from_tcp_z))


def invalidate_stale_demo_completed_entities(
    completed_entities: Set[str],
    runtime_entities: Set[str],
    *,
    deposited_entities: Optional[Set[str]] = None,
) -> Tuple[Set[str], List[str]]:
    if not completed_entities or not runtime_entities:
        return set(completed_entities), []
    deposited = {
        _entity_short(str(e))
        for e in (deposited_entities or set())
        if str(e).strip()
    }
    stale = sorted(
        e
        for e in completed_entities
        if e not in runtime_entities and e not in deposited
    )
    if not stale:
        return set(completed_entities), []
    kept = set(completed_entities) - set(stale)
    return kept, stale


def runtime_entities_from_scene_objects(
    scene_objects: List[Dict[str, Any]],
) -> Set[str]:
    out: Set[str] = set()
    for so in scene_objects:
        if not isinstance(so, dict):
            continue
        ent = _entity_short(str(so.get("entity_name", "")))
        if ent:
            out.add(ent)
    return out
