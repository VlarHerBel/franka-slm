"""Resolución de objetos RuntimeScene para obstáculos autoritativos."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple


def _entity_short(entity: str) -> str:
    ent = str(entity or "").strip()
    return ent.split("::")[-1] if ent else ""


def _label_lower(value: Any) -> str:
    return str(value or "").strip().lower()


def build_executor_object_index(
    executor_objects: Sequence[Dict[str, Any]],
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, List[Dict[str, Any]]]]:
    by_entity: Dict[str, Dict[str, Any]] = {}
    by_label: Dict[str, List[Dict[str, Any]]] = {}
    for raw in executor_objects:
        if not isinstance(raw, dict):
            continue
        ent = _entity_short(str(raw.get("entity_name") or raw.get("gt_entity_name") or ""))
        lb = _label_lower(raw.get("label"))
        if ent:
            by_entity[ent] = raw
        if lb:
            by_label.setdefault(lb, []).append(raw)
    return by_entity, by_label


def _xyz_from_obj(obj: Dict[str, Any], *keys: str) -> Optional[Tuple[float, float, float]]:
    for key in keys:
        val = obj.get(key)
        if isinstance(val, (list, tuple)) and len(val) >= 3:
            try:
                return float(val[0]), float(val[1]), float(val[2])
            except (TypeError, ValueError):
                continue
    return None


def resolve_authoritative_scene_object(
    scene_obj: Dict[str, Any],
    *,
    executor_objects: Sequence[Dict[str, Any]],
    collision_dims_fn: Any = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Enriquece scene_obj con position/collision_dims desde payload u object policy."""
    merged = dict(scene_obj)
    ent = _entity_short(str(scene_obj.get("entity_name", "")))
    label = _label_lower(scene_obj.get("label"))
    by_entity, by_label = build_executor_object_index(executor_objects)
    payload_obj = by_entity.get(ent) if ent else None
    if payload_obj is None and label:
        candidates = by_label.get(label) or []
        if len(candidates) == 1:
            payload_obj = candidates[0]
        elif ent:
            for cand in candidates:
                if _entity_short(str(cand.get("entity_name") or "")) == ent:
                    payload_obj = cand
                    break

    position_source = "semantic_center_base"
    pos = _xyz_from_obj(
        merged,
        "semantic_center_base",
        "position",
        "grasp_center_base",
        "chosen_target_center_base",
        "center",
        "semantic_center",
    )
    if pos is None and payload_obj is not None:
        for src_key in (
            "semantic_center_base",
            "position",
            "grasp_center_base",
            "chosen_target_center_base",
            "center",
        ):
            pos = _xyz_from_obj(payload_obj, src_key)
            if pos is not None:
                position_source = "payload_%s" % src_key
                break
        if pos is not None:
            merged["position"] = [pos[0], pos[1], pos[2]]
            if merged.get("grasp_center_base") is None:
                merged["grasp_center_base"] = [pos[0], pos[1], pos[2]]

    collision_source = "scene_object"
    col_dims = (
        merged.get("collision_dims_moveit")
        or merged.get("collision_dims_inflated")
        or merged.get("collision_dims")
    )
    if col_dims is None and payload_obj is not None:
        col_dims = payload_obj.get("collision_dims")
        if col_dims is not None:
            collision_source = "payload_collision_dims"
    if col_dims is None and callable(collision_dims_fn) and label:
        try:
            col_dims = collision_dims_fn(label, 0.005)
            if col_dims is not None:
                collision_source = "known_object_geometry"
        except Exception:
            col_dims = None
    if col_dims is not None:
        merged["collision_dims"] = col_dims

    if payload_obj is not None:
        for key in (
            "dims_lwh",
            "top_z_m",
            "top_face_center_base",
            "effective_height_m",
            "db_height_m",
            "object_height_m",
            "collision_shape",
        ):
            if merged.get(key) is None and payload_obj.get(key) is not None:
                merged[key] = payload_obj.get(key)

    resolve_meta = {
        "label": label,
        "entity": ent,
        "role_payload": str(scene_obj.get("role", "")),
        "position_source": position_source if pos is not None else "missing",
        "position": pos,
        "collision_source": collision_source if col_dims is not None else "missing",
        "result": "OK" if pos is not None and col_dims is not None else "PARTIAL",
    }
    if pos is None:
        resolve_meta["result"] = "FAIL"
    elif col_dims is None:
        resolve_meta["result"] = "FAIL"
    return merged, resolve_meta


def format_authoritative_obstacle_object_resolve_log(
    *,
    selected_as_target: bool,
    completed: bool,
    meta: Dict[str, Any],
) -> str:
    pos = meta.get("position")
    pos_str = (
        "n/a"
        if pos is None
        else "(%.3f, %.3f, %.3f)" % (float(pos[0]), float(pos[1]), float(pos[2]))
    )
    return (
        "[AUTHORITATIVE_OBSTACLE_OBJECT_RESOLVE]\n"
        "label=%s\n"
        "entity=%s\n"
        "role_payload=%s\n"
        "selected_as_target=%s\n"
        "completed=%s\n"
        "position_source=%s\n"
        "position=%s\n"
        "collision_source=%s\n"
        "result=%s"
        % (
            str(meta.get("label", "")),
            str(meta.get("entity", "")),
            str(meta.get("role_payload", "")),
            str(bool(selected_as_target)).lower(),
            str(bool(completed)).lower(),
            str(meta.get("position_source", "")),
            pos_str,
            str(meta.get("collision_source", "")),
            str(meta.get("result", "")),
        )
    )


def authoritative_non_target_obstacles(
    obstacles: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    return [
        o for o in obstacles if isinstance(o, dict) and not bool(o.get("is_target", False))
    ]


def format_authoritative_obstacle_set_applied_log(
    obstacles: Sequence[Dict[str, Any]],
    *,
    result: str,
) -> str:
    labels = sorted(
        {
            _label_lower(o.get("label"))
            for o in obstacles
            if isinstance(o, dict) and _label_lower(o.get("label"))
        }
    )
    return (
        "[AUTHORITATIVE_OBSTACLE_SET_APPLIED]\n"
        "obstacles=%s\n"
        "result=%s"
        % (labels, str(result))
    )
