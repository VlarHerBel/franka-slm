"""Obstáculos MoveIt desde RuntimeScene (demo_authoritative_scene)."""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

def _entity_short(entity: str) -> str:
    ent = str(entity or "").strip()
    return ent.split("::")[-1] if ent else ""


def _label_lower(value: Any) -> str:
    return str(value or "").strip().lower()


def scene_object_labels(scene_objects: Sequence[Dict[str, Any]]) -> List[str]:
    out: List[str] = []
    for so in scene_objects:
        if not isinstance(so, dict):
            continue
        lb = _label_lower(so.get("label"))
        if lb:
            out.append(lb)
    return sorted(set(out))


def authoritative_target_is_scene_object(
    scene_obj: Dict[str, Any],
    *,
    target_label: str,
    target_entity: str,
) -> bool:
    """Target = candidato actual; ignora role stale del payload."""
    ent = _entity_short(str(scene_obj.get("entity_name", "")))
    obj_label = _label_lower(scene_obj.get("label"))
    target_label_l = _label_lower(target_label)
    target_entity_s = _entity_short(target_entity)
    if target_entity_s and ent and ent == target_entity_s:
        return True
    if (
        target_label_l
        and obj_label == target_label_l
        and (not target_entity_s or not ent or ent == target_entity_s)
    ):
        return True
    return False


def authoritative_obstacle_excluded(
    scene_obj: Dict[str, Any],
    *,
    target_label: str,
    target_entity: str,
    completed_entities: Set[str],
    completed_labels: Set[str],
    live_table_labels: Optional[Set[str]] = None,
) -> Tuple[bool, str]:
    ent = _entity_short(str(scene_obj.get("entity_name", "")))
    obj_label = _label_lower(scene_obj.get("label"))
    if authoritative_target_is_scene_object(
        scene_obj,
        target_label=target_label,
        target_entity=target_entity,
    ):
        return True, "selected_target"
    if ent and ent in completed_entities:
        return True, "completed_entity"
    if obj_label and obj_label in completed_labels:
        return True, "completed_label"
    if live_table_labels is not None and obj_label and obj_label not in live_table_labels:
        return True, "not_live_on_table"
    return False, ""


def expected_authoritative_obstacle_labels(
    scene_objects: Sequence[Dict[str, Any]],
    *,
    target_label: str,
    target_entity: str,
    completed_entities: Set[str],
    completed_labels: Set[str],
    live_table_labels: Optional[Set[str]] = None,
) -> List[str]:
    expected: List[str] = []
    for so in scene_objects:
        if not isinstance(so, dict):
            continue
        excluded, _ = authoritative_obstacle_excluded(
            so,
            target_label=target_label,
            target_entity=target_entity,
            completed_entities=completed_entities,
            completed_labels=completed_labels,
            live_table_labels=live_table_labels,
        )
        if excluded:
            continue
        lb = _label_lower(so.get("label"))
        if lb:
            expected.append(lb)
    return sorted(set(expected))


def build_authoritative_scene_obstacles(
    scene_objects: Sequence[Dict[str, Any]],
    *,
    target_label: str,
    target_entity: str,
    completed_entities: Set[str],
    completed_labels: Set[str],
    build_obstacle_fn: Callable[[int, Dict[str, Any], bool], Optional[Dict[str, Any]]],
    on_skip_obstacle: Optional[Callable[[Dict[str, Any], str], None]] = None,
    live_table_labels: Optional[Set[str]] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """obstacles = all_scene_objects - selected_target - completed_objects."""
    out: List[Dict[str, Any]] = []
    scene_labels: List[str] = []
    obstacle_labels: List[str] = []
    missing: List[str] = []
    for idx, so in enumerate(scene_objects):
        if not isinstance(so, dict):
            continue
        lb = _label_lower(so.get("label"))
        if lb:
            scene_labels.append(lb)
        excluded, exclude_reason = authoritative_obstacle_excluded(
            so,
            target_label=target_label,
            target_entity=target_entity,
            completed_entities=completed_entities,
            completed_labels=completed_labels,
            live_table_labels=live_table_labels,
        )
        is_target = authoritative_target_is_scene_object(
            so,
            target_label=target_label,
            target_entity=target_entity,
        )
        if excluded and not is_target:
            if on_skip_obstacle is not None and exclude_reason:
                on_skip_obstacle(so, exclude_reason)
            continue
        obs = build_obstacle_fn(idx, so, is_target)
        if obs is None:
            if not excluded and lb:
                missing.append(lb)
            continue
        out.append(obs)
        if not is_target:
            olb = _label_lower(obs.get("label"))
            if olb:
                obstacle_labels.append(olb)
    expected = expected_authoritative_obstacle_labels(
        scene_objects,
        target_label=target_label,
        target_entity=target_entity,
        completed_entities=completed_entities,
        completed_labels=completed_labels,
        live_table_labels=live_table_labels,
    )
    obstacle_labels_sorted = sorted(set(obstacle_labels))
    result = "OK"
    reason = ""
    if missing:
        result = "FAIL"
        reason = "missing_remaining_object_as_obstacle:%s" % ",".join(sorted(set(missing)))
    elif set(expected) != set(obstacle_labels_sorted):
        result = "FAIL"
        diff = sorted(set(expected) - set(obstacle_labels_sorted))
        if diff:
            reason = "missing_remaining_object_as_obstacle:%s" % ",".join(diff)
        else:
            extra = sorted(set(obstacle_labels_sorted) - set(expected))
            reason = "unexpected_obstacle:%s" % ",".join(extra)
    log = {
        "scene_objects": sorted(set(scene_labels)),
        "completed_labels": sorted(_label_lower(x) for x in completed_labels if x),
        "live_table_labels": sorted(live_table_labels) if live_table_labels is not None else [],
        "obstacles": obstacle_labels_sorted,
        "expected_obstacles": expected,
        "missing_build": sorted(set(missing)),
        "result": result,
        "reason": reason,
    }
    return out, log


def format_authoritative_obstacle_set_log(
    *,
    scene_id: str,
    target_label: str,
    log: Dict[str, Any],
) -> str:
    return (
        "[AUTHORITATIVE_OBSTACLE_SET]\n"
        "scene_id=%s\n"
        "target_label=%s\n"
        "scene_objects=%s\n"
        "completed_labels=%s\n"
        "obstacles=%s\n"
        "result=%s\n"
        "reason=%s"
        % (
            str(scene_id or ""),
            str(target_label or ""),
            log.get("scene_objects", []),
            log.get("completed_labels", []),
            log.get("obstacles", []),
            str(log.get("result", "")),
            str(log.get("reason", "") or "n/a"),
        )
    )
