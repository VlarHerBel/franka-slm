"""Orden de recogida demo multiobjeto y persistencia de objetos completados."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

DEFAULT_DEMO_OBJECT_PRIORITY_ORDER: Tuple[str, ...] = (
    "cracker_box",
    "chips_can",
    "mustard_bottle",
    "sugar_box",
)

DEFAULT_DEMO_BLOCKING_RULES: Dict[str, List[str]] = {
    "sugar_box": ["cracker_box"],
}


def _label_lower(value: Any) -> str:
    return str(value or "").strip().lower()


def _entity_short(name: Any) -> str:
    return str(name or "").strip().split("::")[-1]


def parse_demo_object_priority_order(raw: Any) -> List[str]:
    if isinstance(raw, (list, tuple)):
        out = [_label_lower(x) for x in raw if _label_lower(x)]
        return out if out else list(DEFAULT_DEMO_OBJECT_PRIORITY_ORDER)
    if isinstance(raw, str) and raw.strip():
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, list):
                return parse_demo_object_priority_order(parsed)
        except json.JSONDecodeError:
            parts = [p.strip() for p in raw.split(",") if p.strip()]
            if parts:
                return [_label_lower(p) for p in parts]
    return list(DEFAULT_DEMO_OBJECT_PRIORITY_ORDER)


def parse_demo_blocking_rules(raw: Any) -> Dict[str, List[str]]:
    data: Any = raw
    if isinstance(raw, str) and raw.strip():
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return dict(DEFAULT_DEMO_BLOCKING_RULES)
    if not isinstance(data, dict):
        return dict(DEFAULT_DEMO_BLOCKING_RULES)
    out: Dict[str, List[str]] = {}
    for key, blockers in data.items():
        req = _label_lower(key)
        if not req:
            continue
        bl: List[str] = []
        if isinstance(blockers, (list, tuple)):
            for b in blockers:
                lb = _label_lower(b)
                if lb:
                    bl.append(lb)
        elif isinstance(blockers, str) and blockers.strip():
            bl = [_label_lower(blockers)]
        if bl:
            out[req] = bl
    return out if out else dict(DEFAULT_DEMO_BLOCKING_RULES)


def _entity_completed(
    entity: str,
    *,
    completed_entities: Set[str],
) -> bool:
    ent = _entity_short(entity)
    if not ent:
        return False
    return ent in completed_entities


def _label_completed(
    label: str,
    *,
    completed_labels: Set[str],
) -> bool:
    lb = _label_lower(label)
    return bool(lb) and lb in completed_labels


def collect_present_table_labels(
    objects: Sequence[Dict[str, Any]],
    *,
    completed_entities: Set[str],
    completed_labels: Set[str],
) -> Set[str]:
    """Labels operativos en mesa (excluye completados)."""
    present: Set[str] = set()
    for obj in objects:
        if not isinstance(obj, dict):
            continue
        lb = _label_lower(obj.get("label"))
        if not lb:
            continue
        ent = _entity_short(obj.get("entity_name") or obj.get("gt_entity_name"))
        if _entity_completed(ent, completed_entities=completed_entities):
            continue
        if _label_completed(lb, completed_labels=completed_labels):
            continue
        present.add(lb)
    return present


def collect_live_table_labels(
    executor_objects: Sequence[Dict[str, Any]],
    *,
    completed_entities: Set[str],
    completed_labels: Set[str],
) -> Set[str]:
    """Labels visibles en el payload live (/vision_to_executor), sin completados."""
    return collect_present_table_labels(
        executor_objects,
        completed_entities=completed_entities,
        completed_labels=completed_labels,
    )


def resolve_present_table_labels(
    *,
    scene_objects: Sequence[Dict[str, Any]],
    executor_objects: Sequence[Dict[str, Any]],
    completed_entities: Set[str],
    completed_labels: Set[str],
) -> Set[str]:
    """
    Labels en mesa para política demo.

    Si hay detecciones live, mandan sobre scene_objects estáticos (RuntimeScene YAML):
    un objeto depositado o ausente no cuenta aunque siga en scene_objects.
    """
    live = collect_live_table_labels(
        executor_objects,
        completed_entities=completed_entities,
        completed_labels=completed_labels,
    )
    if live:
        return live
    return collect_present_table_labels(
        scene_objects,
        completed_entities=completed_entities,
        completed_labels=completed_labels,
    )


def resolve_effective_demo_pick_label(
    requested_label: str,
    *,
    present_labels: Set[str],
    completed_entities: Set[str],
    completed_labels: Set[str],
    blocking_rules: Dict[str, List[str]],
    priority_order: Sequence[str],
) -> Tuple[str, str, str]:
    """
    Devuelve (selected_label, reason, blocker_label).
    reason: direct_request | blocker_before_target | priority_fallback | unchanged
    """
    requested = _label_lower(requested_label)
    if not requested:
        return "", "empty_request", ""

    if _label_completed(requested, completed_labels=completed_labels):
        return requested, "already_completed_label", ""

    blockers = list(blocking_rules.get(requested, []))
    for blocker in blockers:
        if not blocker or _label_completed(blocker, completed_labels=completed_labels):
            continue
        if blocker in present_labels:
            return blocker, "blocker_before_target", blocker

    if requested in present_labels:
        return requested, "direct_request", ""

    for pref in priority_order:
        pl = _label_lower(pref)
        if pl and pl in present_labels and not _label_completed(
            pl, completed_labels=completed_labels
        ):
            if pl != requested:
                return pl, "priority_fallback", ""
            return pl, "direct_request", ""

    return requested, "unchanged", ""


def _normalize_completed_deposits(
    raw_deposits: Any,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    if not isinstance(raw_deposits, list):
        return out
    for ent in raw_deposits:
        if not isinstance(ent, dict):
            continue
        lb = _label_lower(ent.get("label"))
        try:
            x = float(ent.get("x"))
            y = float(ent.get("y"))
        except (TypeError, ValueError):
            continue
        if not lb:
            continue
        try:
            slot_index = int(ent.get("slot_index", ent.get("slot", -1)))
        except (TypeError, ValueError):
            slot_index = -1
        entity = str(ent.get("entity") or ent.get("entity_name") or "").strip()
        slot_name = str(ent.get("slot_name") or "").strip()
        rec: Dict[str, Any] = {
            "label": lb,
            "entity": entity,
            "x": x,
            "y": y,
            "slot_index": slot_index,
            "slot": slot_index,
        }
        if slot_name:
            rec["slot_name"] = slot_name
        rz = ent.get("release_tcp_z")
        if rz is not None:
            try:
                rec["release_tcp_z"] = float(rz)
            except (TypeError, ValueError):
                pass
        out.append(rec)
    return out


def load_demo_completed_state(
    path: str,
) -> Tuple[Set[str], Set[str], List[Dict[str, Any]]]:
    p = Path(str(path or "").strip())
    if not p.is_file():
        return set(), set(), []
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return set(), set(), []
    if not isinstance(raw, dict):
        return set(), set(), []
    entities: Set[str] = set()
    labels: Set[str] = set()
    for ent in raw.get("completed_entities") or []:
        short = _entity_short(ent)
        if short:
            entities.add(short)
    for lb in raw.get("completed_labels") or []:
        norm = _label_lower(lb)
        if norm:
            labels.add(norm)
    deposits = _normalize_completed_deposits(raw.get("completed_deposits"))
    return entities, labels, deposits


def save_demo_completed_state(
    path: str,
    *,
    completed_entities: Set[str],
    completed_labels: Set[str],
    completed_deposits: Optional[List[Dict[str, Any]]] = None,
) -> bool:
    p = Path(str(path or "").strip())
    if not str(p):
        return False
    payload = {
        "completed_entities": sorted(completed_entities),
        "completed_labels": sorted(completed_labels),
        "completed_deposits": list(completed_deposits or []),
    }
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return True
    except OSError:
        return False
