"""Política de orden y selección para demo clear_table (manual y automático)."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from panda_controller.demo_object_order_policy import (
    _label_completed,
    _label_lower,
    collect_present_table_labels,
    parse_demo_object_priority_order,
)
from panda_controller.demo_scene_policy import (
    load_demo_scene_policy,
    resolve_pick_order_from_scene_policy,
)
from panda_vision.spawn.demo_scene_presets import (
    DEMO_SCENE_02_REMAINING_SUGAR_MUSTARD_PICK_ORDER,
    DEMO_SCENE_02_REMAINING_SUGAR_MUSTARD_SCENE_ID,
)

DEFAULT_DEMO_SCENE_02_PICK_ORDER: Tuple[str, ...] = (
    "cracker_box",
    "chips_can",
    "sugar_box",
    "mustard_bottle",
)

# Orden congelado para grabación demo_scene_02 clear_table.
DEMO_SCENE_02_FROZEN_PICK_ORDER: Tuple[str, ...] = DEFAULT_DEMO_SCENE_02_PICK_ORDER


def parse_scene_pick_order(raw: Any, *, fallback: Sequence[str]) -> List[str]:
    return parse_demo_object_priority_order(raw or list(fallback))


def resolve_scene_pick_order(
    scene_id: str,
    *,
    scene_02_order: Sequence[str],
    default_order: Sequence[str] = DEFAULT_DEMO_SCENE_02_PICK_ORDER,
    scene_policy: Optional[Dict[str, Any]] = None,
) -> List[str]:
    sid = str(scene_id or "").strip().lower()
    policy = scene_policy
    if policy is None and sid:
        policy = load_demo_scene_policy(sid)
    if policy:
        order = resolve_pick_order_from_scene_policy(
            policy, fallback=DEMO_SCENE_02_FROZEN_PICK_ORDER
        )
        if order:
            return order
    if sid in ("demo_scene_02", "demo_scene_02_clear_table"):
        order = parse_scene_pick_order(scene_02_order, fallback=DEMO_SCENE_02_FROZEN_PICK_ORDER)
        return order if order else list(DEMO_SCENE_02_FROZEN_PICK_ORDER)
    if sid == DEMO_SCENE_02_REMAINING_SUGAR_MUSTARD_SCENE_ID:
        return list(DEMO_SCENE_02_REMAINING_SUGAR_MUSTARD_PICK_ORDER)
    if sid.startswith("demo_choice_scene"):
        return list(default_order)
    parsed = parse_scene_pick_order(scene_02_order, fallback=default_order)
    return parsed if parsed else list(default_order)


def discover_effective_pick_order(
    *,
    pick_order: Sequence[str],
    present_labels: Set[str],
    completed_labels: Set[str],
) -> List[str]:
    """Orden operativo: presentes en mesa, no completados, respetando pick_order."""
    out: List[str] = []
    for lb in pick_order:
        norm = _label_lower(lb)
        if not norm:
            continue
        if norm not in present_labels:
            continue
        if _label_completed(norm, completed_labels=completed_labels):
            continue
        out.append(norm)
    return out


def resolve_clear_table_manual_step_bootstrap(
    *,
    completed_labels: Set[str],
    pick_order: Sequence[str],
) -> Tuple[str, int, str]:
    """
    Primer label del pick_order no completado (sin filtrar por visibilidad).
    Devuelve (selected_label, order_index, reason).
    """
    for idx, lb in enumerate(pick_order):
        norm = _label_lower(lb)
        if not norm:
            continue
        if _label_completed(norm, completed_labels=completed_labels):
            continue
        return norm, idx, "pick_order"
    return "", -1, "all_completed_or_empty"


def clear_table_manual_step_accepts_payload_target(
    *,
    selected_label: str,
    payload_label: str,
    manual_step_active: bool,
) -> bool:
    """En manual step, solo acepta target_candidate si coincide con selected_label."""
    if not manual_step_active:
        return True
    sel = _label_lower(selected_label)
    pay = _label_lower(payload_label)
    if not sel:
        return False
    return bool(pay) and pay == sel


def resolve_clear_table_manual_target(
    requested_label: str,
    *,
    present_labels: Set[str],
    completed_labels: Set[str],
    pick_order: Sequence[str],
) -> Tuple[str, str, List[str]]:
    """
    Selecciona el target para un paso manual de clear_table.

    Devuelve (selected_label, reason, skipped_completed_labels).
    reason: direct_request | manual_step_next | advance_past_completed_request |
            all_completed_or_empty | requested_not_present_fallback
    """
    effective_order = discover_effective_pick_order(
        pick_order=pick_order,
        present_labels=present_labels,
        completed_labels=completed_labels,
    )
    if not effective_order:
        return "", "all_completed_or_empty", []

    requested = _label_lower(requested_label)
    skipped: List[str] = []

    if requested in ("clear_table", "clear_table_manual", "all", "*"):
        requested = ""

    if requested and _label_completed(requested, completed_labels=completed_labels):
        skipped.append(requested)
        return effective_order[0], "advance_past_completed_request", skipped

    if requested and requested in present_labels:
        return requested, "direct_request", skipped

    if requested and requested not in present_labels:
        return effective_order[0], "requested_not_present_fallback", skipped

    return effective_order[0], "manual_step_next", skipped


def remaining_table_labels(
    objects: Sequence[dict],
    *,
    completed_entities: Set[str],
    completed_labels: Set[str],
) -> Set[str]:
    return collect_present_table_labels(
        objects,
        completed_entities=completed_entities,
        completed_labels=completed_labels,
    )
