"""Estado de ocupación de slots y resolución de «cualquier cajón/slot/hueco»."""

from __future__ import annotations

import re
import unicodedata
from typing import Any, Dict, List, Optional, Tuple

from .semantic_guardrails import resolve_slot_from_text, resolve_target_from_text

_NO_FREE_SLOTS_MSG = "No hay cajones libres disponibles."
_HUMAN_SLOT_NAMES: Tuple[str, ...] = (
    "primer cajón",
    "segundo cajón",
    "tercer cajón",
    "cuarto cajón",
)


def _strip_accents(s: str) -> str:
    return "".join(
        ch for ch in unicodedata.normalize("NFD", s) if unicodedata.category(ch) != "Mn"
    )


def _norm_text(text: str) -> str:
    return _strip_accents(text or "").lower()


def slot_index_to_human(slot_index: int) -> str:
    """Índice 0-based → nombre amigable (primer cajón, …)."""
    if 0 <= slot_index < len(_HUMAN_SLOT_NAMES):
        return _HUMAN_SLOT_NAMES[slot_index]
    return "cajón %d" % (slot_index + 1)


_ANY_SLOT_PATTERNS: Tuple[str, ...] = (
    r"\bcualquier\s+(slot|hueco|espacio|cajon|cajones)\b",
    r"\bdonde\s+sea\b",
    r"\bdonde\s+puedas\b",
    r"\ben\s+algun\s+(slot|hueco|cajon)\b",
    r"\ben\s+algun\s+hueco\b",
    r"\ben\s+un\s+(slot|hueco|cajon)\s+libre\b",
    r"\ben\s+un\s+cajon\s+libre\b",
)


def is_any_slot_request(text: str) -> bool:
    """True si el usuario pide un slot/hueco/cajón no concreto (primer libre)."""
    t = _norm_text(text)
    for pattern in _ANY_SLOT_PATTERNS:
        if re.search(pattern, t):
            return True
    return False


class SlotOccupancy:
    """Ocupación simulada de slots 0..num_slots-1 para sesión chat/web."""

    def __init__(self, num_slots: int = 4) -> None:
        self.num_slots = num_slots
        self.occupied: Dict[int, Optional[str]] = {
            i: None for i in range(num_slots)
        }

    def is_occupied(self, slot_index: int) -> bool:
        return self.get_occupant(slot_index) is not None

    def get_occupant(self, slot_index: int) -> Optional[str]:
        if 0 <= slot_index < self.num_slots:
            return self.occupied.get(slot_index)
        return None

    def first_free_slot(self) -> Optional[int]:
        for i in range(self.num_slots):
            if not self.is_occupied(i):
                return i
        return None

    def can_place(self, slot_index: int, target_label: str) -> Tuple[bool, str]:
        """Comprueba si target_label puede colocarse en slot_index.

        Returns:
            (True, "") si el slot está libre.
            (False, "already_there") si ya contiene el mismo objeto.
            (False, "slot_occupied") si otro objeto lo ocupa.
            (False, "invalid_slot") si el índice no es válido.
        """
        if not 0 <= slot_index < self.num_slots:
            return False, "invalid_slot"
        occupant = self.get_occupant(slot_index)
        if occupant is None:
            return True, ""
        if occupant == target_label:
            return False, "already_there"
        return False, "slot_occupied"

    def mark_occupied(
        self, slot_index: int, target_label: str, overwrite: bool = False
    ) -> bool:
        """Marca el slot ocupado. No sobrescribe por defecto."""
        if not 0 <= slot_index < self.num_slots:
            return False
        occupant = self.get_occupant(slot_index)
        if occupant is not None and occupant != target_label and not overwrite:
            return False
        self.occupied[slot_index] = target_label
        return True

    def mark_free(self, slot_index: int) -> None:
        if 0 <= slot_index < self.num_slots:
            self.occupied[slot_index] = None

    def reset(self) -> None:
        for i in range(self.num_slots):
            self.occupied[i] = None

    def as_dict(self) -> Dict[int, Optional[str]]:
        return dict(self.occupied)

    def format_status(self) -> str:
        parts: List[str] = []
        for i in range(self.num_slots):
            label = self.occupied.get(i)
            if label:
                parts.append("slot %d: %s" % (i, label))
            else:
                parts.append("slot %d: libre" % i)
        return ", ".join(parts)


def _pick_place_template(target_label: str, slot_index: int) -> Dict[str, Any]:
    return {
        "schema_version": "1.1",
        "intent": "pick_place",
        "target_label": target_label,
        "target_selector": {"type": "single"},
        "destination": {
            "type": "slot",
            "slot_index": slot_index,
            "slot_order": None,
        },
        "execution": {"dry_run": True, "require_confirmation": True},
        "safety": {
            "requires_clarification": False,
            "clarification_question": "",
            "reject_reason": "",
        },
    }


def _ask_clarification_template(question: str) -> Dict[str, Any]:
    return {
        "schema_version": "1.1",
        "intent": "ask_clarification",
        "target_label": None,
        "target_selector": {"type": None},
        "destination": {"type": None, "slot_index": None, "slot_order": None},
        "execution": {"dry_run": True, "require_confirmation": True},
        "safety": {
            "requires_clarification": True,
            "clarification_question": question,
            "reject_reason": "",
        },
    }


def _no_free_slots_template() -> Dict[str, Any]:
    return _ask_clarification_template(_NO_FREE_SLOTS_MSG)


def _already_there_message(slot_index: int, target_label: str) -> str:
    return (
        "El %s ya contiene %s; no hace falta moverlo de nuevo."
        % (slot_index_to_human(slot_index), target_label)
    )


def _slot_occupied_message(slot_index: int, occupant: str) -> str:
    return (
        "El %s ya está ocupado por %s. Elige otro cajón o usa 'cualquier cajón'."
        % (slot_index_to_human(slot_index), occupant)
    )


def apply_slot_occupancy_checks(
    intent: Dict[str, Any],
    slot_occupancy: Optional[SlotOccupancy] = None,
) -> Tuple[Dict[str, Any], bool, List[str]]:
    """Bloquea pick_place si el slot destino está ocupado por otro objeto."""
    if intent.get("intent") != "pick_place":
        return intent, False, []

    target = intent.get("target_label")
    dest = intent.get("destination") or {}
    slot = dest.get("slot_index")
    if not target or slot is None:
        return intent, False, []

    occupancy = slot_occupancy if slot_occupancy is not None else SlotOccupancy()
    slot_i = int(slot)
    target_s = str(target)

    ok, reason = occupancy.can_place(slot_i, target_s)
    if ok:
        return intent, False, []

    if reason == "already_there":
        msg = _already_there_message(slot_i, target_s)
        tag = "slot_occupancy_already_there"
    elif reason == "slot_occupied":
        occupant = occupancy.get_occupant(slot_i) or "otro objeto"
        msg = _slot_occupied_message(slot_i, occupant)
        tag = "slot_occupancy_slot_occupied"
    else:
        return intent, False, []

    return _ask_clarification_template(msg), True, [tag]


def apply_deposit_full_checks(
    intent: Dict[str, Any],
    slot_occupancy: Optional[SlotOccupancy] = None,
) -> Tuple[Dict[str, Any], bool, List[str]]:
    """Bloquea pick_place si los cuatro cajones están ocupados."""
    if intent.get("intent") != "pick_place":
        return intent, False, []

    occupancy = slot_occupancy if slot_occupancy is not None else SlotOccupancy()
    if occupancy.first_free_slot() is not None:
        return intent, False, []

    from .deposit_scene_loader import deposit_box_full_message

    return (
        _ask_clarification_template(deposit_box_full_message()),
        True,
        ["deposit_box_full"],
    )


def apply_any_slot_resolution(
    text: str,
    intent: Dict[str, Any],
    slot_occupancy: Optional[SlotOccupancy] = None,
) -> Tuple[Dict[str, Any], bool, List[str]]:
    """Concreta «cualquier cajón/slot» al primer slot libre según estado de sesión."""
    if not is_any_slot_request(text):
        return intent, False, []

    if resolve_slot_from_text(text) is not None:
        return intent, False, []

    target = resolve_target_from_text(text)
    if target is None:
        return intent, False, []

    occupancy = slot_occupancy if slot_occupancy is not None else SlotOccupancy()
    free = occupancy.first_free_slot()
    if free is None:
        return (
            _no_free_slots_template(),
            True,
            ["any_slot_no_free_slots"],
        )

    ok, reason = occupancy.can_place(free, target)
    if not ok:
        if reason == "slot_occupied":
            return (
                _no_free_slots_template(),
                True,
                ["any_slot_inconsistent_free_slot"],
            )
        return (
            _no_free_slots_template(),
            True,
            ["any_slot_no_free_slots"],
        )

    corrected = _pick_place_template(target, free)
    return (
        corrected,
        True,
        ["resolved_any_slot -> slot_index=%d target=%s" % (free, target)],
    )
