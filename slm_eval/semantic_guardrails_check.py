"""Comprobación rápida de semantic_guardrails.py (no usa pytest).

Ejecutar:
  python3 semantic_guardrails_check.py
"""

from __future__ import annotations

import json

from intent_schema import validate_intent_payload
from semantic_guardrails import (
    apply_semantic_guardrails,
    is_clear_table_text,
    resolve_slot_from_text,
    resolve_target_from_text,
)


def _assert(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


def main() -> None:
    # Slots
    _assert(resolve_slot_from_text("tercer hueco") == 2, "tercer hueco -> 2")
    _assert(resolve_slot_from_text("tercer espacio") == 2, "tercer espacio -> 2")
    _assert(resolve_slot_from_text("slot 3") == 3, "slot 3 -> 3")
    _assert(resolve_slot_from_text("hueco 0") == 0, "hueco 0 -> 0")

    # Objetos
    _assert(resolve_target_from_text("chips can al segundo espacio") == "chips_can", "chips can -> chips_can")
    _assert(
        resolve_target_from_text("bote de mostaza al segundo espacio") == "mustard_bottle",
        "bote de mostaza -> mustard_bottle",
    )

    # clear_table
    _assert(is_clear_table_text("recógeme toda la mesa") is True, "toda la mesa -> clear_table")
    _assert(
        is_clear_table_text("recoge chips can al segundo espacio") is False,
        "chips can + segundo espacio -> no clear_table",
    )

    # Aplicación: ask_clarification
    parsed = {
        "schema_version": "1.1",
        "intent": "pick_place",
        "target_label": None,
        "target_selector": {"type": None},
        "destination": {"type": None, "slot_index": None, "slot_order": None},
        "execution": {"dry_run": True, "require_confirmation": True},
        "safety": {"requires_clarification": False, "clarification_question": "", "reject_reason": ""},
    }
    corrected, applied, reasons = apply_semantic_guardrails("ponlo allí", parsed)
    _assert(applied is True, "guardrails deben aplicar en ponlo allí")
    _assert(corrected["intent"] == "ask_clarification", "ponlo allí -> ask_clarification")
    ok, intent_model, _ = validate_intent_payload(corrected)
    _assert(ok, "ask_clarification corrected debe validar schema")

    # Aplicación: pick_place forzado target+slot
    corrected, applied, reasons = apply_semantic_guardrails("deja el azúcar en slot 0", parsed)
    _assert(applied is True, "guardrails deben aplicar en azúcar + slot 0")
    _assert(corrected["intent"] == "pick_place", "deja el azúcar en slot 0 -> pick_place")
    _assert(corrected["target_label"] == "sugar_box", "target_label sugar_box")
    _assert(corrected["destination"]["slot_index"] == 0, "slot_index 0")
    ok, intent_model, _ = validate_intent_payload(corrected)
    _assert(ok, "pick_place corrected debe validar schema")

    # Aplicación: reject objetos no soportados
    corrected, applied, reasons = apply_semantic_guardrails("coge el bote amarillo", parsed)
    _assert(applied is True, "guardrails deben aplicar en objeto no soportado")
    _assert(corrected["intent"] == "reject", "bote amarillo -> reject")
    _assert(corrected["safety"]["reject_reason"] == "object_not_supported", "reject_reason object_not_supported")
    ok, intent_model, _ = validate_intent_payload(corrected)
    _assert(ok, "reject corrected debe validar schema")

    print("OK semantic_guardrails_check")


if __name__ == "__main__":
    main()

