#!/usr/bin/env python3
"""Comprobación offline de guardrails y slot state (sin Ollama)."""

from __future__ import annotations

from typing import Any, Dict, Optional

from .command_dispatcher import dispatch_command
from .intent_schema import validate_intent_payload
from .semantic_guardrails import (
    apply_semantic_guardrails,
    is_clear_table_text,
    resolve_slot_from_text,
    resolve_target_from_text,
)
from .slot_state import (
    SlotOccupancy,
    apply_any_slot_resolution,
    apply_slot_occupancy_checks,
    is_any_slot_request,
)


def _assert(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


def _pipeline(
    text: str,
    occupancy: SlotOccupancy,
    raw: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    corrected, _, _ = apply_semantic_guardrails(text, raw or {})
    corrected, _, _ = apply_any_slot_resolution(text, corrected, occupancy)
    corrected, _, _ = apply_slot_occupancy_checks(corrected, occupancy)
    return corrected


def main() -> None:
    _assert(resolve_slot_from_text("tercer hueco") == 2, "tercer hueco -> 2")
    _assert(resolve_slot_from_text("tercer espacio") == 2, "tercer espacio -> 2")
    _assert(resolve_slot_from_text("tercer cajón") == 2, "tercer cajón -> 2")
    _assert(resolve_slot_from_text("primer cajon") == 0, "primer cajon -> 0")
    _assert(resolve_slot_from_text("cajon 2") == 1, "cajon 2 (humano) -> 1")
    _assert(resolve_slot_from_text("cajon 3") == 2, "cajon 3 (humano) -> 2")
    _assert(resolve_slot_from_text("2 cajon") == 1, "2 cajon (humano) -> 1")
    _assert(resolve_slot_from_text("slot 3") == 3, "slot 3 -> 3")
    _assert(
        resolve_target_from_text("recoge chips can al segundo espacio") == "chips_can",
        "chips_can",
    )
    _assert(
        resolve_target_from_text("recoge el bote de mostaza al segundo espacio")
        == "mustard_bottle",
        "mustard_bottle",
    )
    _assert(is_clear_table_text("recógeme toda la mesa"), "clear_table text")
    _assert(is_clear_table_text("recoge la mesa"), "recoge la mesa")
    _assert(
        not is_clear_table_text("recoge chips can al segundo espacio"),
        "not clear_table",
    )

    occ = SlotOccupancy()
    ok, reason = occ.can_place(0, "cracker_box")
    _assert(ok and reason == "", "free slot can_place")
    occ.mark_occupied(0, "cracker_box")
    ok, reason = occ.can_place(0, "sugar_box")
    _assert(not ok and reason == "slot_occupied", "occupied conflict")
    ok, reason = occ.can_place(0, "cracker_box")
    _assert(not ok and reason == "already_there", "already there")
    occ.mark_occupied(2, "chips_can")
    _assert(
        not occ.mark_occupied(2, "sugar_box")
        and occ.get_occupant(2) == "chips_can",
        "no overwrite",
    )

    corrected, applied, _ = apply_semantic_guardrails("ponlo allí", {})
    _assert(applied and corrected["intent"] == "ask_clarification", "ponlo allí")
    ok, _, _ = validate_intent_payload(corrected)
    _assert(ok, "ask_clarification valid")

    corrected, applied, _ = apply_semantic_guardrails(
        "coge la caja de galletas y déjala en el tercer hueco",
        {
            "schema_version": "1.1",
            "intent": "pick_place",
            "target_label": "cracker_box",
            "target_selector": {"type": "single"},
            "destination": {"type": "slot", "slot_index": 3, "slot_order": None},
            "execution": {"dry_run": True, "require_confirmation": True},
            "safety": {
                "requires_clarification": False,
                "clarification_question": "",
                "reject_reason": "",
            },
        },
    )
    _assert(corrected["destination"]["slot_index"] == 2, "slot 3->2")
    ok, _, _ = validate_intent_payload(corrected)
    _assert(ok, "pick_place valid after slot fix")

    corrected, applied, _ = apply_semantic_guardrails("coge el bote amarillo", {})
    _assert(corrected["intent"] == "reject", "bote amarillo reject")

    corrected, applied, _ = apply_semantic_guardrails(
        "deja el azúcar en el tercer cajón", {}
    )
    _assert(applied and corrected["intent"] == "pick_place", "azúcar tercer cajón")
    _assert(corrected["target_label"] == "sugar_box", "sugar_box")
    _assert(corrected["destination"]["slot_index"] == 2, "slot 2")
    ok, _, _ = validate_intent_payload(corrected)
    _assert(ok, "azúcar tercer cajón valid")

    corrected, applied, _ = apply_semantic_guardrails(
        "recoge las galletas en el primer cajón", {}
    )
    _assert(applied and corrected["intent"] == "pick_place", "galletas primer cajón")
    _assert(corrected["target_label"] == "cracker_box", "cracker_box")
    _assert(corrected["destination"]["slot_index"] == 0, "slot 0")
    ok, _, _ = validate_intent_payload(corrected)
    _assert(ok, "galletas primer cajón valid")

    _assert(is_any_slot_request("deja las galletas en cualquier cajón"), "any slot")
    occupancy = SlotOccupancy()
    corrected, applied, reasons = apply_any_slot_resolution(
        "deja las galletas en cualquier cajón", {}, occupancy
    )
    _assert(applied, "any slot applied")
    _assert(corrected["destination"]["slot_index"] == 0, "first free -> 0")
    _assert(corrected["target_label"] == "cracker_box", "cracker any")
    ok, _, _ = validate_intent_payload(corrected)
    _assert(ok, "any slot pick valid")
    occupancy.mark_occupied(0, "cracker_box")

    corrected, applied, _ = apply_any_slot_resolution(
        "deja el azúcar en cualquier cajón", {}, occupancy
    )
    _assert(applied and corrected["destination"]["slot_index"] == 1, "second free -> 1")
    _assert(corrected["target_label"] == "sugar_box", "sugar any")
    ok, _, _ = validate_intent_payload(corrected)
    _assert(ok, "sugar any valid")

    for i in range(4):
        occupancy.mark_occupied(i, "obj_%d" % i)
    corrected, applied, reasons = apply_any_slot_resolution(
        "deja el azúcar en cualquier cajón", {}, occupancy
    )
    _assert(applied and corrected["intent"] == "ask_clarification", "all full -> clarify")
    _assert(
        "No hay cajones libres" in corrected["safety"]["clarification_question"],
        "no free msg",
    )
    _assert("any_slot_no_free" in reasons[0], "reason tag")

    # Caso A: primer cajón + cualquier cajón con estado
    occ_a = SlotOccupancy()
    c = _pipeline("deja las galletas en el primer cajón", occ_a)
    _assert(c["intent"] == "pick_place" and c["destination"]["slot_index"] == 0, "A1")
    occ_a.mark_occupied(0, "cracker_box")
    c = _pipeline("deja el azúcar en cualquier cajón", occ_a)
    _assert(c["intent"] == "pick_place" and c["destination"]["slot_index"] == 1, "A2")
    action = dispatch_command(c)
    _assert(action.execution_supported, "A2 executable")

    # Caso B: slot explícito ocupado por otro
    occ_b = SlotOccupancy()
    occ_b.mark_occupied(2, "chips_can")
    c = _pipeline("deja el azúcar en el tercer cajón", occ_b)
    _assert(c["intent"] == "ask_clarification", "B intent")
    q = c["safety"]["clarification_question"]
    _assert("tercer cajón" in q and "chips_can" in q, "B message")
    _assert(occ_b.get_occupant(2) == "chips_can", "B no overwrite")
    action = dispatch_command(c)
    _assert(not action.execution_supported, "B not executable")

    # Caso C: ya está en el cajón
    occ_c = SlotOccupancy()
    occ_c.mark_occupied(0, "cracker_box")
    c = _pipeline("recoge las galletas en el primer cajón", occ_c)
    _assert(c["intent"] == "ask_clarification", "C intent")
    _assert("ya contiene" in c["safety"]["clarification_question"], "C already there")
    action = dispatch_command(c)
    _assert(not action.execution_supported, "C not executable")

    # Caso D: todos ocupados, mustard cualquier cajón
    occ_d = SlotOccupancy()
    for i in range(4):
        occ_d.mark_occupied(i, "obj_%d" % i)
    c = _pipeline("deja la mostaza en cualquier cajón", occ_d)
    _assert(c["intent"] == "ask_clarification", "D intent")
    _assert("No hay cajones libres" in c["safety"]["clarification_question"], "D msg")
    action = dispatch_command(c)
    _assert(not action.execution_supported, "D not executable")

    print("OK guardrails_self_check")


if __name__ == "__main__":
    main()
