#!/usr/bin/env python3
"""Batería offline de casuísticas SLM con estado de sesión (sin Ollama)."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from .command_dispatcher import dispatch_command
from .intent_schema import validate_intent_payload
from .semantic_guardrails import apply_semantic_guardrails
from .slot_state import (
    SlotOccupancy,
    apply_any_slot_resolution,
    apply_slot_occupancy_checks,
)


@dataclass
class CasuisticCase:
    case_id: str
    label: str
    text: str
    setup: Callable[[SlotOccupancy], None]
    expected_intent: str
    executable: bool
    message_contains: Optional[str] = None


def _pipeline(text: str, occupancy: SlotOccupancy) -> Dict[str, Any]:
    corrected, _, _ = apply_semantic_guardrails(text, {})
    corrected, _, _ = apply_any_slot_resolution(text, corrected, occupancy)
    corrected, _, _ = apply_slot_occupancy_checks(corrected, occupancy)
    return corrected


def _free_all(occ: SlotOccupancy) -> None:
    occ.reset()


def _occupy_all(occ: SlotOccupancy) -> None:
    for i in range(4):
        occ.mark_occupied(i, "obj_%d" % i)


CASES: List[CasuisticCase] = [
    CasuisticCase(
        "C1",
        "Hueco destino ocupado por otro objeto",
        "deja el azúcar en el tercer cajón",
        lambda o: o.mark_occupied(2, "chips_can"),
        "ask_clarification",
        False,
        "tercer cajón",
    ),
    CasuisticCase(
        "C2",
        "Cajones llenos + cualquier cajón",
        "deja la mostaza en cualquier cajón",
        _occupy_all,
        "ask_clarification",
        False,
        "No hay cajones libres",
    ),
    CasuisticCase(
        "C3",
        "Objeto ya en el cajón solicitado",
        "recoge las galletas en el primer cajón",
        lambda o: o.mark_occupied(0, "cracker_box"),
        "ask_clarification",
        False,
        "ya contiene",
    ),
    CasuisticCase(
        "C4",
        "Cualquier cajón con primer hueco libre",
        "deja el azúcar en cualquier cajón",
        lambda o: o.mark_occupied(0, "cracker_box"),
        "pick_place",
        True,
    ),
    CasuisticCase(
        "C5",
        "Objeto no soportado",
        "coge una banana",
        _free_all,
        "reject",
        False,
    ),
    CasuisticCase(
        "C6",
        "Orden ambigua",
        "ponlo allí",
        _free_all,
        "ask_clarification",
        False,
    ),
    CasuisticCase(
        "C7",
        "Hueco inexistente (quinto cajón)",
        "deja el azúcar en el quinto cajón",
        _free_all,
        "ask_clarification",
        False,
        "cuatro cajones",
    ),
    CasuisticCase(
        "C8",
        "Mesa llena simulada: sin huecos para nuevo depósito",
        "deja el azúcar en cualquier cajón",
        _occupy_all,
        "ask_clarification",
        False,
        "No hay cajones libres",
    ),
]


def run_cases() -> int:
    results: List[Dict[str, Any]] = []
    failed = 0

    for case in CASES:
        occ = SlotOccupancy()
        case.setup(occ)
        intent = _pipeline(case.text, occ)
        ok_schema, _, schema_err = validate_intent_payload(intent)
        action = dispatch_command(intent)
        intent_name = str(intent.get("intent") or "")
        question = str((intent.get("safety") or {}).get("clarification_question") or "")

        intent_ok = intent_name == case.expected_intent
        exec_ok = action.execution_supported == case.executable
        msg_ok = True
        if case.message_contains:
            msg_ok = case.message_contains.lower() in question.lower()
        schema_ok = ok_schema
        passed = intent_ok and exec_ok and msg_ok and schema_ok

        row = {
            "id": case.case_id,
            "label": case.label,
            "text": case.text,
            "expected_intent": case.expected_intent,
            "got_intent": intent_name,
            "executable_expected": case.executable,
            "executable_got": action.execution_supported,
            "schema_ok": schema_ok,
            "passed": passed,
            "clarification": question,
        }
        results.append(row)
        if not passed:
            failed += 1
            print("FAIL %s %s" % (case.case_id, case.label), file=sys.stderr)
            if not intent_ok:
                print("  intent: expected %s got %s" % (case.expected_intent, intent_name), file=sys.stderr)
            if not exec_ok:
                print("  executable: expected %s got %s" % (case.executable, action.execution_supported), file=sys.stderr)
            if not msg_ok:
                print("  message missing: %r in %r" % (case.message_contains, question), file=sys.stderr)
            if not schema_ok:
                print("  schema: %s" % schema_err, file=sys.stderr)

    print(json.dumps(results, ensure_ascii=False, indent=2))
    print("\nResumen: %d/%d OK" % (len(CASES) - failed, len(CASES)))
    return 1 if failed else 0


def main() -> None:
    raise SystemExit(run_cases())


if __name__ == "__main__":
    main()
