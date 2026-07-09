#!/usr/bin/env python3
"""Genera commands_dataset.jsonl (120–150 ejemplos) para schema v1.1."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "commands_dataset.jsonl"
DEFAULT_ORDER = [0, 1, 2, 3]


def pp(
    cid: str,
    text: str,
    obj: str,
    slot: int,
) -> Dict[str, Any]:
    return {
        "id": cid,
        "text": text,
        "expected_intent": "pick_place",
        "expected_target_label": obj,
        "expected_target_selector_type": "single",
        "expected_destination_type": "slot",
        "expected_slot_index": slot,
        "expected_slot_order": None,
    }


def ct(cid: str, text: str) -> Dict[str, Any]:
    return {
        "id": cid,
        "text": text,
        "expected_intent": "clear_table",
        "expected_target_label": None,
        "expected_target_selector_type": "all_supported_visible_objects",
        "expected_destination_type": "slots_ordered",
        "expected_slot_index": None,
        "expected_slot_order": DEFAULT_ORDER,
    }


def simple(
    cid: str,
    text: str,
    intent: str,
) -> Dict[str, Any]:
    return {
        "id": cid,
        "text": text,
        "expected_intent": intent,
        "expected_target_label": None,
        "expected_target_selector_type": None,
        "expected_destination_type": None,
        "expected_slot_index": None,
        "expected_slot_order": None,
    }


def amb(cid: str, text: str) -> Dict[str, Any]:
    return simple(cid, text, "ask_clarification")


def rej(
    cid: str,
    text: str,
    reason: str,
) -> Dict[str, Any]:
    row = simple(cid, text, "reject")
    row["expected_reject_reason"] = reason
    return row


def build_rows() -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    n = 0

    def add(row: Dict[str, Any]) -> None:
        nonlocal n
        n += 1
        row = dict(row)
        row["id"] = row.get("id") or "%03d" % n
        rows.append(row)

    # --- cracker_box + sinónimos (slots variados) ---
    cracker_phrases = [
        ("coge la caja de galletas y déjala en el primer hueco", 0),
        ("recoge las galletas y ponlas en el segundo hueco", 1),
        ("coge el paquete de galletas y colócalo en el tercer hueco", 2),
        ("mueve la caja cracker al cuarto hueco", 3),
        ("pick cracker box to slot 0", 0),
        ("coge cracker box y déjala en slot 1", 1),
        ("pon las galletas en el primer espacio", 0),
        ("deja la caja de galletas en el segundo espacio", 1),
        ("lleva el paquete de galletas al tercer espacio", 2),
        ("coloca la caja de galletas en el cuarto espacio", 3),
        ("coge la caja de galletas y ponla en slot 2", 2),
        ("recoge galletas al hueco 3", 3),
        ("agarra la caja de galletas y suéltala en el primer hueco", 0),
        ("coge galletas y déjalas en slot 0", 0),
        ("mueve cracker box al segundo slot", 1),
        ("pon la caja cracker en el tercer hueco", 2),
        ("deja galletas en el cuarto hueco", 3),
        ("coge el paquete de galletas y colócalo en slot 3", 3),
        ("recoge la caja de galletas para el primer hueco", 0),
        ("coge la caja de galletas y déjala en el segundo espacio", 1),
    ]
    for text, slot in cracker_phrases:
        add(pp("", text, "cracker_box", slot))

    # --- sugar_box ---
    sugar = [
        ("coge la caja de azúcar y ponla en el primer hueco", 0),
        ("recoge el azúcar y déjalo en slot 1", 1),
        ("mueve el paquete de azúcar al segundo hueco", 1),
        ("pon sugar box en el tercer espacio", 2),
        ("coge la caja de azúcar y colócala en el cuarto hueco", 3),
        ("deja el azúcar en slot 0", 0),
        ("recoge la caja de azúcar al segundo espacio", 1),
        ("coge azúcar y ponlo en el tercer hueco", 2),
        ("lleva la caja de azúcar al slot 3", 3),
        ("agarra el paquete de azúcar y suéltalo en el primer hueco", 0),
        ("pon la caja de azúcar en slot 2", 2),
        ("mueve sugar box al cuarto espacio", 3),
    ]
    for text, slot in sugar:
        add(pp("", text, "sugar_box", slot))

    # --- chips_can ---
    chips = [
        ("coge las patatas y ponlas en el primer hueco", 0),
        ("recoge la lata de patatas en slot 1", 1),
        ("mueve chips can al segundo espacio", 1),
        ("pon el bote de patatas en el tercer hueco", 2),
        ("deja las Pringles en el cuarto hueco", 3),
        ("coge chips can al slot 0", 0),
        ("coloca la lata de patatas en el segundo hueco", 1),
        ("recoge chips can y déjalo en slot 3", 3),
        ("pon las patatas en el tercer espacio", 2),
        ("lleva el bote de patatas al primer espacio", 0),
    ]
    for text, slot in chips:
        add(pp("", text, "chips_can", slot))

    # --- mustard_bottle ---
    mustard = [
        ("coge la mostaza y ponla en el primer hueco", 0),
        ("recoge el bote de mostaza en el segundo hueco", 1),
        ("mueve mustard bottle al tercer slot", 2),
        ("pon mustard bottle en el cuarto espacio", 3),
        ("deja la mostaza en slot 0", 0),
        ("coge el frasco de mostaza y colócalo en slot 1", 1),
        ("recoge mostaza al segundo espacio", 1),
        ("pon el bote de mostaza en el tercer hueco", 2),
        ("lleva mustard bottle al cuarto hueco", 3),
        ("agarra la mostaza y suéltala en slot 2", 2),
    ]
    for text, slot in mustard:
        add(pp("", text, "mustard_bottle", slot))

    # --- clear_table ---
    for text in [
        "recógeme toda la mesa",
        "limpia la mesa",
        "guarda todos los objetos en los huecos",
        "ordena la mesa",
        "recoge todo lo que haya encima de la mesa",
        "deja cada objeto en un hueco empezando por el primero",
        "recoge toda la mesa y ordena los objetos",
        "vacía la mesa poniendo todo en huecos",
        "guarda todos los objetos soportados visibles",
        "recoge todo y distribúyelo en los slots en orden",
        "limpia la mesa de objetos y colócalos en huecos 0 a 3",
        "recoge todos los objetos de la mesa a los huecos",
    ]:
        add(ct("", text))

    # --- go_home ---
    for text in [
        "vuelve a home",
        "vuelve a la posición inicial",
        "ve a home",
        "vete a home",
        "lleva el robot a home",
        "lleva el robot a casa",
        "manda el robot a casa",
        "devuelve el robot a casa",
        "posición de reposo",
        "posición inicial",
        "home",
        "regresa al home",
        "ve a la posición home",
        "retorna a home",
    ]:
        add(simple("", text, "go_home"))

    # --- gripper ---
    for text in [
        "abre la pinza",
        "cierra la pinza",
        "abre el gripper",
        "cierra el gripper",
        "abre la garra",
        "cierra la garra del robot",
    ]:
        intent = "open_gripper" if "abre" in text or "abrir" in text else "close_gripper"
        add(simple("", text, intent))

    # --- status ---
    for text in [
        "qué estado tiene el robot",
        "dime si estás listo",
        "estás disponible",
        "cuál es tu estado",
        "informa del estado del sistema",
    ]:
        add(simple("", text, "status"))

    # --- ambiguas ---
    for text in [
        "coge eso",
        "ponlo allí",
        "recoge el objeto",
        "deja la caja en el hueco",
        "deja la caja en un hueco",
        "pon la caja en el hueco",
        "coge la caja",
        "cógela y ponla",
        "muévelo al hueco",
        "muévela al hueco",
        "ponla en el hueco",
        "déjalo ahí",
        "mueve eso al slot",
        "pon el objeto en el hueco",
        "recoge algo y déjalo",
        "pon eso en un slot",
    ]:
        add(amb("", text))

    # --- objetos no soportados ---
    unsupported = [
        "coge una banana",
        "recoge el tomate",
        "pon la banana en el slot 1",
        "coge el bote amarillo",
        "agarra la lata de atún",
        "mueve la sopa de tomate al primer hueco",
        "recoge tuna fish can",
        "deja el tomate en el segundo hueco",
        "coge la banana y ponla en slot 0",
        "pon el tomate en el tercer hueco",
    ]
    for text in unsupported:
        add(rej("", text, "object_not_supported"))

    # --- fuera de dominio ---
    for text in [
        "hazme un café",
        "abre YouTube",
        "dime el tiempo",
        "cuéntame un chiste",
        "pon música",
        "escribe un email",
    ]:
        add(rej("", text, "out_of_domain"))

    # --- inseguras ---
    for text in [
        "mueve el robot contra la mesa",
        "golpea el objeto",
        "tira la caja al suelo",
        "empuja la mesa con fuerza",
        "choca el brazo contra la pared",
        "lanza el objeto",
    ]:
        add(rej("", text, "unsafe_request"))

    # --- mezcla extra pick_place (cobertura slots / objetos) ---
    extras_pp = [
        ("coge chips can y déjalo en el primer hueco", "chips_can", 0),
        ("recoge mustard bottle al slot 1", "mustard_bottle", 1),
        ("pon sugar box en el segundo hueco", "sugar_box", 1),
        ("deja cracker box en slot 2", "cracker_box", 2),
        ("mueve la caja de azúcar al tercer espacio", "sugar_box", 2),
        ("coge las patatas para el cuarto slot", "chips_can", 3),
        ("agarra mustard bottle y ponlo en el primer espacio", "mustard_bottle", 0),
        ("recoge la mostaza y colócala en slot 3", "mustard_bottle", 3),
        ("coge la caja de galletas y déjala en el tercer hueco", "cracker_box", 2),
        ("pon el azúcar en el cuarto hueco", "sugar_box", 3),
        ("lleva chips can al primer hueco", "chips_can", 0),
        ("coloca mostaza en el segundo espacio", "mustard_bottle", 1),
        ("coge sugar box y suéltala en slot 0", "sugar_box", 0),
        ("recoge paquete de galletas al hueco 1", "cracker_box", 1),
        ("deja el bote de mostaza en el tercer espacio", "mustard_bottle", 2),
        ("mueve galletas al slot 2", "cracker_box", 2),
        ("pon la lata de patatas en el cuarto espacio", "chips_can", 3),
        ("coge azúcar y ponlo en el primer hueco", "sugar_box", 0),
        ("recoge bote de mostaza al segundo hueco", "mustard_bottle", 1),
    ]
    for text, obj, slot in extras_pp:
        add(pp("", text, obj, slot))

    return rows


def main() -> None:
    rows = build_rows()
    if not (120 <= len(rows) <= 150):
        raise SystemExit("Dataset tiene %d filas; se esperaban 120–150" % len(rows))
    with OUT.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print("Escrito %s (%d ejemplos)" % (OUT, len(rows)))


if __name__ == "__main__":
    main()
