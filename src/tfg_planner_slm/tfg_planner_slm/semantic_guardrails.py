"""Guardrails semánticos deterministas para intents v1.1.

Pipeline (arquitectura objetivo):
SLM output (texto) -> JSON parser -> schema validation -> guardrails deterministas -> final validated intent.

Estas funciones SOLO aplican reglas cerradas y seguras del dominio:
- resolución determinista de slots (primer/segundo/tercero/cuarto, slot/hueco/espacio/cajón N)
- detección determinista de objetos soportados (cracker/sugar/chips/mustard)
- detección determinista de clear_table (toda la mesa / todos los objetos / mesa)
- desambiguación determinista: pronombres/genéricos -> ask_clarification
- seguridad: objetos no soportados / fuera de dominio / inseguro -> reject

No ejecuta ROS ni remapea nada opaco: si una regla fuerza un intent, se reconstruye
el JSON completo según el schema v1.1.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Any, Dict, List, Optional, Tuple


SUPPORTED_OBJECTS: Tuple[str, ...] = ("cracker_box", "sugar_box", "chips_can", "mustard_bottle")
DEFAULT_SLOT_ORDER: List[int] = [0, 1, 2, 3]


def _strip_accents(s: str) -> str:
    # Normaliza "azúcar" -> "azucar", etc.
    return "".join(
        ch for ch in unicodedata.normalize("NFD", s) if unicodedata.category(ch) != "Mn"
    )


def _norm_text(text: str) -> str:
    return _strip_accents(text or "").lower()


def resolve_slot_from_text(text: str) -> Optional[int]:
    """Resuelve un slot_index determinista desde el texto.

    Reglas:
    - primer/primero -> 0
    - segundo -> 1
    - tercer/tercero -> 2
    - cuarto -> 3
    - "cajon 1".."cajon 4" -> índice 0..3 (numeración humana del depósito)
    - "cajon 0" -> 0 (alias del primer hueco)
    - "slot 0"/"hueco 0"/"espacio 0" -> 0 (índice interno 0..3)
    - "slot/hueco/espacio 1/2/3" -> 1/2/3 (índice interno)
    - "primer cajon", "cajon tercero", etc. (vía ordinales)
    """
    t = _norm_text(text)

    # Cajón(es): numeración humana 1..4 → índice 0..3.
    m = re.search(r"\b(cajon|cajones)\s*([1-4])\b", t)
    if m:
        return int(m.group(2)) - 1

    m = re.search(r"\b([1-4])\s*(cajon|cajones)\b", t)
    if m:
        return int(m.group(1)) - 1

    m = re.search(r"\b([1-4])\s*(?:º|°)\s*(cajon|cajones)\b", t)
    if m:
        return int(m.group(1)) - 1

    m = re.search(r"\b(cajon|cajones)\s*(?:n(?:umero|°|º)?\.?\s*)?([1-4])\b", t)
    if m:
        return int(m.group(2)) - 1

    m = re.search(r"\bcajon\s*0\b", t)
    if m:
        return 0

    # Índice interno 0..3 (API técnica / hueco / espacio / slot).
    m = re.search(r"\bslot\s*([0-3])\b", t)
    if m:
        return int(m.group(1))

    m = re.search(r"\b(hueco|espacio)\s*([0-3])\b", t)
    if m:
        return int(m.group(2))

    m = re.search(r"\b([0-3])\s*(hueco|espacio)\b", t)
    if m:
        return int(m.group(1))

    # Luego: ordinales ("tercer hueco", "tercer cajon" -> 2)
    if re.search(r"\b(cuarto)\b", t):
        return 3
    if re.search(r"\b(tercer|tercero)\b", t):
        return 2
    if re.search(r"\b(segundo)\b", t):
        return 1
    if re.search(r"\b(primer|primero)\b", t):
        return 0
    return None


def has_invalid_slot_reference(text: str) -> bool:
    """True si el texto menciona un hueco/cajón fuera del rango 0..3."""
    t = _norm_text(text)
    if re.search(r"\b(quinto|sexto|septimo|octavo|noveno|decimo)\b", t):
        return True
    if re.search(r"\b(slot|hueco|espacio)\s*([4-9]|[1-9][0-9]+)\b", t):
        return True
    if re.search(r"\b(cajon|cajones)\s*([5-9]|[1-9][0-9]+)\b", t):
        return True
    if re.search(r"\b([5-9]|[1-9][0-9]+)\s*(cajon|cajones)\b", t):
        return True
    return False


def resolve_target_from_text(text: str) -> Optional[str]:
    """Detecta determinísticamente un target_label soportado desde texto."""
    t = _norm_text(text)

    # cracker_box
    if re.search(
        r"\b(galletas|caja\s+de\s+galletas|cracker|cracker\s+box|paquete\s+de\s+galletas)\b",
        t,
    ):
        return "cracker_box"

    # sugar_box
    if re.search(
        r"\b(azucar|caja\s+de\s+azucar|sugar\s+box|azucar(?:es)?|paquete\s+de\s+azucar)\b",
        t,
    ):
        return "sugar_box"

    # chips_can
    if re.search(
        r"\b(chips|patatas|pringles|chips\s+can|chip\s+can|lata\s+de\s+patatas|bote\s+de\s+patatas)\b",
        t,
    ):
        return "chips_can"

    # mustard_bottle
    if re.search(
        r"\b(mostaza|bote\s+de\s+mostaza|mustard|mustard\s+bottle|frasco\s+de\s+mostaza)\b",
        t,
    ):
        return "mustard_bottle"

    return None


def _contains_unsupported_object(text: str) -> Optional[str]:
    """Detecta objetos no soportados con keywords conocidas."""
    t = _norm_text(text)
    if re.search(r"\b(banana|platano|platano|tomate|sopa\s+de\s+tomate|tomato\s+soup|tuna|atun)\b", t):
        return "object_not_supported"
    if re.search(r"\b(bote\s+amarillo|bote\s+amarilla)\b", t):
        return "object_not_supported"
    return None


def _contains_out_of_domain(text: str) -> Optional[str]:
    t = _norm_text(text)
    if re.search(r"\b(cafe|cafe\s+|cafe\s+por\s+favor)\b", t):
        return "out_of_domain"
    if re.search(r"\b(you\s* ?tube|youtube)\b", t):
        return "out_of_domain"
    if re.search(r"\b(tiempo|hora\s+actual)\b", t):
        return "out_of_domain"
    if re.search(r"\b(chiste|cuentame|cuenta\s*me)\b", t):
        return "out_of_domain"
    if re.search(r"\b(email|escribe\s+un\s+email|mensaje)\b", t):
        return "out_of_domain"
    if re.search(r"\b(musica|música|pon\s+musica|pon\s+música|luz)\b", t):
        return "out_of_domain"
    return None


def _contains_unsafe_command(text: str) -> Optional[str]:
    t = _norm_text(text)
    if re.search(r"\b(contra\s+la\s+mesa|contra\s+la\s+pared|choca|golpea)\b", t):
        return "unsafe_request"
    if re.search(r"\b(empuja\s+la\s+mesa|empuja\s+la\s+mesa\s+con\s+fuerza)\b", t):
        return "unsafe_request"
    if re.search(r"\b(tira\s+al\s+suelo|lanza\s+el\s+objeto|lanza|tira)\b", t):
        return "unsafe_request"
    if re.search(r"\b(lanza\s+el\s+objeto)\b", t):
        return "unsafe_request"
    return None


def is_clear_table_text(text: str) -> bool:
    """Devuelve True solo si la frase habla de TODA la mesa / TODOS los objetos.

    Heurística segura:
    - requiere marcadores explícitos de globalidad sobre "mesa" (toda la mesa, limpiar/ordenar/vaciar la mesa, etc.)
    - nunca clasifica como clear_table si detecta un objeto soportado concreto.
    """
    t = _norm_text(text)
    if resolve_target_from_text(t) is not None:
        return False

    # Marcadores típicos del dataset:
    patterns = [
        r"\brecoge?\s+la\s+mesa\b",
        r"\brecoger\s+la\s+mesa\b",
        r"\btoda\s+la\s+mesa\b",
        r"\blimpia\s+la\s+mesa\b",
        r"\bordena\s+la\s+mesa\b",
        r"\bvaci(a|á)\s+la\s+mesa\b",
        r"\bguarda\s+todos?\s+los\s+objetos\b",
        r"\breco(g|g)e\s+todo\s+(y\s+)?distribu(y|y)e(lo)?\b",
        r"\brecoge\s+todo\s+lo\s+que\s+haya\s+encima\s+de\s+la\s+mesa\b",
        r"\brecoge\s+todos?\s+los\s+objetos\s+de\s+la\s+mesa\b",
        r"\bdeja\s+cada\s+objeto\s+en\s+un\s+hueco\b",
        r"\brecoge\s+todo\s+y\s+distribu(y|y)e\w*\s+en\s+los\s+slots\s+en\s+orden\b",
        r"\brecoge\s+toda\s+la\s+mesa\b",
    ]

    for p in patterns:
        if re.search(p, t):
            return True

    # Casos que incluyen "mesa" y "huecos" explícitamente sin nombrar objeto:
    if "mesa" in t and ("huecos" in t or "slots" in t) and (
        "todos" in t or "todo" in t or "vac" in t or "limpi" in t or "ordena" in t
    ):
        # Evita falsos positivos si hay objeto soportado (ya filtrado arriba).
        return True

    return False


def _pronoun_or_generic_reference_without_context(text: str) -> bool:
    """Detecta referencias genéricas tipo 'ponlo allí', 'cógela y ponla', etc."""
    t = _norm_text(text)
    if (
        "ponlo" in t
        or "ponla" in t
        or "dejalo" in t
        or "dejala" in t
        or "cogela" in t
        or "cogelo" in t
        or "muevelo" in t
        or "muevela" in t
        or "coge eso" in t
        or "pon eso" in t
        or "recoge el objeto" in t
        or "recoge algo" in t
        or re.search(r"\b(eso|esto)\b", t)
    ):
        return True
    return False


def apply_semantic_guardrails(
    text: str, parsed_intent: Optional[Dict[str, Any]]
) -> Tuple[Dict[str, Any], bool, List[str]]:
    """Aplica guardrails deterministas y devuelve (corrected_intent, applied, reasons).

    Si ninguna regla fuerza un cambio, se devuelve parsed_intent (o un template vacío si es None).
    """
    reasons: List[str] = []
    applied = False

    corrected: Dict[str, Any] = dict(parsed_intent or {})

    # Plantillas completas para pasar el schema v1.1
    def _template_execution() -> Dict[str, Any]:
        return {"dry_run": True, "require_confirmation": True}

    def _template_safety_clear() -> Dict[str, Any]:
        return {
            "requires_clarification": False,
            "clarification_question": "",
            "reject_reason": "",
        }

    def _template_safety_clarify() -> Dict[str, Any]:
        return {
            "requires_clarification": True,
            "clarification_question": "¿A qué objeto te refieres y en qué slot debo colocarlo?",
            "reject_reason": "",
        }

    def _template_safety_reject(reason: str) -> Dict[str, Any]:
        return {
            "requires_clarification": False,
            "clarification_question": "",
            "reject_reason": reason,
        }

    # E) Seguridad: objetos no soportados / fuera de dominio / inseguro
    unsupported = _contains_unsupported_object(text)
    if unsupported:
        applied = True
        reasons.append("force_reject unsupported_object")
        corrected = {
            "schema_version": "1.1",
            "intent": "reject",
            "target_label": None,
            "target_selector": {"type": None},
            "destination": {"type": None, "slot_index": None, "slot_order": None},
            "execution": _template_execution(),
            "safety": _template_safety_reject(unsupported),
        }
        return corrected, applied, reasons

    out = _contains_out_of_domain(text)
    if out:
        applied = True
        reasons.append("force_reject out_of_domain")
        corrected = {
            "schema_version": "1.1",
            "intent": "reject",
            "target_label": None,
            "target_selector": {"type": None},
            "destination": {"type": None, "slot_index": None, "slot_order": None},
            "execution": _template_execution(),
            "safety": _template_safety_reject(out),
        }
        return corrected, applied, reasons

    unsafe = _contains_unsafe_command(text)
    if unsafe:
        applied = True
        reasons.append("force_reject unsafe_request")
        corrected = {
            "schema_version": "1.1",
            "intent": "reject",
            "target_label": None,
            "target_selector": {"type": None},
            "destination": {"type": None, "slot_index": None, "slot_order": None},
            "execution": _template_execution(),
            "safety": _template_safety_reject(unsafe),
        }
        return corrected, applied, reasons

    if has_invalid_slot_reference(text):
        applied = True
        reasons.append("force_ask_clarification invalid_slot_reference")
        corrected = {
            "schema_version": "1.1",
            "intent": "ask_clarification",
            "target_label": None,
            "target_selector": {"type": None},
            "destination": {"type": None, "slot_index": None, "slot_order": None},
            "execution": _template_execution(),
            "safety": {
                "requires_clarification": True,
                "clarification_question": (
                    "Solo hay cuatro cajones (huecos 0 a 3). Indica un destino válido."
                ),
                "reject_reason": "",
            },
        }
        return corrected, applied, reasons

    # Resoluciones deterministas
    resolved_target = resolve_target_from_text(text)
    resolved_slot = resolve_slot_from_text(text)
    clear_table = is_clear_table_text(text)
    generic_ref = _pronoun_or_generic_reference_without_context(text)
    current_intent = corrected.get("intent")

    # B) clear_table solo para mesa completa / todos los objetos visibles.
    if clear_table:
        applied = True
        reasons.append("force_clear_table is_clear_table_text")
        corrected = {
            "schema_version": "1.1",
            "intent": "clear_table",
            "target_label": None,
            "target_selector": {"type": "all_supported_visible_objects"},
            "destination": {
                "type": "slots_ordered",
                "slot_index": None,
                "slot_order": DEFAULT_SLOT_ORDER,
            },
            "execution": _template_execution(),
            "safety": _template_safety_clear(),
        }
        return corrected, applied, reasons

    # A) target soportado + slot concreto -> pick_place (sin inventar clear_table)
    if resolved_target is not None and resolved_slot is not None:
        applied = True
        reasons.append("force_pick_place target_and_slot_present")
        corrected = {
            "schema_version": "1.1",
            "intent": "pick_place",
            "target_label": resolved_target,
            "target_selector": {"type": "single"},
            "destination": {
                "type": "slot",
                "slot_index": resolved_slot,
                "slot_order": None,
            },
            "execution": _template_execution(),
            "safety": _template_safety_clear(),
        }
        return corrected, applied, reasons

    # C) pick_place: el slot explícito del texto manda sobre el JSON del SLM.
    if resolved_slot is not None and resolved_target is not None and current_intent != "pick_place":
        applied = True
        reasons.append("force_pick_place target_and_slot_override_slm_intent")
        corrected = {
            "schema_version": "1.1",
            "intent": "pick_place",
            "target_label": resolved_target,
            "target_selector": {"type": "single"},
            "destination": {
                "type": "slot",
                "slot_index": resolved_slot,
                "slot_order": None,
            },
            "execution": _template_execution(),
            "safety": _template_safety_clear(),
        }
        return corrected, applied, reasons

    if current_intent == "pick_place" and resolved_slot is not None:
        dest = corrected.get("destination") or {}
        if dest.get("type") != "slot" or dest.get("slot_index") != resolved_slot:
            applied = True
            reasons.append(
                "corrected_slot %s -> %s reason=explicit_slot_text"
                % (dest.get("slot_index"), resolved_slot)
            )
            corrected["destination"] = {
                "type": "slot",
                "slot_index": resolved_slot,
                "slot_order": None,
            }
        if resolved_target is not None and corrected.get("target_label") != resolved_target:
            applied = True
            reasons.append(
                "corrected_target %s -> %s reason=explicit_target_text"
                % (corrected.get("target_label"), resolved_target)
            )
            corrected["target_label"] = resolved_target
            corrected["target_selector"] = {"type": "single"}

    # D) Referencias genéricas / pronombres sin objeto soportado -> ask_clarification.
    if generic_ref and resolved_target is None:
        applied = True
        reasons.append("force_ask_clarification pronoun_without_context")
        corrected = {
            "schema_version": "1.1",
            "intent": "ask_clarification",
            "target_label": None,
            "target_selector": {"type": None},
            "destination": {"type": None, "slot_index": None, "slot_order": None},
            "execution": _template_execution(),
            "safety": _template_safety_clarify(),
        }
        return corrected, applied, reasons

    # Nada que forzar: devolvemos la intención original (si existe)
    # o un template mínimo no válido (que luego fallará validación si se intenta).
    return corrected, applied, reasons

