"""Pipeline: Ollama → JSON → schema → guardrails → intent final validado."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .intent_schema import PYDANTIC_V2, RobotIntentModel, validate_intent_payload
from .json_extract import extract_json_object
from .ollama_client import DEFAULT_MODEL, OllamaGenerateResult, generate_intent
from .semantic_guardrails import apply_semantic_guardrails, resolve_slot_from_text
from .slot_state import (
    SlotOccupancy,
    apply_any_slot_resolution,
    apply_deposit_full_checks,
    apply_slot_occupancy_checks,
)


@dataclass
class ParsedCommandResult:
    """Resultado completo del pipeline de parsing."""

    original_text: str
    raw_model_json: Optional[Dict[str, Any]] = None
    raw_response: str = ""
    validated_json: Optional[Dict[str, Any]] = None
    guardrailed_json: Optional[Dict[str, Any]] = None
    final_intent: Optional[Dict[str, Any]] = None
    guardrails_applied: bool = False
    guardrail_reasons: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    ollama_latency_s: float = 0.0


def _model_to_dict(model: RobotIntentModel) -> Dict[str, Any]:
    if PYDANTIC_V2:
        return model.model_dump()
    return model.dict()  # type: ignore[attr-defined]


def parse_user_command(
    text: str,
    *,
    model: str = DEFAULT_MODEL,
    timeout_s: float = 90.0,
    ollama_url: Optional[str] = None,
    ollama_result: Optional[OllamaGenerateResult] = None,
    slot_occupancy: Optional[SlotOccupancy] = None,
) -> ParsedCommandResult:
    """Pipeline principal: SLM → JSON → Pydantic → guardrails → revalidación."""
    result = ParsedCommandResult(original_text=text)

    pre_corrected, pre_applied, pre_reasons = apply_semantic_guardrails(text, {})
    slm_skipped = False
    parsed: Optional[Dict[str, Any]] = None

    if pre_applied:
        parsed = dict(pre_corrected)
        slm_skipped = True
        result.ollama_latency_s = 0.0
        result.raw_response = ""
        print(
            "[GUARDRAIL] deterministic_bypass_no_slm reasons=%s"
            % ",".join(pre_reasons),
            flush=True,
        )
    elif ollama_result is None:
        ollama_result = generate_intent(
            text, model=model, timeout_s=timeout_s, ollama_url=ollama_url
        )

    if not slm_skipped:
        assert ollama_result is not None
        result.raw_response = ollama_result.raw_response
        result.ollama_latency_s = ollama_result.latency_s

        if ollama_result.request_error:
            result.errors.append(ollama_result.request_error)
            return result

        parsed = ollama_result.parsed_json
        if parsed is None:
            parsed, parse_err = extract_json_object(ollama_result.raw_response)
            if parsed is None:
                result.errors.append(
                    parse_err or ollama_result.parse_error or "json_parse_error"
                )
                return result

    assert parsed is not None
    result.raw_model_json = dict(parsed)

    ok, intent_model, schema_err = validate_intent_payload(parsed)
    if ok and intent_model is not None:
        result.validated_json = _model_to_dict(intent_model)
    else:
        result.errors.append("schema_validation_error:%s" % (schema_err or "unknown"))

    if slm_skipped:
        corrected, guardrails_applied, reasons = dict(pre_corrected), True, list(pre_reasons)
    else:
        corrected, guardrails_applied, reasons = apply_semantic_guardrails(text, parsed)
    corrected, any_applied, any_reasons = apply_any_slot_resolution(
        text, corrected, slot_occupancy
    )
    if any_applied:
        guardrails_applied = True
        reasons.extend(any_reasons)

    corrected, full_applied, full_reasons = apply_deposit_full_checks(
        corrected, slot_occupancy
    )
    if full_applied:
        guardrails_applied = True
        reasons.extend(full_reasons)

    corrected, occ_applied, occ_reasons = apply_slot_occupancy_checks(
        corrected, slot_occupancy
    )
    if occ_applied:
        guardrails_applied = True
        reasons.extend(occ_reasons)

    result.guardrailed_json = corrected
    result.guardrails_applied = guardrails_applied
    result.guardrail_reasons = list(reasons)

    if guardrails_applied and reasons:
        for reason in reasons:
            if reason.startswith("slot_occupancy"):
                tag = "[SLOT_STATE]"
            elif reason == "deposit_box_full":
                tag = "[DEPOSIT_STATE]"
            elif reason.startswith(("resolved_any_slot", "any_slot_")):
                tag = "[ANY_SLOT]"
            else:
                tag = "[GUARDRAIL]"
            print("%s %s" % (tag, reason), flush=True)

    explicit_slot = resolve_slot_from_text(text)
    if (
        explicit_slot is not None
        and isinstance(corrected, dict)
        and corrected.get("intent") == "pick_place"
    ):
        dest = corrected.get("destination") or {}
        if dest.get("slot_index") != explicit_slot or dest.get("type") != "slot":
            corrected = dict(corrected)
            corrected["destination"] = {
                "type": "slot",
                "slot_index": int(explicit_slot),
                "slot_order": None,
            }
            guardrails_applied = True
            reasons.append(
                "final_enforce_slot_from_text -> slot_index=%d" % int(explicit_slot)
            )
            result.guardrailed_json = corrected
            result.guardrail_reasons = list(reasons)
            result.guardrails_applied = guardrails_applied

    ok2, final_model, schema_err2 = validate_intent_payload(corrected)
    if ok2 and final_model is not None:
        result.final_intent = _model_to_dict(final_model)
    else:
        result.errors.append(
            "guardrailed_schema_validation_error:%s" % (schema_err2 or "unknown")
        )
        # Si el modelo raw validaba pero guardrails rompió schema, mantener validated como referencia.
        if result.validated_json and not result.final_intent:
            result.final_intent = None

    return result


def final_intent_json_string(result: ParsedCommandResult) -> str:
    if result.final_intent is None:
        return ""
    return json.dumps(result.final_intent, ensure_ascii=False, indent=2)
