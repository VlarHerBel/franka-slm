"""Schema Pydantic v1.1 + JSON Schema para Ollama structured outputs."""

from __future__ import annotations

import json
from typing import Any, List, Literal, Optional, Tuple

import pydantic

PYDANTIC_V2 = int(pydantic.__version__.split(".", maxsplit=1)[0]) >= 2

if PYDANTIC_V2:
    from pydantic import BaseModel, ConfigDict, ValidationError, field_validator, model_validator
else:
    from pydantic import BaseModel, ValidationError, validator  # type: ignore

SUPPORTED_OBJECTS = ("cracker_box", "sugar_box", "chips_can", "mustard_bottle")
DEFAULT_SLOT_ORDER: List[int] = [0, 1, 2, 3]

ROBOT_INTENT_JSON_SCHEMA: dict = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "schema_version": {"type": "string", "enum": ["1.1"]},
        "intent": {
            "type": "string",
            "enum": [
                "pick_place",
                "clear_table",
                "go_home",
                "open_gripper",
                "close_gripper",
                "status",
                "ask_clarification",
                "reject",
            ],
        },
        "target_label": {
            "anyOf": [
                {
                    "type": "string",
                    "enum": list(SUPPORTED_OBJECTS),
                },
                {"type": "null"},
            ],
        },
        "target_selector": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "type": {
                    "anyOf": [
                        {
                            "type": "string",
                            "enum": ["single", "all_supported_visible_objects"],
                        },
                        {"type": "null"},
                    ],
                },
            },
            "required": ["type"],
        },
        "destination": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "type": {
                    "anyOf": [
                        {
                            "type": "string",
                            "enum": ["slot", "slots_ordered"],
                        },
                        {"type": "null"},
                    ],
                },
                "slot_index": {
                    "anyOf": [
                        {"type": "integer", "enum": [0, 1, 2, 3]},
                        {"type": "null"},
                    ],
                },
                "slot_order": {
                    "anyOf": [
                        {
                            "type": "array",
                            "items": {"type": "integer", "enum": [0, 1, 2, 3]},
                            "minItems": 4,
                            "maxItems": 4,
                        },
                        {"type": "null"},
                    ],
                },
            },
            "required": ["type", "slot_index", "slot_order"],
        },
        "execution": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "dry_run": {"type": "boolean"},
                "require_confirmation": {"type": "boolean"},
            },
            "required": ["dry_run", "require_confirmation"],
        },
        "safety": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "requires_clarification": {"type": "boolean"},
                "clarification_question": {"type": "string"},
                "reject_reason": {"type": "string"},
            },
            "required": [
                "requires_clarification",
                "clarification_question",
                "reject_reason",
            ],
        },
    },
    "required": [
        "schema_version",
        "intent",
        "target_label",
        "target_selector",
        "destination",
        "execution",
        "safety",
    ],
}

FORBIDDEN_TOP_LEVEL_KEYS = frozenset(
    {
        "tarea",
        "accion",
        "objeto",
        "ubicacion",
        "comando",
        "orden",
        "mision",
        "task",
        "action",
        "object",
        "location",
        "command",
    }
)

ALLOWED_TOP_LEVEL_KEYS = frozenset(ROBOT_INTENT_JSON_SCHEMA["required"])


class TargetSelectorModel(BaseModel):
    if PYDANTIC_V2:
        model_config = ConfigDict(extra="forbid")
    else:

        class Config:
            extra = "forbid"

    type: Optional[Literal["single", "all_supported_visible_objects"]] = None


class DestinationModel(BaseModel):
    if PYDANTIC_V2:
        model_config = ConfigDict(extra="forbid")
    else:

        class Config:
            extra = "forbid"

    type: Optional[Literal["slot", "slots_ordered"]] = None
    slot_index: Optional[Literal[0, 1, 2, 3]] = None
    slot_order: Optional[List[Literal[0, 1, 2, 3]]] = None

    if PYDANTIC_V2:

        @field_validator("slot_order", mode="before")
        @classmethod
        def _normalize_slot_order(cls, v: Any) -> Any:
            if v is None:
                return None
            if isinstance(v, list):
                return [int(x) for x in v]
            return v


class ExecutionModel(BaseModel):
    if PYDANTIC_V2:
        model_config = ConfigDict(extra="forbid")
    else:

        class Config:
            extra = "forbid"

    dry_run: bool = True
    require_confirmation: bool = True


class SafetyModel(BaseModel):
    if PYDANTIC_V2:
        model_config = ConfigDict(extra="forbid")
    else:

        class Config:
            extra = "forbid"

    requires_clarification: bool = False
    clarification_question: str = ""
    reject_reason: str = ""

    if PYDANTIC_V2:

        @field_validator("clarification_question", "reject_reason", mode="before")
        @classmethod
        def _coerce_str(cls, v: Any) -> str:
            return "" if v is None else str(v)


class RobotIntentModel(BaseModel):
    if PYDANTIC_V2:
        model_config = ConfigDict(extra="forbid")
    else:

        class Config:
            extra = "forbid"

    schema_version: Literal["1.1"] = "1.1"
    intent: Literal[
        "pick_place",
        "clear_table",
        "go_home",
        "open_gripper",
        "close_gripper",
        "status",
        "ask_clarification",
        "reject",
    ]
    target_label: Optional[Literal["cracker_box", "sugar_box", "chips_can", "mustard_bottle"]] = (
        None
    )
    target_selector: TargetSelectorModel
    destination: DestinationModel
    execution: ExecutionModel
    safety: SafetyModel

    if PYDANTIC_V2:

        @model_validator(mode="after")
        def _cross_field_rules(self) -> "RobotIntentModel":
            intent = self.intent
            if intent == "pick_place":
                if self.target_label is None:
                    raise ValueError("pick_place requires target_label")
                if self.target_selector.type != "single":
                    raise ValueError("pick_place requires target_selector.type=single")
                if self.destination.type != "slot":
                    raise ValueError("pick_place requires destination.type=slot")
                if self.destination.slot_index is None:
                    raise ValueError("pick_place requires destination.slot_index")
                if self.destination.slot_order is not None:
                    raise ValueError("pick_place requires destination.slot_order=null")
            elif intent == "clear_table":
                if self.target_label is not None:
                    raise ValueError("clear_table requires target_label=null")
                if self.target_selector.type != "all_supported_visible_objects":
                    raise ValueError(
                        "clear_table requires target_selector.type=all_supported_visible_objects"
                    )
                if self.destination.type != "slots_ordered":
                    raise ValueError("clear_table requires destination.type=slots_ordered")
                if self.destination.slot_index is not None:
                    raise ValueError("clear_table requires destination.slot_index=null")
                if self.destination.slot_order is None:
                    raise ValueError("clear_table requires destination.slot_order")
                if not self.execution.dry_run or not self.execution.require_confirmation:
                    raise ValueError(
                        "clear_table requires execution.dry_run=true and require_confirmation=true"
                    )
            elif intent in ("go_home", "open_gripper", "close_gripper", "status"):
                if self.target_label is not None:
                    raise ValueError("%s requires target_label=null" % intent)
                if self.target_selector.type is not None:
                    raise ValueError("%s requires target_selector.type=null" % intent)
                if self.destination.type is not None:
                    raise ValueError("%s requires destination.type=null" % intent)
                if self.destination.slot_index is not None:
                    raise ValueError("%s requires destination.slot_index=null" % intent)
                if self.destination.slot_order is not None:
                    raise ValueError("%s requires destination.slot_order=null" % intent)
            elif intent == "ask_clarification":
                if not self.safety.requires_clarification:
                    raise ValueError(
                        "ask_clarification requires safety.requires_clarification=true"
                    )
                if not str(self.safety.clarification_question).strip():
                    raise ValueError(
                        "ask_clarification requires non-empty clarification_question"
                    )
            elif intent == "reject":
                if not str(self.safety.reject_reason).strip():
                    raise ValueError("reject requires non-empty safety.reject_reason")
            return self

    else:

        @validator("safety")
        def _safety_rules(cls, v: SafetyModel, values: dict) -> SafetyModel:
            intent = values.get("intent")
            if intent == "ask_clarification":
                if not v.requires_clarification:
                    raise ValueError("ask_clarification requires requires_clarification")
                if not str(v.clarification_question).strip():
                    raise ValueError("ask_clarification requires clarification_question")
            if intent == "reject" and not str(v.reject_reason).strip():
                raise ValueError("reject requires reject_reason")
            return v


def has_forbidden_keys(data: dict) -> Optional[str]:
    bad = [k for k in data.keys() if k in FORBIDDEN_TOP_LEVEL_KEYS]
    if bad:
        return "forbidden_keys:%s" % ",".join(sorted(bad))
    extra = [k for k in data.keys() if k not in ALLOWED_TOP_LEVEL_KEYS]
    if extra:
        return "extra_top_level_keys:%s" % ",".join(sorted(extra))
    return None


def normalize_slot_order(value: Any) -> Optional[List[int]]:
    if value is None:
        return None
    if isinstance(value, list):
        return [int(x) for x in value]
    return None


def slot_orders_match(expected: Any, predicted: Any) -> bool:
    exp = normalize_slot_order(expected)
    pred = normalize_slot_order(predicted)
    if exp is None and pred is None:
        return True
    if exp is None or pred is None:
        return False
    return exp == pred


def validate_intent_payload(data: dict) -> Tuple[bool, Optional[RobotIntentModel], str]:
    forbidden = has_forbidden_keys(data)
    if forbidden:
        return False, None, forbidden
    try:
        if PYDANTIC_V2:
            model = RobotIntentModel.model_validate(data)
        else:
            model = RobotIntentModel.parse_obj(data)  # type: ignore[attr-defined]
        return True, model, ""
    except ValidationError as exc:
        return False, None, str(exc)
    except Exception as exc:
        return False, None, str(exc)


def predicted_from_model(
    model: RobotIntentModel,
) -> dict:
    return {
        "intent": model.intent,
        "target_label": model.target_label,
        "target_selector_type": model.target_selector.type,
        "destination_type": model.destination.type,
        "slot_index": model.destination.slot_index,
        "slot_order": normalize_slot_order(model.destination.slot_order),
        "reject_reason": str(model.safety.reject_reason or "").strip() or None,
    }


def slot_order_to_csv(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        return json.dumps(value, ensure_ascii=False)
    return str(value)
