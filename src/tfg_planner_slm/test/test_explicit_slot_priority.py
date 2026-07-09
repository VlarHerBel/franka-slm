"""Petición explícita de cajón debe imponerse al JSON del SLM."""

from tfg_planner_slm.intent_parser import parse_user_command
from tfg_planner_slm.ollama_client import OllamaGenerateResult
from tfg_planner_slm.ros_pick_place_cmd import build_clear_table_step_ros2_args
from tfg_planner_slm.semantic_guardrails import apply_semantic_guardrails


def test_cuarto_cajon_overrides_slm_wrong_slot() -> None:
    slm_json = {
        "schema_version": "1.1",
        "intent": "pick_place",
        "target_label": "chips_can",
        "target_selector": {"type": "single"},
        "destination": {"type": "slot", "slot_index": 1, "slot_order": None},
        "execution": {"dry_run": True, "require_confirmation": True},
        "safety": {
            "requires_clarification": False,
            "clarification_question": "",
            "reject_reason": "",
        },
    }
    result = parse_user_command(
        "coge la lata de patatas y dejala en el cuarto cajon",
        ollama_result=OllamaGenerateResult(
            raw_response="",
            parsed_json=slm_json,
            latency_s=0.0,
        ),
    )
    assert result.final_intent is not None
    assert result.final_intent["intent"] == "pick_place"
    assert result.final_intent["target_label"] == "chips_can"
    assert result.final_intent["destination"]["slot_index"] == 3


def test_guardrails_force_pick_place_when_text_has_target_and_slot() -> None:
    wrong = {
        "schema_version": "1.1",
        "intent": "clear_table",
        "target_label": None,
        "target_selector": {"type": "all_supported_visible_objects"},
        "destination": {
            "type": "slots_ordered",
            "slot_index": None,
            "slot_order": [0, 1, 2, 3],
        },
        "execution": {"dry_run": True, "require_confirmation": True},
        "safety": {
            "requires_clarification": False,
            "clarification_question": "",
            "reject_reason": "",
        },
    }
    corrected, applied, reasons = apply_semantic_guardrails(
        "deja la lata de patatas en el cuarto cajon", wrong
    )
    assert applied
    assert corrected["intent"] == "pick_place"
    assert corrected["target_label"] == "chips_can"
    assert corrected["destination"]["slot_index"] == 3
    assert any("target_and_slot" in r for r in reasons)


def test_clear_table_step_passes_user_slot() -> None:
    argv = build_clear_table_step_ros2_args(
        dry_run=False,
        step_label="chips_can",
        scene_id="demo_scene_02_3obj",
        slot_index=2,
        slot_user_specified=True,
    )
    joined = " ".join(argv)
    assert "place_slot_user_specified:=true" in joined
    assert "place_slot_index:=2" in joined
