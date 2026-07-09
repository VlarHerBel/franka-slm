"""Bypass SLM cuando los guardrails resuelven el intent de forma determinista."""

from tfg_planner_slm.intent_parser import parse_user_command
from tfg_planner_slm.semantic_guardrails import is_clear_table_text


def test_is_clear_table_text_recoge_la_mesa() -> None:
    assert is_clear_table_text("recoge la mesa") is True
    assert is_clear_table_text("recoger la mesa") is True


def test_parse_clear_table_skips_slm() -> None:
    result = parse_user_command("recoge la mesa")
    assert result.final_intent is not None
    assert result.final_intent["intent"] == "clear_table"
    assert float(result.ollama_latency_s) == 0.0
    assert result.errors == []
