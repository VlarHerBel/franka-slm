"""Numeración humana de cajones vs índice interno 0..3."""

from tfg_planner_slm.semantic_guardrails import resolve_slot_from_text


def test_cajon_human_numbering() -> None:
    assert resolve_slot_from_text("dejalo en el cajon 1") == 0
    assert resolve_slot_from_text("cajon 2") == 1
    assert resolve_slot_from_text("cajon 3") == 2
    assert resolve_slot_from_text("cajon 4") == 3
    assert resolve_slot_from_text("2 cajon") == 1


def test_cajon_ordinals_match_human_numbering() -> None:
    assert resolve_slot_from_text("primer cajon") == 0
    assert resolve_slot_from_text("segundo cajon") == 1
    assert resolve_slot_from_text("tercer cajon") == 2
    assert resolve_slot_from_text("cuarto cajon") == 3
    assert resolve_slot_from_text("dejala en el 4º cajon") == 3


def test_internal_slot_index_unchanged() -> None:
    assert resolve_slot_from_text("slot 0") == 0
    assert resolve_slot_from_text("slot 3") == 3
    assert resolve_slot_from_text("hueco 2") == 2
