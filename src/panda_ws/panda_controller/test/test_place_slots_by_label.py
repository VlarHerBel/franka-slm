"""Tests: slots físicos y política por defecto por etiqueta."""

from panda_controller.deposit_layout_policy import (
    DEFAULT_LABEL_SLOT_INDEX,
    DEFAULT_PLACE_SLOTS_ORDERED,
    default_slot_name_for_label,
)


def test_default_label_slot_indices() -> None:
    assert DEFAULT_LABEL_SLOT_INDEX["cracker_box"] == 0
    assert DEFAULT_LABEL_SLOT_INDEX["sugar_box"] == 1
    assert DEFAULT_LABEL_SLOT_INDEX["chips_can"] == 2
    assert DEFAULT_LABEL_SLOT_INDEX["mustard_bottle"] == 3


def test_physical_slot_names() -> None:
    names = [str(s["name"]) for s in DEFAULT_PLACE_SLOTS_ORDERED]
    assert names == ["slot_1", "slot_2", "slot_3", "slot_4"]


def test_default_slot_names_by_label() -> None:
    assert default_slot_name_for_label("cracker_box") == "slot_1"
    assert default_slot_name_for_label("chips_can") == "slot_3"
    assert default_slot_name_for_label("sugar_box") == "slot_2"
    assert default_slot_name_for_label("mustard_bottle") == "slot_4"


def test_gelatin_not_in_demo_label_policy() -> None:
    assert "gelatin_box" not in DEFAULT_LABEL_SLOT_INDEX
