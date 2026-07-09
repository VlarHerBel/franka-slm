"""Tests: matriz compacta de depósito demo final."""

from panda_controller.deposit_layout_policy import (
    DEFAULT_DEPOSIT_SAFETY_MARGIN_XY_M,
    DEFAULT_LABEL_SLOT_INDEX,
    DEFAULT_PLACE_SLOTS_ORDERED,
    DEMO_FINAL_DEPOSIT_LABELS,
    deposit_slot_collision_check,
    footprint_radius_xy,
    plan_deposit_layout_slot,
    required_separation_xy,
)


def test_demo_final_has_four_labels_only() -> None:
    assert DEMO_FINAL_DEPOSIT_LABELS == frozenset(
        {"cracker_box", "chips_can", "sugar_box", "mustard_bottle"}
    )
    assert "gelatin_box" not in DEMO_FINAL_DEPOSIT_LABELS


def test_cracker_and_chips_default_slots_separated() -> None:
    occupied = [
        {
            "label": "cracker_box",
            "slot_index": DEFAULT_LABEL_SLOT_INDEX["cracker_box"],
            "x": float(DEFAULT_PLACE_SLOTS_ORDERED[0]["x"]),
            "y": float(DEFAULT_PLACE_SLOTS_ORDERED[0]["y"]),
        }
    ]
    idx, slot, plan = plan_deposit_layout_slot(
        label="chips_can",
        slots=DEFAULT_PLACE_SLOTS_ORDERED,
        occupied=occupied,
        safety_margin_m=DEFAULT_DEPOSIT_SAFETY_MARGIN_XY_M,
    )
    assert plan["result"] == "OK"
    assert plan["selection_mode"] == "label_default_slot"
    assert plan["selected_slot"] == "slot_3"
    assert idx == 2
    dist = (
        (float(slot["x"]) - occupied[0]["x"]) ** 2
        + (float(slot["y"]) - occupied[0]["y"]) ** 2
    ) ** 0.5
    req = required_separation_xy(
        footprint_radius_xy("cracker_box"), footprint_radius_xy("chips_can")
    )
    assert dist >= req - 1e-6


def test_slot2_sugar_x_farther_from_cracker() -> None:
    cracker_x = float(DEFAULT_PLACE_SLOTS_ORDERED[0]["x"])
    sugar_x = float(DEFAULT_PLACE_SLOTS_ORDERED[1]["x"])
    assert sugar_x < cracker_x - 0.15
    assert float(DEFAULT_PLACE_SLOTS_ORDERED[0]["y"]) == 0.08


def test_collision_check_too_close_log_fields() -> None:
    result, dist, req = deposit_slot_collision_check(
        new_label="chips_can",
        new_x=-0.43,
        new_y=0.0,
        existing_label="cracker_box",
        existing_x=-0.37,
        existing_y=0.0,
        new_radius=0.0375,
        existing_radius=0.081,
        safety_margin_m=0.03,
    )
    assert result == "TOO_CLOSE"
    assert dist == 0.06
