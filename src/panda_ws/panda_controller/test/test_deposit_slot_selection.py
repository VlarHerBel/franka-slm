"""Tests: slots físicos vs política por defecto vs petición explícita del usuario."""

from panda_controller.deposit_layout_policy import (
    DEFAULT_LABEL_SLOT_INDEX,
    DEFAULT_PLACE_SLOTS_ORDERED,
    plan_deposit_layout_slot,
    resolve_slot_index_from_name,
)


def test_default_label_slot_policy() -> None:
    assert DEFAULT_LABEL_SLOT_INDEX["cracker_box"] == 0
    assert DEFAULT_LABEL_SLOT_INDEX["sugar_box"] == 1
    assert DEFAULT_LABEL_SLOT_INDEX["chips_can"] == 2
    assert DEFAULT_LABEL_SLOT_INDEX["mustard_bottle"] == 3
    idx, slot, plan = plan_deposit_layout_slot(
        label="chips_can",
        slots=DEFAULT_PLACE_SLOTS_ORDERED,
        occupied=[],
    )
    assert plan["result"] == "OK"
    assert plan["selection_mode"] == "label_default_slot"
    assert plan["requested_slot"] == "none"
    assert plan["default_slot"] == "slot_3"
    assert plan["selected_slot"] == "slot_3"
    assert slot is not None
    assert abs(float(slot["y"]) - (-0.10)) < 1e-6
    assert abs(float(slot["x"]) - (-0.37)) < 1e-6


def test_user_requested_slot_overrides_label_default() -> None:
    occupied = [
        {
            "label": "cracker_box",
            "slot_index": 0,
            "x": -0.37,
            "y": 0.08,
        }
    ]
    idx, slot, plan = plan_deposit_layout_slot(
        label="chips_can",
        slots=DEFAULT_PLACE_SLOTS_ORDERED,
        occupied=occupied,
        user_requested_slot=True,
        requested_slot_index=1,
    )
    assert plan["result"] == "OK"
    assert plan["selection_mode"] == "user_requested_slot"
    assert plan["requested_slot"] == "slot_2"
    assert plan["selected_slot"] == "slot_2"
    assert idx == 1
    assert slot is not None
    assert abs(float(slot["x"]) - (-0.54)) < 1e-6


def test_user_slot_zero_not_overridden_by_scene_preferred_map() -> None:
    idx, slot, plan = plan_deposit_layout_slot(
        label="mustard_bottle",
        slots=DEFAULT_PLACE_SLOTS_ORDERED,
        occupied=[],
        user_requested_slot=True,
        requested_slot_index=0,
        label_slot_map={"mustard_bottle": 1},
    )
    assert plan["result"] == "OK"
    assert plan["selection_mode"] == "user_requested_slot"
    assert plan["selected_slot"] == "slot_1"
    assert idx == 0
    assert slot is not None
    assert abs(float(slot["x"]) - (-0.37)) < 1e-6


def test_default_place_slot_index_zero_is_not_user_request() -> None:
    """place_slot_index=0 sin flag no debe forzar slot_1."""
    idx, slot, plan = plan_deposit_layout_slot(
        label="chips_can",
        slots=DEFAULT_PLACE_SLOTS_ORDERED,
        occupied=[],
        user_requested_slot=False,
        requested_slot_index=0,
    )
    assert plan["selection_mode"] == "label_default_slot"
    assert plan["selected_slot"] == "slot_3"
    assert idx == 2


def test_completed_deposits_saved_after_place(tmp_path) -> None:
    from panda_controller.demo_object_order_policy import (
        load_demo_completed_state,
        save_demo_completed_state,
    )

    path = str(tmp_path / "state.json")
    deposits = [
        {
            "label": "cracker_box",
            "entity": "runtime_ycb_cracker_1",
            "slot_name": "slot_1",
            "slot_index": 0,
            "x": -0.37,
            "y": 0.10,
            "release_tcp_z": 0.42,
        },
        {
            "label": "chips_can",
            "entity": "runtime_ycb_chips_1",
            "slot_name": "slot_3",
            "slot_index": 2,
            "x": -0.37,
            "y": -0.10,
        },
    ]
    save_demo_completed_state(
        path,
        completed_entities={"runtime_ycb_cracker_1", "runtime_ycb_chips_1"},
        completed_labels={"cracker_box", "chips_can"},
        completed_deposits=deposits,
    )
    _, labels, loaded = load_demo_completed_state(path)
    assert labels == {"cracker_box", "chips_can"}
    assert len(loaded) == 2
    assert loaded[1]["slot_name"] == "slot_3"
    assert loaded[1]["y"] == -0.10


def test_occupied_slot_rejected() -> None:
    occupied = [
        {
            "label": "cracker_box",
            "slot_index": 0,
            "x": -0.37,
            "y": 0.10,
        }
    ]
    idx, slot, plan = plan_deposit_layout_slot(
        label="sugar_box",
        slots=DEFAULT_PLACE_SLOTS_ORDERED,
        occupied=occupied,
        user_requested_slot=True,
        requested_slot_index=0,
    )
    assert idx is None
    assert slot is None
    assert plan["result"] == "REJECTED"
    assert "occupied" in str(plan.get("reason", ""))


def test_footprint_collision_rejected() -> None:
    slots = (
        {"name": "slot_1", "x": -0.37, "y": 0.0},
        {"name": "slot_2", "x": -0.43, "y": 0.0},
    )
    occupied = [{"label": "cracker_box", "slot_index": 0, "x": -0.37, "y": 0.0}]
    idx, slot, plan = plan_deposit_layout_slot(
        label="chips_can",
        slots=slots,
        occupied=occupied,
        user_requested_slot=True,
        requested_slot_index=1,
        safety_margin_m=0.03,
    )
    assert plan["result"] == "REJECTED"
    checks = plan.get("collision_checks") or []
    assert checks and checks[0]["result"] == "TOO_CLOSE"


def test_resolve_slot_name_slot_2() -> None:
    assert resolve_slot_index_from_name("slot_2") == 1
    assert resolve_slot_index_from_name("2") == 1
