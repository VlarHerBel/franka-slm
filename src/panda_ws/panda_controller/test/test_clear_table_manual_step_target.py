"""Tests: bootstrap y rechazo de target_candidate en clear_table_manual_step."""

from panda_controller.demo_clear_table_policy import (
    clear_table_manual_step_accepts_payload_target,
    resolve_clear_table_manual_step_bootstrap,
)


def test_bootstrap_run1_selects_cracker_box() -> None:
    selected, idx, reason = resolve_clear_table_manual_step_bootstrap(
        completed_labels=set(),
        pick_order=["cracker_box", "chips_can", "sugar_box", "mustard_bottle"],
    )
    assert selected == "cracker_box"
    assert idx == 0
    assert reason == "pick_order"


def test_bootstrap_run2_selects_chips_can_after_cracker_done() -> None:
    selected, idx, reason = resolve_clear_table_manual_step_bootstrap(
        completed_labels={"cracker_box"},
        pick_order=["cracker_box", "chips_can", "sugar_box", "mustard_bottle"],
    )
    assert selected == "chips_can"
    assert idx == 1
    assert reason == "pick_order"


def test_bootstrap_all_completed() -> None:
    selected, idx, reason = resolve_clear_table_manual_step_bootstrap(
        completed_labels={
            "cracker_box",
            "chips_can",
            "sugar_box",
            "mustard_bottle",
        },
        pick_order=["cracker_box", "chips_can", "sugar_box", "mustard_bottle"],
    )
    assert selected == ""
    assert idx == -1
    assert reason == "all_completed_or_empty"


def test_payload_rejected_when_target_candidate_is_chips_but_selected_cracker() -> None:
    assert not clear_table_manual_step_accepts_payload_target(
        selected_label="cracker_box",
        payload_label="chips_can",
        manual_step_active=True,
    )


def test_payload_accepted_when_target_candidate_matches_selected() -> None:
    assert clear_table_manual_step_accepts_payload_target(
        selected_label="cracker_box",
        payload_label="cracker_box",
        manual_step_active=True,
    )


def test_payload_not_filtered_when_manual_step_inactive() -> None:
    assert clear_table_manual_step_accepts_payload_target(
        selected_label="cracker_box",
        payload_label="chips_can",
        manual_step_active=False,
    )


def test_select_from_objects_not_payload_when_mismatch() -> None:
    """Simula wiring: bootstrap cracker_box; objects incluyen ambos; payload chips_can."""
    pick_order = ["cracker_box", "chips_can", "sugar_box", "mustard_bottle"]
    selected, _, _ = resolve_clear_table_manual_step_bootstrap(
        completed_labels=set(),
        pick_order=pick_order,
    )
    assert selected == "cracker_box"
    payload_label = "chips_can"
    assert not clear_table_manual_step_accepts_payload_target(
        selected_label=selected,
        payload_label=payload_label,
        manual_step_active=True,
    )
    objects = [
        {"label": "cracker_box", "entity_name": "runtime_ycb_cracker_1"},
        {"label": "chips_can", "entity_name": "runtime_ycb_chips_can_1"},
    ]
    pool = [o for o in objects if o["label"] == selected]
    assert len(pool) == 1
    assert pool[0]["label"] == "cracker_box"


def test_missing_selected_label_in_objects_yields_empty_pool() -> None:
    selected, _, _ = resolve_clear_table_manual_step_bootstrap(
        completed_labels=set(),
        pick_order=["cracker_box", "chips_can"],
    )
    objects = [{"label": "chips_can", "entity_name": "runtime_ycb_chips_can_1"}]
    pool = [o for o in objects if o["label"] == selected]
    assert pool == []
