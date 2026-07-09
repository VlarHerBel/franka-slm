"""Tests: ruta pick sugar_box (yaw-free + directo antes que safe_entry)."""

from panda_controller.sugar_box_yaw_free import (
    build_yaw_free_candidate_yaws,
    sugar_pick_route_preference,
)


def test_sugar_box_yaw_variants_ampliadas() -> None:
    names = {n for n, _ in build_yaw_free_candidate_yaws(0.5, joint7_yaw=0.2, current_tcp_yaw=0.3)}
    assert "commanded_yaw" in names
    assert "commanded_yaw_pi" in names
    assert "top_down_yaw_zero" in names
    assert "top_down_yaw_pi_over_2" in names
    assert "top_down_yaw_neg_pi_over_2" in names
    assert "top_down_yaw_pi" in names
    assert "yaw_from_current_joint7" in names
    assert "yaw_from_current_tcp" in names
    assert "top_down_yaw" not in names


def test_si_safe_entry_falla_pero_pregrasp_directo_funciona() -> None:
    assert sugar_pick_route_preference(direct_ok=True, safe_entry_ok=False) == "direct_pregrasp"
    assert sugar_pick_route_preference(direct_ok=True, safe_entry_ok=True) == "direct_pregrasp"


def test_no_abort_sin_probar_directo_cuando_directo_falla() -> None:
    assert sugar_pick_route_preference(direct_ok=False, safe_entry_ok=True) == "safe_entry"
    assert sugar_pick_route_preference(direct_ok=False, safe_entry_ok=False) == "abort"


def test_default_place_slot_index_zero_not_user_request() -> None:
    """place_slot_index=0 sin flag no implica slot físico 1."""
    from panda_controller.deposit_layout_policy import (
        DEFAULT_LABEL_SLOT_INDEX,
        plan_deposit_layout_slot,
        DEFAULT_PLACE_SLOTS_ORDERED,
    )

    assert DEFAULT_LABEL_SLOT_INDEX["chips_can"] == 2
    _, slot, plan = plan_deposit_layout_slot(
        label="chips_can",
        slots=DEFAULT_PLACE_SLOTS_ORDERED,
        occupied=[],
        user_requested_slot=False,
        requested_slot_index=0,
    )
    assert plan["selection_mode"] == "label_default_slot"
    assert slot is not None
    assert plan["selected_slot"] == "slot_3"
