"""Tests transporte online acotado golden fast."""

from __future__ import annotations

from panda_controller.demo_golden_fast_transport import (
    GOLDEN_FAST_DEFAULT_ENTRY_MODE,
    build_golden_fast_bounded_escape_options,
    golden_fast_bounded_hand_z_candidates,
    golden_fast_transport_prevalidated,
    resolve_golden_fast_transport_sequence,
)


def test_resolve_golden_fast_transport_sequence_prefers_golden() -> None:
    seq = resolve_golden_fast_transport_sequence(
        golden_route=["carry_mid_high", "box_high"],
        default_sequence=["carry_front_high", "box_high"],
        scene_policy_route=["carry_front_high", "turn_back_extended_aligned", "box_high"],
        golden_fast_active=True,
    )
    assert seq == ["carry_mid_high", "box_high"]


def test_resolve_golden_fast_transport_sequence_scene_when_not_golden() -> None:
    seq = resolve_golden_fast_transport_sequence(
        golden_route=["carry_mid_high"],
        default_sequence=["carry_front_high", "box_high"],
        scene_policy_route=["carry_mid_high", "box_high"],
        golden_fast_active=False,
    )
    assert seq == ["carry_mid_high", "box_high"]


def test_golden_fast_bounded_hand_z_candidates_limited() -> None:
    zs = golden_fast_bounded_hand_z_candidates(
        0.685,
        {"carry_safe_hand_z": 0.907, "preferred_hand_z": 0.700},
    )
    assert len(zs) <= 3
    assert zs[0] == 0.685


def test_build_golden_fast_bounded_escape_options_vertical_mode() -> None:
    opts = build_golden_fast_bounded_escape_options(
        (0.455, 0.115, 0.685),
        golden_entry_mode=GOLDEN_FAST_DEFAULT_ENTRY_MODE,
    )
    assert 1 <= len(opts) <= 3
    assert all(o["mode"] == GOLDEN_FAST_DEFAULT_ENTRY_MODE for o in opts)
    assert opts[0]["selection_reason"].startswith("golden_fast_bounded_")


def test_golden_fast_transport_prevalidated() -> None:
    assert golden_fast_transport_prevalidated(
        {
            "_demo_golden_fast_execute": True,
            "_golden_transport_prevalidated": True,
            "_golden_transport_route": ["carry_mid_high"],
        }
    )
    assert not golden_fast_transport_prevalidated({"_demo_golden_fast_execute": False})
