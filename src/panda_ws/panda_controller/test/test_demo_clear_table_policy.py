"""Tests: política clear_table manual demo."""

from pathlib import Path

from panda_controller.demo_clear_table_policy import (
    discover_effective_pick_order,
    resolve_clear_table_manual_target,
    resolve_scene_pick_order,
)
from panda_controller.demo_scene_policy import load_demo_scene_policy


def test_scene_02_remaining_pick_order() -> None:
    order = resolve_scene_pick_order(
        "demo_scene_02_remaining_sugar_mustard",
        scene_02_order=["cracker_box", "chips_can", "sugar_box", "mustard_bottle"],
    )
    assert order == ["sugar_box", "mustard_bottle"]


def test_scene_02_remaining_bootstrap_skips_completed_cracker_chips() -> None:
    from panda_controller.demo_clear_table_policy import (
        resolve_clear_table_manual_step_bootstrap,
    )

    selected, idx, reason = resolve_clear_table_manual_step_bootstrap(
        completed_labels={"cracker_box", "chips_can"},
        pick_order=["sugar_box", "mustard_bottle"],
    )
    assert selected == "sugar_box"
    assert idx == 0
    assert reason == "pick_order"


def test_scene_02_pick_order_default() -> None:
    order = resolve_scene_pick_order(
        "demo_scene_02",
        scene_02_order=["cracker_box", "chips_can", "sugar_box", "mustard_bottle"],
    )
    assert order == [
        "cracker_box",
        "chips_can",
        "sugar_box",
        "mustard_bottle",
    ]


def test_scene_02_3obj_pick_order_from_yaml() -> None:
    scenes_dir = str(Path(__file__).resolve().parents[1] / "config" / "demo_scenes")
    policy = load_demo_scene_policy(
        "demo_scene_02_3obj",
        scenes_dir=scenes_dir,
        use_cache=False,
    )
    order = resolve_scene_pick_order(
        "demo_scene_02_3obj",
        scene_02_order=["cracker_box", "chips_can", "sugar_box", "mustard_bottle"],
        scene_policy=policy,
    )
    assert order == ["cracker_box", "chips_can", "sugar_box"]


def test_manual_step_auto_first_remaining() -> None:
    present = {"cracker_box", "chips_can", "mustard_bottle"}
    order = ["cracker_box", "chips_can", "mustard_bottle"]
    selected, reason, skipped = resolve_clear_table_manual_target(
        "",
        present_labels=present,
        completed_labels=set(),
        pick_order=order,
    )
    assert selected == "cracker_box"
    assert reason == "manual_step_next"
    assert skipped == []


def test_manual_step_skips_completed_request() -> None:
    present = {"chips_can", "mustard_bottle"}
    order = ["cracker_box", "chips_can", "mustard_bottle"]
    selected, reason, skipped = resolve_clear_table_manual_target(
        "cracker_box",
        present_labels=present,
        completed_labels={"cracker_box"},
        pick_order=order,
    )
    assert selected == "chips_can"
    assert reason == "advance_past_completed_request"
    assert skipped == ["cracker_box"]


def test_discover_effective_order_excludes_completed() -> None:
    present = {"cracker_box", "chips_can", "sugar_box", "mustard_bottle"}
    order = discover_effective_pick_order(
        pick_order=["cracker_box", "chips_can", "sugar_box", "mustard_bottle"],
        present_labels=present,
        completed_labels={"cracker_box"},
    )
    assert order == ["chips_can", "sugar_box", "mustard_bottle"]


def test_all_completed_returns_empty() -> None:
    present = {"cracker_box"}
    selected, reason, _ = resolve_clear_table_manual_target(
        "clear_table",
        present_labels=present,
        completed_labels={"cracker_box"},
        pick_order=["cracker_box", "chips_can"],
    )
    assert selected == ""
    assert reason == "all_completed_or_empty"
