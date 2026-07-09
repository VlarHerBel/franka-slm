"""Tests: orden demo multiobjeto y estado completado."""

from panda_controller.demo_object_order_policy import (
    collect_present_table_labels,
    load_demo_completed_state,
    parse_demo_blocking_rules,
    resolve_effective_demo_pick_label,
    resolve_present_table_labels,
    save_demo_completed_state,
)


def test_sugar_redirect_when_cracker_on_table() -> None:
    present = {"sugar_box", "cracker_box"}
    selected, reason, blocker = resolve_effective_demo_pick_label(
        "sugar_box",
        present_labels=present,
        completed_entities=set(),
        completed_labels=set(),
        blocking_rules={"sugar_box": ["cracker_box"]},
        priority_order=["cracker_box", "sugar_box"],
    )
    assert selected == "cracker_box"
    assert reason == "blocker_before_target"
    assert blocker == "cracker_box"


def test_sugar_direct_when_cracker_completed() -> None:
    present = {"sugar_box", "cracker_box"}
    selected, reason, blocker = resolve_effective_demo_pick_label(
        "sugar_box",
        present_labels=present,
        completed_entities=set(),
        completed_labels={"cracker_box"},
        blocking_rules={"sugar_box": ["cracker_box"]},
        priority_order=["cracker_box", "sugar_box"],
    )
    assert selected == "sugar_box"
    assert reason == "direct_request"
    assert blocker == ""


def test_collect_present_excludes_completed() -> None:
    objs = [
        {"label": "cracker_box", "entity_name": "runtime_ycb_cracker_1"},
        {"label": "sugar_box", "entity_name": "runtime_ycb_sugar_1"},
    ]
    present = collect_present_table_labels(
        objs,
        completed_entities=set(),
        completed_labels={"cracker_box"},
    )
    assert present == {"sugar_box"}


def test_persist_completed_state_roundtrip(tmp_path) -> None:
    path = str(tmp_path / "demo_completed.json")
    assert save_demo_completed_state(
        path,
        completed_entities={"runtime_ycb_cracker_1"},
        completed_labels={"cracker_box"},
    )
    ents, labels, deposits = load_demo_completed_state(path)
    assert "runtime_ycb_cracker_1" in ents
    assert "cracker_box" in labels
    assert deposits == []

    assert save_demo_completed_state(
        path,
        completed_entities={"runtime_ycb_cracker_1"},
        completed_labels={"cracker_box"},
        completed_deposits=[
            {
                "label": "cracker_box",
                "entity": "runtime_ycb_cracker_1",
                "slot": 0,
                "slot_index": 0,
                "x": -0.37,
                "y": 0.14,
            }
        ],
    )
    _, _, deposits2 = load_demo_completed_state(path)
    assert len(deposits2) == 1
    assert deposits2[0]["label"] == "cracker_box"
    assert deposits2[0]["y"] == 0.14


def test_parse_blocking_rules_json_string() -> None:
    rules = parse_demo_blocking_rules('{"sugar_box": ["cracker_box"]}')
    assert rules["sugar_box"] == ["cracker_box"]


def test_resolve_present_table_labels_prefers_live_executor() -> None:
    scene = [
        {"label": "cracker_box"},
        {"label": "chips_can"},
        {"label": "sugar_box"},
    ]
    live = [{"label": "sugar_box", "entity_name": "runtime_ycb_sugar_2"}]
    present = resolve_present_table_labels(
        scene_objects=scene,
        executor_objects=live,
        completed_entities=set(),
        completed_labels=set(),
    )
    assert present == {"sugar_box"}
