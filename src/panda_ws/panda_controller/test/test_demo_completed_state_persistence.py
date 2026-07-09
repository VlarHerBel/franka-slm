"""Tests: persistencia demo (sin reset automático al arrancar)."""

import json
from pathlib import Path

from panda_controller.demo_object_order_policy import (
    load_demo_completed_state,
    save_demo_completed_state,
)


def test_completed_deposits_roundtrip_with_entity(tmp_path: Path) -> None:
    path = str(tmp_path / "demo_completed.json")
    deposits = [
        {
            "label": "cracker_box",
            "entity": "runtime_ycb_cracker_1",
            "slot": 0,
            "slot_index": 0,
            "x": -0.37,
            "y": 0.14,
        }
    ]
    assert save_demo_completed_state(
        path,
        completed_entities={"runtime_ycb_cracker_1"},
        completed_labels={"cracker_box"},
        completed_deposits=deposits,
    )
    ents, labels, loaded = load_demo_completed_state(path)
    assert "cracker_box" in labels
    assert len(loaded) == 1
    assert loaded[0]["label"] == "cracker_box"
    assert loaded[0]["entity"] == "runtime_ycb_cracker_1"
    assert loaded[0]["x"] == -0.37
    assert loaded[0]["y"] == 0.14


def test_json_schema_includes_deposit_fields(tmp_path: Path) -> None:
    path = tmp_path / "state.json"
    save_demo_completed_state(
        str(path),
        completed_entities=set(),
        completed_labels={"chips_can"},
        completed_deposits=[
            {
                "label": "chips_can",
                "entity": "runtime_ycb_chips_1",
                "slot": 1,
                "x": -0.37,
                "y": -0.14,
            }
        ],
    )
    raw = json.loads(path.read_text(encoding="utf-8"))
    assert "completed_deposits" in raw
    assert raw["completed_deposits"][0]["y"] == -0.14


def test_missing_file_returns_empty_deposits(tmp_path: Path) -> None:
    ents, labels, deposits = load_demo_completed_state(
        str(tmp_path / "missing.json")
    )
    assert ents == set()
    assert labels == set()
    assert deposits == []
