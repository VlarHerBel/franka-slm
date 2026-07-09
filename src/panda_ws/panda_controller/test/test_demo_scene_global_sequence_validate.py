"""Tests validador secuencial global demo_scene_02."""

from __future__ import annotations

from pathlib import Path

from panda_controller.attached_transport_phases import (
    resolve_transport_phase_clearance_thresholds,
)
from panda_controller.demo_scene_global_sequence_validate import (
    evaluate_demo_scene_object_sequence_step,
    format_demo_scene_global_sequence_validate_log,
    format_demo_scene_object_sequence_validate_log,
    remaining_obstacles_for_target,
)
from panda_controller.demo_scene_policy import load_demo_scene_policy


def _scene02_policy() -> dict:
    scenes_dir = str(Path(__file__).resolve().parents[1] / "config" / "demo_scenes")
    policy = load_demo_scene_policy("demo_scene_02", scenes_dir=scenes_dir, use_cache=False)
    assert policy is not None
    return policy


def test_remaining_obstacles_progressive() -> None:
    order = ["cracker_box", "chips_can", "sugar_box", "mustard_bottle"]
    assert remaining_obstacles_for_target(order, "cracker_box") == [
        "chips_can",
        "sugar_box",
        "mustard_bottle",
    ]
    assert remaining_obstacles_for_target(order, "chips_can") == [
        "sugar_box",
        "mustard_bottle",
    ]
    assert remaining_obstacles_for_target(order, "sugar_box") == ["mustard_bottle"]
    assert remaining_obstacles_for_target(order, "mustard_bottle") == []


def test_global_sequence_log_format() -> None:
    policy = _scene02_policy()
    log = format_demo_scene_global_sequence_validate_log(
        scene_id=str(policy["scene_id"]),
        order=policy["pick_order"],
    )
    assert "[DEMO_SCENE_GLOBAL_SEQUENCE_VALIDATE]" in log
    assert "scene_id=demo_scene_02" in log
    assert "cracker_box" in log
    assert "mustard_bottle" in log


def test_object_sequence_validate_log_format() -> None:
    log = format_demo_scene_object_sequence_validate_log(
        target_label="cracker_box",
        remaining_obstacles=["chips_can", "sugar_box", "mustard_bottle"],
        local_exit_min_clearance=0.05,
        global_route_min_clearance=0.10,
        result="OK",
        reason="rear_retreat_x_negative",
    )
    assert "[DEMO_SCENE_OBJECT_SEQUENCE_VALIDATE]" in log
    assert "target_label=cracker_box" in log
    assert "local_exit_min_clearance=0.0500" in log
    assert "global_route_min_clearance=0.1000" in log


def test_phase_clearance_from_scene_yaml() -> None:
    policy = _scene02_policy()
    phase = resolve_transport_phase_clearance_thresholds(policy)
    assert phase["local_exit_required_clearance_m"] == 0.050
    assert phase["local_exit_min_table_clearance_m"] == 0.200
    assert phase["reconfiguration_required_clearance_m"] == 0.080
    assert phase["global_route_required_clearance_m"] == 0.100


def test_evaluate_object_sequence_step_accept() -> None:
    policy = _scene02_policy()
    step = evaluate_demo_scene_object_sequence_step(
        scene_policy=policy,
        target_label="cracker_box",
        transport_score={
            "result": "ACCEPT",
            "transport_entry_possible": True,
            "reconfiguration_zone_ok": True,
            "direct_action_to_hub_ok": True,
            "selected_transport_mode": "rear_retreat_x_negative",
        },
    )
    assert step["result"] == "OK"
    assert step["remaining_obstacles"] == [
        "chips_can",
        "sugar_box",
        "mustard_bottle",
    ]
