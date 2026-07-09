"""Tests carga política de escena demo desde YAML."""

from __future__ import annotations

from pathlib import Path

from panda_controller.demo_scene_policy import (
    apply_scene_policy_to_carry_transport,
    demo_scene_policy_to_spawn_entries,
    filter_transport_route_for_scene,
    load_demo_scene_policy,
    resolve_pick_order_from_scene_policy,
)
from panda_controller.generic_known_scene_carry_planner import (
    resolve_carry_transport_policy,
    resolve_post_pick_transport_entry_target,
)


def _scene02_policy() -> dict:
    scenes_dir = str(
        Path(__file__).resolve().parents[1] / "config" / "demo_scenes"
    )
    policy = load_demo_scene_policy("demo_scene_02", scenes_dir=scenes_dir, use_cache=False)
    assert policy is not None
    return policy


def test_load_chips_mustard_01_policy_random_spawn() -> None:
    scenes_dir = str(
        Path(__file__).resolve().parents[1] / "config" / "demo_scenes"
    )
    policy = load_demo_scene_policy(
        "chips_mustard_01", scenes_dir=scenes_dir, use_cache=False
    )
    assert policy is not None
    assert policy["scene_id"] == "chips_mustard_01"
    assert policy["pick_order"] == ["chips_can", "mustard_bottle"]
    assert policy["objects"]["chips_can"]["preferred_slot"] == 0
    assert policy["objects"]["mustard_bottle"]["preferred_slot"] == 0


def test_load_two_boxes_03_policy_random_spawn() -> None:
    scenes_dir = str(
        Path(__file__).resolve().parents[1] / "config" / "demo_scenes"
    )
    policy = load_demo_scene_policy("two_boxes_03", scenes_dir=scenes_dir, use_cache=False)
    assert policy is not None
    assert policy["scene_id"] == "two_boxes_03"
    assert policy["pick_order"] == ["cracker_box", "sugar_box"]
    assert policy["objects"]["cracker_box"]["preferred_slot"] == 0
    assert policy["objects"]["sugar_box"]["preferred_slot"] == 1


def test_load_demo_scene_02_policy() -> None:
    policy = _scene02_policy()
    assert policy["scene_id"] == "demo_scene_02"
    assert policy["pick_order"][0] == "cracker_box"
    assert "cracker_box" in policy["objects"]
    assert policy["transport_policy"]["backend"] == "direct_action"


def test_spawn_entries_from_policy() -> None:
    policy = _scene02_policy()
    entries = demo_scene_policy_to_spawn_entries(policy)
    labels = [e["label"] for e in entries]
    assert labels[0] == "cracker_box"
    assert set(labels) == {"cracker_box", "chips_can", "sugar_box", "mustard_bottle"}


def test_filter_forbidden_carry_front_high() -> None:
    policy = _scene02_policy()
    before = ["carry_front_high", "carry_mid_high", "box_high"]
    after, forbidden = filter_transport_route_for_scene(
        before, policy, obstacles_remaining=True
    )
    assert forbidden == ["carry_front_high"]
    assert after == ["carry_mid_high", "box_high"]


def test_carry_policy_from_scene_yaml() -> None:
    policy = _scene02_policy()
    candidate = {
        "label": "cracker_box",
        "scene_id": "demo_scene_02",
        "_scene_policy": policy,
        "scene_obstacles": [{"label": "chips_can", "is_target": False}],
    }
    carry = resolve_carry_transport_policy(candidate)
    assert carry["post_pick_skip_carry_front_high"] is True
    assert carry["use_lateral_transport_corridors"] is False
    assert "rear_retreat_x_negative" in carry["local_exit_candidates"]
    assert "rear_retreat_x_negative_far" in carry["local_exit_candidates"]
    assert "rear_retreat_x_negative_raise_far" in carry["local_exit_candidates"]
    assert "vertical_raise_then_rear_retreat" in carry["local_exit_candidates"]


def test_entry_target_from_scene_yaml() -> None:
    policy = _scene02_policy()
    candidate = {
        "label": "cracker_box",
        "scene_id": "demo_scene_02",
        "_scene_policy": policy,
        "scene_obstacles": [{"label": "chips_can", "is_target": False}],
    }
    carry = resolve_carry_transport_policy(candidate)
    entry = resolve_post_pick_transport_entry_target(
        candidate,
        carry,
        default_first_waypoint="carry_front_high",
        waypoints_data={},
    )
    assert entry["skip_carry_front_high"] is True
    assert entry["entry_target_waypoint"] == "carry_mid_high"
    assert entry["defer_entry_hub_to_deterministic_transport"] is True


def test_load_two_boxes_01_transport_policy() -> None:
    scenes_dir = str(
        Path(__file__).resolve().parents[1] / "config" / "demo_scenes"
    )
    policy = load_demo_scene_policy(
        "two_boxes_01", scenes_dir=scenes_dir, use_cache=False
    )
    assert policy is not None
    assert "carry_front_high" in policy["transport_policy"][
        "forbidden_waypoints_when_obstacles_remaining"
    ]
    candidate = {
        "label": "cracker_box",
        "scene_id": "two_boxes_01",
        "_scene_policy": policy,
        "scene_obstacles": [{"label": "sugar_box", "is_target": False}],
    }
    carry = resolve_carry_transport_policy(candidate)
    assert carry["post_pick_skip_carry_front_high"] is True
    entry = resolve_post_pick_transport_entry_target(
        candidate,
        carry,
        default_first_waypoint="carry_front_high",
    )
    assert entry["skip_carry_front_high"] is True
    assert entry["entry_target_waypoint"] == "carry_mid_high"


def test_pick_order_from_scene_policy() -> None:
    policy = _scene02_policy()
    order = resolve_pick_order_from_scene_policy(policy, fallback=[])
    assert order == policy["pick_order"]
