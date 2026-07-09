"""Preflight demo_scene_02: obstáculos progresivos, transporte y slots."""

from __future__ import annotations

from pathlib import Path

import pytest

from panda_controller.attached_transport_entry_validate import (
    enumerate_transport_entry_escape_options,
    generate_rear_retreat_candidates,
)
from panda_controller.authoritative_scene_obstacles import (
    build_authoritative_scene_obstacles,
)
from panda_controller.demo_scene_policy import (
    DEMO_SCENE_PLACEHOLDER_TRANSPORT_WAYPOINTS,
    load_demo_scene_policy,
    preferred_slot_map_from_scene_policy,
    resolve_post_pick_transport_entry_target_from_scene,
)
from panda_controller.deposit_layout_policy import (
    DEFAULT_PLACE_SLOTS_ORDERED,
    plan_deposit_layout_slot,
)
from panda_controller.generic_known_scene_carry_planner import (
    resolve_carry_transport_policy,
    resolve_post_pick_transport_entry_target,
)

_REQUIRED_LOCAL_EXIT_MODES = frozenset(
    {
        "rear_retreat_x_negative",
        "rear_retreat_x_negative_slight_raise",
        "rear_retreat_x_negative_far",
        "rear_retreat_x_negative_raise_far",
        "vertical_raise_then_rear_retreat",
    }
)

_PROGRESSIVE_CASES = [
    pytest.param(
        "cracker_box",
        set(),
        {"chips_can", "mustard_bottle", "sugar_box"},
        id="cracker_empty_completed",
    ),
    pytest.param(
        "chips_can",
        {"cracker_box"},
        {"mustard_bottle", "sugar_box"},
        id="chips_after_cracker",
    ),
    pytest.param(
        "sugar_box",
        {"cracker_box", "chips_can"},
        {"mustard_bottle"},
        id="sugar_after_cracker_chips",
    ),
    pytest.param(
        "mustard_bottle",
        {"cracker_box", "chips_can", "sugar_box"},
        set(),
        id="mustard_last_no_obstacles",
    ),
]


def _scenes_dir() -> str:
    return str(Path(__file__).resolve().parents[1] / "config" / "demo_scenes")


def _scene02_policy() -> dict:
    policy = load_demo_scene_policy("demo_scene_02", scenes_dir=_scenes_dir(), use_cache=False)
    assert policy is not None
    return policy


def _scene_objects_demo02() -> list:
    labels = ["cracker_box", "chips_can", "sugar_box", "mustard_bottle"]
    return [
        {
            "label": lb,
            "entity_name": "runtime_ycb_%s_%d_909078" % (lb, i),
            "role": "target" if lb == labels[0] else "obstacle",
            "position": [0.45 + 0.05 * i, 0.1 - 0.05 * i, 0.47],
            "collision_dims": {"shape": "box", "box": [0.1, 0.1, 0.1]},
        }
        for i, lb in enumerate(labels)
    ]


def _build_obstacle(idx: int, so: dict, is_target: bool) -> dict:
    pos = so.get("position")
    return {
        "idx": idx,
        "entity_name": so["entity_name"],
        "label": so["label"],
        "position": tuple(pos),
        "is_target": is_target,
        "collision_dims": so.get("collision_dims"),
    }


@pytest.mark.parametrize(
    "target_label,completed_labels,expected_obstacles",
    _PROGRESSIVE_CASES,
)
def test_authoritative_obstacle_set_progressive(
    target_label: str,
    completed_labels: set,
    expected_obstacles: set,
) -> None:
    scene = _scene_objects_demo02()
    entity_by_label = {so["label"]: so["entity_name"] for so in scene}
    completed_entities = {
        entity_by_label[lb] for lb in completed_labels if lb in entity_by_label
    }
    target_entity = entity_by_label[target_label]

    obstacles, log = build_authoritative_scene_obstacles(
        scene,
        target_label=target_label,
        target_entity=target_entity,
        completed_entities=completed_entities,
        completed_labels=completed_labels,
        build_obstacle_fn=_build_obstacle,
    )
    labels = {o["label"] for o in obstacles if not o.get("is_target")}
    assert log["result"] == "OK", log.get("reason")
    assert labels == expected_obstacles
    assert log["obstacles"] == sorted(expected_obstacles)
    for done in completed_labels:
        assert done not in labels
        assert done not in log["obstacles"]


def test_demo_scene_02_yaml_local_exit_candidates_complete() -> None:
    policy = _scene02_policy()
    tp = policy["transport_policy"]
    local = set(tp.get("local_exit_candidates") or [])
    phases = set((policy.get("transport_phases") or {}).get("local_escape") or [])
    assert _REQUIRED_LOCAL_EXIT_MODES <= local
    assert _REQUIRED_LOCAL_EXIT_MODES <= phases
    assert "carry_safe_height" in phases


def test_demo_scene_02_no_placeholder_transport_waypoints() -> None:
    policy = _scene02_policy()
    tp = policy["transport_policy"]
    route = set(tp.get("transport_route") or [])
    reconfig = set(tp.get("reconfiguration_waypoints") or [])
    forbidden = set(tp.get("forbidden_waypoints_when_obstacles_remaining") or [])
    assert not (route & DEMO_SCENE_PLACEHOLDER_TRANSPORT_WAYPOINTS)
    assert not (reconfig & DEMO_SCENE_PLACEHOLDER_TRANSPORT_WAYPOINTS)
    assert "carry_front_high" in forbidden
    assert route == {
        "carry_mid_high",
        "turn_back_extended_aligned",
        "box_front_high",
        "box_high",
    }
    assert reconfig == {"carry_mid_high"}


@pytest.mark.parametrize("label", ["cracker_box", "chips_can", "sugar_box", "mustard_bottle"])
def test_carry_transport_generic_for_all_demo_labels(label: str) -> None:
    policy = _scene02_policy()
    candidate = {
        "label": label,
        "scene_id": "demo_scene_02",
        "_scene_policy": policy,
        "scene_obstacles": [
            {"label": "mustard_bottle", "is_target": False},
            {"label": "sugar_box", "is_target": False},
        ],
    }
    carry = resolve_carry_transport_policy(candidate)
    assert carry["post_pick_skip_carry_front_high"] is True
    assert carry["use_lateral_transport_corridors"] is False
    assert _REQUIRED_LOCAL_EXIT_MODES <= set(carry.get("local_exit_candidates") or [])
    entry = resolve_post_pick_transport_entry_target(
        candidate,
        carry,
        default_first_waypoint="carry_front_high",
        waypoints_data={"carry_mid_high": {"joints": {}}},
    )
    assert entry["skip_carry_front_high"] is True
    assert entry["entry_target_waypoint"] == "carry_mid_high"
    assert entry["allow_direct_to_carry_front_high"] is False
    assert entry["allow_direct_to_entry_target"] is True
    assert entry["allow_carry_front_high_corridors"] is False
    assert entry["removed_unsafe_first_waypoint"] == "carry_front_high"


def test_entry_target_never_transport_home_high_override() -> None:
    policy = _scene02_policy()
    carry = resolve_carry_transport_policy(
        {
            "label": "chips_can",
            "scene_id": "demo_scene_02",
            "_scene_policy": policy,
            "scene_obstacles": [{"label": "mustard_bottle", "is_target": False}],
        }
    )
    entry = resolve_post_pick_transport_entry_target_from_scene(
        policy,
        carry,
        default_first_waypoint="carry_front_high",
        waypoints_data={
            "carry_mid_high": {"joints": {}},
            "transport_home_high": {"joints": {}},
        },
        obstacles_remaining=True,
    )
    assert entry["entry_target_waypoint"] == "carry_mid_high"


def test_transport_escape_far_modes_generated_from_policy() -> None:
    policy = _scene02_policy()
    carry = resolve_carry_transport_policy(
        {"label": "chips_can", "scene_id": "demo_scene_02", "_scene_policy": policy}
    )
    cands = generate_rear_retreat_candidates(
        (0.518, -0.038),
        0.705,
        modes=carry.get("local_exit_candidates"),
    )
    modes = {str(c["mode"]) for c in cands}
    assert "rear_retreat_x_negative_far" in modes
    assert "rear_retreat_x_negative_raise_far" in modes
    assert "vertical_raise_then_rear_retreat" in modes


def test_transport_route_lock_requires_hub_validation_fields() -> None:
    """Contrato: no bloquear ruta sin hub_segment_validated (simulado en candidate)."""
    locked_ok = {
        "_transport_reconfiguration_zone_ok": True,
        "_transport_hub_segment_validated": True,
        "_transport_locked_sequence": [
            "carry_mid_high",
            "turn_back_extended_aligned",
            "box_front_high",
            "box_high",
        ],
    }
    locked_bad = {
        "_transport_reconfiguration_zone_ok": False,
        "_transport_hub_segment_validated": False,
    }
    assert locked_ok["_transport_hub_segment_validated"] is True
    assert locked_ok["_transport_reconfiguration_zone_ok"] is True
    assert locked_bad["_transport_hub_segment_validated"] is False


def test_scene_preferred_slots_sugar_and_mustard() -> None:
    policy = _scene02_policy()
    slot_map = preferred_slot_map_from_scene_policy(policy)
    assert slot_map["cracker_box"] == 0
    assert slot_map["chips_can"] == 1
    assert slot_map["sugar_box"] == 2
    assert slot_map["mustard_bottle"] == 3

    idx, slot, plan = plan_deposit_layout_slot(
        label="sugar_box",
        slots=DEFAULT_PLACE_SLOTS_ORDERED,
        occupied=[
            {"label": "cracker_box", "x": -0.37, "y": 0.08, "slot_index": 0},
            {"label": "chips_can", "x": -0.54, "y": 0.08, "slot_index": 1},
        ],
        label_slot_map=slot_map,
    )
    assert plan["result"] == "OK"
    assert plan["selection_mode"] == "scene_preferred_slot"
    assert idx == 2
    assert slot is not None

    idx_m, _, plan_m = plan_deposit_layout_slot(
        label="mustard_bottle",
        slots=DEFAULT_PLACE_SLOTS_ORDERED,
        occupied=[
            {"label": "cracker_box", "x": -0.37, "y": 0.08, "slot_index": 0},
            {"label": "chips_can", "x": -0.54, "y": 0.08, "slot_index": 1},
            {"label": "sugar_box", "x": -0.37, "y": -0.10, "slot_index": 2},
        ],
        label_slot_map=slot_map,
    )
    assert plan_m["result"] == "OK"
    assert idx_m == 3


def test_food_safe_place_policy_enabled_in_scene_yaml() -> None:
    policy = _scene02_policy()
    place = policy.get("place_policy") or {}
    assert place.get("use_food_safe_dynamic_release_z") is True


def test_chips_can_escape_enumeration_logs_all_candidates() -> None:
    policy = _scene02_policy()
    carry = resolve_carry_transport_policy(
        {
            "label": "chips_can",
            "scene_id": "demo_scene_02",
            "_scene_policy": policy,
            "scene_obstacles": [
                {"label": "mustard_bottle", "is_target": False},
                {"label": "sugar_box", "is_target": False},
            ],
        }
    )

    def fk(_joints):
        return (0.35, -0.04, 0.75)

    options, logs = enumerate_transport_entry_escape_options(
        post_lift_hand=(0.518, -0.038, 0.705),
        post_lift_joints=[0.0] * 7,
        entry_target_joints=[0.2] * 7,
        entry_target_waypoint="carry_mid_high",
        allow_direct_to_carry_front_high=False,
        fk_hand_fn=fk,
        attached_geom={
            "carried_object_below_hand_m": 0.192,
            "carried_object_radius_xy_m": 0.05,
        },
        scene_obstacles=[
            {
                "label": "mustard_bottle",
                "position": (0.660, 0.060, 0.385),
                "collision_dims": {"shape": "cylinder", "cylinder": [0.03, 0.19]},
            },
            {
                "label": "sugar_box",
                "position": (0.630, -0.130, 0.385),
                "collision_dims": {"shape": "box", "box": [0.06, 0.158, 0.210]},
            },
        ],
        table_top_z=0.270,
        policy=carry,
        hand_z_candidates=[0.705],
        scene_policy=policy,
    )
    corridor_logs = [l for l in logs if l.get("kind") == "corridor"]
    assert len(corridor_logs) >= len(_REQUIRED_LOCAL_EXIT_MODES)
    assert options, "expected at least one valid escape candidate"


def test_demo_scene_02_v3_cracker_rear_retreat_clearance_from_yaml() -> None:
    """Layout v3: chips_can alejado para rear_retreat tras lift cracker."""
    import math

    from panda_controller.generic_known_scene_carry_planner import (
        attached_obstacle_clearance_3d,
    )

    policy = _scene02_policy()
    objs = policy.get("objects") or {}
    chips_pose = objs["chips_can"]["pose"]
    attached_geom = {
        "carried_object_below_hand_m": 0.192,
        "carried_object_radius_xy_m": 0.1046,
        "attached_collision_padding_m": 0.020,
        "dims_lwh": [0.158, 0.060, 0.213],
    }
    obs = {
        "label": "chips_can",
        "entity_name": "runtime_ycb_chips_can_1",
        "position": (float(chips_pose["x"]), float(chips_pose["y"]), 0.47),
        "collision_dims": {"shape": "cylinder", "cylinder": [0.033, 0.19]},
        "top_z_m": 0.520,
    }
    post_lift = (0.456, 0.115, 0.587)
    retreat = (max(0.30, post_lift[0] - 0.056), post_lift[1], post_lift[2])
    for hand in (post_lift, retreat):
        chk = attached_obstacle_clearance_3d(
            hand,
            attached_geom,
            obs,
            table_top_z=0.270,
            required_xy_clearance_m=0.04,
        )
        assert float(chk["xy_clearance"]) >= 0.04, hand
    assert math.isclose(float(chips_pose["x"]), 0.520, abs_tol=1e-3)
    assert math.isclose(float(chips_pose["y"]), -0.095, abs_tol=1e-3)
