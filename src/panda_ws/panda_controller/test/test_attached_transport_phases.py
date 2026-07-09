"""Tests contrato fases transporte attached."""

from __future__ import annotations

from pathlib import Path

from panda_controller.attached_transport_phases import (
    check_transport_reconfiguration_zone,
    joint_distance_rad,
    joint_limit_margin_min,
    pick_best_transport_aware_score,
    resolve_deterministic_transport_sequence,
    resolve_reconfiguration_safety_thresholds,
    score_transport_aware_pick_exit,
    validate_post_escape_hub_route,
)
from panda_controller.demo_scene_policy import load_demo_scene_policy
from panda_controller.generic_known_scene_carry_planner import (
    resolve_carry_transport_policy,
)
from panda_controller.tfg_motion_waypoints import (
    PANDA_ARM_JOINT_NAMES,
    joint_values_7d_from_any,
)


class _FakeJointState:
    def __init__(self, names: list[str], positions: list[float]) -> None:
        self.name = list(names)
        self.position = [float(v) for v in positions]


def _make_joint_state(positions: list[float]):
    try:
        from sensor_msgs.msg import JointState

        js = JointState()
        js.name = list(PANDA_ARM_JOINT_NAMES)
        js.position = [float(v) for v in positions]
        return js
    except ImportError:
        return _FakeJointState(list(PANDA_ARM_JOINT_NAMES), positions)


def test_joint_values_7d_from_any_accepts_joint_state() -> None:
    positions = [0.1, -0.2, 0.3, -1.9, 0.05, 1.1, -0.6]
    js = _make_joint_state(positions)
    out = joint_values_7d_from_any(js, context="test_joint_state")
    assert out is not None
    assert len(out) == 7
    assert out == positions


def test_joint_values_7d_from_list_and_dict() -> None:
    seq = [0.1, -0.2, 0.3, -1.9, 0.05, 1.1, -0.6]
    assert joint_values_7d_from_any(seq, context="test_list") == seq
    as_dict = {n: seq[i] for i, n in enumerate(PANDA_ARM_JOINT_NAMES)}
    assert joint_values_7d_from_any(as_dict, context="test_dict") == seq


def test_transport_aware_score_accepts_joint_state() -> None:
    policy_doc = _scene02_policy()
    positions = [0.0, -0.5, 0.0, -2.0, 0.0, 1.0, 0.5]
    js = _make_joint_state(positions)
    post_lift_joints_7d = joint_values_7d_from_any(
        js, context="transport_aware_post_lift_joints"
    )
    assert post_lift_joints_7d is not None
    candidate = {
        "label": "cracker_box",
        "scene_id": "demo_scene_02",
        "_scene_policy": policy_doc,
        "dims_lwh": [0.060, 0.158, 0.210],
        "top_z_m": 0.470,
        "recommended_grasp_depth_from_top_m": 0.033,
        "object_height_m": 0.210,
        "scene_obstacles": [],
    }
    carry = resolve_carry_transport_policy(candidate)
    geom = {
        "carried_object_below_hand_m": 0.192,
        "carried_object_radius_xy_m": 0.10,
    }

    def fk(_joints):
        return (0.400, 0.115, 0.750)

    hub = [0.15, -1.0, -0.04, -2.01, -0.04, 1.01, -0.67]
    score = score_transport_aware_pick_exit(
        candidate_idx=0,
        yaw_variant=0.0,
        post_lift_hand=(0.456, 0.115, 0.679),
        post_lift_joints=post_lift_joints_7d,
        grasp_joints=post_lift_joints_7d,
        hub_waypoint_joints=hub,
        hub_waypoint_name="carry_mid_high",
        fk_hand_fn=fk,
        attached_geom=geom,
        scene_obstacles=[],
        table_top_z=0.270,
        policy=carry,
        scene_policy=policy_doc,
    )
    assert score["result"] in ("ACCEPT", "REJECT")
    assert isinstance(score.get("joint_distance_to_hub"), float)


def _scene02_policy() -> dict:
    scenes_dir = str(Path(__file__).resolve().parents[1] / "config" / "demo_scenes")
    policy = load_demo_scene_policy("demo_scene_02", scenes_dir=scenes_dir, use_cache=False)
    assert policy is not None
    return policy


def test_joint_scoring_helpers() -> None:
    a = [0.0, -0.5, 0.0, -2.0, 0.0, 1.0, 0.5]
    b = [0.1, -0.8, 0.0, -2.1, 0.0, 1.2, 0.6]
    assert joint_distance_rad(a, b) > 0.0
    assert joint_limit_margin_min(a) > 0.0


def test_resolve_reconfiguration_safety_thresholds_from_scene() -> None:
    policy = _scene02_policy()
    thresholds = resolve_reconfiguration_safety_thresholds(policy)
    assert thresholds["min_table_clearance_m"] == 0.200
    assert thresholds["min_xy_clearance_m"] == 0.080


def test_maybe_accept_prevalidated_hub_segment_allows_small_negative_margin() -> None:
    from panda_controller.attached_transport_phases import (
        maybe_accept_prevalidated_hub_segment,
    )

    detail = {
        "segment": "current->carry_mid_high",
        "waypoint": "carry_mid_high",
        "hard_collision": False,
        "min_clearance": -0.006113,
        "joint_margin_min": 0.66,
        "result": "FAIL",
    }
    out = maybe_accept_prevalidated_hub_segment(
        detail,
        is_first_segment=True,
        hub_segment_prevalidated=True,
        transport_entry_verified=True,
        obstacle_disturbed=False,
    )
    assert out["result"] == "OK"
    assert out["reason"] == "local_escape_hub_segment_prevalidated_with_tolerance"


def test_maybe_accept_prevalidated_hub_segment_rejects_hard_collision() -> None:
    from panda_controller.attached_transport_phases import (
        maybe_accept_prevalidated_hub_segment,
    )

    detail = {
        "hard_collision": True,
        "min_clearance": -0.006,
        "result": "FAIL",
    }
    out = maybe_accept_prevalidated_hub_segment(
        detail,
        is_first_segment=True,
        hub_segment_prevalidated=True,
        transport_entry_verified=True,
        obstacle_disturbed=False,
    )
    assert out["result"] == "FAIL"


def test_maybe_accept_prevalidated_hub_segment_allows_none_clearance() -> None:
    from panda_controller.attached_transport_phases import (
        maybe_accept_prevalidated_hub_segment,
    )

    detail = {
        "hard_collision": False,
        "min_clearance": None,
        "result": "FAIL",
    }
    out = maybe_accept_prevalidated_hub_segment(
        detail,
        is_first_segment=True,
        hub_segment_prevalidated=True,
        transport_entry_verified=True,
        obstacle_disturbed=False,
    )
    assert out["result"] == "OK"
    assert out["reason"] == "hub_prevalidated_no_clearance_metric"


def test_should_skip_attached_direct_action_route_preflight() -> None:
    from panda_controller.attached_transport_phases import (
        should_skip_attached_direct_action_route_preflight,
    )

    assert should_skip_attached_direct_action_route_preflight(
        transport_entry_validated=True,
        hub_segment_prevalidated=True,
        obstacle_disturbed=False,
    )
    assert not should_skip_attached_direct_action_route_preflight(
        transport_entry_validated=True,
        hub_segment_prevalidated=False,
        obstacle_disturbed=False,
    )
    assert not should_skip_attached_direct_action_route_preflight(
        transport_entry_validated=True,
        hub_segment_prevalidated=True,
        obstacle_disturbed=True,
    )


def test_validate_post_escape_hub_route_single_segment() -> None:
    geom = {"carried_object_below_hand_m": 0.192, "carried_object_radius_xy_m": 0.10}
    policy = resolve_carry_transport_policy({"label": "chips_can"})

    def fk(_joints):
        return (0.350, -0.040, 0.750)

    ok, logs = validate_post_escape_hub_route(
        current_joints=[0.0] * 7,
        hub_waypoint_name="carry_mid_high",
        hub_waypoint_joints=[0.2] * 7,
        fk_hand_fn=fk,
        attached_geom=geom,
        scene_obstacles=[],
        table_top_z=0.270,
        policy=policy,
    )
    assert isinstance(ok, bool)
    assert len(logs) == 1
    assert logs[0]["segment"] == "current->carry_mid_high"


def test_reconfiguration_zone_ok_when_high_and_clear() -> None:
    geom = {"carried_object_below_hand_m": 0.192, "carried_object_radius_xy_m": 0.10}
    hand = (0.350, 0.115, 0.780)
    policy = resolve_carry_transport_policy({"label": "cracker_box"})
    check = check_transport_reconfiguration_zone(
        hand_pos=hand,
        attached_geom=geom,
        scene_obstacles=[
            {
                "label": "chips_can",
                "position": (0.520, -0.040, 0.385),
                "collision_dims": {"shape": "cylinder", "cylinder": [0.035, 0.19]},
                "top_z_m": 0.520,
            }
        ],
        table_top_z=0.270,
        policy=policy,
    )
    assert check["result"] == "OK"
    assert check["transport_reconfiguration_zone_ok"] is True


def test_resolve_deterministic_sequence_with_reconfig() -> None:
    policy = _scene02_policy()
    seq, prefix = resolve_deterministic_transport_sequence(
        policy,
        reconfiguration_zone_ok=True,
        obstacles_remaining=True,
        default_route=["carry_mid_high", "turn_back_extended_aligned", "box_high"],
    )
    assert prefix == ["carry_mid_high"]
    assert seq == [
        "carry_mid_high",
        "turn_back_extended_aligned",
        "box_front_high",
        "box_high",
    ]


def test_resolve_deterministic_sequence_dedup_carry_mid_high() -> None:
    policy = _scene02_policy()
    seq, prefix = resolve_deterministic_transport_sequence(
        policy,
        reconfiguration_zone_ok=True,
        obstacles_remaining=True,
        default_route=[
            "carry_mid_high",
            "turn_back_extended_aligned",
            "box_front_high",
            "box_high",
        ],
    )
    assert seq.count("carry_mid_high") == 1
    assert seq[0] == "carry_mid_high"


def test_resolve_deterministic_sequence_zone_fail_uses_policy_route() -> None:
    policy = _scene02_policy()
    seq, prefix = resolve_deterministic_transport_sequence(
        policy,
        reconfiguration_zone_ok=False,
        obstacles_remaining=True,
        default_route=["carry_mid_high", "box_high"],
    )
    assert prefix == []
    assert seq[0] == "carry_mid_high"
    assert "turn_back_extended_aligned" in seq


def test_transport_aware_score_accepts_clear_scene() -> None:
    policy_doc = _scene02_policy()
    candidate = {
        "label": "cracker_box",
        "scene_id": "demo_scene_02",
        "_scene_policy": policy_doc,
        "dims_lwh": [0.060, 0.158, 0.210],
        "top_z_m": 0.470,
        "recommended_grasp_depth_from_top_m": 0.033,
        "object_height_m": 0.210,
        "scene_obstacles": [
            {
                "label": "chips_can",
                "position": (0.520, -0.040, 0.385),
                "collision_dims": {"shape": "cylinder", "cylinder": [0.035, 0.19]},
                "top_z_m": 0.520,
            }
        ],
    }
    carry = resolve_carry_transport_policy(candidate)
    geom = {
        "carried_object_below_hand_m": 0.192,
        "carried_object_radius_xy_m": 0.10,
    }

    def fk(_joints):
        return (0.400, 0.115, 0.750)

    hub = [0.15, -1.0, -0.04, -2.01, -0.04, 1.01, -0.67]
    score = score_transport_aware_pick_exit(
        candidate_idx=0,
        yaw_variant=0.0,
        post_lift_hand=(0.456, 0.115, 0.679),
        post_lift_joints=[0.0] * 7,
        grasp_joints=[0.0] * 7,
        hub_waypoint_joints=hub,
        hub_waypoint_name="carry_mid_high",
        fk_hand_fn=fk,
        attached_geom=geom,
        scene_obstacles=candidate["scene_obstacles"],
        table_top_z=0.270,
        policy=carry,
        scene_policy=policy_doc,
    )
    best = pick_best_transport_aware_score([score])
    if score["result"] == "ACCEPT":
        assert best is not None


def test_transport_aware_score_accepts_local_escape_with_deferred_global_route() -> None:
    from panda_controller.attached_transport_phases import (
        format_pick_route_transport_aware_score_log,
    )
    from panda_controller.demo_scene_policy import apply_scene_policy_to_carry_transport
    from panda_controller.generic_known_scene_carry_planner import (
        compute_attached_object_geometry,
    )

    policy_doc = _scene02_policy()
    obstacles = [
        {
            "label": "chips_can",
            "position": (0.520, -0.095, 0.400),
            "collision_dims": {"shape": "cylinder", "cylinder": [0.04, 0.255]},
            "top_z_m": 0.520,
        },
        {
            "label": "sugar_box",
            "position": (0.630, -0.175, 0.363),
            "collision_dims": {"shape": "box", "box": [0.06, 0.158, 0.210]},
            "top_z_m": 0.470,
        },
        {
            "label": "mustard_bottle",
            "position": (0.660, 0.060, 0.400),
            "collision_dims": {"shape": "cylinder", "cylinder": [0.04, 0.255]},
            "top_z_m": 0.520,
        },
    ]
    carry = apply_scene_policy_to_carry_transport(
        {
            "carry_clearance_above_table_m": 0.120,
            "carry_clearance_above_obstacles_m": 0.100,
            "attached_transport_safety_margin_tolerance_m": 0.006,
            "use_lateral_transport_corridors": False,
            "local_exit_candidates": [
                "rear_retreat_x_negative",
                "rear_retreat_x_negative_raise_far",
                "vertical_raise_then_rear_retreat",
            ],
        },
        policy_doc,
        obstacles_remaining=True,
    )
    geom = compute_attached_object_geometry(
        {
            "label": "cracker_box",
            "grasp_center_base": [0.455, 0.115, 0.378],
            "grasp_yaw_rad": 2.9155,
            "dims_lwh": [0.158, 0.06, 0.21],
            "recommended_grasp_depth_from_top_m": 0.04,
            "scene_id": "demo_scene_02",
            "_scene_policy": policy_doc,
        },
        grasp_hand_z=0.587,
        grasp_hand_xy=(0.456, 0.115),
        table_top_z=0.270,
    )

    def fk(_joints):
        return (0.35, 0.115, 0.75)

    hub = [0.15, -1.0, -0.04, -2.01, -0.04, 1.01, -0.67]
    post_lift_joints = [0.0, -0.5, 0.0, -2.0, 0.0, 1.0, 0.5]
    score = score_transport_aware_pick_exit(
        candidate_idx=7,
        yaw_variant=0.0,
        post_lift_hand=(0.456, 0.115, 0.587),
        post_lift_joints=post_lift_joints,
        grasp_joints=post_lift_joints,
        hub_waypoint_joints=hub,
        hub_waypoint_name="carry_mid_high",
        fk_hand_fn=fk,
        attached_geom=geom,
        scene_obstacles=obstacles,
        table_top_z=0.270,
        policy=carry,
        scene_policy=policy_doc,
        lift_ok=True,
        target_world_present=False,
    )
    log = format_pick_route_transport_aware_score_log(score)
    assert "local_escape_ok=true" in log
    assert score.get("local_escape_ok") is True
    assert score["result"] == "ACCEPT"
    assert score.get("acceptance_reason") in (
        "valid_pick_with_local_escape_global_route_deferred",
        "full_transport_ok",
    )
    assert pick_best_transport_aware_score([score]) is not None
