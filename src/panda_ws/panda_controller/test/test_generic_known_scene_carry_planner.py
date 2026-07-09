"""Tests GenericKnownSceneCarryPlanner."""

from __future__ import annotations

import math

from panda_controller.generic_known_scene_carry_planner import (
    attached_obstacle_clearance_3d,
    compute_carry_safe_tcp_z,
    generate_transport_entry_candidates,
    resolve_carry_transport_policy,
    validate_attached_hand_pose,
    validate_attached_joint_segment,
)


def _cracker_candidate() -> dict:
    return {
        "label": "cracker_box",
        "dims_lwh": [0.060, 0.158, 0.210],
        "top_z_m": 0.470,
        "recommended_grasp_depth_from_top_m": 0.033,
        "object_height_m": 0.210,
    }


def test_carry_safe_tcp_z_at_least_min_for_cracker() -> None:
    policy = resolve_carry_transport_policy(_cracker_candidate())
    geom = {
        "carried_object_below_tcp_m": 0.190,
        "carried_object_radius_xy_m": 0.100,
    }
    obstacles = [
        {
            "label": "chips_can",
            "position": (0.520, -0.040, 0.385),
            "top_z_m": 0.470,
            "collision_dims": {"shape": "cylinder", "cylinder": [0.0375, 0.25]},
        }
    ]
    carry_z, _detail = compute_carry_safe_tcp_z(
        policy=policy,
        attached_geom=geom,
        table_top_z=0.428,
        remaining_obstacles=obstacles,
        current_tcp_z=0.570,
    )
    assert carry_z >= 0.700 - 1e-6


def test_attached_segment_rejects_low_tcp_near_table() -> None:
    def fk(_joints):
        return (0.45, 0.12, 0.55)

    ok, metrics, _checks = validate_attached_joint_segment(
        [0.0] * 7,
        [0.1] * 7,
        fk_hand_fn=fk,
        attached_geom={"carried_object_below_hand_m": 0.18, "carried_object_radius_xy_m": 0.09},
        table_top_z=0.428,
        obstacles=[],
        min_table_clearance_m=0.120,
        required_xy_clearance_m=0.020,
        n_samples=4,
    )
    assert not ok
    assert metrics.get("reason") == "attached_bottom_near_table"


def test_attached_segment_ok_when_high_enough() -> None:
    z = 0.75

    def fk(_joints):
        return (0.35, -0.20, z)

    ok, metrics, _checks = validate_attached_joint_segment(
        [0.0] * 7,
        [0.2] * 7,
        fk_hand_fn=fk,
        attached_geom={"carried_object_below_hand_m": 0.18, "carried_object_radius_xy_m": 0.09},
        table_top_z=0.428,
        obstacles=[
            {
                "label": "chips_can",
                "position": (0.520, -0.040, 0.385),
                "collision_dims": {"shape": "cylinder", "cylinder": [0.0375, 0.25]},
            }
        ],
        min_table_clearance_m=0.05,
        required_xy_clearance_m=0.01,
        n_samples=4,
    )
    assert ok, metrics
    assert float(metrics["min_clearance_to_table"]) > 0.05


def test_clearance_3d_far_mustard_is_ok_not_collision() -> None:
    geom = {"carried_object_below_hand_m": 0.18, "carried_object_radius_xy_m": 0.09}
    hand = (0.456, 0.115, 0.743)
    obs = {
        "label": "mustard_bottle",
        "entity_name": "runtime_ycb_mustard_bottle_3",
        "position": (0.620, 0.180, 0.385),
        "collision_dims": {"shape": "cylinder", "cylinder": [0.03, 0.19]},
        "top_z_m": 0.520,
    }
    chk = attached_obstacle_clearance_3d(
        hand,
        geom,
        obs,
        table_top_z=0.428,
        required_xy_clearance_m=0.10,
    )
    assert chk["result"] in ("OK", "NEAR")
    assert not chk["hard_collision"]


def test_transport_entry_candidates_have_real_delta_xy() -> None:
    policy = resolve_carry_transport_policy(_cracker_candidate())
    cands = generate_transport_entry_candidates((0.456, 0.115), 0.743, policy)
    assert len(cands) >= 4
    assert any(float(c["delta_xy_from_current"]) >= 0.08 for c in cands)


def test_adaptive_policy_keeps_current_height_when_clearance_ok() -> None:
    candidate = _cracker_candidate()
    candidate["scene_id"] = "demo_scene_02"
    hand = (0.455, 0.115, 0.679)
    tcp = (0.455, 0.115, 0.579)
    geom = {
        "carried_object_below_hand_m": 0.192,
        "carried_object_radius_xy_m": 0.100,
    }
    obstacles = [
        {
            "label": "chips_can",
            "position": (0.520, -0.040, 0.385),
            "collision_dims": {"shape": "cylinder", "cylinder": [0.0375, 0.25]},
            "top_z_m": 0.470,
        },
        {
            "label": "mustard_bottle",
            "position": (0.620, 0.180, 0.385),
            "collision_dims": {"shape": "cylinder", "cylinder": [0.03, 0.19]},
            "top_z_m": 0.520,
        },
    ]
    from panda_controller.generic_known_scene_carry_planner import (
        compute_attached_object_geometry,
        resolve_adaptive_carry_height_policy,
    )

    attached_geom = compute_attached_object_geometry(
        candidate, grasp_hand_z=hand[2], table_top_z=0.270
    )
    adaptive = resolve_adaptive_carry_height_policy(
        candidate=candidate,
        current_hand=hand,
        current_tcp=tcp,
        attached_geom=attached_geom,
        table_top_z=0.270,
        scene_obstacles=obstacles,
    )
    assert adaptive["current_pose_clearance_ok"] is True
    assert abs(float(adaptive["selected_hand_z"]) - 0.679) < 1e-3
    assert adaptive["global_height_required"] is False
    assert float(adaptive["global_over_obstacles_height"]) > 0.80
    assert float(adaptive["max_carry_hand_z"]) <= 0.801
    assert 0.679 in adaptive["hand_z_candidates"]


def test_adaptive_hand_z_candidates_incremental() -> None:
    from panda_controller.generic_known_scene_carry_planner import (
        build_adaptive_hand_z_candidates,
        resolve_carry_transport_policy,
    )

    policy = resolve_carry_transport_policy({"label": "cracker_box", "scene_id": "demo_scene_02"})
    geom = {"carried_object_below_hand_m": 0.192}
    cands = build_adaptive_hand_z_candidates(0.679, policy, geom, 0.270)
    assert cands[0] == 0.679
    assert any(abs(z - 0.709) < 1e-3 for z in cands)
    assert all(z <= 0.800 + 1e-6 for z in cands)


def test_validate_hand_pose_logs_near_vs_collision() -> None:
    geom = {"carried_object_below_hand_m": 0.18, "carried_object_radius_xy_m": 0.09}
    hand = (0.456, 0.115, 0.743)
    obs = {
        "label": "mustard_bottle",
        "position": (0.620, 0.180, 0.385),
        "collision_dims": {"shape": "cylinder", "cylinder": [0.03, 0.19]},
        "top_z_m": 0.520,
    }
    ok, checks, _metrics = validate_attached_hand_pose(
        hand,
        geom,
        [obs],
        table_top_z=0.428,
        min_table_clearance_m=0.05,
        required_xy_clearance_m=0.10,
    )
    assert len(checks) == 1
    assert checks[0]["xy_center_distance"] > 0.1
    assert isinstance(ok, bool)
