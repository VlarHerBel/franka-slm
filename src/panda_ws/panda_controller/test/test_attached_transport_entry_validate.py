"""Tests validación virtual transport_entry."""

from __future__ import annotations

from pathlib import Path

from panda_controller.attached_transport_entry_validate import (
    closest_obstacle_check_from_segment_checks,
    decide_attached_transport_preflight,
    enumerate_transport_entry_escape_options,
    format_transport_exit_clearance_breakdown_log,
    generate_rear_retreat_candidates,
    select_transport_entry_validate_only,
    transport_escape_option_fingerprint,
)
from panda_controller.demo_scene_policy import load_demo_scene_policy
from panda_controller.generic_known_scene_carry_planner import (
    DEFAULT_ATTACHED_TRANSPORT_SAFETY_MARGIN_TOLERANCE_M,
    compute_attached_object_geometry,
    resolve_carry_transport_policy,
    resolve_post_pick_transport_entry_target,
    should_defer_transport_entry_hub_to_deterministic,
)


def _scene02_candidate() -> dict:
    scenes_dir = str(Path(__file__).resolve().parents[1] / "config" / "demo_scenes")
    policy = load_demo_scene_policy("demo_scene_02", scenes_dir=scenes_dir, use_cache=False)
    assert policy is not None
    return {
        "label": "cracker_box",
        "scene_id": "demo_scene_02",
        "_scene_policy": policy,
        "scene_obstacles": [{"label": "chips_can", "is_target": False}],
    }


def test_preflight_fail_on_negative_safety_margin() -> None:
    decision = decide_attached_transport_preflight(
        True,
        {"min_safety_margin_m": -0.12, "min_geometric_xy_clearance_m": -0.12},
        [{"hard_collision": False, "safety_margin_ok": False, "result": "NEAR"}],
    )
    assert decision["decision"] == "FAIL"
    assert decision["reason"] == "safety_margin_insufficient"


def test_preflight_allow_borderline_within_tolerance() -> None:
    decision = decide_attached_transport_preflight(
        False,
        {"min_safety_margin_m": -0.0025, "min_geometric_xy_clearance_m": 0.01},
        [{"hard_collision": False, "safety_margin_ok": False, "result": "NEAR"}],
        tolerance_m=0.006,
    )
    assert decision["decision"] == "ALLOW_BORDERLINE"
    assert decision["reason"] == "within_safety_margin_tolerance"


def test_rear_retreat_candidate_coordinates() -> None:
    cands = generate_rear_retreat_candidates((0.456, 0.115), 0.679)
    assert len(cands) == 9
    assert cands[0]["candidate_hand"] == (0.400, 0.115, 0.679)
    assert cands[1]["candidate_hand"][0] == 0.360
    assert cands[1]["candidate_hand"][1] == 0.115
    assert abs(cands[1]["candidate_hand"][2] - 0.699) < 1e-6
    assert abs(cands[2]["candidate_hand"][2] - 0.719) < 1e-6
    far_modes = [str(c["mode"]) for c in cands]
    assert "rear_retreat_x_negative_far" in far_modes
    assert "rear_retreat_x_negative_raise_far" in far_modes
    assert "vertical_raise_then_rear_retreat" in far_modes


def test_rear_retreat_far_modes_respect_policy_filter() -> None:
    cands = generate_rear_retreat_candidates(
        (0.520, -0.040),
        0.720,
        modes=[
            "rear_retreat_x_negative",
            "vertical_raise_then_rear_retreat",
            "rear_retreat_x_negative_far",
        ],
    )
    modes = [str(c["mode"]) for c in cands]
    assert modes.count("rear_retreat_x_negative") == 1
    assert modes.count("rear_retreat_x_negative_far") == 1
    assert modes.count("vertical_raise_then_rear_retreat") == 3
    assert cands[-1]["candidate_hand"][0] == 0.394  # max(0.30, 0.520 - 0.126)


def test_escape_fingerprint_distinguishes_poses() -> None:
    a = {"mode": "rear_retreat_x_negative", "candidate_hand": (0.40, 0.11, 0.68)}
    b = {"mode": "rear_retreat_x_negative_far", "candidate_hand": (0.30, 0.11, 0.68)}
    assert transport_escape_option_fingerprint(a) != transport_escape_option_fingerprint(b)


def test_enumerate_prefers_zone_ok_candidate_for_chips_can_cluster() -> None:
    candidate = _scene02_candidate()
    candidate["label"] = "chips_can"
    carry = resolve_carry_transport_policy(candidate)
    post_lift = (0.518, -0.038, 0.705)
    obstacles = [
        {
            "label": "mustard_bottle",
            "position": (0.660, 0.060, 0.385),
            "collision_dims": {"shape": "cylinder", "cylinder": [0.03, 0.19]},
            "top_z_m": 0.520,
        },
        {
            "label": "sugar_box",
            "position": (0.630, -0.130, 0.385),
            "collision_dims": {"shape": "box", "box": [0.06, 0.158, 0.210]},
            "top_z_m": 0.470,
        },
    ]

    def fk(_joints):
        return (0.35, -0.04, 0.75)

    options, logs = enumerate_transport_entry_escape_options(
        post_lift_hand=post_lift,
        post_lift_joints=[0.0] * 7,
        entry_target_joints=[0.2] * 7,
        entry_target_waypoint="carry_mid_high",
        allow_direct_to_carry_front_high=False,
        fk_hand_fn=fk,
        attached_geom={
            "carried_object_below_hand_m": 0.192,
            "carried_object_radius_xy_m": 0.05,
        },
        scene_obstacles=obstacles,
        table_top_z=0.270,
        policy=carry,
        hand_z_candidates=[0.705],
        scene_policy=candidate["_scene_policy"],
    )
    assert options, "expected at least one corridor candidate"
    corridor_logs = [l for l in logs if l.get("kind") == "corridor"]
    assert len(corridor_logs) >= 3
    plain_near_logs = [
        l for l in corridor_logs if l.get("mode") == "rear_retreat_x_negative"
    ]
    if plain_near_logs:
        log0 = plain_near_logs[0]
        local_ok = bool(log0.get("local_escape_ok"))
        assert (log0.get("result") == "OK") == local_ok
    selected, _ = select_transport_entry_validate_only(
        post_lift_hand=post_lift,
        post_lift_joints=[0.0] * 7,
        entry_target_joints=[0.2] * 7,
        entry_target_waypoint="carry_mid_high",
        allow_direct_to_carry_front_high=False,
        fk_hand_fn=fk,
        attached_geom={
            "carried_object_below_hand_m": 0.192,
            "carried_object_radius_xy_m": 0.05,
        },
        scene_obstacles=obstacles,
        table_top_z=0.270,
        policy=carry,
        hand_z_candidates=[0.705],
        scene_policy=candidate["_scene_policy"],
    )
    assert selected is not None
    zone_ok_modes = [o["mode"] for o in options if o.get("zone_ok")]
    if zone_ok_modes:
        assert selected["mode"] in zone_ok_modes
        assert selected["mode"] != "rear_retreat_x_negative"


def test_select_tries_rear_retreat_after_direct_fail() -> None:
    post_lift = (0.456, 0.115, 0.679)
    first_hand = (0.35, 0.115, 0.75)

    def fk(joints):
        t = float(joints[0]) if joints else 0.0
        return (
            post_lift[0] + t * (first_hand[0] - post_lift[0]),
            post_lift[1],
            post_lift[2] + t * (first_hand[2] - post_lift[2]),
        )

    geom = {"carried_object_below_hand_m": 0.192, "carried_object_radius_xy_m": 0.10}
    obstacles = [
        {
            "label": "chips_can",
            "position": (0.420, 0.115, 0.385),
            "collision_dims": {"shape": "cylinder", "cylinder": [0.035, 0.19]},
            "top_z_m": 0.520,
        },
    ]
    selected, logs = select_transport_entry_validate_only(
        post_lift_hand=post_lift,
        post_lift_joints=[0.0] * 7,
        entry_target_joints=[1.0] * 7,
        entry_target_waypoint="carry_front_high",
        allow_direct_to_carry_front_high=True,
        fk_hand_fn=fk,
        attached_geom=geom,
        scene_obstacles=obstacles,
        table_top_z=0.270,
        policy={
            "carry_clearance_above_table_m": 0.120,
            "carry_clearance_above_obstacles_m": 0.100,
            "transport_exit_lane_y_m": -0.32,
            "attached_transport_safety_margin_tolerance_m": 0.006,
            "use_lateral_transport_corridors": False,
        },
        hand_z_candidates=[0.679],
    )
    direct_logs = [l for l in logs if l.get("kind") == "direct"]
    corridor_logs = [l for l in logs if l.get("kind") == "corridor"]
    assert direct_logs, "expected direct validation attempt"
    assert corridor_logs, "expected corridor validation after direct fail"
    rear_modes = [str(l.get("mode", "")) for l in corridor_logs]
    assert any("rear_retreat" in m for m in rear_modes)
    assert not any("front_lane" in m for m in rear_modes)
    if selected is not None:
        assert "rear_retreat" in selected["mode"] or selected["mode"] == "direct_to_carry_front_high"


def test_demo_scene_02_cracker_disables_lateral_corridors() -> None:
    candidate = _scene02_candidate()
    policy = resolve_carry_transport_policy(candidate)
    assert policy["use_lateral_transport_corridors"] is False
    assert policy["post_pick_skip_carry_front_high"] is True
    assert policy["first_transport_waypoint_after_rear_retreat"] == "carry_mid_high"
    assert (
        policy["attached_transport_safety_margin_tolerance_m"]
        == DEFAULT_ATTACHED_TRANSPORT_SAFETY_MARGIN_TOLERANCE_M
    )


def test_demo_cracker_entry_target_skips_carry_front_high() -> None:
    candidate = _scene02_candidate()
    carry = resolve_carry_transport_policy(candidate)
    cfg = resolve_post_pick_transport_entry_target(
        candidate,
        carry,
        default_first_waypoint="carry_front_high",
        waypoints_data={},
    )
    assert cfg["skip_carry_front_high"] is True
    assert cfg["entry_target_waypoint"] == "carry_mid_high"
    assert cfg["allow_direct_to_carry_front_high"] is False
    assert cfg["removed_unsafe_first_waypoint"] == "carry_front_high"
    assert cfg["defer_entry_hub_to_deterministic_transport"] is True


def test_defer_hub_for_demo_cracker_rear_retreat() -> None:
    candidate = _scene02_candidate()
    carry = resolve_carry_transport_policy(candidate)
    cfg = resolve_post_pick_transport_entry_target(
        candidate,
        carry,
        default_first_waypoint="carry_front_high",
        waypoints_data={},
    )
    assert should_defer_transport_entry_hub_to_deterministic(
        cfg, "rear_retreat_x_negative"
    )
    assert not should_defer_transport_entry_hub_to_deterministic(
        cfg, "direct_to_carry_front_high"
    )


def test_select_direct_when_joint_segment_clear() -> None:
    def fk(_joints):
        return (0.35, -0.20, 0.75)

    geom = {"carried_object_below_hand_m": 0.192, "carried_object_radius_xy_m": 0.10}
    obstacles = [
        {
            "label": "mustard_bottle",
            "position": (0.620, 0.180, 0.385),
            "collision_dims": {"shape": "cylinder", "cylinder": [0.03, 0.19]},
            "top_z_m": 0.520,
        }
    ]
    selected, logs = select_transport_entry_validate_only(
        post_lift_hand=(0.456, 0.116, 0.675),
        post_lift_joints=[0.0] * 7,
        entry_target_joints=[0.2] * 7,
        entry_target_waypoint="carry_front_high",
        allow_direct_to_carry_front_high=True,
        fk_hand_fn=fk,
        attached_geom=geom,
        scene_obstacles=obstacles,
        table_top_z=0.270,
        policy={
            "carry_clearance_above_table_m": 0.120,
            "carry_clearance_above_obstacles_m": 0.100,
            "transport_exit_lane_y_m": -0.32,
            "attached_transport_safety_margin_tolerance_m": 0.006,
            "use_lateral_transport_corridors": False,
            "local_exit_candidates": [],
        },
        hand_z_candidates=[0.675],
    )
    assert selected is not None
    direct_logs = [entry for entry in logs if entry.get("kind") == "direct"]
    assert direct_logs, "expected direct validation attempt"
    assert direct_logs[0]["result"] == "OK"
    assert selected["mode"] in (
        "direct_to_carry_front_high",
        "carry_front_entry_mid",
    )


def test_transport_exit_clearance_breakdown_log_format() -> None:
    attached_geom = {
        "dims_lwh": [0.158, 0.060, 0.213],
        "carried_object_radius_xy_m": 0.1046,
        "attached_collision_padding_m": 0.020,
    }
    checks = [
        {
            "obstacle_label": "chips_can",
            "obstacle_entity": "runtime_ycb_chips_can_1",
            "xy_clearance": 0.007,
            "obstacle_dims": {"shape": "cylinder", "cylinder": [0.033, 0.19]},
            "result": "NEAR",
        },
        {
            "obstacle_label": "mustard_bottle",
            "obstacle_entity": "runtime_ycb_mustard_bottle_3",
            "xy_clearance": 0.082,
            "result": "OK",
        },
    ]
    closest = closest_obstacle_check_from_segment_checks(checks)
    assert closest is not None
    assert closest["obstacle_label"] == "chips_can"
    log = format_transport_exit_clearance_breakdown_log(
        candidate_idx=18,
        exit_name="rear_retreat_x_negative",
        closest_check=closest,
        required_clearance=0.10,
        attached_geom=attached_geom,
        reason="safety_margin_insufficient",
    )
    assert "[TRANSPORT_EXIT_CLEARANCE_BREAKDOWN]" in log
    assert "closest_obstacle_label=chips_can" in log
    assert "min_obstacle_distance=0.0070" in log
    assert "required_clearance=0.1000" in log


def test_direct_to_carry_mid_high_allowed_when_carry_front_forbidden() -> None:
    """Con obstáculos restantes, directo a carry_mid_high debe evaluarse aunque carry_front_high esté prohibido."""

    def fk(_joints):
        return (0.35, 0.115, 0.75)

    geom = {"carried_object_below_hand_m": 0.192, "carried_object_radius_xy_m": 0.05}
    obstacles = [
        {
            "label": "chips_can",
            "position": (0.520, -0.040, 0.385),
            "collision_dims": {"shape": "cylinder", "cylinder": [0.03, 0.19]},
            "top_z_m": 0.520,
        },
        {
            "label": "sugar_box",
            "position": (0.630, -0.130, 0.385),
            "collision_dims": {"shape": "box", "box": [0.06, 0.158, 0.210]},
            "top_z_m": 0.470,
        },
    ]
    selected, logs = select_transport_entry_validate_only(
        post_lift_hand=(0.456, 0.115, 0.587),
        post_lift_joints=[0.0] * 7,
        entry_target_joints=[0.2] * 7,
        entry_target_waypoint="carry_mid_high",
        allow_direct_to_carry_front_high=False,
        allow_direct_to_entry_target=True,
        allow_carry_front_high_corridors=False,
        fk_hand_fn=fk,
        attached_geom=geom,
        scene_obstacles=obstacles,
        table_top_z=0.270,
        policy={
            "carry_clearance_above_table_m": 0.120,
            "carry_clearance_above_obstacles_m": 0.080,
            "attached_transport_safety_margin_tolerance_m": 0.006,
            "use_lateral_transport_corridors": False,
            "local_exit_candidates": [
                "rear_retreat_x_negative",
                "vertical_raise_then_rear_retreat",
            ],
        },
        hand_z_candidates=[0.587],
    )
    direct_logs = [entry for entry in logs if entry.get("kind") == "direct"]
    assert direct_logs, "direct_to_carry_mid_high must be evaluated"
    assert any(str(l.get("entry_target_waypoint")) == "carry_mid_high" for l in direct_logs)
    assert selected is not None


def test_phased_local_escape_accepts_post_lift_near_obstacle() -> None:
    """Escape local con 0.05 m no debe rechazarse por umbral global 0.10 m en seg_a."""
    from pathlib import Path

    from panda_controller.demo_scene_policy import (
        apply_scene_policy_to_carry_transport,
        load_demo_scene_policy,
    )

    scenes_dir = str(Path(__file__).resolve().parents[1] / "config" / "demo_scenes")
    scene_policy = load_demo_scene_policy("demo_scene_02", scenes_dir=scenes_dir, use_cache=False)
    assert scene_policy is not None

    def fk(_joints):
        return (0.35, 0.115, 0.75)

    geom = compute_attached_object_geometry(
        {
            "label": "cracker_box",
            "grasp_center_base": [0.455, 0.115, 0.378],
            "grasp_yaw_rad": 2.9155,
            "dims_lwh": [0.158, 0.06, 0.21],
            "recommended_grasp_depth_from_top_m": 0.04,
            "scene_id": "demo_scene_02",
            "_scene_policy": scene_policy,
        },
        grasp_hand_z=0.587,
        grasp_hand_xy=(0.456, 0.115),
        table_top_z=0.270,
    )
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
    carry = {
        "carry_clearance_above_table_m": 0.120,
        "carry_clearance_above_obstacles_m": 0.100,
        "attached_transport_safety_margin_tolerance_m": 0.006,
        "use_lateral_transport_corridors": False,
        "local_exit_candidates": [
            "rear_retreat_x_negative",
            "rear_retreat_x_negative_far",
            "vertical_raise_then_rear_retreat",
        ],
    }
    carry = apply_scene_policy_to_carry_transport(
        carry, scene_policy, obstacles_remaining=True
    )
    selected, logs = select_transport_entry_validate_only(
        post_lift_hand=(0.456, 0.115, 0.587),
        post_lift_joints=[0.0] * 7,
        entry_target_joints=[0.2] * 7,
        entry_target_waypoint="carry_mid_high",
        allow_direct_to_carry_front_high=False,
        allow_direct_to_entry_target=True,
        allow_carry_front_high_corridors=False,
        fk_hand_fn=fk,
        attached_geom=geom,
        scene_obstacles=obstacles,
        table_top_z=0.270,
        policy=carry,
        hand_z_candidates=[0.587],
        scene_policy=scene_policy,
    )
    corridor_logs = [entry for entry in logs if entry.get("kind") == "corridor"]
    assert corridor_logs, "corridor candidates must be evaluated"
    seg_a_ok = any(bool(entry.get("seg_start_ok")) for entry in corridor_logs)
    assert seg_a_ok, "local escape seg_a should pass with phased clearance threshold"
    assert selected is not None


def test_local_escape_sweep_ok_overrides_global_xy_overlap_in_logs() -> None:
    """Sweep local OK debe propagarse a TRANSPORT_EXIT_CANDIDATE_VALIDATE."""
    from pathlib import Path

    from panda_controller.attached_transport_entry_validate import (
        emit_transport_exit_candidate_validate_logs,
    )
    from panda_controller.demo_scene_policy import (
        apply_scene_policy_to_carry_transport,
        load_demo_scene_policy,
    )
    from panda_controller.generic_known_scene_carry_planner import (
        compute_attached_object_geometry,
    )

    scenes_dir = str(Path(__file__).resolve().parents[1] / "config" / "demo_scenes")
    scene_policy = load_demo_scene_policy("demo_scene_02", scenes_dir=scenes_dir, use_cache=False)
    assert scene_policy is not None

    def fk(_joints):
        return (0.35, 0.115, 0.75)

    geom = compute_attached_object_geometry(
        {
            "label": "cracker_box",
            "grasp_center_base": [0.455, 0.115, 0.378],
            "grasp_yaw_rad": 2.9155,
            "dims_lwh": [0.158, 0.06, 0.21],
            "recommended_grasp_depth_from_top_m": 0.04,
            "scene_id": "demo_scene_02",
            "_scene_policy": scene_policy,
        },
        grasp_hand_z=0.587,
        grasp_hand_xy=(0.456, 0.115),
        table_top_z=0.270,
    )
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
        scene_policy,
        obstacles_remaining=True,
    )
    _, logs = select_transport_entry_validate_only(
        post_lift_hand=(0.456, 0.115, 0.587),
        post_lift_joints=[0.0] * 7,
        entry_target_joints=[0.2] * 7,
        entry_target_waypoint="carry_mid_high",
        allow_direct_to_carry_front_high=False,
        allow_direct_to_entry_target=True,
        allow_carry_front_high_corridors=False,
        fk_hand_fn=fk,
        attached_geom=geom,
        scene_obstacles=obstacles,
        table_top_z=0.270,
        policy=carry,
        hand_z_candidates=[0.587],
        scene_policy=scene_policy,
    )
    emitted: list[str] = []

    def capture(msg: str) -> None:
        emitted.append(msg)

    emit_transport_exit_candidate_validate_logs(
        logs,
        candidate_idx=0,
        carried_label="cracker_box",
        obstacles_remaining=["chips_can", "sugar_box", "mustard_bottle"],
        target_world_present=True,
        attached_geom=geom,
        local_exit_clearance_m=0.05,
        log_fn=capture,
    )
    ok_logs = [
        msg
        for msg in emitted
        if "[TRANSPORT_EXIT_CANDIDATE_VALIDATE]" in msg
        and "rear_retreat_x_negative" in msg
        and "result=OK" in msg
        and "reason=local_escape_sweep_ok" in msg
    ]
    assert ok_logs, "expected local escape OK in candidate validate log"
    assert any(
        "geom_ok=true" in msg and "plan_checked=false" in msg and "plan_ok=n/a" in msg
        for msg in ok_logs
    )
    decision_logs = [msg for msg in emitted if "[LOCAL_ESCAPE_DECISION]" in msg]
    assert decision_logs
    assert any("decision=ALLOW" in msg for msg in decision_logs)


def test_transport_exit_sweep_debug_log_format() -> None:
    from panda_controller.attached_transport_entry_validate import (
        format_transport_exit_sweep_debug_log,
    )

    log = format_transport_exit_sweep_debug_log(
        {
            "candidate_idx": 25,
            "exit_name": "rear_retreat_x_negative",
            "phase": "local_escape_post_lift",
            "start_hand_xyz": (0.456, 0.115, 0.587),
            "end_hand_xyz": (0.400, 0.115, 0.587),
            "start_carried_center_xyz": (0.455, 0.115, 0.587),
            "end_carried_center_xyz": (0.399, 0.115, 0.587),
            "carried_center_source": "attached_offset_from_grasp",
            "carried_footprint_xy_start": [0.34, 0.56, 0.04, 0.19],
            "carried_footprint_xy_end": [0.29, 0.51, 0.04, 0.19],
            "swept_aabb_xy": [0.29, 0.56, 0.04, 0.19],
            "closest_obstacle_label": "chips_can",
            "closest_check_start": {
                "obstacle_center": (0.52, -0.095, 0.52),
                "obstacle_dims": {"shape": "cylinder", "cylinder": [0.04, 0.255]},
                "xy_center_distance": 0.212,
                "xy_clearance": 0.072,
                "z_clearance": -0.118,
            },
            "closest_check_end": {
                "xy_center_distance": 0.267,
                "xy_clearance": 0.127,
                "z_clearance": -0.118,
            },
            "hard_collision_3d": False,
            "required_clearance_m": 0.05,
            "diagnostic_geometric_xy_overlap": False,
            "result": "OK",
            "reason": "ok",
        }
    )
    assert "[TRANSPORT_EXIT_SWEEP_DEBUG]" in log
    assert "required_clearance=0.0500" in log
    assert "carried_center_source=attached_offset_from_grasp" in log
    assert "hard_collision_3d=false" in log
