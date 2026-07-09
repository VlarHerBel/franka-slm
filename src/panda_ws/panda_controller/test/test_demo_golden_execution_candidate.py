"""Tests golden execution candidate v2 (demo_scene_02 + cracker_box)."""

from __future__ import annotations

import os

from panda_controller.demo_golden_execution_candidate import (
    GOLDEN_EXECUTION_SCHEMA_VERSION,
    PHASE_NAMES,
    RECORDED_NOT_VALIDATED_STATUS,
    VALIDATED_FULL_EXECUTION_STATUS,
    build_recorded_phases_from_snapshots,
    build_v2_from_v1_golden_and_waypoints,
    default_full_execution_golden_path,
    find_golden_phase,
    format_golden_execution_contract_violation_log,
    format_golden_execution_flow_decision_log,
    format_golden_execution_grid_guard_log,
    format_golden_execution_mode_lock_log,
    format_golden_execution_phase_capture_log,
    format_golden_execution_precheck_log,
    format_golden_execution_save_validate_log,
    golden_execution_approach_phase_executable,
    golden_execution_scope_active,
    load_golden_execution_candidate,
    normalize_golden_execution_candidate,
    patch_approach_to_pregrasp_from_snapshots,
    validate_golden_execution_identity,
    validate_golden_execution_save,
    validate_golden_descend_record,
    validate_scene_signature_compatibility,
    resolve_golden_descend_goal_tcp,
)
from panda_controller.demo_golden_pick_candidate import load_golden_pick_candidate
from panda_controller.demo_cracker_candidate_scoring import (
    build_cracker_candidate_score_record,
    compare_cracker_candidate_scores,
    export_cracker_candidate_scores_csv,
)
from panda_controller.demo_profile_loader import (
    load_demo_profile,
    resolve_golden_execution_path_from_profile,
)
from panda_controller.tfg_motion_waypoints import load_waypoints_file, resolve_waypoints_yaml_path


def _sample_v2_raw() -> dict:
    return {
        "golden_candidate_schema_version": 2,
        "type": "full_execution_candidate",
        "scene_id": "demo_scene_02",
        "layout_version": "v3_clear_table_transport",
        "target_label": "cracker_box",
        "status": "validated_full_execution",
        "scene_signature": {
            "scene_id": "demo_scene_02",
            "target_label": "cracker_box",
            "target_center_xyz": [0.455, 0.115, 0.47],
            "target_yaw_rad": 2.9155,
            "target_top_z": 0.47,
            "place_slot_index": 0,
            "compatibility_tolerances": {
                "center_xy_tol_m": 0.01,
                "yaw_tol_deg": 3.0,
                "top_z_tol_m": 0.01,
            },
        },
        "geometric_candidate": {
            "candidate_idx": 0,
            "commanded_tcp_yaw_rad": 1.344703673205025,
            "pregrasp_tcp": [0.455, 0.115, 0.562],
            "grasp_tcp": [0.455, 0.115, 0.437],
            "lift_tcp": [0.455, 0.115, 0.587],
        },
        "transport_route": {
            "selected_transport_entry": "vertical_raise_then_rear_retreat",
            "route": ["carry_mid_high", "turn_back_extended_aligned", "box_front_high", "box_high"],
        },
        "place_policy": {"slot_index": 0, "release_tcp_z": 0.329238},
        "phases": [
            {"name": "home_to_pick_workspace_ready", "type": "joint_trajectory", "duration_s": 5.0},
            {"name": "open_gripper_at_pregrasp", "type": "gripper_command", "duration_s": 1.0},
            {"name": "return_home", "type": "joint_trajectory", "duration_s": 4.0},
        ],
    }


def test_normalize_golden_execution_v2() -> None:
    golden = normalize_golden_execution_candidate(_sample_v2_raw())
    assert golden is not None
    assert golden["golden_candidate_schema_version"] == 2
    assert golden["status"] == VALIDATED_FULL_EXECUTION_STATUS
    assert len(golden["phases"]) == 3


def test_golden_execution_identity_ok() -> None:
    golden = normalize_golden_execution_candidate(_sample_v2_raw())
    assert golden is not None
    ok, reason = validate_golden_execution_identity(
        golden,
        scene_id="demo_scene_02",
        target_label="cracker_box",
        slot_index=0,
    )
    assert ok is True
    assert reason == "OK"


def test_scene_signature_compatibility_ok() -> None:
    golden = normalize_golden_execution_candidate(_sample_v2_raw())
    assert golden is not None
    ok, reason, _ = validate_scene_signature_compatibility(
        golden,
        runtime_xy=(0.455, 0.115),
        runtime_top_z=0.47,
        runtime_scene_yaw_rad=2.9155,
        runtime_scene_yaw_source="runtime_gt_spawn_yaw",
        runtime_commanded_tcp_yaw_rad=1.344703673205025,
        scene_obstacles=[],
    )
    assert ok is True
    assert reason == "OK"


def test_golden_execution_scope_active() -> None:
    assert golden_execution_scope_active(
        scene_id="demo_scene_02",
        target_label="cracker_box",
        place_slot_index=0,
    )
    assert not golden_execution_scope_active(
        scene_id="demo_scene_02",
        target_label="sugar_box",
        place_slot_index=0,
    )


def test_build_v2_from_v1_golden() -> None:
    base = os.path.dirname(__file__)
    v1_path = os.path.join(
        base,
        "..",
        "config",
        "demo_candidate_cache",
        "demo_scene_02_cracker_box_golden.yaml",
    )
    v1 = load_golden_pick_candidate(v1_path)
    assert v1 is not None
    wp = load_waypoints_file(resolve_waypoints_yaml_path(""))
    v2 = build_v2_from_v1_golden_and_waypoints(v1, waypoints_data=wp, slot_index=0)
    assert v2 is not None
    assert v2["status"] == VALIDATED_FULL_EXECUTION_STATUS
    assert len(v2["phases"]) >= 10


def test_v2_yaml_file_loads() -> None:
    path = default_full_execution_golden_path(
        "demo_scene_02", "cracker_box", slot_index=0
    )
    if not os.path.isfile(path):
        return
    golden = load_golden_execution_candidate(path)
    assert golden is not None
    assert golden["golden_candidate_schema_version"] == GOLDEN_EXECUTION_SCHEMA_VERSION


def test_cracker_candidate_scoring_selects_lowest_time() -> None:
    records = [
        build_cracker_candidate_score_record(
            spec={
                "grid_idx": 0,
                "yaw_deg": 10.0,
                "pregrasp_tcp_z": 0.56,
                "grasp_tcp_z": 0.44,
                "depth_from_top_m": 0.03,
                "ik_seed_label": "a",
                "pre_plan": (0.45, 0.11, 0.56),
                "gr_plan": (0.45, 0.11, 0.44),
            },
            pick_ok=True,
            descend_ok=True,
            lift_ok=True,
            transport_ok=True,
            joint_distance_to_hub=1.0,
            wrist_twist_score=0.5,
        ),
        build_cracker_candidate_score_record(
            spec={
                "grid_idx": 1,
                "yaw_deg": 20.0,
                "pregrasp_tcp_z": 0.56,
                "grasp_tcp_z": 0.44,
                "depth_from_top_m": 0.03,
                "ik_seed_label": "b",
                "pre_plan": (0.45, 0.11, 0.56),
                "gr_plan": (0.45, 0.11, 0.44),
            },
            pick_ok=True,
            descend_ok=True,
            lift_ok=True,
            transport_ok=True,
            joint_distance_to_hub=0.8,
            wrist_twist_score=0.3,
        ),
    ]
    best = compare_cracker_candidate_scores(records)
    assert best is not None
    assert int(best.get("candidate_id", best.get("candidate_idx", -1))) == 1
    csv_path = export_cracker_candidate_scores_csv(records)
    assert os.path.isfile(csv_path)


def test_golden_execution_mode_lock_log_format() -> None:
    log = format_golden_execution_mode_lock_log(
        {
            "scene_id": "demo_scene_02",
            "target_label": "cracker_box",
            "slot_index": 0,
            "use_golden_execution_candidate": "true",
            "require_golden_execution_candidate": "true",
            "schema_version": 2,
            "status": "validated_full_execution",
            "grid_search_disabled": "true",
            "v1_golden_disabled": "true",
            "accept_first_valid_disabled": "true",
            "result": "OK",
        }
    )
    assert "[GOLDEN_EXECUTION_MODE_LOCK]" in log
    assert "grid_search_disabled=true" in log


def test_golden_execution_required_flow_logs_block_normal_pipeline() -> None:
    flow = format_golden_execution_flow_decision_log(
        {
            "scene_id": "demo_scene_02",
            "target_label": "cracker_box",
            "slot_index": 0,
            "use_golden_execution_candidate": "true",
            "require_golden_execution_candidate": "true",
            "candidate_benchmark_mode": "false",
            "scope_active": "true",
            "decision": "REPLAY_GOLDEN",
            "normal_pipeline_allowed": "false",
            "reason": "golden_v2_ready",
        }
    )
    assert "[GOLDEN_EXECUTION_FLOW_DECISION]" in flow
    assert "decision=REPLAY_GOLDEN" in flow
    assert "normal_pipeline_allowed=false" in flow

    guard = format_golden_execution_grid_guard_log(
        {
            "use_golden_execution_candidate": "true",
            "require_golden_execution_candidate": "true",
            "candidate_benchmark_mode": "false",
            "scope_active": "true",
            "grid_allowed": "false",
            "result": "VIOLATION",
        }
    )
    assert "[GOLDEN_EXECUTION_GRID_GUARD]" in guard
    assert "grid_allowed=false" in guard

    violation = format_golden_execution_contract_violation_log(
        {"stage": "PLAN_BEFORE_MOTION"}
    )
    assert "[GOLDEN_EXECUTION_CONTRACT_VIOLATION]" in violation
    assert "action=ABORT_NO_FALLBACK" in violation


def test_golden_precheck_allows_closed_gripper_with_pregrasp_open_phase() -> None:
    log = format_golden_execution_precheck_log(
        {
            "scene_id": "demo_scene_02",
            "target_label": "cracker_box",
            "slot_index": 0,
            "candidate_path": "demo_scene_02_cracker_box_slot_0_full_execution_golden.yaml",
            "schema_version": 2,
            "status": "validated_full_execution",
            "scene_signature_ok": "true",
            "robot_start_ok": "true",
            "gripper_commandable_ok": "true",
            "gripper_no_attached_object_ok": "true",
            "gripper_initial_open_ok": "false",
            "golden_has_pregrasp_open_phase_ok": "true",
            "gripper_ok": "true",
            "target_ok": "true",
            "obstacles_ok": "true",
            "slot_ok": "true",
            "phases_ok": "true",
            "result": "OK",
            "reason": "OK",
        }
    )
    assert "gripper_initial_open_ok=false" in log
    assert "golden_has_pregrasp_open_phase_ok=true" in log
    assert "gripper_ok=true" in log
    assert "result=OK" in log


def test_joint_trajectory_tcp_only_is_not_replay_executable_contract() -> None:
    phase = {
        "name": "approach_to_pregrasp",
        "type": "joint_trajectory",
        "tcp_goal": [0.455, 0.115, 0.562],
    }
    executable = bool(
        phase.get("points")
        or phase.get("goal_js")
        or phase.get("goal_waypoint")
        or phase.get("target_waypoint")
    )
    assert executable is False


def test_golden_execution_record_produces_executable_approach_pregrasp() -> None:
    goal_js = [0.1, -0.5, 0.0, -2.0, 0.0, 2.0, 0.8]
    snapshots = {
        "pick_workspace_ready_js": [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785],
        "aligned_pregrasp_js": goal_js,
        "approach_start_js": [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785],
        "approach_duration_s": 3.5,
        "approach_tcp_goal": [0.455, 0.115, 0.562],
    }
    phases = [
        {
            "name": "approach_to_pregrasp",
            "type": "cartesian_or_joint_trajectory",
            "tcp_goal": [0.455, 0.115, 0.562],
            "goal_js": None,
            "duration_s": 4.0,
        }
    ]
    assert patch_approach_to_pregrasp_from_snapshots(
        phases,
        snapshots,
        geometric={"pregrasp_tcp": [0.455, 0.115, 0.562]},
    )
    phase = find_golden_phase(phases, "approach_to_pregrasp")
    assert phase is not None
    assert phase["type"] == "joint_trajectory"
    assert len(phase["goal_js"]) == 7
    assert phase["goal_js"] == goal_js
    assert len(phase["start_js"]) == 7
    assert phase["end_js"] == goal_js
    assert golden_execution_approach_phase_executable(phases)

    log = format_golden_execution_phase_capture_log(
        {
            "phase_name": "approach_to_pregrasp",
            "has_goal_js": "true",
            "goal_js_len": 7,
            "result": "OK",
        }
    )
    assert "[GOLDEN_EXECUTION_PHASE_CAPTURE]" in log
    assert "has_goal_js=true" in log
    assert "goal_js_len=7" in log


def test_golden_execution_without_approach_goal_js_stays_not_validated() -> None:
    phases = [
        {
            "name": "approach_to_pregrasp",
            "type": "cartesian_or_joint_trajectory",
            "tcp_goal": [0.455, 0.115, 0.562],
            "goal_js": None,
        }
    ]
    assert not golden_execution_approach_phase_executable(phases)
    assert not patch_approach_to_pregrasp_from_snapshots(phases, {}, geometric={})
    save_validation = validate_golden_execution_save(phases)
    assert save_validation["result"] == "FAIL"
    status = (
        VALIDATED_FULL_EXECUTION_STATUS
        if save_validation["result"] == "OK"
        else RECORDED_NOT_VALIDATED_STATUS
    )
    assert status == RECORDED_NOT_VALIDATED_STATUS


def _simulated_full_execution_snapshots() -> dict:
    goal_js = [0.1, -0.5, 0.0, -2.0, 0.0, 2.0, 0.8]
    pre_tcp = [0.455, 0.115, 0.562]
    gr_tcp = [0.455, 0.115, 0.437]
    lift_tcp = [0.455, 0.115, 0.649]
    return {
        "pick_workspace_ready_js": [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785],
        "aligned_pregrasp_js": goal_js,
        "approach_start_js": [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785],
        "approach_duration_s": 3.5,
        "approach_tcp_goal": pre_tcp,
        "gripper_axis_joint7_target": 0.8,
        "gripper_axis_expected_error_deg": 1.5,
        "gripper_axis_result": "OK",
        "open_gripper_recorded": True,
        "open_gripper_joint": 0.0399,
        "open_gripper_duration_s": 1.0,
        "descend_start_tcp": pre_tcp,
        "descend_goal_tcp": gr_tcp,
        "descend_depth_from_top_m": 0.033,
        "descend_duration_s": 2.8,
        "descend_start_js": goal_js,
        "grasp_js": [0.11, -0.51, 0.01, -2.01, 0.0, 2.01, 0.81],
        "descend_fraction": 1.0,
        "descend_result": "OK",
        "close_gripper_recorded": True,
        "close_gripper_joint": 0.027,
        "close_expected_width_m": 0.06,
        "attach_recorded": True,
        "attach_result": "OK",
        "attach_contact_policy": "strict",
        "lift_start_tcp": gr_tcp,
        "lift_goal_tcp": lift_tcp,
        "post_lift_js": [0.12, -0.52, 0.02, -2.02, 0.0, 2.02, 0.82],
        "lift_result": "OK",
        "post_lift_escape_recorded": True,
        "post_lift_escape_mode": "lateral_shift",
        "selected_local_exit": "lateral_shift",
        "post_lift_escape_start_tcp": lift_tcp,
        "post_lift_escape_goal_tcp": [0.48, 0.115, 0.649],
        "post_lift_escape_result": "OK",
        "transport_entry_recorded": True,
        "transport_entry_target_waypoint": "carry_mid_high",
        "transport_entry_js": [0.15, -0.55, 0.03, -2.05, 0.0, 2.05, 0.85],
        "transport_entry_result": "OK",
        "transport_sequence": ["carry_mid_high", "carry_front_high", "box_high"],
        "per_segment_times_s": [8.0, 10.0, 6.0],
        "place_approach_recorded": True,
        "place_approach_tcp_z": 0.65,
        "selected_release_tcp_z": 0.3492,
        "place_release_result": "OK",
        "open_detach_recorded": True,
        "open_detach_result": "OK",
        "place_retreat_recorded": True,
        "place_retreat_tcp_z": 0.65,
        "return_home_js": [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785],
        "return_home_duration_s": 4.0,
    }


def test_golden_execution_full_record_has_sixteen_phases() -> None:
    snapshots = _simulated_full_execution_snapshots()
    geometric = {
        "pregrasp_tcp": [0.455, 0.115, 0.562],
        "grasp_tcp": [0.455, 0.115, 0.437],
        "lift_tcp": [0.455, 0.115, 0.649],
        "depth_from_top_m": 0.033,
        "cartesian_descend_fraction": 1.0,
    }
    transport = {
        "route": snapshots["transport_sequence"],
        "backend": "direct_action",
        "first_hub": "carry_mid_high",
        "selected_transport_entry": "lateral_shift",
    }
    place_policy = {
        "slot_index": 0,
        "release_tcp_z": 0.3492,
        "deposit_xy": [-0.37, 0.08],
        "approach_tcp_z": 0.65,
        "retreat_tcp_z": 0.65,
    }
    grasp = {"open_joint": 0.0399, "close_joint": 0.027, "expected_width_m": 0.06}
    v1_stub = {
        "scene_id": "demo_scene_02",
        "layout_version": "v3_clear_table_transport",
        "target_label": "cracker_box",
        "status": VALIDATED_FULL_EXECUTION_STATUS,
        "candidate": geometric,
        "grasp": grasp,
        "transport": transport,
        "place": place_policy,
        "validation": {"result": VALIDATED_FULL_EXECUTION_STATUS},
    }
    wp_path = resolve_waypoints_yaml_path("demo_scene_02")
    waypoints_data = load_waypoints_file(wp_path) if wp_path else {}
    built = build_v2_from_v1_golden_and_waypoints(
        v1_stub, waypoints_data=waypoints_data, slot_index=0
    )
    base_phases = list(built.get("phases") or []) if built else []
    phases = build_recorded_phases_from_snapshots(
        snapshots,
        transport_route=transport["route"],
        per_segment_times_s=snapshots["per_segment_times_s"],
        place_policy=place_policy,
        geometric=geometric,
        base_phases=base_phases,
        transport=transport,
        grasp=grasp,
        waypoints_data=waypoints_data,
    )
    save_validation = validate_golden_execution_save(phases)
    log = format_golden_execution_save_validate_log(save_validation)
    assert "[GOLDEN_EXECUTION_SAVE_VALIDATE]" in log
    assert save_validation["phase_count_real"] == 16
    assert save_validation["result"] == "OK"
    assert len(phases) == 16
    assert [p["name"] for p in phases] == list(PHASE_NAMES)
    status = (
        VALIDATED_FULL_EXECUTION_STATUS
        if save_validation["result"] == "OK"
        else RECORDED_NOT_VALIDATED_STATUS
    )
    assert status == VALIDATED_FULL_EXECUTION_STATUS
    approach = find_golden_phase(phases, "approach_to_pregrasp")
    assert approach is not None
    assert len(approach["goal_js"]) == 7
    assert find_golden_phase(phases, "post_lift_local_escape") is not None
    assert find_golden_phase(phases, "transport_entry_to_safe_hub") is not None
    assert find_golden_phase(phases, "deterministic_transport") is not None
    assert find_golden_phase(phases, "place_release") is not None
    assert find_golden_phase(phases, "return_home") is not None


def test_golden_execution_partial_phases_fail_validation() -> None:
    phases = build_recorded_phases_from_snapshots(
        {"pick_workspace_ready_js": [0.0] * 7, "aligned_pregrasp_js": [0.1] * 7},
        transport_route=["carry_mid_high"],
        geometric={"pregrasp_tcp": [0.1, 0.2, 0.3], "grasp_tcp": [0.1, 0.2, 0.27]},
    )
    save_validation = validate_golden_execution_save(phases)
    assert save_validation["result"] == "FAIL"
    assert len(save_validation.get("missing_phases") or []) > 0


def test_golden_descend_validate_rejects_pregrasp_like_goal() -> None:
    bad = validate_golden_descend_record(
        {
            "start_tcp": [0.455, 0.115, 0.569],
            "goal_tcp": [0.455, 0.115, 0.562],
            "depth_from_top_m": 0.033,
        },
        pregrasp_tcp_z=0.562,
    )
    assert bad["result"] == "FAIL"
    assert bad["reason"] in (
        "descend_delta_z_too_small",
        "goal_tcp_z_too_close_to_pregrasp",
    )


def test_resolve_golden_descend_goal_tcp_from_top_z_depth() -> None:
    goal = resolve_golden_descend_goal_tcp(
        start_tcp=[0.455, 0.115, 0.562],
        post_descend_tcp=None,
        candidate={
            "top_z_m": 0.47,
            "recommended_grasp_depth_from_top_m": 0.033,
            "chosen_target_center_base": [0.455, 0.115],
        },
    )
    assert goal is not None
    assert abs(float(goal[2]) - 0.437) < 0.001


def test_profile_resolves_golden_execution_path() -> None:
    profile = load_demo_profile("demo_scene_02_cracker_box")
    assert profile is not None
    path = resolve_golden_execution_path_from_profile(profile)
    assert "full_execution_golden" in path
