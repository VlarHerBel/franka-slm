"""Tests offline para demo_scene_reachability_scan (sin ROS/MoveIt)."""

from __future__ import annotations

import math

import pytest

from panda_controller.demo_scene_reachability_scan import (
    DEBUG_CALIBRATION_CELLS,
    DEBUG_IK_SEED_LABELS,
    DEMO_SCAN_LABELS,
    MUSTARD_BOTTLE_DEBUG_PREGRASP_ABOVE_TOP_M,
    ScanCellResult,
    SUGAR_BOX_DEBUG_PREGRASP_ABOVE_TOP_M,
    VARIANT_SEARCH_DEBUG_LABELS,
    _as_float_list,
    _as_float_quat,
    aggregate_cell_results,
    aggregate_budgeted_cell_results,
    binary_color_for_cell,
    build_detection_for_cell,
    build_detection_for_reachability_cell,
    build_expanded_debug_yaw_entries,
    build_golden_debug_variants,
    compute_common_reachable_xy,
    compute_compensated_spawn_xy_from_operational,
    compute_top_z_m,
    default_input_mode_for_label,
    format_reachability_binary_cell_decision_log,
    format_reachability_cell_debug_log,
    format_reachability_operational_center_log,
    format_variant_budget_log,
    format_variant_early_stop_log,
    format_variant_search_summary_log,
    is_binary_reachable_cell,
    is_fully_reachable_variant,
    iter_budgeted_variant_jobs,
    iter_debug_calibration_grid,
    iter_expanded_debug_variant_jobs,
    iter_grasp_variants_for_cell,
    normalize_start_joint_positions,
    propose_demo_scene_yaml,
    reachability_heatmap_cell_value,
    resolve_closing_yaw_rad,
    resolve_reachability_cell_coordinates,
    resolve_single_cell_from_args,
    single_cell_best_csv_row,
    summarize_scan_csv,
    summarize_variant_search_results,
    validate_scanner_grasp_hand_target_z,
    wrap_to_pi,
)
from panda_controller.tcp_hand_pose_convert import hand_pose_from_desired_tcp


def test_wrap_to_pi() -> None:
    assert abs(wrap_to_pi(math.pi) - math.pi) < 1e-6
    assert abs(wrap_to_pi(3.5 * math.pi) - (-0.5 * math.pi)) < 1e-5


def test_closing_yaw_short_axis() -> None:
    policy = {"preferred_closing_axis": "short_axis"}
    cy = resolve_closing_yaw_rad(0.0, policy)
    assert abs(cy - math.pi / 2.0) < 1e-6


def test_closing_yaw_free() -> None:
    policy = {"yaw_policy": "yaw_free"}
    cy = resolve_closing_yaw_rad(1.2, policy)
    assert abs(cy - 1.2) < 1e-6


def test_top_z_cracker_box() -> None:
    from panda_controller.demo_scene_reachability_scan import _vision_policy_exports

    export_grasp_policy_for_executor, _, _ = _vision_policy_exports()
    policy = export_grasp_policy_for_executor("cracker_box")
    top_z, geom_z = compute_top_z_m("cracker_box", 0.27, policy)
    assert top_z > 0.27
    assert geom_z > 0.27
    assert top_z > geom_z


def test_build_detection_for_cell_has_axes() -> None:
    from panda_controller.demo_scene_reachability_scan import _vision_policy_exports

    export_grasp_policy_for_executor, _, _ = _vision_policy_exports()
    policy = export_grasp_policy_for_executor("sugar_box")
    det = build_detection_for_cell(
        "sugar_box",
        0.55,
        -0.10,
        1.0,
        table_z_m=0.27,
        policy=policy,
    )
    assert det["top_z_m"] > 0.27
    assert len(det["major_axis_xy"]) == 2
    assert "closing_yaw_rad" in det


def test_iter_grasp_variants_non_empty_for_demo_labels() -> None:
    for label in DEMO_SCAN_LABELS:
        variants = iter_grasp_variants_for_cell(
            label,
            0.55,
            0.0,
            0.0,
            table_z_m=0.27,
        )
        assert variants, "expected variants for %s" % label
        v0 = variants[0]
        assert v0.tcp_pregrasp_xyz[2] > v0.tcp_grasp_xyz[2]


def test_mustard_palm_bridge_raises_grasp_z() -> None:
    plain = iter_grasp_variants_for_cell(
        "mustard_bottle",
        0.60,
        0.05,
        1.6,
        table_z_m=0.27,
    )
    assert plain
    assert plain[0].tcp_grasp_xyz[2] >= 0.40


def test_aggregate_cell_prefers_ok() -> None:
    from panda_controller.demo_scene_reachability_scan import ScanCellResult

    fail = ScanCellResult(
        label="sugar_box",
        x=0.5,
        y=0.0,
        yaw=0.0,
        pregrasp_ok=True,
        reason="endpoint_ik_fail",
    )
    ok = ScanCellResult(
        label="sugar_box",
        x=0.5,
        y=0.0,
        yaw=0.0,
        result="OK",
        reason="reachable",
        pregrasp_ok=True,
        plan_to_pregrasp_ok=True,
        endpoint_ik_ok=True,
        pregrasp_ik_error_code="SUCCESS",
        endpoint_ik_error_code="SUCCESS",
        cartesian_fraction=1.0,
        collision_ok=True,
        joint_limits_ok=True,
    )
    agg = aggregate_cell_results([fail, ok])
    assert agg.result == "OK"
    assert is_binary_reachable_cell(agg)


def test_common_reachable_xy_and_propose_scene() -> None:
    rows = [
        {
            "label": lb,
            "x": "0.5500",
            "y": "0.0000",
            "yaw": "0.0000",
            "result": "OK",
            "binary_color": "yellow",
            "cell_fully_reachable": "true",
        }
        for lb in DEMO_SCAN_LABELS
    ]
    common = compute_common_reachable_xy(rows, DEMO_SCAN_LABELS)
    assert (0.55, 0.0) in common
    scene = propose_demo_scene_yaml(rows)
    assert scene["scene_id"] == "demo_scene_03_reachable"
    assert set(scene["objects"]) == set(DEMO_SCAN_LABELS)


def test_summarize_scan_csv(tmp_path) -> None:
    path = tmp_path / "scan.csv"
    path.write_text(
        "label,x,y,yaw,top_z,grasp_tcp_x,grasp_tcp_y,grasp_tcp_z,"
        "pregrasp_tcp_x,pregrasp_tcp_y,pregrasp_tcp_z,target_hand_x,target_hand_y,"
        "target_hand_z,quat_x,quat_y,quat_z,quat_w,target_link,moveit_target_link,"
        "use_grasp_tcp,seed_state_name,pregrasp_ok,plan_to_pregrasp_ok,"
        "start_tcp_error_m,endpoint_ik_ok,cartesian_fraction,collision_ok,"
        "joint_limits_ok,pregrasp_ik_error_code,endpoint_ik_error_code,result,reason,"
        "variant_budget,attempts_used,early_stop_used,total_possible_variants,"
        "evaluated_variants,cell_fully_reachable\n"
        "cracker_box,0.55,0.00,0.00,0.47,0.55,0.00,0.44,0.55,0.00,0.56,"
        "0.55,0.00,0.66,0,1,0,0,panda_grasp_tcp,panda_hand,true,"
        "pick_workspace_ready,true,true,0.001,true,1.0,true,true,SUCCESS,SUCCESS,OK,reachable,"
        ",0,false,0,0,false\n"
        "sugar_box,0.55,0.00,0.00,,,,,,,,,,,,,,,,,,,,false,false,,false,0.0,false,false,"
        "NO_IK_SOLUTION,,FAIL,pregrasp_ik_fail,fast,3,false,12,3,false\n",
        encoding="utf-8",
    )
    summary = summarize_scan_csv(str(path), labels=("cracker_box", "sugar_box"))
    assert summary["labels"]["cracker_box"]["ok"] == 1
    assert summary["labels"]["sugar_box"]["ok"] == 0


def test_normalize_start_joint_positions_from_dict_and_numpy() -> None:
    import numpy as np

    seed_dict = {
        "panda_joint1": np.float64(-0.0028),
        "panda_joint2": np.float64(0.1647),
        "panda_joint3": np.float64(0.0486),
        "panda_joint4": np.float64(-1.2623),
        "panda_joint5": np.float64(-0.0128),
        "panda_joint6": np.float64(1.4497),
        "panda_joint7": np.float64(0.7736),
    }
    vals = normalize_start_joint_positions(seed_dict)
    assert vals is not None
    assert all(type(v) is float for v in vals)
    quat = _as_float_quat((0.6228, 0.7824, 0.0, 0.0))
    assert all(type(v) is float for v in quat)


def test_debug_calibration_cells_and_log_format() -> None:
    cells = iter_debug_calibration_grid(("cracker_box", "chips_can"))
    assert len(cells) == 2
    assert DEBUG_CALIBRATION_CELLS["cracker_box"][0] == 0.455
    assert DEBUG_CALIBRATION_CELLS["sugar_box"] == (0.630, -0.175, -3.0159)
    assert DEBUG_CALIBRATION_CELLS["mustard_bottle"] == (0.660, 0.060, 1.6392)
    all_cells = iter_debug_calibration_grid(DEMO_SCAN_LABELS)
    assert len(all_cells) == 4
    golden = build_golden_debug_variants("cracker_box")
    assert len(golden) == 1
    assert abs(golden[0].tcp_pregrasp_xyz[2] - 0.5620) < 1e-6
    row = ScanCellResult(
        label="cracker_box",
        x=0.455,
        y=0.115,
        yaw=2.9155,
        top_z=0.47,
        grasp_tcp=(0.455, 0.115, 0.437),
        pregrasp_tcp=(0.455, 0.115, 0.562),
        pregrasp_hand_target=(0.455, 0.115, 0.662),
        grasp_hand_target=(0.455, 0.115, 0.537),
        hand_to_tcp_translation=(0.0, 0.0, 0.10),
        quat=(0.0, 1.0, 0.0, 0.0),
        pregrasp_ik_error_code="NO_IK_SOLUTION",
        reason="pregrasp_ik_fail",
    )
    log = format_reachability_cell_debug_log(row)
    assert "[REACHABILITY_CELL_DEBUG]" in log
    assert "pregrasp_ik_error_code=NO_IK_SOLUTION" in log
    assert "pregrasp_hand_target=" in log
    assert "grasp_hand_target=" in log
    assert "hand_to_tcp_translation=" in log
    assert "target_link=panda_grasp_tcp" in log


def test_variant_search_debug_labels_and_pregrasp_clearances() -> None:
    assert VARIANT_SEARCH_DEBUG_LABELS == frozenset({"sugar_box", "mustard_bottle"})
    assert SUGAR_BOX_DEBUG_PREGRASP_ABOVE_TOP_M == (
        0.055,
        0.070,
        0.085,
        0.100,
        0.120,
    )
    assert MUSTARD_BOTTLE_DEBUG_PREGRASP_ABOVE_TOP_M == (
        0.030,
        0.050,
        0.070,
        0.085,
        0.100,
    )
    assert DEBUG_IK_SEED_LABELS == (
        "pick_workspace_ready",
        "home",
        "current_joint_state",
        "joint7_near_zero",
    )


def test_build_expanded_debug_yaw_entries_demo_scene_02() -> None:
    from panda_controller.demo_scene_reachability_scan import _vision_policy_exports

    export_grasp_policy_for_executor, _, _ = _vision_policy_exports()
    for label, spawn_yaw in (
        ("sugar_box", -3.0159),
        ("mustard_bottle", 1.6392),
    ):
        policy = export_grasp_policy_for_executor(label)
        closing_yaw = resolve_closing_yaw_rad(spawn_yaw, policy)
        entries = build_expanded_debug_yaw_entries(
            spawn_yaw=spawn_yaw,
            closing_yaw=closing_yaw,
            label=label,
            workspace_tcp_yaw=0.42,
        )
        names = {name for name, _y in entries}
        assert "spawn_yaw" in names
        assert "spawn_yaw_pi" in names
        assert "yaw_zero" in names
        assert "yaw_pi_over_2" in names
        assert "yaw_neg_pi_over_2" in names
        assert "yaw_pi" in names
        assert "yaw_from_workspace_tcp" in names
        if label == "mustard_bottle":
            yaws = {round(y, 4) for _name, y in entries}
            assert round(closing_yaw, 4) in yaws
            assert round(wrap_to_pi(closing_yaw + math.pi), 4) in yaws
            assert round(wrap_to_pi(closing_yaw + math.pi / 2.0), 4) in yaws


def test_iter_expanded_debug_variant_jobs_sugar_and_mustard() -> None:
    from panda_controller.demo_scene_reachability_scan import _vision_policy_exports

    export_grasp_policy_for_executor, _, _ = _vision_policy_exports()
    ik_seeds = {
        name: [0.0, 0.0, 0.0, -1.5, 0.0, 1.5, 0.0]
        for name in DEBUG_IK_SEED_LABELS
    }
    for label in ("sugar_box", "mustard_bottle"):
        x, y, yaw = DEBUG_CALIBRATION_CELLS[label]
        policy = export_grasp_policy_for_executor(label)
        det = build_detection_for_cell(
            label,
            x,
            y,
            yaw,
            table_z_m=0.27,
            policy=policy,
        )
        jobs = iter_expanded_debug_variant_jobs(
            label,
            x,
            y,
            yaw,
            table_z_m=0.27,
            policy=policy,
            detection=det,
            ik_seeds=ik_seeds,
            workspace_tcp_yaw=0.25,
        )
        assert len(jobs) >= 200, "expected broad exhaustive search for %s" % label
        seed_names = {job[1] for job in jobs}
        assert seed_names == set(DEBUG_IK_SEED_LABELS)
        notes = jobs[0][0].notes
        assert "pregrasp_z=" in notes
        assert "yaw=" in notes
    assert iter_expanded_debug_variant_jobs(
        "cracker_box",
        0.455,
        0.115,
        2.9155,
        table_z_m=0.27,
        policy=export_grasp_policy_for_executor("cracker_box"),
        detection=build_detection_for_cell(
            "cracker_box",
            0.455,
            0.115,
            2.9155,
            table_z_m=0.27,
            policy=export_grasp_policy_for_executor("cracker_box"),
        ),
        ik_seeds=ik_seeds,
    ) == []


def test_iter_budgeted_variant_jobs_fast_and_balanced() -> None:
    from panda_controller.demo_scene_reachability_scan import _vision_policy_exports

    export_grasp_policy_for_executor, _, _ = _vision_policy_exports()
    ik_seeds = {
        name: [0.0, 0.0, 0.0, -1.5, 0.0, 1.5, 0.0]
        for name in DEBUG_IK_SEED_LABELS
    }
    sugar_det = build_detection_for_cell(
        "sugar_box",
        0.500,
        -0.100,
        0.0,
        table_z_m=0.27,
        policy=export_grasp_policy_for_executor("sugar_box"),
    )
    fast_jobs, fast_meta = iter_budgeted_variant_jobs(
        "sugar_box",
        0.500,
        -0.100,
        0.0,
        table_z_m=0.27,
        policy=export_grasp_policy_for_executor("sugar_box"),
        detection=sugar_det,
        ik_seeds=ik_seeds,
        budget="fast",
    )
    assert 1 <= len(fast_jobs) <= 12
    assert fast_meta["variant_budget"] == "fast"
    assert fast_meta["early_stop"] is True
    first = fast_jobs[0][0]
    assert "closing_yaw" in first.notes
    assert "clearance_above_top=0.0550" in first.notes
    assert "depth=0.0250" in first.notes
    assert fast_jobs[0][1] == "pick_workspace_ready"

    balanced_jobs, balanced_meta = iter_budgeted_variant_jobs(
        "sugar_box",
        0.500,
        -0.100,
        0.0,
        table_z_m=0.27,
        policy=export_grasp_policy_for_executor("sugar_box"),
        detection=sugar_det,
        ik_seeds=ik_seeds,
        budget="balanced",
    )
    assert len(balanced_jobs) <= 40
    assert balanced_meta["variant_budget"] == "balanced"

    x, y, yaw = DEBUG_CALIBRATION_CELLS["mustard_bottle"]
    mustard_det = build_detection_for_cell(
        "mustard_bottle",
        x,
        y,
        yaw,
        table_z_m=0.27,
        policy=export_grasp_policy_for_executor("mustard_bottle"),
    )
    mustard_fast, _ = iter_budgeted_variant_jobs(
        "mustard_bottle",
        x,
        y,
        yaw,
        table_z_m=0.27,
        policy=export_grasp_policy_for_executor("mustard_bottle"),
        detection=mustard_det,
        ik_seeds=ik_seeds,
        budget="fast",
    )
    assert 1 <= len(mustard_fast) <= 12
    assert "palm_bridge" in mustard_fast[0][0].notes

    exhaustive_jobs, exhaustive_meta = iter_budgeted_variant_jobs(
        "sugar_box",
        0.500,
        -0.100,
        0.0,
        table_z_m=0.27,
        policy=export_grasp_policy_for_executor("sugar_box"),
        detection=sugar_det,
        ik_seeds=ik_seeds,
        budget="exhaustive",
    )
    assert len(exhaustive_jobs) >= 200
    assert exhaustive_meta["early_stop"] is False


def test_variant_budget_logs_and_aggregate_budgeted_csv_fields() -> None:
    budget_log = format_variant_budget_log(
        label="sugar_box",
        mode="fast",
        max_variants=12,
        early_stop=True,
    )
    assert "[REACHABILITY_VARIANT_BUDGET]" in budget_log
    assert "mode=fast" in budget_log
    assert "early_stop=true" in budget_log
    stop_log = format_variant_early_stop_log(
        label="sugar_box",
        variant_notes="budget_search yaw=closing_yaw",
        attempts_used=1,
        result="OK",
    )
    assert "[REACHABILITY_VARIANT_SEARCH_EARLY_STOP]" in stop_log
    assert "attempts_used=1" in stop_log

    ok = ScanCellResult(
        label="sugar_box",
        x=0.5,
        y=-0.1,
        yaw=0.0,
        grasp_tcp=(0.5, -0.1, 0.42),
        pregrasp_tcp=(0.5, -0.1, 0.50),
        pregrasp_ok=True,
        plan_to_pregrasp_ok=True,
        endpoint_ik_ok=True,
        pregrasp_ik_error_code="SUCCESS",
        endpoint_ik_error_code="SUCCESS",
        cartesian_fraction=1.0,
        collision_ok=True,
        joint_limits_ok=True,
        result="OK",
        reason="reachable",
    )
    agg = aggregate_budgeted_cell_results(
        [ok],
        label="sugar_box",
        x=0.5,
        y=-0.1,
        yaw=0.0,
        budget_meta={"variant_budget": "fast", "total_possible_variants": 12},
        evaluated_variants=1,
        early_stop_used=True,
    )
    row = agg.to_csv_row()
    assert row["variant_budget"] == "fast"
    assert row["attempts_used"] == "1"
    assert row["early_stop_used"] == "true"
    assert row["total_possible_variants"] == "12"
    assert row["evaluated_variants"] == "1"
    assert row["cell_fully_reachable"] == "true"


def test_summarize_variant_search_results_and_log() -> None:
    rows = [
        ScanCellResult(
            label="sugar_box",
            x=0.63,
            y=-0.175,
            yaw=-3.0159,
            pregrasp_ok=False,
            pregrasp_ik_error_code="ik_timeout",
            variant_notes="debug_search yaw=spawn_yaw pregrasp_z=0.5000",
            seed_state_name="pick_workspace_ready",
            reason="pregrasp_ik_fail",
        ),
        ScanCellResult(
            label="sugar_box",
            x=0.63,
            y=-0.175,
            yaw=-3.0159,
            pregrasp_ok=True,
            plan_to_pregrasp_ok=True,
            endpoint_ik_ok=False,
            variant_notes="debug_search yaw=yaw_zero pregrasp_z=0.5150",
            seed_state_name="home",
            reason="endpoint_ik_fail",
        ),
    ]
    summary = summarize_variant_search_results(rows)
    assert summary["total_variants"] == 2
    assert summary["pregrasp_success"] == 1
    assert summary["plan_success"] == 1
    assert summary["endpoint_success"] == 0
    log = format_variant_search_summary_log(label="sugar_box", summary=summary)
    assert "[REACHABILITY_VARIANT_SEARCH_SUMMARY]" in log
    assert "label=sugar_box" in log
    assert "total_variants=2" in log
    assert "pregrasp_success=1" in log
    assert "best_variant=" in log


def test_resolve_single_cell_from_args() -> None:
    cell = resolve_single_cell_from_args(
        label="sugar_box",
        x=0.5,
        y=-0.1,
        yaw=0.0,
    )
    assert cell == ("sugar_box", 0.5, -0.1, 0.0)
    assert resolve_single_cell_from_args(
        label="",
        x=None,
        y=None,
        yaw=None,
    ) is None
    with pytest.raises(ValueError, match="missing"):
        resolve_single_cell_from_args(label="sugar_box", x=0.5, y=None, yaw=0.0)
    with pytest.raises(ValueError, match="only supports"):
        resolve_single_cell_from_args(
            label="cracker_box",
            x=0.455,
            y=0.115,
            yaw=2.9155,
        )


def test_is_fully_reachable_variant_and_single_cell_csv() -> None:
    ok = ScanCellResult(
        label="sugar_box",
        x=0.5,
        y=-0.1,
        yaw=0.0,
        grasp_tcp=(0.5, -0.1, 0.42),
        pregrasp_tcp=(0.5, -0.1, 0.52),
        commanded_tcp_yaw_rad=0.0,
        pregrasp_ok=True,
        plan_to_pregrasp_ok=True,
        endpoint_ik_ok=True,
        pregrasp_ik_error_code="SUCCESS",
        endpoint_ik_error_code="SUCCESS",
        cartesian_fraction=0.98,
        collision_ok=True,
        joint_limits_ok=True,
        result="OK",
        reason="reachable",
        variant_notes="debug_search yaw=yaw_zero",
        seed_state_name="home",
    )
    assert is_fully_reachable_variant(ok)
    assert binary_color_for_cell(ok) == "yellow"
    fail = ScanCellResult(
        label="sugar_box",
        x=0.5,
        y=-0.1,
        yaw=0.0,
        pregrasp_ok=True,
        plan_to_pregrasp_ok=True,
        endpoint_ik_ok=True,
        cartesian_fraction=0.90,
        collision_ok=True,
        joint_limits_ok=True,
        result="FAIL",
        reason="cartesian_fraction_low",
    )
    assert not is_fully_reachable_variant(fail)
    assert binary_color_for_cell(fail) == "black"
    summary = summarize_variant_search_results([fail, ok])
    row = single_cell_best_csv_row(
        label="sugar_box",
        x=0.5,
        y=-0.1,
        yaw=0.0,
        top_z=0.45,
        summary=summary,
    )
    assert row["cell_fully_reachable"] == "true"
    assert row["best_seed"] == "home"
    assert row["best_result"] == "OK"
    assert row["best_pregrasp_tcp_z"] == "0.5200"
    assert row["best_grasp_tcp_z"] == "0.4200"
    assert row["best_commanded_tcp_yaw_rad"] == "0.000000"
    assert row["best_cartesian_fraction"] == "0.98000"
    assert row["binary_color"] == "yellow"


def test_default_input_mode_mustard_operational() -> None:
    assert default_input_mode_for_label("mustard_bottle") == "operational_grasp_xy"
    assert default_input_mode_for_label("sugar_box") == "spawn_origin"


def test_mustard_operational_to_spawn_compensation() -> None:
    yaw = 1.6392
    op_x, op_y = 0.6600, 0.0600
    spawn_x, spawn_y = compute_compensated_spawn_xy_from_operational(
        op_x, op_y, yaw
    )
    assert math.isclose(spawn_x, 0.6570, abs_tol=0.002)
    assert math.isclose(spawn_y, 0.0360, abs_tol=0.002)
    coords = resolve_reachability_cell_coordinates(
        "mustard_bottle",
        op_x,
        op_y,
        yaw,
        input_mode="operational_grasp_xy",
    )
    assert coords.input_mode == "operational_grasp_xy"
    assert math.isclose(coords.operational_grasp_x, op_x, abs_tol=1e-4)
    assert math.isclose(coords.operational_grasp_y, op_y, abs_tol=1e-4)
    log = format_reachability_operational_center_log(coords)
    assert "[REACHABILITY_OPERATIONAL_CENTER]" in log
    assert "input_mode=operational_grasp_xy" in log


def test_mustard_detection_enriched_for_operational_cell() -> None:
    from panda_controller.demo_scene_reachability_scan import _vision_policy_exports

    export_grasp_policy_for_executor, _, _ = _vision_policy_exports()
    policy = export_grasp_policy_for_executor("mustard_bottle")
    detection, coords = build_detection_for_reachability_cell(
        "mustard_bottle",
        0.6600,
        0.0600,
        1.6392,
        table_z_m=0.27,
        policy=policy,
        input_mode="operational_grasp_xy",
    )
    cap = detection["chosen_target_center_base"]
    assert math.isclose(float(cap[0]), 0.6600, abs_tol=0.002)
    assert math.isclose(float(cap[1]), 0.0600, abs_tol=0.002)


def test_binary_cell_endpoint_ik_fail_is_black() -> None:
    row = ScanCellResult(
        label="mustard_bottle",
        x=0.6593,
        y=0.0603,
        yaw=1.6392,
        pregrasp_ok=True,
        plan_to_pregrasp_ok=True,
        pregrasp_ik_error_code="SUCCESS",
        endpoint_ik_ok=False,
        endpoint_ik_error_code="NO_IK_SOLUTION",
        cartesian_fraction=None,
        result="FAIL",
        reason="endpoint_ik_fail",
    )
    assert not is_binary_reachable_cell(row)
    assert binary_color_for_cell(row) == "black"
    log = format_reachability_binary_cell_decision_log(row)
    assert "binary_color=black" in log
    assert "endpoint_ik_ok=false" in log
    assert reachability_heatmap_cell_value(
        {"binary_color": "black", "result": "FAIL"}
    ) == 0.0


def test_scanner_grasp_hand_target_guard_and_mustard_tcp_conversion() -> None:
    ht = (0.0, 0.0, 0.10)
    quat = (0.0, 1.0, 0.0, 0.0)
    pre_tcp = (0.6593, 0.0603, 0.4909)
    gr_tcp = (0.6593, 0.0603, 0.4274)
    pre_hand, _ = hand_pose_from_desired_tcp(pre_tcp, quat, ht, (0.0, 0.0, 0.0, 1.0))
    gr_hand, _ = hand_pose_from_desired_tcp(gr_tcp, quat, ht, (0.0, 0.0, 0.0, 1.0))
    assert abs(float(pre_hand[2]) - 0.5909) < 1e-4
    assert abs(float(gr_hand[2]) - 0.5274) < 1e-4
    guard = validate_scanner_grasp_hand_target_z(gr_tcp, gr_hand, ht)
    assert guard["ok"] is True
    bad_hand = (gr_hand[0], gr_hand[1], 0.5909)
    bad_guard = validate_scanner_grasp_hand_target_z(gr_tcp, bad_hand, ht)
    assert bad_guard["ok"] is False
    assert bad_guard["reason"] == "scanner_grasp_hand_target_mismatch"
    row = ScanCellResult(
        label="mustard_bottle",
        x=0.6593,
        y=0.0603,
        yaw=1.6392,
        pregrasp_tcp=pre_tcp,
        grasp_tcp=gr_tcp,
        pregrasp_hand_target=pre_hand,
        grasp_hand_target=gr_hand,
        hand_to_tcp_translation=ht,
        quat=quat,
    )
    log = format_reachability_cell_debug_log(row)
    assert "grasp_hand_target=(0.6593, 0.0603, 0.5274)" in log
    assert "pregrasp_hand_target=(0.6593, 0.0603, 0.5909)" in log


def test_binary_cell_full_ok_is_yellow() -> None:
    row = ScanCellResult(
        label="mustard_bottle",
        x=0.5000,
        y=0.1000,
        yaw=1.5708,
        pregrasp_ok=True,
        plan_to_pregrasp_ok=True,
        endpoint_ik_ok=True,
        pregrasp_ik_error_code="SUCCESS",
        endpoint_ik_error_code="SUCCESS",
        cartesian_fraction=1.0,
        collision_ok=True,
        joint_limits_ok=True,
        result="OK",
        reason="reachable",
    )
    assert is_binary_reachable_cell(row)
    assert binary_color_for_cell(row) == "yellow"
    assert reachability_heatmap_cell_value(
        {"binary_color": "yellow", "cell_fully_reachable": "true"}
    ) == 1.0
