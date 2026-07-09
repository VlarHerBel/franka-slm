"""Tests búsqueda depth/descend mustard_bottle demo_scene_02 scanner-aligned."""

from __future__ import annotations

import math

from panda_controller.mustard_depth_search import (
    DEMO_SCENE_02_MUSTARD_MIN_GRASP_Z_M,
    DEMO_SCENE_02_MUSTARD_SCANNER_ALIGNED_COMMANDED_YAW_RAD,
    DEMO_SCENE_02_MUSTARD_SCANNER_ALIGNED_GRASP_Z,
    DEMO_SCENE_02_MUSTARD_SCANNER_ALIGNED_PREGRASP_Z,
    apply_mustard_scanner_aligned_pregrasp_to_seq,
    build_mustard_demo_descend_candidate_specs,
    build_mustard_grasp_elevation_fallback_specs,
    build_mustard_scanner_aligned_descend_spec,
    enforce_mustard_scanner_aligned_pregrasp_plan_targets,
    format_mustard_endpoint_ik_candidate_selected_log,
    format_mustard_scanner_aligned_candidate_log,
    format_mustard_scanner_aligned_pregrasp_applied_log,
    format_mustard_scanner_aligned_post_reachability_verify_log,
    mustard_bottle_extended_pick_scene_active,
    mustard_demo_scene_depth_search_active,
    resolve_mustard_descend_candidate_pose,
    resolve_mustard_descend_fail_reason,
    verify_mustard_scanner_aligned_post_reachability,
)


def test_mustard_demo_scene_depth_search_active() -> None:
    assert mustard_demo_scene_depth_search_active("mustard_bottle", "demo_scene_02")
    assert mustard_demo_scene_depth_search_active("mustard_bottle", "chips_mustard_01")
    assert not mustard_demo_scene_depth_search_active("mustard_bottle", "demo_scene_01")
    assert not mustard_demo_scene_depth_search_active("mustard_bottle", "two_boxes_01")
    assert not mustard_demo_scene_depth_search_active("sugar_box", "demo_scene_02")


def test_mustard_bottle_extended_pick_scene_active() -> None:
    assert mustard_bottle_extended_pick_scene_active("demo_scene_02")
    assert mustard_bottle_extended_pick_scene_active("chips_mustard_01")
    assert mustard_bottle_extended_pick_scene_active("deposit_02_cracker_chips")
    assert mustard_bottle_extended_pick_scene_active("deposit_03_mustard_only")
    assert not mustard_bottle_extended_pick_scene_active("two_boxes_03")
    assert not mustard_bottle_extended_pick_scene_active("deposit_full_1table")


def test_scanner_aligned_spec_first() -> None:
    pre = (0.662, 0.084, 0.492)
    gr_nom = (0.662, 0.084, 0.417)
    specs = build_mustard_demo_descend_candidate_specs(
        pre_plan=pre,
        top_z_m=0.427,
        xy=(0.662, 0.084),
        min_grasp_z_m=0.40,
        gr_plan_nominal=gr_nom,
        effective_top_z_m=0.4609,
    )
    assert specs
    first = specs[0]
    assert first.get("scanner_aligned") is True
    assert abs(float(first["grasp_tcp"][2]) - DEMO_SCENE_02_MUSTARD_SCANNER_ALIGNED_GRASP_Z) < 1e-6
    assert abs(float(first["pregrasp_tcp"][2]) - DEMO_SCENE_02_MUSTARD_SCANNER_ALIGNED_PREGRASP_Z) < 1e-6


def test_elevation_fallback_absolute_z() -> None:
    elev = build_mustard_grasp_elevation_fallback_specs(
        xy=(0.662, 0.084),
        pre_z=0.492,
        base_grasp_z=0.417,
        top_z_m=0.427,
        effective_top_z_m=0.4509,
    )
    grasp_zs = [round(float(s["grasp_tcp"][2]), 3) for s in elev]
    assert grasp_zs == [0.422, 0.427, 0.432]
    assert all(gz >= DEMO_SCENE_02_MUSTARD_MIN_GRASP_Z_M for gz in grasp_zs)


def test_demo_specs_exclude_grasp_below_floor() -> None:
    pre = (0.662, 0.084, 0.492)
    gr_nom = (0.662, 0.084, 0.417)
    specs = build_mustard_demo_descend_candidate_specs(
        pre_plan=pre,
        top_z_m=0.427,
        xy=(0.662, 0.084),
        min_grasp_z_m=0.40,
        gr_plan_nominal=gr_nom,
    )
    assert all(
        float(s["grasp_tcp"][2]) + 1e-6 >= DEMO_SCENE_02_MUSTARD_MIN_GRASP_Z_M
        for s in specs
    )


def test_scanner_aligned_pose_and_yaw() -> None:
    spec = build_mustard_scanner_aligned_descend_spec(
        xy=(0.662, 0.084),
        pre_plan=(0.662, 0.084, 0.492),
        top_z_m=0.427,
        effective_top_z_m=0.4509,
    )
    assert spec is not None
    pre, gr, yaw, source = resolve_mustard_descend_candidate_pose(
        spec,
        nominal_pre_plan=(0.662, 0.084, 0.4918),
        nominal_gr_plan=(0.662, 0.084, 0.4168),
        variant_commanded_yaw_rad=0.0684,
    )
    assert abs(float(pre[2]) - 0.4909) < 1e-6
    assert abs(float(gr[2]) - (0.4509 - 0.046)) < 1e-6
    assert abs(float(yaw) - DEMO_SCENE_02_MUSTARD_SCANNER_ALIGNED_COMMANDED_YAW_RAD) < 1e-6
    assert source == "scanner_aligned"


def test_fail_reason_prefers_endpoint_ik_over_depth_shallow() -> None:
    assert (
        resolve_mustard_descend_fail_reason(
            last_reason="depth_too_shallow",
            endpoint_ik_attempted=True,
            endpoint_ik_failed=True,
        )
        == "endpoint_ik_fail"
    )


def test_mustard_scanner_aligned_candidate_log_format() -> None:
    log = format_mustard_scanner_aligned_candidate_log(
        {
            "controller_pregrasp_tcp_z": 0.4918,
            "controller_grasp_tcp_z": 0.4168,
            "selected": True,
            "result": "EVALUATING",
        }
    )
    assert "[MUSTARD_SCANNER_ALIGNED_CANDIDATE]" in log
    assert "scanner_pregrasp_tcp_z=0.4909" in log
    assert "scanner_grasp_tcp_z=0.4149" in log
    assert "selected=true" in log
    assert "result=EVALUATING" in log


def test_mustard_endpoint_ik_candidate_selected_log_format() -> None:
    log = format_mustard_endpoint_ik_candidate_selected_log(
        {
            "pregrasp_tcp_z": 0.4909,
            "grasp_tcp_z": 0.4149,
            "commanded_tcp_yaw_rad": -3.073189,
            "source": "scanner_aligned",
        }
    )
    assert "[MUSTARD_ENDPOINT_IK_CANDIDATE_SELECTED]" in log
    assert "grasp_tcp_z=0.4149" in log
    assert "source=scanner_aligned" in log


def test_apply_scanner_aligned_pregrasp_overrides_high_approach() -> None:
    candidate = {
        "label": "mustard_bottle",
        "grasp_strategy": "tall_object_topdown",
    }
    seq = {
        "pregrasp_tcp": (0.659, 0.060, 0.5169),
        "grasp_tcp": (0.659, 0.060, 0.4149),
        "safe_pregrasp_tcp": (0.659, 0.060, 0.6200),
    }
    applied = apply_mustard_scanner_aligned_pregrasp_to_seq(
        candidate,
        seq,
        top_z=0.4609,
        scene_id="demo_scene_02",
    )
    assert applied is not None
    assert math.isclose(float(seq["pregrasp_tcp"][2]), 0.4909, abs_tol=1e-6)
    assert math.isclose(float(seq["grasp_tcp"][2]), 0.4149, abs_tol=1e-6)
    assert math.isclose(float(seq["final_descend_m"]), 0.076, abs_tol=1e-3)
    assert bool(candidate.get("mustard_scanner_aligned_pregrasp_locked")) is True


def test_scanner_aligned_pregrasp_applied_log_format() -> None:
    log = format_mustard_scanner_aligned_pregrasp_applied_log(
        {
            "old_pregrasp_tcp_z": 0.5169,
            "new_pregrasp_tcp_z": 0.4909,
            "grasp_tcp_z": 0.4149,
            "top_z": 0.4609,
            "source": "scanner_aligned_contract",
            "result": "OK",
        }
    )
    assert "[MUSTARD_SCANNER_ALIGNED_PREGRASP_APPLIED]" in log
    assert "old_pregrasp_tcp_z=0.5169" in log
    assert "new_pregrasp_tcp_z=0.4909" in log
    assert "source=scanner_aligned_contract" in log


def test_enforce_scanner_aligned_pregrasp_restores_locked_z() -> None:
    candidate = {
        "label": "mustard_bottle",
        "grasp_strategy": "tall_object_topdown",
        "mustard_scanner_aligned_pregrasp_locked": True,
        "_mustard_scanner_aligned_locked_pregrasp_tcp": [0.659, 0.060, 0.4909],
        "_mustard_scanner_aligned_locked_grasp_tcp": [0.659, 0.060, 0.4149],
    }
    plan_targets = {
        "pregrasp_tcp": (0.659, 0.060, 0.5020),
        "grasp_tcp": (0.659, 0.060, 0.4149),
    }
    ok, reason, _sel, _exp = enforce_mustard_scanner_aligned_pregrasp_plan_targets(
        candidate, plan_targets
    )
    assert ok is True
    assert reason == "locked"
    assert math.isclose(float(plan_targets["pregrasp_tcp"][2]), 0.4909, abs_tol=1e-6)


def test_post_reachability_verify_pass_and_fail() -> None:
    ok, fields = verify_mustard_scanner_aligned_post_reachability(
        pregrasp_tcp_z=0.4909,
        grasp_tcp_z=0.4149,
        top_z=0.4609,
    )
    assert ok is True
    assert fields["result"] == "OK"
    bad_ok, bad_fields = verify_mustard_scanner_aligned_post_reachability(
        pregrasp_tcp_z=0.5020,
        grasp_tcp_z=0.4149,
        top_z=0.4609,
    )
    assert bad_ok is False
    assert bad_fields["result"] == "FAIL"
    log = format_mustard_scanner_aligned_post_reachability_verify_log(bad_fields)
    assert "[MUSTARD_SCANNER_ALIGNED_POST_REACHABILITY_VERIFY]" in log
    assert "expected_pregrasp_tcp_z=0.4909" in log
