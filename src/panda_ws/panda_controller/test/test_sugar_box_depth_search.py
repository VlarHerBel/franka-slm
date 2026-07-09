"""Regresión: búsqueda depth/descend sugar_box demo_scene_02."""

from panda_controller.sugar_box_depth_search import (
    SUGAR_BOX_DEPTH_FROM_TOP_M,
    SUGAR_BOX_DESCEND_CANDIDATES_M,
    _safe_float,
    build_sugar_box_depth_descend_tcp_specs,
    format_sugar_box_depth_z_effective_log,
    format_sugar_box_depth_z_search_log,
    format_sugar_box_depth_z_selected_fail_log,
    format_sugar_box_depth_z_selected_log,
    format_sugar_box_prevalidate_defer_final_descend_log,
    format_sugar_box_deferred_prevalidate_authorization_log,
    iter_sugar_paired_prioritized_grid_specs,
    select_sugar_box_depth_z_variant,
    sugar_box_grasp_tcp_z_from_depth,
    sugar_box_pregrasp_tcp_z_from_grasp_and_descend,
)


def test_grasp_z_from_depth_demo_scene_02() -> None:
    top = 0.435
    assert sugar_box_grasp_tcp_z_from_depth(top_z_m=top, depth_from_top_m=0.028) == 0.407


def test_pregrasp_z_uses_descend_not_hard_clamp() -> None:
    top = 0.435
    gr_z = 0.411
    pre = sugar_box_pregrasp_tcp_z_from_grasp_and_descend(
        grasp_tcp_z=gr_z, descend_m=0.050
    )
    assert abs(pre - 0.461) < 1e-6
    assert pre < top + 0.055


def test_depth_spec_rejects_insufficient_clearance() -> None:
    specs = build_sugar_box_depth_descend_tcp_specs(
        xy=(0.630, -0.175),
        top_z_m=0.435,
        depth_candidates_m=(0.024,),
        descend_candidates_m=(0.010,),
        min_pregrasp_clearance_above_top_m=0.020,
    )
    assert specs == []


def test_depth_spec_effective_descend_matches_requested() -> None:
    specs = build_sugar_box_depth_descend_tcp_specs(
        xy=(0.630, -0.175),
        top_z_m=0.435,
        depth_candidates_m=(0.024,),
        descend_candidates_m=(0.050,),
        min_pregrasp_clearance_above_top_m=0.020,
    )
    assert len(specs) == 1
    spec = specs[0]
    assert abs(spec["grasp_tcp_z"] - 0.411) < 1e-6
    assert abs(spec["pregrasp_tcp_z"] - 0.461) < 1e-6
    assert abs(spec["effective_descend_m"] - 0.050) < 1e-6
    assert spec["clamped"] is False


def test_build_demo_specs_count() -> None:
    from panda_controller.sugar_box_depth_search import (
        SUGAR_BOX_DEMO_DESCEND_CANDIDATES_M,
        SUGAR_BOX_DEMO_DEPTH_FROM_TOP_M,
        build_sugar_box_demo_depth_descend_tcp_specs,
    )

    specs = build_sugar_box_demo_depth_descend_tcp_specs(
        xy=(0.630, -0.175),
        top_z_m=0.435,
        min_pregrasp_clearance_above_top_m=0.020,
    )
    max_specs = len(SUGAR_BOX_DEMO_DEPTH_FROM_TOP_M) * len(
        SUGAR_BOX_DEMO_DESCEND_CANDIDATES_M
    )
    assert 0 < len(specs) <= max_specs
    assert max_specs == 12


def test_build_specs_count_full_grid() -> None:
    specs = build_sugar_box_depth_descend_tcp_specs(
        xy=(0.630, -0.175),
        top_z_m=0.435,
        depth_candidates_m=SUGAR_BOX_DEPTH_FROM_TOP_M,
        descend_candidates_m=SUGAR_BOX_DESCEND_CANDIDATES_M,
        min_pregrasp_clearance_above_top_m=0.020,
    )
    max_specs = len(SUGAR_BOX_DEPTH_FROM_TOP_M) * len(SUGAR_BOX_DESCEND_CANDIDATES_M)
    assert 0 < len(specs) <= max_specs


def test_select_deferred_without_fraction() -> None:
    variants = [
        {
            "ok": True,
            "result": "OK",
            "reject_reason": "",
            "defer_final_descend": True,
            "depth_from_top_m": 0.020,
            "effective_descend_m": 0.050,
            "pregrasp_clearance_above_top": 0.035,
            "cartesian_fraction": None,
        },
        {
            "ok": True,
            "result": "OK",
            "reject_reason": "",
            "defer_final_descend": True,
            "depth_from_top_m": 0.024,
            "effective_descend_m": 0.055,
            "pregrasp_clearance_above_top": 0.037,
            "cartesian_fraction": None,
        },
    ]
    selected = select_sugar_box_depth_z_variant(
        variants, fraction_threshold=0.95, defer_final_descend=True
    )
    assert selected is not None
    assert selected["depth_from_top_m"] == 0.024


def test_select_deferred_none_when_no_valid() -> None:
    variants = [
        {
            "ok": True,
            "result": "FAIL",
            "reject_reason": "pregrasp_plan_fail",
            "defer_final_descend": True,
            "depth_from_top_m": 0.020,
            "cartesian_fraction": None,
        }
    ]
    assert (
        select_sugar_box_depth_z_variant(
            variants, fraction_threshold=0.95, defer_final_descend=True
        )
        is None
    )


def test_safe_float_handles_none_and_n_a() -> None:
    assert _safe_float(None, 1.0) == 1.0
    assert _safe_float("n/a", 2.0) == 2.0
    assert _safe_float("0.98", 0.0) == 0.98


def test_depth_search_logs() -> None:
    log = format_sugar_box_depth_z_effective_log(
        {
            "top_z": 0.435,
            "depth_from_top_m": 0.024,
            "grasp_tcp_z": 0.411,
            "requested_descend_m": 0.050,
            "effective_descend_m": 0.050,
            "pregrasp_clearance_above_top": 0.026,
            "clamped": False,
            "result": "OK",
        }
    )
    assert "[SUGAR_BOX_DEPTH_Z_EFFECTIVE]" in log
    assert "effective_descend_m=0.0500" in log
    assert "clamped=false" in log
    defer = format_sugar_box_prevalidate_defer_final_descend_log()
    assert "[SUGAR_BOX_PREVALIDATE_DEFER_FINAL_DESCEND]" in defer
    assert "OK_DEFER_DESCEND_TO_ACTUAL_PREGRASP" in defer
    auth = format_sugar_box_deferred_prevalidate_authorization_log(
        {
            "plan_before_result": "OK_PREGRASP_PENDING_DESCEND_VALIDATE",
            "selected_entry_target": "object_safe_above_tcp",
            "cartesian_descend_pending_at_pregrasp": True,
            "object_safe_above_plan_ok": True,
            "pregrasp_plan_ok": True,
            "motion_authorized": True,
            "result": "OK",
        }
    )
    assert "[SUGAR_BOX_DEFERRED_PREVALIDATE_AUTHORIZATION]" in auth
    assert "motion_authorized=true" in auth
    sel = format_sugar_box_depth_z_selected_log(
        {
            "depth_from_top_m": 0.022,
            "grasp_tcp_z": 0.413,
            "pregrasp_tcp_z": 0.458,
            "descend_m": 0.045,
            "yaw_variant": "top_down_yaw_pi",
            "defer_final_descend": True,
        }
    )
    assert "defer_final_descend=true" in sel
    fail = format_sugar_box_depth_z_selected_fail_log()
    assert "result=FAIL" in fail
    assert "no_valid_deferred_pregrasp_variant" in fail
    assert "[SUGAR_BOX_DEPTH_Z_SEARCH]" in format_sugar_box_depth_z_search_log(
        {
            "depth_from_top_m": 0.024,
            "grasp_tcp_z": 0.411,
            "pregrasp_tcp_z": 0.461,
            "descend_m": 0.050,
            "yaw_variant": "commanded_yaw",
            "cartesian_fraction": None,
            "grasp_ik_ok": True,
            "lift_prevalidate_ok": False,
            "result": "OK",
            "reject_reason": "",
        }
    )


def test_sugar_paired_prioritized_grid_specs_non_empty() -> None:
    specs = iter_sugar_paired_prioritized_grid_specs(
        xy=(0.60, 0.10),
        top_z=0.445,
        commanded_yaw_rad=0.0,
        selected_pregrasp_z=0.50,
        recommended_depth_from_top=0.022,
        max_candidates=36,
    )
    assert len(specs) > 0
    first = specs[0]
    assert first["priority"] == "canonical"
    assert "pre_plan" in first and "gr_plan" in first
    assert first["ik_seed_label"] in ("pick_workspace_ready", "object_safe_above")
