"""Tests búsqueda XY reachability mustard_bottle demo_scene_02."""

from __future__ import annotations

from panda_controller.mustard_xy_reachability_search import (
    DEMO_SCENE_02_MUSTARD_BODY_CENTER_XY,
    build_mustard_demo_xy_reachability_specs,
    build_mustard_operational_xy_reachability_specs,
    build_mustard_scanner_locked_demo_xy_specs,
    build_mustard_xy_anchor_points,
    build_mustard_yaw_candidates,
    format_mustard_grasp_candidate_selected_log,
    format_mustard_xy_reachability_candidate_log,
    mustard_xy_error_acceptable,
    mustard_xy_error_to_cap_m,
    resolve_mustard_body_center_xy,
    resolve_mustard_cap_center_xy,
)


def test_xy_anchor_points_include_cap_body_interpolated_and_offsets() -> None:
    cap = (0.6623, 0.0843)
    body = DEMO_SCENE_02_MUSTARD_BODY_CENTER_XY
    major = (1.0, 0.0)
    minor = (0.0, 1.0)
    anchors = build_mustard_xy_anchor_points(
        cap_center_xy=cap,
        body_center_xy=body,
        major_axis=major,
        minor_axis=minor,
    )
    sources = [a[0] for a in anchors]
    assert "cap_center" in sources
    assert "body_center" in sources
    assert sources.count("interpolated") == 3
    assert "axis_offset_major" in sources
    assert "axis_offset_minor" in sources


def test_yaw_candidates_include_scanner_and_controller_plus_pi() -> None:
    yaws = build_mustard_yaw_candidates(controller_yaw_rad=0.0684)
    assert len(yaws) >= 2
    assert any(abs(y + 3.073189) < 0.01 or abs(y - 0.0684) < 0.01 for y in yaws)


def test_scanner_locked_specs_are_compact_cap_center_only() -> None:
    candidate = {
        "known_box_center_base": [0.660, 0.060, 0.43],
        "grasp_center_base": [0.6623, 0.0843, 0.43],
        "major_axis_xy": [1.0, 0.0],
        "minor_axis_xy": [0.0, 1.0],
    }
    full = build_mustard_demo_xy_reachability_specs(
        candidate,
        controller_grasp_xy=(0.6623, 0.0843),
        controller_yaw_rad=0.0684,
        effective_top_z_m=0.4509,
        min_grasp_z_m=0.40,
    )
    compact = build_mustard_scanner_locked_demo_xy_specs(
        candidate,
        controller_grasp_xy=(0.6623, 0.0843),
        controller_yaw_rad=0.0684,
        controller_pre_plan=(0.6623, 0.0843, 0.4909),
        controller_gr_plan=(0.6623, 0.0843, 0.4360),
        effective_top_z_m=0.4509,
        min_grasp_z_m=0.40,
    )
    assert len(compact) < len(full)
    assert all(s.get("xy_source") == "cap_center" for s in compact)
    assert any(bool(s.get("scanner_aligned")) for s in compact)


def test_xy_reachability_specs_prioritize_cap_center_scanner_z() -> None:
    candidate = {
        "known_box_center_base": [0.660, 0.060, 0.43],
        "grasp_center_base": [0.6623, 0.0843, 0.43],
        "major_axis_xy": [1.0, 0.0],
        "minor_axis_xy": [0.0, 1.0],
    }
    specs = build_mustard_demo_xy_reachability_specs(
        candidate,
        controller_grasp_xy=(0.6623, 0.0843),
        controller_yaw_rad=0.0684,
        effective_top_z_m=0.4509,
        min_grasp_z_m=0.40,
    )
    assert specs
    first = specs[0]
    assert first.get("xy_source") == "cap_center"
    assert abs(float(first["grasp_tcp"][2]) - 0.4269) < 1e-4


def test_body_center_resolved_from_candidate() -> None:
    candidate = {"known_box_center_base": [0.660, 0.060, 0.43]}
    body = resolve_mustard_body_center_xy(candidate)
    assert abs(body[0] - 0.660) < 1e-6
    assert abs(body[1] - 0.060) < 1e-6


def test_cap_center_prefers_grasp_center_base() -> None:
    candidate = {"grasp_center_base": [0.6623, 0.0843, 0.43]}
    cap = resolve_mustard_cap_center_xy(
        candidate, controller_grasp_xy=(0.0, 0.0)
    )
    assert abs(cap[0] - 0.6623) < 1e-6
    assert abs(cap[1] - 0.0843) < 1e-6


def test_xy_error_acceptance() -> None:
    assert mustard_xy_error_acceptable(
        xy_source="body_center", xy_error_to_cap_m=0.024
    )
    assert mustard_xy_error_acceptable(
        xy_source="cap_center", xy_error_to_cap_m=0.005
    )
    assert not mustard_xy_error_acceptable(
        xy_source="cap_center", xy_error_to_cap_m=0.020
    )


def test_xy_reachability_log_format() -> None:
    log = format_mustard_xy_reachability_candidate_log(
        {
            "source": "body_center",
            "grasp_xy": (0.660, 0.060),
            "cap_center_xy": (0.6623, 0.0843),
            "body_center_xy": (0.660, 0.060),
            "xy_error_to_cap_m": mustard_xy_error_to_cap_m(
                (0.660, 0.060), (0.6623, 0.0843)
            ),
            "pregrasp_tcp_z": 0.4909,
            "grasp_tcp_z": 0.4269,
            "commanded_tcp_yaw_rad": -3.073189,
            "endpoint_ik_ok": "true",
            "stepwise_descend_ok": "true",
            "result": "OK",
        }
    )
    assert "[MUSTARD_XY_REACHABILITY_CANDIDATE]" in log
    assert "source=body_center" in log
    assert "result=OK" in log


def test_grasp_candidate_selected_log_format() -> None:
    log = format_mustard_grasp_candidate_selected_log(
        {
            "source": "body_center",
            "grasp_xy": (0.660, 0.060),
            "grasp_tcp_z": 0.4269,
            "commanded_tcp_yaw_rad": -3.073189,
            "result": "OK",
        }
    )
    assert "[MUSTARD_GRASP_CANDIDATE_SELECTED]" in log
    assert "grasp_xy=(0.6600, 0.0600)" in log


def test_operational_xy_specs_are_compact_for_chips_mustard() -> None:
    candidate = {
        "grasp_center_base": [0.545, -0.068, 0.437],
        "top_z_m": 0.437,
    }
    full = build_mustard_demo_xy_reachability_specs(
        candidate,
        controller_grasp_xy=(0.545, -0.068),
        controller_yaw_rad=-0.532,
        effective_top_z_m=0.437,
        min_grasp_z_m=0.40,
    )
    operational = build_mustard_operational_xy_reachability_specs(
        candidate,
        controller_grasp_xy=(0.545, -0.068),
        controller_yaw_rad=-0.532,
        controller_pre_plan=(0.545, -0.068, 0.502),
        controller_gr_plan=(0.545, -0.068, 0.426),
        effective_top_z_m=0.437,
        min_grasp_z_m=0.40,
    )
    assert len(operational) <= 4
    assert len(operational) < len(full)
    assert all(s.get("xy_source") == "cap_center" for s in operational)
