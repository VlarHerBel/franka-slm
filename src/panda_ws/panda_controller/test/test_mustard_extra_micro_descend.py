"""Tests microdescenso extra mustard post-cartesiano."""

import pytest

from panda_controller.mustard_extra_micro_descend import (
    MUSTARD_EXTRA_MICRO_DESCEND_STEP_Z_TOLERANCE_M,
    MUSTARD_GRASP_TCP_Z_TOLERANCE_M,
    build_mustard_extra_micro_descend_steps,
    build_mustard_micro_descend_z_targets,
    evaluate_mustard_post_descend_depth_verify,
    extend_mustard_micro_descend_z_targets_to_grasp,
    mustard_micro_step_target_reached,
    resolve_mustard_descend_shortfall_extra_m,
    resolve_mustard_effective_min_required_depth_m,
    resolve_mustard_extra_micro_descend_apply_m,
    resolve_mustard_nominal_grasp_tcp_z,
)


def test_apply_m_limited_by_palm_bridge_clearance() -> None:
    applied, reason = resolve_mustard_extra_micro_descend_apply_m(
        requested_extra_m=0.010,
        max_extra_m=0.010,
        palm_clearance_observed_m=0.0118,
        min_bridge_clearance_after_m=0.0015,
    )
    assert applied == 0.010
    assert reason == "palm_bridge_limited"


def test_apply_m_skips_when_clearance_insufficient() -> None:
    applied, reason = resolve_mustard_extra_micro_descend_apply_m(
        requested_extra_m=0.010,
        max_extra_m=0.010,
        palm_clearance_observed_m=0.0010,
        min_bridge_clearance_after_m=0.0015,
    )
    assert applied == 0.0
    assert reason == "palm_bridge_clearance_insufficient"


def test_build_steps_10mm_in_2mm_increments() -> None:
    steps = build_mustard_extra_micro_descend_steps(
        start_tcp_z=0.4357,
        applied_extra_m=0.010,
        step_m=0.002,
    )
    assert len(steps) == 5
    assert steps[-1] == pytest.approx(0.4257, abs=1e-4)


def test_shortfall_extra_compensates_cartesian_undershoot() -> None:
    extra, reason = resolve_mustard_descend_shortfall_extra_m(
        requested_tcp_z=0.4360,
        actual_tcp_z=0.4450,
        max_extra_m=0.010,
    )
    assert extra == pytest.approx(0.009, abs=1e-4)
    assert reason == "auto_shortfall_compensation"


def test_runtime_depth_gate_passes_after_10mm_descend() -> None:
    ok, depth = evaluate_mustard_post_descend_depth_verify(
        top_z_m=0.4609,
        actual_tcp_z=0.4257,
        min_required_depth_from_top_m=0.034,
    )
    assert ok is True
    assert abs(depth - 0.0352) < 1e-4


def test_micro_step_within_z_tolerance_counts_as_reached() -> None:
    assert mustard_micro_step_target_reached(
        0.4434,
        0.4443,
        tolerance_m=MUSTARD_EXTRA_MICRO_DESCEND_STEP_Z_TOLERANCE_M,
    )
    assert not mustard_micro_step_target_reached(0.4434, 0.4360)


def test_resolve_nominal_grasp_tcp_z_from_shortfall_key() -> None:
    grasp = resolve_mustard_nominal_grasp_tcp_z(
        {"_mustard_nominal_grasp_tcp_z_m": 0.4360},
        top_z_m=0.4700,
    )
    assert grasp == pytest.approx(0.4360)


def test_effective_min_depth_caps_recommended_above_configured() -> None:
    effective = resolve_mustard_effective_min_required_depth_m(
        configured_min_m=0.034,
        recommended_depth_m=0.035,
        floor_m=0.028,
    )
    assert effective == pytest.approx(0.034)


def test_extend_z_targets_reaches_nominal_grasp() -> None:
    extra = extend_mustard_micro_descend_z_targets_to_grasp(
        start_tcp_z=0.4410,
        nominal_grasp_tcp_z=0.4360,
        step_m=0.002,
    )
    assert extra[-1] == pytest.approx(0.4360, abs=1e-4)
    assert len(extra) == 3


def test_build_targets_include_grasp_extension() -> None:
    targets = build_mustard_micro_descend_z_targets(
        start_tcp_z=0.4496,
        applied_extra_m=0.010,
        step_m=0.002,
        nominal_grasp_tcp_z=0.4360,
    )
    assert targets[-1] == pytest.approx(0.4360, abs=1e-4)
    assert len(targets) > 5


def test_depth_passes_with_grasp_z_tolerance_near_nominal() -> None:
    ok, depth = evaluate_mustard_post_descend_depth_verify(
        top_z_m=0.4700,
        actual_tcp_z=0.4410,
        min_required_depth_from_top_m=0.034,
        nominal_grasp_tcp_z=0.4360,
        grasp_tcp_z_tolerance_m=MUSTARD_GRASP_TCP_Z_TOLERANCE_M,
    )
    assert depth == pytest.approx(0.029, abs=1e-4)
    assert ok is True


def test_depth_passes_when_tcp_at_nominal_grasp_slightly_shallow() -> None:
    """Runtime case: TCP at grasp Z but collision top vs mesh differs by ~0.2 mm."""
    ok, depth = evaluate_mustard_post_descend_depth_verify(
        top_z_m=0.4609,
        actual_tcp_z=0.4261,
        min_required_depth_from_top_m=0.035,
        nominal_grasp_tcp_z=0.4264,
        grasp_tcp_z_tolerance_m=0.004,
        depth_tolerance_m=0.002,
    )
    assert depth == pytest.approx(0.0348, abs=1e-4)
    assert ok is True


def test_depth_tolerance_covers_sub_mm_gap_to_min() -> None:
    ok, depth = evaluate_mustard_post_descend_depth_verify(
        top_z_m=0.4609,
        actual_tcp_z=0.4261,
        min_required_depth_from_top_m=0.035,
        depth_tolerance_m=0.002,
    )
    assert depth == pytest.approx(0.0348, abs=1e-4)
    assert ok is True
