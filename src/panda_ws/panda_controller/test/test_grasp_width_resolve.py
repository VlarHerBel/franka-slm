"""Regresión: pudding_box edge grasp no debe usar db width 0.089 en check yaw."""

import math

from panda_controller.grasp_width_resolve import (
    compute_yaw_uncertainty_effective_width,
    resolve_controller_required_width,
)


def test_pudding_edge_resolves_effective_width_not_db() -> None:
    obj = {
        "label": "pudding_box",
        "grasp_strategy": "edge_grasp",
        "edge_grasp_requested": True,
        "required_grasp_width_m": 0.035,
        "effective_required_grasp_width_m": 0.035,
        "db_required_width_m": 0.089,
        "measured_required_width_m": 0.089,
        "required_width_source": "edge_short_axis_or_height",
    }
    w, src = resolve_controller_required_width(obj)
    assert w == 0.035
    assert src == "effective_required_grasp_width_m"


def test_pudding_yaw_uncertainty_check_passes_with_high_confidence() -> None:
    obj = {
        "label": "pudding_box",
        "grasp_strategy": "edge_grasp",
        "edge_grasp_requested": True,
        "required_grasp_width_m": 0.035,
        "effective_required_grasp_width_m": 0.035,
        "db_required_width_m": 0.089,
        "measured_required_width_m": 0.089,
        "yaw_confidence": 1.0,
        "collision_dims": {"db_dims": [0.110, 0.089, 0.035]},
    }
    policy_w, _ = resolve_controller_required_width(obj)
    open_total = 0.0798
    req_w, eff_w, yaw_unc, _, ok = compute_yaw_uncertainty_effective_width(
        policy_width_m=policy_w,
        open_total_m=open_total,
        yaw_confidence=1.0,
        edge_grasp_requested=True,
        grasp_strategy="edge_grasp",
        collision_db_dims_xy=(0.110, 0.089),
    )
    assert req_w == 0.035
    assert abs(eff_w - 0.035) < 1e-6
    assert yaw_unc == 0.0
    assert ok is True
    assert eff_w <= open_total - 0.003


def test_non_edge_uses_footprint_long_for_yaw_inflation() -> None:
    policy_w = 0.038
    open_total = 0.080
    req_w, eff_w, yaw_unc, _, ok = compute_yaw_uncertainty_effective_width(
        policy_width_m=policy_w,
        open_total_m=open_total,
        yaw_confidence=0.5,
        edge_grasp_requested=False,
        grasp_strategy="top_down_short_axis",
        collision_db_dims_xy=(0.089, 0.038),
    )
    assert req_w == 0.038
    assert yaw_unc > 0.0
    assert eff_w >= policy_w
    assert isinstance(ok, bool)
