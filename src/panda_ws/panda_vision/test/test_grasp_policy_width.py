"""Tests de anchura efectiva de grasp policy (demo multiobjeto)."""

from panda_vision.grasp.object_grasp_policy import (
    export_grasp_policy_for_executor,
    get_grasp_policy,
    resolve_effective_required_grasp_width,
)


def test_pudding_box_uses_short_axis_width_for_edge_grasp() -> None:
    policy = get_grasp_policy("pudding_box", use_measured_dimensions=False)
    assert policy["primary_strategy"] == "edge_grasp"
    assert float(policy["required_grasp_width_m"]) <= 0.040
    assert policy["required_width_source"] in (
        "edge_short_axis_or_height",
        "db_minor_axis",
        "dims_minor_axis",
    )
    open_total = 2.0 * float(policy["recommended_open_joint_m"])
    margin = open_total - float(policy["required_grasp_width_m"])
    assert margin >= float(policy["min_gripper_total_margin_m"])


def test_pudding_export_not_major_axis_width() -> None:
    exp = export_grasp_policy_for_executor("pudding_box")
    assert float(exp["required_grasp_width_m"]) == 0.035
    assert float(exp["db_required_width_m"]) == 0.089
    assert exp["dims_lwh"] == [0.110, 0.089, 0.035]
    assert float(exp["recommended_grasp_depth_from_top_m"]) == 0.008
    assert float(exp["max_cartesian_descend_m"]) == 0.038


def test_sugar_box_required_width_is_short_axis() -> None:
    policy = get_grasp_policy("sugar_box", use_measured_dimensions=False)
    assert abs(float(policy["required_grasp_width_m"]) - 0.038) < 1e-6
    exp = export_grasp_policy_for_executor("sugar_box")
    assert abs(float(exp["recommended_grasp_depth_from_top_m"]) - 0.028) < 1e-6
    assert float(exp["max_cartesian_descend_m"]) == 0.055
    assert float(exp["release_z_m"]) == 0.375


def test_cracker_box_required_width_near_validated_60mm() -> None:
    policy = get_grasp_policy("cracker_box", use_measured_dimensions=False)
    assert abs(float(policy["required_grasp_width_m"]) - 0.060) < 1e-6


def test_resolve_effective_width_prefers_height_for_low_box_wide() -> None:
    entry = {
        "shape": "low_box_wide",
        "dims": (0.110, 0.089, 0.035),
        "required_width": 0.089,
    }
    width, source = resolve_effective_required_grasp_width(
        entry, "edge_grasp", [], 0.089
    )
    assert width == 0.035
    assert source == "edge_short_axis_or_height"
