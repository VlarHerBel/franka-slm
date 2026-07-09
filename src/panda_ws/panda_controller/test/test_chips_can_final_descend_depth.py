"""Tests búsqueda profundidad descenso final chips_can."""

from panda_controller.chips_can_final_descend_depth import (
    chips_can_grasp_tcp_z_from_depth,
    chips_can_high_route_contract_ok,
    infer_chips_can_final_descend_blocker,
    select_chips_can_final_descend_depth_variant,
)
from panda_controller.chips_can_high_route_prevalidate import (
    select_chips_can_high_route_pending_descend_variant,
)


def test_grasp_tcp_z_from_depth() -> None:
    assert chips_can_grasp_tcp_z_from_depth(top_z_m=0.510, depth_from_top_m=0.035) == 0.475


def test_high_route_contract_rejects_collapsed_high() -> None:
    assert not chips_can_high_route_contract_ok(
        object_high_tcp_z=0.610,
        low_pregrasp_tcp_z=0.610,
    )
    assert chips_can_high_route_contract_ok(
        object_high_tcp_z=0.660,
        low_pregrasp_tcp_z=0.610,
    )


def test_select_deepest_passing_depth_variant() -> None:
    variants = [
        {"depth_from_top_m": 0.035, "cartesian_fraction": 0.49, "ok": False},
        {"depth_from_top_m": 0.020, "cartesian_fraction": 0.98, "ok": True},
        {"depth_from_top_m": 0.010, "cartesian_fraction": 1.0, "ok": True},
    ]
    selected = select_chips_can_final_descend_depth_variant(
        variants, fraction_threshold=0.95
    )
    assert selected is not None
    assert selected["depth_from_top_m"] == 0.020


def test_pending_variant_rejects_degenerate_high_equal_low() -> None:
    variants = [
        {
            "variant_name": "degenerate",
            "joint_dist": 0.5,
            "object_high_plan_ok": True,
            "object_high_to_low_fraction": 1.0,
            "entry_tcp": (0.52, -0.095, 0.610),
        },
        {
            "variant_name": "valid_high",
            "joint_dist": 1.0,
            "object_high_plan_ok": True,
            "object_high_to_low_fraction": 1.0,
            "entry_tcp": (0.52, -0.095, 0.660),
        },
    ]
    selected = select_chips_can_high_route_pending_descend_variant(
        variants,
        fraction_threshold=0.95,
        low_pregrasp_tcp_z=0.610,
    )
    assert selected is not None
    assert selected["variant_name"] == "valid_high"


def test_infer_blocker_obstacle_collision() -> None:
    assert (
        infer_chips_can_final_descend_blocker(
            with_obstacles_fraction=0.49,
            without_remaining_obstacles_fraction=0.98,
            fraction_threshold=0.95,
        )
        == "obstacle_collision"
    )


def test_infer_blocker_kinematic_limit() -> None:
    assert (
        infer_chips_can_final_descend_blocker(
            with_obstacles_fraction=0.49,
            without_remaining_obstacles_fraction=0.49,
            fraction_threshold=0.95,
        )
        == "kinematic_limit"
    )
