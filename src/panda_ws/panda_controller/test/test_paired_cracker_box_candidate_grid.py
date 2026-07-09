"""Tests grid paired cracker_box."""

import math

from panda_controller.demo_cracker_box_cartesian_prevalidate import (
    evaluate_paired_safe_geometric_descend_fallback,
)
from panda_controller.paired_cracker_box_candidate_grid import (
    build_cracker_paired_yaw_variants,
    cracker_tcp_targets_for_grid,
    iter_cracker_paired_grid_specs,
    iter_cracker_paired_prioritized_grid_specs,
    PAIRED_GRID_MAX_PRIORITIZED_CANDIDATES,
)


def test_cracker_yaw_grid_includes_pi_and_offsets() -> None:
    yaws = build_cracker_paired_yaw_variants(0.3)
    names = {n for n, _ in yaws}
    assert "yaw" in names
    assert "yaw_pi" in names
    assert "yaw_+5deg" in names
    assert "yaw_pi_-10deg" in names
    assert len(yaws) >= 10


def test_grid_spec_count() -> None:
    specs = iter_cracker_paired_grid_specs(
        xy=(0.455, 0.115),
        top_z=0.470,
        base_yaw_rad=0.2,
    )
    # 10 yaws * 4 pregrasp_z * 4 depth * 4 ik seeds
    assert len(specs) == 10 * 4 * 4 * 4


def test_grasp_z_from_depth_from_top() -> None:
    pre, gr, depth = cracker_tcp_targets_for_grid(
        xy=(0.455, 0.115),
        top_z=0.470,
        pregrasp_tcp_z=0.575,
        depth_from_top_m=0.033,
    )
    assert abs(depth - 0.033) < 1e-9
    assert abs(gr[2] - (0.470 - 0.033)) < 1e-9
    assert pre[2] == 0.575


def test_prioritized_grid_max_36() -> None:
    specs = iter_cracker_paired_prioritized_grid_specs(
        xy=(0.455, 0.115),
        top_z=0.470,
        commanded_yaw_rad=0.2,
        selected_pregrasp_z=0.575,
        recommended_depth_from_top=0.033,
        max_candidates=PAIRED_GRID_MAX_PRIORITIZED_CANDIDATES,
    )
    assert len(specs) <= PAIRED_GRID_MAX_PRIORITIZED_CANDIDATES
    assert specs[0]["priority"] == "canonical"


def test_paired_safe_geometric_requires_endpoint_ik() -> None:
    ok, reason = evaluate_paired_safe_geometric_descend_fallback(
        fk_contract_ok=True,
        endpoint_ik_ok=False,
        label="cracker_box",
        demo_authoritative_scene=True,
        scene_id="demo_scene_02",
        stage_label="pregrasp_to_grasp_cartesian",
        target_collision_removed=True,
        object_safe_above_to_pregrasp_ok=True,
        pre_plan=(0.455, 0.115, 0.575),
        gr_plan=(0.455, 0.115, 0.437),
        candidate={"label": "cracker_box", "top_z_m": 0.470},
        scene_obstacles=[],
        moveit_fraction=0.5,
        table_z_m=0.40,
    )
    assert not ok
    assert reason == "endpoint_ik_not_ok"
