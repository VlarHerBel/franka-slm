"""Tests diagnóstico cartesiano paired/grid."""

import math

from panda_controller.paired_cartesian_descend_diag import (
    build_cartesian_sweep_z_targets,
    compare_descend_profiles,
    interpolate_failed_tcp_z,
)
from panda_controller.paired_cracker_box_candidate_grid import (
    PAIRED_GRID_MAX_PRIORITIZED_CANDIDATES,
    iter_cracker_paired_prioritized_grid_specs,
    resolve_paired_grid_search_mode,
)
from panda_controller.paired_joint7_offline_sim import (
    simulate_joint7_gap_alignment_offline_fast,
)


def test_prioritized_grid_max_36() -> None:
    specs = iter_cracker_paired_prioritized_grid_specs(
        xy=(0.455, 0.115),
        top_z=0.470,
        commanded_yaw_rad=0.2,
        selected_pregrasp_z=0.575,
        recommended_depth_from_top=0.033,
    )
    assert len(specs) <= PAIRED_GRID_MAX_PRIORITIZED_CANDIDATES
    assert specs[0]["priority"] == "canonical"


def test_resolve_grid_mode_full_debug_flag() -> None:
    assert (
        resolve_paired_grid_search_mode(
            mode_param="prioritized",
            enable_full_640_debug=True,
        )
        == "full_debug"
    )
    assert (
        resolve_paired_grid_search_mode(
            mode_param="prioritized",
            enable_full_640_debug=False,
        )
        == "prioritized"
    )


def test_interpolate_failed_tcp_z_half() -> None:
    z = interpolate_failed_tcp_z(0.575, 0.437, 0.5)
    assert z is not None
    assert abs(z - 0.506) < 0.01


def test_build_sweep_z_targets_includes_failed_and_target() -> None:
    zs = build_cartesian_sweep_z_targets(
        start_tcp_z=0.575,
        target_tcp_z=0.437,
        failed_tcp_z=0.4975,
        extra_z=(0.485, 0.470),
    )
    assert 0.437 in zs
    assert any(abs(z - 0.4975) < 1e-3 for z in zs)


def test_compare_descend_profiles_same() -> None:
    profile = {
        "link": "panda_hand",
        "eef_step": 0.0025,
        "jump_threshold": 0.0,
        "avoid_collisions": True,
        "waypoint_count": 1,
        "use_grasp_tcp": False,
        "planning_frame": "panda_link0",
        "target_collision_policy": "present",
        "target_hand_pose": (0.45, 0.11, 0.44),
        "target_hand_quat": (0.0, 1.0, 0.0, 0.0),
        "start_state_source": "aligned_pregrasp_after_joint7",
    }
    result, diffs = compare_descend_profiles(profile, dict(profile))
    assert result == "SAME"
    assert not diffs


def test_joint7_fast_mode_reduces_error() -> None:
    import numpy as np

    desired = np.array([1.0, 0.0])

    def observed_fn(pos):
        return np.array([math.cos(float(pos[6])), math.sin(float(pos[6]))])

    result = simulate_joint7_gap_alignment_offline_fast(
        [0.0] * 6 + [math.radians(28.0)],
        desired_gap_axis_xy=desired,
        observed_gap_axis_fn=observed_fn,
        micro_probe_rad=0.04,
        max_jump_rad=1.2,
        fine_steps=3,
        target_deg=5.0,
    )
    assert result["result"] == "OK"
    assert result["mode"] == "fast"
    assert int(result["iterations"]) <= 4


def test_runtime_descend_profile_missing_collision_attr_does_not_crash() -> None:
    from panda_controller.perception_to_pregrasp_test import PerceptionToPregraspTest

    class _StubMissingCollisionAttr:
        _moveit_target_link = "panda_hand"
        _cartesian_avoid_collisions = True
        _use_grasp_tcp = False
        _planning_frame = "panda_link0"
        _add_target_collision_override = None

        def _default_cartesian_eef_step(self) -> float:
            return 0.0025

        def _default_cartesian_jump_threshold(self) -> float:
            return 0.0

        def _tcp_to_moveit_hand_pose(self, tcp, quat):
            return (float(tcp[0]), float(tcp[1]), float(tcp[2])), tuple(quat)

        def _include_target_collision(self, candidate):
            return True

        def _resolve_add_target_collision_policy(self) -> str:
            return PerceptionToPregraspTest._resolve_add_target_collision_policy(self)

        def _runtime_descend_profile(self, *, gr_plan, quat):
            return PerceptionToPregraspTest._runtime_descend_profile(
                self, gr_plan=gr_plan, quat=quat
            )

    stub = _StubMissingCollisionAttr()
    profile = stub._runtime_descend_profile(
        gr_plan=(0.455, 0.115, 0.437),
        quat=(0.0, 1.0, 0.0, 0.0),
    )
    assert profile["link"] == "panda_hand"
    assert profile["target_collision_policy"] == "removed_before_descend_runtime"
    assert stub._resolve_add_target_collision_policy() == "until_pregrasp"
