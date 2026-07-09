"""Regresión: fallback cartesiano sugar_box (fraction blanda + grasp reducido)."""

from panda_controller.sugar_box_cartesian_descend import (
    build_sugar_box_cartesian_fallback_grasp_z_candidates,
    build_sugar_box_locked_descend_hand_goal,
    build_sugar_box_locked_descend_target_tcp,
    build_sugar_box_segment_descent_waypoints,
    format_sugar_box_descend_orientation_lock_log,
    quaternion_yaw_delta_deg,
    sugar_box_cartesian_fallback_accept,
    sugar_box_cartesian_soft_fail_eligible,
    sugar_box_fallback_grasp_tcp_z_proportional,
)


def test_soft_fail_eligible_in_band() -> None:
    assert sugar_box_cartesian_soft_fail_eligible(0.846, 0.95, soft_min_fraction=0.80)
    assert not sugar_box_cartesian_soft_fail_eligible(0.95, 0.95)
    assert not sugar_box_cartesian_soft_fail_eligible(0.75, 0.95)


def test_proportional_grasp_z_raises_for_low_fraction() -> None:
    pre_z = 0.50
    gr_z = 0.42
    z = sugar_box_fallback_grasp_tcp_z_proportional(pre_z, gr_z, 0.846, 0.95)
    assert z > gr_z
    assert z < pre_z


def test_fallback_candidates_include_proportional_and_steps() -> None:
    cands = build_sugar_box_cartesian_fallback_grasp_z_candidates(
        0.50, 0.42, 0.846, 0.95, z_raise_steps_m=(0.005, 0.010)
    )
    assert len(cands) >= 2
    assert all(c > 0.42 for c in cands)


def test_fallback_accept_requires_threshold_and_points() -> None:
    assert sugar_box_cartesian_fallback_accept(0.96, 0.95, traj_points=3)
    assert not sugar_box_cartesian_fallback_accept(0.90, 0.95, traj_points=3)
    assert not sugar_box_cartesian_fallback_accept(0.96, 0.95, traj_points=1)


def test_locked_descend_target_uses_current_xy() -> None:
    tgt = build_sugar_box_locked_descend_target_tcp((0.63, -0.175, 0.468), 0.413)
    assert tgt == (0.63, -0.175, 0.413)


def test_locked_descend_hand_goal_preserves_vertical_delta() -> None:
    lock = {
        "current_hand_position": [0.63, -0.175, 0.568],
        "current_tcp_position": [0.63, -0.175, 0.468],
    }
    hand = build_sugar_box_locked_descend_hand_goal(lock, 0.458)
    assert abs(hand[2] - 0.558) < 1e-6


def test_segment_waypoints_cover_full_descend() -> None:
    wps = build_sugar_box_segment_descent_waypoints(0.468, 0.413, step_m=0.020)
    assert wps[-1] == 0.413
    assert all(wps[i] > wps[i + 1] for i in range(len(wps) - 1))


def test_orientation_lock_log_and_yaw_delta() -> None:
    q1 = (0.0, 0.0, 0.0, 1.0)
    q2 = (0.0, 0.0, 0.3826834, 0.9238795)
    delta = quaternion_yaw_delta_deg(q1, q2)
    assert 40.0 < delta < 50.0
    log = format_sugar_box_descend_orientation_lock_log(
        {
            "old_candidate_yaw_rad": 0.0,
            "current_hand_yaw_rad": 1.57,
            "orientation_delta_deg": delta,
            "current_tcp": [0.63, -0.175, 0.468],
            "target_tcp": [0.63, -0.175, 0.413],
            "selected_grasp_tcp_z": 0.413,
            "result": "OK",
        }
    )
    assert "[DESCEND_ORIENTATION_LOCK]" in log
    assert "source=current_tf_after_joint7" in log
