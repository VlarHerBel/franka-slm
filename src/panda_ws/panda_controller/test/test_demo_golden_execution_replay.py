"""Tests golden execution replay guards (sin ROS runtime)."""

from __future__ import annotations

from panda_controller.demo_golden_execution_replay import (
    evaluate_replay_duration,
    motion_required_for_goal,
    phase_motion_fields,
    phase_tcp_goal,
)


def test_motion_required_when_far_from_goal_tcp() -> None:
    goal = [0.45, 0.12, 0.53]
    current = [0.45, 0.12, 0.56]
    required, dist, at_goal = motion_required_for_goal(goal, current)
    assert required is True
    assert at_goal is False
    assert dist > 0.02


def test_motion_not_required_when_at_goal_tcp() -> None:
    goal = [0.45, 0.12, 0.53]
    current = [0.45, 0.12, 0.5305]
    required, _dist, at_goal = motion_required_for_goal(goal, current)
    assert required is False
    assert at_goal is True


def test_zero_duration_fails_when_motion_required() -> None:
    ok, reason = evaluate_replay_duration(
        expected_s=3.0,
        real_s=0.0,
        motion_required_flag=True,
        motion_executed=False,
        at_goal=False,
    )
    assert ok is False
    assert reason == "zero_duration_with_motion_required"


def test_zero_duration_ok_when_already_at_goal() -> None:
    ok, reason = evaluate_replay_duration(
        expected_s=4.0,
        real_s=0.0,
        motion_required_flag=False,
        motion_executed=False,
        at_goal=True,
    )
    assert ok is True
    assert reason == "already_at_goal_safe_noop"


def test_descend_phase_has_cartesian_executable() -> None:
    phase = {
        "name": "cartesian_descend_to_grasp",
        "type": "cartesian_or_joint_trajectory",
        "start_tcp": [0.45, 0.12, 0.56],
        "goal_tcp": [0.45, 0.12, 0.53],
    }
    executable, _js, _pts, has_cart, _wp = phase_motion_fields(phase)
    assert executable is True
    assert has_cart is True
    assert phase_tcp_goal(phase) == [0.45, 0.12, 0.53]


def test_descend_without_executable_fields_not_executable() -> None:
    phase = {"name": "cartesian_descend_to_grasp", "type": "cartesian_or_joint_trajectory"}
    executable, has_js, has_pts, has_cart, has_wp = phase_motion_fields(phase)
    assert executable is False
    assert has_js is False
    assert has_cart is False
