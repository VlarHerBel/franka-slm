"""Regresión: goal hand desde target TCP en descenso cartesiano."""

from panda_controller.descend_tcp_hand_goal import (
    descend_goal_semantic_check,
    descend_tcp_hand_delta_mismatch,
    hand_goal_from_tcp_target,
    hand_z_from_tcp_target_delta,
)


def _lock(current_tcp_z: float, current_hand_z: float) -> dict:
    return {
        "current_hand_position": [0.56, 0.12, current_hand_z],
        "current_tcp_position": [0.56, 0.12, current_tcp_z],
        "locked_hand_quat": [0, 0, 0, 1],
    }


def test_hand_z_delta_preserves_tcp_hand_offset() -> None:
    hz = hand_z_from_tcp_target_delta(0.4727, 0.4149, 0.5727)
    assert abs(hz - 0.5149) < 1e-4


def test_mismatch_detects_tcp_used_as_hand() -> None:
    ok, exp, act = descend_tcp_hand_delta_mismatch(
        0.4727, 0.4149, 0.5727, 0.4149, tolerance_m=0.01
    )
    assert not ok
    assert abs(exp + 0.0578) < 1e-4
    assert abs(act + 0.1578) < 1e-4


def test_mismatch_ok_when_hand_goal_correct() -> None:
    ok, _, _ = descend_tcp_hand_delta_mismatch(
        0.4727, 0.4149, 0.5727, 0.5149, tolerance_m=0.01
    )
    assert ok


def test_hand_goal_from_tcp_target_delta_fallback() -> None:
    hand = hand_goal_from_tcp_target(
        _lock(0.4727, 0.5727),
        (0.56, 0.12, 0.4149),
        (0, 0, 0, 1),
        use_grasp_tcp=True,
        tcp_to_moveit=None,
    )
    assert abs(hand[2] - 0.5149) < 1e-4


def test_cracker_box_descend_targets() -> None:
    """Regresión: gr (hand) no debe usarse como target_tcp."""
    lock = _lock(0.5124, 0.6124)
    target_tcp_z = 0.4370
    hand = hand_goal_from_tcp_target(
        lock,
        (0.56, 0.12, target_tcp_z),
        (0, 0, 0, 1),
        use_grasp_tcp=True,
        tcp_to_moveit=None,
    )
    assert abs(hand[2] - 0.5370) < 1e-4
    ok, reason, tcp_dz, hand_dz = descend_goal_semantic_check(
        0.5124,
        target_tcp_z,
        target_tcp_z,
        0.6124,
        hand[2],
    )
    assert ok, reason
    assert abs(tcp_dz + 0.0754) < 1e-3
    assert abs(hand_dz + 0.0754) < 1e-3
    # Bug: usar hand_z como target_tcp (doble conversión)
    wrong_hand = hand_goal_from_tcp_target(
        lock,
        (0.56, 0.12, 0.5370),
        (0, 0, 0, 1),
        use_grasp_tcp=True,
        tcp_to_moveit=None,
    )
    bad, bad_reason, _, _ = descend_goal_semantic_check(
        0.5124,
        0.5370,
        0.4370,
        0.6124,
        wrong_hand[2],
    )
    assert not bad
    assert bad_reason == "target_tcp_not_below_current_tcp"


def test_sugar_box_fallback_descend_targets() -> None:
    lock = _lock(0.4727, 0.5727)
    target_tcp_z = 0.4149
    hand = hand_goal_from_tcp_target(
        lock,
        (0.67, -0.17, target_tcp_z),
        (0, 0, 0, 1),
        use_grasp_tcp=True,
        tcp_to_moveit=None,
    )
    assert abs(hand[2] - 0.5149) < 1e-4
    ok, reason, tcp_dz, _ = descend_goal_semantic_check(
        0.4727,
        target_tcp_z,
        target_tcp_z,
        0.5727,
        hand[2],
    )
    assert ok, reason
    assert abs(tcp_dz + 0.0578) < 1e-3
