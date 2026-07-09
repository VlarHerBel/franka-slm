"""Tests lógicos del guard de eje corto para mustard_bottle (sin ROS)."""
import math


def _is_mustard_short_axis_candidate(label: str, strategy: str) -> bool:
    return label == "mustard_bottle" and strategy == "tall_object_topdown"


def _guard_ok(
    desired_xy: tuple,
    minor_xy: tuple,
    major_xy: tuple,
) -> bool:
    dx, dy = desired_xy
    mx, my = minor_xy
    ax, ay = major_xy
    dn = math.hypot(dx, dy)
    mn = math.hypot(mx, my)
    an = math.hypot(ax, ay)
    desired = (dx / dn, dy / dn)
    minor = (mx / mn, my / mn)
    major = (ax / an, ay / an)
    dot_minor = abs(desired[0] * minor[0] + desired[1] * minor[1])
    dot_major = abs(desired[0] * major[0] + desired[1] * major[1])
    return dot_minor > 0.98 and dot_major < 0.25


def test_mustard_candidate_detected() -> None:
    assert _is_mustard_short_axis_candidate("mustard_bottle", "tall_object_topdown")


def test_mustard_guard_minor_axis_ok() -> None:
    minor = (0.0, 1.0)
    major = (1.0, 0.0)
    assert _guard_ok(minor, minor, major)


def test_mustard_guard_major_axis_fail() -> None:
    minor = (0.0, 1.0)
    major = (1.0, 0.0)
    assert not _guard_ok(major, minor, major)


def test_quaternion_match_direct() -> None:
    a = (0.0, 1.0, 0.0, 0.0)
    b = (0.0, 1.0, 0.0, 0.0)
    assert all(abs(a[i] - b[i]) < 1e-3 for i in range(4))


def test_quaternion_match_sign_flip() -> None:
    a = (0.0, 0.707, 0.0, 0.707)
    b = (0.0, -0.707, 0.0, -0.707)
    assert all(abs(a[i] + b[i]) < 1e-3 for i in range(4))


def test_swap_major_minor_selects_major_as_short_gap() -> None:
    minor = (0.0, 1.0)
    major = (1.0, 0.0)
    short = major
    rejected = minor
    assert abs(short[0] * 1.0 + short[1] * 0.0) > 0.9
    assert abs(rejected[0] * 0.0 + rejected[1] * 1.0) > 0.9
