"""Tests de perfiles de ejecución demo_fast."""

from panda_controller.execution_profile import (
    DEMO_FAST_PROFILE,
    demo_fast_timing_overrides,
    resolve_execution_profile_name,
)


def test_demo_fast_mode_flag() -> None:
    assert (
        resolve_execution_profile_name(
            demo_fast_mode=True, execution_profile="default"
        )
        == DEMO_FAST_PROFILE
    )


def test_execution_profile_demo_alias() -> None:
    assert (
        resolve_execution_profile_name(
            demo_fast_mode=False, execution_profile="demo"
        )
        == DEMO_FAST_PROFILE
    )


def test_default_profile() -> None:
    assert (
        resolve_execution_profile_name(
            demo_fast_mode=False, execution_profile="default"
        )
        == "default"
    )


def test_demo_fast_overrides_values() -> None:
    ov = demo_fast_timing_overrides()
    assert ov["mustard_debug_pause_after_pregrasp_sec"] == 0.0
    assert ov["gripper_motion_time_sec"] == 1.0
    assert ov["post_place_open_settle_s"] == 0.2
    assert ov["gripper_axis_correction_motion_time_sec"] == 1.0
