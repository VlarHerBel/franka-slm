"""Perfiles de temporización para ejecución (demo vs validación)."""

from __future__ import annotations

from typing import Any, Dict

DEMO_FAST_PROFILE = "demo_fast"
DEFAULT_PROFILE = "default"

# Solo retardos / logs repetitivos; no toca palm_bridge, contacto ni place logic.
DEMO_FAST_TIMING_OVERRIDES: Dict[str, Any] = {
    "mustard_debug_pause_after_pregrasp_sec": 0.0,
    "mustard_debug_pause_before_descend_sec": 0.0,
    "gripper_motion_time_sec": 1.0,
    "post_place_open_settle_s": 0.2,
    "post_detach_settle_s": 0.2,
    "post_place_gripper_motion_time_sec": 1.0,
    "gripper_axis_correction_motion_time_sec": 1.0,
    "suppress_repeated_gripper_axis_calibration_warn": True,
}


def resolve_execution_profile_name(
    *,
    demo_fast_mode: bool,
    execution_profile: str,
) -> str:
    """`execution_profile:=demo` o `demo_fast_mode:=true` activan demo_fast."""
    ep = str(execution_profile or DEFAULT_PROFILE).strip().lower()
    if bool(demo_fast_mode) or ep in ("demo", "demo_fast", DEMO_FAST_PROFILE):
        return DEMO_FAST_PROFILE
    return DEFAULT_PROFILE


def demo_fast_timing_overrides() -> Dict[str, Any]:
    return dict(DEMO_FAST_TIMING_OVERRIDES)
