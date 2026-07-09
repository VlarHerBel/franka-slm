"""Tests para analyze_controller_output."""

from tfg_planner_slm.ros_command_executor import (
    _extract_failure_context_lines,
    analyze_controller_output,
)


def test_extract_failure_context_prefers_grasp_lines_over_home_retries() -> None:
    combined = "\n".join(
        [
            "[INFO] [GRASP_CANDIDATE] idx=0: pregrasp FAIL",
            "[INFO] [CHIPS_CAN_TOP_CLEARANCE_VIOLATION] result=BLOCKED",
            "[INFO] [GRASP_CANDIDATE] all candidates failed",
        ]
        + ["[INFO] [MOVE_HOME] attempt=%d/5 result=FAIL" % i for i in range(1, 6)]
        + ["[ERROR] [HOME_FAILED]"]
    )
    context = _extract_failure_context_lines(combined)
    assert any("GRASP_CANDIDATE" in ln for ln in context)
    assert any("TOP_CLEARANCE" in ln for ln in context)
    assert not all("MOVE_HOME" in ln for ln in context)


def test_analyze_rejects_grasp_margin_with_zero_returncode() -> None:
    stderr = (
        "[GEOMETRIC_GRASP_CHECK] Rejecting grasp for chips_can: "
        "gripper margin too small\n"
    )
    ok, reason = analyze_controller_output(
        "", stderr, 0, require_pick_place_success_markers=True
    )
    assert ok is False
    assert "failure_marker" in reason


def test_analyze_accepts_real_success_markers() -> None:
    stdout = (
        "[PLACE_CANDIDATE] idx=0 result=OK\n"
        "[PLACE] deterministic sequence completed successfully\n"
        "[MODE] execution_mode='pick_place' completado.\n"
    )
    ok, reason = analyze_controller_output(
        stdout, "", 0, require_pick_place_success_markers=True
    )
    assert ok is True
    assert reason == "pick_place_success_markers"


def test_analyze_accepts_demo_target_done() -> None:
    stdout = (
        "[DEMO_TARGET_DONE]\n"
        "label=chips_can\n"
        "[PLACE] deterministic sequence completed successfully\n"
    )
    ok, reason = analyze_controller_output(
        stdout, "", 0, require_pick_place_success_markers=True
    )
    assert ok is True
    assert reason == "pick_place_success_markers"


def test_analyze_sim_requires_success_markers_not_only_returncode() -> None:
    ok, reason = analyze_controller_output(
        "dry_run activado", "", 0, require_pick_place_success_markers=True
    )
    assert ok is False
    assert reason == "missing_pick_place_success_markers"
