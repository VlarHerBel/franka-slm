"""Tests para deduplicación de fases en ExecutionProgressTracker."""

from tfg_planner_slm.execution_progress import ExecutionProgressTracker


def test_joint7_phase_not_duplicated_on_repeated_logs() -> None:
    tracker = ExecutionProgressTracker()
    tracker.on_step_start(2, "chips_can")
    tracker.on_log_line("[GRIPPER_AXIS_CALIBRATION] ok")
    tracker.on_log_line("[GRIPPER_AXIS_COMMAND] ok")
    tracker.on_log_line("[GRIPPER_AXIS_SEARCH] step=1")
    tracker.on_log_line("[GRIPPER_AXIS_SEARCH] step=2")
    joint7_rows = [
        r for r in tracker.step_rows if "joint 7" in r["label"].lower()
    ]
    assert len(joint7_rows) == 1


def test_selected_label_parsed_from_multiline_context() -> None:
    tracker = ExecutionProgressTracker()
    tracker.on_log_line("[CLEAR_TABLE_MANUAL_STEP]")
    tracker.on_log_line("selected_label=chips_can")
    assert tracker.object_label == "chips_can"
    assert tracker.object_index == 2
