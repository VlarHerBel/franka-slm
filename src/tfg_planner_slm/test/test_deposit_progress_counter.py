"""Contador x/N con depósito precargado."""

from tfg_planner_slm.execution_progress import ExecutionProgressTracker
from tfg_planner_slm.ros_pick_place_cmd import resolve_execution_progress_config


def test_deposit_2plus2_progress_shows_3_of_4_and_4_of_4() -> None:
    cfg = resolve_execution_progress_config("deposit_2plus2")
    assert cfg.total_objects == 4
    assert cfg.progress_index_offset == 2
    assert cfg.pick_order == ("cracker_box", "mustard_bottle")

    tracker = ExecutionProgressTracker(
        total_objects=cfg.total_objects,
        progress_index_offset=cfg.progress_index_offset,
        pick_order=cfg.pick_order,
    )
    tracker.on_step_start(1, "cracker_box")
    start_rows = [r for r in tracker.step_rows if r.get("phase_key") == "object_start"]
    assert "(3/4)" in start_rows[0]["label"]

    tracker.on_step_start(2, "mustard_bottle")
    start_rows = [r for r in tracker.step_rows if r.get("phase_key") == "object_start"]
    assert any("(4/4)" in r["label"] for r in start_rows)
