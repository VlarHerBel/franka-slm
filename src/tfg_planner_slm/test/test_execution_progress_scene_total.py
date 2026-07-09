"""Tests: contador x/N según objetos de la escena."""

from tfg_planner_slm.execution_progress import ExecutionProgressTracker
from tfg_planner_slm.ros_pick_place_cmd import resolve_clear_table_pick_order


def test_clear_table_progress_uses_scene_total_objects() -> None:
    pick_order = resolve_clear_table_pick_order("demo_scene_02_3obj")
    tracker = ExecutionProgressTracker(
        total_objects=len(pick_order),
        pick_order=pick_order,
    )
    tracker.on_step_start(3, "sugar_box")
    start_rows = [r for r in tracker.step_rows if r.get("phase_key") == "object_start"]
    assert len(start_rows) == 1
    assert "(3/3)" in start_rows[0]["label"]

    public = tracker.to_public_dict()
    tracker.on_step_finished(3, "sugar_box", 12.0, True)
    public = tracker.to_public_dict()
    timing_labels = [t["label"] for t in public["timings"]]
    assert any("(3/3)" in lb for lb in timing_labels)


def test_two_boxes_scene_total_is_two() -> None:
    pick_order = resolve_clear_table_pick_order("two_boxes_01")
    tracker = ExecutionProgressTracker(
        total_objects=len(pick_order),
        pick_order=pick_order,
    )
    tracker.on_step_start(2, "sugar_box")
    start_rows = [r for r in tracker.step_rows if r.get("phase_key") == "object_start"]
    assert "(2/2)" in start_rows[0]["label"]
