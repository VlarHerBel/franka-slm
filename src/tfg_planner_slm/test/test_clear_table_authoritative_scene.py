"""Tests: clear_table hereda demo_authoritative_scene en escenas multiobjeto."""

from tfg_planner_slm.ros_pick_place_cmd import build_clear_table_step_ros2_args


def test_clear_table_3obj_enables_authoritative_scene() -> None:
    argv = build_clear_table_step_ros2_args(
        dry_run=False,
        reset_completed_state=False,
        step_label="sugar_box",
        scene_id="demo_scene_02_3obj",
    )
    joined = " ".join(argv)
    assert "scene_id:=demo_scene_02_3obj" in joined
    assert "demo_authoritative_scene:=true" in joined
