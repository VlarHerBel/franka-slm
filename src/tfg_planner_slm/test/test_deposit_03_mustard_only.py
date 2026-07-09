"""deposit_03_mustard_only: solo mustard en mesa, resto en depósito."""

from tfg_planner_slm.ros_pick_place_cmd import (
    build_clear_table_step_ros2_args,
    resolve_execution_progress_config,
    scene_uses_demo_golden_fast_execute,
)


def test_deposit_03_mustard_only_progress_and_golden() -> None:
    cfg = resolve_execution_progress_config("deposit_03_mustard_only")
    assert cfg.total_objects == 4
    assert cfg.progress_index_offset == 3
    assert cfg.pick_order == ("mustard_bottle",)
    assert scene_uses_demo_golden_fast_execute(
        "deposit_03_mustard_only", target_label="mustard_bottle"
    )
    argv = build_clear_table_step_ros2_args(
        dry_run=False,
        step_label="mustard_bottle",
        scene_id="deposit_03_mustard_only",
    )
    joined = " ".join(argv)
    assert "scene_id:=deposit_03_mustard_only" in joined
    assert "demo_golden_pick_fast_execute:=true" in joined
