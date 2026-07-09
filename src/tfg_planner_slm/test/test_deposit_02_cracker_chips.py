"""deposit_02_cracker_chips progress and golden argv."""

from tfg_planner_slm.ros_pick_place_cmd import (
    build_clear_table_step_ros2_args,
    resolve_execution_progress_config,
    scene_uses_demo_golden_fast_execute,
)


def test_deposit_02_cracker_chips_progress_and_golden() -> None:
    cfg = resolve_execution_progress_config("deposit_02_cracker_chips")
    assert cfg.total_objects == 4
    assert cfg.progress_index_offset == 2
    assert cfg.pick_order == ("sugar_box", "mustard_bottle")
    assert scene_uses_demo_golden_fast_execute(
        "deposit_02_cracker_chips", target_label="sugar_box"
    )
    argv = build_clear_table_step_ros2_args(
        dry_run=False,
        step_label="sugar_box",
        scene_id="deposit_02_cracker_chips",
    )
    assert "demo_golden_pick_fast_execute:=true" in " ".join(argv)


def test_deposit_02_mustard_uses_demo_profile_params() -> None:
    argv = build_clear_table_step_ros2_args(
        dry_run=False,
        step_label="mustard_bottle",
        scene_id="deposit_02_cracker_chips",
    )
    joined = " ".join(argv)
    assert "demo_golden_pick_fast_execute:=true" in joined
    assert "mustard_pregrasp_ik_joint_goal" not in joined
