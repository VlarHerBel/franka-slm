"""Tests integración clear_table: dispatcher + argv ROS + analizador de salida."""

from tfg_planner_slm.command_dispatcher import dispatch_command
from tfg_planner_slm.ros_command_executor import analyze_controller_output
from tfg_planner_slm.ros_pick_place_cmd import (
    build_clear_table_step_ros2_args,
    clear_table_max_steps,
    resolve_clear_table_pick_order,
)


def _clear_table_intent() -> dict:
    return {
        "schema_version": "1.1",
        "intent": "clear_table",
        "target_label": None,
        "target_selector": {"type": "all_supported_visible_objects"},
        "destination": {
            "type": "slots_ordered",
            "slot_index": None,
            "slot_order": [0, 1, 2, 3],
        },
        "execution": {"dry_run": True, "require_confirmation": True},
        "safety": {
            "requires_clarification": False,
            "clarification_question": "",
            "reject_reason": "",
        },
    }


def test_dispatch_clear_table_execution_supported() -> None:
    action = dispatch_command(_clear_table_intent())
    assert action.intent == "clear_table"
    assert action.execution_supported is True
    assert action.slot_order == [0, 1, 2, 3]
    assert action.ros_command_preview is not None
    assert "clear_table" in action.ros_command_preview


def test_build_clear_table_step_argv_two_boxes() -> None:
    argv = build_clear_table_step_ros2_args(
        dry_run=False,
        reset_completed_state=True,
        scene_id="two_boxes_01",
    )
    joined = " ".join(argv)
    assert "perception_to_pregrasp_test" in joined
    assert "execution_mode:=clear_table" in joined
    assert "scene_id:=two_boxes_01" in joined
    assert "demo_authoritative_scene:=false" in joined
    assert "clear_table_manual_step:=true" in joined
    assert "demo_reset_completed_state_on_start:=true" in joined
    assert resolve_clear_table_pick_order("two_boxes_01") == (
        "cracker_box",
        "sugar_box",
    )
    assert clear_table_max_steps("two_boxes_01") == 2


def test_build_clear_table_step_argv_chips_mustard() -> None:
    argv = build_clear_table_step_ros2_args(
        dry_run=False,
        reset_completed_state=True,
        scene_id="chips_mustard_01",
    )
    joined = " ".join(argv)
    assert "scene_id:=chips_mustard_01" in joined
    assert resolve_clear_table_pick_order("chips_mustard_01") == (
        "chips_can",
        "mustard_bottle",
    )
    assert clear_table_max_steps("chips_mustard_01") == 2


def test_resolve_clear_table_pick_order_demo_scene_3obj() -> None:
    assert resolve_clear_table_pick_order("demo_scene_02_3obj") == (
        "cracker_box",
        "chips_can",
        "sugar_box",
    )
    assert clear_table_max_steps("demo_scene_01_3obj") == 3


def test_build_clear_table_step_argv_demo_profile() -> None:
    argv = build_clear_table_step_ros2_args(
        dry_run=False,
        reset_completed_state=True,
        demo_profile=True,
    )
    joined = " ".join(argv)
    assert "scene_id:=demo_scene_02" in joined
    assert "demo_golden_pick_fast_execute:=true" in joined


def test_analyze_clear_table_step_done_marker() -> None:
    stdout = "[CLEAR_TABLE_TARGET_DONE]\nlabel=cracker_box\n"
    ok, reason = analyze_controller_output(
        stdout, "", 0, require_pick_place_success_markers=True
    )
    assert ok is True
    assert reason == "clear_table_step_done"


def test_analyze_clear_table_complete_marker() -> None:
    stdout = "[CLEAR_TABLE_COMPLETE]\nresult=OK\ncompleted_labels=['cracker_box']"
    ok, reason = analyze_controller_output(
        stdout,
        "",
        0,
        require_pick_place_success_markers=False,
        require_clear_table_complete=True,
    )
    assert ok is True
    assert reason == "clear_table_complete"
