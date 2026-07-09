"""Tests para argv pick_place / clear_table."""

from tfg_planner_slm.ros_pick_place_cmd import (
    build_clear_table_step_ros2_args,
    build_pick_place_ros2_args,
    demo_pick_params_for_target,
    scene_uses_demo_golden_fast_execute,
)


def test_pick_place_chips_can_includes_demo_profile() -> None:
    params = dict(demo_pick_params_for_target("chips_can"))
    assert params["scene_id"] == "demo_scene_02"
    assert params["chips_can_use_legacy_successful_pick_policy"] == "true"
    assert params["demo_golden_pick_fast_execute"] == "true"


def test_build_pick_place_simple_profile_has_no_demo_scene_id() -> None:
    argv = build_pick_place_ros2_args(
        dry_run=False,
        target_label="cracker_box",
        slot_index=0,
        slot_user_specified=True,
    )
    joined = " ".join(argv)
    assert "target_label:=cracker_box" in joined
    assert "scene_id:=demo_scene_02" not in joined
    assert "scene_id:=two_boxes_01" in joined
    assert "demo_authoritative_scene:=true" not in joined
    assert "demo_authoritative_scene:=false" in joined
    assert "clear_table_manual_step:=false" in joined
    assert "paired_grid_search_mode:=prioritized_or_cached" in joined
    assert "require_full_pick_route_preplanned_before_prelude:=false" in joined


def test_build_pick_place_argv_includes_demo_params_when_requested() -> None:
    argv = build_pick_place_ros2_args(
        dry_run=False,
        target_label="chips_can",
        slot_index=1,
        slot_user_specified=True,
        demo_profile=True,
    )
    joined = " ".join(argv)
    assert "target_label:=chips_can" in joined
    assert "scene_id:=demo_scene_02" in joined
    assert "chips_can_use_legacy_successful_pick_policy:=true" in joined
    assert "plan_before_prelude_skip_workspace_prelude:=false" in joined


def test_unknown_target_has_no_demo_params() -> None:
    assert demo_pick_params_for_target("tomato_soup_can") == ()


def test_build_pick_place_mustard_includes_grasp_params() -> None:
    argv = build_pick_place_ros2_args(
        dry_run=False,
        target_label="mustard_bottle",
        slot_index=0,
        slot_user_specified=True,
        scene_id="chips_mustard_01",
    )
    joined = " ".join(argv)
    assert "deterministic_transport_time_scale:=2.2" in joined
    assert "mustard_close_joint_m:=0.018" in joined
    assert "mustard_min_required_depth_from_top_m:=0.046" in joined
    assert "post_grasp_pause_sec:=2.0" in joined
    assert "mustard_pregrasp_ik_joint_goal:=false" in joined
    assert "mustard_operational_skip_squeeze_after_close:=false" in joined
    assert "demo_golden_pick_fast_execute:=false" in joined
    assert "mustard_extra_micro_descend_after_cartesian_m:=0.022" in joined
    assert "scene_id:=chips_mustard_01" in joined


def test_build_pick_place_chips_mustard_02_searches_until_golden() -> None:
    argv = build_pick_place_ros2_args(
        dry_run=False,
        target_label="mustard_bottle",
        slot_index=3,
        slot_user_specified=True,
        scene_id="chips_mustard_02",
    )
    joined = " ".join(argv)
    assert "scene_id:=chips_mustard_02" in joined
    assert "demo_golden_pick_fast_execute:=false" in joined
    assert "demo_authoritative_scene:=true" not in joined
    assert "mustard_extra_micro_descend_after_cartesian_m:=0.022" in joined


def test_build_pick_place_chips_mustard_02_chips_no_golden_by_default() -> None:
    argv = build_pick_place_ros2_args(
        dry_run=False,
        target_label="chips_can",
        slot_index=1,
        slot_user_specified=True,
        scene_id="chips_mustard_02",
    )
    joined = " ".join(argv)
    assert "demo_golden_pick_fast_execute:=false" in joined
    assert "paired_grid_search_mode:=prioritized_or_cached" in joined


def test_scene_uses_golden_only_when_chips_mustard_02_yaml_exists(
    tmp_path, monkeypatch
) -> None:
    from panda_controller import demo_golden_pick_candidate as mod
    from panda_controller.demo_golden_pick_candidate import (
        build_chips_can_pick_golden_from_success,
        save_golden_pick_candidate,
    )

    monkeypatch.setattr(mod, "default_demo_config_dir", lambda: str(tmp_path))
    assert not scene_uses_demo_golden_fast_execute(
        "chips_mustard_02", target_label="chips_can"
    )
    golden = build_chips_can_pick_golden_from_success(
        {
            "label": "chips_can",
            "grasp_center_base": [0.500, -0.080, 0.510],
            "_pregrasp_tcp_planning": [0.500, -0.080, 0.545],
            "_grasp_tcp_planning": [0.500, -0.080, 0.475],
        },
        scene_id="chips_mustard_02",
    )
    cache = tmp_path / "demo_candidate_cache"
    cache.mkdir(parents=True)
    save_path = cache / "chips_mustard_02_chips_can_golden.yaml"
    assert save_golden_pick_candidate(golden, str(save_path))
    assert scene_uses_demo_golden_fast_execute(
        "chips_mustard_02", target_label="chips_can"
    )


def test_build_pick_place_demo_scene_3obj_authoritative() -> None:
    argv = build_pick_place_ros2_args(
        dry_run=False,
        target_label="chips_can",
        slot_index=0,
        slot_user_specified=True,
        scene_id="demo_scene_02_3obj",
    )
    joined = " ".join(argv)
    assert "scene_id:=demo_scene_02_3obj" in joined
    assert "demo_authoritative_scene:=true" in joined
    assert "execution_profile:=demo" in joined
    assert "chips_can_use_legacy_successful_pick_policy:=true" in joined


def test_sugar_box_golden_on_demo_scene_02_and_deposit_debug() -> None:
    assert scene_uses_demo_golden_fast_execute(
        "demo_scene_02", target_label="sugar_box"
    )
    assert scene_uses_demo_golden_fast_execute(
        "deposit_02_cracker_chips", target_label="sugar_box"
    )
    assert not scene_uses_demo_golden_fast_execute(
        "demo_scene_02_3obj", target_label="sugar_box"
    )
    argv = build_clear_table_step_ros2_args(
        dry_run=False,
        step_label="sugar_box",
        scene_id="demo_scene_02",
    )
    assert "demo_golden_pick_fast_execute:=true" in " ".join(argv)


def test_nogolden_scene_keeps_golden_for_cracker_disables_for_sugar() -> None:
    assert scene_uses_demo_golden_fast_execute(
        "demo_scene_02_3obj_nogolden", target_label="cracker_box"
    )
    assert scene_uses_demo_golden_fast_execute(
        "demo_scene_02_3obj_nogolden", target_label="chips_can"
    )
    assert not scene_uses_demo_golden_fast_execute(
        "demo_scene_02_3obj_nogolden", target_label="sugar_box"
    )
    argv = build_clear_table_step_ros2_args(
        dry_run=False,
        step_label="cracker_box",
        scene_id="demo_scene_02_3obj_nogolden",
    )
    assert "demo_golden_pick_fast_execute:=true" in " ".join(argv)
    argv_sugar = build_clear_table_step_ros2_args(
        dry_run=False,
        step_label="sugar_box",
        scene_id="demo_scene_02_3obj_nogolden",
    )
    assert "demo_golden_pick_fast_execute:=false" in " ".join(argv_sugar)
