"""Regresión: fallback guarded IK descend sugar_box demo_scene_02."""

from panda_controller.sugar_box_guarded_ik_descend import (
    build_sugar_box_guarded_ik_final_z_candidates,
    build_sugar_box_guarded_ik_hand_goal_from_tcp_delta,
    build_sugar_box_guarded_ik_z_waypoints,
    evaluate_sugar_box_guarded_ik_fk_step,
    format_sugar_box_guarded_ik_descend_step_log,
    format_sugar_box_guarded_ik_descend_step_target_log,
    format_sugar_box_guarded_ik_fail_diag_log,
    sugar_box_guarded_ik_descend_eligible,
    sugar_box_guarded_ik_tcp_z_used_as_hand_z,
)


def test_guarded_ik_eligible_only_demo_scene_02_sugar() -> None:
    assert sugar_box_guarded_ik_descend_eligible(
        label="sugar_box",
        scene_id="demo_scene_02",
        multiobject_safe_route=True,
    )
    assert not sugar_box_guarded_ik_descend_eligible(
        label="sugar_box",
        scene_id="demo_scene_01",
        multiobject_safe_route=True,
    )
    assert not sugar_box_guarded_ik_descend_eligible(
        label="cracker_box",
        scene_id="demo_scene_02",
        multiobject_safe_route=True,
    )


def test_z_waypoints_10mm_steps() -> None:
    wps = build_sugar_box_guarded_ik_z_waypoints(0.4685, 0.4110, step_m=0.010)
    assert wps[0] == 0.4585
    assert wps[-1] == 0.4110


def test_final_z_candidates_include_deeper() -> None:
    cands = build_sugar_box_guarded_ik_final_z_candidates(0.4110)
    assert cands[0] == 0.4110
    assert any(z < 0.4110 for z in cands)


def test_fk_step_tolerance() -> None:
    ok, xy, z, ang = evaluate_sugar_box_guarded_ik_fk_step(
        fk_tcp=(0.6300, -0.1749, 0.4110),
        target_tcp_z=0.4110,
        reference_tcp_xy=(0.6300, -0.1749),
        orientation_error_deg=2.5,
    )
    assert ok
    assert xy < 0.002
    assert z < 0.003
    assert ang <= 5.0


def test_step_log_format() -> None:
    log = format_sugar_box_guarded_ik_descend_step_log(
        {
            "step": "1/6",
            "target_tcp_z": 0.4585,
            "ik_ok": True,
            "fk_tcp": (0.63, -0.175, 0.4585),
            "xy_error": 0.0001,
            "z_error": 0.0002,
            "orientation_error_deg": 2.1,
            "result": "OK",
        }
    )
    assert "[SUGAR_BOX_GUARDED_IK_DESCEND_STEP]" in log
    assert "ik_ok=true" in log


def test_hand_goal_from_tcp_delta_applies_hand_tcp_offset() -> None:
    hand = build_sugar_box_guarded_ik_hand_goal_from_tcp_delta(
        current_hand=(0.63, -0.175, 0.568),
        current_tcp=(0.63, -0.175, 0.468),
        target_tcp=(0.63, -0.175, 0.458),
    )
    assert abs(hand[2] - 0.558) < 1e-6


def test_frame_bug_detects_tcp_z_as_hand_z() -> None:
    assert sugar_box_guarded_ik_tcp_z_used_as_hand_z(
        current_hand_z=0.568,
        current_tcp_z=0.468,
        target_hand_z=0.458,
        target_tcp_z=0.458,
    )
    assert not sugar_box_guarded_ik_tcp_z_used_as_hand_z(
        current_hand_z=0.568,
        current_tcp_z=0.468,
        target_hand_z=0.558,
        target_tcp_z=0.458,
    )


def test_step_target_log_format() -> None:
    log = format_sugar_box_guarded_ik_descend_step_target_log(
        {
            "moveit_target_link": "panda_hand",
            "current_tcp_z": 0.468,
            "current_hand_z": 0.568,
            "target_tcp_z": 0.458,
            "target_hand_z": 0.558,
            "hand_minus_tcp_z": 0.100,
            "quat_source": "current_tf_after_joint7",
        }
    )
    assert "[SUGAR_BOX_GUARDED_IK_DESCEND_STEP_TARGET]" in log
    assert "target_hand_z=0.5580" in log


def test_guarded_ik_fail_diag_log_format() -> None:
    log = format_sugar_box_guarded_ik_fail_diag_log(
        {
            "step": "1",
            "target_tcp_z": 0.458,
            "target_hand_z": 0.558,
            "seed_source": "corrected_pregrasp_js",
            "seed_joint7": 0.12,
            "current_joint7": 0.12,
            "ik_link_name": "panda_hand",
            "orientation_tolerance_deg": 15.0,
            "position_tolerance_m": 0.001,
            "target_collision_present": False,
            "obstacles": ["cracker_box"],
            "joint_limits_near": ["j7=2.850"],
            "joint_dist_pregrasp_to_step": 0.04,
        }
    )
    assert "[SUGAR_BOX_GUARDED_IK_FAIL_DIAG]" in log
    assert "seed_source=corrected_pregrasp_js" in log
    assert "result=NO_IK_SOLUTION" in log
