"""Tests política collision_off descenso final cracker_box demo_scene_02."""

from panda_controller.demo_cracker_collision_off_final_descend import (
    DEMO_COLLISION_OFF_FINAL_DESCEND_SOURCE,
    compute_final_descend_safety_metrics,
    demo_cracker_collision_off_policy_enabled,
    evaluate_demo_cracker_collision_off_final_descend,
    evaluate_staged_collision_off_final_descend_allow,
    select_collision_on_until_z,
)
from panda_controller.paired_pregrasp_descend_validation import (
    PAIRED_ACCEPTED_PREVALIDATION_SOURCES,
    build_paired_candidate_result,
    cartesian_descend_prevalidation_acceptable,
)


def _obstacle(label: str, x: float, y: float) -> dict:
    return {
        "label": label,
        "position": [x, y, 0.45],
        "shape": "box",
        "size": [0.08, 0.08, 0.10],
    }


def test_policy_enabled_auto_only_demo_scene_02_cracker() -> None:
    assert demo_cracker_collision_off_policy_enabled(
        param_value="auto",
        demo_authoritative_scene=True,
        scene_id="demo_scene_02",
        label="cracker_box",
    )
    assert not demo_cracker_collision_off_policy_enabled(
        param_value="auto",
        demo_authoritative_scene=True,
        scene_id="demo_scene_02",
        label="chips_can",
    )
    assert not demo_cracker_collision_off_policy_enabled(
        param_value="false",
        demo_authoritative_scene=True,
        scene_id="demo_scene_02",
        label="cracker_box",
    )


def test_collision_off_allow_when_on_half_off_full() -> None:
    pre = (0.455, 0.115, 0.575)
    gr = (0.455, 0.115, 0.437)
    obstacles = [
        _obstacle("chips_can", 0.52, -0.04),
        _obstacle("sugar_box", 0.63, -0.13),
    ]
    allow, reason, metrics = evaluate_demo_cracker_collision_off_final_descend(
        demo_authoritative_scene=True,
        scene_id="demo_scene_02",
        label="cracker_box",
        pre_plan=pre,
        gr_plan=gr,
        scene_obstacles=obstacles,
        table_z_m=0.40,
        joint_values_7=[0.0, -0.5, 0.0, -2.0, 0.0, 1.5, 0.0],
        target_removed_ok=True,
        collision_on_fraction=0.5,
        collision_off_fraction=1.0,
        start_state_honored=True,
        endpoint_ik_ok=True,
        traj_pts=12,
    )
    assert allow
    assert reason == "collision_on_fraction_low_but_collision_off_ok"
    assert metrics["target_removed"] is True


def test_collision_off_allow_with_endpoint_ik_diagnostic_false() -> None:
    """endpoint_ik_ok=false no debe bloquear si collision_off fraction=1.0."""
    pre = (0.455, 0.115, 0.575)
    gr = (0.455, 0.115, 0.437)
    obstacles = [
        _obstacle("chips_can", 0.52, -0.04),
        _obstacle("sugar_box", 0.63, -0.13),
    ]
    allow, reason, metrics = evaluate_demo_cracker_collision_off_final_descend(
        demo_authoritative_scene=True,
        scene_id="demo_scene_02",
        label="cracker_box",
        pre_plan=pre,
        gr_plan=gr,
        scene_obstacles=obstacles,
        table_z_m=0.40,
        joint_values_7=[0.0, -0.5, 0.0, -2.0, 0.0, 1.5, 0.0],
        target_removed_ok=True,
        collision_on_fraction=0.57407,
        collision_off_fraction=1.0,
        start_state_honored=True,
        endpoint_ik_ok=False,
        traj_pts=12,
        collision_on_fraction_threshold=0.95,
    )
    assert allow
    assert reason == "collision_on_fraction_low_but_collision_off_ok"
    assert metrics["endpoint_ik_diagnostic_ok"] is False
    assert cartesian_descend_prevalidation_acceptable(
        cartesian_ok=True,
        cartesian_fraction=1.0,
        fraction_threshold=0.95,
        prevalidation_source=DEMO_COLLISION_OFF_FINAL_DESCEND_SOURCE,
        paired_validation_required=True,
    )
    result = build_paired_candidate_result(
        label="cracker_box",
        candidate_idx=26,
        yaw_variant=0.2,
        ik_pregrasp_ok=True,
        plan_to_pregrasp_ok=True,
        candidate_pregrasp_js=[0.0] * 7,
        cartesian_descend_fraction=1.0,
        cartesian_descend_ok=True,
        lift_ok=True,
        post_lift_exit_ok=True,
        direct_action_to_hub_ok=True,
        prevalidation_source=DEMO_COLLISION_OFF_FINAL_DESCEND_SOURCE,
    )
    assert result["result"] == "ACCEPT"
    assert DEMO_COLLISION_OFF_FINAL_DESCEND_SOURCE in PAIRED_ACCEPTED_PREVALIDATION_SOURCES


def test_collision_off_reject_when_target_still_present() -> None:
    allow, reason, _ = evaluate_demo_cracker_collision_off_final_descend(
        demo_authoritative_scene=True,
        scene_id="demo_scene_02",
        label="cracker_box",
        pre_plan=(0.455, 0.115, 0.575),
        gr_plan=(0.455, 0.115, 0.437),
        scene_obstacles=[],
        table_z_m=0.40,
        joint_values_7=[0.0] * 7,
        target_removed_ok=False,
        collision_on_fraction=0.5,
        collision_off_fraction=1.0,
        start_state_honored=True,
        endpoint_ik_ok=False,
        traj_pts=10,
    )
    assert not allow
    assert reason == "target_collision_still_present"


def test_obstacle_clearance_ok_not_blocked_by_removed_target() -> None:
    """Con target retirado, obstacle_clearance_ok no debe depender del target."""
    pre = (0.455, 0.115, 0.575)
    gr = (0.455, 0.115, 0.437)
    obstacles = [
        _obstacle("chips_can", 0.52, -0.04),
        _obstacle("sugar_box", 0.63, -0.13),
    ]
    allow, reason, metrics = evaluate_demo_cracker_collision_off_final_descend(
        demo_authoritative_scene=True,
        scene_id="demo_scene_02",
        label="cracker_box",
        pre_plan=pre,
        gr_plan=gr,
        scene_obstacles=obstacles,
        table_z_m=0.40,
        joint_values_7=[0.0, -0.5, 0.0, -2.0, 0.0, 1.5, 0.0],
        target_removed_ok=True,
        collision_on_fraction=0.55357,
        collision_off_fraction=1.0,
        start_state_honored=True,
        endpoint_ik_ok=False,
        traj_pts=12,
        min_obstacle_clearance_m=0.025,
    )
    assert allow
    assert reason == "collision_on_fraction_low_but_collision_off_ok"
    assert metrics["target_removed_ok"] is True
    assert metrics["obstacle_clearance_ok"] is True
    assert metrics["min_obstacle_distance_m"] > 0.025


def test_select_collision_on_until_z_picks_lowest_passing() -> None:
    probes = {0.4975: 1.0, 0.485: 1.0, 0.470: 0.5, 0.440: 1.0}

    def _probe(z: float) -> float:
        return probes.get(round(z, 4), 0.0)

    z = select_collision_on_until_z(
        pregrasp_tcp_z=0.5920,
        grasp_tcp_z=0.4320,
        max_collision_off_descend_m=0.14,
        z_candidates=(0.4975, 0.485, 0.470, 0.455, 0.440),
        probe_collision_on_fraction=_probe,
        fraction_min=0.98,
    )
    assert z == 0.440


def test_staged_allow_when_total_descend_exceeds_limit() -> None:
    pre = (0.455, 0.115, 0.5920)
    gr = (0.455, 0.115, 0.4320)
    obstacles = [
        _obstacle("chips_can", 0.52, -0.04),
        _obstacle("sugar_box", 0.63, -0.13),
    ]
    allow, reason, metrics = evaluate_staged_collision_off_final_descend_allow(
        demo_authoritative_scene=True,
        scene_id="demo_scene_02",
        label="cracker_box",
        pre_plan=pre,
        gr_plan=gr,
        collision_on_until_z=0.4975,
        scene_obstacles=obstacles,
        table_z_m=0.40,
        joint_values_7=[0.0, -0.5, 0.0, -2.0, 0.0, 1.5, 0.0],
        target_removed_ok=True,
        collision_on_stage_fraction=1.0,
        collision_off_fraction=1.0,
        collision_on_fraction=0.58462,
        start_state_honored=True,
        endpoint_ik_ok=False,
        traj_pts=12,
        max_collision_off_descend_m=0.14,
    )
    assert allow
    assert reason == "staged_collision_on_then_off_ok"
    assert metrics["max_descend_ok"] is True
    assert abs(float(metrics["collision_off_descend_m"]) - 0.0655) < 1e-3
    assert float(metrics["total_descend_m"]) > 0.14


def test_safety_metrics_use_collision_off_start_for_max_descend() -> None:
    pre = (0.455, 0.115, 0.5920)
    gr = (0.455, 0.115, 0.4320)
    off_start = (0.455, 0.115, 0.4975)
    metrics = compute_final_descend_safety_metrics(
        pre_plan=pre,
        gr_plan=gr,
        scene_obstacles=[],
        table_z_m=0.40,
        joint_values_7=[0.0] * 7,
        target_removed_ok=True,
        max_collision_off_descend_m=0.14,
        collision_off_start_plan=off_start,
    )
    assert abs(float(metrics["max_descend_m"]) - 0.0655) < 1e-3
    assert metrics["max_descend_ok"] is True
    assert abs(float(metrics["total_descend_m"]) - 0.160) < 1e-3
