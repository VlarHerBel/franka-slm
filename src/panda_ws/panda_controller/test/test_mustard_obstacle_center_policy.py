"""Centro de obstáculo mustard_bottle (body center, no cap center)."""

from panda_controller.mustard_obstacle_center_policy import (
    apply_mustard_obstacle_center_for_planning_scene,
    format_mustard_obstacle_center_policy_log,
    resolve_mustard_obstacle_center_base,
)


def test_obstacle_uses_body_center_not_cap_center() -> None:
    scene_obj = {
        "label": "mustard_bottle",
        "role": "obstacle",
        "grasp_center_base": [0.6623, 0.0843, 0.4300],
        "semantic_center_base": [0.6623, 0.0843, 0.4300],
        "known_object_center_base": [0.6600, 0.0600, 0.4366],
    }
    merged, log_fields = apply_mustard_obstacle_center_for_planning_scene(
        scene_obj,
        is_target=False,
    )
    assert log_fields is not None
    assert log_fields["result"] == "OK"
    assert merged["position"][:2] == [0.6600, 0.0600]
    assert merged["semantic_center_base"][:2] == [0.6600, 0.0600]


def test_target_mustard_unchanged() -> None:
    scene_obj = {
        "label": "mustard_bottle",
        "role": "target",
        "grasp_center_base": [0.6623, 0.0843, 0.4300],
    }
    merged, log_fields = apply_mustard_obstacle_center_for_planning_scene(
        scene_obj,
        is_target=True,
    )
    assert log_fields is None
    assert merged["grasp_center_base"] == [0.6623, 0.0843, 0.4300]


def test_obstacle_defaults_to_demo_body_center_when_only_cap_present() -> None:
    center, source = resolve_mustard_obstacle_center_base(
        {
            "grasp_center_base": [0.6623, 0.0843, 0.4300],
            "semantic_center_base": [0.6623, 0.0843, 0.4300],
        }
    )
    assert center is not None
    assert abs(center[0] - 0.6600) < 1e-6
    assert abs(center[1] - 0.0600) < 1e-6
    assert source == "demo_scene_02_body_center_default"


def test_obstacle_center_policy_log() -> None:
    log = format_mustard_obstacle_center_policy_log(
        {
            "center_source": "body_center",
            "obstacle_pose": (0.660, 0.060, 0.4366),
            "result": "OK",
        }
    )
    assert "[MUSTARD_OBSTACLE_CENTER_POLICY]" in log
    assert "role=obstacle" in log
    assert "center_source=body_center" in log
    assert "obstacle_pose=(0.660, 0.060, 0.437)" in log
    assert "result=OK" in log
