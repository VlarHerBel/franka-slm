"""Tests offline del validador de layouts demo."""

from __future__ import annotations

import json

from panda_controller.demo_scene_layout_validator import (
    CertifiedLayout,
    CertifiedPickRecord,
    build_golden_yaml,
    build_scene_obstacle_dict,
    build_scene_yaml,
    canonical_demo_scene_02_layout,
    layout_pose_from_spawn,
    pick_order_by_robot_proximity,
    write_certified_layout_bundle,
)
from panda_controller.demo_scene_reachability_scan import ScanCellResult


def test_pick_order_closest_to_robot_first() -> None:
    objects = canonical_demo_scene_02_layout()
    order = pick_order_by_robot_proximity(objects)
    assert order[0] == "cracker_box"
    assert order[-1] == "mustard_bottle"


def test_pick_order_varies_with_layout_positions() -> None:
    objects = canonical_demo_scene_02_layout()
    far_cracker = layout_pose_from_spawn("cracker_box", 0.68, 0.15, 2.9155)
    near_chips = layout_pose_from_spawn("chips_can", 0.45, 0.0, 1.3953)
    objects["cracker_box"] = far_cracker
    objects["chips_can"] = near_chips
    order = pick_order_by_robot_proximity(objects)
    assert order[0] == "chips_can"


def test_mustard_spawn_compensation_from_operational() -> None:
    pose = layout_pose_from_spawn("mustard_bottle", 0.660, 0.060, 1.6392)
    assert abs(pose.operational_x - 0.660) < 0.01
    assert abs(pose.spawn_x - 0.6568) < 0.01


def test_canonical_layout_within_table() -> None:
    from panda_controller.demo_scene_layout_validator import (
        is_canonical_demo_scene_02_layout,
        layout_within_table_bounds,
    )

    objects = canonical_demo_scene_02_layout()
    assert layout_within_table_bounds(objects)
    assert is_canonical_demo_scene_02_layout(objects)


def test_scene_and_golden_yaml_builders() -> None:
    pose = layout_pose_from_spawn("sugar_box", 0.630, -0.175, -3.0159)
    scan = ScanCellResult(
        label="sugar_box",
        x=0.630,
        y=-0.175,
        yaw=-3.0159,
        top_z=0.445,
        pregrasp_tcp=(0.630, -0.175, 0.500),
        grasp_tcp=(0.630, -0.175, 0.420),
        commanded_tcp_yaw_rad=-1.445,
        cartesian_fraction=1.0,
        endpoint_ik_ok=True,
        pregrasp_ok=True,
        plan_to_pregrasp_ok=True,
        collision_ok=True,
        joint_limits_ok=True,
        result="OK",
        reason="reachable",
        seed_state_name="pick_workspace_ready",
    )
    layout = CertifiedLayout(
        scene_id="certified_layout_01",
        layout_index=1,
        pick_order=("cracker_box", "chips_can", "sugar_box", "mustard_bottle"),
        objects=canonical_demo_scene_02_layout(),
        picks={
            "sugar_box": CertifiedPickRecord(
                label="sugar_box",
                pick_step=3,
                pose=pose,
                scan_result=scan,
                remaining_obstacles=("mustard_bottle",),
            )
        },
    )
    scene = build_scene_yaml(layout, scene_id="certified_layout_01", description="test")
    assert scene["pick_order"][0] == "cracker_box"
    assert "certification" in scene
    golden = build_golden_yaml(
        layout,
        scene_id="certified_layout_01",
        label="sugar_box",
        place_slot_index=2,
    )
    assert golden["candidate"]["pregrasp_tcp"][2] == 0.500
    assert golden["validation"]["runtime_pick_place_confirm_required"] is True


def test_write_bundle_structure(tmp_path) -> None:
    objects = canonical_demo_scene_02_layout()
    picks = {}
    for idx, label in enumerate(
        ("cracker_box", "chips_can", "sugar_box", "mustard_bottle")
    ):
        pose = objects[label]
        picks[label] = CertifiedPickRecord(
            label=label,
            pick_step=idx + 1,
            pose=pose,
            scan_result=ScanCellResult(
                label=label,
                x=pose.spawn_x,
                y=pose.spawn_y,
                yaw=pose.yaw,
                pregrasp_tcp=(pose.operational_x, pose.operational_y, 0.5),
                grasp_tcp=(pose.operational_x, pose.operational_y, 0.42),
                cartesian_fraction=1.0,
                result="OK",
                reason="reachable",
            ),
            remaining_obstacles=(),
        )
    layout = CertifiedLayout(
        scene_id="certified_layout_01",
        layout_index=1,
        pick_order=("cracker_box", "chips_can", "sugar_box", "mustard_bottle"),
        objects=objects,
        picks=picks,
    )
    bundle = write_certified_layout_bundle(
        layout,
        output_dir=str(tmp_path),
        scene_id="certified_layout_01",
    )
    report_path = f"{bundle}/validation_report.json"
    with open(report_path, "r", encoding="utf-8") as handle:
        report = json.load(handle)
    assert report["pick_order"][0] == "cracker_box"
    assert len(report["golden_files"]) == 4


def test_scene_obstacle_uses_operational_center() -> None:
    pose = layout_pose_from_spawn("mustard_bottle", 0.657, 0.036, 1.6392)
    obs = build_scene_obstacle_dict(pose)
    assert abs(float(obs["x"]) - pose.operational_x) < 0.01
