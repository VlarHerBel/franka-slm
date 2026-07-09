"""Tests offline del certificador plan-only."""

from __future__ import annotations

import json

from panda_controller.demo_scene_layout_validator import (
    canonical_demo_scene_02_layout,
    pick_order_by_robot_proximity,
)
from panda_controller.demo_scene_layout_validator import layout_pose_from_spawn
from panda_controller.demo_scene_plan_preflight_certifier import (
    PLAN_PREFLIGHT_STATUS,
    PickCyclePlanResult,
    PlanPreflightCertifierConfig,
    build_plan_preflight_golden_yaml,
    place_slot_xy,
    remaining_labels_for_pick_step,
    resolve_pick_order_for_layout,
    scene_obstacles_for_pick_step,
    write_plan_preflight_bundle,
)
from panda_controller.demo_scene_reachability_scan import ScanCellResult


def test_place_slot_xy_offsets() -> None:
    origin = (-0.370, 0.080)
    x0, y0 = place_slot_xy(0, origin=origin, spacing=0.07)
    x1, y1 = place_slot_xy(1, origin=origin, spacing=0.07)
    x2, y2 = place_slot_xy(2, origin=origin, spacing=0.07)
    assert abs(x0 + 0.370) < 1e-9 and abs(y0 - 0.080) < 1e-9
    assert abs(x1 + 0.370) < 1e-9 and abs(y1 - 0.150) < 1e-9
    assert abs(x2 + 0.370) < 1e-9 and abs(y2 - 0.010) < 1e-9


def test_remaining_labels_excludes_completed() -> None:
    order = ("cracker_box", "chips_can", "sugar_box", "mustard_bottle")
    completed = {"cracker_box"}
    remaining = remaining_labels_for_pick_step(order, 1, completed)
    assert remaining == ("mustard_bottle", "sugar_box")


def test_resolve_pick_order_not_fixed_label_order() -> None:
    objects = canonical_demo_scene_02_layout()
    objects["cracker_box"] = layout_pose_from_spawn("cracker_box", 0.68, 0.15, 2.9155)
    objects["chips_can"] = layout_pose_from_spawn("chips_can", 0.45, 0.0, 1.3953)
    order = resolve_pick_order_for_layout(objects)
    assert order[0] == "chips_can"


def test_scene_obstacles_for_pick_step() -> None:
    objects = canonical_demo_scene_02_layout()
    obs = scene_obstacles_for_pick_step(
        objects,
        target_label="cracker_box",
        completed_labels=set(),
    )
    labels = {str(o["label"]) for o in obs}
    assert "cracker_box" not in labels
    assert len(labels) == 3


def test_plan_preflight_golden_status() -> None:
    objects = canonical_demo_scene_02_layout()
    pose = objects["sugar_box"]
    scan = ScanCellResult(
        label="sugar_box",
        x=pose.spawn_x,
        y=pose.spawn_y,
        yaw=pose.yaw,
        pregrasp_tcp=(0.630, -0.175, 0.500),
        grasp_tcp=(0.630, -0.175, 0.420),
        commanded_tcp_yaw_rad=-1.445,
        cartesian_fraction=1.0,
        endpoint_ik_ok=True,
        result="OK",
        reason="reachable",
    )
    order = pick_order_by_robot_proximity(objects)
    golden = build_plan_preflight_golden_yaml(
        scene_id="plan_certified_layout_01",
        label="sugar_box",
        pose=pose,
        pick_result=scan,
        pick_step=3,
        pick_order=order,
        remaining_obstacles=("mustard_bottle",),
        place_slot_index=2,
        config=PlanPreflightCertifierConfig(),
    )
    assert golden["status"] == PLAN_PREFLIGHT_STATUS
    assert golden["validation"]["attach_result"] == "NOT_EXECUTED"
    assert golden["validation"]["return_home_plan_ok"] is True


def test_write_plan_preflight_bundle(tmp_path) -> None:
    objects = canonical_demo_scene_02_layout()
    order = pick_order_by_robot_proximity(objects)
    cycles = []
    for idx, label in enumerate(order):
        pose = objects[label]
        cycles.append(
            PickCyclePlanResult(
                label=label,
                pick_step=idx + 1,
                pick_ok=True,
                lift_ok=True,
                place_ok=True,
                home_ok=True,
                pick_result=ScanCellResult(
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
                reason="OK",
                remaining_obstacles=(),
            )
        )
    bundle = write_plan_preflight_bundle(
        scene_id="plan_certified_layout_01",
        output_dir=str(tmp_path),
        objects=objects,
        pick_order=order,
        cycle_results=cycles,
        config=PlanPreflightCertifierConfig(),
    )
    with open(f"{bundle}/validation_report.json", "r", encoding="utf-8") as handle:
        report = json.load(handle)
    assert report["status"] == PLAN_PREFLIGHT_STATUS
    assert len(report["cycles"]) == 4
    assert len(report["golden_files"]) == 4
