"""Regresión: init temprano de place_slot_index tras aplicar demo profile."""

from __future__ import annotations

from unittest.mock import patch

import pytest
import rclpy

from panda_controller.demo_profile_loader import (
    resolve_demo_profile,
    resolve_golden_path_from_profile,
)
from panda_controller.perception_to_pregrasp_test import PerceptionToPregraspTest


@pytest.fixture(scope="module")
def rclpy_context() -> None:
    if not rclpy.ok():
        rclpy.init(
            args=[
                "--ros-args",
                "-p",
                "target_label:=chips_can",
                "-p",
                "scene_id:=demo_scene_02",
                "-p",
                "place_slot_index:=1",
                "-p",
                "dry_run:=true",
                "-p",
                "use_sim_time:=true",
            ]
        )
    yield
    if rclpy.ok():
        rclpy.shutdown()


def test_chips_can_demo_profile_init_place_slot_index(rclpy_context: None) -> None:
    profile = resolve_demo_profile("demo_scene_02", "chips_can")
    assert profile is not None
    assert profile["profile_id"] == "demo_scene_02_chips_can"
    assert profile["parameters"]["place_slot_index"] == 1
    golden_path = resolve_golden_path_from_profile(profile)
    assert golden_path.endswith("demo_scene_02_chips_can_golden.yaml")

    with patch.object(PerceptionToPregraspTest, "_setup_moveit", return_value=None):
        node = PerceptionToPregraspTest()

    try:
        assert node._place_slot_index == 1
        assert node._demo_profile is not None
        assert node._demo_profile["profile_id"] == "demo_scene_02_chips_can"
        assert node._candidate_benchmark_slot_index == 1
        assert str(node._scene_id).strip().lower() == "demo_scene_02"
        assert str(node._target_label).strip().lower() == "chips_can"
        assert node._demo_golden_candidate_path.endswith(
            "demo_scene_02_chips_can_golden.yaml"
        )
    finally:
        node.destroy_node()
