"""Whitelist de grasp_center_source para mesh_local_cap_center (controller/pose gate/preflight)."""

from pathlib import Path

from panda_controller.perception_to_pregrasp_test import EXPLICIT_GRASP_CENTER_SOURCES
from panda_vision.spawn.known_object_geometry import (
    MUSTARD_MESH_LOCAL_CAP_CENTER_SOURCE,
)


def test_mesh_local_cap_center_in_explicit_grasp_sources() -> None:
    assert MUSTARD_MESH_LOCAL_CAP_CENTER_SOURCE in EXPLICIT_GRASP_CENTER_SOURCES


def test_pose_gate_runtime_gt_tall_centers_includes_mesh_local() -> None:
    src_path = (
        Path(__file__).resolve().parents[1]
        / "panda_controller"
        / "perception_to_pregrasp_test.py"
    )
    text = src_path.read_text(encoding="utf-8")
    assert MUSTARD_MESH_LOCAL_CAP_CENTER_SOURCE in text
    assert "runtime_gt_tall_centers" in text
