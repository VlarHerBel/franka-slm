"""Parche visual runtime y offset semántico sin XY residual."""

import math
from pathlib import Path

import pytest

from panda_vision.spawn.runtime_scene_gt_geometry import (
    gazebo_model_origin_from_semantic_center,
    get_known_box_gt_spec,
)
from panda_vision.spawn.ycb_runtime_model_assets import prepare_runtime_spawn_model
from panda_vision.spawn.ycb_visual_normalization import (
    extract_sdf_collision_visual_geometry,
    get_visual_normalization_entry,
    normalize_runtime_sdf_visual_to_collision_box,
    patch_visual_pose_in_sdf,
)


def _gazebo_models_root() -> Path:
    return Path.home() / "tfg_robotics_ws" / "src" / "gazebo_ycb" / "models"


@pytest.mark.skipif(
    not (_gazebo_models_root() / "cracker_box" / "model.sdf").is_file(),
    reason="gazebo_ycb models no instalados",
)
def test_normalize_runtime_sdf_visual_cracker():
    src = _gazebo_models_root() / "cracker_box" / "model.sdf"
    text = src.read_text(encoding="utf-8")
    entry = get_visual_normalization_entry("cracker_box")
    assert entry is not None
    out, ok, old_vis, new_vis = normalize_runtime_sdf_visual_to_collision_box(
        text, "cracker_box", logger=None
    )
    assert ok
    assert old_vis == entry.original_visual_pose
    parsed = extract_sdf_collision_visual_geometry(out)
    assert parsed["original_visual_pose"] == entry.normalized_visual_pose
    assert parsed["original_visual_pose"][:3] == (0.0, 0.0, 0.0)
    assert parsed["original_visual_pose"][3:6] == (0.0, 0.0, 0.0)


def test_cracker_model_origin_offset_xy_zero():
    spec = get_known_box_gt_spec("cracker_box")
    assert spec is not None
    ox, oy, oz = spec.model_origin_to_geometry_center_offset_xyz
    assert ox == pytest.approx(0.0, abs=1e-9)
    assert oy == pytest.approx(0.0, abs=1e-9)
    assert oz == pytest.approx(0.105, abs=1e-4)


def test_gazebo_origin_yaw0_no_xy_shift():
    sem = (0.56, 0.0, 0.366)
    gz = gazebo_model_origin_from_semantic_center(sem, 0.0, "cracker_box")
    assert gz[0] == pytest.approx(sem[0], abs=1e-6)
    assert gz[1] == pytest.approx(sem[1], abs=1e-6)
    assert gz[2] == pytest.approx(sem[2] - 0.105, abs=1e-4)


def test_gazebo_origin_yaw90_only_rotates_z_offset():
    sem = (0.56, 0.0, 0.366)
    yaw = math.pi / 2.0
    gz = gazebo_model_origin_from_semantic_center(sem, yaw, "cracker_box")
    assert gz[0] == pytest.approx(sem[0], abs=1e-6)
    assert gz[1] == pytest.approx(sem[1] - 0.105, abs=1e-4)


@pytest.mark.skipif(
    not (_gazebo_models_root() / "pudding_box" / "model.sdf").is_file(),
    reason="gazebo_ycb models no instalados",
)
def test_prepare_runtime_zeroes_pudding_visual_rpy(tmp_path):
    src_dir = _gazebo_models_root() / "pudding_box"
    sdf_path, _name, runtime_dir = prepare_runtime_spawn_model(
        "pudding_box",
        src_dir,
        runtime_models_root=tmp_path / "runtime_models",
        logger=None,
    )
    parsed = extract_sdf_collision_visual_geometry(sdf_path.read_text(encoding="utf-8"))
    vis = parsed["original_visual_pose"]
    assert vis is not None
    assert vis[3:6] == (0.0, 0.0, 0.0)
    entry = get_visual_normalization_entry("pudding_box")
    assert entry is not None
    assert vis == entry.normalized_visual_pose


def test_patch_visual_pose_format():
    sdf = (
        '<visual name="visual">\n'
        "        <pose>0.015 0.015 0 0 0 0</pose>\n"
        "      </visual>"
    )
    out = patch_visual_pose_in_sdf(sdf, (0.0, 0.0, 0.0, 0.0, 0.0, 0.0))
    assert "0 0 0 0 0 0" in out
