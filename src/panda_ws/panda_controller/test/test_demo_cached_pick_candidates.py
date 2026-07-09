"""Tests candidatos pick cacheados demo_scene_02."""

from __future__ import annotations

import math
import os
import tempfile

import yaml

from panda_controller.demo_cached_pick_candidates import (
    cached_entry_to_grid_spec,
    cached_pick_candidate_from_grasp_winner,
    get_cached_candidate_for_label,
    load_demo_cached_candidates,
    normalize_cached_pick_candidate_entry,
    save_demo_cached_candidate,
    validate_cached_joint7_expectation,
    validate_cached_scene_match,
)
from panda_controller.paired_cracker_box_candidate_grid import (
    PAIRED_GRID_MODE_FULL_DEBUG,
    PAIRED_GRID_MODE_PRIORITIZED,
    PAIRED_GRID_MODE_PRIORITIZED_OR_CACHED,
    resolve_paired_grid_search_mode,
)


def _sample_entry() -> dict:
    return {
        "object_pose": {
            "x": 0.455,
            "y": 0.115,
            "yaw": 2.9155,
            "top_z": 0.470,
        },
        "pregrasp_tcp": [0.455, 0.115, 0.575],
        "grasp_tcp": [0.455, 0.115, 0.437],
        "commanded_tcp_yaw": 2.9155,
        "raw_pregrasp_js": [0.0] * 7,
        "aligned_pregrasp_js": [0.0] * 7,
        "expected_joint7_after_alignment": 0.12,
        "desired_gap_axis_xy": [1.0, 0.0],
        "depth_from_top": 0.033,
        "transport_entry": "rear_retreat_x_negative",
        "place_slot": 0,
        "expected_obstacle_labels": ["chips_can", "mustard_bottle", "sugar_box"],
    }


def test_resolve_paired_grid_search_mode_defaults_to_prioritized_or_cached() -> None:
    assert (
        resolve_paired_grid_search_mode(
            mode_param="",
            enable_full_640_debug=False,
        )
        == PAIRED_GRID_MODE_PRIORITIZED_OR_CACHED
    )


def test_resolve_paired_grid_search_mode_full_debug() -> None:
    assert (
        resolve_paired_grid_search_mode(
            mode_param="full_debug",
            enable_full_640_debug=False,
        )
        == PAIRED_GRID_MODE_FULL_DEBUG
    )
    assert (
        resolve_paired_grid_search_mode(
            mode_param="prioritized",
            enable_full_640_debug=True,
        )
        == PAIRED_GRID_MODE_FULL_DEBUG
    )


def test_resolve_paired_grid_search_mode_prioritized_only() -> None:
    assert (
        resolve_paired_grid_search_mode(
            mode_param="prioritized",
            enable_full_640_debug=False,
        )
        == PAIRED_GRID_MODE_PRIORITIZED
    )


def test_validate_cached_scene_match_ok() -> None:
    entry = normalize_cached_pick_candidate_entry(_sample_entry(), label="cracker_box")
    assert entry is not None
    ok, reason, _ = validate_cached_scene_match(
        entry,
        scene_id="demo_scene_02",
        label="cracker_box",
        runtime_xy=(0.455, 0.115),
        runtime_yaw=2.9155,
        runtime_top_z=0.470,
        scene_obstacles=[
            {"label": "chips_can"},
            {"label": "mustard_bottle"},
            {"label": "sugar_box"},
        ],
        demo_scene_policy=None,
    )
    assert ok
    assert reason == "OK"


def test_validate_cached_scene_match_rejects_pose_drift() -> None:
    entry = normalize_cached_pick_candidate_entry(_sample_entry(), label="cracker_box")
    assert entry is not None
    ok, reason, _ = validate_cached_scene_match(
        entry,
        scene_id="demo_scene_02",
        label="cracker_box",
        runtime_xy=(0.500, 0.115),
        runtime_yaw=2.9155,
        runtime_top_z=0.470,
        scene_obstacles=[
            {"label": "chips_can"},
            {"label": "mustard_bottle"},
            {"label": "sugar_box"},
        ],
        demo_scene_policy=None,
    )
    assert not ok
    assert reason == "object_pose_xy_mismatch"


def test_validate_cached_scene_match_rejects_obstacle_set() -> None:
    entry = normalize_cached_pick_candidate_entry(_sample_entry(), label="cracker_box")
    assert entry is not None
    ok, reason, _ = validate_cached_scene_match(
        entry,
        scene_id="demo_scene_02",
        label="cracker_box",
        runtime_xy=(0.455, 0.115),
        runtime_yaw=2.9155,
        runtime_top_z=0.470,
        scene_obstacles=[{"label": "chips_can"}],
        demo_scene_policy=None,
    )
    assert not ok
    assert reason == "obstacle_set_mismatch"


def test_cached_entry_to_grid_spec() -> None:
    entry = normalize_cached_pick_candidate_entry(_sample_entry(), label="cracker_box")
    assert entry is not None
    spec = cached_entry_to_grid_spec(entry, grid_idx=-1, gripper_physical_yaw_correction_rad=0.0)
    assert spec["source"] == "demo_cached_candidate"
    assert spec["pre_plan"][2] == 0.575
    assert abs(spec["gr_plan"][2] - 0.437) < 1e-9


def test_validate_cached_joint7_expectation() -> None:
    entry = {"expected_joint7_after_alignment": 0.12}
    ok, reason = validate_cached_joint7_expectation(
        entry, joint7_after_sim=0.14, joint7_tol_rad=0.05
    )
    assert ok
    assert reason == "OK"
    ok2, reason2 = validate_cached_joint7_expectation(
        entry, joint7_after_sim=0.30, joint7_tol_rad=0.05
    )
    assert not ok2
    assert reason2 == "joint7_expectation_mismatch"


def test_save_and_load_demo_cached_candidate_yaml() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "demo_scene_02_cached_candidates.yaml")
        save_demo_cached_candidate(
            path,
            _sample_entry(),
            scene_id="demo_scene_02",
            label="cracker_box",
        )
        loaded = load_demo_cached_candidates(path)
        assert loaded is not None
        entry = get_cached_candidate_for_label(loaded, "cracker_box")
        assert entry is not None
        assert math.isclose(entry["pregrasp_tcp"][2], 0.575)
        with open(path, "r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle)
        assert raw["scene_id"] == "demo_scene_02"
        assert "cracker_box" in raw["candidates"]


def test_cached_pick_candidate_from_grasp_winner() -> None:
    winner = {
        "pre_plan": (0.455, 0.115, 0.575),
        "gr_plan": (0.455, 0.115, 0.437),
        "raw_pregrasp_js": [0.1] * 7,
        "aligned_pregrasp_js": [0.2] * 7,
        "selected_joint7_expected_after_alignment": 0.12,
        "selected_commanded_yaw": 2.9155,
        "_cart_frac": 0.98,
    }
    candidate = {"top_z_m": 0.470, "preferred_slot": 0}
    out = cached_pick_candidate_from_grasp_winner(
        scene_id="demo_scene_02",
        label="cracker_box",
        candidate=candidate,
        grasp_winner=winner,
        scene_obstacles=[{"label": "chips_can"}],
    )
    assert out["pregrasp_tcp"][2] == 0.575
    assert out["expected_obstacle_labels"] == ["chips_can"]
