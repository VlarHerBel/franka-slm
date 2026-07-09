"""Tests offline para resolución top_z geométrico mustard_bottle."""

from __future__ import annotations

import math

from panda_controller.mustard_top_z_policy import (
    DEMO_SCENE_02_MUSTARD_FALLBACK_TOP_Z_M,
    MUSTARD_EXPECTED_MIN_TOP_Z_M,
    MUSTARD_KNOWN_PHYSICAL_HEIGHT_M,
    MUSTARD_SCANNER_CONTRACT_GRASP_TCP_Z_M,
    MUSTARD_SCANNER_CONTRACT_PREGRASP_TCP_Z_M,
    MUSTARD_SCANNER_CONTRACT_TOP_Z_M,
    format_mustard_top_z_source_debug_log,
    resolve_mustard_geometry_top_z_m,
    verify_mustard_scanner_aligned_z_contract,
)
from panda_controller.palm_bridge_policy import resolve_effective_top_z_for_palm_bridge


def test_resolve_mustard_geometry_top_z_ignores_effective_height() -> None:
    """Runtime sin corners: usa altura SDF física, no effective_height bajo."""
    candidate = {
        "label": "mustard_bottle",
        "grasp_strategy": "tall_object_topdown",
        "position": [0.6593, 0.0603, 0.4366],
        "grasp_center_base": [0.6593, 0.0603, 0.4366],
        "top_z_m": 0.4366,
        "effective_height_m": 0.1761,
        "height_m": 0.1761,
        "db_height_m": 0.1761,
    }
    top_z, src, meta = resolve_mustard_geometry_top_z_m(
        candidate,
        table_z_m=0.27,
        fallback_z=0.4366,
        scene_id="demo_scene_02",
    )
    assert math.isclose(top_z, 0.4609, abs_tol=0.002)
    assert src in (
        "known_sdf_physical_height",
        "demo_scene_02_fallback",
        "demo_scene_02_reject_low_top_z",
    )
    assert meta["grasp_center_z"] == 0.4366
    assert meta["known_sdf_top_z"] == 0.27 + MUSTARD_KNOWN_PHYSICAL_HEIGHT_M


def test_resolve_mustard_geometry_top_z_from_runtime_corners() -> None:
    candidate = {
        "label": "mustard_bottle",
        "grasp_strategy": "tall_object_topdown",
        "top_face_corners_base": [
            [0.63, 0.04, 0.4610],
            [0.69, 0.04, 0.4610],
            [0.69, 0.08, 0.4610],
            [0.63, 0.08, 0.4610],
        ],
    }
    top_z, src, _meta = resolve_mustard_geometry_top_z_m(
        candidate, table_z_m=0.27, fallback_z=0.4366, scene_id="demo_scene_02"
    )
    assert math.isclose(top_z, 0.4610, abs_tol=0.001)
    assert src.startswith("runtime_top_corners_")


def test_resolve_mustard_demo_scene_02_rejects_low_collision_top() -> None:
    candidate = {
        "label": "mustard_bottle",
        "grasp_strategy": "tall_object_topdown",
        "collision_box_pose": {"z": 0.3535},
        "collision_dims": [0.058, 0.095, 0.1761],
    }
    top_z, src, _meta = resolve_mustard_geometry_top_z_m(
        candidate, table_z_m=0.27, fallback_z=0.4366, scene_id="demo_scene_02"
    )
    assert top_z >= MUSTARD_EXPECTED_MIN_TOP_Z_M
    assert math.isclose(top_z, DEMO_SCENE_02_MUSTARD_FALLBACK_TOP_Z_M, abs_tol=0.002)
    assert src == "known_sdf_physical_height"


def test_palm_bridge_uses_geometry_top_for_mustard_not_cap_center() -> None:
    candidate = {
        "label": "mustard_bottle",
        "top_z_m": 0.4609,
        "height_m": 0.1909,
        "chosen_target_center_base": [0.6593, 0.0603, 0.4366],
        "grasp_center_base": [0.6593, 0.0603, 0.4366],
    }
    eff, src, _meta = resolve_effective_top_z_for_palm_bridge(
        candidate, 0.4609, table_z_m=0.27
    )
    assert eff >= MUSTARD_EXPECTED_MIN_TOP_Z_M
    assert math.isclose(eff, 0.4609, abs_tol=0.002)


def test_scanner_contract_verify_pass_and_fail() -> None:
    ok, fields = verify_mustard_scanner_aligned_z_contract(
        controller_top_z=MUSTARD_SCANNER_CONTRACT_TOP_Z_M,
        controller_pregrasp_tcp_z=MUSTARD_SCANNER_CONTRACT_PREGRASP_TCP_Z_M,
        controller_grasp_tcp_z=MUSTARD_SCANNER_CONTRACT_GRASP_TCP_Z_M,
    )
    assert ok is True
    assert fields["z_match_ok"] is True
    assert fields["verify_mode"] == "strict_absolute"
    bad_ok, bad_fields = verify_mustard_scanner_aligned_z_contract(
        controller_top_z=0.4370,
        controller_pregrasp_tcp_z=0.5020,
        controller_grasp_tcp_z=0.4120,
    )
    assert bad_ok is False
    assert bad_fields["z_match_ok"] is False


def test_scanner_contract_verify_relational_runtime_top() -> None:
    """Regresión log: top_z=0.470 (corners), pre=0.4909, grasp palm_bridge=0.436."""
    ok, fields = verify_mustard_scanner_aligned_z_contract(
        controller_top_z=0.4700,
        controller_pregrasp_tcp_z=0.4909,
        controller_grasp_tcp_z=0.4360,
    )
    assert ok is True
    assert fields["verify_mode"] == "relational_runtime_top"
    assert math.isclose(
        float(fields["expected_grasp_tcp_z_for_runtime_top"]), 0.4365, abs_tol=0.001
    )


def test_top_z_source_debug_log_format() -> None:
    log = format_mustard_top_z_source_debug_log(
        position_z=0.4366,
        grasp_center_z=0.4366,
        candidate_top_z_before=0.4366,
        runtime_top_z=None,
        runtime_top_z_source=None,
        collision_center_z=0.3535,
        collision_height_m=0.1761,
        collision_top_z=0.4415,
        known_sdf_height_m=0.1909,
        table_top_z=0.27,
        known_sdf_top_z=0.4609,
        selected_top_z=0.4609,
        selected_top_z_source="known_sdf_physical_height",
    )
    assert "[MUSTARD_TOP_Z_SOURCE_DEBUG]" in log
    assert "selected_top_z=0.4609" in log
    assert "known_sdf_top_z=0.4609" in log
    assert "collision_top_z=0.4415" in log
