"""Tests candidato depth alineado con reachability scanner sugar_box."""

from panda_controller.sugar_box_depth_search import (
    DEMO_SCENE_02_SUGAR_BOX_SCANNER_ALIGNED_GRASP_Z,
    DEMO_SCENE_02_SUGAR_BOX_SCANNER_ALIGNED_PREGRASP_Z,
    build_sugar_box_demo_depth_descend_tcp_specs,
    build_sugar_box_scanner_aligned_depth_spec,
    format_sugar_box_scanner_aligned_candidate_log,
)


def test_scanner_aligned_depth_spec_values() -> None:
    spec = build_sugar_box_scanner_aligned_depth_spec(
        xy=(0.630, -0.175),
        top_z_m=0.435,
    )
    assert spec is not None
    assert spec["scanner_aligned"] is True
    assert abs(spec["pregrasp_tcp_z"] - DEMO_SCENE_02_SUGAR_BOX_SCANNER_ALIGNED_PREGRASP_Z) < 1e-6
    assert abs(spec["grasp_tcp_z"] - DEMO_SCENE_02_SUGAR_BOX_SCANNER_ALIGNED_GRASP_Z) < 1e-6
    assert abs(spec["pre_plan"][0] - 0.630) < 1e-6
    assert abs(spec["pre_plan"][1] + 0.175) < 1e-6


def test_demo_depth_specs_try_scanner_first() -> None:
    specs = build_sugar_box_demo_depth_descend_tcp_specs(
        xy=(0.630, -0.175),
        top_z_m=0.435,
    )
    assert specs
    assert specs[0].get("scanner_aligned") is True


def test_scanner_aligned_log_format() -> None:
    log = format_sugar_box_scanner_aligned_candidate_log(
        {
            "controller_pregrasp_tcp_z": 0.500,
            "controller_grasp_tcp_z": 0.420,
            "selected": True,
            "seed": "pick_workspace_ready",
        }
    )
    assert "[SUGAR_BOX_SCANNER_ALIGNED_CANDIDATE]" in log
    assert "scanner_pregrasp_tcp_z=0.5000" in log
    assert "scanner_grasp_tcp_z=0.4200" in log
    assert "selected=true" in log
