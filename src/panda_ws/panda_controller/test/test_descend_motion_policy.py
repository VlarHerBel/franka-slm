"""Tests de límite de descenso cartesiano y staging alto (objetos demo)."""

from panda_controller.descend_motion_policy import (
    apply_descend_tcp_sequence,
    should_use_low_object_high_approach,
)
from panda_controller.grasp_centering_target import resolve_gripper_centering_target_xy
from panda_vision.grasp.object_grasp_policy import export_grasp_policy_for_executor


def _descend_seq(label: str, *, top_z: float = 0.285) -> dict:
    cand = export_grasp_policy_for_executor(label)
    cand.setdefault("grasp_strategy", cand.get("grasp_strategy", "top_down_short_axis"))
    return apply_descend_tcp_sequence(
        label=label,
        candidate=cand,
        top_z=top_z,
        grasp_xy=(0.56, -0.02),
        min_grasp_z_from_table=0.252,
        max_target_z=0.95,
        eff_approach_m=float(cand.get("approach_distance_min_m") or 0.10),
        eff_pregrasp_clear_m=float(cand.get("pregrasp_clearance_above_top_m") or 0.07),
        eff_safe_above_m=float(cand.get("safe_pregrasp_clearance_above_top_m") or 0.13),
        eff_safe_extra_m=float(cand.get("safe_pregrasp_extra_above_pregrasp_m") or 0.10),
        global_min_tcp_clearance_m=0.012,
        low_object_high_approach_enabled=True,
        lift_clearance_m=0.12,
    )


def test_pudding_box_vertical_descend_limited() -> None:
    seq = _descend_seq("pudding_box")
    assert 0.045 <= float(seq["final_descend_m"]) <= 0.085


def test_gelatin_box_vertical_descend_limited() -> None:
    seq = _descend_seq("gelatin_box", top_z=0.278)
    assert 0.045 <= float(seq["final_descend_m"]) <= 0.085


def test_sugar_box_vertical_descend_limited() -> None:
    seq = _descend_seq("sugar_box", top_z=0.310)
    assert 0.050 <= float(seq["final_descend_m"]) <= 0.085


def test_cracker_box_allows_larger_descend() -> None:
    seq = _descend_seq("cracker_box", top_z=0.340)
    assert float(seq["final_descend_m"]) >= 0.050


def test_edge_grasp_high_approach_stage_enabled() -> None:
    cand = export_grasp_policy_for_executor("pudding_box")
    assert should_use_low_object_high_approach(cand, 0.285, enabled=True)
    seq = _descend_seq("pudding_box")
    assert seq.get("object_high_pregrasp_tcp") is not None
    assert bool(seq.get("uses_low_object_high_approach_stage"))


def test_edge_grasp_centering_uses_runtime_pregrasp_xy() -> None:
    obj = {
        "label": "pudding_box",
        "grasp_strategy": "edge_grasp",
        "edge_grasp_requested": True,
        "_runtime_pregrasp_tcp_xy": [0.562, -0.010],
        "known_box_center_base": [0.565, -0.025, 0.285],
    }
    target, source = resolve_gripper_centering_target_xy(
        obj, commanded_hand_xy=(0.562, -0.010)
    )
    assert source == "commanded_pregrasp_hand_xy"
    assert target is not None
    assert abs(float(target[0]) - 0.562) < 1e-6
