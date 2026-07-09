"""Regresión: chips_can GT cilindro (centrado/yaw) y suelo de reachability."""

import logging

from panda_controller.perception_to_pregrasp_test import PerceptionToPregraspTest


class _ReachabilityFloorStub(PerceptionToPregraspTest):
    def __init__(self) -> None:
        self._chips_can_min_pregrasp_clearance_above_top_m = 0.075
        self._chips_can_max_cartesian_descend_m = 0.120
        self._min_pregrasp_z = 0.20
        self._max_target_z = 1.20

    def get_logger(self) -> logging.Logger:
        return logging.getLogger("test_chips_can_reachability_floor")

    def _is_chips_can_cylinder_topdown_candidate(self, candidate):  # type: ignore[override]
        return (
            str(candidate.get("label", "")).lower() == "chips_can"
            and str(candidate.get("grasp_strategy", "")) == "cylinder_topdown"
        )


def _compute_chips_can_reachability_z(
    stub: _ReachabilityFloorStub,
    *,
    pre_z: float,
    grasp_z: float,
    top_z: float,
    preferred_descend: float = 0.060,
) -> tuple[float, float, float]:
    """Réplica mínima del suelo chips_can en _resolve_reachable_pregrasp_for_pick."""
    min_descend = 0.04
    max_descend = 0.120
    desired_descend = preferred_descend
    desired_descend = min(max_descend, max(min_descend, desired_descend))
    if stub._is_chips_can_cylinder_topdown_candidate(
        {"label": "chips_can", "grasp_strategy": "cylinder_topdown"}
    ):
        floor_z = max(
            pre_z,
            top_z + float(stub._chips_can_min_pregrasp_clearance_above_top_m),
            grasp_z + min_descend,
        )
        plan_descend = max(0.0, floor_z - grasp_z)
        desired_descend = max(desired_descend, plan_descend)
    desired_pre_z = grasp_z + desired_descend
    min_pre_z = max(float(stub._min_pregrasp_z), grasp_z + min_descend)
    if stub._is_chips_can_cylinder_topdown_candidate(
        {"label": "chips_can", "grasp_strategy": "cylinder_topdown"}
    ):
        min_pre_z = max(min_pre_z, desired_pre_z)
    start_z = max(min_pre_z, desired_pre_z)
    return desired_pre_z, min_pre_z, start_z


def test_reachability_floor_keeps_clearance_pregrasp() -> None:
    stub = _ReachabilityFloorStub()
    top_z = 0.511
    grasp_z = top_z - 0.035
    pre_z = top_z + 0.075
    desired_pre_z, min_pre_z, start_z = _compute_chips_can_reachability_z(
        stub, pre_z=pre_z, grasp_z=grasp_z, top_z=top_z
    )
    assert abs(desired_pre_z - (top_z + 0.075)) < 1e-6
    assert abs(start_z - desired_pre_z) < 1e-6
    assert start_z >= min_pre_z


class _GtCylinderStub(PerceptionToPregraspTest):
    def __init__(self) -> None:
        self._chips_can_skip_gripper_gap_alignment_when_gt = True
        self._chips_can_gt_cylinder_centering_enabled = True
        self._chips_can_gt_cylinder_centering_max_error_xy_m = 0.006

    def get_logger(self) -> logging.Logger:
        return logging.getLogger("test_chips_can_gt_cylinder")


def _gt_candidate() -> dict:
    return {
        "label": "chips_can",
        "grasp_strategy": "cylinder_topdown",
        "grasp_center_source": "runtime_gt_cylinder_center",
        "top_face_source": "runtime_gt_known_cylinder",
        "grasp_center_base": [0.555, 0.190, 0.511],
        "collision_dims": {
            "shape": "cylinder",
            "cylinder": [0.0375, 0.25],
        },
    }


def test_gt_cylinder_candidate_detection() -> None:
    stub = _GtCylinderStub()
    assert stub._is_chips_can_gt_cylinder_candidate(_gt_candidate())
    assert stub._chips_can_skip_gripper_gap_alignment(_gt_candidate())


def test_gt_cylinder_axis_center_from_grasp_center() -> None:
    stub = _GtCylinderStub()
    xy = stub._resolve_chips_can_gt_cylinder_axis_center_xy(_gt_candidate())
    assert xy is not None
    assert abs(float(xy[0]) - 0.555) < 1e-6
    assert abs(float(xy[1]) - 0.190) < 1e-6


def test_gt_cylinder_centering_within_threshold() -> None:
    stub = _GtCylinderStub()
    metrics = {
        "error_xy_m": 0.004,
    }
    assert stub._chips_can_gt_cylinder_centering_within_thresholds(metrics)
