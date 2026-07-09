"""Regresión: override borderline de ancho solo para chips_can cylinder_topdown + RuntimeScene GT."""

import logging
from typing import Any, Dict

from panda_controller.perception_to_pregrasp_test import PerceptionToPregraspTest

# Caso seed1004: pasa GEOMETRIC_GRASP_CHECK pero el gate estricto (max - 3 mm) bloqueaba antes.
_BORDERLINE_REQUIRED_M = 0.0750
_BORDERLINE_MAX_GRIPPER_M = 0.0758
_BORDERLINE_OPEN_TOTAL_M = 0.0798


class _GeometryGateStub(PerceptionToPregraspTest):
    """Stub mínimo: solo métodos del gate OBJECT_GEOMETRY_CONSISTENCY."""

    def __init__(self) -> None:
        self._chips_can_allow_width_match_contact = True
        self._chips_can_geometry_width_tolerance_m = 0.003

    def _effective_open_joint(self) -> float:
        return _BORDERLINE_OPEN_TOTAL_M / 2.0

    def get_logger(self) -> logging.Logger:
        return logging.getLogger("test_chips_can_geometry_width_gate")


def _borderline_candidate(**overrides: Any) -> Dict[str, Any]:
    base: Dict[str, Any] = {
        "label": "chips_can",
        "grasp_strategy": "cylinder_topdown",
        "top_face_source": "runtime_gt_known_cylinder",
        "grasp_center_source": "runtime_gt_cylinder_center",
        "open_total_m": _BORDERLINE_OPEN_TOTAL_M,
        "effective_required_grasp_width_m": _BORDERLINE_REQUIRED_M,
        "max_expected_width_m": _BORDERLINE_MAX_GRIPPER_M,
        "collision_dims": {"shape": "cylinder", "cylinder": [0.0375, 0.25]},
    }
    base.update(overrides)
    return base


def test_chips_can_borderline_override_applied() -> None:
    """Caso 1: chips_can GT + cylinder_topdown dentro de tolerancia => OK_BORDERLINE."""
    stub = _GeometryGateStub()
    supported, verdict = stub._validate_object_geometry_consistency(_borderline_candidate())
    assert verdict == "OK_BORDERLINE_CYLINDER_WIDTH"
    assert supported is True


def test_chips_can_wrong_grasp_center_source_stays_blocked() -> None:
    """Caso 2: source distinta de runtime_gt_cylinder_center => sin override."""
    stub = _GeometryGateStub()
    supported, verdict = stub._validate_object_geometry_consistency(
        _borderline_candidate(grasp_center_source="model_box_center")
    )
    assert verdict == "BLOCKED_BY_GRIPPER_WIDTH"
    assert supported is False


def test_other_label_borderline_width_not_overridden() -> None:
    """Caso 3: mismo ancho borderline pero label != chips_can => bloqueo genérico."""
    stub = _GeometryGateStub()
    supported, verdict = stub._validate_object_geometry_consistency(
        _borderline_candidate(
            label="cracker_box",
            grasp_strategy="top_down_short_axis",
            top_face_source="runtime_gt_box_top",
            grasp_center_source="runtime_gt_box_center",
        )
    )
    assert verdict == "FAIL_GRIPPER_WIDTH_LIMIT"
    assert supported is False


def test_chips_can_required_above_max_plus_tolerance_stays_blocked() -> None:
    """Caso 4: required > max + tolerance => BLOCKED_BY_GRIPPER_WIDTH."""
    stub = _GeometryGateStub()
    supported, verdict = stub._validate_object_geometry_consistency(
        _borderline_candidate(
            effective_required_grasp_width_m=_BORDERLINE_MAX_GRIPPER_M + 0.004
        )
    )
    assert verdict == "BLOCKED_BY_GRIPPER_WIDTH"
    assert supported is False
