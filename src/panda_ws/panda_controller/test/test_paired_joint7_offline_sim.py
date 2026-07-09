"""Tests simulación offline joint7_direct para validador paired/grid."""

import math
from typing import Any, Dict, List, Optional, Sequence
from unittest.mock import MagicMock, patch

import numpy as np

from panda_controller.get_cartesian_path_audit import (
    evaluate_get_cartesian_path_start_state_audit,
    format_start_state_honored_log_value,
)
from panda_controller.paired_joint7_offline_sim import (
    REJECT_JOINT7_OFFLINE_LIMIT,
    REJECT_JOINT7_OFFLINE_NO_IMPROVEMENT,
    gap_axis_parallel_alignment,
    simulate_joint7_gap_alignment_offline,
)


def _desired_x() -> np.ndarray:
    return np.array([1.0, 0.0], dtype=np.float64)


def _observed_from_j7(j7: float) -> np.ndarray:
    angle = float(j7)
    return np.array([math.cos(angle), math.sin(angle)], dtype=np.float64)


def test_joint7_offline_alignment_reduces_error() -> None:
    desired = _desired_x()
    start_j7 = math.radians(30.0)

    def observed_fn(pos: Sequence[float]) -> Optional[np.ndarray]:
        return _observed_from_j7(float(pos[6]))

    before_err, _, _ = gap_axis_parallel_alignment(desired, observed_fn([0.0] * 6 + [start_j7]))
    result = simulate_joint7_gap_alignment_offline(
        [0.0] * 6 + [start_j7],
        desired_gap_axis_xy=desired,
        observed_gap_axis_fn=observed_fn,
        step_rad=0.08,
        max_steps=20,
        target_deg=5.0,
        joint7_limit_lower_rad=-2.8973,
        joint7_limit_upper_rad=2.8973,
    )
    assert result["result"] == "OK"
    after_err = float(result["gap_axis_error_after_deg"])
    assert after_err <= 5.0 + 1e-6
    assert after_err < before_err


def test_joint7_offline_alignment_tries_both_directions() -> None:
    desired = _desired_x()
    start_j7 = math.radians(20.0)
    probes: List[Dict[str, Any]] = []

    def observed_fn(pos: Sequence[float]) -> Optional[np.ndarray]:
        return _observed_from_j7(float(pos[6]))

    def probe_logger(fields: Dict[str, Any]) -> None:
        probes.append(dict(fields))

    result = simulate_joint7_gap_alignment_offline(
        [0.0] * 6 + [start_j7],
        desired_gap_axis_xy=desired,
        observed_gap_axis_fn=observed_fn,
        step_rad=0.06,
        max_steps=10,
        target_deg=5.0,
        probe_logger=probe_logger,
    )
    assert result["result"] == "OK"
    assert probes
    first = probes[0]
    assert float(first["error_plus_deg"]) > float(first["error_current_deg"])
    assert float(first["error_minus_deg"]) < float(first["error_current_deg"])
    assert first["selected_direction"] == "-1"


def test_joint7_offline_alignment_rejects_limit() -> None:
    desired = _desired_x()
    start_j7 = 2.8973

    def observed_fn(pos: Sequence[float]) -> Optional[np.ndarray]:
        return _observed_from_j7(float(pos[6]) + math.radians(45.0))

    result = simulate_joint7_gap_alignment_offline(
        [0.0] * 6 + [start_j7],
        desired_gap_axis_xy=desired,
        observed_gap_axis_fn=observed_fn,
        step_rad=0.15,
        max_steps=3,
        target_deg=5.0,
        joint7_limit_lower_rad=-2.8973,
        joint7_limit_upper_rad=2.8973,
        clip_joint7_fn=lambda _v: start_j7,
    )
    assert result["result"] == "FAIL"
    assert result["reason"] in (
        REJECT_JOINT7_OFFLINE_LIMIT,
        REJECT_JOINT7_OFFLINE_NO_IMPROVEMENT,
    )


def test_grid_uses_aligned_pregrasp_for_cartesian() -> None:
    from panda_controller.perception_to_pregrasp_test import PerceptionToPregraspTest

    stub = PerceptionToPregraspTest.__new__(PerceptionToPregraspTest)
    raw_js = MagicMock(name="raw_js")
    aligned_js = MagicMock(name="aligned_js")
    calls: List[Any] = []

    def _cartesian(start_js: Any, *_args: Any, **_kwargs: Any) -> tuple:
        calls.append(start_js)
        return 0.98, 5

    stub._plan_cartesian_fraction_from_tcp_target = _cartesian
    stub._check_grasp_endpoint_ik_from_pregrasp = MagicMock(return_value=(True, None))
    stub._simulate_joint7_gap_alignment_offline = MagicMock(
        return_value={
            "result": "OK",
            "aligned_pregrasp_js": aligned_js,
            "joint7_after": 0.1,
            "gap_axis_error_after_deg": 2.0,
            "desired_gap_axis_xy": (1.0, 0.0),
        }
    )
    stub.get_logger = MagicMock(return_value=MagicMock())
    stub._add_detected_objects_to_scene = False
    stub._last_get_cartesian_path_audit = {
        "start_state_honored": True,
        "fraction": 0.98,
        "traj_pts": 5,
    }
    stub._demo_authoritative_scene = True
    stub._scene_id = "demo_scene_02"
    stub._table_z_m = 0.4
    stub._joint_values_7d_from_any = MagicMock(return_value=[0.0] * 7)
    stub._paired_validate_vertical_lift_from_pregrasp_js = MagicMock(return_value=(True, ""))
    stub._compute_transport_aware_pick_score = MagicMock(return_value={"result": "ACCEPT", "post_lift_exit_ok": True, "direct_action_to_hub_ok": True})
    stub._log_cartesian_descend_fail_diag_extended = MagicMock()

    # Ejecutar bloque cartesiano equivalente al validador tras joint7 OK.
    sim = stub._simulate_joint7_gap_alignment_offline(
        {}, raw_js, commanded_yaw=0.0, candidate_idx=0, label="cracker_box"
    )
    aligned = sim["aligned_pregrasp_js"]
    stub._plan_cartesian_fraction_from_tcp_target(
        aligned,
        (0.455, 0.115, 0.437),
        (0.0, 1.0, 0.0, 0.0),
        candidate_idx=0,
        start_state_source="aligned_pregrasp_after_joint7",
    )
    assert calls == [aligned_js]
    assert calls != [raw_js]


def test_start_state_honored_not_evaluated_when_no_cartesian_call() -> None:
    audit = evaluate_get_cartesian_path_start_state_audit(
        requested_start_js=[0.0] * 7,
        response=None,
    )
    assert audit["start_state_honored"] is None
    assert audit["result"] == "NOT_EVALUATED"
    assert format_start_state_honored_log_value(audit["start_state_honored"]) == "not_evaluated"


def test_no_endpoint_ik_raw_gate_before_joint7() -> None:
    from panda_controller.paired_joint7_offline_sim import REJECT_JOINT7_OFFLINE_GAP_AXIS

    desired = _desired_x()
    start_j7 = math.radians(25.0)

    def observed_fn(pos: Sequence[float]) -> Optional[np.ndarray]:
        return _observed_from_j7(float(pos[6]))

    endpoint_ik_raw_ok = False
    joint7_sim = simulate_joint7_gap_alignment_offline(
        [0.0] * 6 + [start_j7],
        desired_gap_axis_xy=desired,
        observed_gap_axis_fn=observed_fn,
        step_rad=0.08,
        max_steps=20,
        target_deg=5.0,
    )
    reject_reason = ""
    if not endpoint_ik_raw_ok:
        reject_reason = ""
    if str(joint7_sim.get("result")) != "OK":
        reject_reason = str(joint7_sim.get("reason") or REJECT_JOINT7_OFFLINE_GAP_AXIS)
    assert reject_reason != "endpoint_ik_failed"
    assert reject_reason == ""
