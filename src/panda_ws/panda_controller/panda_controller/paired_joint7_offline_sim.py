"""Simulación offline robusta de joint7_direct para validación paired/grid."""

from __future__ import annotations

import copy
import math
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from panda_controller.tfg_motion_waypoints import PANDA_ARM_JOINT_NAMES, joint_values_7d_from_any

REJECT_JOINT7_OFFLINE_SIM_FAIL = "joint7_offline_sim_fail"
REJECT_JOINT7_OFFLINE_LIMIT = "joint7_offline_limit"
REJECT_JOINT7_OFFLINE_NO_IMPROVEMENT = "joint7_offline_no_improvement"
REJECT_JOINT7_OFFLINE_GAP_AXIS = "joint7_offline_gap_axis_error"
REJECT_JOINT7_OFFLINE_TCP_SHIFT = "joint7_offline_tcp_shift"
REJECT_JOINT7_OFFLINE_TOP_DOWN = "joint7_offline_top_down_invalid"
REJECT_ENDPOINT_IK_RAW_FAILED = "endpoint_ik_raw_failed"

DEFAULT_PAIRED_JOINT7_OFFLINE_STEP_RAD = 0.04
DEFAULT_PAIRED_JOINT7_OFFLINE_MAX_STEPS = 40
DEFAULT_PAIRED_JOINT7_OFFLINE_MIN_IMPROVEMENT_DEG = 0.2
DEFAULT_PAIRED_JOINT7_OFFLINE_HARD_MAX_DEG = 8.0
DEFAULT_PAIRED_JOINT7_OFFLINE_TCP_SHIFT_MAX_M = 0.008


def wrap_to_pi(angle: float) -> float:
    a = float(angle)
    while a > math.pi:
        a -= 2.0 * math.pi
    while a < -math.pi:
        a += 2.0 * math.pi
    return a


def gap_axis_parallel_alignment(
    desired_xy: Sequence[float], observed_xy: Sequence[float]
) -> Tuple[float, float, np.ndarray]:
    d = np.asarray(desired_xy, dtype=np.float64)
    o = np.asarray(observed_xy, dtype=np.float64)
    dn = float(np.linalg.norm(d))
    on = float(np.linalg.norm(o))
    if dn < 1e-9 or on < 1e-9:
        return 180.0, 0.0, d
    d = d / dn
    o = o / on
    dot_pos = float(np.dot(o, d))
    dot_neg = float(np.dot(o, -d))
    target = d if abs(dot_pos) >= abs(dot_neg) else -d
    dot = float(np.clip(np.dot(o, target), -1.0, 1.0))
    angle_deg = math.degrees(math.acos(abs(dot)))
    return angle_deg, dot, target


def unit_xy_or_none(raw: Any) -> Optional[np.ndarray]:
    if isinstance(raw, np.ndarray):
        if raw.size < 2:
            return None
        v = np.asarray(raw[:2], dtype=np.float64)
    elif isinstance(raw, (list, tuple)) and len(raw) >= 2:
        v = np.array([float(raw[0]), float(raw[1])], dtype=np.float64)
    else:
        return None
    n = float(np.linalg.norm(v))
    if n < 1e-9:
        return None
    return v / n


def clip_joint7(
    value_rad: float,
    *,
    lower_rad: float,
    upper_rad: float,
) -> float:
    return float(min(float(upper_rad), max(float(lower_rad), float(value_rad))))


def joint7_limit_ok(
    joint7_rad: float,
    *,
    lower_rad: float,
    upper_rad: float,
    min_margin_rad: float = 0.02,
) -> bool:
    lo_m = float(joint7_rad) - float(lower_rad)
    hi_m = float(upper_rad) - float(joint7_rad)
    return lo_m >= float(min_margin_rad) and hi_m >= float(min_margin_rad)


def simulate_joint7_gap_alignment_offline(
    pregrasp_js_raw: Any,
    *,
    desired_gap_axis_xy: Sequence[float],
    observed_gap_axis_fn: Callable[[Sequence[float]], Optional[Sequence[float]]],
    joint7_idx: int = 6,
    step_rad: float = DEFAULT_PAIRED_JOINT7_OFFLINE_STEP_RAD,
    max_steps: int = DEFAULT_PAIRED_JOINT7_OFFLINE_MAX_STEPS,
    min_improvement_deg: float = DEFAULT_PAIRED_JOINT7_OFFLINE_MIN_IMPROVEMENT_DEG,
    target_deg: float = 5.0,
    hard_max_deg: float = DEFAULT_PAIRED_JOINT7_OFFLINE_HARD_MAX_DEG,
    joint7_limit_lower_rad: float = -2.8973,
    joint7_limit_upper_rad: float = 2.8973,
    clip_joint7_fn: Optional[Callable[[float], float]] = None,
    tcp_shift_fn: Optional[
        Callable[[Sequence[float], Sequence[float]], Optional[float]]
    ] = None,
    hand_shift_fn: Optional[
        Callable[[Sequence[float], Sequence[float]], Optional[float]]
    ] = None,
    top_down_valid_fn: Optional[Callable[[Sequence[float]], bool]] = None,
    tcp_shift_max_m: float = DEFAULT_PAIRED_JOINT7_OFFLINE_TCP_SHIFT_MAX_M,
    probe_logger: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> Dict[str, Any]:
    """Replica offline el bucle joint7_direct probando ±step y eligiendo mejor dirección."""
    raw_positions = joint_values_7d_from_any(
        pregrasp_js_raw, context="paired_joint7_offline_raw"
    )
    desired = unit_xy_or_none(desired_gap_axis_xy)
    if raw_positions is None or desired is None:
        return {
            "result": "FAIL",
            "reason": REJECT_JOINT7_OFFLINE_SIM_FAIL,
            "aligned_pregrasp_js": pregrasp_js_raw,
            "joint7_before": None,
            "joint7_after": None,
            "gap_axis_error_before_deg": None,
            "gap_axis_error_after_deg": None,
        }

    positions = [float(v) for v in raw_positions]
    joint7_before = float(positions[joint7_idx])
    _clip = clip_joint7_fn or (
        lambda v: clip_joint7(
            v,
            lower_rad=joint7_limit_lower_rad,
            upper_rad=joint7_limit_upper_rad,
        )
    )

    def _error_for(pos: Sequence[float]) -> Optional[float]:
        observed = observed_gap_axis_fn(pos)
        if observed is None:
            return None
        err_deg, _, _ = gap_axis_parallel_alignment(desired, observed)
        return float(err_deg)

    error_before = _error_for(positions)
    if error_before is None:
        return {
            "result": "FAIL",
            "reason": REJECT_JOINT7_OFFLINE_GAP_AXIS,
            "aligned_pregrasp_js": pregrasp_js_raw,
            "joint7_before": joint7_before,
            "joint7_after": joint7_before,
            "gap_axis_error_before_deg": None,
            "gap_axis_error_after_deg": None,
        }

    best_direction = "none"
    iterations = 0
    current_err = float(error_before)
    if current_err <= float(target_deg) + 1e-6:
        aligned = copy.deepcopy(positions)
        return {
            "result": "OK",
            "reason": "",
            "aligned_pregrasp_js": aligned,
            "joint7_before": joint7_before,
            "joint7_after": float(aligned[joint7_idx]),
            "joint7_total_delta_deg": 0.0,
            "step_rad": float(step_rad),
            "iterations": 0,
            "gap_axis_error_before_deg": float(error_before),
            "gap_axis_error_after_deg": float(current_err),
            "best_direction": best_direction,
            "joint7_limit_ok": joint7_limit_ok(
                float(aligned[joint7_idx]),
                lower_rad=joint7_limit_lower_rad,
                upper_rad=joint7_limit_upper_rad,
            ),
            "tcp_shift_after_joint7_m": 0.0,
            "hand_shift_after_joint7_m": 0.0,
        }

    if current_err > float(hard_max_deg) + 1e-6:
        pass

    for iteration in range(1, int(max_steps) + 1):
        err_current = _error_for(positions)
        if err_current is None:
            return {
                "result": "FAIL",
                "reason": REJECT_JOINT7_OFFLINE_GAP_AXIS,
                "aligned_pregrasp_js": positions,
                "joint7_before": joint7_before,
                "joint7_after": float(positions[joint7_idx]),
                "gap_axis_error_before_deg": float(error_before),
                "gap_axis_error_after_deg": err_current,
                "iterations": iterations,
            }
        current_err = float(err_current)
        if current_err <= float(target_deg) + 1e-6:
            break

        pos_plus = list(positions)
        pos_minus = list(positions)
        j7_plus = _clip(float(positions[joint7_idx]) + float(step_rad))
        j7_minus = _clip(float(positions[joint7_idx]) - float(step_rad))
        pos_plus[joint7_idx] = j7_plus
        pos_minus[joint7_idx] = j7_minus

        err_plus = _error_for(pos_plus)
        err_minus = _error_for(pos_minus)
        err_plus_v = float(err_plus) if err_plus is not None else float("inf")
        err_minus_v = float(err_minus) if err_minus is not None else float("inf")

        if err_plus_v <= err_minus_v:
            selected_direction = "+1"
            selected_err = err_plus_v
            next_positions = pos_plus
            at_limit = abs(j7_plus - float(positions[joint7_idx])) < 1e-9
        else:
            selected_direction = "-1"
            selected_err = err_minus_v
            next_positions = pos_minus
            at_limit = abs(j7_minus - float(positions[joint7_idx])) < 1e-9

        if probe_logger is not None:
            probe_logger(
                {
                    "iter": int(iteration),
                    "joint7_current": float(positions[joint7_idx]),
                    "error_current_deg": float(current_err),
                    "error_plus_deg": err_plus_v,
                    "error_minus_deg": err_minus_v,
                    "selected_direction": selected_direction,
                    "selected_error_deg": float(selected_err),
                }
            )

        improvement = float(current_err) - float(selected_err)
        if improvement + 1e-6 < float(min_improvement_deg):
            return {
                "result": "FAIL",
                "reason": REJECT_JOINT7_OFFLINE_NO_IMPROVEMENT,
                "aligned_pregrasp_js": positions,
                "joint7_before": joint7_before,
                "joint7_after": float(positions[joint7_idx]),
                "gap_axis_error_before_deg": float(error_before),
                "gap_axis_error_after_deg": float(current_err),
                "iterations": iterations,
                "best_direction": best_direction,
            }

        if at_limit:
            return {
                "result": "FAIL",
                "reason": REJECT_JOINT7_OFFLINE_LIMIT,
                "aligned_pregrasp_js": positions,
                "joint7_before": joint7_before,
                "joint7_after": float(positions[joint7_idx]),
                "gap_axis_error_before_deg": float(error_before),
                "gap_axis_error_after_deg": float(current_err),
                "iterations": iterations,
                "best_direction": best_direction,
            }

        positions = list(next_positions)
        best_direction = selected_direction
        iterations = int(iteration)

    error_after = _error_for(positions)
    if error_after is None:
        return {
            "result": "FAIL",
            "reason": REJECT_JOINT7_OFFLINE_GAP_AXIS,
            "aligned_pregrasp_js": positions,
            "joint7_before": joint7_before,
            "joint7_after": float(positions[joint7_idx]),
            "gap_axis_error_before_deg": float(error_before),
            "gap_axis_error_after_deg": None,
            "iterations": iterations,
        }

    if float(error_after) > float(target_deg) + 1e-6:
        return {
            "result": "FAIL",
            "reason": REJECT_JOINT7_OFFLINE_GAP_AXIS,
            "aligned_pregrasp_js": positions,
            "joint7_before": joint7_before,
            "joint7_after": float(positions[joint7_idx]),
            "gap_axis_error_before_deg": float(error_before),
            "gap_axis_error_after_deg": float(error_after),
            "iterations": iterations,
            "best_direction": best_direction,
        }

    limit_ok = joint7_limit_ok(
        float(positions[joint7_idx]),
        lower_rad=joint7_limit_lower_rad,
        upper_rad=joint7_limit_upper_rad,
    )
    if not limit_ok:
        return {
            "result": "FAIL",
            "reason": REJECT_JOINT7_OFFLINE_LIMIT,
            "aligned_pregrasp_js": positions,
            "joint7_before": joint7_before,
            "joint7_after": float(positions[joint7_idx]),
            "gap_axis_error_before_deg": float(error_before),
            "gap_axis_error_after_deg": float(error_after),
            "iterations": iterations,
            "joint7_limit_ok": False,
        }

    tcp_shift = 0.0
    if tcp_shift_fn is not None:
        shift = tcp_shift_fn(raw_positions, positions)
        if shift is not None:
            tcp_shift = float(shift)
            if tcp_shift > float(tcp_shift_max_m) + 1e-9:
                return {
                    "result": "FAIL",
                    "reason": REJECT_JOINT7_OFFLINE_TCP_SHIFT,
                    "aligned_pregrasp_js": positions,
                    "joint7_before": joint7_before,
                    "joint7_after": float(positions[joint7_idx]),
                    "gap_axis_error_before_deg": float(error_before),
                    "gap_axis_error_after_deg": float(error_after),
                    "tcp_shift_after_joint7_m": tcp_shift,
                    "iterations": iterations,
                }

    hand_shift = 0.0
    if hand_shift_fn is not None:
        shift = hand_shift_fn(raw_positions, positions)
        if shift is not None:
            hand_shift = float(shift)

    if top_down_valid_fn is not None and not bool(top_down_valid_fn(positions)):
        return {
            "result": "FAIL",
            "reason": REJECT_JOINT7_OFFLINE_TOP_DOWN,
            "aligned_pregrasp_js": positions,
            "joint7_before": joint7_before,
            "joint7_after": float(positions[joint7_idx]),
            "gap_axis_error_before_deg": float(error_before),
            "gap_axis_error_after_deg": float(error_after),
            "iterations": iterations,
        }

    total_delta_deg = math.degrees(
        wrap_to_pi(float(positions[joint7_idx]) - float(joint7_before))
    )
    return {
        "result": "OK",
        "reason": "",
        "aligned_pregrasp_js": positions,
        "joint7_before": joint7_before,
        "joint7_after": float(positions[joint7_idx]),
        "joint7_total_delta_deg": float(total_delta_deg),
        "step_rad": float(step_rad),
        "iterations": int(iterations),
        "gap_axis_error_before_deg": float(error_before),
        "gap_axis_error_after_deg": float(error_after),
        "best_direction": best_direction,
        "joint7_limit_ok": True,
        "tcp_shift_after_joint7_m": float(tcp_shift),
        "hand_shift_after_joint7_m": float(hand_shift),
        "mode": "iterative",
    }


def compute_axis_equiv_delta_yaw(
    observed_xy: Sequence[float], desired_xy: Sequence[float]
) -> Tuple[float, np.ndarray, float]:
    o = np.asarray(observed_xy, dtype=np.float64)
    d = np.asarray(desired_xy, dtype=np.float64)
    on = float(np.linalg.norm(o))
    dn = float(np.linalg.norm(d))
    if on < 1e-9 or dn < 1e-9:
        return 0.0, d, 180.0
    o = o / on
    d = d / dn
    d_neg = -d

    def _signed_delta(from_v: np.ndarray, to_v: np.ndarray) -> float:
        cross_z = float(from_v[0] * to_v[1] - from_v[1] * to_v[0])
        dot = float(np.dot(from_v, to_v))
        return math.atan2(cross_z, dot)

    delta_to_desired = _signed_delta(o, d)
    delta_to_neg = _signed_delta(o, d_neg)
    if abs(delta_to_neg) < abs(delta_to_desired):
        selected_target = d_neg
        delta_raw = float(delta_to_neg)
    else:
        selected_target = d
        delta_raw = float(delta_to_desired)
    delta_yaw = wrap_to_pi(delta_raw)
    if delta_yaw > math.pi / 2.0:
        delta_yaw -= math.pi
    elif delta_yaw < -math.pi / 2.0:
        delta_yaw += math.pi
    dot_sel = float(np.clip(np.dot(o, selected_target), -1.0, 1.0))
    angle_error_deg = math.degrees(math.acos(abs(dot_sel)))
    return float(delta_yaw), selected_target, float(angle_error_deg)


def _finalize_joint7_alignment(
    *,
    raw_positions: Sequence[float],
    positions: List[float],
    joint7_idx: int,
    joint7_before: float,
    error_before: float,
    step_rad: float,
    iterations: int,
    best_direction: str,
    target_deg: float,
    joint7_limit_lower_rad: float,
    joint7_limit_upper_rad: float,
    observed_gap_axis_fn: Callable[[Sequence[float]], Optional[Sequence[float]]],
    desired: np.ndarray,
    tcp_shift_fn: Optional[Callable[[Sequence[float], Sequence[float]], Optional[float]]],
    hand_shift_fn: Optional[Callable[[Sequence[float], Sequence[float]], Optional[float]]],
    top_down_valid_fn: Optional[Callable[[Sequence[float]], bool]],
    tcp_shift_max_m: float,
    mode: str,
) -> Dict[str, Any]:
    def _error_for(pos: Sequence[float]) -> Optional[float]:
        observed = observed_gap_axis_fn(pos)
        if observed is None:
            return None
        err_deg, _, _ = gap_axis_parallel_alignment(desired, observed)
        return float(err_deg)

    error_after = _error_for(positions)
    if error_after is None:
        return {
            "result": "FAIL",
            "reason": REJECT_JOINT7_OFFLINE_GAP_AXIS,
            "aligned_pregrasp_js": positions,
            "joint7_before": joint7_before,
            "joint7_after": float(positions[joint7_idx]),
            "gap_axis_error_before_deg": float(error_before),
            "gap_axis_error_after_deg": None,
            "iterations": iterations,
            "mode": mode,
        }
    if float(error_after) > float(target_deg) + 1e-6:
        return {
            "result": "FAIL",
            "reason": REJECT_JOINT7_OFFLINE_GAP_AXIS,
            "aligned_pregrasp_js": positions,
            "joint7_before": joint7_before,
            "joint7_after": float(positions[joint7_idx]),
            "gap_axis_error_before_deg": float(error_before),
            "gap_axis_error_after_deg": float(error_after),
            "iterations": iterations,
            "best_direction": best_direction,
            "mode": mode,
        }
    if not joint7_limit_ok(
        float(positions[joint7_idx]),
        lower_rad=joint7_limit_lower_rad,
        upper_rad=joint7_limit_upper_rad,
    ):
        return {
            "result": "FAIL",
            "reason": REJECT_JOINT7_OFFLINE_LIMIT,
            "aligned_pregrasp_js": positions,
            "joint7_before": joint7_before,
            "joint7_after": float(positions[joint7_idx]),
            "gap_axis_error_before_deg": float(error_before),
            "gap_axis_error_after_deg": float(error_after),
            "iterations": iterations,
            "joint7_limit_ok": False,
            "mode": mode,
        }
    tcp_shift = 0.0
    if tcp_shift_fn is not None:
        shift = tcp_shift_fn(raw_positions, positions)
        if shift is not None:
            tcp_shift = float(shift)
            if tcp_shift > float(tcp_shift_max_m) + 1e-9:
                return {
                    "result": "FAIL",
                    "reason": REJECT_JOINT7_OFFLINE_TCP_SHIFT,
                    "aligned_pregrasp_js": positions,
                    "joint7_before": joint7_before,
                    "joint7_after": float(positions[joint7_idx]),
                    "gap_axis_error_before_deg": float(error_before),
                    "gap_axis_error_after_deg": float(error_after),
                    "tcp_shift_after_joint7_m": tcp_shift,
                    "iterations": iterations,
                    "mode": mode,
                }
    hand_shift = 0.0
    if hand_shift_fn is not None:
        shift = hand_shift_fn(raw_positions, positions)
        if shift is not None:
            hand_shift = float(shift)
    if top_down_valid_fn is not None and not bool(top_down_valid_fn(positions)):
        return {
            "result": "FAIL",
            "reason": REJECT_JOINT7_OFFLINE_TOP_DOWN,
            "aligned_pregrasp_js": positions,
            "joint7_before": joint7_before,
            "joint7_after": float(positions[joint7_idx]),
            "gap_axis_error_before_deg": float(error_before),
            "gap_axis_error_after_deg": float(error_after),
            "iterations": iterations,
            "mode": mode,
        }
    total_delta_deg = math.degrees(
        wrap_to_pi(float(positions[joint7_idx]) - float(joint7_before))
    )
    return {
        "result": "OK",
        "reason": "",
        "aligned_pregrasp_js": positions,
        "joint7_before": joint7_before,
        "joint7_after": float(positions[joint7_idx]),
        "joint7_total_delta_deg": float(total_delta_deg),
        "step_rad": float(step_rad),
        "iterations": int(iterations),
        "gap_axis_error_before_deg": float(error_before),
        "gap_axis_error_after_deg": float(error_after),
        "best_direction": best_direction,
        "joint7_limit_ok": True,
        "tcp_shift_after_joint7_m": float(tcp_shift),
        "hand_shift_after_joint7_m": float(hand_shift),
        "mode": mode,
    }


def simulate_joint7_gap_alignment_offline_fast(
    pregrasp_js_raw: Any,
    *,
    desired_gap_axis_xy: Sequence[float],
    observed_gap_axis_fn: Callable[[Sequence[float]], Optional[Sequence[float]]],
    joint7_idx: int = 6,
    micro_probe_rad: float = 0.04,
    max_jump_rad: float = 1.20,
    fine_steps: int = 3,
    min_improvement_deg: float = DEFAULT_PAIRED_JOINT7_OFFLINE_MIN_IMPROVEMENT_DEG,
    target_deg: float = 5.0,
    joint7_limit_lower_rad: float = -2.8973,
    joint7_limit_upper_rad: float = 2.8973,
    clip_joint7_fn: Optional[Callable[[float], float]] = None,
    tcp_shift_fn: Optional[
        Callable[[Sequence[float], Sequence[float]], Optional[float]]
    ] = None,
    hand_shift_fn: Optional[
        Callable[[Sequence[float], Sequence[float]], Optional[float]]
    ] = None,
    top_down_valid_fn: Optional[Callable[[Sequence[float]], bool]] = None,
    tcp_shift_max_m: float = DEFAULT_PAIRED_JOINT7_OFFLINE_TCP_SHIFT_MAX_M,
    probe_logger: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> Dict[str, Any]:
    """Probe ±micro, salto estimado por delta_yaw, fine-tune corto."""
    raw_positions = joint_values_7d_from_any(
        pregrasp_js_raw, context="paired_joint7_offline_fast_raw"
    )
    desired = unit_xy_or_none(desired_gap_axis_xy)
    if raw_positions is None or desired is None:
        return {
            "result": "FAIL",
            "reason": REJECT_JOINT7_OFFLINE_SIM_FAIL,
            "aligned_pregrasp_js": pregrasp_js_raw,
            "mode": "fast",
        }
    positions = [float(v) for v in raw_positions]
    joint7_before = float(positions[joint7_idx])
    _clip = clip_joint7_fn or (
        lambda v: clip_joint7(
            v,
            lower_rad=joint7_limit_lower_rad,
            upper_rad=joint7_limit_upper_rad,
        )
    )

    def _error_for(pos: Sequence[float]) -> Optional[float]:
        observed = observed_gap_axis_fn(pos)
        if observed is None:
            return None
        err_deg, _, _ = gap_axis_parallel_alignment(desired, observed)
        return float(err_deg)

    error_before = _error_for(positions)
    if error_before is None:
        return {
            "result": "FAIL",
            "reason": REJECT_JOINT7_OFFLINE_GAP_AXIS,
            "aligned_pregrasp_js": pregrasp_js_raw,
            "mode": "fast",
        }
    if float(error_before) <= float(target_deg) + 1e-6:
        return _finalize_joint7_alignment(
            raw_positions=raw_positions,
            positions=positions,
            joint7_idx=joint7_idx,
            joint7_before=joint7_before,
            error_before=float(error_before),
            step_rad=float(micro_probe_rad),
            iterations=0,
            best_direction="none",
            target_deg=float(target_deg),
            joint7_limit_lower_rad=joint7_limit_lower_rad,
            joint7_limit_upper_rad=joint7_limit_upper_rad,
            observed_gap_axis_fn=observed_gap_axis_fn,
            desired=desired,
            tcp_shift_fn=tcp_shift_fn,
            hand_shift_fn=hand_shift_fn,
            top_down_valid_fn=top_down_valid_fn,
            tcp_shift_max_m=tcp_shift_max_m,
            mode="fast",
        )

    probe = float(micro_probe_rad)
    pos_plus = list(positions)
    pos_minus = list(positions)
    pos_plus[joint7_idx] = _clip(float(positions[joint7_idx]) + probe)
    pos_minus[joint7_idx] = _clip(float(positions[joint7_idx]) - probe)
    err_plus = _error_for(pos_plus)
    err_minus = _error_for(pos_minus)
    err_plus_v = float(err_plus) if err_plus is not None else float("inf")
    err_minus_v = float(err_minus) if err_minus is not None else float("inf")
    if err_plus_v <= err_minus_v:
        direction_sign = 1.0
        best_direction = "+1"
    else:
        direction_sign = -1.0
        best_direction = "-1"
    if probe_logger is not None:
        probe_logger(
            {
                "iter": 0,
                "joint7_current": float(positions[joint7_idx]),
                "error_current_deg": float(error_before),
                "error_plus_deg": err_plus_v,
                "error_minus_deg": err_minus_v,
                "selected_direction": best_direction,
                "selected_error_deg": min(err_plus_v, err_minus_v),
            }
        )

    observed_now = observed_gap_axis_fn(positions)
    if observed_now is None:
        return {
            "result": "FAIL",
            "reason": REJECT_JOINT7_OFFLINE_GAP_AXIS,
            "aligned_pregrasp_js": positions,
            "mode": "fast",
        }
    delta_yaw, _, _ = compute_axis_equiv_delta_yaw(observed_now, desired)
    jump = float(direction_sign) * min(float(max_jump_rad), abs(float(delta_yaw)))
    positions[joint7_idx] = _clip(float(positions[joint7_idx]) + jump)
    iterations = 1

    for fine_i in range(1, int(fine_steps) + 1):
        err_current = _error_for(positions)
        if err_current is None:
            break
        if float(err_current) <= float(target_deg) + 1e-6:
            break
        pos_p = list(positions)
        pos_m = list(positions)
        pos_p[joint7_idx] = _clip(float(positions[joint7_idx]) + probe)
        pos_m[joint7_idx] = _clip(float(positions[joint7_idx]) - probe)
        e_p = _error_for(pos_p)
        e_m = _error_for(pos_m)
        e_p_v = float(e_p) if e_p is not None else float("inf")
        e_m_v = float(e_m) if e_m is not None else float("inf")
        if e_p_v <= e_m_v:
            nxt = pos_p
            sel_dir = "+1"
            sel_err = e_p_v
        else:
            nxt = pos_m
            sel_dir = "-1"
            sel_err = e_m_v
        if float(err_current) - float(sel_err) + 1e-6 < float(min_improvement_deg):
            break
        if abs(float(nxt[joint7_idx]) - float(positions[joint7_idx])) < 1e-9:
            break
        positions = list(nxt)
        best_direction = sel_dir
        iterations += 1
        if probe_logger is not None:
            probe_logger(
                {
                    "iter": int(fine_i),
                    "joint7_current": float(positions[joint7_idx]),
                    "error_current_deg": float(err_current),
                    "error_plus_deg": e_p_v,
                    "error_minus_deg": e_m_v,
                    "selected_direction": sel_dir,
                    "selected_error_deg": float(sel_err),
                }
            )

    return _finalize_joint7_alignment(
        raw_positions=raw_positions,
        positions=positions,
        joint7_idx=joint7_idx,
        joint7_before=joint7_before,
        error_before=float(error_before),
        step_rad=float(micro_probe_rad),
        iterations=int(iterations),
        best_direction=best_direction,
        target_deg=float(target_deg),
        joint7_limit_lower_rad=joint7_limit_lower_rad,
        joint7_limit_upper_rad=joint7_limit_upper_rad,
        observed_gap_axis_fn=observed_gap_axis_fn,
        desired=desired,
        tcp_shift_fn=tcp_shift_fn,
        hand_shift_fn=hand_shift_fn,
        top_down_valid_fn=top_down_valid_fn,
        tcp_shift_max_m=tcp_shift_max_m,
        mode="fast",
    )


def joint_state_from_positions(
    positions: Sequence[float],
    joint_names: Sequence[str] = PANDA_ARM_JOINT_NAMES,
) -> Any:
    from sensor_msgs.msg import JointState

    js = JointState()
    js.name = [str(n) for n in joint_names]
    js.position = [float(v) for v in positions]
    return js


def format_paired_joint7_offline_probe_log(fields: Dict[str, Any]) -> str:
    return (
        "[PAIRED_JOINT7_OFFLINE_PROBE]\n"
        "candidate_idx=%s\n"
        "iter=%s\n"
        "joint7_current=%s\n"
        "error_current_deg=%s\n"
        "error_plus_deg=%s\n"
        "error_minus_deg=%s\n"
        "selected_direction=%s\n"
        "selected_error_deg=%s"
        % (
            fields.get("candidate_idx", "n/a"),
            fields.get("iter", "n/a"),
            "n/a"
            if fields.get("joint7_current") is None
            else "%.4f" % float(fields["joint7_current"]),
            "n/a"
            if fields.get("error_current_deg") is None
            else "%.2f" % float(fields["error_current_deg"]),
            "n/a"
            if fields.get("error_plus_deg") is None
            else "%.2f" % float(fields["error_plus_deg"]),
            "n/a"
            if fields.get("error_minus_deg") is None
            else "%.2f" % float(fields["error_minus_deg"]),
            str(fields.get("selected_direction", "n/a")),
            "n/a"
            if fields.get("selected_error_deg") is None
            else "%.2f" % float(fields["selected_error_deg"]),
        )
    )


def format_paired_joint7_offline_sim_log(fields: Dict[str, Any]) -> str:
    return (
        "[PAIRED_JOINT7_OFFLINE_SIM]\n"
        "candidate_idx=%s\n"
        "label=%s\n"
        "desired_gap_axis_source=%s\n"
        "desired_gap_axis_xy=%s\n"
        "observed_gap_axis_source=%s\n"
        "joint7_before=%s\n"
        "joint7_after=%s\n"
        "joint7_total_delta_deg=%s\n"
        "step_rad=%s\n"
        "iterations=%s\n"
        "gap_axis_error_before_deg=%s\n"
        "gap_axis_error_after_deg=%s\n"
        "best_direction=%s\n"
        "joint7_limit_ok=%s\n"
        "tcp_shift_after_joint7_m=%s\n"
        "hand_shift_after_joint7_m=%s\n"
        "result=%s\n"
        "reason=%s"
        % (
            fields.get("candidate_idx", "n/a"),
            fields.get("label", "n/a"),
            fields.get("desired_gap_axis_source", "n/a"),
            fields.get("desired_gap_axis_xy", "n/a"),
            fields.get("observed_gap_axis_source", "n/a"),
            "n/a"
            if fields.get("joint7_before") is None
            else "%.4f" % float(fields["joint7_before"]),
            "n/a"
            if fields.get("joint7_after") is None
            else "%.4f" % float(fields["joint7_after"]),
            "n/a"
            if fields.get("joint7_total_delta_deg") is None
            else "%.2f" % float(fields["joint7_total_delta_deg"]),
            "n/a"
            if fields.get("step_rad") is None
            else "%.4f" % float(fields["step_rad"]),
            fields.get("iterations", "n/a"),
            "n/a"
            if fields.get("gap_axis_error_before_deg") is None
            else "%.2f" % float(fields["gap_axis_error_before_deg"]),
            "n/a"
            if fields.get("gap_axis_error_after_deg") is None
            else "%.2f" % float(fields["gap_axis_error_after_deg"]),
            str(fields.get("best_direction", "none")),
            "n/a"
            if fields.get("joint7_limit_ok") is None
            else str(bool(fields["joint7_limit_ok"])).lower(),
            "n/a"
            if fields.get("tcp_shift_after_joint7_m") is None
            else "%.4f" % float(fields["tcp_shift_after_joint7_m"]),
            "n/a"
            if fields.get("hand_shift_after_joint7_m") is None
            else "%.4f" % float(fields["hand_shift_after_joint7_m"]),
            str(fields.get("result", "FAIL")),
            str(fields.get("reason", "")),
        )
    )
