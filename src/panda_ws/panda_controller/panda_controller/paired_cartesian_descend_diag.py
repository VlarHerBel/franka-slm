"""Diagnóstico de paridad y barrido cartesiano para validador paired/grid."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple


def fmt_pose3(pos: Optional[Sequence[float]]) -> str:
    if pos is None or len(pos) < 3:
        return "n/a"
    return "(%.4f, %.4f, %.4f)" % (float(pos[0]), float(pos[1]), float(pos[2]))


def build_cartesian_sweep_z_targets(
    *,
    start_tcp_z: float,
    target_tcp_z: float,
    failed_tcp_z: Optional[float] = None,
    extra_z: Optional[Sequence[float]] = None,
) -> List[float]:
    lo = min(float(start_tcp_z), float(target_tcp_z))
    hi = max(float(start_tcp_z), float(target_tcp_z))
    out: List[float] = []
    seen: set = set()

    def _add(z: float) -> None:
        zf = round(float(z), 4)
        if zf < lo - 1e-6 or zf > hi + 1e-6:
            return
        key = round(zf, 3)
        if key in seen:
            return
        seen.add(key)
        out.append(zf)

    if failed_tcp_z is not None:
        _add(float(failed_tcp_z))
    if extra_z:
        for z in extra_z:
            _add(float(z))
    _add(float(target_tcp_z))
    mid = lo + 0.5 * (hi - lo)
    for z in (hi, mid, lo + 0.75 * (hi - lo), lo + 0.25 * (hi - lo), lo):
        _add(z)
    out.sort(reverse=True)
    return out


def compare_descend_profiles(
    prevalidator: Dict[str, Any],
    runtime: Dict[str, Any],
) -> Tuple[str, List[str]]:
    diffs: List[str] = []
    keys = (
        "link",
        "eef_step",
        "jump_threshold",
        "avoid_collisions",
        "waypoint_count",
        "use_grasp_tcp",
        "planning_frame",
        "target_collision_policy",
    )
    for key in keys:
        pv = prevalidator.get(key)
        rv = runtime.get(key)
        if pv != rv:
            diffs.append("%s pre=%s runtime=%s" % (key, pv, rv))
    pre_hand = prevalidator.get("target_hand_pose")
    run_hand = runtime.get("target_hand_pose")
    if pre_hand is not None and run_hand is not None:
        for i, label in enumerate(("x", "y", "z")):
            if abs(float(pre_hand[i]) - float(run_hand[i])) > 1e-4:
                diffs.append(
                    "target_hand_%s pre=%.4f runtime=%.4f"
                    % (label, float(pre_hand[i]), float(run_hand[i]))
                )
                break
    pre_quat = prevalidator.get("target_hand_quat")
    run_quat = runtime.get("target_hand_quat")
    if pre_quat is not None and run_quat is not None:
        q_delta = max(
            abs(float(pre_quat[i]) - float(run_quat[i])) for i in range(min(4, len(pre_quat), len(run_quat)))
        )
        if q_delta > 1e-4:
            diffs.append("target_hand_quat_delta=%.5f" % q_delta)
    if prevalidator.get("start_state_source") != runtime.get("start_state_source"):
        diffs.append(
            "start_state_source pre=%s runtime=%s"
            % (prevalidator.get("start_state_source"), runtime.get("start_state_source"))
        )
    result = "SAME" if not diffs else "DIFF"
    return result, diffs


def format_descend_prevalidate_runtime_parity_log(fields: Dict[str, Any]) -> str:
    diffs = fields.get("diffs") or []
    diffs_repr = "; ".join(str(d) for d in diffs) if diffs else "none"
    return (
        "[DESCEND_PREVALIDATE_RUNTIME_PARITY]\n"
        "candidate_idx=%s\n"
        "prevalidator_link=%s\n"
        "runtime_link=%s\n"
        "prevalidator_eef_step=%s\n"
        "runtime_eef_step=%s\n"
        "prevalidator_jump_threshold=%s\n"
        "runtime_jump_threshold=%s\n"
        "prevalidator_avoid_collisions=%s\n"
        "runtime_avoid_collisions=%s\n"
        "prevalidator_waypoint_count=%s\n"
        "runtime_waypoint_count=%s\n"
        "prevalidator_target_hand_pose=%s\n"
        "runtime_target_hand_pose=%s\n"
        "prevalidator_start_state_source=%s\n"
        "runtime_start_state_source=%s\n"
        "prevalidator_target_collision_policy=%s\n"
        "runtime_target_collision_policy=%s\n"
        "result=%s\n"
        "diffs=%s"
        % (
            fields.get("candidate_idx", "n/a"),
            fields.get("prevalidator_link", "n/a"),
            fields.get("runtime_link", "n/a"),
            fields.get("prevalidator_eef_step", "n/a"),
            fields.get("runtime_eef_step", "n/a"),
            fields.get("prevalidator_jump_threshold", "n/a"),
            fields.get("runtime_jump_threshold", "n/a"),
            fields.get("prevalidator_avoid_collisions", "n/a"),
            fields.get("runtime_avoid_collisions", "n/a"),
            fields.get("prevalidator_waypoint_count", "n/a"),
            fields.get("runtime_waypoint_count", "n/a"),
            fmt_pose3(fields.get("prevalidator_target_hand_pose")),
            fmt_pose3(fields.get("runtime_target_hand_pose")),
            fields.get("prevalidator_start_state_source", "n/a"),
            fields.get("runtime_start_state_source", "n/a"),
            fields.get("prevalidator_target_collision_policy", "n/a"),
            fields.get("runtime_target_collision_policy", "n/a"),
            str(fields.get("result", "DIFF")),
            diffs_repr,
        )
    )


def format_cartesian_descend_sweep_diag_log(fields: Dict[str, Any]) -> str:
    return (
        "[CARTESIAN_DESCEND_SWEEP_DIAG]\n"
        "candidate_idx=%s\n"
        "start_tcp_z=%s\n"
        "target_tcp_z=%s\n"
        "test_target_z=%s\n"
        "avoid_collisions=%s\n"
        "jump_threshold=%s\n"
        "target_collision=%s\n"
        "obstacles=%s\n"
        "fraction=%s\n"
        "endpoint_ik_ok=%s\n"
        "result=%s\n"
        "reason=%s"
        % (
            fields.get("candidate_idx", "n/a"),
            fields.get("start_tcp_z", "n/a"),
            fields.get("target_tcp_z", "n/a"),
            fields.get("test_target_z", "n/a"),
            str(fields.get("avoid_collisions", "n/a")).lower()
            if isinstance(fields.get("avoid_collisions"), bool)
            else fields.get("avoid_collisions", "n/a"),
            fields.get("jump_threshold", "n/a"),
            fields.get("target_collision", "n/a"),
            fields.get("obstacles", "n/a"),
            fields.get("fraction", "n/a"),
            str(bool(fields.get("endpoint_ik_ok"))).lower()
            if "endpoint_ik_ok" in fields
            else "n/a",
            str(fields.get("result", "FAIL")),
            str(fields.get("reason", "")),
        )
    )


def format_endpoint_ik_aligned_diag_log(fields: Dict[str, Any]) -> str:
    return (
        "[ENDPOINT_IK_ALIGNED_DIAG]\n"
        "candidate_idx=%s\n"
        "grasp_tcp=%s\n"
        "grasp_hand=%s\n"
        "quat=%s\n"
        "seed=aligned_pregrasp_js\n"
        "ik_link=%s\n"
        "result=%s\n"
        "moveit_error_code=%s"
        % (
            fields.get("candidate_idx", "n/a"),
            fmt_pose3(fields.get("grasp_tcp")),
            fmt_pose3(fields.get("grasp_hand")),
            fields.get("quat_repr", "n/a"),
            fields.get("ik_link", "n/a"),
            str(fields.get("result", "FAIL")),
            fields.get("moveit_error_code", "n/a"),
        )
    )


def interpolate_failed_tcp_z(
    pre_tcp_z: float,
    target_tcp_z: float,
    fraction: Optional[float],
) -> Optional[float]:
    if fraction is None:
        return None
    return float(pre_tcp_z) + float(fraction) * (float(target_tcp_z) - float(pre_tcp_z))
