"""Evaluación candidato pick completo sugar_box demo_scene_02 (sin ROS)."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple


def is_sugar_box_full_pick_candidate_complete(record: Dict[str, Any]) -> bool:
    joint7_ok = bool(
        record.get("joint7_correction_virtual_ok")
        or record.get("joint7_virtual_ok")
    )
    descend_ok = bool(record.get("descend_full_or_guarded_ok"))
    if bool(record.get("micro_descend_ok")):
        descend_ok = True
    return (
        bool(record.get("pregrasp_plan_ok"))
        and joint7_ok
        and bool(record.get("gap_after_joint7_ok"))
        and bool(record.get("centering_after_joint7_ok"))
        and bool(record.get("descend_first_step_ok") or record.get("micro_descend_ok"))
        and descend_ok
        and str(record.get("result", "")).upper() == "OK"
    )


def compute_sugar_box_full_pick_score(record: Dict[str, Any]) -> Tuple[float, ...]:
    """Mayor score = mejor: profundidad, fraction descend, gap/centering bajos."""
    frac = record.get("descend_cartesian_fraction")
    if frac is None:
        if bool(record.get("micro_descend_ok")):
            frac = record.get("descend_cartesian_fraction") or 1.0
        else:
            frac = 1.0 if record.get("guarded_ik_full_ok") else 0.0
    depth = record.get("depth_from_top_m", 0.0)
    if bool(record.get("micro_descend_ok")):
        micro_m = record.get("selected_micro_descend_m")
        if micro_m is not None:
            depth = float(micro_m)
    return (
        float(depth if depth is not None else 0.0),
        float(frac if frac is not None else 0.0),
        -float(record.get("gap_after_joint7_deg", 999.0)),
        -float(record.get("centering_after_joint7_xy", 999.0)),
    )


def select_sugar_box_full_pick_candidate(
    records: Sequence[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    complete = [r for r in records if is_sugar_box_full_pick_candidate_complete(r)]
    if not complete:
        return None
    return max(complete, key=compute_sugar_box_full_pick_score)


def format_sugar_box_full_pick_candidate_eval_log(fields: Dict[str, Any]) -> str:
    frac = fields.get("descend_cartesian_fraction")
    frac_s = "n/a" if frac is None else "%.5f" % float(frac)
    return (
        "[SUGAR_BOX_FULL_PICK_CANDIDATE_EVAL]\n"
        "candidate_id=%s\n"
        "yaw_variant=%s\n"
        "pregrasp_tcp_z=%.4f\n"
        "grasp_tcp_z=%.4f\n"
        "pregrasp_plan_ok=%s\n"
        "joint7_virtual_ok=%s\n"
        "gap_after_joint7_deg=%s\n"
        "centering_after_joint7_xy=%s\n"
        "descend_cartesian_fraction=%s\n"
        "guarded_ik_first_step_ok=%s\n"
        "guarded_ik_full_ok=%s\n"
        "micro_descend_backend=%s\n"
        "micro_descend_ok=%s\n"
        "selected_micro_descend_m=%s\n"
        "score=%s\n"
        "result=%s\n"
        "reject_reason=%s"
        % (
            str(fields.get("candidate_id", "n/a")),
            str(fields.get("yaw_variant", "n/a")),
            float(fields.get("pregrasp_tcp_z", 0.0)),
            float(fields.get("grasp_tcp_z", 0.0)),
            str(bool(fields.get("pregrasp_plan_ok", False))).lower(),
            str(bool(fields.get("joint7_virtual_ok", False))).lower(),
            "n/a"
            if fields.get("gap_after_joint7_deg") is None
            else "%.2f" % float(fields.get("gap_after_joint7_deg")),
            "n/a"
            if fields.get("centering_after_joint7_xy") is None
            else "%.4f" % float(fields.get("centering_after_joint7_xy")),
            frac_s,
            str(bool(fields.get("guarded_ik_first_step_ok", False))).lower(),
            str(bool(fields.get("guarded_ik_full_ok", False))).lower(),
            str(fields.get("micro_descend_backend") or "n/a"),
            str(bool(fields.get("micro_descend_ok", False))).lower(),
            "n/a"
            if fields.get("selected_micro_descend_m") is None
            else "%.4f" % float(fields.get("selected_micro_descend_m")),
            str(fields.get("score", "n/a")),
            str(fields.get("result", "FAIL")),
            str(fields.get("reject_reason", "")),
        )
    )


def format_sugar_box_full_pick_candidate_selected_log(fields: Dict[str, Any]) -> str:
    return (
        "[SUGAR_BOX_FULL_PICK_CANDIDATE_SELECTED]\n"
        "candidate_id=%s\n"
        "yaw_variant=%s\n"
        "pregrasp_tcp=(%.3f, %.3f, %.3f)\n"
        "grasp_tcp=(%.3f, %.3f, %.3f)\n"
        "pregrasp_plan_cached=%s\n"
        "joint7_corrected_js_cached=%s\n"
        "descend_plan_cached=%s\n"
        "micro_descend_backend=%s\n"
        "result=OK"
        % (
            str(fields.get("candidate_id", "n/a")),
            str(fields.get("yaw_variant", "n/a")),
            float(fields.get("pregrasp_tcp_x", 0.0)),
            float(fields.get("pregrasp_tcp_y", 0.0)),
            float(fields.get("pregrasp_tcp_z", 0.0)),
            float(fields.get("grasp_tcp_x", 0.0)),
            float(fields.get("grasp_tcp_y", 0.0)),
            float(fields.get("grasp_tcp_z", 0.0)),
            str(bool(fields.get("pregrasp_plan_cached", False))).lower(),
            str(bool(fields.get("joint7_corrected_js_cached", False))).lower(),
            str(fields.get("descend_plan_cached", "false")),
            str(fields.get("micro_descend_backend") or "n/a"),
        )
    )


def format_full_pick_route_no_candidate_log(
    reason: str = "no_full_pick_candidate",
) -> str:
    return (
        "[FULL_PICK_ROUTE_PREVALIDATE_RESULT]\n"
        "result=FAIL_BEFORE_MOTION\n"
        "reason=%s\n"
        "action=ABORT_IN_HOME"
        % str(reason)
    )


def centering_xy_error_m(
    tcp_xy: Tuple[float, float],
    target_xy: Tuple[float, float],
) -> float:
    return float(
        math.hypot(
            float(tcp_xy[0]) - float(target_xy[0]),
            float(tcp_xy[1]) - float(target_xy[1]),
        )
    )


SUGAR_BOX_OFFLINE_CENTERING_TCP_ACCEPT_M = 0.003


def evaluate_sugar_box_offline_centering(
    *,
    fk_tcp_xy: Tuple[float, float],
    target_center_xy: Tuple[float, float],
    finger_midpoint_xy: Optional[Tuple[float, float]] = None,
    z_err_m: float = 0.0,
    max_z_err_m: float = 0.004,
) -> Tuple[bool, float, Optional[float], str]:
    """Acepta por FK TCP (<=3mm); finger midpoint solo diagnóstico."""
    tcp_err = centering_xy_error_m(fk_tcp_xy, target_center_xy)
    finger_err = (
        centering_xy_error_m(finger_midpoint_xy, target_center_xy)
        if finger_midpoint_xy is not None
        else None
    )
    ok = (
        float(tcp_err) + 1e-9 <= float(SUGAR_BOX_OFFLINE_CENTERING_TCP_ACCEPT_M)
        and float(z_err_m) + 1e-9 <= float(max_z_err_m)
    )
    source = "tcp"
    return bool(ok), float(tcp_err), finger_err, source


def format_sugar_box_offline_centering_diag_log(fields: Dict[str, Any]) -> str:
    finger_err = fields.get("finger_midpoint_error_xy")
    return (
        "[SUGAR_BOX_OFFLINE_CENTERING_DIAG]\n"
        "candidate_id=%s\n"
        "fk_tcp_xy=(%.4f, %.4f)\n"
        "target_center_xy=(%.4f, %.4f)\n"
        "finger_midpoint_xy=%s\n"
        "tcp_error_xy=%.4f\n"
        "finger_midpoint_error_xy=%s\n"
        "centering_source=%s\n"
        "result=%s"
        % (
            str(fields.get("candidate_id", "n/a")),
            float(fields.get("fk_tcp_x", 0.0)),
            float(fields.get("fk_tcp_y", 0.0)),
            float(fields.get("target_center_x", 0.0)),
            float(fields.get("target_center_y", 0.0)),
            "n/a"
            if fields.get("finger_midpoint_xy") is None
            else "(%.4f, %.4f)"
            % (
                float(fields["finger_midpoint_xy"][0]),
                float(fields["finger_midpoint_xy"][1]),
            ),
            float(fields.get("tcp_error_xy", 0.0)),
            "n/a" if finger_err is None else "%.4f" % float(finger_err),
            str(fields.get("centering_source", "tcp")),
            str(fields.get("result", "FAIL")),
        )
    )


def format_sugar_box_endpoint_orientation_contract_log(fields: Dict[str, Any]) -> str:
    expected = fields.get("expected_hand_to_tcp_base")
    actual = fields.get("actual_hand_to_tcp_base")
    desired_q = fields.get("desired_hand_quat") or (0.0, 0.0, 0.0, 1.0)
    fk_q = fields.get("fk_hand_quat")
    return (
        "[SUGAR_BOX_ENDPOINT_ORIENTATION_CONTRACT]\n"
        "candidate_idx=%s\n"
        "yaw_variant=%s\n"
        "desired_hand_quat=(%.4f, %.4f, %.4f, %.4f)\n"
        "fk_hand_quat=%s\n"
        "hand_orientation_error_deg=%s\n"
        "expected_hand_to_tcp_base=%s\n"
        "actual_hand_to_tcp_base=%s\n"
        "tcp_error_xy=%s\n"
        "result=%s\n"
        "reject_reason=%s"
        % (
            str(fields.get("candidate_idx", "n/a")),
            str(fields.get("yaw_variant", "n/a")),
            float(desired_q[0]),
            float(desired_q[1]),
            float(desired_q[2]),
            float(desired_q[3]),
            "n/a"
            if fk_q is None
            else "(%.4f, %.4f, %.4f, %.4f)"
            % (float(fk_q[0]), float(fk_q[1]), float(fk_q[2]), float(fk_q[3])),
            "n/a"
            if fields.get("hand_orientation_error_deg") is None
            else "%.4f" % float(fields.get("hand_orientation_error_deg")),
            "n/a"
            if expected is None
            else "(%.4f, %.4f, %.4f)"
            % (float(expected[0]), float(expected[1]), float(expected[2])),
            "n/a"
            if actual is None
            else "(%.4f, %.4f, %.4f)"
            % (float(actual[0]), float(actual[1]), float(actual[2])),
            "n/a"
            if fields.get("tcp_error_xy") is None
            else "%.4f" % float(fields.get("tcp_error_xy")),
            str(fields.get("result", "FAIL")),
            str(fields.get("reject_reason", "")),
        )
    )


def format_sugar_box_full_pick_descend_scene_log(fields: Dict[str, Any]) -> str:
    obstacles = fields.get("obstacles") or []
    obs_s = ",".join(str(o) for o in obstacles) if obstacles else "none"
    return (
        "[SUGAR_BOX_FULL_PICK_DESCEND_SCENE]\n"
        "target_collision_present=%s\n"
        "obstacles=[%s]\n"
        "result=%s"
        % (
            str(bool(fields.get("target_collision_present", True))).lower(),
            obs_s,
            str(fields.get("result", "FAIL")),
        )
    )


def format_sugar_box_descend_scene_contract_violation_log() -> str:
    return (
        "[SUGAR_BOX_DESCEND_SCENE_CONTRACT_VIOLATION]\n"
        "reason=target_collision_present_during_descend_prevalidate\n"
        "action=ABORT_INTERNAL"
    )
