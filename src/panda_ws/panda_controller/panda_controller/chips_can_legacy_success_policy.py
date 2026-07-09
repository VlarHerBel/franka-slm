"""Política legacy chips_can: entrada alta segura + pregrasp baja + descenso corto.

GOLDEN CONGELADO (demo_scene_02 + chips_can, validado pick_place completo):
  - chips_can_use_legacy_successful_pick_policy
  - object_high → legacy_low_pregrasp (+ borderline high→low)
  - pre-descend pose gate legacy, microsegment descend, inter-segment verify
  - cylinder contact / attach / lift / place
Referencia: config/demo_candidate_cache/demo_scene_02_chips_can_golden.yaml
No modificar este pipeline salvo detrás de flag explícito de experimentación.
"""

from __future__ import annotations

# Escena/objeto del run golden validado; usar como guardrail documental.
CHIPS_CAN_LEGACY_GOLDEN_SCENE_ID = "demo_scene_02"
CHIPS_CAN_LEGACY_GOLDEN_TARGET_LABEL = "chips_can"
CHIPS_CAN_LEGACY_GOLDEN_LAYOUT_VERSION = "v3_clear_table_transport"

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

OK_CHIPS_LEGACY_SUCCESS_POLICY = "OK_CHIPS_LEGACY_SUCCESS_POLICY"
OK_CHIPS_LEGACY_PENDING_ACTUAL_TF_DESCEND = "OK_CHIPS_LEGACY_PENDING_ACTUAL_TF_DESCEND"
CHIPS_CAN_LEGACY_SUCCESS_PREFLIGHT_SOURCE = "chips_can_legacy_success_policy"
CHIPS_CAN_LEGACY_PENDING_ACTUAL_TF_DESCEND_PREFLIGHT_SOURCE = (
    "chips_can_legacy_pending_actual_tf_descend"
)

# Grid de búsqueda (run exitoso histórico: pregrasp_height≈0.025, depth≈0.025–0.033).
CHIPS_CAN_LEGACY_PREGRASP_HEIGHT_ABOVE_TOP_M: Tuple[float, ...] = (
    0.025,
    0.035,
    0.045,
)

CHIPS_CAN_LEGACY_GRASP_DEPTH_FROM_TOP_M: Tuple[float, ...] = (
    0.020,
    0.025,
    0.030,
    0.035,
)

# Gate pre-descenso legacy: pregrasp baja ~0.025–0.045 m sobre top (no 0.100 m high_route).
CHIPS_CAN_LEGACY_PRE_DESCEND_MIN_CLEARANCE_ABOVE_TOP_M = 0.020
CHIPS_CAN_LEGACY_PRE_DESCEND_MAX_CLEARANCE_ABOVE_TOP_M = 0.055
CHIPS_CAN_LEGACY_PRE_DESCEND_ABORT_TOO_LOW_CLEARANCE_M = 0.015
CHIPS_CAN_LEGACY_PRE_DESCEND_ABORT_TOO_HIGH_CLEARANCE_M = 0.070

# Tolerancia TF high→low: aceptar TCP real ligeramente por encima del low planificado.
CHIPS_CAN_HIGH_TO_LOW_TF_TOLERANCE_MIN_M = 0.015
CHIPS_CAN_HIGH_TO_LOW_TF_TOLERANCE_MAX_M = 0.020

# Referencia del pick_place exitoso documentado en logs previos.
CHIPS_CAN_LEGACY_HISTORICAL_SUCCESS_REFERENCE: Dict[str, Any] = {
    "pregrasp_height_above_top_m": 0.025,
    "recommended_grasp_depth_from_top_m": 0.033,
    "contact_policy": "CYLINDER_CONTACT_POLICY contact_ok=true",
    "grasp_contact_strict": "PASS",
    "attach_result": "OK",
    "post_lift_verify": "OK",
    "place": "deterministic sequence completed successfully",
}


def chips_can_legacy_pregrasp_tcp_z(*, top_z_m: float, pregrasp_height_above_top_m: float) -> float:
    return float(top_z_m) + float(pregrasp_height_above_top_m)


def chips_can_legacy_grasp_tcp_z(*, top_z_m: float, depth_from_top_m: float) -> float:
    return float(top_z_m) - float(depth_from_top_m)


def chips_can_legacy_final_descend_m(
    *, pregrasp_height_above_top_m: float, depth_from_top_m: float
) -> float:
    return float(pregrasp_height_above_top_m) + float(depth_from_top_m)


def chips_can_legacy_pregrasp_height_allowed(pregrasp_height_above_top_m: float) -> bool:
    for allowed in CHIPS_CAN_LEGACY_PREGRASP_HEIGHT_ABOVE_TOP_M:
        if abs(float(pregrasp_height_above_top_m) - float(allowed)) < 1e-6:
            return True
    return False


def chips_can_legacy_pending_variant_passes(
    item: Dict[str, Any],
    *,
    fraction_threshold: float,
) -> bool:
    """Acepta variante legacy si high→legacy_low pasa; low→grasp es diagnóstico."""
    threshold = float(fraction_threshold)
    if not bool(item.get("object_high_plan_ok")):
        return False
    pre_h = float(item.get("pregrasp_height_above_top_m", -1.0))
    if not chips_can_legacy_pregrasp_height_allowed(pre_h):
        return False
    if float(item.get("object_high_to_low_fraction", 0.0)) + 1e-6 < threshold:
        return False
    return True


def chips_can_legacy_pre_descend_clearance_ok(
    *,
    actual_tcp_z: float,
    top_z_m: float,
) -> Tuple[bool, float, str]:
    clearance = float(actual_tcp_z) - float(top_z_m)
    if clearance + 1e-9 < CHIPS_CAN_LEGACY_PRE_DESCEND_ABORT_TOO_LOW_CLEARANCE_M:
        return False, clearance, "legacy_pre_descend_too_low"
    if clearance > CHIPS_CAN_LEGACY_PRE_DESCEND_ABORT_TOO_HIGH_CLEARANCE_M + 1e-9:
        return False, clearance, "legacy_pre_descend_not_at_low_pregrasp"
    if (
        clearance + 1e-9 < CHIPS_CAN_LEGACY_PRE_DESCEND_MIN_CLEARANCE_ABOVE_TOP_M
        or clearance > CHIPS_CAN_LEGACY_PRE_DESCEND_MAX_CLEARANCE_ABOVE_TOP_M + 1e-9
    ):
        return False, clearance, "legacy_pre_descend_clearance_out_of_range"
    return True, clearance, "legacy_pre_descend_clearance_ok"


def chips_can_legacy_high_to_low_borderline_ok(
    *,
    requested_low_tcp_z: float,
    actual_tcp_z_after: float,
    top_z_m: float,
) -> Tuple[bool, float, float, str]:
    """Acepta high→low cuando el TCP real queda por encima del low planificado pero en banda legacy."""
    z_error = float(actual_tcp_z_after) - float(requested_low_tcp_z)
    if z_error <= 1e-9:
        return False, z_error, 0.0, "not_above_requested_low"
    clearance = float(actual_tcp_z_after) - float(top_z_m)
    clearance_ok, _, clearance_reason = chips_can_legacy_pre_descend_clearance_ok(
        actual_tcp_z=float(actual_tcp_z_after),
        top_z_m=float(top_z_m),
    )
    if not clearance_ok:
        return False, z_error, clearance, clearance_reason
    return (
        True,
        z_error,
        clearance,
        "tf_z_above_request_but_inside_legacy_low_range",
    )


def evaluate_chips_can_high_to_low_tf_tolerance(
    *,
    requested_tcp_z: float,
    actual_tcp_z_after: float,
    max_tolerance_m: float = CHIPS_CAN_HIGH_TO_LOW_TF_TOLERANCE_MAX_M,
) -> Tuple[bool, float, str]:
    """Acepta high→low borderline cuando el TCP real queda por encima del target pero dentro de tolerancia."""
    z_error = float(actual_tcp_z_after) - float(requested_tcp_z)
    if z_error < -1e-9:
        return False, z_error, "actual_below_requested_contact_risk"
    if z_error <= 1e-9:
        return False, z_error, "at_or_below_requested_not_borderline_high"
    if z_error > float(max_tolerance_m) + 1e-9:
        return False, z_error, "actual_above_requested_plus_tolerance"
    return True, z_error, "tf_z_above_request_within_tolerance"


def chips_can_legacy_low_pregrasp_state_refresh_ok(
    *,
    current_tcp: Sequence[float],
    expected_tcp: Sequence[float],
    tcp_error_threshold_m: float,
    accept_borderline_low_pregrasp: bool,
    top_z_m: Optional[float],
) -> Tuple[bool, Optional[float]]:
    tcp_err = math.sqrt(
        sum(
            (float(current_tcp[i]) - float(expected_tcp[i])) ** 2
            for i in range(3)
        )
    )
    if float(tcp_err) + 1e-9 < float(tcp_error_threshold_m):
        return True, tcp_err
    if not accept_borderline_low_pregrasp:
        return False, tcp_err
    borderline_ok, _, _ = evaluate_chips_can_high_to_low_tf_tolerance(
        requested_tcp_z=float(expected_tcp[2]),
        actual_tcp_z_after=float(current_tcp[2]),
    )
    if borderline_ok:
        return True, tcp_err
    if top_z_m is not None:
        legacy_ok, _, _, _ = chips_can_legacy_high_to_low_borderline_ok(
            requested_low_tcp_z=float(expected_tcp[2]),
            actual_tcp_z_after=float(current_tcp[2]),
            top_z_m=float(top_z_m),
        )
        return bool(legacy_ok), tcp_err
    return False, tcp_err


def evaluate_chips_can_legacy_pre_descend_pose_gate(
    *,
    actual_tcp_z: float,
    top_z_m: float,
    legacy_low_pregrasp_tcp_z: Optional[float],
    centering_ok: bool,
    gripper_open_ok: bool,
    disturbance_ok: bool,
    target_collision_removed_ok: bool,
    descend_route_prepared_ok: bool,
    tcp_error_tolerance_m: float,
) -> Tuple[bool, Dict[str, Any]]:
    clearance_ok, clearance, clearance_reason = chips_can_legacy_pre_descend_clearance_ok(
        actual_tcp_z=float(actual_tcp_z),
        top_z_m=float(top_z_m),
    )
    legacy_low_pregrasp_ok = bool(clearance_ok)
    tcp_near_low = True
    if legacy_low_pregrasp_tcp_z is not None:
        tcp_err = abs(float(actual_tcp_z) - float(legacy_low_pregrasp_tcp_z))
        tcp_near_low = tcp_err + 1e-9 <= float(tcp_error_tolerance_m)
    ok = bool(
        legacy_low_pregrasp_ok
        and centering_ok
        and gripper_open_ok
        and disturbance_ok
        and target_collision_removed_ok
        and descend_route_prepared_ok
    )
    reason = clearance_reason
    if ok:
        reason = "legacy_pre_descend_gate_ok"
    elif clearance_ok and not tcp_near_low:
        reason = "legacy_pre_descend_tcp_not_near_low_pregrasp"
    elif not centering_ok:
        reason = "legacy_pre_descend_centering_failed"
    elif not gripper_open_ok:
        reason = "legacy_pre_descend_gripper_not_open"
    elif not disturbance_ok:
        reason = "legacy_pre_descend_disturbance_failed"
    elif not target_collision_removed_ok:
        reason = "legacy_pre_descend_target_collision_still_present"
    elif not descend_route_prepared_ok:
        reason = "legacy_pre_descend_route_not_prepared"
    return ok, {
        "actual_tcp_z": float(actual_tcp_z),
        "top_z_m": float(top_z_m),
        "clearance_above_top": float(clearance),
        "legacy_low_pregrasp_tcp_z": legacy_low_pregrasp_tcp_z,
        "legacy_low_pregrasp_ok": bool(legacy_low_pregrasp_ok),
        "clearance_ok": bool(clearance_ok),
        "centering_ok": bool(centering_ok),
        "gripper_open_ok": bool(gripper_open_ok),
        "disturbance_ok": bool(disturbance_ok),
        "target_collision_removed_ok": bool(target_collision_removed_ok),
        "descend_route_prepared_ok": bool(descend_route_prepared_ok),
        "reason": reason,
    }


def format_chips_can_legacy_pre_descend_pose_gate_log(fields: Dict[str, Any]) -> str:
    return (
        "[CHIPS_CAN_PRE_DESCEND_POSE_GATE]\n"
        "mode=legacy_successful_pick\n"
        "actual_tcp_z=%.4f\n"
        "top_z=%.4f\n"
        "clearance_above_top=%.4f\n"
        "allowed_clearance_range=[%.3f, %.3f]\n"
        "legacy_low_pregrasp_tcp_z=%s\n"
        "legacy_low_pregrasp_ok=%s\n"
        "centering_ok=%s\n"
        "gripper_open_ok=%s\n"
        "disturbance_ok=%s\n"
        "target_collision_removed_ok=%s\n"
        "descend_route_prepared_ok=%s\n"
        "result=%s\n"
        "reason=%s"
        % (
            float(fields.get("actual_tcp_z", 0.0)),
            float(fields.get("top_z_m", 0.0)),
            float(fields.get("clearance_above_top", 0.0)),
            CHIPS_CAN_LEGACY_PRE_DESCEND_MIN_CLEARANCE_ABOVE_TOP_M,
            CHIPS_CAN_LEGACY_PRE_DESCEND_MAX_CLEARANCE_ABOVE_TOP_M,
            "n/a"
            if fields.get("legacy_low_pregrasp_tcp_z") is None
            else "%.4f" % float(fields.get("legacy_low_pregrasp_tcp_z")),
            str(bool(fields.get("legacy_low_pregrasp_ok", False))).lower(),
            str(bool(fields.get("centering_ok", False))).lower(),
            str(bool(fields.get("gripper_open_ok", False))).lower(),
            str(bool(fields.get("disturbance_ok", False))).lower(),
            str(bool(fields.get("target_collision_removed_ok", False))).lower(),
            str(bool(fields.get("descend_route_prepared_ok", False))).lower(),
            "OK" if bool(fields.get("ok", False)) else "FAIL",
            str(fields.get("reason", "n/a")),
        )
    )


def format_chips_can_legacy_success_policy_start_log(fields: Dict[str, Any]) -> str:
    return (
        "[CHIPS_CAN_LEGACY_SUCCESS_POLICY_START]\n"
        "top_z=%.4f\n"
        "legacy_pregrasp_height_above_top=%.4f\n"
        "legacy_pregrasp_tcp_z=%.4f\n"
        "target_depth_from_top=%.4f\n"
        "target_grasp_tcp_z=%.4f\n"
        "historical_reference_pregrasp_height=%.4f\n"
        "historical_reference_depth=%.4f"
        % (
            float(fields.get("top_z", 0.0)),
            float(fields.get("legacy_pregrasp_height_above_top", 0.0)),
            float(fields.get("legacy_pregrasp_tcp_z", 0.0)),
            float(fields.get("target_depth_from_top", 0.0)),
            float(fields.get("target_grasp_tcp_z", 0.0)),
            float(
                CHIPS_CAN_LEGACY_HISTORICAL_SUCCESS_REFERENCE.get(
                    "pregrasp_height_above_top_m", 0.025
                )
            ),
            float(
                CHIPS_CAN_LEGACY_HISTORICAL_SUCCESS_REFERENCE.get(
                    "recommended_grasp_depth_from_top_m", 0.033
                )
            ),
        )
    )


def format_chips_can_legacy_low_pregrasp_variant_log(fields: Dict[str, Any]) -> str:
    actual_z = fields.get("actual_tcp_z")
    actual_z_s = "n/a" if actual_z is None else "%.4f" % float(actual_z)
    depth_below = fields.get("actual_depth_below_top")
    depth_below_s = "n/a" if depth_below is None else "%.5f" % float(depth_below)
    frac = fields.get("low_to_grasp_fraction")
    frac_s = "n/a" if frac is None else "%.5f" % float(frac)
    return (
        "[CHIPS_CAN_LEGACY_LOW_PREGRASP_VARIANT]\n"
        "pregrasp_height_above_top_m=%.4f\n"
        "depth_from_top_m=%.4f\n"
        "yaw_deg=%.2f\n"
        "plan_to_pregrasp_ok=%s\n"
        "object_high_to_low_fraction=%s\n"
        "low_to_grasp_fraction=%s\n"
        "low_to_grasp_fraction_used_as_diagnostic=%s\n"
        "actual_tcp_z=%s\n"
        "actual_depth_below_top=%s\n"
        "centering_ok=%s\n"
        "disturbance_ok=%s\n"
        "result=%s\n"
        "reject_reason=%s"
        % (
            float(fields.get("pregrasp_height_above_top_m", 0.0)),
            float(fields.get("depth_from_top_m", 0.0)),
            float(fields.get("yaw_deg", 0.0)),
            str(bool(fields.get("plan_to_pregrasp_ok", False))).lower(),
            "n/a"
            if fields.get("object_high_to_low_fraction") is None
            else "%.5f" % float(fields.get("object_high_to_low_fraction")),
            frac_s,
            str(bool(fields.get("low_to_grasp_fraction_used_as_diagnostic", True))).lower(),
            actual_z_s,
            depth_below_s,
            str(bool(fields.get("centering_ok", False))).lower(),
            str(bool(fields.get("disturbance_ok", False))).lower(),
            fields.get("result", "FAIL"),
            fields.get("reject_reason", ""),
        )
    )


def format_chips_can_legacy_success_policy_selected_log(fields: Dict[str, Any]) -> str:
    contract = str(
        fields.get("contract", OK_CHIPS_LEGACY_PENDING_ACTUAL_TF_DESCEND)
    )
    hl_frac = fields.get("object_high_to_low_fraction")
    hl_frac_s = "n/a" if hl_frac is None else "%.5f" % float(hl_frac)
    return (
        "[CHIPS_CAN_LEGACY_SUCCESS_POLICY_SELECTED]\n"
        "pregrasp_height_above_top_m=%.4f\n"
        "depth_from_top_m=%.4f\n"
        "yaw_deg=%.2f\n"
        "object_high_to_low_fraction=%s\n"
        "cached_js_available=%s\n"
        "low_to_grasp_fraction=%.5f\n"
        "low_to_grasp_fraction_used_as_diagnostic=%s\n"
        "final_descend_m=%.4f\n"
        "contract=%s\n"
        "result=%s"
        % (
            float(fields.get("pregrasp_height_above_top_m", 0.0)),
            float(fields.get("depth_from_top_m", 0.0)),
            float(fields.get("yaw_deg", 0.0)),
            hl_frac_s,
            str(bool(fields.get("cached_js_available", False))).lower(),
            float(fields.get("low_to_grasp_fraction", 0.0)),
            str(bool(fields.get("low_to_grasp_fraction_used_as_diagnostic", True))).lower(),
            float(fields.get("final_descend_m", 0.0)),
            contract,
            fields.get("result", "OK"),
        )
    )


def format_chips_can_legacy_high_to_low_prevalidate_reuse_log(
    fields: Dict[str, Any],
) -> str:
    return (
        "[CHIPS_CAN_LEGACY_HIGH_TO_LOW_PREVALIDATE_REUSE]\n"
        "cached_fraction=%s\n"
        "cached_yaw=%.4f\n"
        "cached_js_available=%s\n"
        "revalidate_fraction=%s\n"
        "result=%s"
        % (
            "n/a"
            if fields.get("cached_fraction") is None
            else "%.5f" % float(fields.get("cached_fraction")),
            float(fields.get("cached_yaw", 0.0)),
            str(bool(fields.get("cached_js_available", False))).lower(),
            "n/a"
            if fields.get("revalidate_fraction") is None
            else "%.5f" % float(fields.get("revalidate_fraction")),
            fields.get("result", "OK"),
        )
    )


def format_chips_can_legacy_high_to_low_revalidate_mismatch_log(
    fields: Dict[str, Any],
) -> str:
    return (
        "[CHIPS_CAN_LEGACY_HIGH_TO_LOW_REVALIDATE_MISMATCH]\n"
        "cached_fraction=%s\n"
        "revalidate_fraction=%s\n"
        "cached_high_js_available=%s\n"
        "cached_low_js_available=%s\n"
        "action=%s\n"
        "result=%s"
        % (
            "n/a"
            if fields.get("cached_fraction") is None
            else "%.5f" % float(fields.get("cached_fraction")),
            "n/a"
            if fields.get("revalidate_fraction") is None
            else "%.5f" % float(fields.get("revalidate_fraction")),
            str(bool(fields.get("cached_high_js_available", False))).lower(),
            str(bool(fields.get("cached_low_js_available", False))).lower(),
            str(fields.get("action", "use_cached_trajectory")),
            fields.get("result", "WARN"),
        )
    )


def chips_can_legacy_high_to_low_cache_reusable(
    *,
    cached_fraction: Optional[float],
    cached_high_js: Any,
    cached_low_js: Any,
    fraction_threshold: float,
) -> bool:
    if cached_fraction is None or cached_high_js is None or cached_low_js is None:
        return False
    return float(cached_fraction) + 1e-6 >= float(fraction_threshold)


def format_chips_can_legacy_policy_contract_log(fields: Dict[str, Any]) -> str:
    hl_frac = fields.get("object_high_to_low_fraction")
    lg_frac = fields.get("low_to_grasp_fraction")
    return (
        "[CHIPS_CAN_LEGACY_POLICY_CONTRACT]\n"
        "mode=legacy_successful_pick\n"
        "plan_to_pregrasp_ok=%s\n"
        "object_high_to_low_fraction=%s\n"
        "low_to_grasp_fraction=%s\n"
        "low_to_grasp_used_as_diagnostic=true\n"
        "actual_tf_descend_required=true\n"
        "result=%s"
        % (
            str(bool(fields.get("plan_to_pregrasp_ok", False))).lower(),
            "n/a" if hl_frac is None else "%.5f" % float(hl_frac),
            "n/a" if lg_frac is None else "%.5f" % float(lg_frac),
            fields.get("result", OK_CHIPS_LEGACY_PENDING_ACTUAL_TF_DESCEND),
        )
    )


def format_chips_can_high_to_low_tf_tolerance_log(fields: Dict[str, Any]) -> str:
    return (
        "[CHIPS_CAN_HIGH_TO_LOW_TF_TOLERANCE]\n"
        "requested_tcp_z=%.4f\n"
        "actual_tcp_z_after=%.4f\n"
        "z_error_m=%.4f\n"
        "max_tolerance_m=%.4f\n"
        "reason=%s\n"
        "result=%s"
        % (
            float(fields.get("requested_tcp_z", 0.0)),
            float(fields.get("actual_tcp_z_after", 0.0)),
            float(fields.get("z_error_m", 0.0)),
            float(
                fields.get(
                    "max_tolerance_m",
                    CHIPS_CAN_HIGH_TO_LOW_TF_TOLERANCE_MAX_M,
                )
            ),
            str(fields.get("reason", "n/a")),
            fields.get("result", "FAIL"),
        )
    )


def format_chips_can_low_pregrasp_refresh_from_actual_tf_log(
    fields: Dict[str, Any],
) -> str:
    current_tcp = fields.get("current_tcp")
    expected_tcp = fields.get("expected_tcp")

    def _fmt_tcp(tcp: Any) -> str:
        if isinstance(tcp, (list, tuple)) and len(tcp) >= 3:
            return "(%.3f, %.3f, %.3f)" % (
                float(tcp[0]),
                float(tcp[1]),
                float(tcp[2]),
            )
        return "n/a"

    return (
        "[CHIPS_CAN_LOW_PREGRASP_REFRESH_FROM_ACTUAL_TF]\n"
        "current_tcp=%s\n"
        "expected_tcp=%s\n"
        "tcp_error_m=%s\n"
        "current_js=%s\n"
        "borderline=%s\n"
        "result=%s"
        % (
            _fmt_tcp(current_tcp),
            _fmt_tcp(expected_tcp),
            "n/a"
            if fields.get("tcp_error_m") is None
            else "%.4f" % float(fields.get("tcp_error_m")),
            fields.get("current_js", "n/a"),
            str(bool(fields.get("borderline", False))).lower(),
            fields.get("result", "FAIL"),
        )
    )


def format_chips_can_legacy_high_to_low_accept_borderline_log(
    fields: Dict[str, Any],
) -> str:
    return (
        "[CHIPS_CAN_LEGACY_HIGH_TO_LOW_ACCEPT_BORDERLINE]\n"
        "requested_low_tcp_z=%.4f\n"
        "actual_tcp_z_after=%.4f\n"
        "z_error_m=%.4f\n"
        "top_z=%.4f\n"
        "actual_clearance_above_top=%.4f\n"
        "allowed_clearance_range=[%.3f, %.3f]\n"
        "reason=%s\n"
        "result=OK"
        % (
            float(fields.get("requested_low_tcp_z", 0.0)),
            float(fields.get("actual_tcp_z_after", 0.0)),
            float(fields.get("z_error_m", 0.0)),
            float(fields.get("top_z", 0.0)),
            float(fields.get("actual_clearance_above_top", 0.0)),
            CHIPS_CAN_LEGACY_PRE_DESCEND_MIN_CLEARANCE_ABOVE_TOP_M,
            CHIPS_CAN_LEGACY_PRE_DESCEND_MAX_CLEARANCE_ABOVE_TOP_M,
            str(fields.get("reason", "n/a")),
        )
    )


def format_chips_can_legacy_high_to_low_execute_log(fields: Dict[str, Any]) -> str:
    def _z(key: str) -> str:
        val = fields.get(key)
        if val is None:
            return "n/a"
        return "%.4f" % float(val)

    return (
        "[CHIPS_CAN_LEGACY_HIGH_TO_LOW_EXECUTE]\n"
        "stage=object_high_to_legacy_low_pregrasp\n"
        "high_tcp_z=%s\n"
        "low_tcp_z=%s\n"
        "expected_delta_z=%s\n"
        "plan_fraction=%s\n"
        "actual_tcp_z_before=%s\n"
        "actual_tcp_z_after=%s\n"
        "actual_tcp_error_m=%s\n"
        "result=%s"
        % (
            _z("high_tcp_z"),
            _z("low_tcp_z"),
            _z("expected_delta_z"),
            "n/a"
            if fields.get("plan_fraction") is None
            else "%.5f" % float(fields.get("plan_fraction")),
            _z("actual_tcp_z_before"),
            _z("actual_tcp_z_after"),
            "n/a"
            if fields.get("actual_tcp_error_m") is None
            else "%.4f" % float(fields.get("actual_tcp_error_m")),
            fields.get("result", "FAIL"),
        )
    )


def format_chips_can_legacy_target_collision_restore_log(fields: Dict[str, Any]) -> str:
    return (
        "[LEGACY_TARGET_COLLISION_RESTORE_AFTER_DIAGNOSTIC_PROBE]\n"
        "target_id=%s\n"
        "before_present=%s\n"
        "after_present=%s\n"
        "result=%s"
        % (
            fields.get("target_id", "n/a"),
            str(bool(fields.get("before_present", False))).lower(),
            str(bool(fields.get("after_present", False))).lower(),
            fields.get("result", "FAIL"),
        )
    )


def format_chips_can_legacy_target_collision_required_for_approach_log(
    fields: Dict[str, Any],
) -> str:
    return (
        "[LEGACY_TARGET_COLLISION_REQUIRED_FOR_APPROACH]\n"
        "target_collision_present=%s\n"
        "approach_guard_present=%s\n"
        "result=%s"
        % (
            str(bool(fields.get("target_collision_present", False))).lower(),
            str(bool(fields.get("approach_guard_present", False))).lower(),
            fields.get("result", "FAIL"),
        )
    )


def format_chips_can_legacy_low_pregrasp_state_refresh_log(
    fields: Dict[str, Any],
) -> str:
    current_tcp = fields.get("current_tcp")
    expected_tcp = fields.get("expected_tcp")

    def _fmt_tcp(tcp: Any) -> str:
        if isinstance(tcp, (list, tuple)) and len(tcp) >= 3:
            return "(%.3f, %.3f, %.3f)" % (
                float(tcp[0]),
                float(tcp[1]),
                float(tcp[2]),
            )
        return "n/a"

    return (
        "[CHIPS_CAN_LEGACY_LOW_PREGRASP_STATE_REFRESH]\n"
        "current_tcp=%s\n"
        "expected_tcp=%s\n"
        "tcp_error_m=%s\n"
        "current_js=%s\n"
        "result=%s"
        % (
            _fmt_tcp(current_tcp),
            _fmt_tcp(expected_tcp),
            "n/a"
            if fields.get("tcp_error_m") is None
            else "%.4f" % float(fields.get("tcp_error_m")),
            fields.get("current_js", "n/a"),
            fields.get("result", "FAIL"),
        )
    )


def format_chips_can_micro_descend_inter_segment_verify_log(
    fields: Dict[str, Any],
) -> str:
    return (
        "[CHIPS_CAN_MICRO_DESCEND_INTER_SEGMENT_VERIFY]\n"
        "step_idx=%d\n"
        "actual_tcp_z=%s\n"
        "previous_target_tcp_z=%s\n"
        "next_target_tcp_z=%s\n"
        "z_progress_ok=%s\n"
        "xy_drift_ok=%s\n"
        "centering_ok=%s\n"
        "gripper_open_ok=%s\n"
        "gap_ok=%s\n"
        "segment_tcp_ok=%s\n"
        "tcp_ok_mode=%s\n"
        "result=%s\n"
        "reason=%s"
        % (
            int(fields.get("step_idx", 0)),
            "n/a"
            if fields.get("actual_tcp_z") is None
            else "%.4f" % float(fields.get("actual_tcp_z")),
            "n/a"
            if fields.get("previous_target_tcp_z") is None
            else "%.4f" % float(fields.get("previous_target_tcp_z")),
            "n/a"
            if fields.get("next_target_tcp_z") is None
            else "%.4f" % float(fields.get("next_target_tcp_z")),
            str(bool(fields.get("z_progress_ok", False))).lower(),
            str(bool(fields.get("xy_drift_ok", False))).lower(),
            str(bool(fields.get("centering_ok", False))).lower(),
            str(bool(fields.get("gripper_open_ok", False))).lower(),
            str(bool(fields.get("gap_ok", False))).lower(),
            str(bool(fields.get("segment_tcp_ok", False))).lower(),
            str(fields.get("tcp_ok_mode", "segment_progress_not_pregrasp")),
            fields.get("result", "FAIL"),
            str(fields.get("reason", "n/a")),
        )
    )


def select_chips_can_legacy_success_policy_variant(
    variants: Sequence[Dict[str, Any]],
    *,
    fraction_threshold: float,
) -> Optional[Dict[str, Any]]:
    """Elige variante legacy pending actual-TF; low→grasp solo diagnóstico."""
    threshold = float(fraction_threshold)
    passing: List[Dict[str, Any]] = []
    for item in variants:
        if chips_can_legacy_pending_variant_passes(item, fraction_threshold=threshold):
            passing.append(item)
    if not passing:
        return None
    return min(
        passing,
        key=lambda item: (
            float(item.get("pregrasp_height_above_top_m", math.inf)),
            -float(item.get("depth_from_top_m", 0.0)),
            float(item.get("joint_dist", math.inf)),
        ),
    )
