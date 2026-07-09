"""Prevalidación de ruta chips_can con object_high_stage (demo_scene_02)."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

from panda_controller.chips_can_final_descend_depth import (
    MIN_CHIPS_CAN_HIGH_LOW_PREGRASP_DELTA_M,
)

OK_CHIPS_HIGH_ROUTE_PREVALIDATED = "OK_CHIPS_HIGH_ROUTE_PREVALIDATED"
OK_CHIPS_HIGH_ROUTE_PENDING_FINAL_DESCEND_VALIDATE = (
    "OK_CHIPS_HIGH_ROUTE_PENDING_FINAL_DESCEND_VALIDATE"
)
CHIPS_CAN_HIGH_ROUTE_PREFLIGHT_SOURCE = "chips_can_high_route_prevalidated"
CHIPS_CAN_HIGH_ROUTE_PENDING_PREFLIGHT_SOURCE = (
    "chips_can_high_route_pending_final_descend"
)
CHIPS_CAN_HIGH_TO_LOW_ACTUAL_SOURCE = "chips_can_high_to_low_actual_execution"


def chips_can_high_route_yaw_passes(
    *,
    object_high_plan_ok: bool,
    object_high_to_low_fraction: float,
    low_to_grasp_fraction: float,
    fraction_threshold: float,
) -> bool:
    threshold = float(fraction_threshold)
    return bool(
        object_high_plan_ok
        and float(object_high_to_low_fraction) + 1e-6 >= threshold
        and float(low_to_grasp_fraction) + 1e-6 >= threshold
    )


def select_chips_can_high_route_yaw_variant(
    variants: Sequence[Dict[str, Any]],
    *,
    fraction_threshold: float,
) -> Optional[Dict[str, Any]]:
    """Elige la variante yaw con ruta completa OK y menor joint_dist."""
    passing: List[Dict[str, Any]] = []
    for item in variants:
        if chips_can_high_route_yaw_passes(
            object_high_plan_ok=bool(item.get("object_high_plan_ok")),
            object_high_to_low_fraction=float(
                item.get("object_high_to_low_fraction", 0.0)
            ),
            low_to_grasp_fraction=float(item.get("low_to_grasp_fraction", 0.0)),
            fraction_threshold=fraction_threshold,
        ):
            passing.append(item)
    if not passing:
        return None
    return min(passing, key=lambda item: float(item.get("joint_dist", math.inf)))


def summarize_chips_can_high_route_yaw_variants(
    variants: Sequence[Dict[str, Any]],
) -> Dict[str, float]:
    best_hl = 0.0
    best_lg = 0.0
    for item in variants:
        best_hl = max(
            best_hl, float(item.get("object_high_to_low_fraction", 0.0))
        )
        best_lg = max(best_lg, float(item.get("low_to_grasp_fraction", 0.0)))
    return {
        "best_object_high_to_low_fraction": best_hl,
        "best_low_to_grasp_fraction": best_lg,
    }


def _probe_entry_tcp_z(item: Dict[str, Any]) -> float:
    entry = item.get("entry_tcp")
    if isinstance(entry, (list, tuple)) and len(entry) >= 3:
        return float(entry[2])
    return float(item.get("entry_tcp_z", 0.0))


def select_chips_can_high_route_pending_descend_variant(
    variants: Sequence[Dict[str, Any]],
    *,
    fraction_threshold: float,
    low_pregrasp_tcp_z: float,
    min_high_low_delta_m: float = MIN_CHIPS_CAN_HIGH_LOW_PREGRASP_DELTA_M,
) -> Optional[Dict[str, Any]]:
    """Elige yaw con object_high real (>low) + high→low OK."""
    passing: List[Dict[str, Any]] = []
    threshold = float(fraction_threshold)
    low_z = float(low_pregrasp_tcp_z)
    min_delta = float(min_high_low_delta_m)
    for item in variants:
        if not bool(item.get("object_high_plan_ok")):
            continue
        if float(item.get("object_high_to_low_fraction", 0.0)) + 1e-6 < threshold:
            continue
        entry_z = _probe_entry_tcp_z(item)
        if entry_z + 1e-6 < low_z + min_delta:
            continue
        passing.append(item)
    if not passing:
        return None
    return min(
        passing,
        key=lambda item: (
            -_probe_entry_tcp_z(item),
            float(item.get("joint_dist", math.inf)),
        ),
    )


def chips_can_high_route_pending_preflight_accepts(
    *,
    plan_before_result: str,
    preflight_source: str,
    object_high_plan_ok: bool,
    selected_entry_target: str,
    chips_can_high_route_pending_final_descend: bool,
) -> bool:
    return bool(
        str(plan_before_result or "").strip()
        == OK_CHIPS_HIGH_ROUTE_PENDING_FINAL_DESCEND_VALIDATE
        and str(preflight_source or "").strip()
        == CHIPS_CAN_HIGH_ROUTE_PENDING_PREFLIGHT_SOURCE
        and object_high_plan_ok
        and str(selected_entry_target or "").strip() == "object_high_pregrasp"
        and chips_can_high_route_pending_final_descend
    )


def chips_can_high_route_preflight_accepts(
    *,
    plan_before_result: str,
    cartesian_prevalidated: bool,
    preflight_source: str,
    object_high_plan_ok: bool,
    selected_entry_target: str,
    chips_can_high_route_prevalidated: bool,
) -> bool:
    return bool(
        str(plan_before_result or "").strip() == OK_CHIPS_HIGH_ROUTE_PREVALIDATED
        and cartesian_prevalidated
        and str(preflight_source or "").strip()
        == CHIPS_CAN_HIGH_ROUTE_PREFLIGHT_SOURCE
        and object_high_plan_ok
        and str(selected_entry_target or "").strip() == "object_high_pregrasp"
        and chips_can_high_route_prevalidated
    )


def format_chips_can_high_route_segment_log(fields: Dict[str, Any]) -> str:
    frac = fields.get("cartesian_fraction")
    frac_s = "n/a" if frac is None else "%.5f" % float(frac)
    return (
        "[CHIPS_CAN_HIGH_ROUTE_SEGMENT]\n"
        "segment=%s\n"
        "target_collision_present=%s\n"
        "cartesian_fraction=%s\n"
        "result=%s\n"
        "reason=%s"
        % (
            fields.get("segment", ""),
            fields.get("target_collision_present", "n/a"),
            frac_s,
            fields.get("result", "FAIL"),
            fields.get("reason", ""),
        )
    )


def format_chips_can_low_pregrasp_state_refresh_log(fields: Dict[str, Any]) -> str:
    prev_tcp = fields.get("previous_tcp")
    current_tcp = fields.get("current_tcp")

    def _fmt_tcp(tcp: Any) -> str:
        if isinstance(tcp, (list, tuple)) and len(tcp) >= 3:
            return "(%.3f, %.3f, %.3f)" % (
                float(tcp[0]),
                float(tcp[1]),
                float(tcp[2]),
            )
        return "n/a"

    return (
        "[CHIPS_CAN_LOW_PREGRASP_STATE_REFRESH]\n"
        "stage=after_gripper_open_before_final_descend\n"
        "previous_tcp=%s\n"
        "current_tcp=%s\n"
        "tcp_delta_m=%s\n"
        "previous_js_distance=%s\n"
        "result=%s"
        % (
            _fmt_tcp(prev_tcp),
            _fmt_tcp(current_tcp),
            "n/a"
            if fields.get("tcp_delta_m") is None
            else "%.4f" % float(fields.get("tcp_delta_m")),
            "n/a"
            if fields.get("previous_js_distance") is None
            else "%.4f" % float(fields.get("previous_js_distance")),
            fields.get("result", "FAIL"),
        )
    )


def evaluate_chips_can_high_to_low_pregrasp_verify(
    *,
    tcp_error_m: Optional[float],
    tcp_threshold_m: float,
    js_distance: Optional[float],
    js_threshold: float,
    disturbance_ok: bool,
    centering_ok: bool,
) -> Tuple[bool, str]:
    tcp_ok = tcp_error_m is not None and float(tcp_error_m) + 1e-9 < float(
        tcp_threshold_m
    )
    js_ok = js_distance is not None and float(js_distance) + 1e-9 < float(js_threshold)
    if not bool(disturbance_ok) or not bool(centering_ok):
        return False, "chips_can_pre_descend_checks_failed"
    if tcp_ok:
        if js_ok:
            return True, "chips_can_high_to_low_actual_state_ok"
        return True, "chips_can_high_to_low_tcp_pose_ok_js_refreshed"
    return False, "chips_can_high_to_low_actual_state_mismatch"


def format_chips_can_low_pregrasp_state_cache_log(fields: Dict[str, Any]) -> str:
    current_tcp = fields.get("current_tcp")
    expected_tcp = fields.get("expected_tcp")
    if isinstance(current_tcp, (list, tuple)) and len(current_tcp) >= 3:
        current_tcp_s = "(%.3f, %.3f, %.3f)" % (
            float(current_tcp[0]),
            float(current_tcp[1]),
            float(current_tcp[2]),
        )
    else:
        current_tcp_s = "n/a"
    if isinstance(expected_tcp, (list, tuple)) and len(expected_tcp) >= 3:
        expected_tcp_s = "(%.3f, %.3f, %.3f)" % (
            float(expected_tcp[0]),
            float(expected_tcp[1]),
            float(expected_tcp[2]),
        )
    else:
        expected_tcp_s = "n/a"
    js_s = fields.get("current_js", "n/a")
    return (
        "[CHIPS_CAN_LOW_PREGRASP_STATE_CACHE]\n"
        "current_js=%s\n"
        "current_tcp=%s\n"
        "expected_tcp=%s\n"
        "tcp_error_m=%s\n"
        "result=%s"
        % (
            js_s,
            current_tcp_s,
            expected_tcp_s,
            "n/a"
            if fields.get("tcp_error_m") is None
            else "%.4f" % float(fields.get("tcp_error_m")),
            fields.get("result", "FAIL"),
        )
    )


def format_chips_can_high_route_yaw_variant_log(fields: Dict[str, Any]) -> str:
    return (
        "[CHIPS_CAN_HIGH_ROUTE_YAW_VARIANT]\n"
        "yaw_deg=%.2f\n"
        "object_high_plan_ok=%s\n"
        "object_high_to_low_fraction=%.5f\n"
        "low_to_grasp_fraction=%.5f\n"
        "full_route_ok=%s\n"
        "joint_dist=%.4f\n"
        "selected=%s\n"
        "reject_reason=%s"
        % (
            float(fields.get("yaw_deg", 0.0)),
            str(bool(fields.get("object_high_plan_ok"))).lower(),
            float(fields.get("object_high_to_low_fraction", 0.0)),
            float(fields.get("low_to_grasp_fraction", 0.0)),
            str(bool(fields.get("full_route_ok"))).lower(),
            float(fields.get("joint_dist", 0.0)),
            str(bool(fields.get("selected"))).lower(),
            fields.get("reject_reason", ""),
        )
    )


def format_chips_can_high_route_variant_selected_log(fields: Dict[str, Any]) -> str:
    return (
        "[CHIPS_CAN_HIGH_ROUTE_VARIANT_SELECTED]\n"
        "yaw_deg=%.2f\n"
        "low_to_grasp_fraction=%.5f\n"
        "result=%s"
        % (
            float(fields.get("yaw_deg", 0.0)),
            float(fields.get("low_to_grasp_fraction", 0.0)),
            fields.get("result", "OK"),
        )
    )


def format_chips_can_high_route_yaw_exhausted_log(fields: Dict[str, Any]) -> str:
    return (
        "[CHIPS_CAN_HIGH_ROUTE_YAW_EXHAUSTED]\n"
        "best_object_high_to_low_fraction=%.5f\n"
        "best_low_to_grasp_fraction=%.5f\n"
        "result=FAIL\n"
        "reason=%s"
        % (
            float(fields.get("best_object_high_to_low_fraction", 0.0)),
            float(fields.get("best_low_to_grasp_fraction", 0.0)),
            fields.get("reason", "no_yaw_variant_full_route_ok"),
        )
    )


def format_chips_can_high_route_prevalidate_result_log(fields: Dict[str, Any]) -> str:
    return (
        "[CHIPS_CAN_HIGH_ROUTE_PREVALIDATE_RESULT]\n"
        "result=%s\n"
        "cartesian_prevalidated=%s\n"
        "preflight_source=%s\n"
        "object_high_to_low_cartesian_ok=%s\n"
        "low_to_grasp_cartesian_ok=%s\n"
        "failed_segment=%s\n"
        "reason=%s"
        % (
            fields.get("result", "FAIL_BEFORE_MOTION"),
            fields.get("cartesian_prevalidated", "false"),
            fields.get("preflight_source", ""),
            fields.get("object_high_to_low_cartesian_ok", "false"),
            fields.get("low_to_grasp_cartesian_ok", "false"),
            fields.get("failed_segment", ""),
            fields.get("reason", ""),
        )
    )
