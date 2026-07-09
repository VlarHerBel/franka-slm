"""Microdescenso vertical desde pregrasp para sugar_box demo_scene_02 (sin ROS)."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

SUGAR_BOX_MICRO_DESCEND_SCENE_ID = "demo_scene_02"
SUGAR_BOX_MICRO_DESCEND_STATIC_CANDIDATES_M = (0.035, 0.040, 0.045, 0.050, 0.055)
SUGAR_BOX_MICRO_DESCEND_DEPTH_CANDIDATES_M = (0.010, 0.014, 0.018, 0.022)
SUGAR_BOX_MICRO_DESCEND_CENTERING_XY_TOL_M = 0.004
SUGAR_BOX_MICRO_DESCEND_MIN_DEPTH_BELOW_TOP_M = 0.010
SUGAR_BOX_MICRO_DESCEND_MAX_DEPTH_BELOW_TOP_M = 0.018
SUGAR_BOX_MICRO_DESCEND_DEFAULT_CARTESIAN_FRACTION_THRESHOLD = 0.90

# Backends disponibles para el microdescenso final desde corrected_pregrasp_js.
# cartesian_path se mantiene solo como diagnóstico inicial; ik_stepwise es el
# backend efectivo cuando compute_cartesian_path no rinde.
SUGAR_BOX_MICRO_DESCEND_BACKENDS = (
    "cartesian_path",
    "ik_stepwise",
    "direct_joint_endpoint",
)
SUGAR_BOX_MICRO_DESCEND_DEFAULT_BACKEND = "ik_stepwise"

# Parámetros del backend ik_stepwise.
SUGAR_BOX_MICRO_DESCEND_IK_STEP_DZ_M = 0.005
SUGAR_BOX_MICRO_DESCEND_IK_MIN_STEP_DZ_M = 0.0025
SUGAR_BOX_MICRO_DESCEND_IK_TCP_XY_TO_START_TOL_M = 0.003
SUGAR_BOX_MICRO_DESCEND_IK_TCP_XY_TO_TARGET_TOL_M = 0.004
SUGAR_BOX_MICRO_DESCEND_IK_TCP_Z_TOL_M = 0.004
SUGAR_BOX_MICRO_DESCEND_IK_ORIENTATION_TOL_DEG = 3.0


def normalize_sugar_box_micro_descend_backend(value: Any) -> str:
    """Normaliza el backend solicitado; valor inválido -> ik_stepwise."""
    v = str(value or "").strip().lower()
    if v in SUGAR_BOX_MICRO_DESCEND_BACKENDS:
        return v
    return SUGAR_BOX_MICRO_DESCEND_DEFAULT_BACKEND


def build_sugar_box_micro_descend_ik_step_targets(
    requested_micro_descend_m: float,
    *,
    step_dz_m: float = SUGAR_BOX_MICRO_DESCEND_IK_STEP_DZ_M,
) -> Tuple[float, ...]:
    """Profundidades acumuladas (ΔZ) crecientes desde step hasta requested (incluido)."""
    total = float(requested_micro_descend_m)
    if total <= 1e-6:
        return ()
    step = max(float(step_dz_m), 1e-4)
    out: List[float] = []
    acc = step
    while acc + 1e-9 < total:
        out.append(round(acc, 6))
        acc += step
    out.append(round(total, 6))
    return tuple(out)


def evaluate_sugar_box_micro_descend_ik_step_fk(
    *,
    fk_tcp: Tuple[float, float, float],
    start_tcp_xy: Tuple[float, float],
    target_center_xy: Tuple[float, float],
    target_tcp_z: float,
    orientation_error_deg: float,
    table_top_z: float,
    min_table_clearance_m: float,
    xy_to_start_tol_m: float = SUGAR_BOX_MICRO_DESCEND_IK_TCP_XY_TO_START_TOL_M,
    xy_to_target_tol_m: float = SUGAR_BOX_MICRO_DESCEND_IK_TCP_XY_TO_TARGET_TOL_M,
    z_tol_m: float = SUGAR_BOX_MICRO_DESCEND_IK_TCP_Z_TOL_M,
    orientation_tol_deg: float = SUGAR_BOX_MICRO_DESCEND_IK_ORIENTATION_TOL_DEG,
) -> Tuple[bool, Dict[str, float], str]:
    """Valida un paso IK/FK del backend ik_stepwise."""
    err_xy_start = float(
        math.hypot(
            float(fk_tcp[0]) - float(start_tcp_xy[0]),
            float(fk_tcp[1]) - float(start_tcp_xy[1]),
        )
    )
    err_xy_target = float(
        math.hypot(
            float(fk_tcp[0]) - float(target_center_xy[0]),
            float(fk_tcp[1]) - float(target_center_xy[1]),
        )
    )
    err_z = abs(float(fk_tcp[2]) - float(target_tcp_z))
    table_ok = bool(
        float(fk_tcp[2]) + 1e-9
        >= float(table_top_z) + float(min_table_clearance_m)
    )
    errors = {
        "tcp_error_xy_to_start": err_xy_start,
        "tcp_error_xy_to_target_center": err_xy_target,
        "tcp_error_z": err_z,
        "hand_orientation_error_deg": float(orientation_error_deg),
    }
    reject = ""
    if err_xy_start > float(xy_to_start_tol_m) + 1e-9:
        reject = "ik_step_xy_drift_from_start"
    elif err_xy_target > float(xy_to_target_tol_m) + 1e-9:
        reject = "ik_step_xy_off_target_center"
    elif err_z > float(z_tol_m) + 1e-9:
        reject = "ik_step_z_error"
    elif float(orientation_error_deg) > float(orientation_tol_deg) + 1e-9:
        reject = "ik_step_orientation_error"
    elif not table_ok:
        reject = "ik_step_table_clearance_fail"
    return (reject == ""), errors, reject


def evaluate_sugar_box_micro_descend_ik_success(
    *,
    depth_below_top: Optional[float],
    centering_xy: float,
    final_tcp_z: float,
    table_top_z: float,
    min_table_clearance_m: float,
    all_steps_ok: bool,
    target_collision_present: bool,
    min_depth_below_top_m: float = SUGAR_BOX_MICRO_DESCEND_MIN_DEPTH_BELOW_TOP_M,
    centering_xy_tol_m: float = SUGAR_BOX_MICRO_DESCEND_CENTERING_XY_TOL_M,
) -> Tuple[bool, str]:
    """Criterio de éxito final del microdescenso ik_stepwise."""
    if target_collision_present:
        return False, "target_collision_present"
    if not all_steps_ok:
        return False, "ik_step_fail"
    if depth_below_top is not None and (
        float(depth_below_top) + 1e-9 < float(min_depth_below_top_m)
    ):
        return False, "micro_descend_depth_below_top_fail"
    if float(centering_xy) > float(centering_xy_tol_m) + 1e-9:
        return False, "micro_descend_centering_fail"
    if float(final_tcp_z) + 1e-9 < float(table_top_z) + float(min_table_clearance_m):
        return False, "micro_descend_table_clearance_fail"
    return True, ""


def format_sugar_box_micro_descend_execute_log(fields: Dict[str, Any]) -> str:
    return (
        "[SUGAR_BOX_MICRO_DESCEND_EXECUTE]\n"
        "backend=%s\n"
        "steps=%s\n"
        "selected_micro_descend_m=%s\n"
        "final_tcp_z=%s\n"
        "result=%s"
        % (
            str(fields.get("backend", SUGAR_BOX_MICRO_DESCEND_DEFAULT_BACKEND)),
            "n/a" if fields.get("steps") is None else str(int(fields["steps"])),
            "n/a"
            if fields.get("selected_micro_descend_m") is None
            else "%.4f" % float(fields["selected_micro_descend_m"]),
            "n/a"
            if fields.get("final_tcp_z") is None
            else "%.4f" % float(fields["final_tcp_z"]),
            str(fields.get("result", "FAIL")),
        )
    )


def sugar_box_micro_descend_eligible(
    *,
    label: str,
    scene_id: str,
    enabled: bool = True,
) -> bool:
    return (
        bool(enabled)
        and str(label).strip().lower() == "sugar_box"
        and str(scene_id).strip() == SUGAR_BOX_MICRO_DESCEND_SCENE_ID
    )


def ordered_sugar_box_micro_descend_candidates_m(
    candidates_m: Sequence[float],
) -> Tuple[float, ...]:
    """Prueba primero el descenso más profundo (mayor ΔZ)."""
    uniq = sorted({float(v) for v in candidates_m if float(v) > 1e-6}, reverse=True)
    return tuple(uniq)


def build_sugar_box_dynamic_micro_descend_candidates_m(
    *,
    pregrasp_tcp_z: float,
    top_z: Optional[float],
    depth_candidates_m: Sequence[float] = SUGAR_BOX_MICRO_DESCEND_DEPTH_CANDIDATES_M,
    static_fallback_m: Sequence[float] = SUGAR_BOX_MICRO_DESCEND_STATIC_CANDIDATES_M,
) -> Tuple[float, ...]:
    """
    micro_descend_m = pregrasp_tcp_z - (top_z - depth_below_top_candidate).
    Si no hay top_z, usa fallback estático [0.035..0.055].
    """
    if top_z is None:
        return ordered_sugar_box_micro_descend_candidates_m(static_fallback_m)
    dynamic: List[float] = []
    for depth in depth_candidates_m:
        desired_final_tcp_z = float(top_z) - float(depth)
        micro_m = float(pregrasp_tcp_z) - desired_final_tcp_z
        if micro_m > 1e-4:
            dynamic.append(float(micro_m))
    if not dynamic:
        return ordered_sugar_box_micro_descend_candidates_m(static_fallback_m)
    return ordered_sugar_box_micro_descend_candidates_m(dynamic)


def evaluate_sugar_box_micro_descend_fk(
    *,
    fk_tcp: Tuple[float, float, float],
    target_center_xy: Tuple[float, float],
    top_z: Optional[float],
    table_top_z: float,
    min_table_clearance_m: float,
    centering_xy_tol_m: float = SUGAR_BOX_MICRO_DESCEND_CENTERING_XY_TOL_M,
    min_depth_below_top_m: float = SUGAR_BOX_MICRO_DESCEND_MIN_DEPTH_BELOW_TOP_M,
    max_depth_below_top_m: float = SUGAR_BOX_MICRO_DESCEND_MAX_DEPTH_BELOW_TOP_M,
) -> Tuple[bool, float, Optional[float], bool, str]:
    """Valida FK endpoint del microdescenso."""
    tcp_err_xy = float(
        math.hypot(
            float(fk_tcp[0]) - float(target_center_xy[0]),
            float(fk_tcp[1]) - float(target_center_xy[1]),
        )
    )
    centering_ok = bool(float(tcp_err_xy) + 1e-9 <= float(centering_xy_tol_m))
    table_ok = bool(
        float(fk_tcp[2]) + 1e-9
        >= float(table_top_z) + float(min_table_clearance_m)
    )
    depth_below_top = None
    depth_ok = True
    if top_z is not None:
        depth_below_top = float(top_z) - float(fk_tcp[2])
        depth_ok = bool(
            float(depth_below_top) + 1e-9 >= float(min_depth_below_top_m)
        )
    ok = bool(centering_ok and table_ok and depth_ok)
    reject = ""
    if not ok:
        if not centering_ok:
            reject = "micro_descend_centering_fail"
        elif not table_ok:
            reject = "micro_descend_table_clearance_fail"
        elif not depth_ok:
            reject = "micro_descend_depth_below_top_fail"
    in_preferred_depth = bool(
        depth_below_top is not None
        and float(depth_below_top) + 1e-9 <= float(max_depth_below_top_m)
    )
    return bool(ok), float(tcp_err_xy), depth_below_top, bool(in_preferred_depth), reject


def format_sugar_box_micro_descend_policy_log(fields: Dict[str, Any]) -> str:
    return (
        "[SUGAR_BOX_MICRO_DESCEND_POLICY]\n"
        "candidate_id=%s\n"
        "trigger_reason=%s\n"
        "enabled=%s\n"
        "gated_out_reason=%s\n"
        "backend=%s\n"
        "pregrasp_tcp_z=%s\n"
        "top_z=%s\n"
        "candidates_m=%s\n"
        "requested_micro_descend_m=%s\n"
        "selected_micro_descend_m=%s\n"
        "final_tcp_z=%s\n"
        "depth_below_top=%s\n"
        "centering_xy=%s\n"
        "cartesian_fraction=%s\n"
        "cartesian_fallback_used=%s\n"
        "ik_ok=%s\n"
        "ik_steps_ok=%s\n"
        "fk_ok=%s\n"
        "full_descend_m=%s\n"
        "result=%s\n"
        "reject_reason=%s"
        % (
            str(fields.get("candidate_id", "n/a")),
            str(fields.get("trigger_reason", "n/a")),
            str(bool(fields.get("enabled", False))).lower(),
            str(fields.get("gated_out_reason") or ""),
            str(fields.get("backend", SUGAR_BOX_MICRO_DESCEND_DEFAULT_BACKEND)),
            "n/a"
            if fields.get("pregrasp_tcp_z") is None
            else "%.4f" % float(fields["pregrasp_tcp_z"]),
            "n/a" if fields.get("top_z") is None else "%.4f" % float(fields["top_z"]),
            str(fields.get("candidates_m") or "n/a"),
            "n/a"
            if fields.get("requested_micro_descend_m") is None
            else "%.4f" % float(fields["requested_micro_descend_m"]),
            "n/a"
            if fields.get("selected_micro_descend_m") is None
            else "%.4f" % float(fields["selected_micro_descend_m"]),
            "n/a"
            if fields.get("final_tcp_z") is None
            else "%.4f" % float(fields["final_tcp_z"]),
            "n/a"
            if fields.get("depth_below_top") is None
            else "%.4f" % float(fields["depth_below_top"]),
            "n/a"
            if fields.get("centering_xy") is None
            else "%.4f" % float(fields["centering_xy"]),
            "n/a"
            if fields.get("cartesian_fraction") is None
            else "%.5f" % float(fields["cartesian_fraction"]),
            "n/a"
            if fields.get("cartesian_fallback_used") is None
            else str(bool(fields["cartesian_fallback_used"])).lower(),
            "n/a" if fields.get("ik_ok") is None else str(bool(fields["ik_ok"])).lower(),
            "n/a"
            if fields.get("ik_steps_ok") is None
            else str(int(fields["ik_steps_ok"])),
            "n/a" if fields.get("fk_ok") is None else str(bool(fields["fk_ok"])).lower(),
            "n/a"
            if fields.get("full_descend_m") is None
            else "%.4f" % float(fields.get("full_descend_m")),
            str(fields.get("result", "FAIL")),
            str(fields.get("reject_reason", "")),
        )
    )
