"""Contrato de ruta pick sugar_box: direct_pregrasp vs object_safe_above (sin ROS)."""

from __future__ import annotations

import math
from typing import Any, Dict, Optional

# Yaw de objeto ~alineado con ejes de mesa: joint7 no aporta (plan-before basta).
_SUGAR_BOX_CARDINAL_YAW_TOL_RAD = math.radians(15.0)


def sugar_box_resolve_selected_route(
    candidate: Optional[Dict[str, Any]],
    preplanned_route: Optional[Dict[str, Any]] = None,
) -> str:
    cand = candidate or {}
    pre = preplanned_route or {}
    explicit = str(cand.get("_sugar_box_selected_route") or "").strip()
    if explicit:
        return explicit
    route = str(pre.get("route") or "").strip()
    if route:
        return route
    src = str(pre.get("source") or "")
    if "direct_pregrasp" in src:
        return "direct_pregrasp"
    if "safe_entry" in src or "object_safe_above" in src:
        return "object_safe_above"
    return ""


def sugar_box_skip_object_safe_above_stage(
    candidate: Optional[Dict[str, Any]],
    preplanned_route: Optional[Dict[str, Any]] = None,
) -> bool:
    """True cuando sugar_box debe ejecutar direct_pregrasp, no object_safe_above."""
    cand = candidate or {}
    if str(cand.get("label", "")).strip().lower() != "sugar_box":
        return False
    route = sugar_box_resolve_selected_route(cand, preplanned_route)
    if route == "direct_pregrasp":
        return True
    pre = preplanned_route or {}
    if str(pre.get("route") or "") == "direct_pregrasp":
        return True
    entry = cand.get("selected_entry_target") or pre.get("selected_entry_target")
    if entry is not None and str(entry) != "object_safe_above_tcp":
        return True
    return False


def _object_yaw_rad_from_candidate(
    candidate: Optional[Dict[str, Any]],
) -> Optional[float]:
    cand = candidate or {}
    for key in (
        "active_yaw_rad",
        "object_yaw_rad",
        "yaw_rad",
        "_base_commanded_tcp_yaw_rad",
        "_final_tcp_yaw_rad",
    ):
        raw = cand.get(key)
        if raw is None:
            continue
        try:
            return float(raw)
        except (TypeError, ValueError):
            continue
    return None


def sugar_box_object_yaw_cardinal_aligned(
    candidate: Optional[Dict[str, Any]],
    *,
    tolerance_rad: float = _SUGAR_BOX_CARDINAL_YAW_TOL_RAD,
) -> bool:
    """True si el yaw del objeto está ~0°/90° respecto a la mesa (eje corto obvio)."""
    yaw = _object_yaw_rad_from_candidate(candidate)
    if yaw is None:
        return True
    y = abs((float(yaw) + math.pi) % (2.0 * math.pi) - math.pi)
    for cardinal in (0.0, math.pi / 2.0):
        if abs(y - cardinal) <= float(tolerance_rad):
            return True
    return False


def sugar_box_direct_pregrasp_yaw_locked(candidate: Optional[Dict[str, Any]]) -> bool:
    """True si el yaw/quat de ejecución está fijado por plan-before (direct pregrasp)."""
    cand = candidate or {}
    if str(cand.get("label", "")).strip().lower() != "sugar_box":
        return False
    if bool(cand.get("_direct_pregrasp_yaw_execution_locked")):
        return True
    validated = cand.get("_plan_before_motion_validated") or {}
    if not bool(validated.get("ok")):
        return False
    if str(validated.get("mode", "")) != "direct_pregrasp":
        return False
    return bool(
        cand.get("_full_pick_route_prevalidated")
        or cand.get("_post_prelude_pregrasp_locked")
        or cand.get("_sugar_box_direct_pregrasp_cached_traj") is not None
    )


def sugar_box_skip_joint7_axis_in_place_correction(
    candidate: Optional[Dict[str, Any]],
) -> bool:
    """
    sugar_box direct_pregrasp validado: no rotar solo joint7 para alinear gap
    cuando el yaw del objeto ya es cardinal (0°/90°). Con yaw oblicuo (p. ej.
  two_boxes_02) hace falta joint7 para alinear el gap con el eje corto.
    """
    if not sugar_box_direct_pregrasp_yaw_locked(candidate):
        return False
    return sugar_box_object_yaw_cardinal_aligned(candidate)


def sugar_box_skip_gripper_gap_alignment(
    candidate: Optional[Dict[str, Any]],
) -> bool:
    """Omite verify/corrección gap en runtime solo con yaw cardinal validado."""
    if not sugar_box_direct_pregrasp_yaw_locked(candidate):
        return False
    return sugar_box_object_yaw_cardinal_aligned(candidate)


# Muñeca perpendicular a la mesa: dot(-Z_tcp, world_Z) >= este umbral.
SUGAR_BOX_STRICT_TOP_DOWN_DOT_DEFAULT = 0.98
# Error máximo hand_quat actual vs quat validado en plan-before (grados).
SUGAR_BOX_STRICT_HAND_ORIENTATION_TOL_DEG_DEFAULT = 10.0
# Umbral plan-before: FK del endpoint de trayectoria suele ser ~0.94–0.95 aunque
# el plan sea válido; el ajuste estricto ocurre en runtime (0.98).
SUGAR_BOX_PLAN_TOP_DOWN_DOT_MIN = 0.88


def sugar_box_skip_gripper_centering_verify(
    candidate: Optional[Dict[str, Any]],
) -> bool:
    """
    Omite verify/corrección XY de centering en runtime para direct_pregrasp validado.
    El plan-before ya fijó TCP/quat; finger_midpoint vs modelo suele dar falsos FAIL.
    """
    return sugar_box_direct_pregrasp_yaw_locked(candidate)


def sugar_box_use_gap_aligned_descend_orientation_lock(
    candidate: Optional[Dict[str, Any]],
) -> bool:
    """
    Tras corrección joint7 el hand_quat difiere del plan-before validado.
    En descend basta top-down + gap alineado (como el lock genérico post-yaw).
    """
    cand = candidate or {}
    return bool(
        sugar_box_direct_pregrasp_yaw_locked(cand)
        and cand.get("_axis_correction_applied")
    )


def sugar_box_descend_wrist_ok_with_gap_lock(
    *,
    top_down_dot: float,
    gap_angle_error_deg: float,
    strict_top_down_dot: float,
    gap_target_angle_deg: float,
) -> bool:
    return bool(
        float(top_down_dot) + 1e-9 >= float(strict_top_down_dot)
        and float(gap_angle_error_deg) <= float(gap_target_angle_deg) + 1e-6
    )


def sugar_box_lift_gap_prerequisite_required(
    candidate: Optional[Dict[str, Any]],
) -> bool:
    """Micro-lift: si False, no exige gap alignment runtime (yaw/quat ya validados)."""
    return not sugar_box_skip_gripper_gap_alignment(candidate)


def sugar_box_lift_centering_prerequisite_required(
    candidate: Optional[Dict[str, Any]],
) -> bool:
    """Micro-lift: si False, no exige centering verify runtime."""
    return not sugar_box_skip_gripper_centering_verify(candidate)


def sugar_box_pregrasp_plan_ik_fallback_eligible(
    candidate: Optional[Dict[str, Any]],
) -> bool:
    """
    Plan-before workspace→pregrasp: si MoveIt plan falla pero IK es válido,
    aceptar la variante (runtime usa moveit_pose → joint_configuration_fallback).
    """
    if not isinstance(candidate, dict):
        return False
    return str(candidate.get("label", "")).strip().lower() == "sugar_box"


def sugar_box_direct_pregrasp_route_contract_violation(
    candidate: Optional[Dict[str, Any]],
    preplanned_route: Optional[Dict[str, Any]] = None,
    *,
    attempted_stage: str,
) -> bool:
    """True si la ejecución intenta object_safe_above con ruta direct_pregrasp."""
    if str(attempted_stage) != "object_safe_above_to_pregrasp":
        return False
    return sugar_box_resolve_selected_route(candidate, preplanned_route) == "direct_pregrasp"
