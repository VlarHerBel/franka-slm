"""Ruta pick simplificada: workspace -> pregrasp -> descenso vertical (sin object_safe_above alto)."""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple

from panda_controller.paired_pregrasp_descend_validation import (
    PAIRED_PREGRASP_FK_ERROR_THRESHOLD_M,
    evaluate_paired_pregrasp_fk_contract,
)
from panda_controller.sugar_box_safe_entry import (
    sugar_box_multiobject_full_pick_prevalidate_required,
)

SIMPLE_DIRECT_PICK_ROUTE_LABELS = frozenset({"sugar_box", "mustard_bottle"})
SIMPLE_DIRECT_PREGRASP_START_FK_TCP_ERROR_THRESHOLD_M = (
    PAIRED_PREGRASP_FK_ERROR_THRESHOLD_M
)


def simple_direct_pick_route_eligible(label: str) -> bool:
    return str(label or "").strip().lower() in SIMPLE_DIRECT_PICK_ROUTE_LABELS


def simple_direct_pick_route_eligible_for_candidate(
    candidate: Optional[Dict[str, Any]],
) -> bool:
    if not isinstance(candidate, dict):
        return False
    return simple_direct_pick_route_eligible(str(candidate.get("label", "")))


def simple_direct_pick_route_prevalidate_required(
    candidate: Dict[str, Any],
    *,
    enable_pick_workspace_prelude: bool,
    plan_before_prelude_skip_workspace_prelude: bool,
) -> bool:
    """Usar prevalidación workspace->pregrasp directa en lugar de object_safe_above alto."""
    if sugar_box_multiobject_full_pick_prevalidate_required(candidate):
        return False
    if not simple_direct_pick_route_eligible_for_candidate(candidate):
        return False
    if not bool(enable_pick_workspace_prelude):
        return False
    if bool(plan_before_prelude_skip_workspace_prelude):
        return False
    return True


def simple_direct_mustard_bottle_route_active(
    candidate: Optional[Dict[str, Any]],
) -> bool:
    if not isinstance(candidate, dict):
        return False
    label = str(candidate.get("label", "")).strip().lower()
    return label == "mustard_bottle" and bool(candidate.get("_simple_direct_pick_route"))


def mustard_pregrasp_plan_ik_fallback_eligible(
    candidate: Optional[Dict[str, Any]],
    *,
    mustard_pregrasp_ik_joint_goal: bool = False,
) -> bool:
    """
    workspace→pregrasp: si MoveIt plan falla pero IK/FK es válido, aceptar variante.
    En chips_mustard_* la ejecución real usa IK joint goal tras pick_workspace_ready.
    """
    if not simple_direct_mustard_bottle_route_active(candidate):
        return False
    if bool(mustard_pregrasp_ik_joint_goal):
        return True
    scene_id = str((candidate or {}).get("scene_id") or "").strip().lower()
    return scene_id.startswith("chips_mustard")


def _fmt_xyz(pos: Optional[Tuple[float, float, float]]) -> str:
    if pos is None:
        return "n/a"
    return "(%.4f, %.4f, %.4f)" % (float(pos[0]), float(pos[1]), float(pos[2]))


def evaluate_simple_direct_pregrasp_start_fk(
    *,
    pregrasp_tcp_desired: Tuple[float, float, float],
    pre_hand_plan: Tuple[float, float, float],
    fk_tcp: Optional[Tuple[float, float, float]],
    fk_hand: Optional[Tuple[float, float, float]],
    error_threshold_m: float = SIMPLE_DIRECT_PREGRASP_START_FK_TCP_ERROR_THRESHOLD_M,
) -> Dict[str, Any]:
    fk_result = evaluate_paired_pregrasp_fk_contract(
        expected_tcp=pregrasp_tcp_desired,
        expected_hand=pre_hand_plan,
        actual_tcp=fk_tcp,
        actual_hand=fk_hand,
        error_threshold_m=float(error_threshold_m),
    )
    tcp_err = fk_result.get("tcp_error_m")
    hand_err = fk_result.get("hand_error_m")
    ok = bool(fk_result.get("ok"))
    reason = "fk_contract_ok" if ok else "start_tcp_or_hand_fk_out_of_tolerance"
    if not ok and fk_result.get("frame_mismatch"):
        reason = "paired_pregrasp_frame_mismatch"
    return {
        "ok": ok,
        "start_tcp_error_m": tcp_err,
        "start_hand_error_m": hand_err,
        "pregrasp_tcp_desired": pregrasp_tcp_desired,
        "fk_grasp_tcp": fk_tcp,
        "fk_panda_hand": fk_hand,
        "reason": reason,
    }


def format_simple_direct_pregrasp_start_fk_validate_log(
    *,
    label: str,
    fk_eval: Dict[str, Any],
    result: str,
) -> str:
    tcp_err = fk_eval.get("start_tcp_error_m")
    hand_err = fk_eval.get("start_hand_error_m")
    return (
        "[SIMPLE_DIRECT_PREGRASP_START_FK_VALIDATE]\n"
        "label=%s\n"
        "start_hand_error_m=%s\n"
        "start_tcp_error_m=%s\n"
        "pregrasp_tcp_desired=%s\n"
        "fk_grasp_tcp=%s\n"
        "fk_panda_hand=%s\n"
        "result=%s\n"
        "reason=%s"
        % (
            label,
            "n/a" if hand_err is None else "%.4f" % float(hand_err),
            "n/a" if tcp_err is None else "%.4f" % float(tcp_err),
            _fmt_xyz(fk_eval.get("pregrasp_tcp_desired")),
            _fmt_xyz(fk_eval.get("fk_grasp_tcp")),
            _fmt_xyz(fk_eval.get("fk_panda_hand")),
            result,
            str(fk_eval.get("reason") or "n/a"),
        )
    )


def evaluate_simple_direct_vertical_descend_geometry(
    pre_plan: Tuple[float, float, float],
    gr_plan: Tuple[float, float, float],
    *,
    xy_tol_m: float = 0.002,
) -> Tuple[bool, str]:
    dxy = math.hypot(
        float(pre_plan[0]) - float(gr_plan[0]),
        float(pre_plan[1]) - float(gr_plan[1]),
    )
    if dxy > float(xy_tol_m):
        return False, "xy_mismatch_dxy=%.4f" % dxy
    if float(pre_plan[2]) <= float(gr_plan[2]) + 1e-6:
        return False, "non_vertical_descend"
    return True, "vertical_ok"


def format_sugar_box_simple_descend_prevalidate_log(payload: Dict[str, Any]) -> str:
    return (
        "[SUGAR_BOX_SIMPLE_DESCEND_PREVALIDATE]\n"
        "pregrasp_tcp=%s\n"
        "grasp_tcp=%s\n"
        "descend_delta=%s\n"
        "target_collision_present=false\n"
        "cartesian_fraction=%s\n"
        "endpoint_ik_ok=%s\n"
        "result=%s\n"
        "reason=%s"
        % (
            payload.get("pregrasp_tcp", "n/a"),
            payload.get("grasp_tcp", "n/a"),
            payload.get("descend_delta", "n/a"),
            payload.get("cartesian_fraction", "n/a"),
            payload.get("endpoint_ik_ok", "n/a"),
            payload.get("result", "n/a"),
            payload.get("reason", ""),
        )
    )
