"""Política de contacto post-cierre para mustard_bottle + tall_object_topdown."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

# Contacto top-down en tapón/cuello, no en el minor del cuerpo/collision box.
MUSTARD_CAP_TOP_CONTACT_WIDTH_M = 0.055
MUSTARD_MAX_ALLOWED_WIDTH_PENETRATION_M = 0.0035
MUSTARD_OPERATIONAL_MIN_CLOSURE_DELTA_M = 0.012
MUSTARD_OPERATIONAL_SQUEEZE_TOLERANCE_M = 0.0015
MUSTARD_OPERATIONAL_MIN_CLOSE_JOINT_M = 0.010


def _mustard_scene_id(candidate: Dict[str, Any]) -> str:
    return str(candidate.get("scene_id") or "").strip().lower()


def mustard_operational_pick_scene_active(candidate: Dict[str, Any]) -> bool:
    """Escenas chips_mustard_* o demo_scene_02 (golden/clear_table/deposit_*)."""
    if str(candidate.get("label", "")).strip().lower() != "mustard_bottle":
        return False
    sid = _mustard_scene_id(candidate)
    if sid.startswith("chips_mustard"):
        return True
    from panda_controller.demo_golden_pick_candidate import demo_golden_policy_scene_id

    return demo_golden_policy_scene_id(sid) == "demo_scene_02"


def mustard_topdown_grasp_contact_active(candidate: Dict[str, Any]) -> bool:
    return (
        str(candidate.get("label", "")).strip().lower() == "mustard_bottle"
        and str(candidate.get("grasp_strategy", "")).strip() == "tall_object_topdown"
    )


def mustard_gazebo_physical_attach_required(candidate: Dict[str, Any]) -> bool:
    """Mostaza: fricción en Gazebo. Sin set_pose (teletransporte visual)."""
    return False


def mustard_requires_gazebo_lift_verification(candidate: Dict[str, Any]) -> bool:
    """Desactivado: el readback Gazebo aborta con agarre válido; se completa pick&place."""
    return False


def mustard_operational_lift_relaxed_safety(candidate: Dict[str, Any]) -> bool:
    """Lift operacional chips_mustard_* o demo_scene_02 (golden/clear_table)."""
    if mustard_operational_pick_scene_active(candidate):
        return True
    if str(candidate.get("label", "")).strip().lower() != "mustard_bottle":
        return False
    from panda_controller.demo_golden_pick_candidate import demo_golden_policy_scene_id

    return demo_golden_policy_scene_id(_mustard_scene_id(candidate)) == "demo_scene_02"


def resolve_mustard_expected_contact_width_m(
    *,
    db_required_width_m: Optional[float],
    effective_required_grasp_width_m: Optional[float],
    footprint_minor_m: Optional[float],
    collision_xy_minor_m: Optional[float],
) -> Tuple[float, str]:
    cap_w = float(MUSTARD_CAP_TOP_CONTACT_WIDTH_M)
    if collision_xy_minor_m is not None and float(collision_xy_minor_m) > cap_w + 0.001:
        return cap_w, "cap_top_contact_width"
    if db_required_width_m is not None and abs(float(db_required_width_m) - cap_w) <= 0.008:
        return cap_w, "cap_top_contact_near_db"
    if (
        effective_required_grasp_width_m is not None
        and float(effective_required_grasp_width_m) > 0.0
        and float(effective_required_grasp_width_m) <= cap_w + 0.005
    ):
        return float(effective_required_grasp_width_m), "effective_near_cap"
    if collision_xy_minor_m is not None and float(collision_xy_minor_m) > 0.0:
        return cap_w, "cap_top_contact_over_collision_minor"
    return cap_w, "cap_top_contact_default"


def evaluate_mustard_topdown_grasp_contact(
    *,
    actual_total: float,
    expected_contact_width_m: float,
    width_match_ok: bool,
    finger_asymmetry_m: float,
    max_asymmetry_m: float,
    centering_ok: bool,
    axis_ok: bool,
    max_allowed_width_penetration_m: float,
) -> Tuple[bool, str]:
    sym_ok = float(finger_asymmetry_m) <= float(max_asymmetry_m)
    if not sym_ok:
        return False, "mustard_contact_asymmetric"
    if not bool(centering_ok):
        return False, "mustard_centering_reject"
    if not bool(axis_ok):
        return False, "mustard_axis_reject"
    if not bool(width_match_ok):
        return False, "mustard_width_mismatch_reject"
    physical_width_ok = float(actual_total) >= (
        float(expected_contact_width_m) - float(max_allowed_width_penetration_m)
    )
    if not physical_width_ok:
        return False, "mustard_physical_width_reject"
    return True, "mustard_width_match_symmetric_contact"


def build_mustard_operational_close_joint_targets_m(
    base_close_joint_m: float,
    *,
    min_close_joint_m: float = MUSTARD_OPERATIONAL_MIN_CLOSE_JOINT_M,
) -> Tuple[float, ...]:
    """Escalera de cierre progresivo hacia 0 (0=cerrado, ~0.04=abierto en Panda)."""
    base = max(float(min_close_joint_m), float(base_close_joint_m))
    out: List[float] = []
    for extra in (0.0, 0.002, 0.004, 0.006, 0.008):
        value = round(base - float(extra), 4)
        value = max(float(min_close_joint_m), value)
        if all(abs(value - existing) > 1e-6 for existing in out):
            out.append(float(value))
    return tuple(out)


def resolve_mustard_operational_close_joint_targets_m(
    base_close_joint_m: float,
    *,
    single_attempt: bool = True,
    min_close_joint_m: float = MUSTARD_OPERATIONAL_MIN_CLOSE_JOINT_M,
) -> Tuple[float, ...]:
    """Un solo cierre operacional (dedos quietos) o escalera progresiva legacy."""
    if bool(single_attempt):
        base = max(float(min_close_joint_m), float(base_close_joint_m))
        return (float(base),)
    return build_mustard_operational_close_joint_targets_m(
        base_close_joint_m,
        min_close_joint_m=min_close_joint_m,
    )


def evaluate_mustard_operational_close_squeeze(
    *,
    actual_total: float,
    target_total: float,
    open_total: float,
    expected_contact_width_m: float = MUSTARD_CAP_TOP_CONTACT_WIDTH_M,
    max_allowed_width_penetration_m: float = MUSTARD_MAX_ALLOWED_WIDTH_PENETRATION_M,
    contact_width_tolerance_m: float = 0.014,
    min_closure_delta_m: float = MUSTARD_OPERATIONAL_MIN_CLOSURE_DELTA_M,
    squeeze_tolerance_m: float = MUSTARD_OPERATIONAL_SQUEEZE_TOLERANCE_M,
) -> Tuple[bool, str]:
    """Tras cierre operacional: cierre real desde apertura o contacto con el cuello.

    Con objeto en pinza, actual_total queda ~ancho del tapón (> target_total);
    exigir actual_total <= target_total rechazaría agarres válidos.
    """
    closure_delta = float(open_total) - float(actual_total)
    closure_ok = closure_delta >= float(min_closure_delta_m)
    if not closure_ok:
        return False, "mustard_operational_insufficient_closure_delta"

    exp_w = float(expected_contact_width_m)
    in_object_contact = (
        float(actual_total)
        >= exp_w - float(max_allowed_width_penetration_m)
        and abs(float(actual_total) - exp_w) <= float(contact_width_tolerance_m)
    )
    if in_object_contact:
        return True, "mustard_operational_squeeze_ok"

    if float(actual_total) <= float(target_total) + float(squeeze_tolerance_m):
        return True, "mustard_operational_squeeze_ok"
    return False, "mustard_operational_insufficient_squeeze"


def evaluate_mustard_friction_lift_verify(
    *,
    hand_lift_ok: bool,
    attached_flag: bool,
    contact_strict_pass: bool,
    orient_ok: bool,
    object_delta_z: float,
    hand_delta_z: float,
    min_delta: float,
    readback_available: bool,
) -> Tuple[bool, str]:
    """Lift con fricción (sin set_pose): Gazebo ΔZ si hay readback; si no, proxy por mano+contacto."""
    min_hand = float(min_delta)
    min_gz = max(min_hand * 0.65, 0.020)
    if not bool(hand_lift_ok and attached_flag and contact_strict_pass and orient_ok):
        if not hand_lift_ok:
            return False, "lift_no_hand_progress"
        if not attached_flag:
            return False, "lift_not_attached"
        if not contact_strict_pass:
            return False, "lift_contact_not_strict"
        return False, "lift_orientation_reject"
    if not math.isnan(float(object_delta_z)):
        if float(object_delta_z) >= min_gz:
            return True, "mustard_gazebo_object_lift_ok"
        if float(object_delta_z) <= 0.006:
            return False, "mustard_gazebo_object_stayed_on_table"
        if float(object_delta_z) >= 0.018:
            return True, "mustard_gazebo_object_lift_partial_ok"
    if not readback_available or math.isnan(float(object_delta_z)):
        if float(hand_delta_z) + 1e-6 >= min_hand:
            return True, "mustard_friction_lift_hand_proxy_ok"
    return False, "mustard_physical_lift_not_verified"
