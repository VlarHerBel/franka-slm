"""Prevalidación cartesiana mustard_bottle demo_scene_02 (fallback geométrico, sin ROS)."""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Sequence, Tuple

from panda_controller.demo_cracker_box_cartesian_prevalidate import (
    vertical_descend_volume_clear_of_obstacles,
)
from panda_controller.mustard_depth_search import mustard_bottle_extended_pick_scene_active

MUSTARD_GEOMETRIC_FALLBACK_REASON = "mustard_demo_vertical_descend_volume_clear"


def mustard_demo_scene_02_policy_active(*, label: str, scene_id: str) -> bool:
    return (
        str(label).strip().lower() == "mustard_bottle"
        and mustard_bottle_extended_pick_scene_active(scene_id)
    )


def evaluate_mustard_demo_geometric_vertical_descend_fallback(
    *,
    label: str,
    scene_id: str,
    stage_label: str,
    target_collision_removed: bool,
    object_safe_above_to_pregrasp_ok: bool,
    pre_plan: Tuple[float, float, float],
    gr_plan: Tuple[float, float, float],
    candidate: Dict[str, Any],
    scene_obstacles: Sequence[Dict[str, Any]],
    moveit_fraction: float,
    table_z_m: float,
    endpoint_ik_ok: bool = True,
    xy_tol_m: float = 0.002,
    gripper_xy_radius_m: float = 0.055,
    min_lateral_clearance_m: float = 0.025,
) -> Tuple[bool, str]:
    if not mustard_demo_scene_02_policy_active(label=label, scene_id=scene_id):
        return False, "not_demo_scene_02_mustard_bottle"
    if str(stage_label or "").strip() != "pregrasp_to_grasp_cartesian":
        return False, "wrong_stage"
    if not bool(target_collision_removed):
        return False, "target_collision_not_removed"
    if not bool(object_safe_above_to_pregrasp_ok):
        return False, "object_safe_above_to_pregrasp_not_ok"
    dxy = math.hypot(
        float(pre_plan[0]) - float(gr_plan[0]),
        float(pre_plan[1]) - float(gr_plan[1]),
    )
    if dxy > float(xy_tol_m):
        return False, "xy_mismatch_dxy=%.4f" % dxy
    if float(pre_plan[2]) <= float(gr_plan[2]) + 1e-6:
        return False, "non_vertical_descend"
    eff_top = candidate.get("_effective_top_z_for_palm_bridge_m")
    try:
        eff_top_f = float(eff_top) if eff_top is not None else None
    except (TypeError, ValueError):
        eff_top_f = None
    top_z = candidate.get("top_z_m")
    try:
        top_z_f = float(top_z) if top_z is not None else None
    except (TypeError, ValueError):
        top_z_f = None
    if top_z_f is None and eff_top_f is None:
        return False, "missing_top_z"
    depth_top_z = eff_top_f if eff_top_f is not None else float(top_z_f)
    depth_from_top = float(depth_top_z) - float(gr_plan[2])
    insertion_lim = candidate.get("insertion_depth_limit_m")
    if insertion_lim is None:
        insertion_lim = candidate.get("recommended_grasp_depth_from_top_m")
    if insertion_lim is None:
        insertion_lim = candidate.get("new_depth_from_effective_top_m")
    try:
        insertion_lim_f = float(insertion_lim) if insertion_lim is not None else 0.040
    except (TypeError, ValueError):
        insertion_lim_f = 0.040
    if bool(candidate.get("mustard_scanner_aligned_pregrasp_locked")):
        insertion_lim_f = max(
            insertion_lim_f,
            float(depth_from_top) + 1e-6,
        )
    if depth_from_top > insertion_lim_f + 1e-6:
        return False, "depth_from_top_exceeds_limit"
    min_table_clr = candidate.get("min_tcp_clearance_above_table_m")
    try:
        min_table_clr_f = (
            float(min_table_clr) if min_table_clr is not None else 0.012
        )
    except (TypeError, ValueError):
        min_table_clr_f = 0.012
    if float(gr_plan[2]) <= float(table_z_m) + min_table_clr_f - 1e-6:
        return False, "grasp_tcp_below_table_clearance"
    descend_m = float(pre_plan[2]) - float(gr_plan[2])
    max_descend = candidate.get("max_cartesian_descend_m")
    try:
        max_descend_f = float(max_descend) if max_descend is not None else 0.120
    except (TypeError, ValueError):
        max_descend_f = 0.120
    if descend_m > max_descend_f + 1e-6:
        return False, "descend_exceeds_max_cartesian_descend_m"
    obs_ok, obs_reason = vertical_descend_volume_clear_of_obstacles(
        pre_plan,
        gr_plan,
        scene_obstacles,
        gripper_xy_radius_m=gripper_xy_radius_m,
        min_lateral_clearance_m=min_lateral_clearance_m,
    )
    if not bool(endpoint_ik_ok) and not obs_ok:
        return False, "endpoint_ik_not_ok"
    if not obs_ok:
        return False, obs_reason
    if float(moveit_fraction) + 1e-6 >= 0.95:
        return False, "moveit_not_false_negative"
    return True, MUSTARD_GEOMETRIC_FALLBACK_REASON


def mustard_geometric_lift_pregrasp_proxy_eligible(
    candidate: Dict[str, Any],
    *,
    gr_plan: Tuple[float, float, float],
    pregrasp_js: Any = None,
) -> bool:
    """Proxy lift desde pregrasp cuando descend geométrico no tiene IK de grasp."""
    if str(candidate.get("label", "")).strip().lower() != "mustard_bottle":
        return False
    if (
        str(candidate.get("_cartesian_descend_prevalidation_source") or "").strip()
        != "geometric_fallback"
    ):
        return False
    if not bool(candidate.get("_simple_direct_pick_route")):
        return False
    if pregrasp_js is None:
        return False
    pre_raw = candidate.get("_mustard_selected_pregrasp_tcp")
    if isinstance(pre_raw, (list, tuple)) and len(pre_raw) >= 3:
        pre_plan = (float(pre_raw[0]), float(pre_raw[1]), float(pre_raw[2]))
    else:
        pre_plan = (float(gr_plan[0]), float(gr_plan[1]), float(gr_plan[2]) + 0.05)
    vol_ok, _ = vertical_descend_volume_clear_of_obstacles(
        pre_plan,
        gr_plan,
        candidate.get("scene_obstacles") or [],
    )
    return bool(vol_ok)


def mustard_geometric_fallback_runtime_descend_eligible(
    candidate: Dict[str, Any],
) -> bool:
    if str(candidate.get("label", "")).strip().lower() != "mustard_bottle":
        return False
    if (
        str(candidate.get("_cartesian_descend_prevalidation_source") or "").strip()
        != "geometric_fallback"
    ):
        return False
    if not bool(candidate.get("_simple_direct_pick_route")):
        return False
    if bool(candidate.get("_mustard_geometric_fallback_runtime_prepared")):
        return False
    if bool(candidate.get("_sugar_box_guarded_ik_descend_validated")):
        return False
    return True


MUSTARD_GUARDED_IK_Z_STEP_M = 0.005
MUSTARD_IK_STEPWISE_DZ_M = 0.005
MUSTARD_GEOMETRIC_FALLBACK_DESCEND_FRACTION_THRESHOLD = 0.15
MUSTARD_IK_STEPWISE_MIN_DEPTH_BELOW_TOP_M = 0.028


def mustard_final_descend_avoid_collisions_effective(
    candidate: Dict[str, Any],
    *,
    target_collision_removed: bool,
) -> Optional[bool]:
    """Override MoveIt avoid_collisions en descenso final mustard (None = global)."""
    if str(candidate.get("label", "")).strip().lower() != "mustard_bottle":
        return None
    src = str(candidate.get("_cartesian_descend_prevalidation_source") or "").strip()
    if src in (
        "geometric_fallback",
        "micro_descend_from_pregrasp",
        "guarded_ik_after_joint7",
    ) or bool(candidate.get("_mustard_geometric_fallback_runtime_prepared")):
        return False
    if bool(candidate.get("_mustard_final_descend_relaxed_cartesian")):
        return False
    return None


def mustard_final_descend_fraction_threshold_effective(
    candidate: Dict[str, Any],
    *,
    default_threshold: float,
    geometric_fallback_threshold: float = MUSTARD_GEOMETRIC_FALLBACK_DESCEND_FRACTION_THRESHOLD,
) -> float:
    """Umbral de fracción cartesiana para descend mustard (geometric_fallback permisivo)."""
    src = str(candidate.get("_cartesian_descend_prevalidation_source") or "").strip()
    if src == "geometric_fallback" or bool(
        candidate.get("_mustard_final_descend_relaxed_cartesian")
    ) or bool(candidate.get("_mustard_geometric_fallback_runtime_prepared")):
        return min(float(default_threshold), float(geometric_fallback_threshold))
    return float(default_threshold)


def format_mustard_cartesian_descend_prevalidate_fallback_log(
    *,
    result: str,
    reason: str,
    moveit_fraction: float,
) -> str:
    return (
        "[MUSTARD_CARTESIAN_DESCEND_PREVALIDATE_FALLBACK]\n"
        "result=%s\n"
        "reason=%s\n"
        "moveit_fraction=%.5f"
        % (str(result), str(reason), float(moveit_fraction))
    )
