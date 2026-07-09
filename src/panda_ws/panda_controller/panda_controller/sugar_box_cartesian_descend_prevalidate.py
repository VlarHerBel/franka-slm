"""Fallback geométrico de prevalidación cartesiana sugar_box demo_scene_02 (sin ROS)."""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Sequence, Tuple

from panda_controller.demo_cracker_box_cartesian_prevalidate import (
    DEMO_SCENE_02_IDS,
    KNOWN_BOX_GEOMETRIC_FALLBACK_REASON,
    TWO_BOXES_SCENE_IDS,
    evaluate_known_box_geometric_vertical_descend_fallback,
    vertical_descend_volume_clear_of_obstacles,
)
from panda_controller.demo_pick_route_preflight import active_table_obstacle_count
from panda_controller.sugar_box_safe_entry import (
    sugar_box_multiobject_use_object_safe_above_stage,
)
from panda_controller.simple_direct_pick_route import (
    simple_direct_pick_route_eligible_for_candidate,
)

SUGAR_BOX_GEOMETRIC_FALLBACK_REASON = (
    "demo_geometric_vertical_descend_safe_after_moveit_false_negative"
)


def sugar_box_demo_scene_02_policy_active(
    *,
    label: str,
    scene_id: str,
) -> bool:
    return (
        str(label or "").strip().lower() == "sugar_box"
        and str(scene_id or "").strip().lower() in DEMO_SCENE_02_IDS
    )


def sugar_box_generic_geometric_fallback_policy_active(
    *,
    label: str,
    scene_id: str,
    candidate: Dict[str, Any],
) -> bool:
    """Fallback geométrico sugar fuera de demo_scene_02 (p. ej. two_boxes_*)."""
    if str(label or "").strip().lower() != "sugar_box":
        return False
    sid = str(scene_id or "").strip().lower()
    if sid in DEMO_SCENE_02_IDS:
        return False
    if sid in TWO_BOXES_SCENE_IDS:
        return sugar_box_cartesian_geometric_fallback_eligible(candidate)
    return sugar_box_cartesian_geometric_fallback_eligible(candidate)


def sugar_box_cartesian_geometric_fallback_eligible(
    candidate: Dict[str, Any],
) -> bool:
    """Fallback geométrico sugar: ruta multiobjeto safe o simple_direct activa."""
    if sugar_box_multiobject_use_object_safe_above_stage(candidate):
        return True
    return bool(simple_direct_pick_route_eligible_for_candidate(candidate)) and bool(
        candidate.get("_simple_direct_pick_route")
    )


def sync_sugar_box_max_cartesian_descend_after_pregrasp_raise(
    candidate: Dict[str, Any],
    *,
    selected_pregrasp_tcp_z: float,
    grasp_tcp_z: float,
) -> bool:
    """Tras elevar pregrasp por IK, alinear max_cartesian_descend_m con el descend real."""
    if str(candidate.get("label", "")).strip().lower() != "sugar_box":
        return False
    required = max(0.0, float(selected_pregrasp_tcp_z) - float(grasp_tcp_z))
    try:
        current_max = float(candidate.get("max_cartesian_descend_m") or 0.0)
    except (TypeError, ValueError):
        current_max = 0.0
    if required <= current_max + 1e-6:
        return False
    candidate["generic_max_cartesian_descend_m"] = current_max
    candidate["max_cartesian_descend_m"] = required
    candidate["sugar_box_pregrasp_raise_max_descend_sync"] = True
    return True


def evaluate_sugar_box_demo_geometric_vertical_descend_fallback(
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
    if sugar_box_generic_geometric_fallback_policy_active(
        label=label, scene_id=scene_id, candidate=candidate
    ):
        ok, reason = evaluate_known_box_geometric_vertical_descend_fallback(
            label=label,
            stage_label=stage_label,
            target_collision_removed=target_collision_removed,
            object_safe_above_to_pregrasp_ok=object_safe_above_to_pregrasp_ok,
            pre_plan=pre_plan,
            gr_plan=gr_plan,
            candidate=candidate,
            scene_obstacles=scene_obstacles,
            moveit_fraction=float(moveit_fraction),
            table_z_m=float(table_z_m),
            xy_tol_m=xy_tol_m,
            gripper_xy_radius_m=gripper_xy_radius_m,
            min_lateral_clearance_m=min_lateral_clearance_m,
        )
        if ok:
            return True, KNOWN_BOX_GEOMETRIC_FALLBACK_REASON
        if not bool(endpoint_ik_ok):
            obs_ok, _ = vertical_descend_volume_clear_of_obstacles(
                pre_plan,
                gr_plan,
                scene_obstacles,
                gripper_xy_radius_m=gripper_xy_radius_m,
                min_lateral_clearance_m=min_lateral_clearance_m,
            )
            if not obs_ok:
                return False, "endpoint_ik_not_ok"
        return ok, reason
    if not sugar_box_demo_scene_02_policy_active(label=label, scene_id=scene_id):
        return False, "not_demo_scene_02_sugar_box"
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
    top_z = candidate.get("top_z_m")
    try:
        top_z_f = float(top_z) if top_z is not None else None
    except (TypeError, ValueError):
        top_z_f = None
    if top_z_f is None:
        return False, "missing_top_z"
    depth_from_top = top_z_f - float(gr_plan[2])
    insertion_lim = candidate.get("insertion_depth_limit_m")
    if insertion_lim is None:
        insertion_lim = candidate.get("recommended_grasp_depth_from_top_m")
    try:
        insertion_lim_f = float(insertion_lim) if insertion_lim is not None else 0.036
    except (TypeError, ValueError):
        insertion_lim_f = 0.036
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
    return True, SUGAR_BOX_GEOMETRIC_FALLBACK_REASON


def resolve_sugar_box_pregrasp_tcp_xyz(
    candidate: Dict[str, Any],
    gr_plan: Tuple[float, float, float],
) -> Tuple[float, float, float]:
    """Resuelve pregrasp TCP para comprobaciones geométricas (p. ej. lift proxy)."""
    raw = candidate.get("pregrasp_tcp")
    if isinstance(raw, (list, tuple)) and len(raw) >= 3:
        return (float(raw[0]), float(raw[1]), float(raw[2]))
    plan_targets = candidate.get("_plan_targets") or candidate.get("plan_targets")
    if isinstance(plan_targets, dict):
        pt = plan_targets.get("pregrasp_tcp")
        if isinstance(pt, (list, tuple)) and len(pt) >= 3:
            return (float(pt[0]), float(pt[1]), float(pt[2]))
    pz = candidate.get("selected_pregrasp_tcp_z")
    if pz is None:
        pz = candidate.get("pregrasp_tcp_z")
    if pz is not None:
        return (float(gr_plan[0]), float(gr_plan[1]), float(pz))
    return (float(gr_plan[0]), float(gr_plan[1]), float(gr_plan[2]))


def sugar_box_geometric_lift_pregrasp_proxy_eligible(
    candidate: Dict[str, Any],
    *,
    gr_plan: Tuple[float, float, float],
    pregrasp_js: Any = None,
) -> bool:
    """Proxy lift desde pregrasp cuando descend geométrico no tiene IK de grasp."""
    if str(candidate.get("label", "")).strip().lower() != "sugar_box":
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
    scene_obstacles = candidate.get("scene_obstacles") or []
    if int(active_table_obstacle_count(scene_obstacles)) == 0:
        return True
    pre_plan = resolve_sugar_box_pregrasp_tcp_xyz(candidate, gr_plan)
    vol_ok, _ = vertical_descend_volume_clear_of_obstacles(
        pre_plan,
        gr_plan,
        scene_obstacles,
    )
    return bool(vol_ok)


def sugar_box_final_descend_avoid_collisions_effective(
    candidate: Dict[str, Any],
    *,
    policy_in_contact_zone: bool,
    target_collision_removed: bool,
) -> Optional[bool]:
    """Override MoveIt avoid_collisions en descenso final sugar_box (None = global)."""
    if str(candidate.get("label", "")).strip().lower() != "sugar_box":
        return None
    src = str(candidate.get("_cartesian_descend_prevalidation_source") or "").strip()
    if src == "geometric_fallback" or bool(
        candidate.get("_sugar_box_geometric_fallback_runtime_prepared")
    ):
        return False
    if bool(candidate.get("_sugar_box_final_descend_relaxed_cartesian")):
        return False
    if target_collision_removed:
        return bool(policy_in_contact_zone)
    return None


def sugar_box_final_descend_fraction_threshold_effective(
    candidate: Dict[str, Any],
    *,
    default_threshold: float,
    geometric_fallback_threshold: float = 0.80,
) -> float:
    """Umbral de fracción cartesiana para descenso sugar (geometric_fallback más permisivo)."""
    src = str(candidate.get("_cartesian_descend_prevalidation_source") or "").strip()
    if src == "geometric_fallback" or bool(
        candidate.get("_sugar_box_final_descend_relaxed_cartesian")
    ):
        return min(float(default_threshold), float(geometric_fallback_threshold))
    return float(default_threshold)


def sugar_box_geometric_fallback_runtime_descend_eligible(
    candidate: Dict[str, Any],
    *,
    micro_descend_enabled: bool = True,
) -> bool:
    """Runtime descend debe usar IK/micro cuando prevalidate aceptó geometric_fallback."""
    if str(candidate.get("label", "")).strip().lower() != "sugar_box":
        return False
    if (
        str(candidate.get("_cartesian_descend_prevalidation_source") or "").strip()
        != "geometric_fallback"
    ):
        return False
    if bool(candidate.get("_sugar_box_micro_descend_from_pregrasp_used")):
        return False
    if bool(candidate.get("_sugar_box_guarded_ik_descend_validated")):
        return False
    if bool(candidate.get("_sugar_box_descend_use_segmented")):
        return False
    if bool(candidate.get("_sugar_box_geometric_fallback_runtime_prepared")):
        return False
    return bool(micro_descend_enabled)


def format_sugar_box_cartesian_descend_prevalidate_fallback_log(
    *,
    result: str,
    reason: str,
    moveit_fraction: float,
) -> str:
    return (
        "[SUGAR_BOX_CARTESIAN_DESCEND_PREVALIDATE_FALLBACK]\n"
        "moveit_fraction=%.5f\n"
        "result=%s\n"
        "reason=%s"
        % (float(moveit_fraction), str(result), str(reason))
    )
