"""Prevalidación cartesiana demo_scene_02 cracker_box (fallback geométrico seguro)."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

from panda_controller.target_collision_policy import object_collision_radius_xy

DEMO_GOLDEN_PARENT_SCENE_IDS = frozenset(
    {"demo_scene_01", "demo_scene_02", "demo_scene_03"}
)
DEMO_SCENE_02_CLEAR_TABLE_SCENE_ID = "demo_scene_02_clear_table"
DEMO_SCENE_02_IDS = DEMO_GOLDEN_PARENT_SCENE_IDS | frozenset(
    {DEMO_SCENE_02_CLEAR_TABLE_SCENE_ID}
)
TWO_BOXES_SCENE_IDS = frozenset(
    {"two_boxes_01", "two_boxes_02", "two_boxes_03", "two_boxes_easy"}
)

DEFAULT_CRACKER_BOX_DEMO_PREGRASP_CLEARANCE_M = 0.085
DEFAULT_CRACKER_BOX_DEMO_OBJECT_SAFE_ABOVE_CLEARANCE_M = 0.150
DEFAULT_CRACKER_BOX_DEMO_MAX_CARTESIAN_DESCEND_M = 0.135
DEMO_CRACKER_DESCEND_LIMIT_MARGIN_M = 0.010

GEOMETRIC_FALLBACK_REASON = (
    "demo_geometric_vertical_descend_safe_after_moveit_false_negative"
)
KNOWN_BOX_GEOMETRIC_FALLBACK_REASON = (
    "known_box_geometric_vertical_descend_safe_after_moveit_false_negative"
)
GENERIC_KNOWN_BOX_FK_ERROR_THRESHOLD_M = 0.015


def effective_demo_cracker_max_cartesian_descend_m(
    *,
    selected_pregrasp_tcp_z: float,
    grasp_tcp_z: float,
    demo_max_cartesian_descend_m: float,
    margin_m: float = DEMO_CRACKER_DESCEND_LIMIT_MARGIN_M,
) -> Tuple[float, float]:
    required = max(0.0, float(selected_pregrasp_tcp_z) - float(grasp_tcp_z))
    effective = max(
        float(demo_max_cartesian_descend_m),
        required + float(margin_m),
    )
    return required, effective


def apply_demo_cracker_descend_limit_policy(
    candidate: Dict[str, Any],
    *,
    selected_pregrasp_tcp_z: float,
    grasp_tcp_z: float,
    generic_max_cartesian_descend_m: float,
    demo_max_cartesian_descend_m: float = DEFAULT_CRACKER_BOX_DEMO_MAX_CARTESIAN_DESCEND_M,
) -> Dict[str, Any]:
    """Alinea max_cartesian_descend_m demo con pregrasp alto bloqueado."""
    required, effective = effective_demo_cracker_max_cartesian_descend_m(
        selected_pregrasp_tcp_z=selected_pregrasp_tcp_z,
        grasp_tcp_z=grasp_tcp_z,
        demo_max_cartesian_descend_m=demo_max_cartesian_descend_m,
    )
    generic = float(generic_max_cartesian_descend_m)
    candidate["generic_max_cartesian_descend_m"] = generic
    candidate["demo_max_cartesian_descend_m"] = float(demo_max_cartesian_descend_m)
    candidate["demo_required_cartesian_descend_m"] = required
    candidate["demo_effective_max_cartesian_descend_m"] = effective
    candidate["max_cartesian_descend_m"] = effective
    candidate["vertical_descend_tcp_m"] = required
    candidate["effective_approach_distance_m"] = required
    ok = required <= effective + 1e-6
    return {
        "label": "cracker_box",
        "selected_pregrasp_tcp_z": float(selected_pregrasp_tcp_z),
        "grasp_tcp_z": float(grasp_tcp_z),
        "required_descend_m": required,
        "generic_max_cartesian_descend_m": generic,
        "demo_max_cartesian_descend_m": float(demo_max_cartesian_descend_m),
        "effective_max_cartesian_descend_m": effective,
        "result": "OK" if ok else "FAIL",
    }


def demo_scene_02_cracker_box_policy_active(
    *,
    label: str,
    demo_authoritative_scene: bool,
    scene_id: str,
) -> bool:
    from panda_vision.spawn.demo_scene_presets import demo_scene_policy_scene_id_for_preset

    parent = demo_scene_policy_scene_id_for_preset(str(scene_id or ""))
    return (
        bool(demo_authoritative_scene)
        and parent in DEMO_GOLDEN_PARENT_SCENE_IDS
        and str(label or "").strip().lower() == "cracker_box"
    )


def cracker_box_paired_grid_search_active(
    *,
    label: str,
    demo_authoritative_scene: bool,
    scene_id: str,
    paired_grid_search_mode: str,
    execution_profile: str,
) -> bool:
    """Compat: delega en known_box_paired_grid_search_active."""
    return known_box_paired_grid_search_active(
        label=label,
        paired_grid_search_mode=paired_grid_search_mode,
    )


KNOWN_BOX_GRID_LABELS = frozenset({"cracker_box", "sugar_box"})


def known_box_paired_grid_search_active(
    *,
    label: str,
    paired_grid_search_mode: str,
) -> bool:
    """Grid paired (pregrasp+joint7+descend+lift) para cajas conocidas sin golden."""
    if str(label or "").strip().lower() not in KNOWN_BOX_GRID_LABELS:
        return False
    mode = str(paired_grid_search_mode or "").strip().lower()
    return mode not in ("", "off", "disabled", "none")


def apply_demo_scene_02_cracker_box_descend_sequence(
    seq: Dict[str, Any],
    *,
    top_z: float,
    pregrasp_clearance_m: float,
    object_safe_above_clearance_m: float,
    max_target_z: float,
) -> Dict[str, Any]:
    """Eleva pregrasp y object_safe_above para demo multiobjeto cracker_box."""
    grasp_tcp = tuple(seq["grasp_tcp"])
    clear_m = max(0.080, float(pregrasp_clearance_m))
    safe_clear = max(clear_m + 0.065, float(object_safe_above_clearance_m))
    max_descend = max(
        float(seq.get("max_cartesian_descend_m") or 0.10),
        clear_m + 0.025,
    )
    min_pregrasp_z = float(top_z) + clear_m
    pregrasp_z = max(float(grasp_tcp[2]) + clear_m, min_pregrasp_z)
    pregrasp_z = min(float(pregrasp_z), float(max_target_z))
    pregrasp_tcp = (float(grasp_tcp[0]), float(grasp_tcp[1]), float(pregrasp_z))
    object_safe_z = max(
        float(top_z) + safe_clear,
        float(pregrasp_z) + 0.050,
    )
    object_safe_z = min(float(object_safe_z), float(max_target_z))
    out = dict(seq)
    out["pregrasp_tcp"] = pregrasp_tcp
    out["final_descend_m"] = float(pregrasp_z) - float(grasp_tcp[2])
    out["max_cartesian_descend_m"] = float(max_descend)
    out["object_safe_above_tcp"] = (
        float(pregrasp_tcp[0]),
        float(pregrasp_tcp[1]),
        float(object_safe_z),
    )
    out["demo_pregrasp_clearance_m"] = clear_m
    out["demo_object_safe_above_clearance_m"] = safe_clear
    return out


def _obs_xy(obs: Dict[str, Any]) -> Optional[Tuple[float, float]]:
    pos = obs.get("position") or obs.get("center") or obs.get("grasp_center_base")
    if isinstance(pos, (list, tuple)) and len(pos) >= 2:
        try:
            return float(pos[0]), float(pos[1])
        except (TypeError, ValueError):
            return None
    x = obs.get("x")
    y = obs.get("y")
    try:
        if x is not None and y is not None:
            return float(x), float(y)
    except (TypeError, ValueError):
        pass
    return None


def vertical_descend_volume_clear_of_obstacles(
    pre_plan: Tuple[float, float, float],
    gr_plan: Tuple[float, float, float],
    scene_obstacles: Sequence[Dict[str, Any]],
    *,
    gripper_xy_radius_m: float = 0.055,
    min_lateral_clearance_m: float = 0.025,
) -> Tuple[bool, str]:
    px, py = float(pre_plan[0]), float(pre_plan[1])
    for obs in scene_obstacles:
        if not isinstance(obs, dict) or bool(obs.get("is_target", False)):
            continue
        oxy = _obs_xy(obs)
        if oxy is None:
            continue
        obs_r = float(object_collision_radius_xy(obs))
        required = obs_r + float(gripper_xy_radius_m) + float(min_lateral_clearance_m)
        dxy = math.hypot(px - oxy[0], py - oxy[1])
        if dxy + 1e-6 < required:
            lb = str(obs.get("label", "unknown"))
            return (
                False,
                "obstacle_%s_dxy=%.4f required=%.4f"
                % (lb, dxy, required),
            )
    return True, "ok"


def _evaluate_vertical_descend_geometric_core(
    *,
    label: str,
    stage_label: str,
    target_collision_removed: bool,
    object_safe_above_to_pregrasp_ok: bool,
    pre_plan: Tuple[float, float, float],
    gr_plan: Tuple[float, float, float],
    candidate: Dict[str, Any],
    scene_obstacles: Sequence[Dict[str, Any]],
    moveit_fraction: float,
    table_z_m: float,
    xy_tol_m: float = 0.002,
    gripper_xy_radius_m: float = 0.055,
    min_lateral_clearance_m: float = 0.025,
    max_descend_default_m: float = 0.120,
    insertion_depth_default_m: float = 0.036,
    success_reason: str = GEOMETRIC_FALLBACK_REASON,
) -> Tuple[bool, str]:
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
        insertion_lim_f = (
            float(insertion_lim) if insertion_lim is not None else insertion_depth_default_m
        )
    except (TypeError, ValueError):
        insertion_lim_f = insertion_depth_default_m
    if str(label or "").strip().lower() == "sugar_box":
        insertion_lim_f = max(insertion_lim_f, 0.028)
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
    max_descend = candidate.get("demo_effective_max_cartesian_descend_m")
    if max_descend is None:
        max_descend = candidate.get("max_cartesian_descend_m")
    try:
        max_descend_f = float(max_descend) if max_descend is not None else max_descend_default_m
    except (TypeError, ValueError):
        max_descend_f = max_descend_default_m
    if bool(candidate.get("demo_pregrasp_policy_locked")):
        demo_cap = candidate.get("demo_max_cartesian_descend_m")
        try:
            demo_cap_f = (
                float(demo_cap)
                if demo_cap is not None
                else DEFAULT_CRACKER_BOX_DEMO_MAX_CARTESIAN_DESCEND_M
            )
        except (TypeError, ValueError):
            demo_cap_f = DEFAULT_CRACKER_BOX_DEMO_MAX_CARTESIAN_DESCEND_M
        _, max_descend_f = effective_demo_cracker_max_cartesian_descend_m(
            selected_pregrasp_tcp_z=float(pre_plan[2]),
            grasp_tcp_z=float(gr_plan[2]),
            demo_max_cartesian_descend_m=demo_cap_f,
        )
    if descend_m > max_descend_f + 1e-6:
        return False, "descend_exceeds_max_cartesian_descend_m"
    yaw_policy = str(candidate.get("yaw_policy") or candidate.get("grasp_strategy") or "")
    if yaw_policy and yaw_policy not in (
        "align_short_axis",
        "topdown",
        "tall_object_topdown",
        "keep_current",
        "no_change",
        "",
    ):
        return False, "orientation_not_topdown_locked"
    obs_ok, obs_reason = vertical_descend_volume_clear_of_obstacles(
        pre_plan,
        gr_plan,
        scene_obstacles,
        gripper_xy_radius_m=gripper_xy_radius_m,
        min_lateral_clearance_m=min_lateral_clearance_m,
    )
    if not obs_ok:
        return False, obs_reason
    if float(moveit_fraction) + 1e-6 >= 0.95:
        return False, "moveit_not_false_negative"
    return True, success_reason


def evaluate_demo_geometric_vertical_descend_fallback(
    *,
    label: str,
    demo_authoritative_scene: bool,
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
    xy_tol_m: float = 0.002,
    gripper_xy_radius_m: float = 0.055,
    min_lateral_clearance_m: float = 0.025,
) -> Tuple[bool, str]:
    if not demo_scene_02_cracker_box_policy_active(
        label=label,
        demo_authoritative_scene=demo_authoritative_scene,
        scene_id=scene_id,
    ):
        return False, "not_demo_scene_02_cracker_box"
    return _evaluate_vertical_descend_geometric_core(
        label=label,
        stage_label=stage_label,
        target_collision_removed=target_collision_removed,
        object_safe_above_to_pregrasp_ok=object_safe_above_to_pregrasp_ok,
        pre_plan=pre_plan,
        gr_plan=gr_plan,
        candidate=candidate,
        scene_obstacles=scene_obstacles,
        moveit_fraction=moveit_fraction,
        table_z_m=table_z_m,
        xy_tol_m=xy_tol_m,
        gripper_xy_radius_m=gripper_xy_radius_m,
        min_lateral_clearance_m=min_lateral_clearance_m,
        success_reason=GEOMETRIC_FALLBACK_REASON,
    )


def evaluate_known_box_geometric_vertical_descend_fallback(
    *,
    label: str,
    stage_label: str,
    target_collision_removed: bool,
    object_safe_above_to_pregrasp_ok: bool,
    pre_plan: Tuple[float, float, float],
    gr_plan: Tuple[float, float, float],
    candidate: Dict[str, Any],
    scene_obstacles: Sequence[Dict[str, Any]],
    moveit_fraction: float,
    table_z_m: float,
    xy_tol_m: float = 0.002,
    gripper_xy_radius_m: float = 0.055,
    min_lateral_clearance_m: float = 0.025,
) -> Tuple[bool, str]:
    """Fallback geométrico para pick_place genérico (cracker/sugar, sin demo_scene_02)."""
    if str(label or "").strip().lower() not in KNOWN_BOX_GRID_LABELS:
        return False, "not_known_box_label"
    return _evaluate_vertical_descend_geometric_core(
        label=label,
        stage_label=stage_label,
        target_collision_removed=target_collision_removed,
        object_safe_above_to_pregrasp_ok=object_safe_above_to_pregrasp_ok,
        pre_plan=pre_plan,
        gr_plan=gr_plan,
        candidate=candidate,
        scene_obstacles=scene_obstacles,
        moveit_fraction=moveit_fraction,
        table_z_m=table_z_m,
        xy_tol_m=xy_tol_m,
        gripper_xy_radius_m=gripper_xy_radius_m,
        min_lateral_clearance_m=min_lateral_clearance_m,
        max_descend_default_m=0.135,
        success_reason=KNOWN_BOX_GEOMETRIC_FALLBACK_REASON,
    )


def known_box_geometric_lift_pregrasp_proxy_eligible(
    candidate: Dict[str, Any],
    *,
    gr_plan: Tuple[float, float, float],
    pregrasp_js: Any = None,
    pre_plan: Optional[Tuple[float, float, float]] = None,
) -> bool:
    """Proxy lift desde pregrasp cuando descend geométrico no tiene IK de grasp."""
    if str(candidate.get("label", "")).strip().lower() not in KNOWN_BOX_GRID_LABELS:
        return False
    if not bool(candidate.get("_known_box_geometric_fallback_validated")):
        src = str(
            candidate.get("_cartesian_descend_prevalidation_source")
            or candidate.get("cartesian_descend_source")
            or ""
        ).strip()
        if src not in ("geometric_fallback", "paired_safe_geometric"):
            return False
    if pregrasp_js is None:
        return False
    pp = pre_plan
    if pp is None:
        stored = candidate.get("_known_box_geometric_pre_plan")
        if isinstance(stored, (list, tuple)) and len(stored) >= 3:
            pp = (float(stored[0]), float(stored[1]), float(stored[2]))
    if pp is None:
        pz = candidate.get("selected_pregrasp_tcp_z")
        if pz is None:
            pz = candidate.get("pregrasp_tcp_z")
        if pz is not None:
            pp = (float(gr_plan[0]), float(gr_plan[1]), float(pz))
        else:
            pp = (float(gr_plan[0]), float(gr_plan[1]), float(gr_plan[2]) + 0.08)
    vol_ok, _ = vertical_descend_volume_clear_of_obstacles(
        pp,
        gr_plan,
        candidate.get("scene_obstacles") or [],
    )
    return bool(vol_ok)


def evaluate_paired_known_box_geometric_descend_fallback(
    *,
    fk_contract_ok: bool,
    endpoint_ik_ok: bool,
    joint7_offline_ok: bool,
    label: str,
    stage_label: str,
    target_collision_removed: bool,
    object_safe_above_to_pregrasp_ok: bool,
    pre_plan: Tuple[float, float, float],
    gr_plan: Tuple[float, float, float],
    candidate: Dict[str, Any],
    scene_obstacles: Sequence[Dict[str, Any]],
    moveit_fraction: float,
    table_z_m: float,
) -> Tuple[bool, str]:
    if not bool(fk_contract_ok):
        return False, "fk_contract_not_ok"
    if not bool(endpoint_ik_ok) and not bool(joint7_offline_ok):
        return False, "endpoint_ik_and_joint7_fail"
    return evaluate_known_box_geometric_vertical_descend_fallback(
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
    )


def evaluate_paired_safe_geometric_descend_fallback(
    *,
    fk_contract_ok: bool,
    endpoint_ik_ok: bool,
    label: str,
    demo_authoritative_scene: bool,
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
) -> Tuple[bool, str]:
    """Fallback geométrico estricto solo tras FK+endpoint IK OK (paired cracker_box)."""
    if not bool(fk_contract_ok):
        return False, "fk_contract_not_ok"
    if not bool(endpoint_ik_ok):
        return False, "endpoint_ik_not_ok"
    return evaluate_demo_geometric_vertical_descend_fallback(
        label=label,
        demo_authoritative_scene=demo_authoritative_scene,
        scene_id=scene_id,
        stage_label=stage_label,
        target_collision_removed=target_collision_removed,
        object_safe_above_to_pregrasp_ok=object_safe_above_to_pregrasp_ok,
        pre_plan=pre_plan,
        gr_plan=gr_plan,
        candidate=candidate,
        scene_obstacles=scene_obstacles,
        moveit_fraction=float(moveit_fraction),
        table_z_m=float(table_z_m),
    )


def cartesian_descend_start_state_diagnostics(
    *,
    pre_plan: Tuple[float, float, float],
    pregrasp_js: Any,
    current_js: Any,
    joint_state_distance_fn: Any,
    fk_tcp_fn: Any,
    current_tcp_fn: Any,
    tcp_xyz_error_fn: Any,
    pre_hand_plan: Optional[Tuple[float, float, float]] = None,
    fk_hand_fn: Any = None,
) -> Dict[str, Any]:
    virtual_available = pregrasp_js is not None
    current_tcp = current_tcp_fn() if callable(current_tcp_fn) else None
    fk_tcp = fk_tcp_fn(pregrasp_js) if virtual_available and callable(fk_tcp_fn) else None
    fk_hand = (
        fk_hand_fn(pregrasp_js)
        if virtual_available and callable(fk_hand_fn)
        else None
    )
    js_dist = None
    if (
        virtual_available
        and current_js is not None
        and callable(joint_state_distance_fn)
    ):
        try:
            js_dist = float(joint_state_distance_fn(current_js, pregrasp_js))
        except (TypeError, ValueError):
            js_dist = None
    start_tcp_err = (
        tcp_xyz_error_fn(fk_tcp, pre_plan)
        if fk_tcp is not None and callable(tcp_xyz_error_fn)
        else None
    )
    start_hand_err = None
    if (
        pre_hand_plan is not None
        and fk_hand is not None
        and callable(tcp_xyz_error_fn)
    ):
        start_hand_err = tcp_xyz_error_fn(fk_hand, pre_hand_plan)
    current_err = (
        tcp_xyz_error_fn(current_tcp, pre_plan)
        if current_tcp is not None and callable(tcp_xyz_error_fn)
        else None
    )
    tol = 0.015
    using_virtual = bool(
        virtual_available
        and (
            current_err is None
            or float(current_err) > tol
            or js_dist is None
            or float(js_dist) > 0.05
        )
    )
    if virtual_available and using_virtual:
        source = "virtual_pregrasp"
    elif current_tcp is not None:
        source = "current_state"
    else:
        source = "fallback"
    return {
        "source": source,
        "using_virtual_start_state": using_virtual and virtual_available,
        "virtual_pregrasp_available": virtual_available,
        "current_js_distance_to_virtual_pregrasp": js_dist,
        "start_tcp_error_m": start_tcp_err,
        "start_hand_error_m": start_hand_err,
        "current_tcp_error_m": current_err,
        "fk_start_tcp": fk_tcp,
        "fk_start_hand": fk_hand,
        "current_tcp": current_tcp,
    }
