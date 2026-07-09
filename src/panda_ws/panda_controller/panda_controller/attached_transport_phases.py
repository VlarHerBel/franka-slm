"""Contrato por fases: escape local, zona de reconfiguración y ruta direct_action."""

from __future__ import annotations

import math
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from panda_controller.attached_transport_entry_validate import (
    decide_attached_transport_preflight,
    emit_transport_exit_candidate_validate_logs,
    select_transport_entry_validate_only,
)
from panda_controller.demo_scene_policy import has_remaining_table_obstacles
from panda_controller.generic_known_scene_carry_planner import (
    DEFAULT_ATTACHED_TRANSPORT_SAFETY_MARGIN_TOLERANCE_M,
    validate_attached_joint_segment,
)

# Límites Franka Panda (rad) — margen conservador para scoring.
PANDA_JOINT_LIMITS_RAD: Tuple[Tuple[float, float], ...] = (
    (-2.8973, 2.8973),
    (-1.7628, 1.7628),
    (-2.8973, 2.8973),
    (-3.0718, -0.0698),
    (-2.8973, 2.8973),
    (-0.0175, 3.7525),
    (-2.8973, 2.8973),
)

DEFAULT_RECONFIGURATION_MIN_TABLE_CLEARANCE_M = 0.200
DEFAULT_RECONFIGURATION_MIN_XY_CLEARANCE_M = 0.080
DEFAULT_LOCAL_EXIT_REQUIRED_CLEARANCE_M = 0.050
DEFAULT_LOCAL_EXIT_MIN_TABLE_CLEARANCE_M = 0.200
DEFAULT_RECONFIGURATION_REQUIRED_CLEARANCE_M = 0.080
DEFAULT_GLOBAL_ROUTE_REQUIRED_CLEARANCE_M = 0.100


def resolve_reconfiguration_safety_thresholds(
    scene_policy: Optional[Dict[str, Any]],
) -> Dict[str, float]:
    safety = (scene_policy or {}).get("safety") or {}
    return {
        "min_table_clearance_m": float(
            safety.get(
                "reconfiguration_min_table_clearance_m",
                DEFAULT_RECONFIGURATION_MIN_TABLE_CLEARANCE_M,
            )
        ),
        "min_xy_clearance_m": float(
            safety.get(
                "reconfiguration_min_xy_clearance_m",
                DEFAULT_RECONFIGURATION_MIN_XY_CLEARANCE_M,
            )
        ),
    }


def resolve_transport_phase_clearance_thresholds(
    scene_policy: Optional[Dict[str, Any]],
    carry_policy: Optional[Dict[str, Any]] = None,
) -> Dict[str, float]:
    """Umbrales por fase: escape local, zona segura y transporte global."""
    safety = (scene_policy or {}).get("safety") or {}
    carry = carry_policy or {}
    reconfig_xy = float(
        safety.get(
            "reconfiguration_required_clearance_m",
            safety.get(
                "reconfiguration_min_xy_clearance_m",
                DEFAULT_RECONFIGURATION_REQUIRED_CLEARANCE_M,
            ),
        )
    )
    global_route = float(
        safety.get(
            "global_route_required_clearance_m",
            carry.get(
                "carry_clearance_above_obstacles_m",
                DEFAULT_GLOBAL_ROUTE_REQUIRED_CLEARANCE_M,
            ),
        )
    )
    return {
        "local_exit_required_clearance_m": float(
            safety.get(
                "local_exit_required_clearance_m",
                carry.get(
                    "local_exit_required_clearance_m",
                    DEFAULT_LOCAL_EXIT_REQUIRED_CLEARANCE_M,
                ),
            )
        ),
        "local_exit_min_table_clearance_m": float(
            safety.get(
                "local_exit_min_table_clearance_m",
                carry.get(
                    "local_exit_min_table_clearance_m",
                    DEFAULT_LOCAL_EXIT_MIN_TABLE_CLEARANCE_M,
                ),
            )
        ),
        "reconfiguration_required_clearance_m": reconfig_xy,
        "global_route_required_clearance_m": global_route,
    }


def validate_post_escape_hub_route(
    *,
    current_joints: Sequence[float],
    hub_waypoint_name: str,
    hub_waypoint_joints: Sequence[float],
    fk_hand_fn: Callable[[Any], Optional[Tuple[float, float, float]]],
    attached_geom: Dict[str, Any],
    scene_obstacles: Sequence[Dict[str, Any]],
    table_top_z: float,
    policy: Dict[str, Any],
) -> Tuple[bool, List[Dict[str, Any]]]:
    """Valida el primer segmento joint current->hub con objeto attached."""
    return validate_attached_direct_action_route(
        current_joints=current_joints,
        waypoint_names=[str(hub_waypoint_name)],
        waypoint_joints=[list(hub_waypoint_joints)],
        fk_hand_fn=fk_hand_fn,
        attached_geom=attached_geom,
        scene_obstacles=scene_obstacles,
        table_top_z=float(table_top_z),
        policy=policy,
    )


def joint_limit_margin_min(joints: Sequence[float]) -> float:
    margins: List[float] = []
    for i, (lo, hi) in enumerate(PANDA_JOINT_LIMITS_RAD):
        if i >= len(joints):
            break
        q = float(joints[i])
        lo_f, hi_f = (lo, hi) if lo <= hi else (hi, lo)
        margins.append(min(q - lo_f, hi_f - q))
    return min(margins) if margins else float("inf")


def joint_distance_rad(a: Sequence[float], b: Sequence[float]) -> float:
    n = min(len(a), len(b))
    if n == 0:
        return 0.0
    return math.sqrt(sum((float(a[i]) - float(b[i])) ** 2 for i in range(n)))


def wrist_twist_score(joints: Sequence[float]) -> float:
    """Menor es mejor: penaliza |j6|+|j7| lejos de postura transporte neutra."""
    if len(joints) < 7:
        return 0.0
    j6 = abs(float(joints[5]))
    j7 = abs(float(joints[6]))
    return j6 + j7


def elbow_posture_score(joints: Sequence[float]) -> float:
    """Menor es mejor: joint4 cerca del centro del rango."""
    if len(joints) < 4:
        return 0.0
    j4 = float(joints[3])
    lo, hi = PANDA_JOINT_LIMITS_RAD[3]
    lo_f, hi_f = (lo, hi) if lo <= hi else (hi, lo)
    mid = 0.5 * (lo_f + hi_f)
    span = max(1e-6, hi_f - lo_f)
    return abs(j4 - mid) / span


def check_transport_reconfiguration_zone(
    *,
    hand_pos: Tuple[float, float, float],
    attached_geom: Dict[str, Any],
    scene_obstacles: Sequence[Dict[str, Any]],
    table_top_z: float,
    policy: Dict[str, Any],
    min_table_clearance_m: float = DEFAULT_RECONFIGURATION_MIN_TABLE_CLEARANCE_M,
    min_xy_clearance_m: float = DEFAULT_RECONFIGURATION_MIN_XY_CLEARANCE_M,
) -> Dict[str, Any]:
    """True si el objeto está alto y lejos de obstáculos para reconfiguración articular."""
    from panda_controller.generic_known_scene_carry_planner import (
        attached_obstacle_clearance_3d,
        validate_attached_hand_pose,
    )

    below = float(attached_geom.get("carried_object_below_hand_m", 0.19))
    att_bottom = float(hand_pos[2]) - below
    table_clr = att_bottom - float(table_top_z)
    req_xy = float(
        policy.get(
            "reconfiguration_required_clearance_m",
            policy.get("carry_clearance_above_obstacles_m", 0.10),
        )
    )
    tol = float(
        policy.get(
            "attached_transport_safety_margin_tolerance_m",
            DEFAULT_ATTACHED_TRANSPORT_SAFETY_MARGIN_TOLERANCE_M,
        )
    )
    ok_pose, checks, _metrics = validate_attached_hand_pose(
        hand_pos,
        attached_geom,
        scene_obstacles,
        table_top_z=float(table_top_z),
        min_table_clearance_m=min_table_clearance_m * 0.5,
        required_xy_clearance_m=req_xy,
        safety_margin_tolerance_m=tol,
    )
    min_xy = float("inf")
    outside_cluster = True
    for chk in checks:
        if chk.get("result") == "SKIP":
            continue
        xy = float(chk.get("xy_clearance", 0.0))
        min_xy = min(min_xy, xy)
        if bool(chk.get("xy_overlap")) or xy < min_xy_clearance_m:
            outside_cluster = False
    hard = any(bool(c.get("hard_collision")) for c in checks)
    zone_ok = (
        table_clr >= float(min_table_clearance_m)
        and min_xy >= float(min_xy_clearance_m)
        and ok_pose
        and not hard
        and outside_cluster
    )
    return {
        "current_hand": hand_pos,
        "attached_bottom_z": att_bottom,
        "table_clearance": table_clr,
        "min_xy_clearance_to_obstacles": min_xy if min_xy != float("inf") else None,
        "outside_object_cluster": outside_cluster,
        "hard_collision": hard,
        "result": "OK" if zone_ok else "FAIL",
        "transport_reconfiguration_zone_ok": zone_ok,
    }


def format_transport_reconfiguration_zone_log(check: Dict[str, Any]) -> str:
    hand = check.get("current_hand") or (0.0, 0.0, 0.0)
    return (
        "[TRANSPORT_RECONFIGURATION_ZONE_CHECK]\n"
        "current_hand=(%.3f, %.3f, %.3f)\n"
        "attached_bottom_z=%s\n"
        "table_clearance=%s\n"
        "min_xy_clearance_to_obstacles=%s\n"
        "outside_object_cluster=%s\n"
        "result=%s"
        % (
            float(hand[0]),
            float(hand[1]),
            float(hand[2]),
            check.get("attached_bottom_z", "n/a"),
            check.get("table_clearance", "n/a"),
            check.get("min_xy_clearance_to_obstacles", "n/a"),
            str(bool(check.get("outside_object_cluster"))).lower(),
            str(check.get("result", "FAIL")),
        )
    )


HUB_SEGMENT_PREVALIDATED_MIN_CLEARANCE_FLOOR_M = -0.010


def should_skip_attached_direct_action_route_preflight(
    *,
    transport_entry_validated: bool,
    hub_segment_prevalidated: bool,
    obstacle_disturbed: bool,
) -> bool:
    """Evita re-validar ruta ya certificada en transport entry (falsos FAIL post-defer)."""
    return bool(
        transport_entry_validated
        and hub_segment_prevalidated
        and not obstacle_disturbed
    )


def format_attached_direct_action_route_skip_log(reason: str) -> str:
    return (
        "[ATTACHED_DIRECT_ACTION_ROUTE_VALIDATE]\n"
        "result=SKIP\n"
        "reason=%s"
        % str(reason)
    )


def maybe_accept_prevalidated_hub_segment(
    detail: Dict[str, Any],
    *,
    is_first_segment: bool,
    hub_segment_prevalidated: bool,
    transport_entry_verified: bool,
    obstacle_disturbed: bool,
    min_clearance_floor_m: float = HUB_SEGMENT_PREVALIDATED_MIN_CLEARANCE_FLOOR_M,
) -> Dict[str, Any]:
    """Acepta current->hub si ya fue prevalidado en escape local con margen pequeño."""
    out = dict(detail)
    if str(out.get("result", "")).upper() == "OK":
        out.setdefault("reason", "ok")
        return out
    if not is_first_segment:
        return out
    if not (
        hub_segment_prevalidated
        and transport_entry_verified
        and not obstacle_disturbed
    ):
        return out
    hard = bool(out.get("hard_collision"))
    min_clr = out.get("min_clearance")
    try:
        min_clr_f = float(min_clr) if min_clr is not None else float("-inf")
    except (TypeError, ValueError):
        min_clr_f = float("-inf")
    if math.isinf(min_clr_f) and min_clr_f > 0:
        min_clr_f = float("inf")
    if not hard and min_clr is None:
        out["result"] = "OK"
        out["reason"] = "hub_prevalidated_no_clearance_metric"
        out["decision"] = "ALLOW_BORDERLINE"
        return out
    if not hard and min_clr_f == float("inf"):
        out["result"] = "OK"
        out["reason"] = "hub_prevalidated_unobstructed_path"
        out["decision"] = "ALLOW_BORDERLINE"
        return out
    if not hard and min_clr_f + 1e-9 >= float(min_clearance_floor_m):
        out["result"] = "OK"
        out["reason"] = "local_escape_hub_segment_prevalidated_with_tolerance"
        out["decision"] = "ALLOW_BORDERLINE"
    return out


def validate_attached_direct_action_segment(
    start_joints: Sequence[float],
    end_joints: Sequence[float],
    *,
    segment_name: str,
    fk_hand_fn: Callable[[Any], Optional[Tuple[float, float, float]]],
    attached_geom: Dict[str, Any],
    scene_obstacles: Sequence[Dict[str, Any]],
    table_top_z: float,
    policy: Dict[str, Any],
) -> Tuple[bool, Dict[str, Any]]:
    min_table = float(policy.get("carry_clearance_above_table_m", 0.12)) * 0.5
    req_xy = float(
        policy.get(
            "global_route_required_clearance_m",
            policy.get("carry_clearance_above_obstacles_m", 0.10),
        )
    )
    tol = float(
        policy.get(
            "attached_transport_safety_margin_tolerance_m",
            DEFAULT_ATTACHED_TRANSPORT_SAFETY_MARGIN_TOLERANCE_M,
        )
    )
    seg_ok, metrics, checks = validate_attached_joint_segment(
        start_joints,
        end_joints,
        fk_hand_fn=fk_hand_fn,
        attached_geom=attached_geom,
        table_top_z=float(table_top_z),
        obstacles=scene_obstacles,
        min_table_clearance_m=min_table,
        required_xy_clearance_m=req_xy,
        safety_margin_tolerance_m=tol,
    )
    decision = decide_attached_transport_preflight(
        seg_ok, metrics, checks, tolerance_m=tol
    )
    ok = decision.get("decision") in ("OK", "ALLOW_BORDERLINE")
    j_margin = min(
        joint_limit_margin_min(start_joints),
        joint_limit_margin_min(end_joints),
    )
    return ok, {
        "segment": segment_name,
        "hard_collision": decision.get("hard_collision", False),
        "min_clearance": metrics.get("min_safety_margin_m"),
        "joint_margin_min": j_margin,
        "decision": decision.get("decision"),
        "result": "OK" if ok else "FAIL",
    }


def validate_attached_direct_action_route(
    *,
    current_joints: Sequence[float],
    waypoint_names: Sequence[str],
    waypoint_joints: Sequence[Sequence[float]],
    fk_hand_fn: Callable[[Any], Optional[Tuple[float, float, float]]],
    attached_geom: Dict[str, Any],
    scene_obstacles: Sequence[Dict[str, Any]],
    table_top_z: float,
    policy: Dict[str, Any],
    hub_segment_prevalidated: bool = False,
    transport_entry_verified: bool = False,
    obstacle_disturbed: bool = False,
    hub_min_clearance_floor_m: float = HUB_SEGMENT_PREVALIDATED_MIN_CLEARANCE_FLOOR_M,
) -> Tuple[bool, List[Dict[str, Any]]]:
    logs: List[Dict[str, Any]] = []
    if not waypoint_names:
        return True, logs
    prev = list(current_joints)
    all_ok = True
    for idx, (name, target) in enumerate(zip(waypoint_names, waypoint_joints)):
        seg_name = "current->%s" if idx == 0 else "%s->%s" % (
            waypoint_names[idx - 1],
            name,
        )
        if idx == 0:
            seg_name = "current->%s" % name
        ok, detail = validate_attached_direct_action_segment(
            prev,
            target,
            segment_name=seg_name,
            fk_hand_fn=fk_hand_fn,
            attached_geom=attached_geom,
            scene_obstacles=scene_obstacles,
            table_top_z=table_top_z,
            policy=policy,
        )
        detail = maybe_accept_prevalidated_hub_segment(
            detail,
            is_first_segment=(idx == 0),
            hub_segment_prevalidated=hub_segment_prevalidated,
            transport_entry_verified=transport_entry_verified,
            obstacle_disturbed=obstacle_disturbed,
            min_clearance_floor_m=hub_min_clearance_floor_m,
        )
        detail["waypoint"] = name
        ok = str(detail.get("result", "FAIL")).upper() == "OK"
        logs.append(detail)
        if not ok:
            all_ok = False
        prev = list(target)
    return all_ok, logs


def format_attached_direct_action_route_validate_log(
    sequence: Sequence[str], result: str
) -> str:
    return (
        "[ATTACHED_DIRECT_ACTION_ROUTE_VALIDATE]\n"
        "sequence=%s\n"
        "object_attached=true\n"
        "result=%s"
        % (list(sequence), result)
    )


def format_attached_direct_action_segment_validate_log(detail: Dict[str, Any]) -> str:
    min_clr = detail.get("min_clearance", "n/a")
    if isinstance(min_clr, float):
        min_clr = "%.6f" % float(min_clr)
    return (
        "[ATTACHED_DIRECT_ACTION_SEGMENT_VALIDATE]\n"
        "segment=%s\n"
        "waypoint=%s\n"
        "hard_collision=%s\n"
        "min_clearance=%s\n"
        "joint_margin_min=%s\n"
        "result=%s\n"
        "reason=%s"
        % (
            str(detail.get("segment", "")),
            str(detail.get("waypoint", "")),
            str(bool(detail.get("hard_collision"))).lower(),
            str(min_clr),
            detail.get("joint_margin_min", "n/a"),
            str(detail.get("result", "FAIL")),
            str(detail.get("reason", "n/a")),
        )
    )


def resolve_deterministic_transport_sequence(
    scene_policy: Optional[Dict[str, Any]],
    *,
    reconfiguration_zone_ok: bool,
    obstacles_remaining: bool,
    default_route: Sequence[str],
) -> Tuple[List[str], List[str]]:
    """Devuelve (execute_sequence, reconfiguration_prefix)."""
    tp = (scene_policy or {}).get("transport_policy") or {}
    phases = (scene_policy or {}).get("transport_phases") or {}
    reconfig_cfg = phases.get("reconfiguration") or {}
    global_cfg = phases.get("global_transport") or {}

    reconfig = list(
        reconfig_cfg.get("waypoints") or tp.get("reconfiguration_waypoints") or []
    )
    global_route = list(global_cfg.get("waypoints") or [])
    policy_route = list(tp.get("transport_route") or default_route)
    forbidden = list(tp.get("forbidden_waypoints_when_obstacles_remaining") or [])
    if obstacles_remaining and forbidden:
        forbidden_set = set(forbidden)
        policy_route = [wp for wp in policy_route if wp not in forbidden_set]
        global_route = [wp for wp in global_route if wp not in forbidden_set]
    prefix: List[str] = []
    if reconfiguration_zone_ok and reconfig:
        prefix = list(reconfig)
        route = global_route or policy_route
    else:
        route = policy_route or global_route
    execute: List[str] = []
    seen: set = set()
    for wp in prefix + route:
        wp_name = str(wp).strip()
        if not wp_name or wp_name in seen:
            continue
        seen.add(wp_name)
        execute.append(wp_name)
    return execute, prefix


def score_transport_aware_pick_exit(
    *,
    candidate_idx: int,
    yaw_variant: float,
    post_lift_hand: Tuple[float, float, float],
    post_lift_joints: Sequence[float],
    grasp_joints: Sequence[float],
    hub_waypoint_joints: Sequence[float],
    hub_waypoint_name: str,
    fk_hand_fn: Callable[[Any], Optional[Tuple[float, float, float]]],
    attached_geom: Dict[str, Any],
    scene_obstacles: Sequence[Dict[str, Any]],
    table_top_z: float,
    policy: Dict[str, Any],
    scene_policy: Optional[Dict[str, Any]],
    pick_plan_ok: bool = True,
    grasp_ok: bool = True,
    lift_ok: bool = True,
    hand_z_candidates: Optional[Sequence[float]] = None,
    allow_direct_to_entry_target: Optional[bool] = None,
    allow_carry_front_high_corridors: Optional[bool] = None,
    carried_label: str = "",
    lift_start_state_source: str = "post_lift_endpoint",
    target_world_present: bool = False,
    transport_exit_log_fn: Optional[Callable[[str], None]] = None,
) -> Dict[str, Any]:
    """Score virtual post-lift exit hacia transporte (sin mover robot)."""
    tp = (scene_policy or {}).get("transport_policy") or {}
    label = str(carried_label or policy.get("_label", ""))
    obstacles_remaining = has_remaining_table_obstacles(
        scene_obstacles, target_label=label
    )
    obstacle_labels = [
        str(o.get("label", "")).strip()
        for o in scene_obstacles
        if isinstance(o, dict) and str(o.get("label", "")).strip()
    ]
    if allow_direct_to_entry_target is None:
        forbidden = list(tp.get("forbidden_waypoints_when_obstacles_remaining") or [])
        if str(hub_waypoint_name) == "carry_front_high":
            allow_direct_to_entry_target = not (
                obstacles_remaining and "carry_front_high" in forbidden
            )
        else:
            allow_direct_to_entry_target = True
    if allow_carry_front_high_corridors is None:
        forbidden = list(tp.get("forbidden_waypoints_when_obstacles_remaining") or [])
        allow_carry_front_high_corridors = not (
            obstacles_remaining and "carry_front_high" in forbidden
        )
    selected, validation_logs = select_transport_entry_validate_only(
        post_lift_hand=post_lift_hand,
        post_lift_joints=post_lift_joints,
        entry_target_joints=hub_waypoint_joints,
        entry_target_waypoint=hub_waypoint_name,
        allow_direct_to_carry_front_high=bool(allow_carry_front_high_corridors),
        allow_direct_to_entry_target=allow_direct_to_entry_target,
        allow_carry_front_high_corridors=allow_carry_front_high_corridors,
        fk_hand_fn=fk_hand_fn,
        attached_geom=attached_geom,
        scene_obstacles=scene_obstacles,
        table_top_z=float(table_top_z),
        policy=policy,
        hand_z_candidates=list(hand_z_candidates or [post_lift_hand[2]]),
        scene_policy=scene_policy,
    )
    phase_clr = resolve_transport_phase_clearance_thresholds(scene_policy, policy)
    emit_transport_exit_candidate_validate_logs(
        validation_logs,
        candidate_idx=int(candidate_idx),
        carried_label=label,
        obstacles_remaining=obstacle_labels,
        target_world_present=bool(target_world_present),
        start_state_source=str(lift_start_state_source or "post_lift_endpoint"),
        attached_geom=attached_geom,
        local_exit_clearance_m=float(phase_clr["local_exit_required_clearance_m"]),
        global_route_clearance_m=float(phase_clr["global_route_required_clearance_m"]),
        log_fn=transport_exit_log_fn,
    )
    rear_ok = selected is not None and "rear_retreat" in str(
        selected.get("mode", "")
    )
    transport_entry_ok = selected is not None
    selected_local_exit = str(selected.get("mode", "")) if selected else ""
    local_escape_ok = False
    hard_collision_3d = False
    if selected is not None:
        local_escape_ok = bool(
            selected.get("local_escape_ok")
            or (selected.get("metrics") or {}).get("local_escape_ok")
        )
        if not local_escape_ok:
            local_escape_ok = transport_entry_ok
        sweep_dbg = dict((selected.get("metrics") or {}).get("sweep_debug") or {})
        hard_collision_3d = bool(sweep_dbg.get("hard_collision_3d", False))
        if not hard_collision_3d:
            hard_collision_3d = bool((selected.get("decision") or {}).get("hard_collision"))
    reconfig_safety = resolve_reconfiguration_safety_thresholds(scene_policy)
    zone_hand = post_lift_hand
    if selected is not None and selected.get("candidate_hand") is not None:
        ch = selected["candidate_hand"]
        zone_hand = (float(ch[0]), float(ch[1]), float(ch[2]))
    elif selected is not None and bool(selected.get("need_raise")):
        zone_hand = (
            float(post_lift_hand[0]),
            float(post_lift_hand[1]),
            float(selected.get("hand_z", post_lift_hand[2])),
        )
    zone = check_transport_reconfiguration_zone(
        hand_pos=zone_hand,
        attached_geom=attached_geom,
        scene_obstacles=scene_obstacles,
        table_top_z=float(table_top_z),
        policy=policy,
        min_table_clearance_m=float(reconfig_safety["min_table_clearance_m"]),
        min_xy_clearance_m=float(reconfig_safety["min_xy_clearance_m"]),
    )
    route_seq, _prefix = resolve_deterministic_transport_sequence(
        scene_policy,
        reconfiguration_zone_ok=bool(zone.get("transport_reconfiguration_zone_ok")),
        obstacles_remaining=obstacles_remaining,
        default_route=[hub_waypoint_name],
    )
    resolve_wp = (scene_policy or {}).get("_resolve_waypoint_joints")
    route_names: List[str] = []
    route_joints: List[List[float]] = []
    for wp in route_seq:
        js: Optional[List[float]] = None
        if callable(resolve_wp):
            js = resolve_wp(str(wp))
        if js is None and str(wp) == hub_waypoint_name:
            js = list(hub_waypoint_joints)
        if js is None:
            continue
        route_names.append(str(wp))
        route_joints.append(list(js))
    if not route_names:
        route_names = [hub_waypoint_name]
        route_joints = [list(hub_waypoint_joints)]
    global_route_names = list(route_names)
    global_route_joints = list(route_joints)
    if (
        global_route_names
        and str(global_route_names[0]) == str(hub_waypoint_name)
        and transport_entry_ok
    ):
        global_route_names = global_route_names[1:]
        global_route_joints = global_route_joints[1:]
    if not global_route_names:
        direct_ok = bool(transport_entry_ok and zone.get("transport_reconfiguration_zone_ok"))
        _seg_logs: List[Dict[str, Any]] = []
    else:
        direct_ok, _seg_logs = validate_attached_direct_action_route(
            current_joints=list(hub_waypoint_joints),
            waypoint_names=global_route_names,
            waypoint_joints=global_route_joints,
            fk_hand_fn=fk_hand_fn,
            attached_geom=attached_geom,
            scene_obstacles=scene_obstacles,
            table_top_z=float(table_top_z),
            policy=policy,
        )
    j_dist = joint_distance_rad(post_lift_joints, hub_waypoint_joints)
    j_margin = joint_limit_margin_min(post_lift_joints)
    wrist = wrist_twist_score(post_lift_joints)
    elbow = elbow_posture_score(post_lift_joints)
    reconfiguration_zone_ok = bool(zone.get("transport_reconfiguration_zone_ok"))
    global_route_ok = bool(direct_ok)
    full_transport_ok = bool(
        transport_entry_ok and reconfiguration_zone_ok and global_route_ok
    )
    local_escape_deferred_ok = bool(
        local_escape_ok and not hard_collision_3d and not target_world_present
    )
    accept = bool(
        pick_plan_ok
        and grasp_ok
        and lift_ok
        and j_margin > -0.05
        and (full_transport_ok or local_escape_deferred_ok)
    )
    if accept:
        if full_transport_ok:
            acceptance_reason = "full_transport_ok"
        else:
            acceptance_reason = "valid_pick_with_local_escape_global_route_deferred"
    else:
        acceptance_reason = ""
    acceptance_blocker = "none"
    if not accept:
        if not pick_plan_ok:
            acceptance_blocker = "pick_plan_fail"
        elif not grasp_ok:
            acceptance_blocker = "grasp_fail"
        elif not lift_ok:
            acceptance_blocker = "lift_fail"
        elif not local_escape_ok:
            acceptance_blocker = "no_local_escape"
        elif hard_collision_3d:
            acceptance_blocker = "hard_collision_3d"
        elif target_world_present:
            acceptance_blocker = "target_world_collision_present"
        elif not global_route_ok:
            acceptance_blocker = "global_route_not_validated"
        elif not reconfiguration_zone_ok:
            acceptance_blocker = "reconfiguration_zone_not_reached"
        elif j_margin <= -0.05:
            acceptance_blocker = "joint_limit_margin"
        else:
            acceptance_blocker = "transport_reject"
    post_lift_exit_ok = bool(local_escape_ok)
    from panda_controller.demo_scene_global_sequence_validate import (
        emit_demo_scene_object_sequence_validate_log,
    )

    emit_demo_scene_object_sequence_validate_log(
        scene_policy=scene_policy,
        target_label=label,
        carry_policy=policy,
        transport_score={
            "transport_entry_possible": local_escape_ok,
            "reconfiguration_zone_ok": reconfiguration_zone_ok,
            "direct_action_to_hub_ok": global_route_ok,
            "result": "ACCEPT" if accept else "REJECT",
            "selected_transport_mode": selected_local_exit,
        },
        log_fn=transport_exit_log_fn,
    )
    return {
        "candidate_idx": int(candidate_idx),
        "yaw_variant": float(yaw_variant),
        "pick_plan_ok": bool(pick_plan_ok),
        "grasp_ok": bool(grasp_ok),
        "lift_ok": bool(lift_ok),
        "rear_retreat_possible": rear_ok,
        "transport_entry_possible": local_escape_ok,
        "local_escape_ok": local_escape_ok,
        "selected_local_exit": selected_local_exit,
        "hard_collision_3d": hard_collision_3d,
        "post_lift_exit_ok": post_lift_exit_ok,
        "direct_action_to_hub_ok": global_route_ok,
        "global_route_ok": global_route_ok,
        "reconfiguration_zone_ok": reconfiguration_zone_ok,
        "joint_distance_to_hub": j_dist,
        "joint_margin_min": j_margin,
        "wrist_twist_score": wrist,
        "elbow_score": elbow,
        "selected_transport_mode": selected_local_exit,
        "acceptance_reason": acceptance_reason,
        "acceptance_blocker": acceptance_blocker,
        "result": "ACCEPT" if accept else "REJECT",
    }


def format_pick_route_transport_aware_score_log(score: Dict[str, Any]) -> str:
    return (
        "[PICK_ROUTE_TRANSPORT_AWARE_SCORE]\n"
        "candidate_idx=%s\n"
        "yaw_variant=%s\n"
        "pick_plan_ok=%s\n"
        "local_escape_ok=%s\n"
        "selected_local_exit=%s\n"
        "reconfiguration_zone_ok=%s\n"
        "global_route_ok=%s\n"
        "transport_entry_possible=%s\n"
        "post_lift_exit_ok=%s\n"
        "direct_action_to_hub_ok=%s\n"
        "joint_distance_to_hub=%.4f\n"
        "joint_margin_min=%.4f\n"
        "wrist_twist_score=%.4f\n"
        "elbow_score=%.4f\n"
        "selected_transport_mode=%s\n"
        "acceptance_blocker=%s\n"
        "acceptance_reason=%s\n"
        "result=%s"
        % (
            score.get("candidate_idx", "n/a"),
            score.get("yaw_variant", "n/a"),
            str(bool(score.get("pick_plan_ok"))).lower(),
            str(bool(score.get("local_escape_ok", score.get("post_lift_exit_ok")))).lower(),
            str(score.get("selected_local_exit", score.get("selected_transport_mode", ""))),
            str(bool(score.get("reconfiguration_zone_ok"))).lower(),
            str(bool(score.get("global_route_ok", score.get("direct_action_to_hub_ok")))).lower(),
            str(bool(score.get("transport_entry_possible"))).lower(),
            str(bool(score.get("post_lift_exit_ok"))).lower(),
            str(bool(score.get("direct_action_to_hub_ok"))).lower(),
            float(score.get("joint_distance_to_hub", 0.0)),
            float(score.get("joint_margin_min", 0.0)),
            float(score.get("wrist_twist_score", 0.0)),
            float(score.get("elbow_score", 0.0)),
            str(score.get("selected_transport_mode", "")),
            str(score.get("acceptance_blocker", "n/a")),
            str(score.get("acceptance_reason", "n/a")),
            str(score.get("result", "REJECT")),
        )
    )


def format_pick_route_transport_aware_selected_log(
    *,
    candidate_idx: int,
    selected_local_exit: str,
    reason: str,
    yaw_variant: Any = "n/a",
    joint_distance_to_hub: float = 0.0,
) -> str:
    return (
        "[PICK_ROUTE_TRANSPORT_AWARE_SELECTED]\n"
        "candidate_idx=%s\n"
        "yaw_variant=%s\n"
        "selected_local_exit=%s\n"
        "joint_distance_to_hub=%.4f\n"
        "reason=%s\n"
        "result=SELECTED"
        % (
            int(candidate_idx),
            str(yaw_variant),
            str(selected_local_exit or "n/a"),
            float(joint_distance_to_hub),
            str(reason or "n/a"),
        )
    )


def pick_best_transport_aware_score(
    scores: Sequence[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    accepted = [s for s in scores if s.get("result") == "ACCEPT"]
    if not accepted:
        return None
    return min(
        accepted,
        key=lambda s: (
            float(s.get("joint_distance_to_hub", 1e9)),
            float(s.get("wrist_twist_score", 1e9)),
            float(s.get("elbow_score", 1e9)),
            -float(s.get("joint_margin_min", -1e9)),
        ),
    )
