"""Validación virtual de transport_entry con objeto attached (sin exploración física)."""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from panda_controller.generic_known_scene_carry_planner import (
    DEFAULT_ATTACHED_TRANSPORT_SAFETY_MARGIN_TOLERANCE_M,
    generate_transport_entry_candidates,
    validate_attached_hand_pose,
    validate_attached_joint_segment,
)


def validate_attached_cartesian_hand_segment(
    hand_start: Tuple[float, float, float],
    hand_end: Tuple[float, float, float],
    attached_geom: Dict[str, Any],
    obstacles: Sequence[Dict[str, Any]],
    *,
    table_top_z: float,
    min_table_clearance_m: float,
    required_xy_clearance_m: float,
    safety_margin_tolerance_m: float = 0.0,
    n_samples: int = 12,
    obstacle_margin_mode: str = "3d",
) -> Tuple[bool, Dict[str, Any], List[Dict[str, Any]]]:
    all_checks: List[Dict[str, Any]] = []
    min_table_clr = float("inf")
    min_geom_xy = float("inf")
    min_safety_margin = float("inf")
    min_bottom = float("inf")
    fail_metrics: Optional[Dict[str, Any]] = None
    local_xy_mode = str(obstacle_margin_mode) == "xy_unless_hard_collision"
    for i in range(max(2, int(n_samples))):
        t = float(i) / float(max(1, n_samples - 1))
        hand = (
            float(hand_start[0]) + t * (float(hand_end[0]) - float(hand_start[0])),
            float(hand_start[1]) + t * (float(hand_end[1]) - float(hand_start[1])),
            float(hand_start[2]) + t * (float(hand_end[2]) - float(hand_start[2])),
        )
        ok, checks, metrics = validate_attached_hand_pose(
            hand,
            attached_geom,
            obstacles,
            table_top_z=table_top_z,
            min_table_clearance_m=min_table_clearance_m,
            required_xy_clearance_m=required_xy_clearance_m,
            safety_margin_tolerance_m=float(safety_margin_tolerance_m),
            obstacle_margin_mode=str(obstacle_margin_mode),
        )
        all_checks.extend(checks)
        if local_xy_mode:
            if metrics.get("min_geometric_xy_clearance_m") is not None:
                min_geom_xy = min(
                    min_geom_xy, float(metrics["min_geometric_xy_clearance_m"])
                )
            sample_margin = metrics.get("min_safety_margin_m")
            if sample_margin is None or sample_margin == float("inf"):
                sample_margin = metrics.get("min_geometric_xy_clearance_m")
            if sample_margin is not None and sample_margin != float("inf"):
                min_safety_margin = min(min_safety_margin, float(sample_margin))
        else:
            for chk in checks:
                if chk.get("result") == "SKIP":
                    continue
                min_geom_xy = min(min_geom_xy, float(chk.get("xy_clearance", 0.0)))
                if bool(chk.get("xy_overlap")) or bool(chk.get("z_overlap")) or chk.get(
                    "result"
                ) in ("NEAR", "COLLISION"):
                    margin = min(
                        float(chk.get("xy_clearance", 0.0)),
                        float(chk.get("z_clearance", 0.0)),
                    )
                    min_safety_margin = min(min_safety_margin, margin)
        if metrics.get("min_attached_object_bottom_z") is not None:
            min_bottom = min(
                min_bottom, float(metrics["min_attached_object_bottom_z"])
            )
        if metrics.get("min_clearance_to_table") is not None:
            min_table_clr = min(min_table_clr, float(metrics["min_clearance_to_table"]))
        if not ok:
            fail_metrics = dict(metrics)
            fail_metrics["sample"] = i
            fail_metrics["hand_pos"] = hand
            fail_metrics["min_geometric_xy_clearance_m"] = min_geom_xy
            fail_metrics["min_safety_margin_m"] = min_safety_margin
            fail_metrics["min_clearance_to_obstacles"] = min_safety_margin
    ok_metrics = {
        "min_attached_object_bottom_z": min_bottom,
        "min_clearance_to_table": min_table_clr,
        "min_geometric_xy_clearance_m": min_geom_xy,
        "min_safety_margin_m": min_safety_margin,
        "min_clearance_to_obstacles": min_safety_margin,
    }
    if fail_metrics is not None:
        decision = decide_attached_transport_preflight(
            False,
            fail_metrics,
            all_checks,
            tolerance_m=safety_margin_tolerance_m,
            clearance_mode=str(obstacle_margin_mode),
        )
        if decision.get("decision") in ("OK", "ALLOW_BORDERLINE"):
            return True, ok_metrics, all_checks
        return False, fail_metrics, all_checks
    decision = decide_attached_transport_preflight(
        True,
        ok_metrics,
        all_checks,
        tolerance_m=safety_margin_tolerance_m,
        clearance_mode=str(obstacle_margin_mode),
    )
    if decision.get("decision") in ("OK", "ALLOW_BORDERLINE"):
        return True, ok_metrics, all_checks
    return False, {**ok_metrics, "reason": decision.get("reason", "segment_not_clear")}, all_checks


def decide_attached_transport_preflight(
    swept_ok: bool,
    metrics: Dict[str, Any],
    checks: Sequence[Dict[str, Any]],
    *,
    tolerance_m: float = DEFAULT_ATTACHED_TRANSPORT_SAFETY_MARGIN_TOLERANCE_M,
    clearance_mode: str = "3d",
) -> Dict[str, Any]:
    hard = any(bool(c.get("hard_collision")) for c in checks)
    min_geom = metrics.get("min_geometric_xy_clearance_m")
    min_margin = metrics.get("min_safety_margin_m")
    if min_margin is None or min_margin == float("inf"):
        min_margin = metrics.get("min_clearance_to_obstacles")
    try:
        min_geom_f = float(min_geom) if min_geom is not None else float("inf")
    except (TypeError, ValueError):
        min_geom_f = float("inf")
    try:
        min_margin_f = float(min_margin) if min_margin is not None else float("inf")
    except (TypeError, ValueError):
        min_margin_f = float("inf")
    tol = float(tolerance_m)
    mode = str(clearance_mode or "3d")
    diag_geom_xy_overlap = bool(
        min_geom_f < 0.0 and any(bool(c.get("xy_overlap")) for c in checks)
    )
    geom_overlap = any(
        bool(c.get("xy_overlap")) and bool(c.get("z_overlap")) for c in checks
    )
    if mode == "local_escape_post_lift":
        margin_ok = not any(
            bool(c.get("hard_collision"))
            or (
                bool(c.get("xy_overlap"))
                and bool(c.get("z_overlap"))
                and float(c.get("z_clearance", 0.0)) < -tol
            )
            for c in checks
            if c.get("result") != "SKIP"
        )
        if min_margin_f == float("inf"):
            min_margin_f = min_geom_f
        if not margin_ok and min_margin_f >= -tol:
            margin_ok = True
        reason = "ok"
        decision = "OK"
        if hard:
            decision = "FAIL"
            reason = "hard_collision_3d"
        elif str(metrics.get("reason", "")) == "attached_bottom_near_table":
            decision = "FAIL"
            reason = "attached_bottom_near_table"
        elif not swept_ok and not margin_ok and min_margin_f < -tol:
            decision = "FAIL"
            reason = str(metrics.get("reason", "safety_margin_insufficient"))
        elif not swept_ok and str(metrics.get("reason", "")) not in ("", "ok"):
            if str(metrics.get("reason", "")) == "near_obstacle_margin" and margin_ok:
                decision = "ALLOW_BORDERLINE"
                reason = "within_safety_margin_tolerance"
            else:
                decision = "FAIL"
                reason = str(metrics.get("reason", "segment_not_clear"))
        elif min_margin_f < 0.0 and min_margin_f >= -tol:
            decision = "ALLOW_BORDERLINE"
            reason = "within_safety_margin_tolerance"
        return {
            "hard_collision": hard,
            "min_clearance_to_obstacles": min_margin_f,
            "min_geometric_xy_clearance_m": min_geom_f,
            "min_safety_margin_m": min_margin_f,
            "tolerance_m": tol,
            "safety_margin_ok": margin_ok,
            "decision": decision,
            "reason": reason,
            "diagnostic_geometric_xy_overlap": diag_geom_xy_overlap,
        }
    if mode == "xy_unless_hard_collision":
        margin_ok = not any(
            bool(c.get("hard_collision"))
            or (
                bool(c.get("xy_overlap"))
                and float(c.get("xy_clearance", 0.0)) + tol + 1e-6 < 0.0
            )
            for c in checks
            if c.get("result") != "SKIP"
        )
        if min_margin_f == float("inf"):
            min_margin_f = min_geom_f
    else:
        margin_ok = all(
            bool(c.get("safety_margin_ok", True)) for c in checks if c.get("result") != "SKIP"
        )
    if not margin_ok and min_margin_f >= -tol:
        margin_ok = True
    reason = "ok"
    decision = "OK"
    if mode == "xy_unless_hard_collision":
        if hard:
            decision = "FAIL"
            reason = "hard_collision_3d"
        elif min_margin_f < -tol:
            decision = "FAIL"
            reason = "safety_margin_insufficient"
        elif not swept_ok and min_margin_f < -tol:
            decision = "FAIL"
            reason = str(metrics.get("reason", "segment_not_clear"))
        elif min_margin_f < 0.0 and min_margin_f >= -tol:
            decision = "ALLOW_BORDERLINE"
            reason = "within_safety_margin_tolerance"
        elif not margin_ok and min_margin_f >= -tol:
            decision = "ALLOW_BORDERLINE"
            reason = "within_safety_margin_tolerance"
        elif not swept_ok:
            decision = "FAIL"
            reason = str(metrics.get("reason", "segment_not_clear"))
        return {
            "hard_collision": hard,
            "min_clearance_to_obstacles": min_margin_f,
            "min_geometric_xy_clearance_m": min_geom_f,
            "min_safety_margin_m": min_margin_f,
            "tolerance_m": tol,
            "safety_margin_ok": margin_ok,
            "decision": decision,
            "reason": reason,
            "diagnostic_geometric_xy_overlap": diag_geom_xy_overlap,
        }
    if hard or geom_overlap:
        decision = "FAIL"
        reason = "hard_collision"
    elif min_geom_f < 0.0 and any(bool(c.get("xy_overlap")) for c in checks):
        decision = "FAIL"
        reason = "geometric_xy_overlap"
    elif min_margin_f < -tol:
        decision = "FAIL"
        reason = "safety_margin_insufficient"
    elif not swept_ok and min_margin_f < -tol:
        decision = "FAIL"
        reason = str(metrics.get("reason", "segment_not_clear"))
    elif min_margin_f < 0.0 and min_margin_f >= -tol:
        decision = "ALLOW_BORDERLINE"
        reason = "within_safety_margin_tolerance"
    elif not margin_ok and min_margin_f >= -tol:
        decision = "ALLOW_BORDERLINE"
        reason = "within_safety_margin_tolerance"
    elif not swept_ok:
        decision = "FAIL"
        reason = str(metrics.get("reason", "segment_not_clear"))
    return {
        "hard_collision": hard,
        "min_clearance_to_obstacles": min_margin_f,
        "min_geometric_xy_clearance_m": min_geom_f,
        "min_safety_margin_m": min_margin_f,
        "safety_margin_ok": margin_ok,
        "tolerance_m": tol,
        "decision": decision,
        "reason": reason,
    }


def extract_segment_fail_detail(
    *,
    candidate_mode: str,
    segment: str,
    metrics: Dict[str, Any],
    checks: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    blocking = ""
    min_clr = float("inf")
    hard = False
    for chk in checks:
        if chk.get("result") in ("NEAR", "COLLISION") or bool(chk.get("hard_collision")):
            m = min(float(chk.get("xy_clearance", 0.0)), float(chk.get("z_clearance", 0.0)))
            if m < min_clr:
                min_clr = m
                blocking = str(chk.get("obstacle_label", ""))
            if bool(chk.get("hard_collision")):
                hard = True
    hand = metrics.get("hand_pos") or (0.0, 0.0, 0.0)
    below = metrics.get("min_attached_object_bottom_z")
    if below is None and isinstance(hand, (list, tuple)) and len(hand) >= 3:
        below = float(hand[2]) - 0.19
    return {
        "candidate_mode": candidate_mode,
        "segment": segment,
        "blocking_obstacle": blocking or "n/a",
        "min_clearance": min_clr if min_clr != float("inf") else metrics.get("min_safety_margin_m"),
        "hard_collision": hard,
        "sample_idx": metrics.get("sample", "n/a"),
        "sample_hand": hand,
        "sample_attached_bottom_z": below,
        "reason": metrics.get("reason", "n/a"),
    }


def format_transport_entry_segment_fail_detail_log(detail: Dict[str, Any]) -> str:
    hand = detail.get("sample_hand") or (0.0, 0.0, 0.0)
    hand_str = (
        "n/a"
        if not isinstance(hand, (list, tuple))
        else "(%.3f, %.3f, %.3f)" % (float(hand[0]), float(hand[1]), float(hand[2]))
    )
    return (
        "[TRANSPORT_ENTRY_SEGMENT_FAIL_DETAIL]\n"
        "candidate_mode=%s\n"
        "segment=%s\n"
        "blocking_obstacle=%s\n"
        "min_clearance=%s\n"
        "hard_collision=%s\n"
        "sample_idx=%s\n"
        "sample_hand=%s\n"
        "sample_attached_bottom_z=%s\n"
        "reason=%s"
        % (
            str(detail.get("candidate_mode", "")),
            str(detail.get("segment", "")),
            str(detail.get("blocking_obstacle", "")),
            detail.get("min_clearance", "n/a"),
            str(bool(detail.get("hard_collision"))).lower(),
            str(detail.get("sample_idx", "n/a")),
            hand_str,
            detail.get("sample_attached_bottom_z", "n/a"),
            str(detail.get("reason", "")),
        )
    )


def format_attached_transport_preflight_decision_log(decision: Dict[str, Any]) -> str:
    return (
        "[ATTACHED_TRANSPORT_PREFLIGHT_DECISION]\n"
        "hard_collision=%s\n"
        "min_geometric_xy_clearance_m=%s\n"
        "min_safety_margin_m=%s\n"
        "tolerance_m=%s\n"
        "safety_margin_ok=%s\n"
        "decision=%s\n"
        "reason=%s"
        % (
            str(bool(decision.get("hard_collision"))).lower(),
            decision.get("min_geometric_xy_clearance_m", "n/a"),
            decision.get("min_safety_margin_m", "n/a"),
            decision.get("tolerance_m", "n/a"),
            str(bool(decision.get("safety_margin_ok"))).lower(),
            str(decision.get("decision", "")),
            str(decision.get("reason", "")),
        )
    )


def format_transport_exit_candidate_validate_log(
    *,
    candidate_idx: int,
    exit_name: str,
    start_state_source: str = "post_lift_endpoint",
    attached_object: bool = True,
    carried_label: str = "",
    obstacles_remaining: Sequence[str] = (),
    target_world_present: bool = False,
    fraction: str = "n/a",
    plan_ok: Any = False,
    plan_checked: Optional[bool] = None,
    geom_ok: Optional[bool] = None,
    collision_ok: bool = False,
    min_obstacle_distance: str = "n/a",
    table_clearance: str = "n/a",
    result: str = "FAIL",
    reason: str = "",
) -> str:
    obs_repr = "[%s]" % ", ".join(str(o) for o in obstacles_remaining)
    if isinstance(plan_ok, str):
        plan_ok_repr = str(plan_ok)
    else:
        plan_ok_repr = str(bool(plan_ok)).lower()
    geom_repr = "n/a" if geom_ok is None else str(bool(geom_ok)).lower()
    plan_checked_repr = (
        "n/a" if plan_checked is None else str(bool(plan_checked)).lower()
    )
    return (
        "[TRANSPORT_EXIT_CANDIDATE_VALIDATE]\n"
        "candidate_idx=%d\n"
        "exit_name=%s\n"
        "start_state_source=%s\n"
        "attached_object=%s\n"
        "carried_label=%s\n"
        "obstacles_remaining=%s\n"
        "target_world_present=%s\n"
        "fraction=%s\n"
        "geom_ok=%s\n"
        "plan_checked=%s\n"
        "plan_ok=%s\n"
        "collision_ok=%s\n"
        "min_obstacle_distance=%s\n"
        "table_clearance=%s\n"
        "result=%s\n"
        "reason=%s"
        % (
            int(candidate_idx),
            str(exit_name),
            str(start_state_source),
            str(bool(attached_object)).lower(),
            str(carried_label or "n/a"),
            obs_repr,
            str(bool(target_world_present)).lower(),
            str(fraction),
            geom_repr,
            plan_checked_repr,
            plan_ok_repr,
            str(bool(collision_ok)).lower(),
            str(min_obstacle_distance),
            str(table_clearance),
            str(result),
            str(reason or "n/a"),
        )
    )


def format_transport_exit_clearance_breakdown_log(
    *,
    candidate_idx: int,
    exit_name: str,
    closest_check: Optional[Dict[str, Any]],
    required_clearance: float,
    attached_geom: Dict[str, Any],
    reason: str = "",
) -> str:
    chk = dict(closest_check or {})
    pad = attached_geom.get("attached_collision_padding_m", "n/a")
    if isinstance(pad, float):
        pad = "%.4f" % float(pad)
    carried_fp = attached_geom.get("dims_lwh")
    if carried_fp is None:
        carried_fp = "radius_xy=%.4f" % float(
            attached_geom.get("carried_object_radius_xy_m", 0.0)
        )
    min_dist = chk.get("xy_clearance", "n/a")
    if isinstance(min_dist, float):
        min_dist = "%.4f" % float(min_dist)
    return (
        "[TRANSPORT_EXIT_CLEARANCE_BREAKDOWN]\n"
        "candidate_idx=%d\n"
        "exit_name=%s\n"
        "closest_obstacle_label=%s\n"
        "closest_obstacle_entity=%s\n"
        "min_obstacle_distance=%s\n"
        "required_clearance=%.4f\n"
        "attached_object_padding=%s\n"
        "carried_object_footprint=%s\n"
        "obstacle_footprint=%s\n"
        "reason=%s"
        % (
            int(candidate_idx),
            str(exit_name),
            str(chk.get("obstacle_label", "n/a")),
            str(chk.get("obstacle_entity", "n/a")),
            str(min_dist),
            float(required_clearance),
            str(pad),
            str(carried_fp),
            str(chk.get("obstacle_dims", "n/a")),
            str(reason or chk.get("result", "n/a")),
        )
    )


def closest_obstacle_check_from_segment_checks(
    checks: Sequence[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    best: Optional[Dict[str, Any]] = None
    best_xy = float("inf")
    for chk in checks:
        if str(chk.get("result", "")).upper() == "SKIP":
            continue
        try:
            xy = float(chk.get("xy_clearance", float("inf")))
        except (TypeError, ValueError):
            continue
        if xy < best_xy:
            best_xy = xy
            best = dict(chk)
    return best


def format_local_escape_decision_log(
    *,
    candidate_idx: int,
    exit_name: str,
    sweep_result: str,
    hard_collision_3d: bool,
    old_geometric_xy_overlap: bool,
    old_breakdown_reason: str,
    decision: str,
    reason: str,
) -> str:
    return (
        "[LOCAL_ESCAPE_DECISION]\n"
        "candidate_idx=%d\n"
        "exit_name=%s\n"
        "sweep_result=%s\n"
        "hard_collision_3d=%s\n"
        "old_geometric_xy_overlap=%s\n"
        "old_breakdown_reason=%s\n"
        "decision=%s\n"
        "reason=%s"
        % (
            int(candidate_idx),
            str(exit_name),
            str(sweep_result),
            str(bool(hard_collision_3d)).lower(),
            str(bool(old_geometric_xy_overlap)).lower(),
            str(old_breakdown_reason or "n/a"),
            str(decision),
            str(reason or "n/a"),
        )
    )


def closest_check_from_sweep_debug(
    sweep_debug: Optional[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    dbg = dict(sweep_debug or {})
    start_chk = dict(dbg.get("closest_check_start") or {})
    end_chk = dict(dbg.get("closest_check_end") or {})
    if not start_chk and not end_chk:
        return None
    if not end_chk:
        return start_chk
    if not start_chk:
        return end_chk
    start_xy = float(start_chk.get("xy_clearance", float("inf")))
    end_xy = float(end_chk.get("xy_clearance", float("inf")))
    return end_chk if end_xy <= start_xy else start_chk


def emit_transport_exit_candidate_validate_logs(
    validation_logs: Sequence[Dict[str, Any]],
    *,
    candidate_idx: int,
    carried_label: str,
    obstacles_remaining: Sequence[str],
    target_world_present: bool,
    start_state_source: str = "post_lift_endpoint",
    attached_geom: Optional[Dict[str, Any]] = None,
    required_clearance_m: float = 0.0,
    local_exit_clearance_m: Optional[float] = None,
    global_route_clearance_m: Optional[float] = None,
    log_fn: Optional[Callable[[str], None]] = None,
) -> None:
    if log_fn is None:
        return
    breakdown_clearance = float(
        global_route_clearance_m
        if global_route_clearance_m is not None
        else required_clearance_m
    )
    local_clearance = (
        float(local_exit_clearance_m)
        if local_exit_clearance_m is not None
        else breakdown_clearance
    )
    for vlog in validation_logs:
        kind = str(vlog.get("kind", ""))
        if kind == "corridor":
            metrics = dict(vlog.get("metrics") or {})
            local_decision = dict(vlog.get("local_decision") or vlog.get("decision") or {})
            global_decision = dict(vlog.get("global_decision") or {})
            sweep_dbg = dict(metrics.get("sweep_debug") or {})
            local_escape_ok = bool(
                vlog.get("local_escape_ok", vlog.get("local_escape_result") == "OK")
            )
            seg_b_ok = bool(vlog.get("seg_to_first_ok"))
            geom_ok = local_escape_ok
            plan_checked = False
            hard_collision = bool(
                sweep_dbg.get("hard_collision_3d", local_decision.get("hard_collision"))
            )
            cand_result = "OK" if local_escape_ok else str(vlog.get("result", "FAIL"))
            collision_ok = bool(local_escape_ok) and not hard_collision
            closest_sweep = closest_check_from_sweep_debug(sweep_dbg)
            legacy_closest = closest_obstacle_check_from_segment_checks(
                list(vlog.get("all_checks") or [])
            )
            if closest_sweep is not None:
                min_obs_val = closest_sweep.get("xy_clearance", "n/a")
            else:
                min_obs_val = local_decision.get(
                    "min_geometric_xy_clearance_m",
                    metrics.get("min_geometric_xy_clearance_m", "n/a"),
                )
            if isinstance(min_obs_val, float):
                min_obs = "%.4f" % float(min_obs_val)
            else:
                min_obs = str(min_obs_val)
            table_clr = metrics.get("min_clearance_to_table", "n/a")
            if isinstance(table_clr, float):
                table_clr = "%.4f" % float(table_clr)
            if local_escape_ok:
                reason = "local_escape_sweep_ok"
            else:
                reason = str(local_decision.get("reason", ""))
            old_geom_overlap = bool(
                global_decision.get("diagnostic_geometric_xy_overlap")
                or global_decision.get("reason") == "geometric_xy_overlap"
                or (
                    isinstance(legacy_closest, dict)
                    and float(legacy_closest.get("xy_clearance", 0.0)) < 0.0
                    and bool(legacy_closest.get("xy_overlap"))
                )
            )
            old_breakdown_reason = str(
                global_decision.get("reason", metrics.get("reason", "n/a"))
            )
            log_fn(
                format_transport_exit_candidate_validate_log(
                    candidate_idx=int(candidate_idx),
                    exit_name=str(vlog.get("mode", "unknown")),
                    start_state_source=start_state_source,
                    attached_object=True,
                    carried_label=carried_label,
                    obstacles_remaining=obstacles_remaining,
                    target_world_present=target_world_present,
                    fraction="n/a",
                    geom_ok=geom_ok,
                    plan_checked=plan_checked,
                    plan_ok="n/a",
                    collision_ok=collision_ok,
                    min_obstacle_distance=str(min_obs),
                    table_clearance=str(table_clr),
                    result=str(cand_result),
                    reason=reason,
                )
            )
            if attached_geom is not None and log_fn is not None:
                breakdown_check = closest_sweep or legacy_closest
                breakdown_reason = (
                    "local_escape_sweep_ok"
                    if local_escape_ok
                    else reason or old_breakdown_reason
                )
                log_fn(
                    format_transport_exit_clearance_breakdown_log(
                        candidate_idx=int(candidate_idx),
                        exit_name=str(vlog.get("mode", "unknown")),
                        closest_check=breakdown_check,
                        required_clearance=float(local_clearance),
                        attached_geom=attached_geom,
                        reason=breakdown_reason,
                    )
                )
                if sweep_dbg:
                    sweep_dbg["candidate_idx"] = int(candidate_idx)
                    sweep_dbg["exit_name"] = str(vlog.get("mode", "unknown"))
                    log_fn(format_transport_exit_sweep_debug_log(sweep_dbg))
                log_fn(
                    format_local_escape_decision_log(
                        candidate_idx=int(candidate_idx),
                        exit_name=str(vlog.get("mode", "unknown")),
                        sweep_result=str(sweep_dbg.get("result", cand_result)),
                        hard_collision_3d=bool(sweep_dbg.get("hard_collision_3d", hard_collision)),
                        old_geometric_xy_overlap=old_geom_overlap,
                        old_breakdown_reason=old_breakdown_reason,
                        decision="ALLOW" if local_escape_ok else "REJECT",
                        reason=(
                            "sweep_ok_overrides_xy_projection"
                            if local_escape_ok and old_geom_overlap
                            else reason
                        ),
                    )
                )
        elif kind == "direct":
            metrics = dict(vlog.get("metrics") or {})
            decision = dict(vlog.get("decision") or {})
            cand_result = str(vlog.get("result", "FAIL"))
            plan_ok = cand_result == "OK"
            hard_collision = bool(decision.get("hard_collision"))
            min_obs = metrics.get("min_geometric_xy_clearance_m", "n/a")
            if isinstance(min_obs, float):
                min_obs = "%.4f" % float(min_obs)
            table_clr = metrics.get("min_clearance_to_table", "n/a")
            if isinstance(table_clr, float):
                table_clr = "%.4f" % float(table_clr)
            entry_wp = str(vlog.get("entry_target_waypoint", "carry_front_high"))
            log_fn(
                format_transport_exit_candidate_validate_log(
                    candidate_idx=int(candidate_idx),
                    exit_name="direct_to_%s" % entry_wp,
                    start_state_source=start_state_source,
                    attached_object=True,
                    carried_label=carried_label,
                    obstacles_remaining=obstacles_remaining,
                    target_world_present=target_world_present,
                    fraction="1.00000" if plan_ok else "0.00000",
                    plan_ok=plan_ok,
                    collision_ok=plan_ok and not hard_collision,
                    min_obstacle_distance=str(min_obs),
                    table_clearance=str(table_clr),
                    result=cand_result,
                    reason=str(decision.get("reason", metrics.get("reason", ""))),
                )
            )
            if attached_geom is not None and log_fn is not None:
                closest = closest_obstacle_check_from_segment_checks(
                    list(vlog.get("all_checks") or [])
                )
                log_fn(
                    format_transport_exit_clearance_breakdown_log(
                        candidate_idx=int(candidate_idx),
                        exit_name="direct_to_%s" % entry_wp,
                        closest_check=closest,
                        required_clearance=float(required_clearance_m),
                        attached_geom=attached_geom,
                        reason=str(decision.get("reason", metrics.get("reason", ""))),
                    )
                )


def format_transport_entry_candidate_validate_log(
    idx: int,
    *,
    mode: str,
    candidate_hand: Tuple[float, float, float],
    seg_start_ok: bool,
    seg_to_first_ok: bool,
    metrics: Dict[str, Any],
    decision: Dict[str, Any],
    result: str,
    entry_target_waypoint: str = "carry_front_high",
) -> str:
    seg_b_name = (
        "candidate_to_safe_transport_hub"
        if entry_target_waypoint != "carry_front_high"
        else "candidate_to_carry_front_high"
    )
    return (
        "[TRANSPORT_ENTRY_CANDIDATE_VALIDATE]\n"
        "idx=%d\n"
        "mode=%s\n"
        "validate_only=true\n"
        "start_state=post_lift_frozen\n"
        "candidate_hand=(%.3f, %.3f, %.3f)\n"
        "segment_start_to_candidate=%s\n"
        "segment_%s=%s\n"
        "entry_target_waypoint=%s\n"
        "hard_collision=%s\n"
        "min_geometric_clearance_m=%s\n"
        "min_safety_margin_m=%s\n"
        "result=%s"
        % (
            idx,
            mode,
            candidate_hand[0],
            candidate_hand[1],
            candidate_hand[2],
            "OK" if seg_start_ok else "FAIL",
            seg_b_name,
            "OK" if seg_to_first_ok else "FAIL",
            entry_target_waypoint,
            str(bool(decision.get("hard_collision"))).lower(),
            decision.get("min_geometric_xy_clearance_m", "n/a"),
            decision.get("min_safety_margin_m", "n/a"),
            result,
        )
    )


def format_transport_entry_direct_validate_log(
    *,
    swept_ok: bool,
    metrics: Dict[str, Any],
    decision: Dict[str, Any],
    result: str,
) -> str:
    return (
        "[TRANSPORT_ENTRY_DIRECT_VALIDATE]\n"
        "start_state=post_lift_frozen\n"
        "target=carry_front_high\n"
        "swept_collision_free=%s\n"
        "min_geometric_clearance_m=%s\n"
        "min_safety_margin_m=%s\n"
        "decision=%s\n"
        "result=%s"
        % (
            str(bool(swept_ok and decision.get("decision") in ("OK", "ALLOW_BORDERLINE"))).lower(),
            decision.get("min_geometric_xy_clearance_m", "n/a"),
            decision.get("min_safety_margin_m", "n/a"),
            str(decision.get("decision", "")),
            result,
        )
    )


def generate_rear_retreat_candidates(
    current_hand_xy: Tuple[float, float],
    hand_z: float,
    *,
    modes: Optional[Sequence[str]] = None,
) -> List[Dict[str, Any]]:
    x, y = float(current_hand_xy[0]), float(current_hand_xy[1])
    hz = float(hand_z)
    all_specs: List[Tuple[str, Tuple[float, float, float]]] = [
        ("rear_retreat_x_negative", (max(0.30, x - 0.056), y, hz)),
        ("rear_retreat_x_negative_slight_raise", (max(0.30, x - 0.096), y, hz + 0.020)),
        ("rear_retreat_x_negative_slight_raise", (max(0.30, x - 0.126), y, hz + 0.040)),
        ("rear_retreat_x_negative_far", (max(0.30, x - 0.156), y, hz)),
        ("rear_retreat_x_negative_raise_far", (max(0.30, x - 0.156), y, hz + 0.050)),
        ("rear_retreat_x_negative_raise_far", (max(0.30, x - 0.186), y, hz + 0.060)),
    ]
    vertical_specs: List[Tuple[str, Tuple[float, float, float]]] = [
        ("vertical_raise_then_rear_retreat", (max(0.30, x - 0.056), y, hz + 0.026)),
        ("vertical_raise_then_rear_retreat", (max(0.30, x - 0.096), y, hz + 0.046)),
        ("vertical_raise_then_rear_retreat", (max(0.30, x - 0.126), y, hz + 0.066)),
    ]
    allowed = {str(m) for m in (modes or []) if str(m).strip()}
    specs: List[Tuple[str, Tuple[float, float, float]]] = []
    if not allowed:
        specs = all_specs + vertical_specs
    else:
        if "rear_retreat_x_negative" in allowed:
            specs.append(all_specs[0])
        if "rear_retreat_x_negative_slight_raise" in allowed:
            specs.extend(all_specs[1:3])
        if "rear_retreat_x_negative_far" in allowed:
            specs.append(all_specs[3])
        if "rear_retreat_x_negative_raise_far" in allowed:
            specs.extend(all_specs[4:])
        if "vertical_raise_then_rear_retreat" in allowed:
            specs.extend(vertical_specs)
        if not specs:
            specs = all_specs + vertical_specs
    out: List[Dict[str, Any]] = []
    for mode, pose in specs:
        delta_xy = abs(float(pose[0]) - x)
        out.append(
            {
                "mode": mode,
                "candidate_hand": pose,
                "candidate_hand_z": float(pose[2]),
                "delta_xy_from_current": float(delta_xy),
            }
        )
    return out


def generate_local_exit_candidates(
    current_hand_xy: Tuple[float, float],
    hand_z: float,
    policy: Dict[str, Any],
) -> List[Dict[str, Any]]:
    modes = policy.get("local_exit_candidates")
    if isinstance(modes, list) and len(modes) == 0:
        return []
    mode_list = list(modes or [])
    return generate_rear_retreat_candidates(
        current_hand_xy, hand_z, modes=mode_list if mode_list else None
    )


def generate_carry_front_entry_mid(
    current_hand_xy: Tuple[float, float],
    hand_z: float,
) -> List[Dict[str, Any]]:
    x, y = float(current_hand_xy[0]), float(current_hand_xy[1])
    hz = float(hand_z)
    pose = (0.35, y, hz)
    return [
        {
            "mode": "carry_front_entry_mid",
            "candidate_hand": pose,
            "candidate_hand_z": hz,
            "delta_xy_from_current": abs(float(pose[0]) - x),
        }
    ]


def _limiting_obstacle_clearance_at_hand(
    hand_pos: Tuple[float, float, float],
    attached_geom: Dict[str, Any],
    obstacles: Sequence[Dict[str, Any]],
    *,
    table_top_z: float,
    required_xy_clearance_m: float,
) -> Tuple[float, str]:
    from panda_controller.generic_known_scene_carry_planner import (
        attached_obstacle_clearance_3d,
    )

    min_margin = float("inf")
    limiting = ""
    for obs in obstacles:
        if bool(obs.get("is_target", False)):
            continue
        chk = attached_obstacle_clearance_3d(
            hand_pos,
            attached_geom,
            obs,
            table_top_z=float(table_top_z),
            required_xy_clearance_m=float(required_xy_clearance_m),
        )
        if chk.get("result") == "SKIP":
            continue
        if bool(chk.get("hard_collision")):
            margin = min(
                float(chk.get("xy_clearance", 0.0)),
                float(chk.get("z_clearance", 0.0)),
            )
        elif bool(chk.get("xy_overlap")) or bool(chk.get("z_overlap")):
            margin = min(
                float(chk.get("xy_clearance", 0.0)),
                float(chk.get("z_clearance", 0.0)),
            )
        else:
            margin = float(chk.get("xy_clearance", 0.0))
        if margin < min_margin:
            min_margin = margin
            limiting = str(chk.get("obstacle_label", ""))
    if min_margin == float("inf"):
        return float("inf"), ""
    return float(min_margin), limiting


def _local_escape_maintains_clearance(
    *,
    start_hand: Tuple[float, float, float],
    end_hand: Tuple[float, float, float],
    attached_geom: Dict[str, Any],
    obstacles: Sequence[Dict[str, Any]],
    table_top_z: float,
    local_required_clearance_m: float,
    tolerance_m: float,
) -> Tuple[bool, str]:
    start_margin, _ = _limiting_obstacle_clearance_at_hand(
        start_hand,
        attached_geom,
        obstacles,
        table_top_z=float(table_top_z),
        required_xy_clearance_m=local_required_clearance_m,
    )
    end_margin, _ = _limiting_obstacle_clearance_at_hand(
        end_hand,
        attached_geom,
        obstacles,
        table_top_z=float(table_top_z),
        required_xy_clearance_m=local_required_clearance_m,
    )
    if start_margin == float("inf") or end_margin == float("inf"):
        return True, ""
    if float(end_margin) + float(tolerance_m) + 1e-6 < float(start_margin):
        return False, "limiting_obstacle_clearance_worsened"
    return True, ""


def build_transport_exit_sweep_debug(
    *,
    candidate_idx: int,
    exit_name: str,
    start_hand: Tuple[float, float, float],
    end_hand: Tuple[float, float, float],
    attached_geom: Dict[str, Any],
    scene_obstacles: Sequence[Dict[str, Any]],
    table_top_z: float,
    local_required_clearance_m: float,
    local_escape_ok: bool,
    seg_a_metrics: Dict[str, Any],
    seg_a_checks: Sequence[Dict[str, Any]],
    seg_a_decision: Dict[str, Any],
    fail_reason: str = "",
) -> Dict[str, Any]:
    from panda_controller.generic_known_scene_carry_planner import (
        attached_obstacle_clearance_3d,
        carried_footprint_aabb_xy,
        resolve_carried_object_center_xy,
    )

    start_center = resolve_carried_object_center_xy(start_hand, attached_geom)
    end_center = resolve_carried_object_center_xy(end_hand, attached_geom)
    start_aabb = carried_footprint_aabb_xy(start_hand, attached_geom)
    end_aabb = carried_footprint_aabb_xy(end_hand, attached_geom)
    swept_aabb = (
        min(start_aabb[0], end_aabb[0]),
        max(start_aabb[1], end_aabb[1]),
        min(start_aabb[2], end_aabb[2]),
        max(start_aabb[3], end_aabb[3]),
    )
    closest_label = ""
    closest_check: Optional[Dict[str, Any]] = None
    min_xy = float("inf")
    for hand in (start_hand, end_hand):
        for obs in scene_obstacles:
            if bool(obs.get("is_target", False)):
                continue
            chk = attached_obstacle_clearance_3d(
                hand,
                attached_geom,
                obs,
                table_top_z=float(table_top_z),
                required_xy_clearance_m=float(local_required_clearance_m),
            )
            if chk.get("result") == "SKIP":
                continue
            xy = float(chk.get("xy_clearance", 0.0))
            if xy < min_xy:
                min_xy = xy
                closest_label = str(chk.get("obstacle_label", ""))
                closest_check = dict(chk)
    start_chk = closest_check
    end_chk = closest_check
    if closest_label:
        for obs in scene_obstacles:
            if str(obs.get("label", "")) != closest_label:
                continue
            start_chk = attached_obstacle_clearance_3d(
                start_hand,
                attached_geom,
                obs,
                table_top_z=float(table_top_z),
                required_xy_clearance_m=float(local_required_clearance_m),
            )
            end_chk = attached_obstacle_clearance_3d(
                end_hand,
                attached_geom,
                obs,
                table_top_z=float(table_top_z),
                required_xy_clearance_m=float(local_required_clearance_m),
            )
            break
    hard_3d = any(bool(c.get("hard_collision")) for c in seg_a_checks)
    if local_escape_ok:
        reason = "local_escape_sweep_ok"
        result = "OK"
    else:
        reason = fail_reason or str(
            seg_a_decision.get("reason", seg_a_metrics.get("reason", "n/a"))
        )
        result = "FAIL"
    return {
        "candidate_idx": int(candidate_idx),
        "exit_name": str(exit_name),
        "phase": "local_escape_post_lift",
        "start_hand_xyz": start_hand,
        "end_hand_xyz": end_hand,
        "start_carried_center_xyz": (
            float(start_center[0]),
            float(start_center[1]),
            float(start_hand[2]),
        ),
        "end_carried_center_xyz": (
            float(end_center[0]),
            float(end_center[1]),
            float(end_hand[2]),
        ),
        "carried_center_source": str(start_center[2]),
        "carried_footprint_xy_start": list(start_aabb),
        "carried_footprint_xy_end": list(end_aabb),
        "swept_aabb_xy": list(swept_aabb),
        "closest_obstacle_label": closest_label or "n/a",
        "closest_check_start": start_chk,
        "closest_check_end": end_chk,
        "hard_collision_3d": hard_3d,
        "result": result,
        "reason": reason,
        "required_clearance_m": float(local_required_clearance_m),
        "diagnostic_geometric_xy_overlap": bool(
            seg_a_decision.get("diagnostic_geometric_xy_overlap", False)
        ),
    }


def format_transport_exit_sweep_debug_log(debug: Dict[str, Any]) -> str:
    sh = debug.get("start_hand_xyz") or (0.0, 0.0, 0.0)
    eh = debug.get("end_hand_xyz") or (0.0, 0.0, 0.0)
    sc = debug.get("start_carried_center_xyz") or (0.0, 0.0, 0.0)
    ec = debug.get("end_carried_center_xyz") or (0.0, 0.0, 0.0)
    start_chk = dict(debug.get("closest_check_start") or {})
    end_chk = dict(debug.get("closest_check_end") or {})
    obs_center = start_chk.get("obstacle_center") or end_chk.get("obstacle_center")
    obs_repr = "n/a"
    if isinstance(obs_center, (list, tuple)) and len(obs_center) >= 2:
        obs_repr = "(%.4f, %.4f)" % (float(obs_center[0]), float(obs_center[1]))
    obs_dims = start_chk.get("obstacle_dims") or end_chk.get("obstacle_dims") or "n/a"
    return (
        "[TRANSPORT_EXIT_SWEEP_DEBUG]\n"
        "candidate_idx=%d\n"
        "exit_name=%s\n"
        "phase=%s\n"
        "start_hand_xyz=(%.4f, %.4f, %.4f)\n"
        "end_hand_xyz=(%.4f, %.4f, %.4f)\n"
        "start_carried_center_xyz=(%.4f, %.4f, %.4f)\n"
        "end_carried_center_xyz=(%.4f, %.4f, %.4f)\n"
        "carried_center_source=%s\n"
        "carried_footprint_xy_start=%s\n"
        "carried_footprint_xy_end=%s\n"
        "swept_aabb_xy=%s\n"
        "closest_obstacle_label=%s\n"
        "obstacle_center_xy=%s\n"
        "obstacle_radius_or_aabb=%s\n"
        "xy_distance_start=%s\n"
        "xy_distance_end=%s\n"
        "xy_distance_swept=%s\n"
        "vertical_clearance_start=%s\n"
        "vertical_clearance_end=%s\n"
        "hard_collision_3d=%s\n"
        "required_clearance=%.4f\n"
        "diagnostic_geometric_xy_overlap=%s\n"
        "result=%s\n"
        "reason=%s"
        % (
            int(debug.get("candidate_idx", -1)),
            str(debug.get("exit_name", "")),
            str(debug.get("phase", "local_escape_post_lift")),
            float(sh[0]),
            float(sh[1]),
            float(sh[2]),
            float(eh[0]),
            float(eh[1]),
            float(eh[2]),
            float(sc[0]),
            float(sc[1]),
            float(sc[2]),
            float(ec[0]),
            float(ec[1]),
            float(ec[2]),
            str(debug.get("carried_center_source", "n/a")),
            list(debug.get("carried_footprint_xy_start") or []),
            list(debug.get("carried_footprint_xy_end") or []),
            list(debug.get("swept_aabb_xy") or []),
            str(debug.get("closest_obstacle_label", "n/a")),
            obs_repr,
            str(obs_dims),
            start_chk.get("xy_center_distance", "n/a"),
            end_chk.get("xy_center_distance", "n/a"),
            min(
                float(start_chk.get("xy_clearance", float("inf"))),
                float(end_chk.get("xy_clearance", float("inf"))),
            )
            if start_chk or end_chk
            else "n/a",
            start_chk.get("z_clearance", "n/a"),
            end_chk.get("z_clearance", "n/a"),
            str(bool(debug.get("hard_collision_3d"))).lower(),
            float(debug.get("required_clearance_m", 0.0)),
            str(bool(debug.get("diagnostic_geometric_xy_overlap"))).lower(),
            str(debug.get("result", "FAIL")),
            str(debug.get("reason", "n/a")),
        )
    )


def _validate_corridor_candidate(
    *,
    start_hand: Tuple[float, float, float],
    candidate_hand: Tuple[float, float, float],
    first_hand: Tuple[float, float, float],
    attached_geom: Dict[str, Any],
    scene_obstacles: Sequence[Dict[str, Any]],
    table_top_z: float,
    min_table_clr: float,
    local_req_xy_clr: float,
    local_min_table_clr: float,
    global_req_xy_clr: float,
    tolerance_m: float,
    reconfig_safety: Dict[str, float],
    policy: Dict[str, Any],
) -> Tuple[bool, bool, Dict[str, Any], List[Dict[str, Any]], Dict[str, Any]]:
    from panda_controller.attached_transport_phases import (
        check_transport_reconfiguration_zone,
    )

    seg_a_ok, seg_a_metrics, seg_a_checks = validate_attached_cartesian_hand_segment(
        start_hand,
        candidate_hand,
        attached_geom,
        scene_obstacles,
        table_top_z=float(table_top_z),
        min_table_clearance_m=min_table_clr,
        required_xy_clearance_m=local_req_xy_clr,
        safety_margin_tolerance_m=tolerance_m,
        obstacle_margin_mode="local_escape_post_lift",
    )
    seg_a_decision = decide_attached_transport_preflight(
        seg_a_ok,
        seg_a_metrics,
        seg_a_checks,
        tolerance_m=tolerance_m,
        clearance_mode="local_escape_post_lift",
    )
    seg_a_ok = seg_a_decision.get("decision") in ("OK", "ALLOW_BORDERLINE")
    monotonic_ok, monotonic_reason = _local_escape_maintains_clearance(
        start_hand=start_hand,
        end_hand=candidate_hand,
        attached_geom=attached_geom,
        obstacles=scene_obstacles,
        table_top_z=float(table_top_z),
        local_required_clearance_m=local_req_xy_clr,
        tolerance_m=tolerance_m,
    )
    endpoint_ok, endpoint_checks, endpoint_metrics = validate_attached_hand_pose(
        candidate_hand,
        attached_geom,
        scene_obstacles,
        table_top_z=float(table_top_z),
        min_table_clearance_m=min_table_clr,
        required_xy_clearance_m=local_req_xy_clr,
        safety_margin_tolerance_m=tolerance_m,
        obstacle_margin_mode="local_escape_post_lift",
    )
    endpoint_decision = decide_attached_transport_preflight(
        endpoint_ok,
        endpoint_metrics,
        endpoint_checks,
        tolerance_m=tolerance_m,
        clearance_mode="local_escape_post_lift",
    )
    endpoint_ok = endpoint_decision.get("decision") in ("OK", "ALLOW_BORDERLINE")
    zone = check_transport_reconfiguration_zone(
        hand_pos=candidate_hand,
        attached_geom=attached_geom,
        scene_obstacles=scene_obstacles,
        table_top_z=float(table_top_z),
        policy=policy,
        min_table_clearance_m=float(reconfig_safety["min_table_clearance_m"]),
        min_xy_clearance_m=float(reconfig_safety["min_xy_clearance_m"]),
    )
    zone_ok = bool(zone.get("transport_reconfiguration_zone_ok"))
    seg_b_ok = False
    seg_b_metrics: Dict[str, Any] = {"reason": "reconfiguration_zone_not_reached"}
    seg_b_checks: List[Dict[str, Any]] = list(endpoint_checks)
    if zone_ok:
        seg_b_ok, seg_b_metrics, seg_b_checks = validate_attached_cartesian_hand_segment(
            candidate_hand,
            first_hand,
            attached_geom,
            scene_obstacles,
            table_top_z=float(table_top_z),
            min_table_clearance_m=min_table_clr,
            required_xy_clearance_m=global_req_xy_clr,
            safety_margin_tolerance_m=tolerance_m,
        )
    all_checks = list(seg_a_checks) + list(seg_b_checks)
    combined_ok = bool(
        seg_a_ok and monotonic_ok and endpoint_ok and zone_ok and seg_b_ok
    )
    if not combined_ok:
        if not seg_a_ok:
            combined_metrics = dict(seg_a_metrics)
            combined_metrics["reason"] = str(
                seg_a_metrics.get("reason", "local_escape_segment_not_clear")
            )
        elif not monotonic_ok:
            combined_metrics = dict(endpoint_metrics)
            combined_metrics["reason"] = monotonic_reason
        elif not endpoint_ok:
            combined_metrics = dict(endpoint_metrics)
            combined_metrics.setdefault("reason", "local_escape_endpoint_not_clear")
        elif not zone_ok:
            combined_metrics = {
                "reason": "reconfiguration_zone_not_reached",
                "min_clearance_to_table": zone.get("table_clearance"),
                "min_geometric_xy_clearance_m": zone.get(
                    "min_xy_clearance_to_obstacles"
                ),
            }
        else:
            combined_metrics = dict(seg_b_metrics)
            combined_metrics.setdefault("reason", "global_route_segment_not_clear")
    else:
        combined_metrics = dict(seg_b_metrics)
        combined_metrics["local_escape_phase"] = "OK"
        combined_metrics["reconfiguration_zone"] = "OK"
        combined_metrics["global_route_phase"] = "OK"
    if not seg_a_ok:
        decision = dict(seg_a_decision)
    elif not monotonic_ok:
        decision = {
            "decision": "FAIL",
            "reason": monotonic_reason,
            "hard_collision": False,
            "min_geometric_xy_clearance_m": endpoint_metrics.get(
                "min_geometric_xy_clearance_m"
            ),
            "diagnostic_geometric_xy_overlap": seg_a_decision.get(
                "diagnostic_geometric_xy_overlap", False
            ),
        }
    elif not endpoint_ok:
        decision = dict(endpoint_decision)
    elif not zone_ok:
        decision = {
            "decision": "FAIL",
            "reason": "reconfiguration_zone_not_reached",
            "hard_collision": False,
            "min_geometric_xy_clearance_m": zone.get("min_xy_clearance_to_obstacles"),
        }
    elif not seg_b_ok:
        decision = decide_attached_transport_preflight(
            False,
            seg_b_metrics,
            seg_b_checks,
            tolerance_m=tolerance_m,
            clearance_mode="3d",
        )
    else:
        decision = {
            "decision": "OK",
            "reason": "ok",
            "hard_collision": False,
            "min_geometric_xy_clearance_m": combined_metrics.get(
                "min_geometric_xy_clearance_m"
            ),
        }
    hard_3d = any(bool(c.get("hard_collision")) for c in seg_a_checks)
    local_escape_ok = bool(
        seg_a_ok
        and monotonic_ok
        and endpoint_ok
        and not hard_3d
    )
    local_fail_reason = ""
    if not local_escape_ok:
        if hard_3d:
            local_fail_reason = "hard_collision_3d"
        elif not seg_a_ok:
            local_fail_reason = str(
                seg_a_decision.get("reason", seg_a_metrics.get("reason", "local_escape_segment_not_clear"))
            )
        elif not monotonic_ok:
            local_fail_reason = monotonic_reason
        elif not endpoint_ok:
            local_fail_reason = str(
                endpoint_decision.get("reason", "local_escape_endpoint_not_clear")
            )
    local_decision = {
        "decision": "OK" if local_escape_ok else "FAIL",
        "reason": "local_escape_sweep_ok" if local_escape_ok else local_fail_reason,
        "hard_collision": hard_3d,
        "min_geometric_xy_clearance_m": seg_a_decision.get(
            "min_geometric_xy_clearance_m", seg_a_metrics.get("min_geometric_xy_clearance_m")
        ),
        "diagnostic_geometric_xy_overlap": bool(
            seg_a_decision.get("diagnostic_geometric_xy_overlap", False)
        ),
        "required_clearance_m": float(local_req_xy_clr),
    }
    global_decision = dict(decision)
    combined_metrics["local_escape_ok"] = local_escape_ok
    combined_metrics["local_decision"] = local_decision
    combined_metrics["global_decision"] = global_decision
    combined_metrics["sweep_debug"] = build_transport_exit_sweep_debug(
        candidate_idx=-1,
        exit_name="",
        start_hand=start_hand,
        end_hand=candidate_hand,
        attached_geom=attached_geom,
        scene_obstacles=scene_obstacles,
        table_top_z=float(table_top_z),
        local_required_clearance_m=local_req_xy_clr,
        local_escape_ok=local_escape_ok,
        seg_a_metrics=seg_a_metrics,
        seg_a_checks=seg_a_checks,
        seg_a_decision=seg_a_decision,
        fail_reason=local_fail_reason,
    )
    return seg_a_ok, seg_b_ok, combined_metrics, all_checks, global_decision


def transport_escape_option_fingerprint(option: Dict[str, Any]) -> Tuple[Any, ...]:
    """Identificador estable para no repetir el mismo escape local."""
    hand = option.get("candidate_hand")
    if isinstance(hand, (list, tuple)) and len(hand) >= 3:
        return (
            str(option.get("mode", "")),
            round(float(hand[0]), 3),
            round(float(hand[1]), 3),
            round(float(hand[2]), 3),
        )
    return (
        str(option.get("mode", "")),
        round(float(option.get("hand_z", 0.0)), 3),
    )


def enumerate_transport_entry_escape_options(
    *,
    post_lift_hand: Tuple[float, float, float],
    post_lift_joints: Sequence[float],
    entry_target_joints: Sequence[float],
    entry_target_waypoint: str = "carry_front_high",
    allow_direct_to_carry_front_high: bool = True,
    allow_direct_to_entry_target: Optional[bool] = None,
    allow_carry_front_high_corridors: Optional[bool] = None,
    fk_hand_fn: Callable[[Any], Optional[Tuple[float, float, float]]],
    attached_geom: Dict[str, Any],
    scene_obstacles: Sequence[Dict[str, Any]],
    table_top_z: float,
    policy: Dict[str, Any],
    hand_z_candidates: Sequence[float],
    scene_policy: Optional[Dict[str, Any]] = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Evalúa todos los candidatos de escape (sin mover el robot)."""
    if allow_direct_to_entry_target is None:
        if str(entry_target_waypoint) == "carry_front_high":
            allow_direct_to_entry_target = bool(allow_direct_to_carry_front_high)
        else:
            allow_direct_to_entry_target = True
    if allow_carry_front_high_corridors is None:
        allow_carry_front_high_corridors = bool(allow_direct_to_carry_front_high)
    min_table_clr = float(policy["carry_clearance_above_table_m"]) * 0.5
    from panda_controller.attached_transport_phases import (
        check_transport_reconfiguration_zone,
        resolve_reconfiguration_safety_thresholds,
        resolve_transport_phase_clearance_thresholds,
    )

    phase_clr = resolve_transport_phase_clearance_thresholds(scene_policy, policy)
    local_req_xy_clr = float(phase_clr["local_exit_required_clearance_m"])
    local_min_table_clr = float(phase_clr["local_exit_min_table_clearance_m"])
    global_req_xy_clr = float(phase_clr["global_route_required_clearance_m"])
    tolerance_m = float(
        policy.get(
            "attached_transport_safety_margin_tolerance_m",
            DEFAULT_ATTACHED_TRANSPORT_SAFETY_MARGIN_TOLERANCE_M,
        )
    )
    use_lateral = bool(policy.get("use_lateral_transport_corridors", True))
    frozen_z = float(post_lift_hand[2])
    hand_xy = (float(post_lift_hand[0]), float(post_lift_hand[1]))
    target_hand = fk_hand_fn(entry_target_joints)
    if target_hand is None:
        return [], []

    reconfig_safety = resolve_reconfiguration_safety_thresholds(scene_policy)
    hub_segment = (
        "candidate_to_safe_transport_hub"
        if entry_target_waypoint != "carry_front_high"
        else "candidate_to_carry_front_high"
    )

    validation_logs: List[Dict[str, Any]] = []
    all_options: List[Dict[str, Any]] = []

    backoff_hand_zs: List[float] = []
    seen_z: set = set()
    for z in list(hand_z_candidates) + [frozen_z + 0.026, frozen_z + 0.046, frozen_z + 0.066]:
        key = round(float(z), 4)
        if key not in seen_z:
            seen_z.add(key)
            backoff_hand_zs.append(float(z))

    def _zone_for_hand(hand: Tuple[float, float, float]) -> Dict[str, Any]:
        return check_transport_reconfiguration_zone(
            hand_pos=hand,
            attached_geom=attached_geom,
            scene_obstacles=scene_obstacles,
            table_top_z=float(table_top_z),
            policy=policy,
            min_table_clearance_m=float(reconfig_safety["min_table_clearance_m"]),
            min_xy_clearance_m=float(reconfig_safety["min_xy_clearance_m"]),
        )

    def _annotate_option(opt: Dict[str, Any]) -> Dict[str, Any]:
        if opt.get("candidate_hand") is not None:
            hand = opt["candidate_hand"]
        else:
            hand = (hand_xy[0], hand_xy[1], float(opt.get("hand_z", frozen_z)))
        zone = _zone_for_hand(
            (float(hand[0]), float(hand[1]), float(hand[2]))
        )
        out = dict(opt)
        out["zone_check"] = zone
        out["zone_ok"] = bool(zone.get("transport_reconfiguration_zone_ok"))
        out["corridor_ok"] = True
        return out

    def _try_direct(hand_z: float) -> Optional[Dict[str, Any]]:
        need_raise = float(hand_z) > frozen_z + 1e-4
        start_hand = post_lift_hand if not need_raise else (hand_xy[0], hand_xy[1], float(hand_z))
        _, _, start_metrics = validate_attached_hand_pose(
            start_hand,
            attached_geom,
            scene_obstacles,
            table_top_z=float(table_top_z),
            min_table_clearance_m=min_table_clr,
            required_xy_clearance_m=local_req_xy_clr,
            safety_margin_tolerance_m=tolerance_m,
        )
        baseline_margin = start_metrics.get("min_geometric_xy_clearance_m", float("inf"))
        try:
            baseline_f = float(baseline_margin)
        except (TypeError, ValueError):
            baseline_f = float("inf")
        require_local_first = bool(
            policy.get("require_local_escape_before_global_transport", False)
        )
        if (
            require_local_first
            and baseline_f != float("inf")
            and baseline_f + float(tolerance_m) < float(global_req_xy_clr)
        ):
            validation_logs.append(
                {
                    "kind": "direct",
                    "hand_z": float(hand_z),
                    "entry_target_waypoint": entry_target_waypoint,
                    "metrics": {
                        "min_geometric_xy_clearance_m": baseline_margin,
                        "reason": "direct_requires_local_escape_first",
                    },
                    "decision": {
                        "decision": "FAIL",
                        "reason": "direct_requires_local_escape_first",
                    },
                    "result": "FAIL",
                    "all_checks": [],
                }
            )
            return None
        if not need_raise:
            direct_ok, direct_metrics, direct_checks = validate_attached_joint_segment(
                post_lift_joints,
                entry_target_joints,
                fk_hand_fn=fk_hand_fn,
                attached_geom=attached_geom,
                table_top_z=float(table_top_z),
                obstacles=scene_obstacles,
                min_table_clearance_m=min_table_clr,
                required_xy_clearance_m=global_req_xy_clr,
                safety_margin_tolerance_m=tolerance_m,
            )
        else:
            hand_at_h = (hand_xy[0], hand_xy[1], float(hand_z))
            raise_ok, raise_metrics, raise_checks = validate_attached_cartesian_hand_segment(
                post_lift_hand,
                hand_at_h,
                attached_geom,
                scene_obstacles,
                table_top_z=float(table_top_z),
                min_table_clearance_m=min_table_clr,
                required_xy_clearance_m=local_req_xy_clr,
                safety_margin_tolerance_m=tolerance_m,
                obstacle_margin_mode="xy_unless_hard_collision",
            )
            to_first_ok, direct_metrics, direct_checks = validate_attached_cartesian_hand_segment(
                hand_at_h,
                target_hand,
                attached_geom,
                scene_obstacles,
                table_top_z=float(table_top_z),
                min_table_clearance_m=min_table_clr,
                required_xy_clearance_m=global_req_xy_clr,
                safety_margin_tolerance_m=tolerance_m,
            )
            direct_ok = bool(raise_ok and to_first_ok)
            direct_checks = list(raise_checks) + list(direct_checks)
            if not raise_ok:
                direct_metrics = raise_metrics
        direct_decision = decide_attached_transport_preflight(
            direct_ok, direct_metrics, direct_checks, tolerance_m=tolerance_m
        )
        direct_result = (
            "OK" if direct_decision.get("decision") in ("OK", "ALLOW_BORDERLINE") else "FAIL"
        )
        validation_logs.append(
            {
                "kind": "direct",
                "hand_z": float(hand_z),
                "entry_target_waypoint": entry_target_waypoint,
                "metrics": direct_metrics,
                "decision": direct_decision,
                "result": direct_result,
                "all_checks": list(direct_checks),
            }
        )
        if direct_result != "OK":
            if not direct_ok:
                validation_logs.append(
                    {
                        "kind": "segment_fail_detail",
                        "detail": extract_segment_fail_detail(
                            candidate_mode="direct_to_carry_front_high",
                            segment="post_lift_to_carry_front_high",
                            metrics=direct_metrics,
                            checks=direct_checks,
                        ),
                    }
                )
            return None
        return _annotate_option(
            {
                "idx": -1,
                "mode": "direct_to_carry_front_high",
                "hand_z": float(hand_z),
                "candidate_hand": None,
                "need_raise": need_raise,
                "metrics": direct_metrics,
                "decision": direct_decision,
                "priority": (0, float(hand_z), 0.0),
                "selection_reason": "direct_path_clear",
                "entry_target_waypoint": entry_target_waypoint,
            }
        )

    def _evaluate_corridor_list(
        cands: Sequence[Dict[str, Any]],
        *,
        hand_z: float,
        start_hand: Tuple[float, float, float],
        priority_base: int,
        selection_reason: str,
    ) -> List[Dict[str, Any]]:
        found: List[Dict[str, Any]] = []
        for idx, cand in enumerate(cands):
            pose = cand.get("candidate_hand")
            if not isinstance(pose, (list, tuple)) or len(pose) < 3:
                continue
            candidate_hand = (float(pose[0]), float(pose[1]), float(pose[2]))
            seg_a_ok, seg_b_ok, combined_metrics, all_checks, global_decision = (
                _validate_corridor_candidate(
                    start_hand=start_hand,
                    candidate_hand=candidate_hand,
                    first_hand=target_hand,
                    attached_geom=attached_geom,
                    scene_obstacles=scene_obstacles,
                    table_top_z=float(table_top_z),
                    min_table_clr=min_table_clr,
                    local_req_xy_clr=local_req_xy_clr,
                    local_min_table_clr=local_min_table_clr,
                    global_req_xy_clr=global_req_xy_clr,
                    tolerance_m=tolerance_m,
                    reconfig_safety=reconfig_safety,
                    policy=policy,
                )
            )
            local_escape_ok = bool(combined_metrics.get("local_escape_ok"))
            local_decision = dict(combined_metrics.get("local_decision") or {})
            cand_result = "OK" if local_escape_ok else "FAIL"
            validation_logs.append(
                {
                    "kind": "corridor",
                    "idx": idx,
                    "mode": str(cand.get("mode", "")),
                    "hand_z": float(hand_z),
                    "candidate_hand": candidate_hand,
                    "seg_start_ok": seg_a_ok,
                    "seg_to_first_ok": seg_b_ok,
                    "metrics": combined_metrics,
                    "decision": global_decision,
                    "local_decision": local_decision,
                    "global_decision": global_decision,
                    "local_escape_ok": local_escape_ok,
                    "local_escape_result": "OK" if local_escape_ok else "FAIL",
                    "result": cand_result,
                    "delta_xy": float(cand.get("delta_xy_from_current", 0.0)),
                    "zone_ok": False,
                    "all_checks": list(all_checks),
                }
            )
            if not local_escape_ok:
                fail_seg = "start_to_candidate" if not seg_a_ok else hub_segment
                validation_logs.append(
                    {
                        "kind": "segment_fail_detail",
                        "detail": extract_segment_fail_detail(
                            candidate_mode=str(cand.get("mode", "")),
                            segment=fail_seg,
                            metrics=combined_metrics,
                            checks=all_checks,
                        ),
                    }
                )
                continue
            found.append(
                _annotate_option(
                    {
                        "idx": idx,
                        "mode": str(cand.get("mode", "")),
                        "hand_z": float(hand_z),
                        "candidate_hand": candidate_hand,
                        "need_raise": float(hand_z) > frozen_z + 1e-4,
                        "metrics": combined_metrics,
                        "decision": local_decision,
                        "global_decision": global_decision,
                        "local_escape_ok": local_escape_ok,
                        "priority": (
                            priority_base,
                            float(hand_z),
                            float(cand.get("delta_xy_from_current", 0.0)),
                        ),
                        "selection_reason": selection_reason,
                        "entry_target_waypoint": entry_target_waypoint,
                    }
                )
            )
        return found

    for hand_z in backoff_hand_zs:
        hand_at_h = (hand_xy[0], hand_xy[1], float(hand_z))
        need_raise = float(hand_z) > frozen_z + 1e-4
        if need_raise:
            ok_pose, _, _ = validate_attached_hand_pose(
                hand_at_h,
                attached_geom,
                scene_obstacles,
                table_top_z=float(table_top_z),
                min_table_clearance_m=min_table_clr,
                required_xy_clearance_m=local_req_xy_clr,
                safety_margin_tolerance_m=tolerance_m,
            )
            if not ok_pose:
                continue
        start_hand = post_lift_hand if not need_raise else hand_at_h

        if allow_direct_to_entry_target:
            direct_opt = _try_direct(hand_z)
            if direct_opt is not None:
                all_options.append(direct_opt)

        all_options.extend(
            _evaluate_corridor_list(
                generate_local_exit_candidates(hand_xy, float(hand_z), policy),
                hand_z=float(hand_z),
                start_hand=start_hand,
                priority_base=1,
                selection_reason="simple_backoff_escape",
            )
        )

        if allow_carry_front_high_corridors:
            all_options.extend(
                _evaluate_corridor_list(
                    generate_carry_front_entry_mid(hand_xy, float(hand_z)),
                    hand_z=float(hand_z),
                    start_hand=start_hand,
                    priority_base=2,
                    selection_reason="carry_front_entry_mid",
                )
            )

        if use_lateral and allow_carry_front_high_corridors:
            all_options.extend(
                _evaluate_corridor_list(
                    generate_transport_entry_candidates(
                        hand_xy, float(hand_z), policy, include_lateral=True
                    ),
                    hand_z=float(hand_z),
                    start_hand=start_hand,
                    priority_base=3,
                    selection_reason="lateral_corridor_fallback",
                )
            )

    if not all_options and allow_direct_to_entry_target:
        direct_opt = _try_direct(frozen_z)
        if direct_opt is not None:
            fallback = dict(direct_opt)
            fallback["selection_reason"] = "conservative_direct_to_entry_fallback"
            fallback["mode"] = "conservative_direct_to_%s" % str(entry_target_waypoint)
            all_options.append(fallback)

    # Actualizar zone_ok en logs de corridor para trazabilidad.
    zone_by_mode: Dict[Tuple[str, Tuple[float, float, float]], bool] = {}
    for opt in all_options:
        hand = opt.get("candidate_hand")
        if hand is None:
            hand = (hand_xy[0], hand_xy[1], float(opt.get("hand_z", frozen_z)))
        key = (str(opt.get("mode", "")), tuple(round(float(v), 3) for v in hand))
        zone_by_mode[key] = bool(opt.get("zone_ok"))
    for vlog in validation_logs:
        if vlog.get("kind") != "corridor":
            continue
        cand_hand = vlog.get("candidate_hand")
        if not isinstance(cand_hand, (list, tuple)):
            continue
        key = (
            str(vlog.get("mode", "")),
            tuple(round(float(v), 3) for v in cand_hand),
        )
        if key in zone_by_mode:
            vlog["zone_ok"] = zone_by_mode[key]

    all_options.sort(
        key=lambda o: (
            not bool(o.get("zone_ok")),
            o.get("priority", (99, 0.0, 0.0)),
        )
    )
    return all_options, validation_logs


def select_transport_entry_validate_only(
    *,
    post_lift_hand: Tuple[float, float, float],
    post_lift_joints: Sequence[float],
    entry_target_joints: Sequence[float],
    entry_target_waypoint: str = "carry_front_high",
    allow_direct_to_carry_front_high: bool = True,
    allow_direct_to_entry_target: Optional[bool] = None,
    allow_carry_front_high_corridors: Optional[bool] = None,
    fk_hand_fn: Callable[[Any], Optional[Tuple[float, float, float]]],
    attached_geom: Dict[str, Any],
    scene_obstacles: Sequence[Dict[str, Any]],
    table_top_z: float,
    policy: Dict[str, Any],
    hand_z_candidates: Sequence[float],
    scene_policy: Optional[Dict[str, Any]] = None,
) -> Tuple[Optional[Dict[str, Any]], List[Dict[str, Any]]]:
    """Selecciona un único candidato sin mover el robot."""
    options, validation_logs = enumerate_transport_entry_escape_options(
        post_lift_hand=post_lift_hand,
        post_lift_joints=post_lift_joints,
        entry_target_joints=entry_target_joints,
        entry_target_waypoint=entry_target_waypoint,
        allow_direct_to_carry_front_high=allow_direct_to_carry_front_high,
        allow_direct_to_entry_target=allow_direct_to_entry_target,
        allow_carry_front_high_corridors=allow_carry_front_high_corridors,
        fk_hand_fn=fk_hand_fn,
        attached_geom=attached_geom,
        scene_obstacles=scene_obstacles,
        table_top_z=table_top_z,
        policy=policy,
        hand_z_candidates=hand_z_candidates,
        scene_policy=scene_policy,
    )
    if not options:
        return None, validation_logs
    zone_ok_opts = [o for o in options if bool(o.get("zone_ok"))]
    if zone_ok_opts:
        return zone_ok_opts[0], validation_logs
    return options[0], validation_logs
