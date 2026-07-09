"""Validación secuencial global de escenas demo (orden de pick progresivo)."""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence

from panda_controller.attached_transport_phases import (
    resolve_transport_phase_clearance_thresholds,
)
from panda_controller.demo_scene_policy import resolve_pick_order_from_scene_policy


def remaining_obstacles_for_target(
    pick_order: Sequence[str],
    target_label: str,
) -> List[str]:
    """Obstáculos aún presentes en mesa cuando se recoge target_label."""
    order = [str(x).strip().lower() for x in pick_order if str(x).strip()]
    tgt = str(target_label or "").strip().lower()
    if tgt not in order:
        return []
    idx = order.index(tgt)
    return order[idx + 1 :]


def format_demo_scene_global_sequence_validate_log(
    *,
    scene_id: str,
    order: Sequence[str],
) -> str:
    order_repr = "[%s]" % ", ".join(str(x) for x in order)
    return (
        "[DEMO_SCENE_GLOBAL_SEQUENCE_VALIDATE]\n"
        "scene_id=%s\n"
        "order=%s\n"
        "result=OK"
        % (str(scene_id), order_repr)
    )


def format_demo_scene_object_sequence_validate_log(
    *,
    target_label: str,
    remaining_obstacles: Sequence[str],
    local_exit_min_clearance: float,
    global_route_min_clearance: float,
    result: str,
    reason: str,
) -> str:
    obs_repr = "[%s]" % ", ".join(str(o) for o in remaining_obstacles)
    return (
        "[DEMO_SCENE_OBJECT_SEQUENCE_VALIDATE]\n"
        "target_label=%s\n"
        "remaining_obstacles=%s\n"
        "local_exit_min_clearance=%.4f\n"
        "global_route_min_clearance=%.4f\n"
        "result=%s\n"
        "reason=%s"
        % (
            str(target_label),
            obs_repr,
            float(local_exit_min_clearance),
            float(global_route_min_clearance),
            str(result),
            str(reason or "n/a"),
        )
    )


def evaluate_demo_scene_object_sequence_step(
    *,
    scene_policy: Optional[Dict[str, Any]],
    target_label: str,
    carry_policy: Optional[Dict[str, Any]] = None,
    transport_score: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    if not isinstance(scene_policy, dict):
        return {
            "target_label": str(target_label),
            "remaining_obstacles": [],
            "result": "SKIP",
            "reason": "no_scene_policy",
        }
    pick_order = resolve_pick_order_from_scene_policy(scene_policy, fallback=[])
    remaining = remaining_obstacles_for_target(pick_order, target_label)
    phase = resolve_transport_phase_clearance_thresholds(scene_policy, carry_policy)
    score = dict(transport_score or {})
    result = "OK"
    reason = "sequence_policy_ready"
    if score:
        if str(score.get("result", "")).upper() == "ACCEPT":
            result = "OK"
            reason = str(score.get("selected_transport_mode") or "transport_accept")
        elif score.get("transport_entry_possible") is False:
            result = "FAIL"
            reason = "no_local_escape_exit"
        elif score.get("reconfiguration_zone_ok") is False:
            result = "FAIL"
            reason = "reconfiguration_zone_not_reached"
        elif score.get("direct_action_to_hub_ok") is False:
            result = "FAIL"
            reason = "global_route_not_clear"
        else:
            result = "FAIL"
            reason = "transport_reject"
    return {
        "scene_id": str(scene_policy.get("scene_id", "")),
        "target_label": str(target_label).strip().lower(),
        "remaining_obstacles": remaining,
        "local_exit_min_clearance": float(phase["local_exit_required_clearance_m"]),
        "global_route_min_clearance": float(phase["global_route_required_clearance_m"]),
        "result": result,
        "reason": reason,
    }


def emit_demo_scene_global_sequence_validate_log(
    *,
    scene_policy: Optional[Dict[str, Any]],
    log_fn: Optional[Callable[[str], None]] = None,
) -> None:
    if log_fn is None or not isinstance(scene_policy, dict):
        return
    pick_order = resolve_pick_order_from_scene_policy(scene_policy, fallback=[])
    if not pick_order:
        return
    log_fn(
        format_demo_scene_global_sequence_validate_log(
            scene_id=str(scene_policy.get("scene_id", "")),
            order=pick_order,
        )
    )


def emit_demo_scene_object_sequence_validate_log(
    *,
    scene_policy: Optional[Dict[str, Any]],
    target_label: str,
    carry_policy: Optional[Dict[str, Any]] = None,
    transport_score: Optional[Dict[str, Any]] = None,
    log_fn: Optional[Callable[[str], None]] = None,
) -> None:
    if log_fn is None or not isinstance(scene_policy, dict):
        return
    step = evaluate_demo_scene_object_sequence_step(
        scene_policy=scene_policy,
        target_label=target_label,
        carry_policy=carry_policy,
        transport_score=transport_score,
    )
    if step.get("result") == "SKIP":
        return
    log_fn(
        format_demo_scene_object_sequence_validate_log(
            target_label=str(step.get("target_label", "")),
            remaining_obstacles=list(step.get("remaining_obstacles") or []),
            local_exit_min_clearance=float(step.get("local_exit_min_clearance", 0.0)),
            global_route_min_clearance=float(step.get("global_route_min_clearance", 0.0)),
            result=str(step.get("result", "FAIL")),
            reason=str(step.get("reason", "")),
        )
    )
