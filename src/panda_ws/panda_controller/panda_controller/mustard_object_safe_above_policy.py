"""Política de altura object_safe_above para mustard_bottle + tall_object_topdown (sin ROS)."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

MUSTARD_OBJECT_SAFE_ABOVE_PREGRASP_OFFSET_M = 0.030
MUSTARD_OBJECT_SAFE_ABOVE_TOP_OFFSET_M = 0.085
MUSTARD_OBJECT_SAFE_ABOVE_PREGRASP_INCREMENTS_M: Tuple[float, ...] = (
    0.020,
    0.030,
    0.040,
    0.050,
)


def mustard_object_safe_above_policy_active(label: str, strategy: str) -> bool:
    return (
        str(label or "").strip().lower() == "mustard_bottle"
        and str(strategy or "").strip().lower() == "tall_object_topdown"
    )


def mustard_object_safe_above_policy_active_for_candidate(
    candidate: Dict[str, Any],
) -> bool:
    return mustard_object_safe_above_policy_active(
        str(candidate.get("label", "")),
        str(candidate.get("grasp_strategy", "")),
    )


def resolve_mustard_object_safe_above_tcp_z(
    *,
    top_z_m: float,
    selected_pregrasp_tcp_z: float,
    old_safe_above_tcp_z: Optional[float] = None,
) -> Dict[str, Any]:
    """Altura baja: min(pregrasp+0.030, top_z+0.085) en lugar del clearance genérico 0.150."""
    policy_a_z = float(selected_pregrasp_tcp_z) + MUSTARD_OBJECT_SAFE_ABOVE_PREGRASP_OFFSET_M
    policy_b_z = float(top_z_m) + MUSTARD_OBJECT_SAFE_ABOVE_TOP_OFFSET_M
    new_z = min(policy_a_z, policy_b_z)
    return {
        "top_z_m": float(top_z_m),
        "selected_pregrasp_tcp_z": float(selected_pregrasp_tcp_z),
        "old_safe_above_tcp_z": (
            float(old_safe_above_tcp_z)
            if old_safe_above_tcp_z is not None
            else None
        ),
        "new_safe_above_tcp_z": float(new_z),
        "policy_a_pregrasp_offset_z": float(policy_a_z),
        "policy_b_top_offset_z": float(policy_b_z),
        "clearance_above_top_m": float(new_z) - float(top_z_m),
        "clearance_above_pregrasp_m": float(new_z) - float(selected_pregrasp_tcp_z),
    }


def build_mustard_object_safe_above_z_candidates(
    *,
    selected_pregrasp_tcp_z: float,
    top_z_m: float,
) -> List[float]:
    """Primero la política baja; luego incrementos pregrasp+[0.020..0.050] sin duplicados."""
    primary = resolve_mustard_object_safe_above_tcp_z(
        top_z_m=float(top_z_m),
        selected_pregrasp_tcp_z=float(selected_pregrasp_tcp_z),
    )["new_safe_above_tcp_z"]
    seen: set = set()
    out: List[float] = []
    for z in [float(primary)] + [
        float(selected_pregrasp_tcp_z) + float(d)
        for d in MUSTARD_OBJECT_SAFE_ABOVE_PREGRASP_INCREMENTS_M
    ]:
        key = round(z, 6)
        if key in seen:
            continue
        seen.add(key)
        out.append(float(z))
    return out


def format_mustard_object_safe_above_height_policy_log(payload: Dict[str, Any]) -> str:
    return (
        "[MUSTARD_OBJECT_SAFE_ABOVE_HEIGHT_POLICY]\n"
        "top_z=%s\n"
        "selected_pregrasp_tcp_z=%s\n"
        "old_safe_above_tcp_z=%s\n"
        "new_safe_above_tcp_z=%s\n"
        "old_hand_z=%s\n"
        "new_hand_z=%s\n"
        "ik_ok=%s\n"
        "plan_ok=%s\n"
        "result=%s\n"
        "reason=%s"
        % (
            payload.get("top_z", "n/a"),
            payload.get("selected_pregrasp_tcp_z", "n/a"),
            payload.get("old_safe_above_tcp_z", "n/a"),
            payload.get("new_safe_above_tcp_z", "n/a"),
            payload.get("old_hand_z", "n/a"),
            payload.get("new_hand_z", "n/a"),
            payload.get("ik_ok", "n/a"),
            payload.get("plan_ok", "n/a"),
            payload.get("result", "n/a"),
            payload.get("reason", ""),
        )
    )
