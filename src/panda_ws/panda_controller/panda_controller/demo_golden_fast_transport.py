"""Transporte online acotado cuando pick/place usan golden fast execute."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

from panda_controller.attached_transport_entry_validate import (
    generate_rear_retreat_candidates,
)

GOLDEN_FAST_DEFAULT_ENTRY_MODE = "vertical_raise_then_rear_retreat"
GOLDEN_FAST_MAX_ESCAPE_ROUNDS = 2
GOLDEN_FAST_MAX_ENTRY_CANDIDATES = 3
GOLDEN_FAST_CARRY_CARTESIAN_FRACTION = 0.85


def golden_fast_transport_active(candidate: Optional[Dict[str, Any]]) -> bool:
    return bool(isinstance(candidate, dict) and candidate.get("_demo_golden_fast_execute"))


def golden_fast_transport_prevalidated(candidate: Optional[Dict[str, Any]]) -> bool:
    if not golden_fast_transport_active(candidate):
        return False
    return bool(
        isinstance(candidate, dict)
        and (
            candidate.get("_golden_transport_prevalidated")
            or candidate.get("_golden_transport_route")
        )
    )


def resolve_golden_fast_transport_sequence(
    *,
    golden_route: Optional[Sequence[str]],
    default_sequence: Sequence[str],
    scene_policy_route: Optional[Sequence[str]],
    golden_fast_active: bool,
) -> List[str]:
    """Golden route gana sobre scene_policy cuando golden fast está activo."""
    if golden_fast_active and isinstance(golden_route, (list, tuple)) and golden_route:
        seq = [str(x) for x in golden_route]
        if "box_high" not in seq:
            seq.append("box_high")
        return seq
    if isinstance(scene_policy_route, (list, tuple)) and scene_policy_route:
        seq = [str(x) for x in scene_policy_route]
        if "box_high" not in seq:
            seq.append("box_high")
        return seq
    return [str(x) for x in default_sequence]


def golden_fast_bounded_hand_z_candidates(
    frozen_hand_z: float,
    plan: Optional[Dict[str, Any]] = None,
) -> List[float]:
    """Máximo 3 alturas: actual, carry_safe del plan, bump golden estándar."""
    frozen = float(frozen_hand_z)
    seen = {round(frozen, 4)}
    out = [frozen]
    plan = plan or {}
    for key in ("carry_safe_hand_z", "preferred_hand_z"):
        val = plan.get(key)
        if val is None:
            continue
        try:
            z = float(val)
        except (TypeError, ValueError):
            continue
        key_r = round(z, 4)
        if key_r not in seen and z > frozen + 1e-4:
            seen.add(key_r)
            out.append(z)
    for bump in (0.026, 0.046):
        z = frozen + float(bump)
        key_r = round(z, 4)
        if key_r not in seen:
            seen.add(key_r)
            out.append(z)
        if len(out) >= GOLDEN_FAST_MAX_ENTRY_CANDIDATES:
            break
    return out[:GOLDEN_FAST_MAX_ENTRY_CANDIDATES]


def build_golden_fast_bounded_escape_options(
    post_lift_hand: Tuple[float, float, float],
    *,
    golden_entry_mode: str = GOLDEN_FAST_DEFAULT_ENTRY_MODE,
) -> List[Dict[str, Any]]:
    """Candidatos de escape limitados al modo validado en golden (sin grid lateral)."""
    frozen_z = float(post_lift_hand[2])
    hand_xy = (float(post_lift_hand[0]), float(post_lift_hand[1]))
    mode = str(golden_entry_mode or GOLDEN_FAST_DEFAULT_ENTRY_MODE).strip()
    raw = generate_rear_retreat_candidates(hand_xy, frozen_z, modes=[mode])
    if not raw:
        raw = generate_rear_retreat_candidates(
            hand_xy, frozen_z, modes=[GOLDEN_FAST_DEFAULT_ENTRY_MODE]
        )
    out: List[Dict[str, Any]] = []
    for idx, cand in enumerate(raw[:GOLDEN_FAST_MAX_ENTRY_CANDIDATES]):
        pose = cand.get("candidate_hand")
        if not isinstance(pose, (list, tuple)) or len(pose) < 3:
            continue
        hz = float(pose[2])
        cx, cy, cz = float(pose[0]), float(pose[1]), float(pose[2])
        need_raise = bool(hz > frozen_z + 1e-4 and abs(cx - hand_xy[0]) < 1e-3 and abs(cy - hand_xy[1]) < 1e-3)
        if str(cand.get("mode", "")) == GOLDEN_FAST_DEFAULT_ENTRY_MODE:
            need_raise = hz > frozen_z + 1e-4
        out.append(
            {
                "idx": int(idx),
                "mode": str(cand.get("mode", mode)),
                "hand_z": hz,
                "candidate_hand": (cx, cy, cz),
                "need_raise": need_raise,
                "selection_reason": "golden_fast_bounded_%s" % mode,
                "zone_ok": True,
                "corridor_ok": True,
                "metrics": {"reason": "golden_transport_prevalidated"},
                "decision": {
                    "decision": "OK",
                    "reason": "golden_transport_prevalidated",
                },
            }
        )
    return out


def format_golden_fast_bounded_transport_log(fields: Dict[str, Any]) -> str:
    return (
        "[GOLDEN_FAST_BOUNDED_TRANSPORT]\n"
        "target_label=%s\n"
        "entry_mode=%s\n"
        "candidate_count=%d\n"
        "hand_z_candidates=%s\n"
        "max_escape_rounds=%d\n"
        "cartesian_fraction=%.2f\n"
        "hub_segment_prevalidated=%s\n"
        "result=%s"
        % (
            fields.get("target_label", ""),
            fields.get("entry_mode", ""),
            int(fields.get("candidate_count", 0)),
            fields.get("hand_z_candidates", []),
            int(fields.get("max_escape_rounds", GOLDEN_FAST_MAX_ESCAPE_ROUNDS)),
            float(fields.get("cartesian_fraction", GOLDEN_FAST_CARRY_CARTESIAN_FRACTION)),
            str(fields.get("hub_segment_prevalidated", False)).lower(),
            fields.get("result", "OK"),
        )
    )
