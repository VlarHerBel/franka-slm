"""Métricas y scoring genéricos para benchmark de candidatos manipulación."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple


@dataclass
class ManipulationCandidate:
    candidate_id: int
    scene_id: str
    target_label: str
    slot_index: int
    yaw_variant_rad: float = 0.0
    yaw_variant_deg: float = 0.0
    pregrasp_tcp_z: float = 0.0
    grasp_tcp_z: float = 0.0
    depth_from_top_m: float = 0.0
    ik_seed_name: str = ""
    grasp_strategy: str = ""
    closing_yaw: float = 0.0
    selected_local_exit: str = ""
    selected_transport_route: str = ""
    pre_plan: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    gr_plan: Tuple[float, float, float] = (0.0, 0.0, 0.0)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["pre_plan"] = list(self.pre_plan)
        d["gr_plan"] = list(self.gr_plan)
        return d


@dataclass
class CandidateEvaluationResult:
    pick_plan_ok: bool = False
    cartesian_descend_ok: bool = False
    contact_policy_ok: bool = False
    lift_precheck_ok: bool = False
    local_escape_ok: bool = False
    transport_entry_ok: bool = False
    transport_route_ok: bool = False
    place_precheck_ok: bool = False
    return_home_precheck_ok: bool = False
    result: str = "REJECT"
    rejection_reason: str = ""

    def all_gates_ok(self) -> bool:
        return (
            self.pick_plan_ok
            and self.cartesian_descend_ok
            and self.contact_policy_ok
            and self.lift_precheck_ok
            and self.local_escape_ok
            and self.transport_entry_ok
            and self.transport_route_ok
            and self.place_precheck_ok
            and self.return_home_precheck_ok
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class CandidateMetrics:
    estimated_pick_time_s: float = 0.0
    estimated_lift_time_s: float = 0.0
    estimated_transport_entry_time_s: float = 0.0
    estimated_transport_route_time_s: float = 0.0
    estimated_place_time_s: float = 0.0
    estimated_return_home_time_s: float = 0.0
    estimated_total_time_s: float = 0.0
    joint_distance_post_lift_to_hub: float = float("inf")
    joint_distance_total: float = float("inf")
    wrist_twist_score: float = float("inf")
    joint6_delta: float = 0.0
    joint7_delta: float = 0.0
    number_of_segments: int = 0
    number_of_fallbacks: int = 0
    requires_unwind: bool = False
    obstacle_disturbed: bool = False
    score: float = float("inf")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class CandidateRealTiming:
    pick_real_s: float = 0.0
    lift_real_s: float = 0.0
    transport_entry_real_s: float = 0.0
    transport_route_real_s: float = 0.0
    place_real_s: float = 0.0
    return_home_real_s: float = 0.0
    total_real_s: float = 0.0
    obstacle_disturbed: bool = False
    fallback_count: int = 0
    unwind_count: int = 0
    result: str = "FAIL"
    failure_reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


DEFAULT_SCORE_PENALTIES: Dict[str, float] = {
    "unwind": 8.0,
    "obstacle_disturbed": 5.0,
    "fallback": 4.0,
    "wrist_twist": 3.0,
    "joint_distance_to_hub": 2.0,
    "joint7_delta": 2.0,
    "segments": 1.0,
    "carry_front_high": 6.0,
    "transport_fallback": 5.0,
    "place_fallback": 5.0,
}


def compute_estimated_total_time(metrics: CandidateMetrics) -> float:
    return (
        float(metrics.estimated_pick_time_s)
        + float(metrics.estimated_lift_time_s)
        + float(metrics.estimated_transport_entry_time_s)
        + float(metrics.estimated_transport_route_time_s)
        + float(metrics.estimated_place_time_s)
        + float(metrics.estimated_return_home_time_s)
    )


def compute_candidate_score(
    metrics: CandidateMetrics,
    *,
    penalties: Optional[Dict[str, float]] = None,
    extra_penalties: Optional[Dict[str, float]] = None,
) -> float:
    """Score configurable: menor es mejor."""
    pen = dict(DEFAULT_SCORE_PENALTIES)
    if penalties:
        pen.update(penalties)
    total_time = compute_estimated_total_time(metrics)
    if metrics.estimated_total_time_s <= 0.0:
        metrics.estimated_total_time_s = total_time
    score = float(metrics.estimated_total_time_s)
    if metrics.requires_unwind:
        score += float(pen.get("unwind", 8.0))
    if metrics.obstacle_disturbed:
        score += float(pen.get("obstacle_disturbed", 5.0))
    score += float(pen.get("fallback", 4.0)) * float(metrics.number_of_fallbacks)
    score += float(pen.get("wrist_twist", 3.0)) * float(metrics.wrist_twist_score)
    score += float(pen.get("joint_distance_to_hub", 2.0)) * float(
        metrics.joint_distance_post_lift_to_hub
    )
    score += float(pen.get("joint7_delta", 2.0)) * abs(float(metrics.joint7_delta))
    score += float(pen.get("segments", 1.0)) * float(metrics.number_of_segments)
    if extra_penalties:
        for _key, val in extra_penalties.items():
            score += float(val)
    metrics.score = float(score)
    return float(score)


def evaluation_from_try_one(
    *,
    ik_pregrasp_ok: bool,
    plan_to_pregrasp_ok: bool,
    cart_ok: bool,
    lift_ok: bool,
    local_escape_ok: bool,
    global_route_ok: bool,
    reject_reason: str = "",
    place_precheck_ok: bool = True,
    return_home_precheck_ok: bool = True,
    contact_policy_ok: bool = True,
) -> CandidateEvaluationResult:
    ev = CandidateEvaluationResult(
        pick_plan_ok=bool(ik_pregrasp_ok and plan_to_pregrasp_ok),
        cartesian_descend_ok=bool(cart_ok),
        contact_policy_ok=bool(contact_policy_ok),
        lift_precheck_ok=bool(lift_ok),
        local_escape_ok=bool(local_escape_ok),
        transport_entry_ok=bool(local_escape_ok),
        transport_route_ok=bool(global_route_ok),
        place_precheck_ok=bool(place_precheck_ok),
        return_home_precheck_ok=bool(return_home_precheck_ok),
    )
    if ev.all_gates_ok():
        ev.result = "VALID"
        ev.rejection_reason = ""
    else:
        ev.result = "REJECT"
        ev.rejection_reason = str(reject_reason or _first_failed_gate(ev))
    return ev


def _first_failed_gate(ev: CandidateEvaluationResult) -> str:
    gates = [
        ("pick_plan_ok", ev.pick_plan_ok),
        ("cartesian_descend_ok", ev.cartesian_descend_ok),
        ("contact_policy_ok", ev.contact_policy_ok),
        ("lift_precheck_ok", ev.lift_precheck_ok),
        ("local_escape_ok", ev.local_escape_ok),
        ("transport_entry_ok", ev.transport_entry_ok),
        ("transport_route_ok", ev.transport_route_ok),
        ("place_precheck_ok", ev.place_precheck_ok),
        ("return_home_precheck_ok", ev.return_home_precheck_ok),
    ]
    for name, ok in gates:
        if not ok:
            return "%s_fail" % name
    return "unknown_reject"


def metrics_from_transport_score(
    transport_score: Optional[Dict[str, Any]],
    *,
    estimated_pick_time_s: float = 5.0,
    estimated_lift_time_s: float = 3.0,
    estimated_place_time_s: float = 10.0,
    estimated_return_home_time_s: float = 4.0,
) -> CandidateMetrics:
    ts = transport_score or {}
    route = ts.get("selected_transport_route") or ts.get("route") or []
    if isinstance(route, (list, tuple)):
        n_seg = len(route)
        route_str = ",".join(str(x) for x in route)
    else:
        n_seg = 0
        route_str = str(route)
    requires_unwind = bool(ts.get("requires_unwind")) or "carry_front_high" in route_str
    return CandidateMetrics(
        estimated_pick_time_s=float(estimated_pick_time_s),
        estimated_lift_time_s=float(estimated_lift_time_s),
        estimated_transport_entry_time_s=float(
            ts.get("estimated_transport_entry_time_s", 6.0)
        ),
        estimated_transport_route_time_s=float(
            ts.get("estimated_transport_route_time_s", 34.0)
        ),
        estimated_place_time_s=float(estimated_place_time_s),
        estimated_return_home_time_s=float(estimated_return_home_time_s),
        joint_distance_post_lift_to_hub=float(
            ts.get("joint_distance_to_hub", float("inf"))
        ),
        wrist_twist_score=float(ts.get("wrist_twist_score", float("inf"))),
        joint6_delta=float(ts.get("joint6_delta", 0.0)),
        joint7_delta=float(ts.get("joint7_delta", 0.0)),
        number_of_segments=int(n_seg),
        number_of_fallbacks=int(ts.get("number_of_fallbacks", 0)),
        requires_unwind=requires_unwind,
        obstacle_disturbed=bool(ts.get("obstacle_disturbed", False)),
    )


def build_candidate_score_record(
    candidate: ManipulationCandidate,
    evaluation: CandidateEvaluationResult,
    metrics: CandidateMetrics,
    *,
    selected: bool = False,
    penalties: Optional[Dict[str, float]] = None,
    extra_penalties: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    compute_candidate_score(metrics, penalties=penalties, extra_penalties=extra_penalties)
    rec = {
        "scene_id": candidate.scene_id,
        "target_label": candidate.target_label,
        "slot_index": candidate.slot_index,
        "candidate_id": candidate.candidate_id,
        "yaw_deg": float(candidate.yaw_variant_deg),
        "pregrasp_tcp_z": float(candidate.pregrasp_tcp_z),
        "grasp_tcp_z": float(candidate.grasp_tcp_z),
        "depth_from_top_m": float(candidate.depth_from_top_m),
        "ik_seed_name": str(candidate.ik_seed_name),
        "grasp_strategy": str(candidate.grasp_strategy),
        "pick_plan_ok": evaluation.pick_plan_ok,
        "cartesian_descend_ok": evaluation.cartesian_descend_ok,
        "contact_policy_ok": evaluation.contact_policy_ok,
        "lift_precheck_ok": evaluation.lift_precheck_ok,
        "local_escape_ok": evaluation.local_escape_ok,
        "transport_entry_ok": evaluation.transport_entry_ok,
        "transport_route_ok": evaluation.transport_route_ok,
        "place_precheck_ok": evaluation.place_precheck_ok,
        "return_home_precheck_ok": evaluation.return_home_precheck_ok,
        "selected_local_exit": str(candidate.selected_local_exit),
        "selected_transport_route": str(candidate.selected_transport_route),
        "joint_distance_post_lift_to_hub": float(metrics.joint_distance_post_lift_to_hub),
        "joint_distance_total": float(metrics.joint_distance_total),
        "wrist_twist_score": float(metrics.wrist_twist_score),
        "joint6_delta": float(metrics.joint6_delta),
        "joint7_delta": float(metrics.joint7_delta),
        "requires_unwind": metrics.requires_unwind,
        "number_of_segments": int(metrics.number_of_segments),
        "number_of_fallbacks": int(metrics.number_of_fallbacks),
        "estimated_pick_time_s": float(metrics.estimated_pick_time_s),
        "estimated_lift_time_s": float(metrics.estimated_lift_time_s),
        "estimated_transport_entry_time_s": float(metrics.estimated_transport_entry_time_s),
        "estimated_transport_route_time_s": float(metrics.estimated_transport_route_time_s),
        "estimated_place_time_s": float(metrics.estimated_place_time_s),
        "estimated_return_home_time_s": float(metrics.estimated_return_home_time_s),
        "estimated_total_time_s": float(metrics.estimated_total_time_s),
        "score": float(metrics.score),
        "result": evaluation.result,
        "rejection_reason": evaluation.rejection_reason,
        "selected": bool(selected),
    }
    return rec


def compare_candidate_scores(
    records: Sequence[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    valid = [
        r
        for r in records
        if str(r.get("result", "")).upper().startswith("VALID")
        or str(r.get("result", "")).upper().startswith("EXECUTED_OK")
    ]
    if not valid:
        return None

    def _key(r: Dict[str, Any]) -> tuple:
        return (
            float(r.get("score", 1e9)),
            float(r.get("estimated_total_time_s", 1e9)),
            float(r.get("joint_distance_post_lift_to_hub", 1e9)),
            int(r.get("candidate_id", 1e9)),
        )

    return min(valid, key=_key)


def select_top_k_candidates(
    records: Sequence[Dict[str, Any]],
    k: int,
) -> List[Dict[str, Any]]:
    valid = [
        r
        for r in records
        if str(r.get("result", "")).upper().startswith("VALID")
        or str(r.get("result", "")).upper().startswith("EXECUTED_OK")
    ]
    valid.sort(
        key=lambda r: (
            float(r.get("score", 1e9)),
            float(r.get("estimated_total_time_s", 1e9)),
        )
    )
    return valid[: max(0, int(k))]


def validation_summary_stats(
    timings: Sequence[CandidateRealTiming],
) -> Dict[str, Any]:
    ok_runs = [t for t in timings if str(t.result).upper() == "OK"]
    totals = [float(t.total_real_s) for t in ok_runs]
    if not totals:
        return {
            "repetitions": len(timings),
            "success_count": 0,
            "fail_count": len(timings),
            "mean_total_real_s": 0.0,
            "std_total_real_s": 0.0,
            "min_total_real_s": 0.0,
            "max_total_real_s": 0.0,
            "obstacle_disturbed_count": sum(1 for t in timings if t.obstacle_disturbed),
            "fallback_count": sum(t.fallback_count for t in timings),
            "unwind_count": sum(t.unwind_count for t in timings),
            "result": "REJECTED",
        }
    mean = sum(totals) / len(totals)
    var = sum((x - mean) ** 2 for x in totals) / max(1, len(totals))
    return {
        "repetitions": len(timings),
        "success_count": len(ok_runs),
        "fail_count": len(timings) - len(ok_runs),
        "mean_total_real_s": float(mean),
        "std_total_real_s": float(math.sqrt(var)),
        "min_total_real_s": float(min(totals)),
        "max_total_real_s": float(max(totals)),
        "obstacle_disturbed_count": sum(1 for t in timings if t.obstacle_disturbed),
        "fallback_count": sum(t.fallback_count for t in timings),
        "unwind_count": sum(t.unwind_count for t in timings),
        "result": "VALIDATED_FULL_EXECUTION" if len(ok_runs) == len(timings) else "REJECTED",
    }
