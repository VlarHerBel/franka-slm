"""Benchmark genérico de candidatos manipulación (object-agnostic)."""

from __future__ import annotations

import csv
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import yaml

from panda_controller.candidate_metrics import (
    CandidateEvaluationResult,
    CandidateMetrics,
    CandidateRealTiming,
    ManipulationCandidate,
    build_candidate_score_record,
    compare_candidate_scores,
    compute_candidate_score,
    evaluation_from_try_one,
    metrics_from_transport_score,
    select_top_k_candidates,
    validation_summary_stats,
)
from panda_controller.demo_golden_pick_candidate import default_demo_config_dir

CANDIDATE_SCORE_CSV_TEMPLATE = "/tmp/tfg_candidate_scores_{scene_id}_{target_label}.csv"
CANDIDATE_SUMMARY_CSV_TEMPLATE = "/tmp/tfg_candidate_summary_{scene_id}.csv"
CANDIDATE_REAL_TIMING_CSV_TEMPLATE = (
    "/tmp/tfg_candidate_real_timings_{scene_id}_{target_label}.csv"
)

SCORE_CSV_FIELDS: Tuple[str, ...] = (
    "scene_id",
    "target_label",
    "slot_index",
    "candidate_id",
    "yaw_deg",
    "pregrasp_tcp_z",
    "grasp_tcp_z",
    "depth_from_top_m",
    "ik_seed_name",
    "grasp_strategy",
    "pick_plan_ok",
    "cartesian_descend_ok",
    "contact_policy_ok",
    "lift_precheck_ok",
    "local_escape_ok",
    "transport_entry_ok",
    "transport_route_ok",
    "transport_deferred_ok",
    "legacy_acceptance_ok",
    "final_candidate_executed",
    "place_precheck_ok",
    "return_home_precheck_ok",
    "selected_local_exit",
    "selected_transport_route",
    "joint_distance_post_lift_to_hub",
    "joint_distance_total",
    "wrist_twist_score",
    "joint6_delta",
    "joint7_delta",
    "requires_unwind",
    "number_of_segments",
    "number_of_fallbacks",
    "estimated_pick_time_s",
    "estimated_lift_time_s",
    "estimated_transport_entry_time_s",
    "estimated_transport_route_time_s",
    "estimated_place_time_s",
    "estimated_return_home_time_s",
    "estimated_total_time_s",
    "score",
    "result",
    "execution_result",
    "rejection_reason",
    "selected",
)

REAL_TIMING_CSV_FIELDS: Tuple[str, ...] = (
    "scene_id",
    "target_label",
    "slot_index",
    "candidate_id",
    "yaw_deg",
    "pregrasp_tcp_z",
    "grasp_tcp_z",
    "selected_local_exit",
    "selected_transport_route",
    "pick_real_s",
    "lift_real_s",
    "transport_entry_real_s",
    "transport_route_real_s",
    "place_real_s",
    "return_home_real_s",
    "total_real_s",
    "obstacle_disturbed",
    "fallback_count",
    "unwind_count",
    "execution_result",
    "place_had_cartesian_fail",
    "place_release_attempts",
    "selected_release_tcp_z",
    "mode_completed",
    "place_completed",
    "return_home_ok",
    "failure_reason",
)

SUMMARY_CSV_FIELDS: Tuple[str, ...] = (
    "scene_id",
    "target_label",
    "slot_index",
    "total_candidates",
    "total_candidates_evaluated",
    "total_candidates_executed",
    "valid_candidates",
    "rejected_candidates",
    "selected_candidate_id",
    "selected_score",
    "selected_estimated_total_time_s",
    "selected_reason",
    "top_k_candidates",
    "execution_result",
    "total_real_s",
    "selected_release_tcp_z",
    "fallback_count",
    "golden_status",
)


def default_benchmark_config_dir() -> str:
    fallback = str(
        Path(__file__).resolve().parent.parent / "config" / "candidate_benchmarking"
    )
    try:
        from ament_index_python.packages import get_package_share_directory

        share = os.path.join(
            get_package_share_directory("panda_controller"),
            "config",
            "candidate_benchmarking",
        )
        if os.path.isdir(share):
            return share
    except Exception:
        pass
    return fallback


def benchmark_config_path(scene_id: str, *, config_dir: Optional[str] = None) -> str:
    sid = str(scene_id or "").strip().lower()
    base = str(config_dir or default_benchmark_config_dir())
    return os.path.join(base, f"{sid}.yaml")


def load_benchmark_config(
    scene_id: str,
    *,
    config_dir: Optional[str] = None,
) -> Dict[str, Any]:
    path = benchmark_config_path(scene_id, config_dir=config_dir)
    if not os.path.isfile(path):
        return {}
    with open(path, "r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    return dict(raw) if isinstance(raw, dict) else {}


def get_object_benchmark_config(
    config: Dict[str, Any],
    target_label: str,
) -> Optional[Dict[str, Any]]:
    label = str(target_label or "").strip().lower()
    obj = config.get(label)
    if not isinstance(obj, dict):
        return None
    if not bool(obj.get("enabled", False)):
        return None
    return dict(obj)


def benchmark_active_for_object(
    *,
    scene_id: str,
    target_label: str,
    slot_index: int,
    config: Optional[Dict[str, Any]] = None,
) -> bool:
    cfg = config if config is not None else load_benchmark_config(scene_id)
    obj = get_object_benchmark_config(cfg, target_label)
    if obj is None:
        return False
    req_slot = obj.get("slot_index")
    if req_slot is not None and int(req_slot) != int(slot_index):
        return False
    return True


def _wrap_to_pi(angle: float) -> float:
    a = float(angle)
    while a > math.pi:
        a -= 2.0 * math.pi
    while a < -math.pi:
        a += 2.0 * math.pi
    return a


def build_yaw_variants_rad(
    base_yaw_rad: float,
    yaw_offsets_deg: Sequence[float],
    *,
    include_pi_flip: bool = True,
) -> List[Tuple[str, float]]:
    base = _wrap_to_pi(float(base_yaw_rad))
    out: List[Tuple[str, float]] = []
    seen: set = set()
    pi_vals = (0.0, math.pi) if include_pi_flip else (0.0,)

    def _add(name: str, yaw_rad: float) -> None:
        y = _wrap_to_pi(yaw_rad)
        key = round(y, 5)
        if key in seen:
            return
        seen.add(key)
        out.append((str(name), y))

    for pi_add in pi_vals:
        root = _wrap_to_pi(base + pi_add)
        tag = "yaw" if abs(pi_add) < 1e-6 else "yaw_pi"
        _add(tag, root)
        for off_deg in yaw_offsets_deg:
            if abs(float(off_deg)) < 1e-9:
                continue
            off = math.radians(float(off_deg))
            _add("%s_%+.0fdeg" % (tag, float(off_deg)), root + off)
    return out


def iter_benchmark_grid_specs(
    *,
    scene_id: str,
    target_label: str,
    slot_index: int,
    xy: Tuple[float, float],
    top_z: float,
    base_yaw_rad: float,
    object_config: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Genera specs de grid desde configuración por objeto."""
    pregrasp_zs = tuple(
        float(z) for z in (object_config.get("pregrasp_tcp_z_values") or [0.575])
    )
    depths = tuple(
        float(d) for d in (object_config.get("depth_from_top_values") or [0.033])
    )
    yaw_offsets = tuple(
        float(y) for y in (object_config.get("yaw_variants_deg") or [0.0])
    )
    ik_seeds = tuple(
        str(s) for s in (object_config.get("ik_seeds") or ["pick_workspace_ready"])
    )
    yaws = build_yaw_variants_rad(base_yaw_rad, yaw_offsets)
    specs: List[Dict[str, Any]] = []
    idx = 0
    for pre_z in pregrasp_zs:
        for depth in depths:
            grasp_z = float(top_z) - float(depth)
            pre_plan = (float(xy[0]), float(xy[1]), float(pre_z))
            gr_plan = (float(xy[0]), float(xy[1]), float(grasp_z))
            for yaw_name, yaw_rad in yaws:
                for seed_label in ik_seeds:
                    specs.append(
                        {
                            "grid_idx": int(idx),
                            "candidate_id": int(idx),
                            "yaw_name": str(yaw_name),
                            "yaw_rad": float(yaw_rad),
                            "yaw_deg": math.degrees(float(yaw_rad)),
                            "pregrasp_tcp_z": float(pre_z),
                            "grasp_tcp_z": float(grasp_z),
                            "depth_from_top_m": float(depth),
                            "pre_plan": pre_plan,
                            "gr_plan": gr_plan,
                            "ik_seed_label": str(seed_label),
                            "scene_id": str(scene_id),
                            "target_label": str(target_label),
                            "slot_index": int(slot_index),
                        }
                    )
                    idx += 1
    return specs


def score_csv_path(scene_id: str, target_label: str) -> str:
    return CANDIDATE_SCORE_CSV_TEMPLATE.format(
        scene_id=str(scene_id or "").strip().lower(),
        target_label=str(target_label or "").strip().lower(),
    )


def summary_csv_path(scene_id: str) -> str:
    return CANDIDATE_SUMMARY_CSV_TEMPLATE.format(
        scene_id=str(scene_id or "").strip().lower(),
    )


def real_timing_csv_path(scene_id: str, target_label: str) -> str:
    return CANDIDATE_REAL_TIMING_CSV_TEMPLATE.format(
        scene_id=str(scene_id or "").strip().lower(),
        target_label=str(target_label or "").strip().lower(),
    )


def build_score_record_from_try_one(
    *,
    spec: Dict[str, Any],
    scene_id: str,
    target_label: str,
    slot_index: int,
    ik_pregrasp_ok: bool,
    plan_to_pregrasp_ok: bool,
    cart_ok: bool,
    lift_ok: bool,
    local_escape_ok: bool,
    global_route_ok: bool,
    reject_reason: str,
    transport_score: Optional[Dict[str, Any]] = None,
    penalties: Optional[Dict[str, float]] = None,
    selected: bool = False,
) -> Dict[str, Any]:
    cand = ManipulationCandidate(
        candidate_id=int(spec.get("grid_idx", spec.get("candidate_id", -1))),
        scene_id=str(scene_id),
        target_label=str(target_label),
        slot_index=int(slot_index),
        yaw_variant_rad=float(spec.get("yaw_rad", 0.0)),
        yaw_variant_deg=float(spec.get("yaw_deg", 0.0)),
        pregrasp_tcp_z=float(spec.get("pregrasp_tcp_z", 0.0)),
        grasp_tcp_z=float(spec.get("grasp_tcp_z", 0.0)),
        depth_from_top_m=float(spec.get("depth_from_top_m", 0.0)),
        ik_seed_name=str(spec.get("ik_seed_label", "")),
        selected_local_exit=str(
            (transport_score or {}).get("selected_local_exit", "")
            or (transport_score or {}).get("selected_transport_mode", "")
        ),
        selected_transport_route=",".join(
            str(x)
            for x in (
                (transport_score or {}).get("route")
                or (transport_score or {}).get("selected_transport_route")
                or []
            )
        ),
        pre_plan=tuple(spec.get("pre_plan") or (0, 0, 0)),
        gr_plan=tuple(spec.get("gr_plan") or (0, 0, 0)),
    )
    ev = evaluation_from_try_one(
        ik_pregrasp_ok=ik_pregrasp_ok,
        plan_to_pregrasp_ok=plan_to_pregrasp_ok,
        cart_ok=cart_ok,
        lift_ok=lift_ok,
        local_escape_ok=local_escape_ok,
        global_route_ok=global_route_ok,
        reject_reason=reject_reason,
    )
    metrics = metrics_from_transport_score(transport_score)
    extra: Dict[str, float] = {}
    route_str = cand.selected_transport_route
    if "carry_front_high" in route_str:
        extra["carry_front_high"] = float(
            (penalties or {}).get("carry_front_high", 6.0)
        )
    if metrics.requires_unwind:
        extra["requires_unwind_extra"] = 0.0
    return build_candidate_score_record(
        cand,
        ev,
        metrics,
        selected=selected,
        penalties=penalties,
        extra_penalties=extra or None,
    )


def format_candidate_score_log(record: Dict[str, Any]) -> str:
    return (
        "[CANDIDATE_SCORE]\n"
        "scene_id=%s\n"
        "target_label=%s\n"
        "slot_index=%s\n"
        "candidate_id=%s\n"
        "yaw_deg=%.2f\n"
        "pregrasp_tcp_z=%.4f\n"
        "grasp_tcp_z=%.4f\n"
        "depth_from_top_m=%.4f\n"
        "ik_seed_name=%s\n"
        "grasp_strategy=%s\n"
        "pick_plan_ok=%s\n"
        "cartesian_descend_ok=%s\n"
        "contact_policy_ok=%s\n"
        "lift_precheck_ok=%s\n"
        "local_escape_ok=%s\n"
        "transport_entry_ok=%s\n"
        "transport_route_ok=%s\n"
        "transport_deferred_ok=%s\n"
        "legacy_acceptance_ok=%s\n"
        "final_candidate_executed=%s\n"
        "place_precheck_ok=%s\n"
        "return_home_precheck_ok=%s\n"
        "selected_local_exit=%s\n"
        "selected_transport_route=%s\n"
        "joint_distance_post_lift_to_hub=%.4f\n"
        "joint_distance_total=%.4f\n"
        "wrist_twist_score=%.4f\n"
        "joint6_delta=%.4f\n"
        "joint7_delta=%.4f\n"
        "requires_unwind=%s\n"
        "number_of_segments=%s\n"
        "number_of_fallbacks=%s\n"
        "estimated_pick_time_s=%.3f\n"
        "estimated_lift_time_s=%.3f\n"
        "estimated_transport_entry_time_s=%.3f\n"
        "estimated_transport_route_time_s=%.3f\n"
        "estimated_place_time_s=%.3f\n"
        "estimated_return_home_time_s=%.3f\n"
        "estimated_total_time_s=%.3f\n"
        "score=%.4f\n"
        "result=%s\n"
        "execution_result=%s\n"
        "rejection_reason=%s"
        % (
            record.get("scene_id", ""),
            record.get("target_label", ""),
            record.get("slot_index", ""),
            record.get("candidate_id", "n/a"),
            float(record.get("yaw_deg", 0.0)),
            float(record.get("pregrasp_tcp_z", 0.0)),
            float(record.get("grasp_tcp_z", 0.0)),
            float(record.get("depth_from_top_m", 0.0)),
            record.get("ik_seed_name", ""),
            record.get("grasp_strategy", ""),
            str(record.get("pick_plan_ok", False)).lower(),
            str(record.get("cartesian_descend_ok", False)).lower(),
            str(record.get("contact_policy_ok", False)).lower(),
            str(record.get("lift_precheck_ok", False)).lower(),
            str(record.get("local_escape_ok", False)).lower(),
            str(record.get("transport_entry_ok", False)).lower(),
            str(record.get("transport_route_ok", False)).lower(),
            str(record.get("transport_deferred_ok", False)).lower(),
            str(record.get("legacy_acceptance_ok", False)).lower(),
            str(record.get("final_candidate_executed", False)).lower(),
            str(record.get("place_precheck_ok", False)).lower(),
            str(record.get("return_home_precheck_ok", False)).lower(),
            record.get("selected_local_exit", ""),
            record.get("selected_transport_route", ""),
            float(record.get("joint_distance_post_lift_to_hub", 0.0)),
            float(record.get("joint_distance_total", 0.0)),
            float(record.get("wrist_twist_score", 0.0)),
            float(record.get("joint6_delta", 0.0)),
            float(record.get("joint7_delta", 0.0)),
            str(record.get("requires_unwind", False)).lower(),
            record.get("number_of_segments", 0),
            record.get("number_of_fallbacks", 0),
            float(record.get("estimated_pick_time_s", 0.0)),
            float(record.get("estimated_lift_time_s", 0.0)),
            float(record.get("estimated_transport_entry_time_s", 0.0)),
            float(record.get("estimated_transport_route_time_s", 0.0)),
            float(record.get("estimated_place_time_s", 0.0)),
            float(record.get("estimated_return_home_time_s", 0.0)),
            float(record.get("estimated_total_time_s", 0.0)),
            float(record.get("score", 0.0)),
            record.get("result", "REJECT"),
            record.get("execution_result", ""),
            record.get("rejection_reason", ""),
        )
    )


def format_candidate_selection_summary_log(fields: Dict[str, Any]) -> str:
    top_k = fields.get("top_k_candidates") or []
    top_s = ",".join(str(x) for x in top_k) if top_k else ""
    return (
        "[CANDIDATE_SELECTION_SUMMARY]\n"
        "scene_id=%s\n"
        "target_label=%s\n"
        "slot_index=%s\n"
        "total_candidates=%s\n"
        "valid_candidates=%s\n"
        "rejected_candidates=%s\n"
        "selected_candidate_id=%s\n"
        "selected_score=%.4f\n"
        "selected_estimated_total_time_s=%.3f\n"
        "selected_reason=%s\n"
        "top_k_candidates=[%s]"
        % (
            fields.get("scene_id", ""),
            fields.get("target_label", ""),
            fields.get("slot_index", ""),
            fields.get("total_candidates", 0),
            fields.get("valid_candidates", 0),
            fields.get("rejected_candidates", 0),
            fields.get("selected_candidate_id", "n/a"),
            float(fields.get("selected_score", 0.0)),
            float(fields.get("selected_estimated_total_time_s", 0.0)),
            fields.get("selected_reason", ""),
            top_s,
        )
    )


def format_candidate_real_timing_log(
    *,
    scene_id: str,
    target_label: str,
    slot_index: int,
    candidate_id: int,
    timing: CandidateRealTiming,
) -> str:
    return (
        "[CANDIDATE_REAL_TIMING]\n"
        "scene_id=%s\n"
        "target_label=%s\n"
        "slot_index=%s\n"
        "candidate_id=%s\n"
        "pick_real_s=%.3f\n"
        "lift_real_s=%.3f\n"
        "transport_entry_real_s=%.3f\n"
        "transport_route_real_s=%.3f\n"
        "place_real_s=%.3f\n"
        "return_home_real_s=%.3f\n"
        "total_real_s=%.3f\n"
        "obstacle_disturbed=%s\n"
        "fallback_count=%s\n"
        "unwind_count=%s\n"
        "execution_result=%s\n"
        "failure_reason=%s"
        % (
            scene_id,
            target_label,
            slot_index,
            candidate_id,
            float(timing.pick_real_s),
            float(timing.lift_real_s),
            float(timing.transport_entry_real_s),
            float(timing.transport_route_real_s),
            float(timing.place_real_s),
            float(timing.return_home_real_s),
            float(timing.total_real_s),
            str(timing.obstacle_disturbed).lower(),
            int(timing.fallback_count),
            int(timing.unwind_count),
            timing.result,
            timing.failure_reason,
        )
    )


def format_golden_validation_summary_log(fields: Dict[str, Any]) -> str:
    return (
        "[GOLDEN_VALIDATION_SUMMARY]\n"
        "scene_id=%s\n"
        "target_label=%s\n"
        "slot_index=%s\n"
        "candidate_id=%s\n"
        "repetitions=%s\n"
        "success_count=%s\n"
        "fail_count=%s\n"
        "mean_total_real_s=%.3f\n"
        "std_total_real_s=%.3f\n"
        "min_total_real_s=%.3f\n"
        "max_total_real_s=%.3f\n"
        "obstacle_disturbed_count=%s\n"
        "fallback_count=%s\n"
        "unwind_count=%s\n"
        "result=%s"
        % (
            fields.get("scene_id", ""),
            fields.get("target_label", ""),
            fields.get("slot_index", ""),
            fields.get("candidate_id", ""),
            fields.get("repetitions", 0),
            fields.get("success_count", 0),
            fields.get("fail_count", 0),
            float(fields.get("mean_total_real_s", 0.0)),
            float(fields.get("std_total_real_s", 0.0)),
            float(fields.get("min_total_real_s", 0.0)),
            float(fields.get("max_total_real_s", 0.0)),
            fields.get("obstacle_disturbed_count", 0),
            fields.get("fallback_count", 0),
            fields.get("unwind_count", 0),
            fields.get("result", "REJECTED"),
        )
    )


def export_candidate_scores_csv(
    records: Sequence[Dict[str, Any]],
    *,
    scene_id: str,
    target_label: str,
) -> str:
    path = score_csv_path(scene_id, target_label)
    _write_csv(path, SCORE_CSV_FIELDS, records, bool_fields=_BOOL_SCORE_FIELDS)
    return path


def export_candidate_summary_csv(
    summary: Dict[str, Any],
    *,
    scene_id: str,
) -> str:
    path = summary_csv_path(scene_id)
    _write_csv(path, SUMMARY_CSV_FIELDS, [summary])
    return path


def export_candidate_real_timings_csv(
    records: Sequence[Dict[str, Any]],
    *,
    scene_id: str,
    target_label: str,
) -> str:
    path = real_timing_csv_path(scene_id, target_label)
    _write_csv(
        path,
        REAL_TIMING_CSV_FIELDS,
        records,
        bool_fields=(
            "obstacle_disturbed",
            "place_had_cartesian_fail",
            "mode_completed",
            "place_completed",
            "return_home_ok",
        ),
    )
    return path


_BOOL_SCORE_FIELDS: Tuple[str, ...] = (
    "pick_plan_ok",
    "cartesian_descend_ok",
    "contact_policy_ok",
    "lift_precheck_ok",
    "local_escape_ok",
    "transport_entry_ok",
    "transport_route_ok",
    "transport_deferred_ok",
    "legacy_acceptance_ok",
    "final_candidate_executed",
    "place_precheck_ok",
    "return_home_precheck_ok",
    "requires_unwind",
    "selected",
)


def _write_csv(
    path: str,
    fields: Sequence[str],
    records: Sequence[Dict[str, Any]],
    *,
    bool_fields: Sequence[str] = (),
) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        for rec in records:
            row = {k: rec.get(k, "") for k in fields}
            for bk in bool_fields:
                if bk in row:
                    row[bk] = str(bool(rec.get(bk))).lower()
            writer.writerow(row)


def finalize_benchmark_selection(
    records: Sequence[Dict[str, Any]],
    *,
    scene_id: str,
    target_label: str,
    slot_index: int,
    top_k: int = 1,
) -> Tuple[Optional[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    best = compare_candidate_scores(records)
    top = select_top_k_candidates(records, top_k)
    for rec in records:
        rec["selected"] = best is not None and int(rec.get("candidate_id", -2)) == int(
            best.get("candidate_id", -1)
        )
    valid_count = sum(
        1
        for r in records
        if str(r.get("result", "")).upper().startswith("VALID")
        or str(r.get("result", "")).upper().startswith("EXECUTED_OK")
    )
    summary = {
        "scene_id": scene_id,
        "target_label": target_label,
        "slot_index": slot_index,
        "total_candidates": len(records),
        "total_candidates_evaluated": len(records),
        "total_candidates_executed": sum(
            1 for r in records if bool(r.get("final_candidate_executed"))
        ),
        "valid_candidates": valid_count,
        "rejected_candidates": len(records) - valid_count,
        "selected_candidate_id": best.get("candidate_id") if best else "n/a",
        "selected_score": float((best or {}).get("score", 0.0)),
        "selected_estimated_total_time_s": float(
            (best or {}).get("estimated_total_time_s", 0.0)
        ),
        "selected_reason": (
            "lowest_configurable_score"
            if best
            else "no_valid_candidate"
        ),
        "top_k_candidates": [int(t.get("candidate_id", -1)) for t in top],
        "execution_result": "",
        "total_real_s": "",
        "selected_release_tcp_z": "",
        "fallback_count": "",
        "golden_status": "",
    }
    return best, top, summary
