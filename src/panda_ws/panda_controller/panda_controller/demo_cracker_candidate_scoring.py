"""Compatibilidad cracker_box → benchmark genérico (delegación a candidate_*)."""

from __future__ import annotations

import csv
import os
from typing import Any, Dict, List, Optional, Sequence

from panda_controller.candidate_benchmarking import (
    SCORE_CSV_FIELDS,
    build_score_record_from_try_one,
    finalize_benchmark_selection,
    format_candidate_score_log,
    format_candidate_selection_summary_log,
)
from panda_controller.candidate_metrics import compare_candidate_scores

CRACKER_CANDIDATE_SCORE_CSV_PATH = "/tmp/tfg_cracker_candidate_scores.csv"


def build_cracker_candidate_score_record(
    *,
    spec: Dict[str, Any],
    pick_ok: bool = False,
    descend_ok: bool = False,
    lift_ok: bool = False,
    local_escape_ok: bool = False,
    transport_ok: bool = False,
    place_ok: bool = False,
    return_home_ok: bool = False,
    joint_distance_to_hub: float = float("inf"),
    wrist_twist_score: float = float("inf"),
    estimated_pick_time_s: float = 0.0,
    estimated_transport_time_s: float = 0.0,
    estimated_place_time_s: float = 0.0,
    rejection_reason: str = "",
    selected: bool = False,
    scene_id: str = "demo_scene_02",
    target_label: str = "cracker_box",
    slot_index: int = 0,
    transport_score: Optional[Dict[str, Any]] = None,
    penalties: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    ts = dict(transport_score or {})
    ts.setdefault("joint_distance_to_hub", joint_distance_to_hub)
    ts.setdefault("wrist_twist_score", wrist_twist_score)
    rec = build_score_record_from_try_one(
        spec=spec,
        scene_id=scene_id,
        target_label=target_label,
        slot_index=slot_index,
        ik_pregrasp_ok=bool(pick_ok),
        plan_to_pregrasp_ok=bool(pick_ok),
        cart_ok=bool(descend_ok),
        lift_ok=bool(lift_ok),
        local_escape_ok=bool(local_escape_ok or transport_ok),
        global_route_ok=bool(transport_ok),
        reject_reason=rejection_reason,
        transport_score=ts,
        penalties=penalties,
        selected=selected,
    )
    if place_ok:
        rec["place_precheck_ok"] = True
    if return_home_ok:
        rec["return_home_precheck_ok"] = True
    if (
        rec.get("pick_plan_ok")
        and rec.get("cartesian_descend_ok")
        and rec.get("lift_precheck_ok")
        and rec.get("local_escape_ok")
        and rec.get("transport_route_ok")
        and not rejection_reason
    ):
        rec["result"] = "VALID"
    rec["candidate_idx"] = rec.get("candidate_id")
    rec["depth_from_top"] = rec.get("depth_from_top_m")
    rec["ik_seed"] = rec.get("ik_seed_name")
    rec["pick_ok"] = rec.get("pick_plan_ok")
    rec["descend_ok"] = rec.get("cartesian_descend_ok")
    rec["lift_ok"] = rec.get("lift_precheck_ok")
    rec["transport_ok"] = rec.get("transport_route_ok")
    rec["place_ok"] = place_ok
    rec["return_home_ok"] = return_home_ok
    rec["joint_distance_to_hub"] = rec.get("joint_distance_post_lift_to_hub")
    rec["estimated_transport_time_s"] = rec.get("estimated_transport_route_time_s")
    return rec


compare_cracker_candidate_scores = compare_candidate_scores


def format_cracker_candidate_score_log(record: Dict[str, Any]) -> str:
    return format_candidate_score_log(record)


def format_cracker_candidate_selection_summary_log(fields: Dict[str, Any]) -> str:
    mapped = dict(fields)
    if "selected_candidate_idx" in mapped and "selected_candidate_id" not in mapped:
        mapped["selected_candidate_id"] = mapped["selected_candidate_idx"]
    return format_candidate_selection_summary_log(mapped)


def export_cracker_candidate_scores_csv(
    records: Sequence[Dict[str, Any]],
    *,
    csv_path: str = CRACKER_CANDIDATE_SCORE_CSV_PATH,
) -> str:
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    with open(csv_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(SCORE_CSV_FIELDS), extrasaction="ignore"
        )
        writer.writeheader()
        for rec in records:
            writer.writerow({field: rec.get(field, "") for field in SCORE_CSV_FIELDS})
    return csv_path
