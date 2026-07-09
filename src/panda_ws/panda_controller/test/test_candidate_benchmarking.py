"""Tests benchmark genérico de candidatos."""

from __future__ import annotations

import os

from panda_controller.candidate_benchmarking import (
    benchmark_active_for_object,
    build_score_record_from_try_one,
    finalize_benchmark_selection,
    iter_benchmark_grid_specs,
    load_benchmark_config,
)
from panda_controller.candidate_metrics import (
    compute_candidate_score,
    evaluation_from_try_one,
    metrics_from_transport_score,
    select_top_k_candidates,
)


def test_load_demo_scene_02_benchmark_config() -> None:
    cfg = load_benchmark_config("demo_scene_02")
    assert "cracker_box" in cfg
    assert cfg["cracker_box"]["enabled"] is True


def test_iter_benchmark_grid_specs_cracker() -> None:
    cfg = load_benchmark_config("demo_scene_02")
    obj = cfg["cracker_box"]
    specs = iter_benchmark_grid_specs(
        scene_id="demo_scene_02",
        target_label="cracker_box",
        slot_index=0,
        xy=(0.455, 0.115),
        top_z=0.47,
        base_yaw_rad=1.34,
        object_config=obj,
    )
    assert len(specs) > 0
    assert specs[0]["scene_id"] == "demo_scene_02"


def test_build_score_record_valid() -> None:
    spec = {
        "grid_idx": 0,
        "yaw_deg": 77.0,
        "yaw_rad": 1.34,
        "pregrasp_tcp_z": 0.575,
        "grasp_tcp_z": 0.437,
        "depth_from_top_m": 0.033,
        "ik_seed_label": "pick_workspace_ready",
        "pre_plan": (0.455, 0.115, 0.575),
        "gr_plan": (0.455, 0.115, 0.437),
    }
    rec = build_score_record_from_try_one(
        spec=spec,
        scene_id="demo_scene_02",
        target_label="cracker_box",
        slot_index=0,
        ik_pregrasp_ok=True,
        plan_to_pregrasp_ok=True,
        cart_ok=True,
        lift_ok=True,
        local_escape_ok=True,
        global_route_ok=True,
        reject_reason="",
        transport_score={
            "joint_distance_to_hub": 0.5,
            "wrist_twist_score": 0.2,
            "route": ["carry_mid_high", "box_high"],
        },
        penalties={"unwind": 8.0},
    )
    assert rec["result"] == "VALID"
    assert float(rec["score"]) > 0.0


def test_finalize_benchmark_selection_top_k() -> None:
    records = []
    for i, score in enumerate([12.0, 8.0, 15.0]):
        records.append(
            build_score_record_from_try_one(
                spec={
                    "grid_idx": i,
                    "yaw_deg": 10.0 * i,
                    "pregrasp_tcp_z": 0.575,
                    "grasp_tcp_z": 0.437,
                    "depth_from_top_m": 0.033,
                    "ik_seed_label": "seed",
                    "pre_plan": (0, 0, 0.575),
                    "gr_plan": (0, 0, 0.437),
                },
                scene_id="demo_scene_02",
                target_label="cracker_box",
                slot_index=0,
                ik_pregrasp_ok=True,
                plan_to_pregrasp_ok=True,
                cart_ok=True,
                lift_ok=True,
                local_escape_ok=True,
                global_route_ok=True,
                reject_reason="",
                transport_score={"joint_distance_to_hub": score, "wrist_twist_score": 0.1},
            )
        )
    best, top, summary = finalize_benchmark_selection(
        records,
        scene_id="demo_scene_02",
        target_label="cracker_box",
        slot_index=0,
        top_k=2,
    )
    assert best is not None
    assert int(best["candidate_id"]) == 1
    assert len(top) == 2
    assert summary["valid_candidates"] == 3


def test_benchmark_active_for_cracker_slot_0() -> None:
    cfg = load_benchmark_config("demo_scene_02")
    assert benchmark_active_for_object(
        scene_id="demo_scene_02",
        target_label="cracker_box",
        slot_index=0,
        config=cfg,
    )
    assert not benchmark_active_for_object(
        scene_id="demo_scene_02",
        target_label="chips_can",
        slot_index=0,
        config=cfg,
    )
