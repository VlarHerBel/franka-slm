#!/usr/bin/env python3
"""Validador secuencial de layouts demo: certifica escenas de 4 objetos con golden."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import yaml

from panda_controller.demo_scene_reachability_scan import (
    DEBUG_CALIBRATION_CELLS,
    DEMO_SCAN_LABELS,
    DEFAULT_CARTESIAN_FRACTION_THRESHOLD,
    DEFAULT_JOINT_LIMIT_MARGIN_MIN_RAD,
    DEFAULT_TABLE_Z_M,
    ReachabilityMoveItBackend,
    ReachabilityScanConfig,
    ScanCellResult,
    VARIANT_SEARCH_DEBUG_LABELS,
    aggregate_budgeted_cell_results,
    apply_reachability_coordinate_metadata,
    build_detection_for_reachability_cell,
    default_input_mode_for_label,
    format_reachability_operational_center_log,
    format_variant_search_summary_log,
    is_binary_reachable_cell,
    iter_budgeted_variant_jobs,
    iter_grasp_variants_for_cell,
    resolve_reachability_cell_coordinates,
    summarize_variant_search_results,
    validate_variant_with_moveit,
    _vision_policy_exports,
)

LAYOUT_VERSION = "v3_clear_table_transport"
CERTIFIED_STATUS = "scanner_certified_sequential"
ROBOT_BASE_XY = (0.0, 0.0)
TABLE_X_RANGE = (0.44, 0.70)
TABLE_Y_RANGE = (-0.20, 0.18)
MIN_OBJECT_SEPARATION_M = 0.03
CHIPS_CAN_DEFAULT_SEED = 1003

OBJECT_FOOTPRINT_LW_M: Dict[str, Tuple[float, float]] = {
    "cracker_box": (0.158, 0.060),
    "sugar_box": (0.089, 0.038),
    "mustard_bottle": (0.121, 0.106),
    "chips_can": (0.075, 0.075),
}

GRASP_STRATEGY_BY_LABEL: Dict[str, str] = {
    "cracker_box": "top_down_short_axis",
    "chips_can": "cylinder_topdown",
    "sugar_box": "top_down",
    "mustard_bottle": "palm_bridge",
}


@dataclass(frozen=True)
class LayoutObjectPose:
    label: str
    spawn_x: float
    spawn_y: float
    yaw: float
    operational_x: float
    operational_y: float
    input_mode: str

    @property
    def robot_distance_m(self) -> float:
        return math.hypot(
            float(self.operational_x) - float(ROBOT_BASE_XY[0]),
            float(self.operational_y) - float(ROBOT_BASE_XY[1]),
        )


@dataclass
class CertifiedPickRecord:
    label: str
    pick_step: int
    pose: LayoutObjectPose
    scan_result: ScanCellResult
    remaining_obstacles: Tuple[str, ...]


@dataclass
class CertifiedLayout:
    scene_id: str
    layout_index: int
    pick_order: Tuple[str, ...]
    objects: Dict[str, LayoutObjectPose]
    picks: Dict[str, CertifiedPickRecord]
    attempts_to_find: int = 0

    def fingerprint(self) -> Tuple[Tuple[str, float, float, float], ...]:
        return tuple(
            (
                lb,
                round(self.objects[lb].spawn_x, 3),
                round(self.objects[lb].spawn_y, 3),
                round(self.objects[lb].yaw, 3),
            )
            for lb in sorted(self.objects.keys())
        )


def layout_pose_from_spawn(
    label: str,
    spawn_x: float,
    spawn_y: float,
    yaw: float,
) -> LayoutObjectPose:
    coords = resolve_reachability_cell_coordinates(
        label,
        float(spawn_x),
        float(spawn_y),
        float(yaw),
        input_mode=default_input_mode_for_label(label),
    )
    return LayoutObjectPose(
        label=str(label),
        spawn_x=float(coords.spawn_x),
        spawn_y=float(coords.spawn_y),
        yaw=float(coords.yaw),
        operational_x=float(coords.operational_grasp_x),
        operational_y=float(coords.operational_grasp_y),
        input_mode=str(coords.input_mode),
    )


def layout_fingerprint_seed(objects: Dict[str, LayoutObjectPose]) -> int:
    """Semilla estable por layout (misma escena → mismo desempate)."""
    fp = tuple(
        (
            lb,
            round(objects[lb].spawn_x, 3),
            round(objects[lb].spawn_y, 3),
            round(objects[lb].yaw, 3),
        )
        for lb in sorted(objects.keys())
    )
    return int(hash(fp) & 0x7FFFFFFF)


def pick_order_distance_breakdown(
    objects: Dict[str, LayoutObjectPose],
) -> List[Tuple[str, float]]:
    return sorted(
        [(str(lb), float(objects[lb].robot_distance_m)) for lb in objects.keys()],
        key=lambda item: (float(item[1]), str(item[0])),
    )


def pick_order_by_robot_proximity(
    objects: Dict[str, LayoutObjectPose],
    *,
    tie_break_seed: Optional[int] = None,
) -> Tuple[str, ...]:
    """Orden dinámico: el más cercano al robot primero (no fijo por label).

    ``tie_break_seed`` desempata distancias iguales de forma pseudoaleatoria pero
    estable por layout (misma escena → mismo orden).
    """
    rng = random.Random(int(tie_break_seed)) if tie_break_seed is not None else None

    def _sort_key(lb: str) -> Tuple[float, float]:
        dist = float(objects[lb].robot_distance_m)
        if rng is not None:
            return (dist, float(rng.random()))
        return (dist, float(objects[lb].spawn_x))

    return tuple(sorted(objects.keys(), key=_sort_key))


def object_collision_radius_m(label: str) -> float:
    l_m, w_m = OBJECT_FOOTPRINT_LW_M.get(str(label), (0.10, 0.10))
    return 0.5 * math.hypot(float(l_m), float(w_m))


def build_scene_obstacle_dict(pose: LayoutObjectPose) -> Dict[str, Any]:
    l_m, w_m = OBJECT_FOOTPRINT_LW_M.get(pose.label, (0.10, 0.10))
    return {
        "label": pose.label,
        "x": float(pose.operational_x),
        "y": float(pose.operational_y),
        "footprint_major_m": float(l_m),
        "footprint_minor_m": float(w_m),
        "collision_dims": {"box": [float(l_m), float(w_m), 0.12]},
    }


def layout_within_table_bounds(objects: Dict[str, LayoutObjectPose]) -> bool:
    for pose in objects.values():
        if not (
            TABLE_X_RANGE[0] <= pose.spawn_x <= TABLE_X_RANGE[1]
            and TABLE_Y_RANGE[0] <= pose.spawn_y <= TABLE_Y_RANGE[1]
        ):
            return False
    return True


def layout_has_minimum_separation(objects: Dict[str, LayoutObjectPose]) -> bool:
    labels = list(objects.keys())
    for i, la in enumerate(labels):
        pa = objects[la]
        for lb in labels[i + 1 :]:
            pb = objects[lb]
            dist = math.hypot(
                pa.operational_x - pb.operational_x,
                pa.operational_y - pb.operational_y,
            )
            required = (
                object_collision_radius_m(la)
                + object_collision_radius_m(lb)
                + MIN_OBJECT_SEPARATION_M
            )
            if dist + 1e-6 < required:
                return False
    return True


def layout_is_physically_plausible(
    objects: Dict[str, LayoutObjectPose],
    *,
    require_separation: bool = True,
) -> bool:
    if not layout_within_table_bounds(objects):
        return False
    if require_separation and not layout_has_minimum_separation(objects):
        return False
    return True


def is_canonical_demo_scene_02_layout(objects: Dict[str, LayoutObjectPose]) -> bool:
    canonical = canonical_demo_scene_02_layout()
    for label, pose in canonical.items():
        other = objects.get(label)
        if other is None:
            return False
        if (
            abs(other.spawn_x - pose.spawn_x) > 1e-4
            or abs(other.spawn_y - pose.spawn_y) > 1e-4
            or abs(other.yaw - pose.yaw) > 1e-4
        ):
            return False
    return True


def canonical_demo_scene_02_layout() -> Dict[str, LayoutObjectPose]:
    out: Dict[str, LayoutObjectPose] = {}
    for label, (x, y, yaw) in DEBUG_CALIBRATION_CELLS.items():
        out[label] = layout_pose_from_spawn(label, x, y, yaw)
    return out


def iter_layout_candidates(
    *,
    max_candidates: int = 120,
    rng_seed: int = 42,
    prepend_canonical: bool = True,
) -> Iterator[Dict[str, LayoutObjectPose]]:
    seen: set = set()
    yielded = 0

    def _yield_unique(
        objects: Dict[str, LayoutObjectPose],
        *,
        require_separation: bool = True,
    ) -> bool:
        nonlocal yielded
        plausible = layout_is_physically_plausible(
            objects, require_separation=require_separation
        )
        if not plausible and not is_canonical_demo_scene_02_layout(objects):
            return False
        if not layout_within_table_bounds(objects):
            return False
        fp = tuple(
            (
                lb,
                round(objects[lb].spawn_x, 3),
                round(objects[lb].spawn_y, 3),
                round(objects[lb].yaw, 3),
            )
            for lb in sorted(objects.keys())
        )
        if fp in seen:
            return False
        seen.add(fp)
        yielded += 1
        return True

    canonical = canonical_demo_scene_02_layout()
    if prepend_canonical and _yield_unique(canonical, require_separation=False):
        yield dict(canonical)

    offsets = (-0.05, -0.03, 0.03, 0.05)
    for dx in offsets:
        for dy in offsets:
            perturbed: Dict[str, LayoutObjectPose] = {}
            for label, pose in canonical.items():
                perturbed[label] = layout_pose_from_spawn(
                    label,
                    pose.spawn_x + dx,
                    pose.spawn_y + dy,
                    pose.yaw,
                )
            if _yield_unique(perturbed):
                yield perturbed
                if yielded >= max_candidates:
                    return

    rng = random.Random(int(rng_seed))
    labels = list(DEMO_SCAN_LABELS)
    while yielded < max_candidates:
        objects: Dict[str, LayoutObjectPose] = {}
        for label in labels:
            x = rng.uniform(TABLE_X_RANGE[0], TABLE_X_RANGE[1])
            y = rng.uniform(TABLE_Y_RANGE[0], TABLE_Y_RANGE[1])
            _, _, yaw = DEBUG_CALIBRATION_CELLS[label]
            objects[label] = layout_pose_from_spawn(label, x, y, yaw)
        if _yield_unique(objects):
            yield objects


def evaluate_object_pick_with_backend(
    backend: ReachabilityMoveItBackend,
    *,
    pose: LayoutObjectPose,
    workspace_js: Any,
    table_z_m: float,
    scene_obstacles: Sequence[Dict[str, Any]],
    variant_budget: str,
    cartesian_fraction_threshold: float,
    joint_limit_margin_min_rad: float,
    gripper_jaw_axis_offset_rad: float,
    logger: Any = None,
) -> Tuple[Optional[ScanCellResult], Dict[str, Any]]:
    export_grasp_policy_for_executor, _, _ = _vision_policy_exports()
    policy = export_grasp_policy_for_executor(pose.label)
    policy["label"] = pose.label
    grid_x = (
        pose.operational_x
        if pose.input_mode == "operational_grasp_xy"
        else pose.spawn_x
    )
    grid_y = (
        pose.operational_y
        if pose.input_mode == "operational_grasp_xy"
        else pose.spawn_y
    )
    detection, cell_coords = build_detection_for_reachability_cell(
        pose.label,
        grid_x,
        grid_y,
        pose.yaw,
        table_z_m=float(table_z_m),
        policy=policy,
        input_mode=pose.input_mode,
    )
    if logger is not None:
        logger.info(format_reachability_operational_center_log(cell_coords))
    cell_meta = {
        "label": pose.label,
        "x": grid_x,
        "y": grid_y,
        "yaw": pose.yaw,
        "top_z": float(detection["top_z_m"]),
        "cell_coords": cell_coords,
    }
    variant_results: List[ScanCellResult] = []
    use_budgeted = str(pose.label) in VARIANT_SEARCH_DEBUG_LABELS
    budget = str(variant_budget or "fast").strip().lower()

    if use_budgeted:
        ik_seeds = backend.resolve_debug_ik_seeds(workspace_js)
        workspace_tcp_yaw = backend.fk_tcp_yaw_from_joint_state(
            ik_seeds.get("pick_workspace_ready", workspace_js)
        )
        jobs, budget_meta = iter_budgeted_variant_jobs(
            pose.label,
            grid_x,
            grid_y,
            pose.yaw,
            table_z_m=float(table_z_m),
            policy=policy,
            detection=detection,
            ik_seeds=ik_seeds,
            workspace_tcp_yaw=workspace_tcp_yaw,
            budget=budget,
        )
        early_stop_enabled = bool(budget_meta.get("early_stop"))
        early_stop_used = False
        for variant, seed_name, seed_js in jobs:
            notes = str(getattr(variant, "notes", "") or "")
            variant.notes = "%s;seed=%s" % (notes, seed_name)
            row = validate_variant_with_moveit(
                backend,
                variant,
                workspace_js=workspace_js,
                table_z_m=float(table_z_m),
                scene_obstacles=scene_obstacles,
                joint_limit_margin_min_rad=float(joint_limit_margin_min_rad),
                cell_meta=cell_meta,
                ik_seed_js=seed_js,
                seed_state_name=seed_name,
            )
            apply_reachability_coordinate_metadata(
                row,
                cell_coords,
                cartesian_fraction_threshold=float(cartesian_fraction_threshold),
            )
            variant_results.append(row)
            if early_stop_enabled and is_binary_reachable_cell(row):
                early_stop_used = True
                break
        summary = summarize_variant_search_results(variant_results)
        summary.update(
            {
                "variant_budget": budget,
                "attempts_used": len(variant_results),
                "early_stop_used": early_stop_used,
                "total_possible_variants": int(
                    budget_meta.get("total_possible_variants", len(jobs))
                ),
                "evaluated_variants": len(variant_results),
            }
        )
        if logger is not None:
            logger.info(
                format_variant_search_summary_log(label=pose.label, summary=summary)
            )
        best = summary.get("best")
        if isinstance(best, ScanCellResult) and is_binary_reachable_cell(best):
            return best, summary
        return None, summary

    variants = list(
        iter_grasp_variants_for_cell(
            pose.label,
            grid_x,
            grid_y,
            pose.yaw,
            table_z_m=float(table_z_m),
            gripper_jaw_axis_offset_rad=float(gripper_jaw_axis_offset_rad),
        )
    )
    for variant in variants:
        row = validate_variant_with_moveit(
            backend,
            variant,
            workspace_js=workspace_js,
            table_z_m=float(table_z_m),
            scene_obstacles=scene_obstacles,
            joint_limit_margin_min_rad=float(joint_limit_margin_min_rad),
            cell_meta=cell_meta,
        )
        apply_reachability_coordinate_metadata(
            row,
            cell_coords,
            cartesian_fraction_threshold=float(cartesian_fraction_threshold),
        )
        variant_results.append(row)
        if is_binary_reachable_cell(row):
            summary = summarize_variant_search_results(variant_results)
            return row, summary
    summary = summarize_variant_search_results(variant_results)
    return None, summary


def validate_layout_sequential(
    backend: ReachabilityMoveItBackend,
    objects: Dict[str, LayoutObjectPose],
    *,
    workspace_js: Any,
    table_z_m: float,
    variant_budget: str,
    cartesian_fraction_threshold: float,
    joint_limit_margin_min_rad: float,
    gripper_jaw_axis_offset_rad: float,
    logger: Any = None,
) -> Tuple[bool, CertifiedLayout, str]:
    pick_order = pick_order_by_robot_proximity(objects)
    picks: Dict[str, CertifiedPickRecord] = {}
    remaining = set(pick_order)

    for step_idx, label in enumerate(pick_order):
        remaining_obstacles = [
            build_scene_obstacle_dict(objects[lb])
            for lb in remaining
            if lb != label
        ]
        if logger is not None:
            logger.info(
                "[LAYOUT_VALIDATOR_PICK_STEP]\n"
                "step=%d\n"
                "label=%s\n"
                "remaining_obstacles=%s"
                % (step_idx + 1, label, sorted(lb for lb in remaining if lb != label))
            )
        best, _summary = evaluate_object_pick_with_backend(
            backend,
            pose=objects[label],
            workspace_js=workspace_js,
            table_z_m=float(table_z_m),
            scene_obstacles=remaining_obstacles,
            variant_budget=variant_budget,
            cartesian_fraction_threshold=cartesian_fraction_threshold,
            joint_limit_margin_min_rad=joint_limit_margin_min_rad,
            gripper_jaw_axis_offset_rad=gripper_jaw_axis_offset_rad,
            logger=logger,
        )
        if best is None or not is_binary_reachable_cell(best):
            reason = "pick_fail:%s" % label
            if best is not None:
                reason = "%s:%s" % (reason, str(best.reason))
            fail_layout = CertifiedLayout(
                scene_id="",
                layout_index=-1,
                pick_order=pick_order,
                objects=objects,
                picks=picks,
            )
            return False, fail_layout, reason
        picks[label] = CertifiedPickRecord(
            label=label,
            pick_step=step_idx + 1,
            pose=objects[label],
            scan_result=best,
            remaining_obstacles=tuple(
                sorted(lb for lb in remaining if lb != label)
            ),
        )
        remaining.remove(label)

    certified = CertifiedLayout(
        scene_id="",
        layout_index=-1,
        pick_order=pick_order,
        objects=objects,
        picks=picks,
    )
    return True, certified, "OK"


def build_scene_yaml(
    certified: CertifiedLayout,
    *,
    scene_id: str,
    description: str,
) -> Dict[str, Any]:
    objects_yaml: Dict[str, Any] = {}
    for step_idx, label in enumerate(certified.pick_order):
        pose = certified.objects[label]
        entry: Dict[str, Any] = {
            "role": "target_first" if step_idx == 0 else "obstacle_then_target",
            "pose": {
                "x": round(float(pose.spawn_x), 4),
                "y": round(float(pose.spawn_y), 4),
                "yaw": round(float(pose.yaw), 4),
            },
            "preferred_slot": int(step_idx),
        }
        if label == "chips_can":
            entry["seed"] = CHIPS_CAN_DEFAULT_SEED
        if label == "mustard_bottle":
            entry["operational_grasp_xy"] = [
                round(float(pose.operational_x), 4),
                round(float(pose.operational_y), 4),
            ]
        objects_yaml[label] = entry
    return {
        "scene_id": str(scene_id),
        "description": str(description),
        "layout_version": LAYOUT_VERSION,
        "pick_order": list(certified.pick_order),
        "objects": objects_yaml,
        "certification": {
            "status": CERTIFIED_STATUS,
            "validator": "demo_scene_layout_validator",
            "criteria": (
                "sequential_pick: pregrasp+plan+grasp IK+cartesian>=0.95 "
                "per step with remaining object obstacles"
            ),
            "runtime_pick_place_pending": True,
        },
        "transport_policy": {
            "forbidden_waypoints_when_obstacles_remaining": ["carry_front_high"],
            "local_exit_candidates": [
                "rear_retreat_x_negative",
                "rear_retreat_x_negative_slight_raise",
                "vertical_raise_then_rear_retreat",
            ],
            "transport_route": [
                "carry_mid_high",
                "turn_back_extended_aligned",
                "box_front_high",
                "box_high",
            ],
            "backend": "direct_action",
        },
    }


def build_golden_yaml(
    certified: CertifiedLayout,
    *,
    scene_id: str,
    label: str,
    place_slot_index: int,
) -> Dict[str, Any]:
    pick = certified.picks[label]
    pose = certified.objects[label]
    result = pick.scan_result
    pre = result.pregrasp_tcp or (0.0, 0.0, 0.0)
    gr = result.grasp_tcp or (0.0, 0.0, 0.0)
    top_z = float(result.top_z or gr[2] + 0.025)
    depth = max(0.0, float(pre[2]) - float(gr[2]))
    lift_z = float(pre[2]) + 0.10
    yaw_rad = float(result.commanded_tcp_yaw_rad or pose.yaw)
    return {
        "scene_id": str(scene_id),
        "layout_version": LAYOUT_VERSION,
        "target_label": str(label),
        "status": CERTIFIED_STATUS,
        "object_pose": {
            "semantic_center_xy": [
                round(float(pose.operational_x), 4),
                round(float(pose.operational_y), 4),
            ],
            "spawn_xy": [round(float(pose.spawn_x), 4), round(float(pose.spawn_y), 4)],
            "top_z": round(top_z, 4),
            "yaw_rad": round(float(pose.yaw), 4),
        },
        "candidate": {
            "candidate_idx": 0,
            "yaw_deg": round(math.degrees(yaw_rad), 2),
            "commanded_tcp_yaw_rad": float(yaw_rad),
            "pregrasp_tcp": [round(pre[0], 4), round(pre[1], 4), round(pre[2], 4)],
            "grasp_tcp": [round(gr[0], 4), round(gr[1], 4), round(gr[2], 4)],
            "lift_tcp": [round(pre[0], 4), round(pre[1], 4), round(lift_z, 4)],
            "depth_from_top_m": round(depth, 4),
            "ik_seed": str(result.seed_state_name or "pick_workspace_ready"),
            "prevalidation_source": "demo_scene_layout_validator_sequential",
            "cartesian_descend_fraction": float(result.cartesian_fraction or 1.0),
            "variant_notes": str(result.variant_notes or ""),
        },
        "grasp": {
            "strategy": GRASP_STRATEGY_BY_LABEL.get(label, "top_down"),
            "open_joint": 0.0399,
            "close_joint": 0.0270,
            "expected_width_m": 0.0600,
        },
        "transport": {
            "selected_transport_entry": "vertical_raise_then_rear_retreat",
            "route": [
                "carry_mid_high",
                "turn_back_extended_aligned",
                "box_front_high",
                "box_high",
            ],
            "backend": "direct_action",
        },
        "place": {
            "slot_index": int(place_slot_index),
            "slot_name": "slot_%d" % (int(place_slot_index) + 1),
        },
        "validation": {
            "result": CERTIFIED_STATUS,
            "pick_step": int(pick.pick_step),
            "pick_order": list(certified.pick_order),
            "remaining_obstacles_at_pick": list(pick.remaining_obstacles),
            "endpoint_ik_ok": bool(result.endpoint_ik_ok),
            "cartesian_fraction": float(result.cartesian_fraction or 0.0),
            "runtime_pick_place_confirm_required": True,
        },
    }


def write_certified_layout_bundle(
    certified: CertifiedLayout,
    *,
    output_dir: str,
    scene_id: str,
) -> str:
    bundle_dir = os.path.join(str(output_dir), scene_id)
    os.makedirs(bundle_dir, exist_ok=True)
    scene = build_scene_yaml(
        certified,
        scene_id=scene_id,
        description=(
            "Escena certificada por demo_scene_layout_validator "
            "(pick secuencial con obstáculos restantes)."
        ),
    )
    scene_path = os.path.join(bundle_dir, "scene.yaml")
    with open(scene_path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(
            scene,
            handle,
            default_flow_style=False,
            sort_keys=False,
            allow_unicode=True,
        )
    golden_paths: Dict[str, str] = {}
    for step_idx, label in enumerate(certified.pick_order):
        golden = build_golden_yaml(
            certified,
            scene_id=scene_id,
            label=label,
            place_slot_index=step_idx,
        )
        golden_name = "golden_%s.yaml" % label
        golden_path = os.path.join(bundle_dir, golden_name)
        with open(golden_path, "w", encoding="utf-8") as handle:
            yaml.safe_dump(
                golden,
                handle,
                default_flow_style=False,
                sort_keys=False,
                allow_unicode=True,
            )
        golden_paths[label] = golden_path

    report = {
        "scene_id": scene_id,
        "pick_order": list(certified.pick_order),
        "fingerprint": list(certified.fingerprint()),
        "picks": {
            label: {
                "pick_step": pick.pick_step,
                "spawn_xy": [pick.pose.spawn_x, pick.pose.spawn_y],
                "operational_xy": [pick.pose.operational_x, pick.pose.operational_y],
                "yaw": pick.pose.yaw,
                "pregrasp_tcp": list(pick.scan_result.pregrasp_tcp or ()),
                "grasp_tcp": list(pick.scan_result.grasp_tcp or ()),
                "cartesian_fraction": pick.scan_result.cartesian_fraction,
                "remaining_obstacles": list(pick.remaining_obstacles),
            }
            for label, pick in certified.picks.items()
        },
        "golden_files": golden_paths,
        "scene_file": scene_path,
    }
    report_path = os.path.join(bundle_dir, "validation_report.json")
    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
    return bundle_dir


@dataclass
class LayoutValidatorConfig:
    num_scenes: int = 3
    output_dir: str = "/tmp/certified_demo_layouts"
    scene_id_prefix: str = "certified_layout"
    variant_budget: str = "fast"
    max_layout_attempts: int = 120
    table_z_m: float = DEFAULT_TABLE_Z_M
    cartesian_fraction_threshold: float = DEFAULT_CARTESIAN_FRACTION_THRESHOLD
    joint_limit_margin_min_rad: float = DEFAULT_JOINT_LIMIT_MARGIN_MIN_RAD
    gripper_jaw_axis_offset_rad: float = 0.0
    waypoints_yaml: str = ""
    rng_seed: int = 42


class DemoSceneLayoutValidatorNode:
    """Busca layouts certificados y escribe escena + golden por objeto."""

    def __init__(self, config: LayoutValidatorConfig) -> None:
        import rclpy
        from rclpy.node import Node

        self._config = config
        self._node = Node("demo_scene_layout_validator")
        self._logger = self._node.get_logger()
        scan_cfg = ReachabilityScanConfig(
            labels=DEMO_SCAN_LABELS,
            table_z_m=float(config.table_z_m),
            waypoints_yaml=str(config.waypoints_yaml or ""),
            variant_budget=str(config.variant_budget),
            cartesian_fraction_threshold=float(config.cartesian_fraction_threshold),
            joint_limit_margin_min_rad=float(config.joint_limit_margin_min_rad),
            gripper_jaw_axis_offset_rad=float(config.gripper_jaw_axis_offset_rad),
        )
        self._backend = ReachabilityMoveItBackend(self._node, scan_cfg)

    def run(self) -> int:
        import rclpy

        self._backend.setup()
        self._backend.apply_table_collision()
        time.sleep(0.5)
        try:
            workspace_js = self._backend.workspace_joint_state()
        except RuntimeError as exc:
            self._logger.error(str(exc))
            return 1

        os.makedirs(self._config.output_dir, exist_ok=True)
        certified_layouts: List[CertifiedLayout] = []
        seen_fingerprints: set = set()
        attempts = 0

        self._logger.info(
            "[LAYOUT_VALIDATOR_START]\n"
            "target_scenes=%d\n"
            "max_layout_attempts=%d\n"
            "variant_budget=%s\n"
            "output_dir=%s"
            % (
                int(self._config.num_scenes),
                int(self._config.max_layout_attempts),
                str(self._config.variant_budget),
                str(self._config.output_dir),
            )
        )

        for objects in iter_layout_candidates(
            max_candidates=int(self._config.max_layout_attempts),
            rng_seed=int(self._config.rng_seed),
        ):
            attempts += 1
            fp = tuple(
                (
                    lb,
                    round(objects[lb].spawn_x, 3),
                    round(objects[lb].spawn_y, 3),
                    round(objects[lb].yaw, 3),
                )
                for lb in sorted(objects.keys())
            )
            if fp in seen_fingerprints:
                continue
            self._logger.info(
                "[LAYOUT_VALIDATOR_TRY]\nattempt=%d\nfingerprint=%s"
                % (attempts, str(fp))
            )
            ok, layout, reason = validate_layout_sequential(
                self._backend,
                objects,
                workspace_js=workspace_js,
                table_z_m=float(self._config.table_z_m),
                variant_budget=str(self._config.variant_budget),
                cartesian_fraction_threshold=float(
                    self._config.cartesian_fraction_threshold
                ),
                joint_limit_margin_min_rad=float(
                    self._config.joint_limit_margin_min_rad
                ),
                gripper_jaw_axis_offset_rad=float(
                    self._config.gripper_jaw_axis_offset_rad
                ),
                logger=self._logger,
            )
            if not ok:
                self._logger.info(
                    "[LAYOUT_VALIDATOR_REJECT]\nreason=%s" % str(reason)
                )
                continue
            seen_fingerprints.add(fp)
            layout.attempts_to_find = attempts
            layout.layout_index = len(certified_layouts) + 1
            layout.scene_id = "%s_%02d" % (
                str(self._config.scene_id_prefix),
                layout.layout_index,
            )
            certified_layouts.append(layout)
            bundle_dir = write_certified_layout_bundle(
                layout,
                output_dir=str(self._config.output_dir),
                scene_id=layout.scene_id,
            )
            self._logger.info(
                "[LAYOUT_VALIDATOR_CERTIFIED]\n"
                "scene_id=%s\n"
                "pick_order=%s\n"
                "bundle_dir=%s"
                % (
                    layout.scene_id,
                    ",".join(layout.pick_order),
                    bundle_dir,
                )
            )
            if len(certified_layouts) >= int(self._config.num_scenes):
                break
            rclpy.spin_once(self._node, timeout_sec=0.0)

        manifest = {
            "certified_count": len(certified_layouts),
            "attempts_used": attempts,
            "target_scenes": int(self._config.num_scenes),
            "layouts": [
                {
                    "scene_id": layout.scene_id,
                    "pick_order": list(layout.pick_order),
                    "bundle_dir": os.path.join(
                        str(self._config.output_dir), layout.scene_id
                    ),
                    "fingerprint": list(layout.fingerprint()),
                }
                for layout in certified_layouts
            ],
        }
        manifest_path = os.path.join(str(self._config.output_dir), "manifest.json")
        with open(manifest_path, "w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2, sort_keys=True)

        self._logger.info(
            "[LAYOUT_VALIDATOR_DONE]\n"
            "certified=%d\n"
            "attempts=%d\n"
            "manifest=%s"
            % (len(certified_layouts), attempts, manifest_path)
        )
        return 0 if certified_layouts else 2


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Validador secuencial de layouts demo: genera escenas certificadas "
            "con golden YAML por objeto (sin heatmaps)."
        )
    )
    parser.add_argument("--num-scenes", type=int, default=3)
    parser.add_argument(
        "--output-dir",
        default="/tmp/certified_demo_layouts",
        help="Directorio raíz para escenas certificadas",
    )
    parser.add_argument("--scene-id-prefix", default="certified_layout")
    parser.add_argument(
        "--variant-budget",
        default="fast",
        choices=("fast", "balanced", "exhaustive"),
    )
    parser.add_argument("--max-layout-attempts", type=int, default=120)
    parser.add_argument("--table-z", type=float, default=DEFAULT_TABLE_Z_M)
    parser.add_argument("--waypoints-yaml", default="")
    parser.add_argument("--rng-seed", type=int, default=42)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    try:
        import rclpy
    except ImportError:
        print("ROS/rclpy required for layout validation.", file=sys.stderr)
        return 1

    config = LayoutValidatorConfig(
        num_scenes=int(args.num_scenes),
        output_dir=str(args.output_dir),
        scene_id_prefix=str(args.scene_id_prefix),
        variant_budget=str(args.variant_budget),
        max_layout_attempts=int(args.max_layout_attempts),
        table_z_m=float(args.table_z),
        waypoints_yaml=str(args.waypoints_yaml or ""),
        rng_seed=int(args.rng_seed),
    )
    rclpy.init()
    node = DemoSceneLayoutValidatorNode(config)
    try:
        return int(node.run())
    finally:
        node._node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
