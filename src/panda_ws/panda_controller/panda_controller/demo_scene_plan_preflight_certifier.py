#!/usr/bin/env python3
"""Certificador plan-only de escenas demo (sin mover el robot).

Valida secuencialmente pick + lift + place + home para 4 objetos usando MoveIt
con colisiones reales en planning scene. Escribe escenas + golden candidatos.
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import yaml

from panda_controller.authoritative_scene_obstacles import (
    authoritative_obstacle_excluded,
)
from panda_controller.demo_golden_pick_candidate import VALIDATED_STATUS
from panda_controller.demo_scene_layout_validator import (
    CertifiedLayout,
    CertifiedPickRecord,
    LayoutObjectPose,
    build_golden_yaml,
    build_scene_obstacle_dict,
    build_scene_yaml,
    canonical_demo_scene_02_layout,
    iter_layout_candidates,
    layout_fingerprint_seed,
    pick_order_by_robot_proximity,
    pick_order_distance_breakdown,
    evaluate_object_pick_with_backend,
)
from panda_controller.demo_scene_reachability_scan import (
    DEFAULT_CARTESIAN_FRACTION_THRESHOLD,
    DEFAULT_JOINT_LIMIT_MARGIN_MIN_RAD,
    DEFAULT_TABLE_Z_M,
    ReachabilityMoveItBackend,
    ReachabilityScanConfig,
    ScanCellResult,
    downward_yaw_quaternion,
    is_binary_reachable_cell,
)
from panda_controller.planning_scene_collision_builder import (
    MoveItCollisionScenePublisher,
    build_scene_objects_from_layout,
)

PLAN_PREFLIGHT_STATUS = "plan_preflight_certified"
PLAN_PREFLIGHT_PICK_ONLY_STATUS = "plan_preflight_pick_only_certified"
LIFT_CARTESIAN_THRESHOLD = 0.90
DEFAULT_PLACE_ORIGIN_XY = (-0.370, 0.080)
DEFAULT_PLACE_SLOT_SPACING_M = 0.070
DEFAULT_PLACE_RELEASE_Z_M = 0.330
DEFAULT_PLACE_APPROACH_Z_M = 0.650
LIFT_CLEARANCE_ABOVE_PREGRASP_M = 0.10


@dataclass
class PickCyclePlanResult:
    label: str
    pick_step: int
    pick_ok: bool
    lift_ok: bool
    place_ok: bool
    home_ok: bool
    pick_result: Optional[ScanCellResult]
    reason: str = ""
    remaining_obstacles: Tuple[str, ...] = ()

    def passed(self, *, pick_only: bool = False) -> bool:
        if pick_only:
            return bool(self.pick_ok)
        return bool(self.pick_ok and self.lift_ok and self.place_ok and self.home_ok)


@dataclass
class PlanPreflightCertifierConfig:
    num_scenes: int = 3
    output_dir: str = "/tmp/plan_preflight_certified_layouts"
    scene_id_prefix: str = "plan_certified_layout"
    variant_budget: str = "fast"
    max_layout_attempts: int = 120
    table_z_m: float = DEFAULT_TABLE_Z_M
    cartesian_fraction_threshold: float = DEFAULT_CARTESIAN_FRACTION_THRESHOLD
    joint_limit_margin_min_rad: float = DEFAULT_JOINT_LIMIT_MARGIN_MIN_RAD
    gripper_jaw_axis_offset_rad: float = 0.0
    waypoints_yaml: str = ""
    rng_seed: int = 42
    prepend_canonical: bool = False
    certify_pick_only: bool = False
    place_origin_xy: Tuple[float, float] = DEFAULT_PLACE_ORIGIN_XY
    place_slot_spacing_m: float = DEFAULT_PLACE_SLOT_SPACING_M


def resolve_pick_order_for_layout(
    objects: Dict[str, LayoutObjectPose],
) -> Tuple[str, ...]:
    return pick_order_by_robot_proximity(
        objects,
        tie_break_seed=layout_fingerprint_seed(objects),
    )


def _pick_fail_reason(
    pick_scan: Optional[ScanCellResult],
    summary: Dict[str, Any],
) -> str:
    if isinstance(pick_scan, ScanCellResult):
        code = str(pick_scan.pregrasp_ik_error_code or "").strip()
        base = str(pick_scan.reason or pick_scan.result or "unreachable")
        if code and code not in ("SUCCESS", "OK"):
            return "pick_fail:%s:ik_code=%s" % (base, code)
        return "pick_fail:%s" % base
    best = summary.get("best")
    if isinstance(best, ScanCellResult):
        code = str(best.pregrasp_ik_error_code or "").strip()
        base = str(best.reason or best.result or "unreachable")
        if code and code not in ("SUCCESS", "OK"):
            return "pick_fail:%s:ik_code=%s" % (base, code)
        if base:
            return "pick_fail:%s" % base
    for key in ("failure_reason", "reason", "status"):
        if summary.get(key):
            return "pick_fail:%s" % str(summary[key])
    evaluated = summary.get("evaluated_variants")
    if evaluated == 0:
        return "pick_fail:no_variants_evaluated"
    return "pick_fail:no_reachable_variant"


def wait_for_joint_states(node: Any, *, timeout_sec: float = 8.0) -> bool:
    import rclpy
    from sensor_msgs.msg import JointState

    ready = {"ok": False}

    def _cb(_msg: JointState) -> None:
        ready["ok"] = True

    sub = node.create_subscription(JointState, "/joint_states", _cb, 10)
    deadline = time.monotonic() + float(timeout_sec)
    while not ready["ok"] and time.monotonic() < deadline:
        rclpy.spin_once(node, timeout_sec=0.1)
    node.destroy_subscription(sub)
    return bool(ready["ok"])


def probe_moveit_pick_pipeline(
    backend: ReachabilityMoveItBackend,
    publisher: MoveItCollisionScenePublisher,
    *,
    workspace_js: Any,
    config: PlanPreflightCertifierConfig,
    logger: Any = None,
) -> Tuple[bool, str]:
    """Prueba rápida: mesa vacía + cracker canónico (como el scanner single-cell)."""
    objects = canonical_demo_scene_02_layout()
    pose = objects["cracker_box"]
    publisher.publish_table()
    time.sleep(0.2)
    pick_scan, summary = evaluate_object_pick_with_backend(
        backend,
        pose=pose,
        workspace_js=workspace_js,
        table_z_m=float(config.table_z_m),
        scene_obstacles=[],
        variant_budget=str(config.variant_budget),
        cartesian_fraction_threshold=float(config.cartesian_fraction_threshold),
        joint_limit_margin_min_rad=float(config.joint_limit_margin_min_rad),
        gripper_jaw_axis_offset_rad=float(config.gripper_jaw_axis_offset_rad),
        logger=logger,
    )
    if pick_scan is not None and is_binary_reachable_cell(pick_scan):
        return True, "OK"
    reason = _pick_fail_reason(pick_scan, summary)
    if logger is not None:
        logger.error(
            "[PLAN_PREFLIGHT_PROBE_FAIL]\n"
            "probe=canonical_cracker_empty_table\nreason=%s\nsummary=%s"
            % (reason, json.dumps(summary, default=str)[:800])
        )
    return False, reason


def write_rejection_summary(
    output_dir: str,
    *,
    rejection_counts: Dict[str, int],
    last_rejections: Sequence[Dict[str, Any]],
    probe_ok: bool,
    probe_reason: str,
) -> str:
    path = os.path.join(str(output_dir), "rejection_summary.json")
    payload = {
        "probe_ok": bool(probe_ok),
        "probe_reason": str(probe_reason),
        "rejection_counts": dict(rejection_counts),
        "last_rejections": list(last_rejections),
        "hints": [
            "Si probe_ok=false con pregrasp_ik_fail: falta /joint_states + robot_state_publisher.",
            "Usa: ros2 launch panda_bringup moveit_plan_preflight_bringup.launch.py",
            "Comprueba: ros2 topic echo /joint_states --once",
            "Comprueba: ros2 service list | grep compute_cartesian_path",
        ],
    }
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    return path


def place_slot_xy(slot_index: int, *, origin: Tuple[float, float], spacing: float) -> Tuple[float, float]:
    """Misma convención que DEFAULT_PLACE_SLOTS del controller (primeros 4 slots)."""
    offsets = (
        (0.0, 0.0),
        (0.0, 1.0),
        (0.0, -1.0),
        (-1.0, 0.0),
    )
    idx = max(0, min(int(slot_index), len(offsets) - 1))
    dx, dy = offsets[idx]
    return (
        float(origin[0]) + float(dx) * float(spacing),
        float(origin[1]) + float(dy) * float(spacing),
    )


def _hand_from_tcp(
    backend: ReachabilityMoveItBackend,
    tcp: Tuple[float, float, float],
    yaw_rad: float,
) -> Tuple[Tuple[float, float, float], Tuple[float, float, float, float]]:
    quat = downward_yaw_quaternion(float(yaw_rad))
    return backend.tcp_to_hand_pose(tcp, quat)


def _ik_hand(
    backend: ReachabilityMoveItBackend,
    hand_pos: Tuple[float, float, float],
    hand_quat: Tuple[float, float, float, float],
    seed_js: Any,
) -> Tuple[Any, str]:
    return backend.compute_ik_with_error(
        hand_pos,
        hand_quat,
        backend._config.moveit_target_link,
        seed_js,
    )


def _cartesian_ok(
    backend: ReachabilityMoveItBackend,
    start_js: Any,
    hand_goal: Tuple[float, float, float],
    hand_quat: Tuple[float, float, float, float],
    *,
    threshold: float,
    avoid_collisions: bool = True,
    max_step: float = 0.0025,
) -> bool:
    from panda_controller.get_cartesian_path_audit import (
        build_get_cartesian_path_request,
        evaluate_get_cartesian_path_start_state_audit,
    )

    import rclpy

    if backend._moveit2 is None or backend._cartesian_client is None:
        return False
    if not backend._cartesian_client.wait_for_service(timeout_sec=3.0):
        return False
    req = build_get_cartesian_path_request(
        planning_frame=str(backend._planning_frame),
        group_name=str(backend._config.group_name),
        link_name=str(backend._config.moveit_target_link),
        start_js=start_js,
        hand_goal=hand_goal,
        hand_quat=hand_quat,
        max_step=float(max_step),
        jump_threshold=0.0,
        avoid_collisions=bool(avoid_collisions),
    )
    if req is None:
        return False
    future = backend._cartesian_client.call_async(req)
    deadline = time.monotonic() + max(
        5.0, float(backend._config.planning_time) + 2.0
    )
    while not future.done() and time.monotonic() < deadline:
        rclpy.spin_once(backend._node, timeout_sec=0.05)
    if not future.done():
        return False
    response = future.result()
    audit = evaluate_get_cartesian_path_start_state_audit(
        requested_start_js=start_js,
        response=response,
    )
    fraction = audit.get("fraction")
    honored = audit.get("start_state_honored")
    return bool(
        honored
        and fraction is not None
        and float(fraction) + 1e-6 >= float(threshold)
    )


def _resolve_pick_grasp_joint_state(
    backend: ReachabilityMoveItBackend,
    pick_result: ScanCellResult,
    workspace_js: Any,
) -> Tuple[Optional[Any], Optional[Any], str]:
    """Replica la cadena IK del pick validado: workspace -> pregrasp -> grasp."""
    pre_hand = pick_result.pregrasp_hand_target
    gr_hand = pick_result.grasp_hand_target
    quat = pick_result.quat
    if pre_hand is None or gr_hand is None or quat is None:
        return None, None, "missing_hand_targets"
    pregrasp_js, pre_err = _ik_hand(
        backend,
        (
            float(pre_hand[0]),
            float(pre_hand[1]),
            float(pre_hand[2]),
        ),
        (
            float(quat[0]),
            float(quat[1]),
            float(quat[2]),
            float(quat[3]),
        ),
        workspace_js,
    )
    if pregrasp_js is None:
        return None, None, "pregrasp_js:%s" % pre_err
    grasp_js, gr_err = _ik_hand(
        backend,
        (
            float(gr_hand[0]),
            float(gr_hand[1]),
            float(gr_hand[2]),
        ),
        (
            float(quat[0]),
            float(quat[1]),
            float(quat[2]),
            float(quat[3]),
        ),
        pregrasp_js,
    )
    if grasp_js is None:
        return None, None, "grasp_js:%s" % gr_err
    return pregrasp_js, grasp_js, "OK"


def _lift_hand_quat_from_pick(
    pick_result: ScanCellResult,
) -> Tuple[float, float, float, float]:
    quat = pick_result.quat
    if quat is None:
        raise ValueError("missing_pick_quat")
    return (
        float(quat[0]),
        float(quat[1]),
        float(quat[2]),
        float(quat[3]),
    )


def _lift_hand_candidates_from_pick(
    pick_result: ScanCellResult,
    *,
    label: str,
) -> List[Tuple[float, float, float]]:
    """Candidatos hand-Z ascendentes (grasp micro-rise, pregrasp+clearance)."""
    out: List[Tuple[float, float, float]] = []
    seen: set = set()
    gr_hand = pick_result.grasp_hand_target
    pre_hand = pick_result.pregrasp_hand_target
    rises_m = (0.040, 0.070, LIFT_CLEARANCE_ABOVE_PREGRASP_M)
    if str(label).strip().lower() == "chips_can":
        rises_m = (0.025, 0.040, 0.070, LIFT_CLEARANCE_ABOVE_PREGRASP_M)
    if gr_hand is not None:
        for rise in rises_m:
            pt = (
                round(float(gr_hand[0]), 4),
                round(float(gr_hand[1]), 4),
                round(float(gr_hand[2]) + float(rise), 4),
            )
            if pt not in seen:
                seen.add(pt)
                out.append(pt)
    if pre_hand is not None:
        pt = (
            round(float(pre_hand[0]), 4),
            round(float(pre_hand[1]), 4),
            round(
                float(pre_hand[2]) + float(LIFT_CLEARANCE_ABOVE_PREGRASP_M),
                4,
            ),
        )
        if pt not in seen:
            seen.add(pt)
            out.append(pt)
    return out


def _validate_lift_plan_only(
    backend: ReachabilityMoveItBackend,
    *,
    label: str,
    pregrasp_js: Any,
    grasp_js: Any,
    lift_candidates: Sequence[Tuple[float, float, float]],
    lift_quat: Tuple[float, float, float, float],
    pregrasp_hand: Optional[Tuple[float, float, float]],
    config: PlanPreflightCertifierConfig,
) -> Tuple[bool, str, Any]:
    """Plan-only: sin objeto attach en scene → lift siempre sin colisiones estrictas."""
    _ = label
    cart_steps = (0.040, 0.025, 0.010)
    lift_threshold = min(
        float(LIFT_CARTESIAN_THRESHOLD),
        float(config.cartesian_fraction_threshold),
    )

    if pregrasp_hand is not None:
        for max_step in cart_steps:
            if _cartesian_ok(
                backend,
                grasp_js,
                pregrasp_hand,
                lift_quat,
                threshold=lift_threshold,
                avoid_collisions=False,
                max_step=float(max_step),
            ):
                return True, "lift_pregrasp_cartesian", pregrasp_js
        if backend.plan_to_joint_state(grasp_js, pregrasp_js):
            return True, "lift_pregrasp_plan", pregrasp_js

    last_lift_err = "no_lift_candidate"
    for lift_hand in lift_candidates:
        for max_step in cart_steps:
            if _cartesian_ok(
                backend,
                grasp_js,
                lift_hand,
                lift_quat,
                threshold=lift_threshold,
                avoid_collisions=False,
                max_step=float(max_step),
            ):
                lift_js, lift_err = _ik_hand(
                    backend, lift_hand, lift_quat, grasp_js
                )
                if lift_js is not None:
                    return True, "lift_cartesian_ok", lift_js
                last_lift_err = str(lift_err)
        lift_js, lift_err = _ik_hand(backend, lift_hand, lift_quat, grasp_js)
        last_lift_err = str(lift_err)
        if lift_js is not None and backend.plan_to_joint_state(grasp_js, lift_js):
            return True, "lift_plan_ok", lift_js

    return False, "lift_fail:%s" % last_lift_err, None


def _validate_place_and_home_plan_only(
    backend: ReachabilityMoveItBackend,
    *,
    start_js: Any,
    workspace_js: Any,
    place_hand: Tuple[float, float, float],
    place_quat: Tuple[float, float, float, float],
    extra_seeds: Sequence[Any],
) -> Tuple[bool, bool, str]:
    seeds: List[Any] = []
    for seed in [start_js] + list(extra_seeds) + [workspace_js]:
        if seed is not None and seed not in seeds:
            seeds.append(seed)

    place_js = None
    last_place_err = "no_place_seed"
    for seed in seeds:
        candidate, place_err = _ik_hand(backend, place_hand, place_quat, seed)
        last_place_err = str(place_err)
        if candidate is None:
            continue
        if backend.plan_to_joint_state(seed, candidate):
            place_js = candidate
            break

    if place_js is None:
        return False, False, "place_fail:%s" % last_place_err

    home_js = backend.home_joint_state()
    if home_js is None:
        return True, False, "home_waypoint_missing"
    if backend.plan_to_joint_state(place_js, home_js):
        return True, True, "OK"
    if backend.plan_to_joint_state(workspace_js, home_js):
        return True, True, "OK"
    return True, False, "home_plan_fail"


def validate_lift_place_home_plan_only(
    backend: ReachabilityMoveItBackend,
    *,
    label: str,
    pick_result: ScanCellResult,
    place_slot_index: int,
    workspace_js: Any,
    config: PlanPreflightCertifierConfig,
) -> Tuple[bool, bool, bool, str]:
    if pick_result.grasp_tcp is None or pick_result.pregrasp_tcp is None:
        return False, False, False, "missing_pick_tcp"
    yaw = float(pick_result.commanded_tcp_yaw_rad or 0.0)
    pregrasp_js, grasp_js, chain_err = _resolve_pick_grasp_joint_state(
        backend, pick_result, workspace_js
    )
    if grasp_js is None:
        return False, False, False, "grasp_chain_fail:%s" % chain_err

    try:
        lift_quat = _lift_hand_quat_from_pick(pick_result)
        lift_candidates = _lift_hand_candidates_from_pick(
            pick_result, label=str(label)
        )
    except ValueError:
        return False, False, False, "missing_lift_hand_target"
    if not lift_candidates:
        return False, False, False, "missing_lift_hand_target"

    pre_hand = pick_result.pregrasp_hand_target
    pregrasp_hand = None
    if pre_hand is not None:
        pregrasp_hand = (
            float(pre_hand[0]),
            float(pre_hand[1]),
            float(pre_hand[2]),
        )

    lift_ok, lift_reason, transport_js = _validate_lift_plan_only(
        backend,
        label=str(label),
        pregrasp_js=pregrasp_js,
        grasp_js=grasp_js,
        lift_candidates=lift_candidates,
        lift_quat=lift_quat,
        pregrasp_hand=pregrasp_hand,
        config=config,
    )
    if not lift_ok or transport_js is None:
        return False, False, False, lift_reason

    px, py = place_slot_xy(
        place_slot_index,
        origin=config.place_origin_xy,
        spacing=config.place_slot_spacing_m,
    )
    place_tcp = (float(px), float(py), float(DEFAULT_PLACE_RELEASE_Z_M))
    place_hand, place_quat = _hand_from_tcp(backend, place_tcp, yaw)
    place_ok, home_ok, tail_reason = _validate_place_and_home_plan_only(
        backend,
        start_js=transport_js,
        workspace_js=workspace_js,
        place_hand=place_hand,
        place_quat=place_quat,
        extra_seeds=(pregrasp_js, grasp_js),
    )
    if not place_ok:
        return True, False, False, tail_reason
    if not home_ok:
        return True, True, False, tail_reason
    return True, True, True, "OK"


def remaining_labels_for_pick_step(
    pick_order: Sequence[str],
    step_index: int,
    completed_labels: Set[str],
) -> Tuple[str, ...]:
    remaining = [
        str(lb)
        for lb in pick_order[step_index + 1 :]
        if str(lb) not in completed_labels
    ]
    return tuple(sorted(remaining))


def scene_obstacles_for_pick_step(
    objects: Dict[str, LayoutObjectPose],
    *,
    target_label: str,
    completed_labels: Set[str],
) -> List[Dict[str, Any]]:
    """Obstáculos 2D autoritativos para el pre-check geométrico del descend."""
    out: List[Dict[str, Any]] = []
    target_l = str(target_label).strip().lower()
    for label, pose in objects.items():
        lb = str(label).strip().lower()
        if lb == target_l or lb in {str(x).strip().lower() for x in completed_labels}:
            continue
        out.append(build_scene_obstacle_dict(pose))
    return out


def publish_planning_scene_for_pick_step(
    publisher: MoveItCollisionScenePublisher,
    scene_objects: Sequence[Dict[str, Any]],
    *,
    target_label: str,
    completed_labels: Set[str],
) -> None:
    include: List[str] = []
    for obj in scene_objects:
        lb = str(obj.get("label", "")).strip().lower()
        ent = str(obj.get("entity_name", "")).strip()
        excluded, _ = authoritative_obstacle_excluded(
            {"label": lb, "entity_name": ent},
            target_label=str(target_label),
            target_entity="",
            completed_entities=set(),
            completed_labels={str(x).strip().lower() for x in completed_labels},
        )
        if excluded:
            continue
        include.append(lb)
    publisher.publish_objects(
        scene_objects,
        include_labels=include,
        target_label=str(target_label),
    )
    time.sleep(0.15)


def validate_layout_plan_preflight(
    backend: ReachabilityMoveItBackend,
    publisher: MoveItCollisionScenePublisher,
    objects: Dict[str, LayoutObjectPose],
    *,
    workspace_js: Any,
    config: PlanPreflightCertifierConfig,
    logger: Any = None,
) -> Tuple[bool, List[PickCyclePlanResult], str]:
    pick_order = resolve_pick_order_for_layout(objects)
    scene_objects = build_scene_objects_from_layout(
        objects, table_z_m=float(config.table_z_m)
    )
    completed: Set[str] = set()
    results: List[PickCyclePlanResult] = []

    for step_idx, label in enumerate(pick_order):
        remaining = remaining_labels_for_pick_step(pick_order, step_idx, completed)
        publish_planning_scene_for_pick_step(
            publisher,
            scene_objects,
            target_label=str(label),
            completed_labels=completed,
        )
        if logger is not None:
            logger.info(
                "[PLAN_PREFLIGHT_PICK_STEP]\n"
                "step=%d\nlabel=%s\ncompleted=%s\nremaining=%s"
                % (step_idx + 1, label, sorted(completed), list(remaining))
            )

        pose = objects[label]
        obstacles = scene_obstacles_for_pick_step(
            objects,
            target_label=str(label),
            completed_labels=completed,
        )
        pick_scan, _summary = evaluate_object_pick_with_backend(
            backend,
            pose=pose,
            workspace_js=workspace_js,
            table_z_m=float(config.table_z_m),
            scene_obstacles=obstacles,
            variant_budget=str(config.variant_budget),
            cartesian_fraction_threshold=float(config.cartesian_fraction_threshold),
            joint_limit_margin_min_rad=float(config.joint_limit_margin_min_rad),
            gripper_jaw_axis_offset_rad=float(config.gripper_jaw_axis_offset_rad),
            logger=logger,
        )
        pick_ok = pick_scan is not None and is_binary_reachable_cell(pick_scan)
        lift_ok = False
        place_ok = False
        home_ok = False
        reason = _pick_fail_reason(pick_scan, _summary)
        if pick_ok and pick_scan is not None:
            if bool(config.certify_pick_only):
                lift_ok = True
                place_ok = True
                home_ok = True
                reason = "pick_only_deferred_transport"
            else:
                lift_ok, place_ok, home_ok, reason = validate_lift_place_home_plan_only(
                    backend,
                    label=str(label),
                    pick_result=pick_scan,
                    place_slot_index=step_idx,
                    workspace_js=workspace_js,
                    config=config,
                )
                if not lift_ok:
                    reason = "lift_fail:" + reason
                elif not place_ok:
                    reason = "place_fail:" + reason
                elif not home_ok:
                    reason = "home_fail:" + reason
                else:
                    reason = "OK"

        cycle = PickCyclePlanResult(
            label=str(label),
            pick_step=step_idx + 1,
            pick_ok=bool(pick_ok),
            lift_ok=bool(lift_ok),
            place_ok=bool(place_ok),
            home_ok=bool(home_ok),
            pick_result=pick_scan,
            reason=str(reason),
            remaining_obstacles=remaining,
        )
        results.append(cycle)
        if logger is not None:
            logger.info(
                "[PLAN_PREFLIGHT_STEP_RESULT]\n"
                "label=%s\npick_ok=%s\nlift_ok=%s\nplace_ok=%s\nhome_ok=%s\nreason=%s"
                % (
                    label,
                    str(cycle.pick_ok).lower(),
                    str(cycle.lift_ok).lower(),
                    str(cycle.place_ok).lower(),
                    str(cycle.home_ok).lower(),
                    cycle.reason,
                )
            )
        if not cycle.passed(pick_only=bool(config.certify_pick_only)):
            return False, results, "step_fail:%s:%s" % (label, cycle.reason)
        completed.add(str(label))

    return True, results, "OK"


def certified_status_for_config(config: PlanPreflightCertifierConfig) -> str:
    if bool(config.certify_pick_only):
        return PLAN_PREFLIGHT_PICK_ONLY_STATUS
    return PLAN_PREFLIGHT_STATUS


def build_plan_preflight_golden_yaml(
    *,
    scene_id: str,
    label: str,
    pose: LayoutObjectPose,
    pick_result: ScanCellResult,
    pick_step: int,
    pick_order: Sequence[str],
    remaining_obstacles: Sequence[str],
    place_slot_index: int,
    config: PlanPreflightCertifierConfig,
) -> Dict[str, Any]:
    layout = CertifiedLayout(
        scene_id=str(scene_id),
        layout_index=1,
        pick_order=tuple(pick_order),
        objects={str(label): pose},
        picks={
            str(label): CertifiedPickRecord(
                label=str(label),
                pick_step=int(pick_step),
                pose=pose,
                scan_result=pick_result,
                remaining_obstacles=tuple(remaining_obstacles),
            )
        },
    )
    golden = build_golden_yaml(
        layout,
        scene_id=scene_id,
        label=label,
        place_slot_index=place_slot_index,
    )
    if str(label).strip().lower() == "chips_can":
        from panda_controller.demo_golden_pick_candidate import (
            enrich_chips_can_legacy_golden_fields,
        )

        pose_xy = (
            round(float(pose.operational_x), 4),
            round(float(pose.operational_y), 4),
        )
        golden = enrich_chips_can_legacy_golden_fields(
            golden,
            grasp_xy=pose_xy,
        )
    cert_status = certified_status_for_config(config)
    golden["status"] = cert_status
    px, py = place_slot_xy(
        place_slot_index,
        origin=config.place_origin_xy,
        spacing=config.place_slot_spacing_m,
    )
    golden["place"] = {
        "slot_index": int(place_slot_index),
        "slot_name": "slot_%d" % (int(place_slot_index) + 1),
        "deposit_xy": [round(px, 3), round(py, 3)],
        "release_tcp_z": float(DEFAULT_PLACE_RELEASE_Z_M),
        "approach_tcp_z": float(DEFAULT_PLACE_APPROACH_Z_M),
    }
    pick_only = bool(config.certify_pick_only)
    golden["validation"] = {
        "result": cert_status,
        "validation_mode": (
            "moveit_plan_only_pick_sequence"
            if pick_only
            else "moveit_plan_only_no_motion"
        ),
        "pick_plan_ok": True,
        "lift_plan_ok": not pick_only,
        "place_plan_ok": not pick_only,
        "return_home_plan_ok": not pick_only,
        "lift_place_home_deferred": pick_only,
        "attach_result": "NOT_EXECUTED",
        "pick_step": int(pick_step),
        "pick_order": list(pick_order),
        "remaining_obstacles_at_pick": list(remaining_obstacles),
        "runtime_pick_place_confirm_required": True,
        "promote_to_status_after_runtime": VALIDATED_STATUS,
    }
    return golden


def write_plan_preflight_bundle(
    *,
    scene_id: str,
    output_dir: str,
    objects: Dict[str, LayoutObjectPose],
    pick_order: Sequence[str],
    cycle_results: Sequence[PickCyclePlanResult],
    config: PlanPreflightCertifierConfig,
) -> str:
    picks: Dict[str, CertifiedPickRecord] = {}
    for cycle in cycle_results:
        if cycle.pick_result is None:
            continue
        picks[cycle.label] = CertifiedPickRecord(
            label=cycle.label,
            pick_step=cycle.pick_step,
            pose=objects[cycle.label],
            scan_result=cycle.pick_result,
            remaining_obstacles=cycle.remaining_obstacles,
        )
    layout = CertifiedLayout(
        scene_id=scene_id,
        layout_index=1,
        pick_order=tuple(pick_order),
        objects=objects,
        picks=picks,
    )
    bundle_dir = os.path.join(str(output_dir), str(scene_id))
    os.makedirs(bundle_dir, exist_ok=True)

    cert_status = certified_status_for_config(config)
    pick_only = bool(config.certify_pick_only)
    scene = build_scene_yaml(
        layout,
        scene_id=scene_id,
        description=(
            "Escena certificada plan-only (pick secuencial)"
            if pick_only
            else "Escena certificada plan-only (pick+lift+place+home) sin movimiento real."
        ),
    )
    scene["certification"] = {
        "status": cert_status,
        "validator": "demo_scene_plan_preflight_certifier",
        "validation_mode": (
            "moveit_plan_only_pick_sequence"
            if pick_only
            else "moveit_plan_only_no_motion"
        ),
        "criteria": (
            "sequential pick IK+descend with remaining MoveIt obstacles"
            if pick_only
            else (
                "sequential: pick IK+descend, lift cartesian, place plan, home plan "
                "with MoveIt collision objects (YCB dims)"
            )
        ),
        "runtime_confirm_required": True,
    }
    scene_path = os.path.join(bundle_dir, "scene.yaml")
    with open(scene_path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(scene, handle, default_flow_style=False, sort_keys=False)

    golden_paths: Dict[str, str] = {}
    for step_idx, label in enumerate(pick_order):
        cycle = next(c for c in cycle_results if c.label == label)
        if cycle.pick_result is None:
            continue
        golden = build_plan_preflight_golden_yaml(
            scene_id=scene_id,
            label=label,
            pose=objects[label],
            pick_result=cycle.pick_result,
            pick_step=cycle.pick_step,
            pick_order=pick_order,
            remaining_obstacles=cycle.remaining_obstacles,
            place_slot_index=step_idx,
            config=config,
        )
        path = os.path.join(bundle_dir, "golden_%s.yaml" % label)
        with open(path, "w", encoding="utf-8") as handle:
            yaml.safe_dump(golden, handle, default_flow_style=False, sort_keys=False)
        golden_paths[label] = path

    report = {
        "scene_id": scene_id,
        "status": cert_status,
        "pick_order": list(pick_order),
        "cycles": [
            {
                "label": c.label,
                "pick_step": c.pick_step,
                "pick_ok": c.pick_ok,
                "lift_ok": c.lift_ok,
                "place_ok": c.place_ok,
                "home_ok": c.home_ok,
                "reason": c.reason,
                "remaining_obstacles": list(c.remaining_obstacles),
            }
            for c in cycle_results
        ],
        "golden_files": golden_paths,
        "scene_file": scene_path,
    }
    report_path = os.path.join(bundle_dir, "validation_report.json")
    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
    return bundle_dir


class DemoScenePlanPreflightCertifierNode:
    def __init__(self, config: PlanPreflightCertifierConfig) -> None:
        import rclpy
        from rclpy.node import Node

        self._config = config
        self._node = Node("demo_scene_plan_preflight_certifier")
        self._logger = self._node.get_logger()
        scan_cfg = ReachabilityScanConfig(
            table_z_m=float(config.table_z_m),
            waypoints_yaml=str(config.waypoints_yaml or ""),
            variant_budget=str(config.variant_budget),
            cartesian_fraction_threshold=float(config.cartesian_fraction_threshold),
            joint_limit_margin_min_rad=float(config.joint_limit_margin_min_rad),
            gripper_jaw_axis_offset_rad=float(config.gripper_jaw_axis_offset_rad),
        )
        self._backend = ReachabilityMoveItBackend(self._node, scan_cfg)
        self._publisher = MoveItCollisionScenePublisher(self._backend)

    def run(self) -> int:
        import rclpy

        self._backend.setup()
        if not wait_for_joint_states(self._node, timeout_sec=10.0):
            self._logger.error(
                "[PLAN_PREFLIGHT_NO_JOINT_STATES]\n"
                "Sin /joint_states MoveIt no puede resolver IK con colisiones.\n"
                "Lanza: ros2 launch panda_bringup moveit_plan_preflight_bringup.launch.py"
            )
            write_rejection_summary(
                str(self._config.output_dir),
                rejection_counts={"no_joint_states": 1},
                last_rejections=[],
                probe_ok=False,
                probe_reason="no_joint_states",
            )
            return 3
        self._publisher.publish_table()
        time.sleep(2.0)
        try:
            workspace_js = self._backend.workspace_joint_state()
        except RuntimeError as exc:
            self._logger.error(str(exc))
            return 1

        os.makedirs(self._config.output_dir, exist_ok=True)
        probe_ok, probe_reason = probe_moveit_pick_pipeline(
            self._backend,
            self._publisher,
            workspace_js=workspace_js,
            config=self._config,
            logger=self._logger,
        )
        if probe_ok:
            self._logger.info(
                "[PLAN_PREFLIGHT_PROBE_OK]\nprobe=canonical_cracker_empty_table"
            )
        else:
            self._logger.error(
                "[PLAN_PREFLIGHT_PROBE_FAIL]\n"
                "Aborting: MoveIt pipeline not healthy.\nreason=%s"
                % probe_reason
            )
            write_rejection_summary(
                str(self._config.output_dir),
                rejection_counts={"probe_fail": 1},
                last_rejections=[{"reason": probe_reason}],
                probe_ok=False,
                probe_reason=probe_reason,
            )
            return 3

        certified = 0
        attempts = 0
        seen: set = set()
        manifest_layouts: List[Dict[str, Any]] = []
        rejection_counts: Dict[str, int] = collections.Counter()
        last_rejections: List[Dict[str, Any]] = []

        self._logger.info(
            "[PLAN_PREFLIGHT_START]\n"
            "target_scenes=%d\nmax_attempts=%d\npick_only=%s\noutput=%s"
            % (
                int(self._config.num_scenes),
                int(self._config.max_layout_attempts),
                str(bool(self._config.certify_pick_only)).lower(),
                str(self._config.output_dir),
            )
        )

        for objects in iter_layout_candidates(
            max_candidates=int(self._config.max_layout_attempts),
            rng_seed=int(self._config.rng_seed),
            prepend_canonical=bool(self._config.prepend_canonical),
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
            if fp in seen:
                continue
            seen.add(fp)

            pick_preview = resolve_pick_order_for_layout(objects)
            dist_report = pick_order_distance_breakdown(objects)
            self._logger.info(
                "[PLAN_PREFLIGHT_LAYOUT_TRY]\n"
                "attempt=%d\npick_order=%s\ndistances_m=%s"
                % (
                    attempts + 1,
                    list(pick_preview),
                    [(lb, round(d, 3)) for lb, d in dist_report],
                )
            )

            ok, cycles, reason = validate_layout_plan_preflight(
                self._backend,
                self._publisher,
                objects,
                workspace_js=workspace_js,
                config=self._config,
                logger=self._logger,
            )
            if not ok:
                rejection_counts[str(reason)] += 1
                last_rejections.append(
                    {
                        "attempt": int(attempts),
                        "reason": str(reason),
                        "pick_order": list(pick_preview),
                    }
                )
                if len(last_rejections) > 30:
                    last_rejections.pop(0)
                self._logger.info(
                    "[PLAN_PREFLIGHT_REJECT]\nattempt=%d\nreason=%s"
                    % (attempts, reason)
                )
                continue

            certified += 1
            scene_id = "%s_%02d" % (
                str(self._config.scene_id_prefix),
                certified,
            )
            pick_order = resolve_pick_order_for_layout(objects)
            bundle = write_plan_preflight_bundle(
                scene_id=scene_id,
                output_dir=str(self._config.output_dir),
                objects=objects,
                pick_order=pick_order,
                cycle_results=cycles,
                config=self._config,
            )
            manifest_layouts.append(
                {
                    "scene_id": scene_id,
                    "bundle_dir": bundle,
                    "pick_order": list(pick_order),
                    "fingerprint": list(fp),
                }
            )
            self._logger.info(
                "[PLAN_PREFLIGHT_CERTIFIED]\nscene_id=%s\nbundle=%s"
                % (scene_id, bundle)
            )
            if certified >= int(self._config.num_scenes):
                break
            rclpy.spin_once(self._node, timeout_sec=0.0)

        rejection_path = write_rejection_summary(
            str(self._config.output_dir),
            rejection_counts=dict(rejection_counts),
            last_rejections=last_rejections,
            probe_ok=True,
            probe_reason="OK",
        )
        manifest = {
            "certified_count": certified,
            "attempts_used": attempts,
            "status": certified_status_for_config(self._config),
            "validation_mode": (
                "moveit_plan_only_pick_sequence"
                if bool(self._config.certify_pick_only)
                else "moveit_plan_only_no_motion"
            ),
            "probe_ok": True,
            "layouts": manifest_layouts,
            "rejection_summary": rejection_path,
        }
        manifest_path = os.path.join(str(self._config.output_dir), "manifest.json")
        with open(manifest_path, "w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2, sort_keys=True)

        self._logger.info(
            "[PLAN_PREFLIGHT_DONE]\ncertified=%d\nmanifest=%s\nrejections=%s"
            % (certified, manifest_path, rejection_path)
        )
        if certified == 0:
            top = sorted(rejection_counts.items(), key=lambda kv: -kv[1])[:5]
            self._logger.error(
                "[PLAN_PREFLIGHT_ZERO_CERTIFIED]\n"
                "top_rejection_reasons=%s\nsee=%s"
                % (top, rejection_path)
            )
        return 0 if certified > 0 else 2


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Certificador plan-only de escenas demo (pick+lift+place+home, sin movimiento)."
        )
    )
    parser.add_argument("--num-scenes", type=int, default=3)
    parser.add_argument(
        "--output-dir",
        default="/tmp/plan_preflight_certified_layouts",
    )
    parser.add_argument("--scene-id-prefix", default="plan_certified_layout")
    parser.add_argument("--variant-budget", default="fast")
    parser.add_argument("--max-layout-attempts", type=int, default=120)
    parser.add_argument("--table-z", type=float, default=DEFAULT_TABLE_Z_M)
    parser.add_argument("--waypoints-yaml", default="")
    parser.add_argument("--rng-seed", type=int, default=42)
    parser.add_argument(
        "--prepend-canonical",
        action="store_true",
        help="Incluir demo_scene_02 canónica como primer candidato (cracker suele ir primero).",
    )
    parser.add_argument(
        "--probe-only",
        action="store_true",
        help="Solo prueba MoveIt con cracker en mesa vacía y sale.",
    )
    parser.add_argument(
        "--pick-only",
        action="store_true",
        help=(
            "Certifica solo pick secuencial (4 rondas). Lift/place/home quedan para runtime."
        ),
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    try:
        import rclpy
    except ImportError:
        print("ROS/rclpy required.", file=sys.stderr)
        return 1

    config = PlanPreflightCertifierConfig(
        num_scenes=int(args.num_scenes),
        output_dir=str(args.output_dir),
        scene_id_prefix=str(args.scene_id_prefix),
        variant_budget=str(args.variant_budget),
        max_layout_attempts=int(args.max_layout_attempts),
        table_z_m=float(args.table_z),
        waypoints_yaml=str(args.waypoints_yaml or ""),
        rng_seed=int(args.rng_seed),
        prepend_canonical=bool(args.prepend_canonical),
        certify_pick_only=bool(args.pick_only),
    )
    rclpy.init()
    node = DemoScenePlanPreflightCertifierNode(config)
    try:
        if bool(args.probe_only):
            node._backend.setup()
            if not wait_for_joint_states(node._node, timeout_sec=8.0):
                node._logger.error(
                    "[PLAN_PREFLIGHT_NO_JOINT_STATES]\n"
                    "No hay /joint_states. Lanza moveit_plan_preflight_bringup.launch.py "
                    "o publica joint_states antes del probe."
                )
                return 3
            node._publisher.publish_table()
            time.sleep(2.0)
            workspace_js = node._backend.workspace_joint_state()
            ok, reason = probe_moveit_pick_pipeline(
                node._backend,
                node._publisher,
                workspace_js=workspace_js,
                config=config,
                logger=node._logger,
            )
            node._logger.info(
                "[PLAN_PREFLIGHT_PROBE_ONLY]\nok=%s\nreason=%s"
                % (str(ok).lower(), reason)
            )
            return 0 if ok else 3
        return int(node.run())
    finally:
        node._node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
