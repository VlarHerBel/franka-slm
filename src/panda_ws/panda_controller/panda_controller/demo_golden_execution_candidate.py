"""Golden execution candidate v2: ruta end-to-end reproducible para demo_scene_02 + cracker_box."""

from __future__ import annotations

import csv
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import yaml

from panda_controller.demo_golden_pick_candidate import (
    GOLDEN_CENTER_XY_TOL_M,
    GOLDEN_TOP_Z_TOL_M,
    GOLDEN_YAW_TOL_DEG,
    VALIDATED_STATUS as V1_VALIDATED_STATUS,
    _normalize_xyz,
    _normalize_xy,
    _wrap_to_pi,
    _yaw_delta_rad,
    default_demo_config_dir,
    resolve_golden_candidate_path,
    resolve_runtime_scene_yaw_rad,
)
from panda_controller.tfg_motion_waypoints import PANDA_ARM_JOINT_NAMES

GOLDEN_EXECUTION_SCHEMA_VERSION = 2
GOLDEN_EXECUTION_TYPE = "full_execution_candidate"
VALIDATED_FULL_EXECUTION_STATUS = "validated_full_execution"
VALIDATED_PICK_ONLY_STATUS = "validated_pick_only"
VALIDATED_PICK_TRANSPORT_STATUS = "validated_pick_transport"
RECORDED_NOT_VALIDATED_STATUS = "recorded_not_validated"

GOLDEN_EXECUTION_SCOPE_SCENE_ID = "demo_scene_02"
GOLDEN_EXECUTION_SCOPE_TARGET_LABEL = "cracker_box"
GOLDEN_EXECUTION_SCOPE_SLOT_INDEX = 0

OBSTACLE_POSE_TOL_M = 0.02
HOME_JOINT_TOL_RAD = 0.05
GRIPPER_OPEN_MIN_M = 0.035
MIN_GOLDEN_DESCEND_DELTA_Z_M = 0.03
MIN_GOLDEN_LIFT_DELTA_Z_M = 0.03
PREGRASP_GOAL_Z_MARGIN_M = 0.03

PHASE_NAMES: Tuple[str, ...] = (
    "home_to_pick_workspace_ready",
    "approach_to_pregrasp",
    "gripper_axis_alignment",
    "open_gripper_at_pregrasp",
    "cartesian_descend_to_grasp",
    "close_gripper",
    "attach_and_verify",
    "cartesian_lift",
    "post_lift_local_escape",
    "transport_entry_to_safe_hub",
    "deterministic_transport",
    "place_approach",
    "place_release",
    "open_detach",
    "place_retreat",
    "return_home",
)


def default_full_execution_golden_path(
    scene_id: str,
    target_label: str,
    *,
    slot_index: int = 0,
    config_dir: Optional[str] = None,
) -> str:
    sid = str(scene_id or "").strip().lower()
    label = str(target_label or "").strip().lower()
    rel = (
        f"demo_candidate_cache/{sid}_{label}_slot_{int(slot_index)}"
        f"_full_execution_golden.yaml"
    )
    return resolve_golden_candidate_path(rel, config_dir=config_dir)


def golden_execution_scope_active(
    *,
    scene_id: str,
    target_label: str,
    place_slot_index: int,
) -> bool:
    return (
        str(scene_id or "").strip().lower() == GOLDEN_EXECUTION_SCOPE_SCENE_ID
        and str(target_label or "").strip().lower() == GOLDEN_EXECUTION_SCOPE_TARGET_LABEL
        and int(place_slot_index) == GOLDEN_EXECUTION_SCOPE_SLOT_INDEX
    )


def _joint_list_from_dict(joints: Dict[str, Any]) -> List[float]:
    return [float(joints.get(n, 0.0)) for n in PANDA_ARM_JOINT_NAMES]


def _joint_dict_from_list(values: Sequence[float]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for i, name in enumerate(PANDA_ARM_JOINT_NAMES):
        if i < len(values):
            out[name] = float(values[i])
    return out


def load_golden_execution_candidate(yaml_path: str) -> Optional[Dict[str, Any]]:
    path = str(yaml_path or "").strip()
    if not path or not os.path.isfile(path):
        return None
    with open(path, "r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    return normalize_golden_execution_candidate(raw, source_file=path)


def normalize_golden_execution_candidate(
    raw: Dict[str, Any],
    *,
    source_file: str = "",
) -> Optional[Dict[str, Any]]:
    if not isinstance(raw, dict):
        return None
    schema = int(raw.get("golden_candidate_schema_version", 0))
    if schema != GOLDEN_EXECUTION_SCHEMA_VERSION:
        return None
    if str(raw.get("type", "")).strip() != GOLDEN_EXECUTION_TYPE:
        return None
    sid = str(raw.get("scene_id", "")).strip().lower()
    label = str(raw.get("target_label", "")).strip().lower()
    status = str(raw.get("status", "")).strip().lower()
    if not sid or not label:
        return None
    phases = raw.get("phases")
    if not isinstance(phases, list) or not phases:
        return None
    normalized_phases: List[Dict[str, Any]] = []
    for phase in phases:
        if not isinstance(phase, dict):
            return None
        name = str(phase.get("name", "")).strip()
        ptype = str(phase.get("type", "")).strip()
        if not name or not ptype:
            return None
        normalized_phases.append(dict(phase))
    return {
        "golden_candidate_schema_version": GOLDEN_EXECUTION_SCHEMA_VERSION,
        "type": GOLDEN_EXECUTION_TYPE,
        "scene_id": sid,
        "target_label": label,
        "layout_version": str(raw.get("layout_version", "")).strip(),
        "status": status,
        "source_file": str(source_file),
        "scene_signature": dict(raw.get("scene_signature") or {}),
        "preconditions": dict(raw.get("preconditions") or {}),
        "geometric_candidate": dict(raw.get("geometric_candidate") or {}),
        "kinematic_solution": dict(raw.get("kinematic_solution") or {}),
        "execution_route": dict(raw.get("execution_route") or {}),
        "transport_route": dict(raw.get("transport_route") or {}),
        "place_policy": dict(raw.get("place_policy") or {}),
        "phases": normalized_phases,
        "validation": dict(raw.get("validation") or {}),
        "record_metadata": dict(raw.get("record_metadata") or {}),
    }


def validate_golden_execution_identity(
    golden: Dict[str, Any],
    *,
    scene_id: str,
    target_label: str,
    slot_index: int,
) -> Tuple[bool, str]:
    sid = str(scene_id or "").strip().lower()
    label = str(target_label or "").strip().lower()
    if str(golden.get("scene_id", "")).strip().lower() != sid:
        return False, "scene_id_mismatch"
    if str(golden.get("target_label", "")).strip().lower() != label:
        return False, "target_label_mismatch"
    sig = golden.get("scene_signature") or {}
    sig_slot = sig.get("place_slot_index")
    if sig_slot is not None and int(sig_slot) != int(slot_index):
        return False, "place_slot_index_mismatch"
    status = str(golden.get("status", "")).strip().lower()
    if status != VALIDATED_FULL_EXECUTION_STATUS:
        return False, "status_not_validated_full_execution"
    return True, "OK"


def validate_scene_signature_compatibility(
    golden: Dict[str, Any],
    *,
    runtime_xy: Tuple[float, float],
    runtime_top_z: float,
    runtime_scene_yaw_rad: Optional[float],
    runtime_scene_yaw_source: str,
    runtime_commanded_tcp_yaw_rad: Optional[float],
    scene_obstacles: Sequence[Dict[str, Any]],
    completed_labels: Optional[Sequence[str]] = None,
) -> Tuple[bool, str, Dict[str, Any]]:
    sig = golden.get("scene_signature") or {}
    tol = sig.get("compatibility_tolerances") or {}
    center_tol = float(tol.get("center_xy_tol_m", GOLDEN_CENTER_XY_TOL_M))
    yaw_tol = float(tol.get("yaw_tol_deg", GOLDEN_YAW_TOL_DEG))
    top_z_tol = float(tol.get("top_z_tol_m", GOLDEN_TOP_Z_TOL_M))
    obs_tol = float(tol.get("obstacle_pose_tol_m", OBSTACLE_POSE_TOL_M))

    exp_xy = _normalize_xy(sig.get("target_center_xyz"))
    if exp_xy is None:
        exp_xy = _normalize_xy(sig.get("semantic_center_xy"))
    if exp_xy is None:
        return False, "missing_golden_target_xy", {}

    golden_yaw = _to_float(sig.get("target_yaw_rad"))
    golden_cmd = _to_float((golden.get("geometric_candidate") or {}).get("commanded_tcp_yaw_rad"))
    exp_top_z = _to_float(sig.get("target_top_z"))
    if exp_top_z is None:
        exp_top_z = float(runtime_top_z)

    center_err = math.hypot(
        float(runtime_xy[0]) - float(exp_xy[0]),
        float(runtime_xy[1]) - float(exp_xy[1]),
    )
    top_z_err = abs(float(runtime_top_z) - float(exp_top_z))

    scene_yaw_source = str(runtime_scene_yaw_source or "none").strip().lower()
    has_explicit_scene_yaw = (
        runtime_scene_yaw_rad is not None
        and scene_yaw_source not in ("", "none", "missing")
        and not scene_yaw_source.endswith("non_explicit_source")
    )
    yaw_compare_mode = ""
    yaw_err_deg = float("inf")
    if has_explicit_scene_yaw and golden_yaw is not None:
        yaw_compare_mode = "scene_yaw_vs_scene_yaw"
        yaw_err_deg = math.degrees(_yaw_delta_rad(float(runtime_scene_yaw_rad), golden_yaw))
    elif runtime_commanded_tcp_yaw_rad is not None and golden_cmd is not None:
        yaw_compare_mode = "commanded_tcp_yaw_vs_commanded_tcp_yaw"
        yaw_err_deg = math.degrees(
            _yaw_delta_rad(float(runtime_commanded_tcp_yaw_rad), golden_cmd)
        )

    details: Dict[str, Any] = {
        "center_error_m": float(center_err),
        "yaw_error_deg": float(yaw_err_deg),
        "top_z_error_m": float(top_z_err),
        "yaw_compare_mode": yaw_compare_mode,
        "center_xy_tol_m": center_tol,
        "yaw_tol_deg": yaw_tol,
        "top_z_tol_m": top_z_tol,
    }

    if center_err > center_tol:
        return False, "center_xy_out_of_tolerance", details
    if not yaw_compare_mode:
        return False, "missing_runtime_yaw", details
    if yaw_err_deg > yaw_tol:
        return False, "yaw_out_of_tolerance", details
    if top_z_err > top_z_tol:
        return False, "top_z_out_of_tolerance", details

    golden_obstacles = sig.get("obstacles") or []
    if isinstance(golden_obstacles, list) and golden_obstacles:
        obs_ok, obs_reason = _obstacles_match(
            golden_obstacles, scene_obstacles, tol_m=obs_tol
        )
        details["obstacles_ok"] = obs_ok
        if not obs_ok:
            return False, obs_reason, details

    if completed_labels is not None:
        exp_completed = sig.get("completed_labels") or []
        if isinstance(exp_completed, list) and exp_completed:
            for label in exp_completed:
                if str(label).strip().lower() not in {
                    str(x).strip().lower() for x in completed_labels
                }:
                    return False, "completed_labels_mismatch", details

    return True, "OK", details


def _to_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _obstacles_match(
    golden_obstacles: Sequence[Dict[str, Any]],
    runtime_obstacles: Sequence[Dict[str, Any]],
    *,
    tol_m: float,
) -> Tuple[bool, str]:
    runtime_by_label: Dict[str, Dict[str, Any]] = {}
    for obs in runtime_obstacles:
        if not isinstance(obs, dict):
            continue
        lbl = str(obs.get("label") or "").strip().lower()
        if lbl:
            runtime_by_label[lbl] = obs

    for g_obs in golden_obstacles:
        if not isinstance(g_obs, dict):
            continue
        lbl = str(g_obs.get("label") or "").strip().lower()
        if not lbl:
            continue
        r_obs = runtime_by_label.get(lbl)
        if r_obs is None:
            return False, "obstacle_missing:%s" % lbl
        g_xyz = _normalize_xyz(g_obs.get("center_xyz"))
        r_xyz = _normalize_xyz(
            r_obs.get("center_xyz")
            or r_obs.get("position")
            or [
                r_obs.get("x"),
                r_obs.get("y"),
                r_obs.get("z"),
            ]
        )
        if g_xyz is None or r_xyz is None:
            return False, "obstacle_pose_missing:%s" % lbl
        err = math.sqrt(
            (g_xyz[0] - r_xyz[0]) ** 2
            + (g_xyz[1] - r_xyz[1]) ** 2
            + (g_xyz[2] - r_xyz[2]) ** 2
        )
        if err > float(tol_m):
            return False, "obstacle_pose_out_of_tolerance:%s" % lbl
    return True, "OK"


def build_scene_signature_from_runtime(
    *,
    scene_id: str,
    target_label: str,
    candidate: Dict[str, Any],
    scene_obstacles: Sequence[Dict[str, Any]],
    place_slot_index: int,
    place_slot_xyz: Optional[Tuple[float, float, float]] = None,
    completed_labels: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    label = str(target_label or "").strip().lower()
    center = (
        _normalize_xyz(candidate.get("chosen_target_center_base"))
        or _normalize_xyz(candidate.get("position"))
        or (0.0, 0.0, 0.0)
    )
    top_z = _to_float(candidate.get("top_z_m") or candidate.get("top_z_estimated")) or 0.0
    scene_yaw, _ = resolve_runtime_scene_yaw_rad(candidate, target_label=label)
    geom = candidate.get("_golden_geometric") or {}
    obstacles: List[Dict[str, Any]] = []
    for obs in scene_obstacles:
        if not isinstance(obs, dict):
            continue
        obs_label = str(obs.get("label") or "").strip().lower()
        if not obs_label or obs_label == label:
            continue
        xyz = _normalize_xyz(
            obs.get("center_xyz")
            or obs.get("position")
            or [obs.get("x"), obs.get("y"), obs.get("z")]
        )
        if xyz is None:
            continue
        obstacles.append(
            {
                "label": obs_label,
                "entity_name": str(obs.get("entity_name") or obs.get("gazebo_name") or ""),
                "center_xyz": list(xyz),
                "yaw_rad": _to_float(obs.get("yaw_rad") or obs.get("spawn_yaw_rad")),
                "dims": obs.get("dims") or obs.get("dimensions"),
            }
        )
    return {
        "scene_id": str(scene_id or "").strip().lower(),
        "target_label": label,
        "target_entity_name": str(
            candidate.get("entity_name") or candidate.get("gazebo_name") or ""
        ),
        "target_center_xyz": list(center),
        "target_yaw_rad": float(scene_yaw if scene_yaw is not None else 0.0),
        "target_top_z": float(top_z),
        "target_dims_lwh": candidate.get("dims_lwh") or candidate.get("dimensions"),
        "obstacles": obstacles,
        "place_slot_index": int(place_slot_index),
        "place_slot_xyz": list(place_slot_xyz) if place_slot_xyz else None,
        "completed_labels": list(completed_labels or []),
        "compatibility_tolerances": {
            "center_xy_tol_m": GOLDEN_CENTER_XY_TOL_M,
            "yaw_tol_deg": GOLDEN_YAW_TOL_DEG,
            "top_z_tol_m": GOLDEN_TOP_Z_TOL_M,
            "obstacle_pose_tol_m": OBSTACLE_POSE_TOL_M,
        },
    }


def apply_golden_execution_runtime_overrides(
    candidate: Dict[str, Any],
    golden: Dict[str, Any],
) -> None:
    """Aplica overrides de transporte/place desde golden v2 (o v1 compatible)."""
    from panda_controller.demo_golden_pick_candidate import apply_golden_runtime_overrides

    transport = golden.get("transport_route") or golden.get("transport") or {}
    place = golden.get("place_policy") or golden.get("place") or {}
    cand = golden.get("geometric_candidate") or golden.get("candidate") or {}
    apply_golden_runtime_overrides(
        candidate,
        {
            "candidate": cand,
            "transport": {
                "selected_transport_entry": transport.get("selected_transport_entry"),
                "route": transport.get("route"),
            },
            "place": place,
        },
    )


def golden_execution_to_plan_targets(
    golden: Dict[str, Any],
) -> Dict[str, Any]:
    """Construye plan_targets y grasp_valid sintéticos para replay."""
    geom = golden.get("geometric_candidate") or {}
    kin = golden.get("kinematic_solution") or {}
    place = golden.get("place_policy") or {}
    pre = _normalize_xyz(geom.get("pregrasp_tcp")) or (0.0, 0.0, 0.0)
    gr = _normalize_xyz(geom.get("grasp_tcp")) or pre
    lift = _normalize_xyz(geom.get("lift_tcp")) or (
        pre[0],
        pre[1],
        pre[2] + 0.025,
    )
    cmd_yaw = float(geom.get("commanded_tcp_yaw_rad", 0.0))
    return {
        "pregrasp_tcp": pre,
        "grasp_tcp": gr,
        "lift_tcp": lift,
        "safe_pregrasp_tcp": pre,
        "pregrasp_legacy_with_offset": pre,
        "commanded_tcp_yaw_rad": cmd_yaw,
        "candidate_idx": int(geom.get("candidate_idx", 0)),
        "aligned_pregrasp_js": kin.get("aligned_pregrasp_js"),
        "grasp_js": kin.get("grasp_js"),
        "lift_js": kin.get("lift_js"),
        "release_tcp_z": float(place.get("release_tcp_z", 0.0)),
        "place_slot_index": int(place.get("slot_index", 0)),
    }


def save_golden_execution_candidate(
    golden: Dict[str, Any],
    yaml_path: str,
) -> bool:
    path = str(yaml_path or "").strip()
    if not path:
        return False
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    payload = {
        "golden_candidate_schema_version": GOLDEN_EXECUTION_SCHEMA_VERSION,
        "type": GOLDEN_EXECUTION_TYPE,
        "scene_id": golden.get("scene_id"),
        "layout_version": golden.get("layout_version"),
        "target_label": golden.get("target_label"),
        "status": golden.get("status", VALIDATED_FULL_EXECUTION_STATUS),
        "scene_signature": golden.get("scene_signature") or {},
        "preconditions": golden.get("preconditions") or {},
        "geometric_candidate": golden.get("geometric_candidate") or {},
        "kinematic_solution": golden.get("kinematic_solution") or {},
        "execution_route": golden.get("execution_route") or {},
        "transport_route": golden.get("transport_route") or {},
        "place_policy": golden.get("place_policy") or {},
        "phases": golden.get("phases") or [],
        "validation": golden.get("validation") or {},
        "record_metadata": golden.get("record_metadata") or {},
    }
    with open(path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, default_flow_style=False, sort_keys=False)
    return True


def build_v2_from_v1_golden_and_waypoints(
    v1_golden: Dict[str, Any],
    *,
    waypoints_data: Dict[str, Any],
    slot_index: int = 0,
) -> Optional[Dict[str, Any]]:
    """Construye plantilla v2 a partir del golden v1 y waypoints articulares."""
    from panda_controller.tfg_motion_waypoints import get_waypoint_joint_positions

    cand = v1_golden.get("candidate") or {}
    transport = v1_golden.get("transport") or {}
    place = v1_golden.get("place") or {}
    grasp = v1_golden.get("grasp") or {}
    pose = v1_golden.get("object_pose") or {}

    def _wp_js(name: str) -> Optional[List[float]]:
        joints = get_waypoint_joint_positions(waypoints_data, name)
        return [float(v) for v in joints] if joints else None

    home_js = _wp_js("home")
    pwr_js = _wp_js("pick_workspace_ready")
    if home_js is None or pwr_js is None:
        return None

    route = list(transport.get("route") or [])
    transport_js: List[Dict[str, Any]] = []
    for wp in route:
        js = _wp_js(str(wp))
        if js is not None:
            transport_js.append({"waypoint": str(wp), "joints": js})

    pre = cand.get("pregrasp_tcp") or [0.0, 0.0, 0.0]
    gr = cand.get("grasp_tcp") or pre
    lift = cand.get("lift_tcp") or [pre[0], pre[1], pre[2] + 0.025]

    phases: List[Dict[str, Any]] = [
        {
            "name": "home_to_pick_workspace_ready",
            "type": "joint_trajectory",
            "joint_names": list(PANDA_ARM_JOINT_NAMES),
            "goal_waypoint": "pick_workspace_ready",
            "points": [{"positions": pwr_js, "time_from_start_s": 5.0}],
            "duration_s": 5.0,
        },
        {
            "name": "approach_to_pregrasp",
            "type": "cartesian_or_joint_trajectory",
            "tcp_goal": list(pre),
            "goal_js": None,
            "duration_s": 4.0,
        },
        {
            "name": "gripper_axis_alignment",
            "type": "joint_adjustment",
            "joint7_target": None,
            "expected_gap_axis_error_deg": 2.0,
        },
        {
            "name": "open_gripper_at_pregrasp",
            "type": "gripper_command",
            "open_joint": float(grasp.get("open_joint", 0.0399)),
            "command": "open",
            "duration_s": 1.0,
        },
        {
            "name": "cartesian_descend_to_grasp",
            "type": "cartesian_or_joint_trajectory",
            "start_tcp": list(pre),
            "goal_tcp": list(gr),
            "depth_from_top_m": float(cand.get("depth_from_top_m", 0.033)),
            "fraction": float(cand.get("cartesian_descend_fraction", 1.0)),
            "duration_s": 3.0,
        },
        {
            "name": "close_gripper",
            "type": "gripper_command",
            "open_joint": float(grasp.get("open_joint", 0.0399)),
            "close_joint": float(grasp.get("close_joint", 0.0270)),
            "expected_width": float(grasp.get("expected_width_m", 0.06)),
        },
        {"name": "attach_and_verify", "type": "attach", "expected_contact_policy": "strict"},
        {
            "name": "cartesian_lift",
            "type": "trajectory",
            "start_tcp": list(gr),
            "goal_tcp": list(lift),
            "duration_s": 2.5,
        },
        {
            "name": "post_lift_local_escape",
            "type": "trajectory",
            "selected_mode": str(transport.get("selected_transport_entry", "")),
            "duration_s": 4.0,
        },
        {
            "name": "transport_entry_to_safe_hub",
            "type": "direct_action_or_joint_trajectory",
            "target_waypoint": str(transport.get("first_hub", "carry_mid_high")),
            "duration_s": 6.0,
        },
        {
            "name": "deterministic_transport",
            "type": "waypoint_sequence",
            "backend": str(transport.get("backend", "direct_action")),
            "sequence": route,
            "per_segment_times_s": [5.0, 10.0, 20.0, 27.0],
            "joint_waypoints": transport_js,
        },
        {
            "name": "place_approach",
            "type": "cartesian_or_joint_trajectory",
            "tcp_goal": [
                float(place.get("deposit_xy", [-0.37, 0.08])[0]),
                float(place.get("deposit_xy", [-0.37, 0.08])[1]),
                float(place.get("approach_tcp_z", 0.65)),
            ],
            "duration_s": 4.0,
        },
        {
            "name": "place_release",
            "type": "cartesian_or_joint_trajectory",
            "tcp_goal": [
                float(place.get("deposit_xy", [-0.37, 0.08])[0]),
                float(place.get("deposit_xy", [-0.37, 0.08])[1]),
                float(place.get("release_tcp_z", 0.329)),
            ],
            "release_tcp_z": float(place.get("release_tcp_z", 0.329)),
            "duration_s": 3.0,
        },
        {"name": "open_detach", "type": "gripper_and_detach"},
        {
            "name": "place_retreat",
            "type": "cartesian_or_joint_trajectory",
            "tcp_goal": [
                float(place.get("deposit_xy", [-0.37, 0.08])[0]),
                float(place.get("deposit_xy", [-0.37, 0.08])[1]),
                float(place.get("retreat_tcp_z", 0.65)),
            ],
            "duration_s": 3.0,
        },
        {
            "name": "return_home",
            "type": "joint_trajectory",
            "goal_waypoint": "home",
            "points": [{"positions": home_js, "time_from_start_s": 4.0}],
            "duration_s": 4.0,
        },
    ]

    return normalize_golden_execution_candidate(
        {
            "golden_candidate_schema_version": GOLDEN_EXECUTION_SCHEMA_VERSION,
            "type": GOLDEN_EXECUTION_TYPE,
            "scene_id": v1_golden.get("scene_id"),
            "layout_version": v1_golden.get("layout_version"),
            "target_label": v1_golden.get("target_label"),
            "status": VALIDATED_FULL_EXECUTION_STATUS,
            "scene_signature": {
                "scene_id": v1_golden.get("scene_id"),
                "target_label": v1_golden.get("target_label"),
                "target_center_xyz": list(pose.get("semantic_center_xy") or [0.455, 0.115])
                + [float(pose.get("top_z", 0.47))],
                "target_yaw_rad": float(pose.get("yaw_rad", 0.0)),
                "target_top_z": float(pose.get("top_z", 0.47)),
                "place_slot_index": int(slot_index),
                "place_slot_xyz": list(place.get("deposit_xy") or [-0.37, 0.08])
                + [float(place.get("release_tcp_z", 0.329))],
                "obstacles": [],
                "completed_labels": [],
                "compatibility_tolerances": {
                    "center_xy_tol_m": GOLDEN_CENTER_XY_TOL_M,
                    "yaw_tol_deg": GOLDEN_YAW_TOL_DEG,
                    "top_z_tol_m": GOLDEN_TOP_Z_TOL_M,
                    "obstacle_pose_tol_m": OBSTACLE_POSE_TOL_M,
                },
            },
            "preconditions": {
                "robot_start": "HOME",
                "gripper_open": True,
                "no_object_attached": True,
                "target_present": True,
                "target_not_completed": True,
                "target_collision_before_approach": True,
                "runtime_scene_gt_available": True,
                "place_slot_free": True,
            },
            "geometric_candidate": {
                "candidate_idx": int(cand.get("candidate_idx", 0)),
                "yaw_deg": float(cand.get("yaw_deg", 0.0)),
                "commanded_tcp_yaw_rad": float(cand.get("commanded_tcp_yaw_rad", 0.0)),
                "pregrasp_tcp": list(pre),
                "grasp_tcp": list(gr),
                "lift_tcp": list(lift),
                "depth_from_top_m": float(cand.get("depth_from_top_m", 0.033)),
                "ik_seed": str(cand.get("ik_seed", "pick_workspace_ready")),
                "prevalidation_source": str(cand.get("prevalidation_source", "")),
                "cartesian_descend_fraction": float(
                    cand.get("cartesian_descend_fraction", 1.0)
                ),
            },
            "kinematic_solution": {
                "ik_seed": str(cand.get("ik_seed", "pick_workspace_ready")),
            },
            "execution_route": {"phase_count": len(phases)},
            "transport_route": {
                "selected_transport_entry": str(
                    transport.get("selected_transport_entry", "")
                ),
                "route": route,
                "backend": str(transport.get("backend", "direct_action")),
            },
            "place_policy": {
                "slot_index": int(slot_index),
                "slot_name": str(place.get("slot_name", "slot_1")),
                "deposit_xy": list(place.get("deposit_xy") or [-0.37, 0.08]),
                "release_tcp_z": float(place.get("release_tcp_z", 0.329)),
                "approach_tcp_z": float(place.get("approach_tcp_z", 0.65)),
                "retreat_tcp_z": float(place.get("retreat_tcp_z", 0.65)),
                "release_source": str(place.get("release_source", "")),
            },
            "phases": phases,
            "validation": dict(v1_golden.get("validation") or {}),
            "record_metadata": {
                "source": "v1_golden_migration",
                "v1_status": str(v1_golden.get("status", V1_VALIDATED_STATUS)),
            },
        }
    )


def build_recorded_kinematic_solution(
    snapshots: Dict[str, Any],
) -> Dict[str, Any]:
    """Construye kinematic_solution desde snapshots JS capturados en ejecución real."""
    out: Dict[str, Any] = {}
    for key in (
        "home_js",
        "pick_workspace_ready_js",
        "pregrasp_js",
        "aligned_pregrasp_js",
        "grasp_js",
        "post_lift_js",
        "transport_entry_js",
        "box_high_js",
        "place_approach_js",
        "place_release_js",
        "place_retreat_js",
        "return_home_js",
    ):
        val = snapshots.get(key)
        if isinstance(val, (list, tuple)) and val:
            out[key] = [float(v) for v in val]
    if snapshots.get("selected_local_exit"):
        out["selected_local_exit"] = str(snapshots["selected_local_exit"])
    if snapshots.get("ik_seed"):
        out["ik_seed"] = str(snapshots["ik_seed"])
    return out


def build_recorded_phases_from_snapshots(
    snapshots: Dict[str, Any],
    *,
    transport_route: Sequence[str],
    per_segment_times_s: Optional[Sequence[float]] = None,
    place_policy: Optional[Dict[str, Any]] = None,
    geometric: Optional[Dict[str, Any]] = None,
    base_phases: Optional[Sequence[Dict[str, Any]]] = None,
    transport: Optional[Dict[str, Any]] = None,
    grasp: Optional[Dict[str, Any]] = None,
    waypoints_data: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """Parchea plantilla base de 16 fases; nunca devuelve lista parcial."""
    phases = [dict(p) for p in (base_phases or [])]
    merge_golden_execution_phases_from_snapshots(
        phases,
        snapshots=snapshots,
        geometric=geometric or {},
        transport=dict(
            transport
            or {"route": list(transport_route), "backend": "direct_action"}
        ),
        place_policy=place_policy or {},
        grasp=grasp or {},
        waypoints_data=waypoints_data,
        per_segment_times_s=per_segment_times_s,
    )
    return order_golden_execution_phases(phases)


def order_golden_execution_phases(
    phases: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    by_name: Dict[str, Dict[str, Any]] = {}
    for phase in phases:
        if not isinstance(phase, dict):
            continue
        name = str(phase.get("name", "")).strip().lower()
        if name:
            by_name[name] = dict(phase)
    ordered: List[Dict[str, Any]] = []
    for name in PHASE_NAMES:
        ordered.append(by_name.get(name, {"name": name, "type": "unknown"}))
    return ordered


def merge_golden_execution_phases_from_snapshots(
    phases: List[Dict[str, Any]],
    *,
    snapshots: Dict[str, Any],
    geometric: Dict[str, Any],
    transport: Dict[str, Any],
    place_policy: Dict[str, Any],
    grasp: Optional[Dict[str, Any]] = None,
    waypoints_data: Optional[Dict[str, Any]] = None,
    per_segment_times_s: Optional[Sequence[float]] = None,
) -> None:
    """Parchea fases base in-place con datos capturados en ejecución real."""
    from panda_controller.tfg_motion_waypoints import get_waypoint_joint_positions

    grasp = grasp or {}
    ordered = order_golden_execution_phases(phases)
    phases[:] = ordered

    def _phase(name: str) -> Dict[str, Any]:
        phase = find_golden_phase(phases, name)
        if phase is None:
            phase = {"name": name, "type": "unknown"}
            phases.append(phase)
        return phase

    pwr = _normalize_arm_joint_list(snapshots.get("pick_workspace_ready_js"))
    if pwr is not None:
        _phase("home_to_pick_workspace_ready").update(
            {
                "type": "joint_trajectory",
                "joint_names": list(PANDA_ARM_JOINT_NAMES),
                "goal_waypoint": "pick_workspace_ready",
                "goal_js": pwr,
                "duration_s": float(
                    snapshots.get("home_to_pwr_duration_s", 5.0)
                ),
            }
        )

    patch_approach_to_pregrasp_from_snapshots(phases, snapshots, geometric)

    j7_target = snapshots.get("gripper_axis_joint7_target")
    if j7_target is not None:
        axis_phase = _phase("gripper_axis_alignment")
        axis_phase.update(
            {
                "type": "joint_adjustment",
                "joint7_target": float(j7_target),
                "expected_gap_axis_error_deg": float(
                    snapshots.get("gripper_axis_expected_error_deg", 2.0)
                ),
                "result": str(snapshots.get("gripper_axis_result", "OK")),
            }
        )
        start_js = _normalize_arm_joint_list(snapshots.get("gripper_axis_start_js"))
        end_js = _normalize_arm_joint_list(snapshots.get("gripper_axis_end_js"))
        if start_js is not None:
            axis_phase["start_js"] = start_js
        if end_js is not None:
            axis_phase["end_js"] = end_js

    if snapshots.get("open_gripper_recorded") or snapshots.get("open_gripper_joint"):
        _phase("open_gripper_at_pregrasp").update(
            {
                "type": "gripper_command",
                "command": "open",
                "open_joint": float(
                    snapshots.get(
                        "open_gripper_joint", grasp.get("open_joint", 0.0399)
                    )
                ),
                "duration_s": float(snapshots.get("open_gripper_duration_s", 1.0)),
                "result": "OK",
            }
        )

    pre_tcp = list(
        snapshots.get("descend_start_tcp")
        or snapshots.get("descend_pre_tcp")
        or snapshots.get("approach_tcp_goal")
        or geometric.get("pregrasp_tcp")
        or []
    )
    gr_tcp = list(
        snapshots.get("descend_goal_tcp")
        or snapshots.get("descend_end_tcp")
        or []
    )
    if not gr_tcp:
        built = resolve_golden_descend_goal_tcp(
            start_tcp=pre_tcp if pre_tcp else None,
            post_descend_tcp=None,
            geometric=geometric,
            snapshots=snapshots,
        )
        if built is not None:
            gr_tcp = list(built)
    if not gr_tcp:
        gr_tcp = list(geometric.get("grasp_tcp") or [])
    if pre_tcp and gr_tcp:
        descend = _phase("cartesian_descend_to_grasp")
        depth_m = float(
            snapshots.get("descend_depth_from_top_m")
            or geometric.get("depth_from_top_m", 0.033)
        )
        descend.update(
            {
                "type": "cartesian_or_joint_trajectory",
                "start_tcp": pre_tcp,
                "goal_tcp": gr_tcp,
                "depth_from_top_m": depth_m,
                "fraction": float(
                    snapshots.get(
                        "descend_fraction",
                        geometric.get("cartesian_descend_fraction", 1.0),
                    )
                ),
                "duration_s": float(snapshots.get("descend_duration_s", 3.0)),
                "result": str(snapshots.get("descend_result", "OK")),
            }
        )
        grasp_js = _normalize_arm_joint_list(snapshots.get("grasp_js"))
        if grasp_js is not None:
            descend["goal_js"] = grasp_js
            descend["end_js"] = grasp_js
        start_js = _normalize_arm_joint_list(snapshots.get("descend_start_js"))
        if start_js is not None:
            descend["start_js"] = start_js

    if snapshots.get("close_gripper_recorded"):
        _phase("close_gripper").update(
            {
                "type": "gripper_command",
                "command": "close",
                "close_joint": float(
                    snapshots.get(
                        "close_gripper_joint", grasp.get("close_joint", 0.027)
                    )
                ),
                "expected_width": float(
                    snapshots.get(
                        "close_expected_width_m",
                        grasp.get("expected_width_m", 0.06),
                    )
                ),
                "result": "OK",
            }
        )

    if snapshots.get("attach_recorded"):
        _phase("attach_and_verify").update(
            {
                "type": "attach",
                "expected_contact_policy": str(
                    snapshots.get("attach_contact_policy", "strict")
                ),
                "result": str(snapshots.get("attach_result", "OK")),
            }
        )

    lift_pre = list(
        snapshots.get("lift_start_tcp")
        or snapshots.get("descend_goal_tcp")
        or snapshots.get("descend_end_tcp")
        or pre_tcp
        or []
    )
    lift_goal = list(
        snapshots.get("lift_goal_tcp") or geometric.get("lift_tcp") or []
    )
    if lift_pre and lift_goal:
        lift_phase = _phase("cartesian_lift")
        lift_phase.update(
            {
                "type": "trajectory",
                "start_tcp": lift_pre,
                "goal_tcp": lift_goal,
                "duration_s": float(snapshots.get("lift_duration_s", 2.5)),
                "result": str(snapshots.get("lift_result", "OK")),
            }
        )
        post_lift_js = _normalize_arm_joint_list(snapshots.get("post_lift_js"))
        if post_lift_js is not None:
            lift_phase["goal_js"] = post_lift_js
            lift_phase["end_js"] = post_lift_js

    local_exit = snapshots.get("selected_local_exit") or transport.get(
        "selected_transport_entry"
    )
    if local_exit or snapshots.get("post_lift_escape_recorded"):
        escape = _phase("post_lift_local_escape")
        escape.update(
            {
                "type": "trajectory",
                "selected_mode": str(
                    snapshots.get("post_lift_escape_mode") or local_exit or ""
                ),
                "selected_local_exit": str(local_exit or ""),
                "duration_s": float(snapshots.get("local_escape_duration_s", 4.0)),
                "result": str(snapshots.get("post_lift_escape_result", "OK")),
            }
        )
        if snapshots.get("post_lift_escape_start_tcp"):
            escape["start_tcp"] = list(snapshots["post_lift_escape_start_tcp"])
        if snapshots.get("post_lift_escape_goal_tcp"):
            escape["goal_tcp"] = list(snapshots["post_lift_escape_goal_tcp"])
        escape_js = _normalize_arm_joint_list(snapshots.get("post_lift_escape_js"))
        if escape_js is not None:
            escape["goal_js"] = escape_js

    entry_wp = snapshots.get("transport_entry_target_waypoint") or transport.get(
        "first_hub"
    )
    if entry_wp or snapshots.get("transport_entry_recorded"):
        entry = _phase("transport_entry_to_safe_hub")
        entry.update(
            {
                "type": "direct_action_or_joint_trajectory",
                "selected_local_exit": str(local_exit or ""),
                "duration_s": float(snapshots.get("transport_entry_duration_s", 6.0)),
                "result": str(snapshots.get("transport_entry_result", "OK")),
            }
        )
        if entry_wp:
            entry["target_waypoint"] = str(entry_wp)
        entry_js = _normalize_arm_joint_list(snapshots.get("transport_entry_js"))
        if entry_js is not None:
            entry["goal_js"] = entry_js

    route = [
        str(x)
        for x in (
            snapshots.get("transport_sequence")
            or transport.get("route")
            or []
        )
    ]
    if route:
        transport_js: List[Dict[str, Any]] = []
        if waypoints_data:
            for wp in route:
                js = get_waypoint_joint_positions(waypoints_data, str(wp))
                if js is not None:
                    transport_js.append(
                        {"waypoint": str(wp), "joints": [float(v) for v in js]}
                    )
        _phase("deterministic_transport").update(
            {
                "type": "waypoint_sequence",
                "backend": str(transport.get("backend", "direct_action")),
                "sequence": route,
                "per_segment_times_s": list(
                    per_segment_times_s
                    or snapshots.get("per_segment_times_s")
                    or []
                ),
                "joint_waypoints": transport_js,
                "duration_s": float(snapshots.get("transport_duration_s", 34.0)),
            }
        )

    deposit = list(place_policy.get("deposit_xy") or [-0.37, 0.08])
    approach_z = float(
        place_policy.get("approach_tcp_z", snapshots.get("place_approach_tcp_z", 0.65))
    )
    if snapshots.get("place_approach_recorded") or approach_z:
        _phase("place_approach").update(
            {
                "type": "cartesian_or_joint_trajectory",
                "tcp_goal": [float(deposit[0]), float(deposit[1]), approach_z],
                "duration_s": float(snapshots.get("place_approach_duration_s", 4.0)),
                "result": "OK",
            }
        )

    rel_z = (
        place_policy.get("release_tcp_z")
        or snapshots.get("selected_release_tcp_z")
    )
    if rel_z is not None:
        _phase("place_release").update(
            {
                "type": "cartesian_or_joint_trajectory",
                "tcp_goal": [float(deposit[0]), float(deposit[1]), float(rel_z)],
                "release_tcp_z": float(rel_z),
                "duration_s": float(snapshots.get("place_release_duration_s", 3.0)),
                "result": str(snapshots.get("place_release_result", "OK")),
            }
        )
        release_js = _normalize_arm_joint_list(snapshots.get("place_release_js"))
        if release_js is not None:
            _phase("place_release")["goal_js"] = release_js

    if snapshots.get("open_detach_recorded"):
        _phase("open_detach").update(
            {
                "type": "gripper_and_detach",
                "command": "open",
                "result": str(snapshots.get("open_detach_result", "OK")),
            }
        )

    retreat_z = float(
        place_policy.get("retreat_tcp_z", snapshots.get("place_retreat_tcp_z", 0.65))
    )
    if snapshots.get("place_retreat_recorded") or retreat_z:
        _phase("place_retreat").update(
            {
                "type": "cartesian_or_joint_trajectory",
                "tcp_goal": [float(deposit[0]), float(deposit[1]), retreat_z],
                "duration_s": float(snapshots.get("place_retreat_duration_s", 3.0)),
                "result": "OK",
            }
        )

    ret_js = _normalize_arm_joint_list(snapshots.get("return_home_js"))
    if ret_js is not None:
        _phase("return_home").update(
            {
                "type": "joint_trajectory",
                "goal_waypoint": "home",
                "goal_js": ret_js,
                "duration_s": float(snapshots.get("return_home_duration_s", 4.0)),
                "result": "OK",
            }
        )

    phases[:] = order_golden_execution_phases(phases)


def _normalize_arm_joint_list(values: Any) -> Optional[List[float]]:
    if not isinstance(values, (list, tuple)) or len(values) < 7:
        return None
    try:
        return [float(values[i]) for i in range(7)]
    except (TypeError, ValueError):
        return None


def phase_has_executable_joint_goal(phase: Dict[str, Any]) -> bool:
    if not isinstance(phase, dict):
        return False
    if phase.get("points"):
        return True
    if phase.get("goal_waypoint") or phase.get("target_waypoint"):
        return True
    return _normalize_arm_joint_list(phase.get("goal_js")) is not None


def find_golden_phase(
    phases: Sequence[Dict[str, Any]],
    name: str,
) -> Optional[Dict[str, Any]]:
    target = str(name or "").strip().lower()
    for phase in phases:
        if str(phase.get("name", "")).strip().lower() == target:
            return phase
    return None


def golden_execution_approach_phase_executable(
    phases: Sequence[Dict[str, Any]],
) -> bool:
    phase = find_golden_phase(phases, "approach_to_pregrasp")
    return phase is not None and phase_has_executable_joint_goal(phase)


def patch_approach_to_pregrasp_from_snapshots(
    phases: List[Dict[str, Any]],
    snapshots: Dict[str, Any],
    geometric: Optional[Dict[str, Any]] = None,
) -> bool:
    goal_js = _normalize_arm_joint_list(
        snapshots.get("aligned_pregrasp_js")
        or snapshots.get("pregrasp_js")
        or snapshots.get("approach_end_js")
    )
    if goal_js is None:
        return False
    start_js = _normalize_arm_joint_list(
        snapshots.get("approach_start_js") or snapshots.get("pick_workspace_ready_js")
    )
    geom = geometric or {}
    pre_tcp = snapshots.get("approach_tcp_goal") or geom.get("pregrasp_tcp") or []
    updated: Dict[str, Any] = {
        "name": "approach_to_pregrasp",
        "type": "joint_trajectory",
        "joint_names": list(PANDA_ARM_JOINT_NAMES),
        "tcp_goal": list(pre_tcp) if pre_tcp else None,
        "goal_js": goal_js,
        "end_js": goal_js,
        "duration_s": float(snapshots.get("approach_duration_s", 4.0)),
    }
    if start_js is not None:
        updated["start_js"] = start_js
    phase = find_golden_phase(phases, "approach_to_pregrasp")
    if phase is not None:
        phase.clear()
        phase.update(updated)
    else:
        phases.append(updated)
    return True


def format_golden_execution_phase_capture_log(fields: Dict[str, Any]) -> str:
    lines = [
        "[GOLDEN_EXECUTION_PHASE_CAPTURE]",
        "phase_name=%s" % fields.get("phase_name", ""),
    ]
    skip = {"phase_name"}
    for key in (
        "has_goal_js",
        "goal_js_len",
        "target_waypoint",
        "selected_local_exit",
        "start_tcp",
        "goal_tcp",
        "result",
        "reason",
    ):
        if key in fields and fields.get(key) not in (None, ""):
            lines.append("%s=%s" % (key, fields.get(key)))
            skip.add(key)
    for key, val in sorted(fields.items()):
        if key in skip or val in (None, ""):
            continue
        lines.append("%s=%s" % (key, val))
    if "result" not in fields:
        lines.append("result=FAIL")
    return "\n".join(lines)


def _phase_has_min_executable_data(name: str, phase: Optional[Dict[str, Any]]) -> bool:
    if phase is None:
        return False
    n = str(name).strip().lower()
    if n == "home_to_pick_workspace_ready":
        return bool(
            phase.get("goal_waypoint") or phase.get("goal_js") or phase.get("points")
        )
    if n == "approach_to_pregrasp":
        return _normalize_arm_joint_list(phase.get("goal_js")) is not None
    if n == "gripper_axis_alignment":
        return phase.get("joint7_target") is not None
    if n == "open_gripper_at_pregrasp":
        return str(phase.get("command", "")).lower() == "open"
    if n == "cartesian_descend_to_grasp":
        return bool(phase.get("start_tcp") and phase.get("goal_tcp"))
    if n == "close_gripper":
        return str(phase.get("command", "")).lower() == "close" or phase.get(
            "close_joint"
        ) is not None
    if n == "attach_and_verify":
        return bool(phase.get("expected_contact_policy") or phase.get("result"))
    if n == "cartesian_lift":
        return bool(phase.get("start_tcp") and phase.get("goal_tcp"))
    if n == "post_lift_local_escape":
        return bool(phase.get("selected_mode") or phase.get("selected_local_exit"))
    if n == "transport_entry_to_safe_hub":
        return bool(
            phase.get("target_waypoint")
            or _normalize_arm_joint_list(phase.get("goal_js"))
        )
    if n == "deterministic_transport":
        seq = phase.get("sequence") or []
        return isinstance(seq, (list, tuple)) and len(seq) > 0
    if n == "place_approach":
        return bool(phase.get("tcp_goal"))
    if n == "place_release":
        return bool(phase.get("tcp_goal") and phase.get("release_tcp_z") is not None)
    if n == "open_detach":
        return bool(phase.get("result") or phase.get("type") == "gripper_and_detach")
    if n == "place_retreat":
        return bool(phase.get("tcp_goal"))
    if n == "return_home":
        return bool(
            phase.get("goal_waypoint") or _normalize_arm_joint_list(phase.get("goal_js"))
        )
    return False


def _tcp_triplet(values: Any) -> Optional[List[float]]:
    if not isinstance(values, (list, tuple)) or len(values) < 3:
        return None
    try:
        return [float(values[0]), float(values[1]), float(values[2])]
    except (TypeError, ValueError):
        return None


def _tcp_delta_z(
    start_tcp: Optional[Sequence[float]],
    goal_tcp: Optional[Sequence[float]],
) -> Optional[float]:
    if start_tcp is None or goal_tcp is None:
        return None
    if len(start_tcp) < 3 or len(goal_tcp) < 3:
        return None
    return abs(float(start_tcp[2]) - float(goal_tcp[2]))


def resolve_golden_descend_goal_tcp(
    *,
    start_tcp: Optional[Sequence[float]],
    post_descend_tcp: Optional[Sequence[float]],
    geometric: Optional[Dict[str, Any]] = None,
    snapshots: Optional[Dict[str, Any]] = None,
    candidate: Optional[Dict[str, Any]] = None,
    dx: float = 0.0,
    dy: float = 0.0,
) -> Optional[List[float]]:
    """Construye goal_tcp de grasp; nunca reutiliza pregrasp/object_safe_above."""
    geometric = geometric or {}
    snapshots = snapshots or {}
    candidate = candidate or {}

    post = _tcp_triplet(post_descend_tcp)
    if post is not None:
        start = _tcp_triplet(start_tcp)
        if start is None or _tcp_delta_z(start, post) >= MIN_GOLDEN_DESCEND_DELTA_Z_M:
            return post

    for key in ("_grasp_tcp_planning", "grasp_tcp"):
        if key.startswith("_"):
            raw = candidate.get(key)
        else:
            raw = geometric.get(key) or (snapshots.get("grasp_tcp"))
        grasp = _tcp_triplet(raw)
        if grasp is None:
            continue
        gx = float(grasp[0]) + float(dx)
        gy = float(grasp[1]) + float(dy)
        gz = float(grasp[2])
        start = _tcp_triplet(start_tcp)
        if start is not None:
            if float(start[2]) - gz >= MIN_GOLDEN_DESCEND_DELTA_Z_M:
                return [gx, gy, gz]
        elif gz < 0.55:
            return [gx, gy, gz]

    top_z = snapshots.get("top_z_m")
    if top_z is None:
        top_z = candidate.get("top_z_m") or candidate.get("top_z_estimated")
    depth = (
        snapshots.get("descend_depth_from_top_m")
        or candidate.get("recommended_grasp_depth_from_top_m")
        or candidate.get("depth_from_top_m")
        or geometric.get("depth_from_top_m")
        or 0.033
    )
    center = (
        candidate.get("chosen_target_center_base")
        or candidate.get("grasp_center_base")
        or candidate.get("position")
    )
    if top_z is not None and isinstance(center, (list, tuple)) and len(center) >= 2:
        gz = float(top_z) - float(depth)
        start = _tcp_triplet(start_tcp)
        if start is None or float(start[2]) - gz >= MIN_GOLDEN_DESCEND_DELTA_Z_M:
            return [
                float(center[0]) + float(dx),
                float(center[1]) + float(dy),
                gz,
            ]
    return None


def validate_golden_descend_record(
    phase: Optional[Dict[str, Any]],
    *,
    pregrasp_tcp_z: Optional[float] = None,
    min_delta_z: float = MIN_GOLDEN_DESCEND_DELTA_Z_M,
) -> Dict[str, Any]:
    start = _tcp_triplet(phase.get("start_tcp") if phase else None)
    goal = _tcp_triplet(phase.get("goal_tcp") if phase else None)
    delta = _tcp_delta_z(start, goal)
    top_z = phase.get("top_z") if phase else None
    depth = phase.get("depth_from_top_m") if phase else None
    if depth is None and phase:
        depth = phase.get("depth_from_top_m")
    ok = bool(start and goal and delta is not None)
    reason = "OK"
    if not start or not goal:
        ok = False
        reason = "missing_start_or_goal_tcp"
    elif delta is not None and delta < float(min_delta_z):
        ok = False
        reason = "descend_delta_z_too_small"
    elif pregrasp_tcp_z is not None and goal is not None:
        if float(goal[2]) >= float(pregrasp_tcp_z) - PREGRASP_GOAL_Z_MARGIN_M:
            ok = False
            reason = "goal_tcp_z_too_close_to_pregrasp"
    return {
        "start_tcp_z": start[2] if start else None,
        "goal_tcp_z": goal[2] if goal else None,
        "delta_z": delta,
        "expected_min_delta_z": float(min_delta_z),
        "top_z": top_z,
        "depth_from_top_m": depth,
        "result": "OK" if ok else "FAIL",
        "reason": reason,
    }


def validate_golden_lift_record(
    phase: Optional[Dict[str, Any]],
    *,
    min_delta_z: float = MIN_GOLDEN_LIFT_DELTA_Z_M,
) -> Dict[str, Any]:
    start = _tcp_triplet(phase.get("start_tcp") if phase else None)
    goal = _tcp_triplet(phase.get("goal_tcp") if phase else None)
    delta = _tcp_delta_z(start, goal)
    ok = bool(start and goal and delta is not None and delta >= float(min_delta_z))
    return {
        "start_tcp_z": start[2] if start else None,
        "goal_tcp_z": goal[2] if goal else None,
        "delta_z": delta,
        "expected_min_delta_z": float(min_delta_z),
        "result": "OK" if ok else "FAIL",
    }


def format_golden_record_descend_validate_log(fields: Dict[str, Any]) -> str:
    return (
        "[GOLDEN_RECORD_DESCEND_VALIDATE]\n"
        "start_tcp_z=%s\n"
        "goal_tcp_z=%s\n"
        "delta_z=%s\n"
        "expected_min_delta_z=%s\n"
        "top_z=%s\n"
        "depth_from_top_m=%s\n"
        "result=%s\n"
        "reason=%s"
        % (
            fields.get("start_tcp_z", "n/a"),
            fields.get("goal_tcp_z", "n/a"),
            fields.get("delta_z", "n/a"),
            fields.get("expected_min_delta_z", MIN_GOLDEN_DESCEND_DELTA_Z_M),
            fields.get("top_z", "n/a"),
            fields.get("depth_from_top_m", "n/a"),
            fields.get("result", "FAIL"),
            fields.get("reason", ""),
        )
    )


def validate_golden_execution_save(
    phases: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    raw_phases = [dict(p) for p in phases if isinstance(p, dict)]
    ordered = order_golden_execution_phases(raw_phases)
    missing = [
        name
        for name in PHASE_NAMES
        if not _phase_has_min_executable_data(
            name, find_golden_phase(ordered, name)
        )
    ]
    approach = find_golden_phase(ordered, "approach_to_pregrasp")
    approach_js = _normalize_arm_joint_list(
        approach.get("goal_js") if approach else None
    )
    descend_phase = find_golden_phase(ordered, "cartesian_descend_to_grasp")
    lift_phase = find_golden_phase(ordered, "cartesian_lift")
    place_release = find_golden_phase(ordered, "place_release")
    pregrasp_z = None
    if approach is not None:
        tcp = _tcp_triplet(approach.get("tcp_goal"))
        if tcp is not None:
            pregrasp_z = float(tcp[2])
    descend_val = validate_golden_descend_record(
        descend_phase, pregrasp_tcp_z=pregrasp_z
    )
    lift_val = validate_golden_lift_record(lift_phase)
    place_release_z = (
        place_release.get("release_tcp_z") if place_release else None
    )
    min_ok = len(missing) == 0
    required_present = (
        len(raw_phases) >= len(PHASE_NAMES)
        and len(ordered) == len(PHASE_NAMES)
        and min_ok
    )
    semantics_ok = (
        str(descend_val.get("result", "")).upper() == "OK"
        and str(lift_val.get("result", "")).upper() == "OK"
        and place_release_z is not None
    )
    result_ok = (
        required_present
        and len(approach_js or []) == 7
        and semantics_ok
    )
    return {
        "phase_count_real": len(ordered),
        "required_phases_present": required_present and min_ok,
        "missing_phases": missing,
        "approach_goal_js_len": len(approach_js or []),
        "post_lift_local_escape_present": find_golden_phase(
            ordered, "post_lift_local_escape"
        )
        is not None,
        "transport_entry_present": find_golden_phase(
            ordered, "transport_entry_to_safe_hub"
        )
        is not None,
        "deterministic_transport_present": find_golden_phase(
            ordered, "deterministic_transport"
        )
        is not None,
        "place_release_present": find_golden_phase(ordered, "place_release") is not None,
        "return_home_present": find_golden_phase(ordered, "return_home") is not None,
        "descend_delta_z": descend_val.get("delta_z"),
        "descend_semantics_ok": str(descend_val.get("result", "")).upper() == "OK",
        "lift_delta_z": lift_val.get("delta_z"),
        "lift_semantics_ok": str(lift_val.get("result", "")).upper() == "OK",
        "place_release_tcp_z": place_release_z,
        "descend_validate": descend_val,
        "lift_validate": lift_val,
        "result": "OK" if result_ok else "FAIL",
    }


def format_golden_execution_save_validate_log(fields: Dict[str, Any]) -> str:
    missing = fields.get("missing_phases") or []
    missing_s = ",".join(str(x) for x in missing) if missing else ""
    return (
        "[GOLDEN_EXECUTION_SAVE_VALIDATE]\n"
        "phase_count_real=%s\n"
        "required_phases_present=%s\n"
        "missing_phases=[%s]\n"
        "approach_goal_js_len=%s\n"
        "post_lift_local_escape_present=%s\n"
        "transport_entry_present=%s\n"
        "deterministic_transport_present=%s\n"
        "place_release_present=%s\n"
        "return_home_present=%s\n"
        "descend_delta_z=%s\n"
        "descend_semantics_ok=%s\n"
        "lift_delta_z=%s\n"
        "lift_semantics_ok=%s\n"
        "place_release_tcp_z=%s\n"
        "result=%s"
        % (
            fields.get("phase_count_real", 0),
            str(bool(fields.get("required_phases_present"))).lower(),
            missing_s,
            fields.get("approach_goal_js_len", 0),
            str(bool(fields.get("post_lift_local_escape_present"))).lower(),
            str(bool(fields.get("transport_entry_present"))).lower(),
            str(bool(fields.get("deterministic_transport_present"))).lower(),
            str(bool(fields.get("place_release_present"))).lower(),
            str(bool(fields.get("return_home_present"))).lower(),
            fields.get("descend_delta_z", "n/a"),
            str(bool(fields.get("descend_semantics_ok"))).lower(),
            fields.get("lift_delta_z", "n/a"),
            str(bool(fields.get("lift_semantics_ok"))).lower(),
            fields.get("place_release_tcp_z", "n/a"),
            fields.get("result", "FAIL"),
        )
    )


# --- Log formatters ---


def format_golden_execution_mode_lock_log(fields: Dict[str, Any]) -> str:
    return (
        "[GOLDEN_EXECUTION_MODE_LOCK]\n"
        "scene_id=%s\n"
        "target_label=%s\n"
        "slot_index=%s\n"
        "use_golden_execution_candidate=%s\n"
        "require_golden_execution_candidate=%s\n"
        "schema_version=%s\n"
        "status=%s\n"
        "grid_search_disabled=%s\n"
        "v1_golden_disabled=%s\n"
        "accept_first_valid_disabled=%s\n"
        "result=%s"
        % (
            fields.get("scene_id", ""),
            fields.get("target_label", ""),
            fields.get("slot_index", ""),
            fields.get("use_golden_execution_candidate", "false"),
            fields.get("require_golden_execution_candidate", "false"),
            fields.get("schema_version", ""),
            fields.get("status", ""),
            fields.get("grid_search_disabled", "true"),
            fields.get("v1_golden_disabled", "true"),
            fields.get("accept_first_valid_disabled", "true"),
            fields.get("result", "FAIL"),
        )
    )


def format_golden_execution_precheck_log(fields: Dict[str, Any]) -> str:
    return (
        "[GOLDEN_EXECUTION_PRECHECK]\n"
        "scene_id=%s\n"
        "target_label=%s\n"
        "slot_index=%s\n"
        "candidate_path=%s\n"
        "schema_version=%s\n"
        "status=%s\n"
        "scene_signature_ok=%s\n"
        "robot_start_ok=%s\n"
        "gripper_commandable_ok=%s\n"
        "gripper_no_attached_object_ok=%s\n"
        "gripper_initial_open_ok=%s\n"
        "golden_has_pregrasp_open_phase_ok=%s\n"
        "gripper_ok=%s\n"
        "target_ok=%s\n"
        "obstacles_ok=%s\n"
        "slot_ok=%s\n"
        "phases_ok=%s\n"
        "result=%s\n"
        "reason=%s"
        % (
            fields.get("scene_id", ""),
            fields.get("target_label", ""),
            fields.get("slot_index", ""),
            fields.get("candidate_path", ""),
            fields.get("schema_version", ""),
            fields.get("status", ""),
            fields.get("scene_signature_ok", "n/a"),
            fields.get("robot_start_ok", "n/a"),
            fields.get("gripper_commandable_ok", "n/a"),
            fields.get("gripper_no_attached_object_ok", "n/a"),
            fields.get("gripper_initial_open_ok", "n/a"),
            fields.get("golden_has_pregrasp_open_phase_ok", "n/a"),
            fields.get("gripper_ok", "n/a"),
            fields.get("target_ok", fields.get("target_collision_ok", "n/a")),
            fields.get("obstacles_ok", "n/a"),
            fields.get("slot_ok", fields.get("place_slot_ok", "n/a")),
            fields.get("phases_ok", "n/a"),
            fields.get("result", "FAIL"),
            fields.get("reason", ""),
        )
    )


def format_golden_execution_flow_decision_log(fields: Dict[str, Any]) -> str:
    return (
        "[GOLDEN_EXECUTION_FLOW_DECISION]\n"
        "scene_id=%s\n"
        "target_label=%s\n"
        "slot_index=%s\n"
        "use_golden_execution_candidate=%s\n"
        "require_golden_execution_candidate=%s\n"
        "candidate_benchmark_mode=%s\n"
        "scope_active=%s\n"
        "decision=%s\n"
        "normal_pipeline_allowed=%s\n"
        "reason=%s"
        % (
            fields.get("scene_id", ""),
            fields.get("target_label", ""),
            fields.get("slot_index", ""),
            fields.get("use_golden_execution_candidate", "false"),
            fields.get("require_golden_execution_candidate", "false"),
            fields.get("candidate_benchmark_mode", "false"),
            fields.get("scope_active", "false"),
            fields.get("decision", ""),
            fields.get("normal_pipeline_allowed", "true"),
            fields.get("reason", ""),
        )
    )


def format_golden_execution_contract_violation_log(fields: Dict[str, Any]) -> str:
    return (
        "[GOLDEN_EXECUTION_CONTRACT_VIOLATION]\n"
        "stage=%s\n"
        "reason=%s\n"
        "action=ABORT_NO_FALLBACK"
        % (
            fields.get("stage", ""),
            fields.get(
                "reason", "normal_pipeline_reached_while_golden_required"
            ),
        )
    )


def format_golden_execution_grid_guard_log(fields: Dict[str, Any]) -> str:
    return (
        "[GOLDEN_EXECUTION_GRID_GUARD]\n"
        "use_golden_execution_candidate=%s\n"
        "require_golden_execution_candidate=%s\n"
        "candidate_benchmark_mode=%s\n"
        "scope_active=%s\n"
        "grid_allowed=%s\n"
        "result=%s"
        % (
            fields.get("use_golden_execution_candidate", "false"),
            fields.get("require_golden_execution_candidate", "false"),
            fields.get("candidate_benchmark_mode", "false"),
            fields.get("scope_active", "false"),
            fields.get("grid_allowed", "true"),
            fields.get("result", "OK"),
        )
    )


def format_golden_execution_reject_detail_log(fields: Dict[str, Any]) -> str:
    return (
        "[GOLDEN_EXECUTION_REJECT_DETAIL]\n"
        "stage=%s\n"
        "reason=%s\n"
        "expected=%s\n"
        "observed=%s\n"
        "action=ABORT_NO_FALLBACK"
        % (
            fields.get("stage", ""),
            fields.get("reason", ""),
            fields.get("expected", ""),
            fields.get("observed", ""),
        )
    )


def format_golden_execution_fallback_log(fields: Dict[str, Any]) -> str:
    return (
        "[GOLDEN_EXECUTION_FALLBACK_TO_GRID]\n"
        "reason=%s"
        % (fields.get("reason", ""),)
    )


def format_golden_execution_phase_start_log(fields: Dict[str, Any]) -> str:
    return (
        "[GOLDEN_EXECUTION_PHASE_START]\n"
        "phase_idx=%s\n"
        "phase_name=%s\n"
        "phase_type=%s"
        % (
            fields.get("phase_idx", ""),
            fields.get("phase_name", ""),
            fields.get("phase_type", fields.get("type", "")),
        )
    )


def format_golden_execution_phase_done_log(fields: Dict[str, Any]) -> str:
    lines = [
        "[GOLDEN_EXECUTION_PHASE_DONE]",
        "phase_idx=%s" % fields.get("phase_idx", ""),
        "phase_name=%s" % fields.get("phase_name", ""),
        "real_duration_s=%.3f" % float(fields.get("duration_real_s", 0.0)),
        "expected_duration_s=%.3f" % float(fields.get("expected_duration_s", 0.0)),
        "result=%s" % fields.get("result", "FAIL"),
    ]
    if fields.get("reason") not in (None, ""):
        lines.append("reason=%s" % fields.get("reason"))
    if fields.get("motion_executed") not in (None, ""):
        lines.append("motion_executed=%s" % fields.get("motion_executed"))
    if fields.get("object_attached") not in (None, ""):
        lines.append("object_attached=%s" % fields.get("object_attached"))
    return "\n".join(lines)


def format_golden_execution_done_log(fields: Dict[str, Any]) -> str:
    return (
        "[GOLDEN_EXECUTION_DONE]\n"
        "scene_id=%s\n"
        "target_label=%s\n"
        "slot_index=%s\n"
        "total_expected_time_s=%.3f\n"
        "total_real_time_s=%.3f\n"
        "result=%s"
        % (
            fields.get("scene_id", ""),
            fields.get("target_label", ""),
            fields.get("slot_index", ""),
            float(fields.get("total_expected_time_s", 0.0)),
            float(fields.get("total_real_time_s", 0.0)),
            fields.get("result", "FAIL"),
        )
    )


def format_golden_execution_abort_log(fields: Dict[str, Any]) -> str:
    return (
        "[GOLDEN_EXECUTION_ABORT]\n"
        "phase_name=%s\n"
        "reason=%s\n"
        "object_attached=%s\n"
        "safe_state_action=%s"
        % (
            fields.get("phase_name", ""),
            fields.get("reason", ""),
            fields.get("object_attached", "unknown"),
            fields.get("safe_state_action", ""),
        )
    )
