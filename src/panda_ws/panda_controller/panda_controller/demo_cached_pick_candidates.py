"""Candidatos pick prevalidados/cacheados para escenas demo (demo_scene_02)."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import yaml

from panda_controller.demo_scene_policy import (
    default_demo_scenes_dir,
    obstacle_disturbance_thresholds,
)

DEFAULT_SCENE_OBJECT_XY_TOL_M = 0.015
DEFAULT_SCENE_OBJECT_YAW_TOL_RAD = 0.08
DEFAULT_SCENE_TOP_Z_TOL_M = 0.025
DEFAULT_CACHED_JOINT7_TOL_RAD = 0.05

_CACHE_VERSION = 1


def default_cached_candidates_path(
    scene_id: str,
    *,
    scenes_dir: Optional[str] = None,
) -> str:
    sid = str(scene_id or "").strip().lower()
    base = str(scenes_dir or default_demo_scenes_dir())
    return os.path.join(base, f"{sid}_cached_candidates.yaml")


def _normalize_xyz(raw: Any) -> Optional[Tuple[float, float, float]]:
    if not isinstance(raw, (list, tuple)) or len(raw) < 3:
        return None
    return (float(raw[0]), float(raw[1]), float(raw[2]))


def _normalize_js(raw: Any) -> Optional[List[float]]:
    if not isinstance(raw, (list, tuple)):
        return None
    if len(raw) < 7:
        return None
    return [float(v) for v in raw[:7]]


def _normalize_xy(raw: Any) -> Optional[Tuple[float, float]]:
    if not isinstance(raw, (list, tuple)) or len(raw) < 2:
        return None
    return (float(raw[0]), float(raw[1]))


def _wrap_to_pi(angle: float) -> float:
    a = float(angle)
    while a > math.pi:
        a -= 2.0 * math.pi
    while a < -math.pi:
        a += 2.0 * math.pi
    return a


def _yaw_delta_rad(a: float, b: float) -> float:
    return abs(_wrap_to_pi(float(a) - float(b)))


def load_demo_cached_candidates(
    yaml_path: str,
) -> Optional[Dict[str, Any]]:
    path = str(yaml_path or "").strip()
    if not path or not os.path.isfile(path):
        return None
    ext = Path(path).suffix.lower()
    with open(path, "r", encoding="utf-8") as handle:
        if ext == ".json":
            raw = json.load(handle) or {}
        else:
            raw = yaml.safe_load(handle) or {}
    return normalize_demo_cached_candidates(raw, source_file=path)


def normalize_demo_cached_candidates(
    raw: Dict[str, Any],
    *,
    source_file: str = "",
) -> Dict[str, Any]:
    sid = str(raw.get("scene_id", "")).strip().lower()
    tol_raw = raw.get("tolerances") or {}
    tolerances = {
        "object_xy_m": float(tol_raw.get("object_xy_m", DEFAULT_SCENE_OBJECT_XY_TOL_M)),
        "object_yaw_rad": float(
            tol_raw.get("object_yaw_rad", DEFAULT_SCENE_OBJECT_YAW_TOL_RAD)
        ),
        "top_z_m": float(tol_raw.get("top_z_m", DEFAULT_SCENE_TOP_Z_TOL_M)),
        "joint7_rad": float(tol_raw.get("joint7_rad", DEFAULT_CACHED_JOINT7_TOL_RAD)),
    }
    candidates_raw = raw.get("candidates") or {}
    candidates: Dict[str, Dict[str, Any]] = {}
    if isinstance(candidates_raw, dict):
        for label, spec in candidates_raw.items():
            lb = str(label).strip().lower()
            if not lb or not isinstance(spec, dict):
                continue
            norm = normalize_cached_pick_candidate_entry(spec, label=lb)
            if norm is not None:
                candidates[lb] = norm
    return {
        "scene_id": sid,
        "version": int(raw.get("version", _CACHE_VERSION)),
        "source_file": str(source_file),
        "tolerances": tolerances,
        "candidates": candidates,
    }


def normalize_cached_pick_candidate_entry(
    spec: Dict[str, Any],
    *,
    label: str,
) -> Optional[Dict[str, Any]]:
    pose_raw = spec.get("object_pose") or spec.get("expected_object_pose") or {}
    if not isinstance(pose_raw, dict):
        pose_raw = {}
    pregrasp = _normalize_xyz(spec.get("pregrasp_tcp"))
    grasp = _normalize_xyz(spec.get("grasp_tcp"))
    if pregrasp is None or grasp is None:
        return None
    raw_js = _normalize_js(spec.get("raw_pregrasp_js"))
    aligned_js = _normalize_js(spec.get("aligned_pregrasp_js"))
    gap_xy = _normalize_xy(spec.get("desired_gap_axis_xy"))
    obstacles = spec.get("expected_obstacle_labels") or spec.get("obstacle_labels") or []
    obs_labels: Set[str] = set()
    if isinstance(obstacles, (list, tuple, set)):
        obs_labels = {str(x).strip().lower() for x in obstacles if str(x).strip()}
    return {
        "label": str(label).strip().lower(),
        "object_pose": {
            "x": float(pose_raw.get("x", pregrasp[0])),
            "y": float(pose_raw.get("y", pregrasp[1])),
            "yaw": float(pose_raw.get("yaw", spec.get("commanded_tcp_yaw", 0.0))),
            "top_z": float(pose_raw.get("top_z", grasp[2] + float(spec.get("depth_from_top", 0.033)))),
        },
        "pregrasp_tcp": pregrasp,
        "grasp_tcp": grasp,
        "commanded_tcp_yaw": float(spec.get("commanded_tcp_yaw", 0.0)),
        "raw_pregrasp_js": raw_js,
        "aligned_pregrasp_js": aligned_js,
        "expected_joint7_after_alignment": spec.get("expected_joint7_after_alignment"),
        "desired_gap_axis_xy": gap_xy,
        "depth_from_top": float(spec.get("depth_from_top", 0.033)),
        "transport_entry": spec.get("transport_entry"),
        "place_slot": spec.get("place_slot"),
        "expected_obstacle_labels": sorted(obs_labels),
        "cartesian_fraction": spec.get("cartesian_fraction"),
        "validated_at": spec.get("validated_at"),
    }


def get_cached_candidate_for_label(
    cache: Dict[str, Any],
    label: str,
) -> Optional[Dict[str, Any]]:
    lb = str(label or "").strip().lower()
    if not lb:
        return None
    entry = (cache.get("candidates") or {}).get(lb)
    if not isinstance(entry, dict):
        return None
    return dict(entry)


def obstacle_labels_from_scene(
    scene_obstacles: Sequence[Dict[str, Any]],
    *,
    exclude_label: str,
) -> Set[str]:
    ex = str(exclude_label or "").strip().lower()
    out: Set[str] = set()
    for obs in scene_obstacles or []:
        if not isinstance(obs, dict):
            continue
        lb = str(obs.get("label") or obs.get("id") or "").strip().lower()
        if lb and lb != ex:
            out.add(lb)
    return out


def validate_cached_scene_match(
    entry: Dict[str, Any],
    *,
    scene_id: str,
    label: str,
    runtime_xy: Tuple[float, float],
    runtime_yaw: float,
    runtime_top_z: float,
    scene_obstacles: Sequence[Dict[str, Any]],
    demo_scene_policy: Optional[Dict[str, Any]],
    cache_tolerances: Optional[Dict[str, float]] = None,
) -> Tuple[bool, str, Dict[str, Any]]:
    sid = str(scene_id or "").strip().lower()
    lb = str(label or "").strip().lower()
    exp = entry.get("object_pose") or {}
    tol = dict(cache_tolerances or {})
    xy_thr = float(tol.get("object_xy_m", DEFAULT_SCENE_OBJECT_XY_TOL_M))
    yaw_thr = float(tol.get("object_yaw_rad", DEFAULT_SCENE_OBJECT_YAW_TOL_RAD))
    z_thr = float(tol.get("top_z_m", DEFAULT_SCENE_TOP_Z_TOL_M))
    if demo_scene_policy:
        pol_xy, pol_z = obstacle_disturbance_thresholds(demo_scene_policy)
        xy_thr = max(xy_thr, float(pol_xy))
        z_thr = max(z_thr, float(pol_z))

    dx = float(runtime_xy[0]) - float(exp.get("x", runtime_xy[0]))
    dy = float(runtime_xy[1]) - float(exp.get("y", runtime_xy[1]))
    dxy = math.hypot(dx, dy)
    dyaw = _yaw_delta_rad(float(runtime_yaw), float(exp.get("yaw", runtime_yaw)))
    dz = abs(float(runtime_top_z) - float(exp.get("top_z", runtime_top_z)))

    details: Dict[str, Any] = {
        "scene_id": sid,
        "label": lb,
        "delta_xy_m": float(dxy),
        "delta_yaw_rad": float(dyaw),
        "delta_top_z_m": float(dz),
        "xy_threshold_m": float(xy_thr),
        "yaw_threshold_rad": float(yaw_thr),
        "top_z_threshold_m": float(z_thr),
    }

    if dxy > xy_thr:
        return False, "object_pose_xy_mismatch", details
    if dyaw > yaw_thr:
        return False, "object_pose_yaw_mismatch", details
    if dz > z_thr:
        return False, "object_top_z_mismatch", details

    expected_obs = set(entry.get("expected_obstacle_labels") or [])
    runtime_obs = obstacle_labels_from_scene(scene_obstacles, exclude_label=lb)
    details["expected_obstacle_labels"] = sorted(expected_obs)
    details["runtime_obstacle_labels"] = sorted(runtime_obs)
    if expected_obs and expected_obs != runtime_obs:
        missing = sorted(expected_obs - runtime_obs)
        extra = sorted(runtime_obs - expected_obs)
        details["missing_obstacles"] = missing
        details["extra_obstacles"] = extra
        return False, "obstacle_set_mismatch", details

    if demo_scene_policy and expected_obs:
        objects = demo_scene_policy.get("objects") or {}
        obs_xy_thr, obs_z_thr = obstacle_disturbance_thresholds(demo_scene_policy)
        for obs_lb in expected_obs:
            gt = objects.get(obs_lb) or {}
            gt_pose = gt.get("pose") or {}
            if not gt_pose:
                continue
            runtime_pose = _runtime_obstacle_pose(scene_obstacles, obs_lb)
            if runtime_pose is None:
                continue
            odxy = math.hypot(
                float(runtime_pose[0]) - float(gt_pose.get("x", runtime_pose[0])),
                float(runtime_pose[1]) - float(gt_pose.get("y", runtime_pose[1])),
            )
            odz = abs(float(runtime_pose[2]) - float(gt_pose.get("z", runtime_pose[2])))
            if odxy > obs_xy_thr or odz > obs_z_thr:
                details["obstacle_pose_fail"] = obs_lb
                details["obstacle_delta_xy_m"] = float(odxy)
                details["obstacle_delta_z_m"] = float(odz)
                return False, "obstacle_pose_disturbed", details

    return True, "OK", details


def _runtime_obstacle_pose(
    scene_obstacles: Sequence[Dict[str, Any]],
    label: str,
) -> Optional[Tuple[float, float, float]]:
    lb = str(label or "").strip().lower()
    for obs in scene_obstacles or []:
        if not isinstance(obs, dict):
            continue
        olb = str(obs.get("label") or obs.get("id") or "").strip().lower()
        if olb != lb:
            continue
        pos = obs.get("position") or obs.get("pose") or obs.get("center")
        if isinstance(pos, dict):
            return (
                float(pos.get("x", 0.0)),
                float(pos.get("y", 0.0)),
                float(pos.get("z", 0.0)),
            )
        xyz = _normalize_xyz(pos)
        if xyz is not None:
            return xyz
    return None


def cached_entry_to_grid_spec(
    entry: Dict[str, Any],
    *,
    grid_idx: int = -1,
    gripper_physical_yaw_correction_rad: float = 0.0,
) -> Dict[str, Any]:
    pre = entry["pregrasp_tcp"]
    gr = entry["grasp_tcp"]
    cmd = float(entry["commanded_tcp_yaw"])
    phys = float(gripper_physical_yaw_correction_rad)
    yaw_rad = _wrap_to_pi(cmd - phys)
    depth = float(entry.get("depth_from_top") or (pre[2] - gr[2] if pre[2] > gr[2] else 0.033))
    return {
        "grid_idx": int(grid_idx),
        "yaw_name": "cached",
        "yaw_rad": float(yaw_rad),
        "yaw_deg": math.degrees(float(yaw_rad)),
        "pregrasp_tcp_z": float(pre[2]),
        "grasp_tcp_z": float(gr[2]),
        "depth_from_top_m": float(depth),
        "pre_plan": (float(pre[0]), float(pre[1]), float(pre[2])),
        "gr_plan": (float(gr[0]), float(gr[1]), float(gr[2])),
        "ik_seed_label": "pick_workspace_ready",
        "priority": "cached",
        "source": "demo_cached_candidate",
    }


def validate_cached_joint7_expectation(
    entry: Dict[str, Any],
    *,
    joint7_after_sim: Optional[float],
    joint7_tol_rad: float,
) -> Tuple[bool, str]:
    expected = entry.get("expected_joint7_after_alignment")
    if expected is None or joint7_after_sim is None:
        return True, "not_checked"
    delta = abs(float(joint7_after_sim) - float(expected))
    if delta > float(joint7_tol_rad):
        return False, "joint7_expectation_mismatch"
    return True, "OK"


def cached_pick_candidate_from_grasp_winner(
    *,
    scene_id: str,
    label: str,
    candidate: Dict[str, Any],
    grasp_winner: Dict[str, Any],
    scene_obstacles: Sequence[Dict[str, Any]],
    transport_score: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    lb = str(label or "").strip().lower()
    pre_plan = grasp_winner.get("pre_plan") or ()
    gr_plan = grasp_winner.get("gr_plan") or ()
    raw_js = grasp_winner.get("raw_pregrasp_js") or grasp_winner.get("pregrasp_js")
    aligned_js = grasp_winner.get("aligned_pregrasp_js") or grasp_winner.get("pregrasp_js")
    top_z = candidate.get("top_z_m")
    yaw = candidate.get("_base_commanded_tcp_yaw_rad")
    if yaw is None:
        yaw = grasp_winner.get("selected_commanded_yaw") or grasp_winner.get(
            "commanded_yaw_rad"
        )
    return {
        "object_pose": {
            "x": float(pre_plan[0]) if len(pre_plan) >= 1 else float(candidate.get("x", 0.0)),
            "y": float(pre_plan[1]) if len(pre_plan) >= 2 else float(candidate.get("y", 0.0)),
            "yaw": float(yaw or 0.0),
            "top_z": float(top_z) if top_z is not None else None,
        },
        "pregrasp_tcp": list(pre_plan) if pre_plan else None,
        "grasp_tcp": list(gr_plan) if gr_plan else None,
        "commanded_tcp_yaw": float(
            grasp_winner.get("selected_commanded_yaw")
            or grasp_winner.get("commanded_yaw_rad")
            or 0.0
        ),
        "raw_pregrasp_js": _joint_values_list(raw_js),
        "aligned_pregrasp_js": _joint_values_list(aligned_js),
        "expected_joint7_after_alignment": grasp_winner.get(
            "selected_joint7_expected_after_alignment"
        ),
        "desired_gap_axis_xy": grasp_winner.get("selected_desired_gap_axis_xy"),
        "depth_from_top": float(grasp_winner.get("depth_from_top_m") or 0.033)
        if grasp_winner.get("depth_from_top_m") is not None
        else (
            float(top_z) - float(gr_plan[2])
            if top_z is not None and len(gr_plan) >= 3
            else None
        ),
        "transport_entry": (
            (transport_score or {}).get("transport_entry_mode")
            or (transport_score or {}).get("local_exit_mode")
        ),
        "place_slot": candidate.get("preferred_slot"),
        "expected_obstacle_labels": sorted(
            obstacle_labels_from_scene(scene_obstacles, exclude_label=lb)
        ),
        "cartesian_fraction": grasp_winner.get("_cart_frac"),
        "label": lb,
        "scene_id": str(scene_id or "").strip().lower(),
    }


def _joint_values_list(js: Any) -> Optional[List[float]]:
    if js is None:
        return None
    if hasattr(js, "position"):
        pos = getattr(js, "position", None)
        if pos is not None:
            return [float(v) for v in list(pos)[:7]]
    if isinstance(js, (list, tuple)):
        return [float(v) for v in js[:7]]
    return None


def save_demo_cached_candidate(
    yaml_path: str,
    entry: Dict[str, Any],
    *,
    scene_id: str,
    label: str,
) -> None:
    sid = str(scene_id or "").strip().lower()
    lb = str(label or entry.get("label") or "").strip().lower()
    path = str(yaml_path or "").strip()
    if not path or not lb:
        return
    existing: Dict[str, Any] = {}
    if os.path.isfile(path):
        loaded = load_demo_cached_candidates(path)
        if loaded is not None:
            existing = {
                "scene_id": loaded.get("scene_id") or sid,
                "version": loaded.get("version", _CACHE_VERSION),
                "tolerances": loaded.get("tolerances") or {},
                "candidates": dict(loaded.get("candidates") or {}),
            }
    else:
        existing = {
            "scene_id": sid,
            "version": _CACHE_VERSION,
            "tolerances": {
                "object_xy_m": DEFAULT_SCENE_OBJECT_XY_TOL_M,
                "object_yaw_rad": DEFAULT_SCENE_OBJECT_YAW_TOL_RAD,
                "top_z_m": DEFAULT_SCENE_TOP_Z_TOL_M,
                "joint7_rad": DEFAULT_CACHED_JOINT7_TOL_RAD,
            },
            "candidates": {},
        }
    norm = normalize_cached_pick_candidate_entry(entry, label=lb)
    if norm is None:
        return
    serial = _serialize_candidate_for_yaml(norm)
    candidates = dict(existing.get("candidates") or {})
    candidates[lb] = serial
    existing["candidates"] = candidates
    existing["scene_id"] = sid or existing.get("scene_id")
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(existing, handle, sort_keys=False, allow_unicode=True)


def _serialize_candidate_for_yaml(entry: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "object_pose": dict(entry.get("object_pose") or {}),
        "pregrasp_tcp": [float(v) for v in entry["pregrasp_tcp"]],
        "grasp_tcp": [float(v) for v in entry["grasp_tcp"]],
        "commanded_tcp_yaw": float(entry["commanded_tcp_yaw"]),
        "depth_from_top": float(entry.get("depth_from_top", 0.033)),
        "expected_obstacle_labels": list(entry.get("expected_obstacle_labels") or []),
    }
    if entry.get("raw_pregrasp_js") is not None:
        out["raw_pregrasp_js"] = [float(v) for v in entry["raw_pregrasp_js"]]
    if entry.get("aligned_pregrasp_js") is not None:
        out["aligned_pregrasp_js"] = [float(v) for v in entry["aligned_pregrasp_js"]]
    if entry.get("expected_joint7_after_alignment") is not None:
        out["expected_joint7_after_alignment"] = float(
            entry["expected_joint7_after_alignment"]
        )
    if entry.get("desired_gap_axis_xy") is not None:
        out["desired_gap_axis_xy"] = [float(v) for v in entry["desired_gap_axis_xy"]]
    if entry.get("transport_entry") is not None:
        out["transport_entry"] = str(entry["transport_entry"])
    if entry.get("place_slot") is not None:
        out["place_slot"] = int(entry["place_slot"])
    if entry.get("cartesian_fraction") is not None:
        out["cartesian_fraction"] = float(entry["cartesian_fraction"])
    if entry.get("validated_at") is not None:
        out["validated_at"] = str(entry["validated_at"])
    return out


def format_demo_cached_candidate_load_log(fields: Dict[str, Any]) -> str:
    return (
        "[DEMO_CACHED_CANDIDATE_LOAD]\n"
        "scene_id=%s\n"
        "label=%s\n"
        "yaml_path=%s\n"
        "found=%s\n"
        "candidate_count=%d"
        % (
            fields.get("scene_id", "n/a"),
            fields.get("label", "n/a"),
            fields.get("yaml_path", "n/a"),
            str(bool(fields.get("found"))).lower(),
            int(fields.get("candidate_count", 0)),
        )
    )


def format_demo_cached_candidate_validate_log(fields: Dict[str, Any]) -> str:
    return (
        "[DEMO_CACHED_CANDIDATE_VALIDATE]\n"
        "scene_id=%s\n"
        "label=%s\n"
        "stage=%s\n"
        "result=%s\n"
        "reason=%s\n"
        "delta_xy_m=%s\n"
        "delta_yaw_rad=%s\n"
        "delta_top_z_m=%s"
        % (
            fields.get("scene_id", "n/a"),
            fields.get("label", "n/a"),
            fields.get("stage", "n/a"),
            fields.get("result", "n/a"),
            fields.get("reason", ""),
            fields.get("delta_xy_m", "n/a"),
            fields.get("delta_yaw_rad", "n/a"),
            fields.get("delta_top_z_m", "n/a"),
        )
    )


def format_demo_cached_candidate_accept_log(fields: Dict[str, Any]) -> str:
    return (
        "[DEMO_CACHED_CANDIDATE_ACCEPT]\n"
        "scene_id=%s\n"
        "label=%s\n"
        "cartesian_fraction=%s\n"
        "joint7_after=%s\n"
        "expected_joint7=%s\n"
        "transport_entry=%s\n"
        "place_slot=%s"
        % (
            fields.get("scene_id", "n/a"),
            fields.get("label", "n/a"),
            fields.get("cartesian_fraction", "n/a"),
            fields.get("joint7_after", "n/a"),
            fields.get("expected_joint7", "n/a"),
            fields.get("transport_entry", "n/a"),
            fields.get("place_slot", "n/a"),
        )
    )


def format_demo_cached_candidate_reject_log(fields: Dict[str, Any]) -> str:
    return (
        "[DEMO_CACHED_CANDIDATE_REJECT]\n"
        "scene_id=%s\n"
        "label=%s\n"
        "stage=%s\n"
        "reason=%s\n"
        "fallback=prioritized_grid"
        % (
            fields.get("scene_id", "n/a"),
            fields.get("label", "n/a"),
            fields.get("stage", "n/a"),
            fields.get("reason", ""),
        )
    )


def format_paired_grid_search_mode_log(fields: Dict[str, Any]) -> str:
    return (
        "[PAIRED_GRID_SEARCH_MODE]\n"
        "scene_id=%s\n"
        "label=%s\n"
        "mode_param=%s\n"
        "resolved_mode=%s\n"
        "max_candidates=%d\n"
        "cache_enabled=%s"
        % (
            fields.get("scene_id", "n/a"),
            fields.get("label", "n/a"),
            fields.get("mode_param", "n/a"),
            fields.get("resolved_mode", "n/a"),
            int(fields.get("max_candidates", 0)),
            str(bool(fields.get("cache_enabled"))).lower(),
        )
    )
