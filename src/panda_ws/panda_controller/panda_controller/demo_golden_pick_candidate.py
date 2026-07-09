"""Candidato pick golden validado para escenas demo (clave estable scene+layout+label)."""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

GOLDEN_CENTER_XY_TOL_M = 0.01
GOLDEN_YAW_TOL_DEG = 3.0
GOLDEN_TOP_Z_TOL_M = 0.01
GOLDEN_YAW_TOL_RAD = math.radians(GOLDEN_YAW_TOL_DEG)
GOLDEN_COMPARE_EPS_M = 1e-6
CHIPS_CAN_RUNTIME_HEIGHT_M = 0.25
CHIPS_CAN_LEGACY_DEPTH_FROM_TOP_M = 0.035
CHIPS_CAN_LEGACY_PREGRASP_ABOVE_TOP_M = 0.035
CHIPS_CAN_OBJECT_HIGH_CLEARANCE_ABOVE_TOP_M = 0.150
VALIDATED_STATUS = "validated_pick_place"
PLAN_PREFLIGHT_PICK_ONLY_STATUS = "plan_preflight_pick_only_certified"
ACCEPTED_GOLDEN_STATUSES = frozenset(
    {VALIDATED_STATUS, PLAN_PREFLIGHT_PICK_ONLY_STATUS}
)
EXPLICIT_SCENE_YAW_SOURCES = frozenset(
    {
        "runtime_gt_spawn_yaw",
        "runtime_scene_gt_spawn_yaw",
    }
)


def demo_golden_policy_scene_id(runtime_scene_id: str) -> str:
    """scene_id padre para golden/cache (demo_scene_02_3obj → demo_scene_02)."""
    from panda_vision.spawn.demo_scene_presets import demo_scene_policy_scene_id_for_preset

    return demo_scene_policy_scene_id_for_preset(str(runtime_scene_id or "").strip())


DEMO_GOLDEN_PARENT_SCENE_IDS = frozenset(
    {"demo_scene_01", "demo_scene_02", "demo_scene_03"}
)
POSE_ADAPTIVE_GOLDEN_LABELS = frozenset(
    {"cracker_box", "mustard_bottle", "sugar_box"}
)
GOLDEN_TEMPLATE_FALLBACK_SCENE_IDS = frozenset({"demo_scene_01", "demo_scene_03"})


def runtime_uses_demo_scene_02_golden(runtime_scene_id: str) -> bool:
    """Escenas demo multiobjeto (01/02/03) y deposit_* equivalentes pueden usar golden."""
    return demo_golden_policy_scene_id(runtime_scene_id) in DEMO_GOLDEN_PARENT_SCENE_IDS


def _to_float_val(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _yaw_from_runtime_scene_object(scene_obj: Dict[str, Any]) -> Optional[float]:
    col_pose = scene_obj.get("collision_box_pose") or {}
    if isinstance(col_pose, dict):
        yaw = _to_float_val(col_pose.get("yaw"))
        if yaw is not None:
            return yaw
    for key in ("yaw_rad", "spawn_yaw_rad", "object_yaw_rad"):
        yaw = _to_float_val(scene_obj.get(key))
        if yaw is not None:
            return yaw
    return None


def resolve_runtime_scene_yaw_rad(
    candidate: Optional[Dict[str, Any]],
    *,
    target_label: str = "",
) -> Tuple[Optional[float], str]:
    """Yaw geométrico RuntimeScene/GT/spawn. Nunca grasp/fit/TCP derivado."""
    if not isinstance(candidate, dict):
        return None, "missing"

    label = str(target_label or candidate.get("label") or "").strip().lower()
    scene_objects = candidate.get("_runtime_scene_objects")
    if isinstance(scene_objects, list) and label:
        for obj in scene_objects:
            if not isinstance(obj, dict):
                continue
            obj_label = str(obj.get("label") or "").strip().lower()
            if obj_label != label:
                continue
            src = str(obj.get("yaw_source") or "runtime_gt_spawn_yaw").strip().lower()
            if src in EXPLICIT_SCENE_YAW_SOURCES:
                yaw = _yaw_from_runtime_scene_object(obj)
                if yaw is not None:
                    return float(yaw), "runtime_scene_object:%s" % src
            return None, "runtime_scene_object:non_explicit_source"

    yaw_src = str(candidate.get("yaw_source") or "").strip().lower()
    if yaw_src in EXPLICIT_SCENE_YAW_SOURCES:
        for key in ("object_yaw_rad", "known_box_yaw_rad", "spawn_yaw_rad"):
            val = _to_float_val(candidate.get(key))
            if val is not None:
                return float(val), "%s:%s" % (key, yaw_src)

    spawn_yaw = _to_float_val(candidate.get("spawn_yaw_rad"))
    if spawn_yaw is not None:
        return float(spawn_yaw), "spawn_yaw_rad"

    return None, "none"


def default_demo_config_dir() -> str:
    fallback = str(Path(__file__).resolve().parent.parent / "config")
    try:
        from ament_index_python.packages import get_package_share_directory

        share = os.path.join(get_package_share_directory("panda_controller"), "config")
        if os.path.isdir(share):
            return share
    except Exception:
        pass
    return fallback


def resolve_golden_candidate_path(
    path_or_relative: str,
    *,
    config_dir: Optional[str] = None,
) -> str:
    raw = str(path_or_relative or "").strip()
    if not raw:
        return ""
    if os.path.isabs(raw) and os.path.isfile(raw):
        return raw
    base = str(config_dir or default_demo_config_dir())
    candidate = os.path.join(base, raw)
    if os.path.isfile(candidate):
        return candidate
    return candidate


def default_golden_candidate_path(
    scene_id: str,
    target_label: str,
    *,
    config_dir: Optional[str] = None,
) -> str:
    sid = demo_golden_policy_scene_id(scene_id)
    label = str(target_label or "").strip().lower()
    rel = "demo_candidate_cache/%s_%s_golden.yaml" % (sid, label)
    path = resolve_golden_candidate_path(rel, config_dir=config_dir)
    if os.path.isfile(path):
        return path
    if sid in GOLDEN_TEMPLATE_FALLBACK_SCENE_IDS:
        fallback_rel = "demo_candidate_cache/demo_scene_02_%s_golden.yaml" % label
        fallback_path = resolve_golden_candidate_path(
            fallback_rel, config_dir=config_dir
        )
        if os.path.isfile(fallback_path):
            return fallback_path
    return path


def _normalize_xyz(raw: Any) -> Optional[Tuple[float, float, float]]:
    if not isinstance(raw, (list, tuple)) or len(raw) < 3:
        return None
    return (float(raw[0]), float(raw[1]), float(raw[2]))


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


def load_golden_pick_candidate(yaml_path: str) -> Optional[Dict[str, Any]]:
    path = str(yaml_path or "").strip()
    if not path or not os.path.isfile(path):
        return None
    with open(path, "r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    return normalize_golden_pick_candidate(raw, source_file=path)


def normalize_golden_pick_candidate(
    raw: Dict[str, Any],
    *,
    source_file: str = "",
) -> Optional[Dict[str, Any]]:
    if not isinstance(raw, dict):
        return None
    sid = str(raw.get("scene_id", "")).strip().lower()
    label = str(raw.get("target_label", "")).strip().lower()
    layout = str(raw.get("layout_version", "")).strip()
    status = str(raw.get("status", "")).strip().lower()
    if not sid or not label:
        return None

    pose_raw = raw.get("object_pose") or {}
    center_xy = _normalize_xy(pose_raw.get("semantic_center_xy"))
    if center_xy is None:
        center_xy = _normalize_xy(
            [pose_raw.get("x"), pose_raw.get("y")]
            if pose_raw.get("x") is not None
            else None
        )
    if center_xy is None:
        return None

    cand_raw = raw.get("candidate") or {}
    if not isinstance(cand_raw, dict):
        return None
    pregrasp = _normalize_xyz(cand_raw.get("pregrasp_tcp"))
    grasp = _normalize_xyz(cand_raw.get("grasp_tcp"))
    lift = _normalize_xyz(cand_raw.get("lift_tcp"))
    if pregrasp is None or grasp is None:
        return None

    return {
        "scene_id": sid,
        "layout_version": layout,
        "target_label": label,
        "status": status,
        "source_file": str(source_file),
        "object_pose": {
            "semantic_center_xy": center_xy,
            "top_z": float(pose_raw.get("top_z", grasp[2] + 0.033)),
            "yaw_rad": float(pose_raw.get("yaw_rad", cand_raw.get("commanded_tcp_yaw_rad", 0.0))),
        },
        "candidate": {
            "candidate_idx": int(cand_raw.get("candidate_idx", 0)),
            "yaw_deg": float(cand_raw.get("yaw_deg", 0.0)),
            "commanded_tcp_yaw_rad": float(cand_raw.get("commanded_tcp_yaw_rad", 0.0)),
            "pregrasp_tcp": pregrasp,
            "grasp_tcp": grasp,
            "lift_tcp": lift,
            "depth_from_top_m": float(
                cand_raw.get(
                    "depth_from_top_m",
                    pregrasp[2] - grasp[2] if pregrasp[2] > grasp[2] else 0.033,
                )
            ),
            "ik_seed": str(cand_raw.get("ik_seed", "pick_workspace_ready")),
            "prevalidation_source": str(
                cand_raw.get("prevalidation_source", "demo_collision_off_final_descend")
            ),
            "cartesian_descend_fraction": float(
                cand_raw.get("cartesian_descend_fraction", 1.0)
            ),
        },
        "grasp": dict(raw.get("grasp") or {}),
        "transport": dict(raw.get("transport") or {}),
        "place": dict(raw.get("place") or {}),
        "legacy": dict(raw.get("legacy") or {}),
        "validation": dict(raw.get("validation") or {}),
    }


def golden_scene_id_compatible(golden_scene_id: str, runtime_scene_id: str) -> bool:
    golden_sid = str(golden_scene_id or "").strip().lower()
    runtime_sid = str(runtime_scene_id or "").strip().lower()
    if not golden_sid or not runtime_sid:
        return False
    if golden_sid == runtime_sid:
        return True
    parent_runtime = demo_golden_policy_scene_id(runtime_sid)
    if golden_sid == parent_runtime:
        return True
    return (
        parent_runtime == "demo_scene_02"
        and golden_sid.startswith("plan_certified_layout_")
    )


def validate_golden_candidate_identity(
    golden: Dict[str, Any],
    *,
    scene_id: str,
    layout_version: str,
    target_label: str,
) -> Tuple[bool, str]:
    sid = str(scene_id or "").strip().lower()
    label = str(target_label or "").strip().lower()
    layout = str(layout_version or "").strip()
    golden_sid = str(golden.get("scene_id", "")).strip().lower()
    if not golden_scene_id_compatible(golden_sid, sid):
        return False, "scene_id_mismatch"
    if str(golden.get("target_label", "")).strip().lower() != label:
        return False, "target_label_mismatch"
    if layout and str(golden.get("layout_version", "")).strip() != layout:
        return False, "layout_version_mismatch"
    status = str(golden.get("status", "")).strip().lower()
    if status not in ACCEPTED_GOLDEN_STATUSES:
        return False, "status_not_validated"
    return True, "OK"


def validate_golden_candidate_compatibility(
    golden: Dict[str, Any],
    *,
    scene_id: str,
    layout_version: str,
    target_label: str,
    runtime_xy: Tuple[float, float],
    runtime_top_z: float,
    runtime_scene_yaw_rad: Optional[float] = None,
    runtime_scene_yaw_source: str = "none",
    runtime_commanded_tcp_yaw_rad: Optional[float] = None,
) -> Tuple[bool, str, Dict[str, Any]]:
    ok_id, id_reason = validate_golden_candidate_identity(
        golden,
        scene_id=scene_id,
        layout_version=layout_version,
        target_label=target_label,
    )
    exp_pose = golden.get("object_pose") or {}
    exp_xy = exp_pose.get("semantic_center_xy") or (0.0, 0.0)
    golden_scene_yaw = float(exp_pose.get("yaw_rad", 0.0))
    golden_commanded = float(
        (golden.get("candidate") or {}).get("commanded_tcp_yaw_rad", 0.0)
    )
    center_error_m = math.hypot(
        float(runtime_xy[0]) - float(exp_xy[0]),
        float(runtime_xy[1]) - float(exp_xy[1]),
    )
    top_z_error_m = abs(float(runtime_top_z) - float(exp_pose.get("top_z", runtime_top_z)))

    scene_yaw_source = str(runtime_scene_yaw_source or "none").strip().lower()
    has_explicit_scene_yaw = (
        runtime_scene_yaw_rad is not None
        and scene_yaw_source not in ("", "none", "missing")
        and not scene_yaw_source.endswith("non_explicit_source")
    )

    yaw_compare_mode = ""
    yaw_error_deg = float("inf")
    if has_explicit_scene_yaw:
        yaw_compare_mode = "scene_yaw_vs_scene_yaw"
        yaw_error_deg = math.degrees(
            _yaw_delta_rad(float(runtime_scene_yaw_rad), golden_scene_yaw)
        )
    elif runtime_commanded_tcp_yaw_rad is not None:
        yaw_compare_mode = "commanded_tcp_yaw_vs_commanded_tcp_yaw"
        yaw_error_deg = math.degrees(
            _yaw_delta_rad(float(runtime_commanded_tcp_yaw_rad), golden_commanded)
        )

    details: Dict[str, Any] = {
        "center_error_m": float(center_error_m),
        "yaw_error_deg": float(yaw_error_deg),
        "top_z_error_m": float(top_z_error_m),
        "center_xy_tol_m": float(GOLDEN_CENTER_XY_TOL_M),
        "yaw_tol_deg": float(GOLDEN_YAW_TOL_DEG),
        "top_z_tol_m": float(GOLDEN_TOP_Z_TOL_M),
        "current_scene_yaw_rad": runtime_scene_yaw_rad if has_explicit_scene_yaw else None,
        "current_scene_yaw_source": scene_yaw_source,
        "golden_scene_yaw_rad": golden_scene_yaw,
        "current_commanded_tcp_yaw_rad": runtime_commanded_tcp_yaw_rad,
        "golden_commanded_tcp_yaw_rad": golden_commanded,
        "yaw_compare_mode": yaw_compare_mode,
    }
    if not ok_id:
        details["identity_reason"] = id_reason
        return False, id_reason, details
    if center_error_m > GOLDEN_CENTER_XY_TOL_M + GOLDEN_COMPARE_EPS_M:
        return False, "center_xy_out_of_tolerance", details
    if not yaw_compare_mode:
        return False, "missing_runtime_yaw", details
    if yaw_error_deg > GOLDEN_YAW_TOL_DEG + GOLDEN_COMPARE_EPS_M:
        return False, "yaw_out_of_tolerance", details
    if top_z_error_m > GOLDEN_TOP_Z_TOL_M + GOLDEN_COMPARE_EPS_M:
        return False, "top_z_out_of_tolerance", details
    return True, "OK", details


def enrich_box_golden_pose_adaptive_from_runtime(
    golden: Dict[str, Any],
    *,
    runtime_xy: Tuple[float, float],
    runtime_top_z: float,
    runtime_commanded_tcp_yaw_rad: float,
    runtime_scene_yaw_rad: Optional[float] = None,
) -> Dict[str, Any]:
    """Reancla el golden demo_scene_02 al centro/yaw runtime (escenas 01/03)."""
    pose = dict(golden.get("object_pose") or {})
    exp_xy = pose.get("semantic_center_xy") or [0.0, 0.0]
    dx = float(runtime_xy[0]) - float(exp_xy[0])
    dy = float(runtime_xy[1]) - float(exp_xy[1])
    golden_top_z = float(pose.get("top_z", runtime_top_z))
    dz = float(runtime_top_z) - golden_top_z

    def _shift_xyz(raw: Any) -> Optional[list]:
        pt = _normalize_xyz(raw)
        if pt is None:
            return None
        return [
            round(float(pt[0]) + dx, 4),
            round(float(pt[1]) + dy, 4),
            round(float(pt[2]) + dz, 4),
        ]

    cand = dict(golden.get("candidate") or {})
    for key in ("pregrasp_tcp", "grasp_tcp", "lift_tcp"):
        shifted = _shift_xyz(cand.get(key))
        if shifted is not None:
            cand[key] = shifted
    cand["commanded_tcp_yaw_rad"] = float(runtime_commanded_tcp_yaw_rad)
    if runtime_scene_yaw_rad is not None:
        cand["yaw_deg"] = math.degrees(float(runtime_scene_yaw_rad))

    out = dict(golden)
    out["object_pose"] = {
        **pose,
        "semantic_center_xy": [round(float(runtime_xy[0]), 4), round(float(runtime_xy[1]), 4)],
        "top_z": round(float(runtime_top_z), 4),
    }
    if runtime_scene_yaw_rad is not None:
        out["object_pose"]["yaw_rad"] = float(runtime_scene_yaw_rad)
    out["candidate"] = cand
    out["_pose_adaptive_from_demo_scene_02"] = True
    return out


def prepare_demo_golden_for_runtime(
    golden: Optional[Dict[str, Any]],
    *,
    scene_id: str,
    layout_version: str,
    target_label: str,
    runtime_xy: Tuple[float, float],
    runtime_top_z: float,
    runtime_scene_yaw_rad: Optional[float] = None,
    runtime_scene_yaw_source: str = "none",
    runtime_commanded_tcp_yaw_rad: Optional[float] = None,
) -> Tuple[Optional[Dict[str, Any]], bool, str, Dict[str, Any]]:
    """Carga golden estricto o reanclado pose-adaptive para escenas 01/03."""
    if golden is None:
        return None, False, "golden_missing", {}
    compat_ok, compat_reason, compat_details = validate_golden_candidate_compatibility(
        golden,
        scene_id=scene_id,
        layout_version=layout_version,
        target_label=target_label,
        runtime_xy=runtime_xy,
        runtime_top_z=float(runtime_top_z),
        runtime_scene_yaw_rad=runtime_scene_yaw_rad,
        runtime_scene_yaw_source=runtime_scene_yaw_source,
        runtime_commanded_tcp_yaw_rad=runtime_commanded_tcp_yaw_rad,
    )
    if compat_ok:
        return golden, True, "OK", compat_details

    label = str(target_label or "").strip().lower()
    runtime_parent = demo_golden_policy_scene_id(scene_id)
    golden_sid = str(golden.get("scene_id", "")).strip().lower()
    pose_adaptive_reasons = frozenset(
        {
            "scene_id_mismatch",
            "center_xy_out_of_tolerance",
            "yaw_out_of_tolerance",
            "top_z_out_of_tolerance",
        }
    )
    mustard_top_z_pose_adaptive_max_m = 0.030
    if (
        label in POSE_ADAPTIVE_GOLDEN_LABELS
        and golden_sid == "demo_scene_02"
        and (
            runtime_parent in GOLDEN_TEMPLATE_FALLBACK_SCENE_IDS
            or runtime_parent == "demo_scene_02"
        )
        and compat_reason in pose_adaptive_reasons
        and runtime_commanded_tcp_yaw_rad is not None
    ):
        top_z_error_m = float(compat_details.get("top_z_error_m", 0.0))
        if compat_reason == "top_z_out_of_tolerance":
            if (
                label == "mustard_bottle"
                and top_z_error_m
                > float(mustard_top_z_pose_adaptive_max_m) + GOLDEN_COMPARE_EPS_M
            ):
                return golden, False, "top_z_out_of_tolerance", compat_details
        elif top_z_error_m > GOLDEN_TOP_Z_TOL_M + GOLDEN_COMPARE_EPS_M:
            return golden, False, "top_z_out_of_tolerance", compat_details
        adapted = enrich_box_golden_pose_adaptive_from_runtime(
            golden,
            runtime_xy=runtime_xy,
            runtime_top_z=float(runtime_top_z),
            runtime_commanded_tcp_yaw_rad=float(runtime_commanded_tcp_yaw_rad),
            runtime_scene_yaw_rad=runtime_scene_yaw_rad,
        )
        compat_details = dict(compat_details)
        compat_details["pose_adaptive"] = True
        compat_details["pose_adaptive_reason"] = compat_reason
        return adapted, True, "OK_pose_adaptive", compat_details
    return golden, False, compat_reason, compat_details


def golden_entry_to_grid_spec(
    golden: Dict[str, Any],
    *,
    gripper_physical_yaw_correction_rad: float = 0.0,
) -> Dict[str, Any]:
    cand = golden.get("candidate") or {}
    pre = cand["pregrasp_tcp"]
    gr = cand["grasp_tcp"]
    cmd = float(cand["commanded_tcp_yaw_rad"])
    phys = float(gripper_physical_yaw_correction_rad)
    yaw_rad = _wrap_to_pi(cmd - phys)
    depth = float(cand.get("depth_from_top_m") or (pre[2] - gr[2]))
    return {
        "grid_idx": int(cand.get("candidate_idx", 0)),
        "yaw_name": "golden",
        "yaw_rad": float(yaw_rad),
        "yaw_deg": float(cand.get("yaw_deg", math.degrees(yaw_rad))),
        "pregrasp_tcp_z": float(pre[2]),
        "grasp_tcp_z": float(gr[2]),
        "depth_from_top_m": float(depth),
        "pre_plan": (float(pre[0]), float(pre[1]), float(pre[2])),
        "gr_plan": (float(gr[0]), float(gr[1]), float(gr[2])),
        "ik_seed_label": str(cand.get("ik_seed", "pick_workspace_ready")),
        "priority": "golden",
        "source": "demo_golden_candidate",
        "prevalidation_source": str(cand.get("prevalidation_source", "")),
        "cartesian_descend_fraction": float(cand.get("cartesian_descend_fraction", 1.0)),
    }


def apply_golden_runtime_overrides(
    candidate: Dict[str, Any],
    golden: Dict[str, Any],
) -> None:
    transport = golden.get("transport") or {}
    place = golden.get("place") or {}
    cand = golden.get("candidate") or {}
    if transport.get("selected_transport_entry"):
        candidate["_golden_selected_transport_entry"] = str(
            transport["selected_transport_entry"]
        )
    route = transport.get("route")
    if isinstance(route, (list, tuple)) and route:
        candidate["_golden_transport_route"] = [str(x) for x in route]
    if place.get("slot_index") is not None:
        slot_idx = int(place["slot_index"])
        candidate["_golden_place_slot_index"] = slot_idx
        if not bool(candidate.get("_place_slot_request_locked")):
            candidate["place_slot_user_specified"] = True
            candidate["place_slot_index"] = slot_idx
    slot_name = str(place.get("slot_name") or "").strip()
    if slot_name and not bool(candidate.get("_place_slot_request_locked")):
        candidate["place_slot_name"] = slot_name
    release_z = _to_float_val(place.get("release_tcp_z"))
    if release_z is not None:
        candidate["_golden_place_release_tcp_z"] = float(release_z)
        candidate["selected_release_tcp_z"] = float(release_z)
    rel_cands = place.get("release_tcp_z_candidates")
    if isinstance(rel_cands, (list, tuple)) and rel_cands:
        candidate["_golden_place_release_tcp_z_candidates"] = [
            float(z) for z in rel_cands
        ]
    approach_z = _to_float_val(place.get("approach_tcp_z"))
    if approach_z is not None:
        candidate["_golden_place_approach_tcp_z"] = float(approach_z)
    retreat_z = _to_float_val(place.get("retreat_tcp_z"))
    if retreat_z is not None:
        candidate["_golden_place_retreat_tcp_z"] = float(retreat_z)
    if cand.get("lift_tcp") is not None:
        candidate["_golden_lift_tcp"] = list(cand["lift_tcp"])
    depth_from_top = _to_float_val(cand.get("depth_from_top_m"))
    if depth_from_top is not None:
        candidate["recommended_grasp_depth_from_top_m"] = float(depth_from_top)
        candidate["depth_from_top_m"] = float(depth_from_top)
    grasp = golden.get("grasp") or {}
    open_j = _to_float_val(grasp.get("open_joint"))
    close_j = _to_float_val(grasp.get("close_joint"))
    width_m = _to_float_val(grasp.get("expected_width_m"))
    if open_j is not None:
        candidate["recommended_open_joint_m"] = float(open_j)
        candidate["open_joint_m"] = float(open_j)
    if close_j is not None:
        candidate["recommended_close_joint_m"] = float(close_j)
    if width_m is not None:
        candidate["expected_object_width_m"] = float(width_m)
        candidate["object_width_m"] = float(width_m)
    obj_pose = golden.get("object_pose") or {}
    golden_top_z = _to_float_val(obj_pose.get("top_z"))
    if golden_top_z is not None:
        candidate["top_z_m"] = float(golden_top_z)
        candidate["top_z_estimated"] = float(golden_top_z)
    candidate["_demo_golden_candidate"] = True
    candidate["_demo_golden_prevalidation_source"] = str(
        cand.get("prevalidation_source", "")
    )


def format_golden_candidate_load_log(fields: Dict[str, Any]) -> str:
    return (
        "[GOLDEN_CANDIDATE_LOAD]\n"
        "path=%s\n"
        "scene_id=%s\n"
        "target_label=%s\n"
        "status=%s\n"
        "result=%s"
        % (
            fields.get("path", ""),
            fields.get("scene_id", ""),
            fields.get("target_label", ""),
            fields.get("status", ""),
            fields.get("result", "FAIL"),
        )
    )


def format_golden_candidate_compatibility_log(fields: Dict[str, Any]) -> str:
    def _fmt_yaw(value: Any) -> str:
        if value is None:
            return "n/a"
        try:
            return "%.4f" % float(value)
        except (TypeError, ValueError):
            return str(value)

    return (
        "[GOLDEN_CANDIDATE_COMPATIBILITY]\n"
        "center_error_m=%s\n"
        "current_scene_yaw_rad=%s\n"
        "current_scene_yaw_source=%s\n"
        "golden_scene_yaw_rad=%s\n"
        "current_commanded_tcp_yaw_rad=%s\n"
        "golden_commanded_tcp_yaw_rad=%s\n"
        "yaw_compare_mode=%s\n"
        "yaw_error_deg=%s\n"
        "top_z_error_m=%s\n"
        "result=%s\n"
        "reason=%s"
        % (
            fields.get("center_error_m", "n/a"),
            _fmt_yaw(fields.get("current_scene_yaw_rad")),
            fields.get("current_scene_yaw_source", "none"),
            _fmt_yaw(fields.get("golden_scene_yaw_rad")),
            _fmt_yaw(fields.get("current_commanded_tcp_yaw_rad")),
            _fmt_yaw(fields.get("golden_commanded_tcp_yaw_rad")),
            fields.get("yaw_compare_mode", ""),
            fields.get("yaw_error_deg", "n/a"),
            fields.get("top_z_error_m", "n/a"),
            fields.get("result", "FAIL"),
            fields.get("reason", ""),
        )
    )


def format_golden_candidate_selected_log(fields: Dict[str, Any]) -> str:
    route = fields.get("route") or []
    route_s = ",".join(str(x) for x in route) if route else ""
    return (
        "[GOLDEN_CANDIDATE_SELECTED]\n"
        "target_label=%s\n"
        "candidate_idx=%s\n"
        "pregrasp_tcp=%s\n"
        "grasp_tcp=%s\n"
        "transport_entry=%s\n"
        "route=[%s]\n"
        "result=%s"
        % (
            fields.get("target_label", ""),
            fields.get("candidate_idx", "n/a"),
            fields.get("pregrasp_tcp", ""),
            fields.get("grasp_tcp", ""),
            fields.get("transport_entry", ""),
            route_s,
            fields.get("result", "OK"),
        )
    )


def format_golden_candidate_fallback_log(fields: Dict[str, Any]) -> str:
    return (
        "[GOLDEN_CANDIDATE_FALLBACK_TO_GRID]\n"
        "reason=%s"
        % (fields.get("reason", ""),)
    )


def enrich_chips_can_legacy_golden_fields(
    golden: Dict[str, Any],
    *,
    grasp_xy: Tuple[float, float],
) -> Dict[str, Any]:
    """Alinea golden chips_can con top_z runtime (0.510) y bloque legacy del executor."""
    gx, gy = float(grasp_xy[0]), float(grasp_xy[1])
    pose = dict(golden.get("object_pose") or {})
    cand = dict(golden.get("candidate") or {})
    pose_top_z = _to_float_val(pose.get("top_z"))
    if pose_top_z is not None and abs(float(pose_top_z) - 0.510) <= GOLDEN_TOP_Z_TOL_M:
        runtime_top_z = round(float(pose_top_z), 4)
    else:
        runtime_top_z = 0.5100
    legacy_pre_z = round(
        runtime_top_z + CHIPS_CAN_LEGACY_PREGRASP_ABOVE_TOP_M, 4
    )
    legacy_gr_z = round(runtime_top_z - CHIPS_CAN_LEGACY_DEPTH_FROM_TOP_M, 4)
    object_high_z = round(
        runtime_top_z + CHIPS_CAN_OBJECT_HIGH_CLEARANCE_ABOVE_TOP_M, 4
    )
    cmd_yaw = float(cand.get("commanded_tcp_yaw_rad", math.pi))
    golden = dict(golden)
    golden["object_pose"] = {
        **pose,
        "semantic_center_xy": [round(gx, 4), round(gy, 4)],
        "top_z": runtime_top_z,
    }
    golden["candidate"] = {
        **cand,
        "pregrasp_tcp": [round(gx, 4), round(gy, 4), legacy_pre_z],
        "grasp_tcp": [round(gx, 4), round(gy, 4), legacy_gr_z],
        "lift_tcp": [
            round(gx, 4),
            round(gy, 4),
            round(legacy_pre_z + 0.08, 4),
        ],
        "depth_from_top_m": CHIPS_CAN_LEGACY_DEPTH_FROM_TOP_M,
        "commanded_tcp_yaw_rad": cmd_yaw,
        "prevalidation_source": str(
            cand.get("prevalidation_source")
            or "chips_can_legacy_pending_actual_tf_descend"
        ),
    }
    golden["legacy"] = {
        "policy": "legacy_successful_pick",
        "contract": "OK_CHIPS_LEGACY_PENDING_ACTUAL_TF_DESCEND",
        "pregrasp_height_above_top_m": CHIPS_CAN_LEGACY_PREGRASP_ABOVE_TOP_M,
        "depth_from_top_m": CHIPS_CAN_LEGACY_DEPTH_FROM_TOP_M,
        "object_high_tcp_z": object_high_z,
        "legacy_low_pregrasp_tcp_z": legacy_pre_z,
        "object_high_clearance_above_top_m": CHIPS_CAN_OBJECT_HIGH_CLEARANCE_ABOVE_TOP_M,
        "object_high_to_low_fraction": float(
            cand.get("cartesian_descend_fraction") or 1.0
        ),
        "variant_name": "golden",
    }
    return golden


def chips_can_golden_legacy_probe_targets(
    golden: Dict[str, Any],
    *,
    grasp_xy: Tuple[float, float],
    top_z_m: float,
) -> Optional[Dict[str, Any]]:
    """Parámetros legacy del golden chips_can para un único probe (sin grid completo)."""
    legacy = golden.get("legacy") or {}
    cand = golden.get("candidate") or {}
    pre_h = _to_float_val(legacy.get("pregrasp_height_above_top_m"))
    depth = _to_float_val(legacy.get("depth_from_top_m") or cand.get("depth_from_top_m"))
    cmd_yaw = _to_float_val(cand.get("commanded_tcp_yaw_rad"))
    if pre_h is None or depth is None or cmd_yaw is None:
        return None
    gx, gy = float(grasp_xy[0]), float(grasp_xy[1])
    legacy_pre_z = float(top_z_m) + float(pre_h)
    gr_z = float(top_z_m) - float(depth)
    high_clear = _to_float_val(legacy.get("object_high_clearance_above_top_m"))
    if high_clear is None:
        high_clear = 0.150
    entry_z = float(top_z_m) + float(high_clear)
    return {
        "pregrasp_height_above_top_m": float(pre_h),
        "depth_from_top_m": float(depth),
        "commanded_tcp_yaw_rad": float(cmd_yaw),
        "variant_name": str(legacy.get("variant_name") or "golden"),
        "pre_plan": (gx, gy, float(legacy_pre_z)),
        "gr_plan": (gx, gy, float(gr_z)),
        "entry_tcp": (gx, gy, float(entry_z)),
        "object_high_to_low_fraction": float(
            legacy.get("object_high_to_low_fraction")
            or cand.get("cartesian_descend_fraction")
            or 1.0
        ),
    }


def format_chips_can_golden_required_missing_log(fields: Dict[str, Any]) -> str:
    return (
        "[CHIPS_CAN_GOLDEN_REQUIRED_MISSING]\n"
        "scene_id=%s\n"
        "target_label=%s\n"
        "reason=%s\n"
        "result=FAIL"
        % (
            fields.get("scene_id", ""),
            fields.get("target_label", ""),
            fields.get("reason", "n/a"),
        )
    )


def format_chips_can_golden_legacy_variant_log(fields: Dict[str, Any]) -> str:
    return (
        "[CHIPS_CAN_GOLDEN_LEGACY_VARIANT]\n"
        "scene_id=%s\n"
        "target_label=%s\n"
        "pregrasp_height_above_top_m=%.4f\n"
        "depth_from_top_m=%.4f\n"
        "commanded_tcp_yaw_rad=%.4f\n"
        "object_high_to_low_fraction=%.5f\n"
        "high_to_low_result=%s\n"
        "result=%s"
        % (
            fields.get("scene_id", ""),
            fields.get("target_label", ""),
            float(fields.get("pregrasp_height_above_top_m", 0.0)),
            float(fields.get("depth_from_top_m", 0.0)),
            float(fields.get("commanded_tcp_yaw_rad", 0.0)),
            float(fields.get("object_high_to_low_fraction", 0.0)),
            str(fields.get("high_to_low_result", "n/a")),
            fields.get("result", "OK"),
        )
    )


GOLDEN_FAST_EXECUTE_LABELS = frozenset(
    {"cracker_box", "chips_can", "sugar_box", "mustard_bottle"}
)


def golden_fast_execute_label_supported(target_label: str) -> bool:
    return str(target_label or "").strip().lower() in GOLDEN_FAST_EXECUTE_LABELS


def apply_golden_plan_targets(
    plan_targets: Dict[str, Any],
    golden: Dict[str, Any],
) -> None:
    cand = golden.get("candidate") or {}
    pre = _normalize_xyz(cand.get("pregrasp_tcp"))
    gr = _normalize_xyz(cand.get("grasp_tcp"))
    lift = _normalize_xyz(cand.get("lift_tcp"))
    if pre is not None:
        plan_targets["pregrasp_tcp"] = pre
    if gr is not None:
        plan_targets["grasp_tcp"] = gr
    if lift is not None:
        plan_targets["lift_tcp"] = lift


def build_golden_fast_execute_detection_entry(
    golden: Dict[str, Any],
    *,
    target_label: str = "",
    entity_name: str = "",
) -> Optional[Dict[str, Any]]:
    """Detección mínima desde golden cuando YOLO no ve el target (clear_table demo)."""
    if not isinstance(golden, dict):
        return None
    label = str(target_label or golden.get("target_label") or "").strip().lower()
    if not label:
        return None
    obj_pose = golden.get("object_pose") or {}
    xy = obj_pose.get("semantic_center_xy")
    top_z = _to_float_val(obj_pose.get("top_z"))
    if not isinstance(xy, (list, tuple)) or len(xy) < 2 or top_z is None:
        return None
    yaw = _to_float_val(obj_pose.get("yaw_rad"))
    ent = str(entity_name or "").strip()
    cx, cy = float(xy[0]), float(xy[1])
    top = float(top_z)
    if label == "chips_can":
        grasp_center_source = "runtime_gt_cylinder_center"
        top_face_source = "runtime_gt_known_cylinder"
    elif label in ("mustard_bottle",):
        grasp_center_source = "runtime_gt_tall_object_center"
        top_face_source = "runtime_gt_known_object"
    else:
        grasp_center_source = "runtime_gt_box_center"
        top_face_source = "runtime_gt_known_box"
    entry: Dict[str, Any] = {
        "label": label,
        "score": 1.0,
        "position": (cx, cy, top),
        "grasp_center_base": [cx, cy, top],
        "known_box_center_base": [cx, cy, top],
        "chosen_target_center_base": [cx, cy, top],
        "approach_position": (cx, cy, top + 0.08),
        "top_z_m": top,
        "top_z_estimated": top,
        "grasp_yaw_rad": yaw,
        "object_yaw_rad": yaw,
        "closing_yaw_rad": yaw,
        "active_yaw_rad": yaw,
        "yaw": yaw,
        "grasp_center_source": grasp_center_source,
        "yaw_source": "runtime_gt_spawn_yaw",
        "top_face_source": top_face_source,
        "supported_for_demo": True,
        "entity_name": ent,
        "gt_entity_name": ent,
        "_golden_fast_execute_detection_fallback": True,
    }
    grasp = golden.get("grasp") or {}
    if grasp.get("strategy"):
        entry["grasp_strategy"] = str(grasp["strategy"])
    open_j = _to_float_val(grasp.get("open_joint"))
    close_j = _to_float_val(grasp.get("close_joint"))
    if open_j is not None:
        entry["recommended_open_joint_m"] = float(open_j)
    if close_j is not None:
        entry["recommended_close_joint_m"] = float(close_j)
    width = _to_float_val(grasp.get("expected_width_m"))
    if width is not None:
        entry["required_grasp_width_m"] = float(width)
    return entry


def format_golden_fast_execute_detection_fallback_log(fields: Dict[str, Any]) -> str:
    return (
        "[GOLDEN_FAST_EXECUTE_DETECTION_FALLBACK]\n"
        "target_label=%s\n"
        "path=%s\n"
        "entity_name=%s\n"
        "result=%s"
        % (
            fields.get("target_label", ""),
            fields.get("path", ""),
            fields.get("entity_name", "n/a"),
            fields.get("result", "OK"),
        )
    )


def build_golden_fast_execute_grasp_valid_entry(
    golden: Dict[str, Any],
    *,
    gripper_physical_yaw_correction_rad: float = 0.0,
    pregrasp_js: Any = None,
    quat: Optional[Tuple[float, float, float, float]] = None,
    hand_q: Optional[Tuple[float, float, float, float]] = None,
) -> Dict[str, Any]:
    spec = golden_entry_to_grid_spec(
        golden,
        gripper_physical_yaw_correction_rad=gripper_physical_yaw_correction_rad,
    )
    cand = golden.get("candidate") or {}
    cmd_yaw = float(cand.get("commanded_tcp_yaw_rad", 0.0))
    prevalidation_source = str(
        cand.get("prevalidation_source", "demo_golden_validated")
    )
    cart_frac = float(cand.get("cartesian_descend_fraction", 1.0))
    return {
        "variant_index": int(spec["grid_idx"]),
        "variant_name": "demo_golden_fast_execute",
        "quat": quat,
        "hand_quat": hand_q,
        "commanded_yaw_rad": cmd_yaw,
        "pregrasp_js": pregrasp_js,
        "aligned_pregrasp_js": pregrasp_js,
        "raw_pregrasp_js": pregrasp_js,
        "pre_plan": spec["pre_plan"],
        "gr_plan": spec["gr_plan"],
        "plan_to_pregrasp_ok": True,
        "ik_pregrasp_ok": True,
        "_cart_frac": cart_frac,
        "_lift_ok": True,
        "_prevalidation_source": prevalidation_source,
        "_demo_golden_fast_execute": True,
        "_demo_golden_candidate": True,
    }


def format_golden_fast_execute_log(fields: Dict[str, Any]) -> str:
    route = fields.get("route") or []
    route_s = ",".join(str(x) for x in route) if route else ""
    return (
        "[GOLDEN_FAST_EXECUTE]\n"
        "target_label=%s\n"
        "path=%s\n"
        "pregrasp_tcp=%s\n"
        "grasp_tcp=%s\n"
        "transport_entry=%s\n"
        "route=%s\n"
        "prevalidation_source=%s\n"
        "result=%s"
        % (
            fields.get("target_label", ""),
            fields.get("path", ""),
            fields.get("pregrasp_tcp", ""),
            fields.get("grasp_tcp", ""),
            fields.get("transport_entry", ""),
            route_s,
            fields.get("prevalidation_source", ""),
            fields.get("result", "OK"),
        )
    )


def sugar_pick_golden_persist_eligible(scene_id: str) -> bool:
    """Tras pick+place OK, puede actualizar demo_scene_02_sugar_box_golden.yaml."""
    return demo_golden_policy_scene_id(str(scene_id or "").strip()) == "demo_scene_02"


def _xyz_list_from_candidate(
    candidate: Dict[str, Any], *keys: str
) -> Optional[List[float]]:
    for key in keys:
        raw = candidate.get(key)
        pt = _normalize_xyz(raw)
        if pt is not None:
            return [round(float(pt[0]), 4), round(float(pt[1]), 4), round(float(pt[2]), 4)]
    return None


def build_sugar_pick_golden_from_success(
    candidate: Dict[str, Any],
    *,
    place_payload: Optional[Dict[str, Any]] = None,
    existing_golden: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Construye golden pick sugar desde ejecución validada (deposit_02 / demo_scene_02)."""
    base = dict(existing_golden or {})
    place_payload = place_payload if isinstance(place_payload, dict) else {}
    center = _normalize_xyz(
        candidate.get("grasp_center_base")
        or candidate.get("chosen_target_center_base")
        or candidate.get("position")
    )
    if center is None:
        pose = (base.get("object_pose") or {}) if isinstance(base, dict) else {}
        xy = pose.get("semantic_center_xy") or [0.630, -0.175]
        center = (float(xy[0]), float(xy[1]), float(pose.get("top_z", 0.435)))
    top_z = _to_float_val(candidate.get("top_z_m"))
    if top_z is None:
        top_z = _to_float_val(candidate.get("top_z_estimated"))
    if top_z is None:
        top_z = float(center[2]) if len(center) > 2 else 0.435

    pregrasp = _xyz_list_from_candidate(
        candidate, "_pregrasp_tcp_planning", "pregrasp_tcp"
    )
    grasp = _xyz_list_from_candidate(candidate, "_grasp_tcp_planning", "grasp_tcp")
    lift = _xyz_list_from_candidate(candidate, "_lift_tcp_planning", "lift_tcp")
    if pregrasp is None or grasp is None:
        cand = (base.get("candidate") or {}) if isinstance(base, dict) else {}
        pregrasp = list(cand.get("pregrasp_tcp") or [0.630, -0.175, 0.472])
        grasp = list(cand.get("grasp_tcp") or [0.630, -0.175, 0.407])
    if lift is None:
        lift = [pregrasp[0], pregrasp[1], round(float(grasp[2]) + 0.150, 4)]

    cmd_yaw = _to_float_val(candidate.get("_base_commanded_tcp_yaw_rad"))
    if cmd_yaw is None:
        cmd_yaw = _to_float_val(
            ((base.get("candidate") or {}) if isinstance(base, dict) else {}).get(
                "commanded_tcp_yaw_rad"
            )
        )
    if cmd_yaw is None:
        cmd_yaw = 1.6965

    scene_yaw = _to_float_val(candidate.get("object_yaw_rad"))
    if scene_yaw is None:
        scene_yaw = _to_float_val(
            ((base.get("object_pose") or {}) if isinstance(base, dict) else {}).get(
                "yaw_rad"
            )
        )
    if scene_yaw is None:
        scene_yaw = -3.0159

    depth = float(pregrasp[2]) - float(grasp[2])
    slot_idx = place_payload.get("place_slot_index")
    if slot_idx is None:
        slot_idx = candidate.get("place_slot_index")
    try:
        slot_index = int(slot_idx)
    except (TypeError, ValueError):
        slot_index = 2
    release_z = _to_float_val(place_payload.get("release_tcp_z"))
    if release_z is None:
        release_z = _to_float_val(candidate.get("_final_release_tcp_z"))
    if release_z is None:
        release_z = _to_float_val(
            ((base.get("place") or {}) if isinstance(base, dict) else {}).get(
                "release_tcp_z"
            )
        )
    if release_z is None:
        release_z = 0.200

    dep_x = _to_float_val(place_payload.get("x"))
    dep_y = _to_float_val(place_payload.get("y"))
    if dep_x is None or dep_y is None:
        dep = ((base.get("place") or {}) if isinstance(base, dict) else {}).get(
            "deposit_xy"
        ) or [-0.370, -0.100]
        dep_x, dep_y = float(dep[0]), float(dep[1])

    transport = dict((base.get("transport") or {}) if isinstance(base, dict) else {})
    grasp_meta = dict((base.get("grasp") or {}) if isinstance(base, dict) else {})
    return {
        "scene_id": "demo_scene_02",
        "layout_version": str(
            base.get("layout_version") or "v3_clear_table_transport"
        ),
        "target_label": "sugar_box",
        "status": VALIDATED_STATUS,
        "object_pose": {
            "semantic_center_xy": [round(float(center[0]), 4), round(float(center[1]), 4)],
            "top_z": round(float(top_z), 4),
            "yaw_rad": float(scene_yaw),
        },
        "candidate": {
            "candidate_idx": 0,
            "yaw_deg": round(math.degrees(float(scene_yaw)), 2),
            "commanded_tcp_yaw_rad": round(float(cmd_yaw), 4),
            "pregrasp_tcp": pregrasp,
            "grasp_tcp": grasp,
            "lift_tcp": lift,
            "depth_from_top_m": round(float(depth), 4),
            "ik_seed": "pick_workspace_ready",
            "prevalidation_source": "geometric_fallback",
            "cartesian_descend_fraction": 1.0,
        },
        "grasp": grasp_meta
        or {
            "strategy": "top_down_short_axis",
            "open_joint": 0.0399,
            "close_joint": 0.0270,
            "expected_width_m": 0.0380,
        },
        "transport": transport
        or {
            "selected_transport_entry": "vertical_raise_then_rear_retreat",
            "route": [
                "carry_mid_high",
                "turn_back_extended_aligned",
                "box_front_high",
                "box_high",
            ],
            "first_hub": "carry_mid_high",
            "backend": "direct_action",
        },
        "place": {
            "slot_index": int(slot_index),
            "slot_name": "slot_%d" % (int(slot_index) + 1),
            "deposit_xy": [round(float(dep_x), 3), round(float(dep_y), 3)],
            "release_tcp_z": round(float(release_z), 4),
            "release_tcp_z_candidates": [0.220, 0.230, 0.240, 0.250, 0.200],
            "release_source": "food_safe_dynamic_with_fallback",
            "approach_tcp_z": 0.650,
            "retreat_tcp_z": 0.650,
        },
        "validation": {
            "result": VALIDATED_STATUS,
            "golden_run_date": "2026-07-01",
            "prevalidation_source": "geometric_fallback",
            "validated_in_scenes": [
                "deposit_02_cracker_chips",
                "deposit_03_mustard_only",
                "demo_scene_02",
            ],
            "pick_place_pipeline_completed": True,
            "notes": (
                "Golden actualizado tras pick+place OK en deposit_02_cracker_chips "
                "(sugar+mustard en mesa). pregrasp=%s grasp=%s release_z=%.3f"
                % (pregrasp, grasp, float(release_z))
            ),
        },
    }


def save_golden_pick_candidate(golden: Dict[str, Any], yaml_path: str) -> bool:
    path = str(yaml_path or "").strip()
    if not path or not isinstance(golden, dict):
        return False
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(golden, handle, default_flow_style=False, sort_keys=False)
    return True


def persist_sugar_demo_scene_02_golden_from_success(
    candidate: Dict[str, Any],
    *,
    scene_id: str,
    place_payload: Optional[Dict[str, Any]] = None,
    config_dir: Optional[str] = None,
) -> Tuple[bool, str]:
    """Escribe demo_scene_02_sugar_box_golden.yaml tras ejecución exitosa."""
    if not sugar_pick_golden_persist_eligible(scene_id):
        return False, "scene_not_eligible"
    if str(candidate.get("label", "")).strip().lower() != "sugar_box":
        return False, "not_sugar_box"
    yaml_path = default_golden_candidate_path(
        "demo_scene_02", "sugar_box", config_dir=config_dir
    )
    existing = load_golden_pick_candidate(yaml_path)
    golden = build_sugar_pick_golden_from_success(
        candidate,
        place_payload=place_payload,
        existing_golden=existing,
    )
    if not save_golden_pick_candidate(golden, yaml_path):
        return False, "save_failed"
    return True, yaml_path


def format_sugar_golden_persist_log(*, result: str, path: str = "", reason: str = "") -> str:
    return (
        "[SUGAR_BOX_GOLDEN_PERSIST]\n"
        "result=%s\n"
        "path=%s\n"
        "reason=%s"
        % (result, path or "n/a", reason or "n/a")
    )


def golden_fast_execute_available(
    scene_id: str,
    target_label: str,
    *,
    config_dir: Optional[str] = None,
) -> bool:
    """True si existe golden validado para scene_id+label (sin fallback a otra escena)."""
    sid = str(scene_id or "").strip().lower()
    label = str(target_label or "").strip().lower()
    if not sid or not golden_fast_execute_label_supported(label):
        return False
    yaml_path = default_golden_candidate_path(sid, label, config_dir=config_dir)
    golden = load_golden_pick_candidate(yaml_path)
    if golden is None:
        return False
    if str(golden.get("scene_id", "")).strip().lower() != sid:
        return False
    status = str(golden.get("status", "")).strip().lower()
    return status in ACCEPTED_GOLDEN_STATUSES


def chips_can_pick_golden_persist_eligible(scene_id: str) -> bool:
    return str(scene_id or "").strip().lower() == "chips_mustard_02"


def mustard_pick_golden_persist_eligible(scene_id: str) -> bool:
    sid = str(scene_id or "").strip().lower()
    if sid == "chips_mustard_02":
        return True
    return demo_golden_policy_scene_id(sid) == "demo_scene_02"


def build_mustard_pick_golden_from_success(
    candidate: Dict[str, Any],
    *,
    scene_id: str = "demo_scene_02",
    place_payload: Optional[Dict[str, Any]] = None,
    existing_golden: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Golden mustard tras pick+place OK (demo_scene_02 / deposit_* / chips_mustard_02)."""
    base = dict(existing_golden or {})
    sid = str(scene_id or "demo_scene_02").strip().lower()
    place_payload = place_payload if isinstance(place_payload, dict) else {}
    center = _normalize_xyz(
        candidate.get("grasp_center_base")
        or candidate.get("chosen_target_center_base")
        or candidate.get("position")
    )
    if center is None:
        pose = (base.get("object_pose") or {}) if isinstance(base, dict) else {}
        xy = pose.get("semantic_center_xy") or [0.662, 0.084]
        center = (float(xy[0]), float(xy[1]), float(pose.get("top_z", 0.4609)))
    top_z = _to_float_val(candidate.get("top_z_m"))
    if top_z is None:
        top_z = _to_float_val(candidate.get("top_z_estimated"))
    if top_z is None:
        top_z = float(center[2]) if len(center) > 2 else 0.4609

    pregrasp = _xyz_list_from_candidate(
        candidate, "_pregrasp_tcp_planning", "pregrasp_tcp"
    )
    grasp = _xyz_list_from_candidate(candidate, "_grasp_tcp_planning", "grasp_tcp")
    lift = _xyz_list_from_candidate(candidate, "_lift_tcp_planning", "lift_tcp")
    cand_base = (base.get("candidate") or {}) if isinstance(base, dict) else {}
    if pregrasp is None or grasp is None:
        pregrasp = list(cand_base.get("pregrasp_tcp") or [0.662, 0.084, 0.491])
        grasp = list(cand_base.get("grasp_tcp") or [0.662, 0.084, 0.427])
    if lift is None:
        lift = list(cand_base.get("lift_tcp") or [0.662, 0.084, 0.548])

    cmd_yaw = _to_float_val(candidate.get("_base_commanded_tcp_yaw_rad"))
    if cmd_yaw is None:
        cmd_yaw = _to_float_val(candidate.get("_final_tcp_yaw_rad"))
    if cmd_yaw is None:
        cmd_yaw = _to_float_val(cand_base.get("commanded_tcp_yaw_rad"))
    if cmd_yaw is None:
        scene_yaw = _to_float_val(
            ((base.get("object_pose") or {}) if isinstance(base, dict) else {}).get(
                "yaw_rad"
            )
        )
        cmd_yaw = scene_yaw if scene_yaw is not None else -3.0732

    depth = float(pregrasp[2]) - float(grasp[2])
    slot_idx = place_payload.get("place_slot_index")
    if slot_idx is None:
        slot_idx = candidate.get("place_slot_index")
    try:
        slot_index = int(slot_idx)
    except (TypeError, ValueError):
        slot_index = 3
    release_z = _to_float_val(place_payload.get("release_tcp_z"))
    if release_z is None:
        release_z = _to_float_val(candidate.get("_final_release_tcp_z"))
    if release_z is None:
        release_z = _to_float_val(
            ((base.get("place") or {}) if isinstance(base, dict) else {}).get(
                "release_tcp_z"
            )
        )
    if release_z is None:
        release_z = 0.3284

    dep_x = _to_float_val(place_payload.get("x"))
    dep_y = _to_float_val(place_payload.get("y"))
    if dep_x is None or dep_y is None:
        dep = ((base.get("place") or {}) if isinstance(base, dict) else {}).get(
            "deposit_xy"
        ) or [-0.540, -0.100]
        dep_x, dep_y = float(dep[0]), float(dep[1])

    transport = dict((base.get("transport") or {}) if isinstance(base, dict) else {})
    grasp_meta = dict((base.get("grasp") or {}) if isinstance(base, dict) else {})
    return {
        "scene_id": sid,
        "layout_version": str(
            base.get("layout_version") or "v3_clear_table_transport"
        ),
        "target_label": "mustard_bottle",
        "status": VALIDATED_STATUS,
        "object_pose": {
            "semantic_center_xy": [round(float(center[0]), 4), round(float(center[1]), 4)],
            "top_z": round(float(top_z), 4),
            "yaw_rad": float(cmd_yaw),
        },
        "candidate": {
            "candidate_idx": 0,
            "yaw_deg": round(math.degrees(float(cmd_yaw)), 2),
            "commanded_tcp_yaw_rad": round(float(cmd_yaw), 4),
            "pregrasp_tcp": pregrasp,
            "grasp_tcp": grasp,
            "lift_tcp": lift,
            "depth_from_top_m": round(float(depth), 4),
            "ik_seed": "pick_workspace_ready",
            "prevalidation_source": "geometric_fallback",
            "cartesian_descend_fraction": 1.0,
            "grasp_strategy": "tall_object_topdown",
            "joint7_rad": cand_base.get("joint7_rad", 1.1709),
        },
        "grasp": grasp_meta
        or {
            "strategy": "tall_object_topdown",
            "open_joint": 0.0399,
            "close_joint": 0.0220,
            "expected_width_m": 0.0440,
        },
        "transport": transport
        or {
            "selected_transport_entry": "direct_to_carry_front_high",
            "route": [
                "carry_mid_high",
                "turn_back_extended_aligned",
                "box_front_high",
                "box_high",
            ],
            "first_hub": "carry_mid_high",
            "backend": "direct_action",
        },
        "place": {
            "slot_index": int(slot_index),
            "slot_name": "slot_%d" % (int(slot_index) + 1),
            "deposit_xy": [round(float(dep_x), 3), round(float(dep_y), 3)],
            "release_tcp_z": round(float(release_z), 4),
            "release_tcp_z_candidates": [0.3084, 0.3284],
            "release_source": "golden_place",
            "approach_tcp_z": 0.650,
            "retreat_tcp_z": 0.650,
        },
        "validation": {
            "result": VALIDATED_STATUS,
            "golden_run_date": "2026-07-01",
            "prevalidation_source": "geometric_fallback",
            "validated_in_scenes": [
                "deposit_02_cracker_chips",
                "deposit_03_mustard_only",
                "demo_scene_02",
            ],
            "pick_place_pipeline_completed": True,
            "notes": (
                "Golden actualizado tras pick+place OK. pregrasp=%s grasp=%s release_z=%.3f"
                % (pregrasp, grasp, float(release_z))
            ),
        },
    }


def persist_mustard_demo_scene_02_golden_from_success(
    candidate: Dict[str, Any],
    *,
    scene_id: str,
    place_payload: Optional[Dict[str, Any]] = None,
    config_dir: Optional[str] = None,
) -> Tuple[bool, str]:
    if not mustard_pick_golden_persist_eligible(scene_id):
        return False, "scene_not_eligible"
    if str(candidate.get("label", "")).strip().lower() != "mustard_bottle":
        return False, "not_mustard_bottle"
    sid = str(scene_id or "").strip().lower()
    yaml_path = default_golden_candidate_path(
        sid, "mustard_bottle", config_dir=config_dir
    )
    existing = load_golden_pick_candidate(yaml_path)
    golden = build_mustard_pick_golden_from_success(
        candidate,
        scene_id=sid,
        place_payload=place_payload,
        existing_golden=existing,
    )
    if not save_golden_pick_candidate(golden, yaml_path):
        return False, "save_failed"
    return True, yaml_path


def format_mustard_golden_persist_log(*, result: str, path: str = "", reason: str = "") -> str:
    return (
        "[MUSTARD_GOLDEN_PERSIST]\n"
        "result=%s\n"
        "path=%s\n"
        "reason=%s"
        % (result, path or "n/a", reason or "n/a")
    )


def build_chips_can_pick_golden_from_success(
    candidate: Dict[str, Any],
    *,
    scene_id: str = "chips_mustard_02",
    place_payload: Optional[Dict[str, Any]] = None,
    existing_golden: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Golden chips_can tras pick+place OK en chips_mustard_02."""
    base = dict(existing_golden or {})
    sid = str(scene_id or "chips_mustard_02").strip().lower()
    place_payload = place_payload if isinstance(place_payload, dict) else {}
    center = _normalize_xyz(
        candidate.get("grasp_center_base")
        or candidate.get("chosen_target_center_base")
        or candidate.get("position")
    )
    if center is None:
        pose = (base.get("object_pose") or {}) if isinstance(base, dict) else {}
        xy = pose.get("semantic_center_xy") or [0.500, -0.080]
        center = (float(xy[0]), float(xy[1]), float(pose.get("top_z", 0.510)))
    top_z = _to_float_val(candidate.get("top_z_m"))
    if top_z is None:
        top_z = _to_float_val(candidate.get("top_z_estimated"))
    if top_z is None:
        top_z = float(center[2]) if len(center) > 2 else 0.510

    pregrasp = _xyz_list_from_candidate(
        candidate, "_pregrasp_tcp_planning", "pregrasp_tcp"
    )
    grasp = _xyz_list_from_candidate(candidate, "_grasp_tcp_planning", "grasp_tcp")
    lift = _xyz_list_from_candidate(candidate, "_lift_tcp_planning", "lift_tcp")
    cand_base = (base.get("candidate") or {}) if isinstance(base, dict) else {}
    if pregrasp is None or grasp is None:
        pregrasp = list(cand_base.get("pregrasp_tcp") or [0.500, -0.080, 0.545])
        grasp = list(cand_base.get("grasp_tcp") or [0.500, -0.080, 0.475])
    if lift is None:
        lift = list(cand_base.get("lift_tcp") or [0.500, -0.080, 0.625])

    cmd_yaw = _to_float_val(candidate.get("_base_commanded_tcp_yaw_rad"))
    if cmd_yaw is None:
        cmd_yaw = _to_float_val(candidate.get("_final_tcp_yaw_rad"))
    if cmd_yaw is None:
        cmd_yaw = _to_float_val(cand_base.get("commanded_tcp_yaw_rad"))
    if cmd_yaw is None:
        cmd_yaw = math.pi

    depth = float(pregrasp[2]) - float(grasp[2])
    slot_idx = place_payload.get("place_slot_index")
    if slot_idx is None:
        slot_idx = candidate.get("place_slot_index")
    try:
        slot_index = int(slot_idx)
    except (TypeError, ValueError):
        slot_index = 1
    release_z = _to_float_val(place_payload.get("release_tcp_z"))
    if release_z is None:
        release_z = _to_float_val(candidate.get("_final_release_tcp_z"))
    if release_z is None:
        release_z = _to_float_val(
            ((base.get("place") or {}) if isinstance(base, dict) else {}).get(
                "release_tcp_z"
            )
        )
    if release_z is None:
        release_z = 0.195

    transport = dict((base.get("transport") or {}) if isinstance(base, dict) else {})
    grasp_meta = dict((base.get("grasp") or {}) if isinstance(base, dict) else {})
    spawn_yaw = _to_float_val(candidate.get("spawn_yaw_rad"))
    if spawn_yaw is None:
        spawn_yaw = _to_float_val(
            ((base.get("object_pose") or {}) if isinstance(base, dict) else {}).get(
                "yaw_rad"
            )
        )
    if spawn_yaw is None:
        spawn_yaw = 0.0
    golden = {
        "scene_id": sid,
        "layout_version": str(
            base.get("layout_version") or "chips_mustard_02_easy_layout"
        ),
        "target_label": "chips_can",
        "status": VALIDATED_STATUS,
        "object_pose": {
            "semantic_center_xy": [round(float(center[0]), 4), round(float(center[1]), 4)],
            "top_z": round(float(top_z), 4),
            "yaw_rad": float(spawn_yaw),
        },
        "candidate": {
            "candidate_idx": int(
                candidate.get("candidate_idx", cand_base.get("candidate_idx", 0))
            ),
            "yaw_deg": round(math.degrees(float(cmd_yaw)), 2),
            "commanded_tcp_yaw_rad": round(float(cmd_yaw), 4),
            "pregrasp_tcp": pregrasp,
            "grasp_tcp": grasp,
            "lift_tcp": lift,
            "depth_from_top_m": round(float(depth), 4),
            "ik_seed": str(
                candidate.get("ik_seed")
                or cand_base.get("ik_seed")
                or "pick_workspace_ready"
            ),
            "prevalidation_source": str(
                candidate.get("_prevalidation_source")
                or cand_base.get("prevalidation_source")
                or "chips_can_legacy_pending_actual_tf_descend"
            ),
            "cartesian_descend_fraction": float(
                candidate.get("_cartesian_descend_fraction")
                or cand_base.get("cartesian_descend_fraction")
                or 1.0
            ),
        },
        "grasp": grasp_meta
        or {
            "strategy": "cylinder_topdown",
            "open_joint": 0.0399,
            "close_joint": 0.0270,
            "expected_width_m": 0.0600,
        },
        "transport": transport
        or {
            "selected_transport_entry": "vertical_raise_then_rear_retreat",
            "route": [
                "carry_mid_high",
                "turn_back_extended_aligned",
                "box_front_high",
                "box_high",
            ],
            "forbidden_waypoints": ["carry_front_high"],
            "first_hub": "carry_mid_high",
            "backend": "direct_action",
        },
        "place": {
            "slot_index": int(slot_index),
            "slot_name": "slot_%d" % (int(slot_index) + 1),
            "release_tcp_z": round(float(release_z), 4),
            "release_tcp_z_candidates": [0.195, 0.210, 0.225],
            "release_source": "golden_place",
            "approach_tcp_z": 0.400,
            "retreat_tcp_z": 0.450,
        },
        "validation": {
            "result": VALIDATED_STATUS,
            "pick_place_pipeline_completed": True,
            "notes": (
                "Golden chips_mustard_02 tras pick+place OK. pregrasp=%s grasp=%s release_z=%.3f"
                % (pregrasp, grasp, float(release_z))
            ),
        },
    }
    return enrich_chips_can_legacy_golden_fields(
        golden, grasp_xy=(float(center[0]), float(center[1]))
    )


def persist_chips_can_golden_from_success(
    candidate: Dict[str, Any],
    *,
    scene_id: str,
    place_payload: Optional[Dict[str, Any]] = None,
    config_dir: Optional[str] = None,
) -> Tuple[bool, str]:
    sid = str(scene_id or "").strip().lower()
    if not chips_can_pick_golden_persist_eligible(sid):
        return False, "scene_not_eligible"
    if str(candidate.get("label", "")).strip().lower() != "chips_can":
        return False, "not_chips_can"
    yaml_path = default_golden_candidate_path(sid, "chips_can", config_dir=config_dir)
    existing = load_golden_pick_candidate(yaml_path)
    golden = build_chips_can_pick_golden_from_success(
        candidate,
        scene_id=sid,
        place_payload=place_payload,
        existing_golden=existing,
    )
    if not save_golden_pick_candidate(golden, yaml_path):
        return False, "save_failed"
    return True, yaml_path


def format_chips_can_golden_persist_log(*, result: str, path: str = "", reason: str = "") -> str:
    return (
        "[CHIPS_CAN_GOLDEN_PERSIST]\n"
        "result=%s\n"
        "path=%s\n"
        "reason=%s"
        % (result, path or "n/a", reason or "n/a")
    )
