"""Resolución de top_z geométrico real para mustard_bottle (tall_object_topdown)."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

MUSTARD_EXPECTED_MIN_TOP_Z_M = 0.455
MUSTARD_DEFAULT_HEIGHT_M = 0.1909
MUSTARD_KNOWN_PHYSICAL_HEIGHT_M = 0.1909
DEMO_SCENE_02_MUSTARD_FALLBACK_TOP_Z_M = 0.4609
MUSTARD_SCANNER_CONTRACT_TOP_Z_M = 0.4609
MUSTARD_SCANNER_CONTRACT_PREGRASP_TCP_Z_M = 0.4909
# Profundidad operacional: dedos en tapón/cuello (~46 mm bajo top_z contrato).
MUSTARD_OPERATIONAL_GRASP_DEPTH_FROM_TOP_M = 0.046
MUSTARD_SCANNER_CONTRACT_GRASP_TCP_Z_M = (
    MUSTARD_SCANNER_CONTRACT_TOP_Z_M - MUSTARD_OPERATIONAL_GRASP_DEPTH_FROM_TOP_M
)
MUSTARD_SCANNER_Z_MATCH_TOLERANCE_M = 0.003
# Profundidad de grasp respecto a top_z del contrato scanner (reachability golden).
MUSTARD_SCANNER_DEPTH_FROM_TOP_M = (
    MUSTARD_SCANNER_CONTRACT_TOP_Z_M - MUSTARD_SCANNER_CONTRACT_GRASP_TCP_Z_M
)
MUSTARD_SCANNER_PREGRASP_MINUS_GRASP_M = (
    MUSTARD_SCANNER_CONTRACT_PREGRASP_TCP_Z_M - MUSTARD_SCANNER_CONTRACT_GRASP_TCP_Z_M
)
# Runtime corners pueden situar top_z ligeramente por encima del contrato SDF fijo.
MUSTARD_SCANNER_RUNTIME_TOP_SLACK_M = 0.015


def expected_mustard_grasp_tcp_z_for_runtime_top(top_z_m: float) -> float:
    """Grasp TCP coherente con top_z medido y profundidad del contrato scanner."""
    return float(top_z_m) - float(MUSTARD_SCANNER_DEPTH_FROM_TOP_M)

_RUNTIME_CORNER_KEYS = (
    "top_face_corners_base",
    "gt_top_face_corners_base",
    "runtime_gt_top_face_corners_base",
    "model_top_face_corners_base",
    "top_corners_base",
)

_RUNTIME_TOP_FACE_CENTER_KEYS = (
    "top_face_center_base",
    "gt_top_face_center_base",
    "runtime_gt_top_face_center_base",
)


def _to_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(v):
        return None
    return v


def _z_from_xyz_field(candidate: Dict[str, Any], key: str) -> Optional[float]:
    raw = candidate.get(key)
    if isinstance(raw, (list, tuple)) and len(raw) >= 3:
        return _to_float(raw[2])
    return None


def _max_corner_z(corners: Any) -> Optional[float]:
    if not isinstance(corners, list) or not corners:
        return None
    zs: List[float] = []
    for corner in corners:
        if isinstance(corner, (list, tuple)) and len(corner) >= 3:
            z = _to_float(corner[2])
            if z is not None:
                zs.append(float(z))
    return max(zs) if zs else None


def _iter_runtime_geometry_dicts(candidate: Dict[str, Any]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = [candidate]
    for obs in candidate.get("scene_obstacles") or []:
        if isinstance(obs, dict):
            out.append(obs)
    for scene_obj in candidate.get("_runtime_scene_objects") or []:
        if isinstance(scene_obj, dict):
            out.append(scene_obj)
    return out


def _resolve_runtime_top_z_m(candidate: Dict[str, Any]) -> Tuple[Optional[float], str]:
    """A) RuntimeScene top face / top corners si existen."""
    best_z: Optional[float] = None
    best_src = ""

    for src_dict in _iter_runtime_geometry_dicts(candidate):
        for key in _RUNTIME_CORNER_KEYS:
            z = _max_corner_z(src_dict.get(key))
            if z is None or z < MUSTARD_EXPECTED_MIN_TOP_Z_M:
                continue
            if best_z is None or z > best_z:
                best_z = float(z)
                best_src = "runtime_top_corners_%s" % str(key)

        for key in _RUNTIME_TOP_FACE_CENTER_KEYS:
            z = _z_from_xyz_field(src_dict, key)
            if z is None or z < MUSTARD_EXPECTED_MIN_TOP_Z_M:
                continue
            if best_z is None or z > best_z:
                best_z = float(z)
                best_src = "runtime_top_face_%s" % str(key)

    runtime_scene = _to_float(candidate.get("runtime_scene_top_z_m"))
    if runtime_scene is not None and runtime_scene >= MUSTARD_EXPECTED_MIN_TOP_Z_M:
        if best_z is None or runtime_scene > best_z:
            best_z = float(runtime_scene)
            best_src = "runtime_scene_top_z"

    return best_z, best_src


def _resolve_collision_top_z_m(
    candidate: Dict[str, Any],
) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """B) collision_pose_center_z + collision_height / 2."""
    collision_center_z: Optional[float] = None
    collision_height_m: Optional[float] = None
    collision_top_z: Optional[float] = None

    for src_dict in _iter_runtime_geometry_dicts(candidate):
        col_pose = src_dict.get("collision_box_pose")
        center_z: Optional[float] = None
        if isinstance(col_pose, dict):
            center_z = _to_float(col_pose.get("z"))
            if center_z is None:
                pos = col_pose.get("position")
                if isinstance(pos, (list, tuple)) and len(pos) >= 3:
                    center_z = _to_float(pos[2])
        if center_z is None:
            center_z = _z_from_xyz_field(src_dict, "semantic_box_center_base")
        if center_z is None:
            center_z = _z_from_xyz_field(src_dict, "gt_geometry_center_base")
        if center_z is None and isinstance(src_dict.get("position"), (list, tuple)):
            center_z = _z_from_xyz_field(src_dict, "position")

        col_dims = (
            src_dict.get("collision_dims_moveit")
            or src_dict.get("collision_dims_inflated")
            or src_dict.get("collision_dims")
        )
        height_m: Optional[float] = None
        if isinstance(col_dims, (list, tuple)) and len(col_dims) >= 3:
            height_m = _to_float(col_dims[2])
        if height_m is None:
            dims_lwh = src_dict.get("dims_lwh")
            if isinstance(dims_lwh, (list, tuple)) and len(dims_lwh) >= 3:
                height_m = _to_float(dims_lwh[2])

        if center_z is None or height_m is None or height_m <= 1e-6:
            continue
        top_z = float(center_z) + 0.5 * float(height_m)
        if collision_top_z is None or top_z > collision_top_z:
            collision_center_z = float(center_z)
            collision_height_m = float(height_m)
            collision_top_z = float(top_z)

    return collision_center_z, collision_height_m, collision_top_z


def mustard_tall_object_topdown_active(candidate: Dict[str, Any]) -> bool:
    return (
        str(candidate.get("label", "")).strip().lower() == "mustard_bottle"
        and str(candidate.get("grasp_strategy", "")).strip() == "tall_object_topdown"
    )


def resolve_mustard_geometry_top_z_m(
    candidate: Dict[str, Any],
    *,
    table_z_m: float,
    fallback_z: float,
    scene_id: Optional[str] = None,
) -> Tuple[float, str, Dict[str, Any]]:
    """Top físico del objeto; nunca cap center / grasp_center / height efectivo bajo."""
    position_z = _z_from_xyz_field(candidate, "position")
    grasp_center_z = _z_from_xyz_field(candidate, "grasp_center_base")
    candidate_top_before = _to_float(candidate.get("top_z_m"))

    runtime_top_z, runtime_top_z_source = _resolve_runtime_top_z_m(candidate)
    collision_center_z, collision_height_m, collision_top_z = _resolve_collision_top_z_m(
        candidate
    )

    table_top_z = float(table_z_m)
    known_sdf_height_m = float(MUSTARD_KNOWN_PHYSICAL_HEIGHT_M)
    known_sdf_top_z = table_top_z + known_sdf_height_m

    scene_l = str(scene_id or candidate.get("scene_id") or "").strip().lower()
    demo_scene_02 = scene_l == "demo_scene_02"

    selected: Optional[float] = None
    selected_src = ""

    if runtime_top_z is not None:
        selected = float(runtime_top_z)
        selected_src = str(runtime_top_z_source or "runtime_scene_top_z")
    elif (
        collision_top_z is not None
        and float(collision_top_z) >= MUSTARD_EXPECTED_MIN_TOP_Z_M
    ):
        selected = float(collision_top_z)
        selected_src = "collision_top_z"
    elif known_sdf_top_z >= MUSTARD_EXPECTED_MIN_TOP_Z_M:
        selected = float(known_sdf_top_z)
        selected_src = "known_sdf_physical_height"
    elif demo_scene_02:
        selected = float(DEMO_SCENE_02_MUSTARD_FALLBACK_TOP_Z_M)
        selected_src = "demo_scene_02_fallback"
    else:
        selected = float(fallback_z)
        selected_src = "fallback_z"

    if demo_scene_02 and float(selected) < MUSTARD_EXPECTED_MIN_TOP_Z_M:
        selected = float(DEMO_SCENE_02_MUSTARD_FALLBACK_TOP_Z_M)
        selected_src = "demo_scene_02_reject_low_top_z"

    meta = {
        "position_z": position_z,
        "grasp_center_z": grasp_center_z,
        "runtime_top_z": runtime_top_z,
        "runtime_top_z_source": runtime_top_z_source or None,
        "candidate_top_z_before": candidate_top_before,
        "collision_center_z": collision_center_z,
        "collision_height_m": collision_height_m,
        "collision_top_z": collision_top_z,
        "known_sdf_height_m": known_sdf_height_m,
        "table_top_z": table_top_z,
        "known_sdf_top_z": known_sdf_top_z,
        "demo_scene_02_fallback_top_z": (
            float(DEMO_SCENE_02_MUSTARD_FALLBACK_TOP_Z_M) if demo_scene_02 else None
        ),
        "scene_id": scene_l or None,
    }
    return float(selected), str(selected_src), meta


def format_mustard_top_z_source_debug_log(
    *,
    position_z: Optional[float],
    grasp_center_z: Optional[float],
    candidate_top_z_before: Optional[float],
    runtime_top_z: Optional[float],
    runtime_top_z_source: Optional[str],
    collision_center_z: Optional[float],
    collision_height_m: Optional[float],
    collision_top_z: Optional[float],
    known_sdf_height_m: Optional[float],
    table_top_z: Optional[float],
    known_sdf_top_z: Optional[float],
    selected_top_z: float,
    selected_top_z_source: str,
    result: str = "OK",
) -> str:
    def _f(v: Optional[float]) -> str:
        return "n/a" if v is None else "%.4f" % float(v)

    def _s(v: Optional[str]) -> str:
        return "n/a" if not v else str(v)

    return (
        "[MUSTARD_TOP_Z_SOURCE_DEBUG]\n"
        "label=mustard_bottle\n"
        "position_z=%s\n"
        "grasp_center_z=%s\n"
        "candidate_top_z_before=%s\n"
        "runtime_top_z=%s\n"
        "runtime_top_z_source=%s\n"
        "collision_center_z=%s\n"
        "collision_height_m=%s\n"
        "collision_top_z=%s\n"
        "known_sdf_height_m=%s\n"
        "table_top_z=%s\n"
        "known_sdf_top_z=%s\n"
        "selected_top_z=%.4f\n"
        "selected_top_z_source=%s\n"
        "result=%s"
        % (
            _f(position_z),
            _f(grasp_center_z),
            _f(candidate_top_z_before),
            _f(runtime_top_z),
            _s(runtime_top_z_source),
            _f(collision_center_z),
            _f(collision_height_m),
            _f(collision_top_z),
            _f(known_sdf_height_m),
            _f(table_top_z),
            _f(known_sdf_top_z),
            float(selected_top_z),
            str(selected_top_z_source),
            str(result),
        )
    )


def format_mustard_top_z_sanity_log(*, selected_top_z: float) -> str:
    return (
        "[MUSTARD_TOP_Z_SANITY]\n"
        "selected_top_z=%.4f\n"
        "expected_min_top_z=%.3f\n"
        "result=WARN_BAD_TOP_Z"
        % (float(selected_top_z), float(MUSTARD_EXPECTED_MIN_TOP_Z_M))
    )


def verify_mustard_scanner_aligned_z_contract(
    *,
    controller_top_z: float,
    controller_pregrasp_tcp_z: float,
    controller_grasp_tcp_z: float,
    tolerance_m: float = MUSTARD_SCANNER_Z_MATCH_TOLERANCE_M,
) -> Tuple[bool, Dict[str, Any]]:
    tol = float(tolerance_m)
    top_z = float(controller_top_z)
    pre_z = float(controller_pregrasp_tcp_z)
    grasp_z = float(controller_grasp_tcp_z)
    pre_ok = (
        abs(pre_z - MUSTARD_SCANNER_CONTRACT_PREGRASP_TCP_Z_M) <= tol
    )
    top_ok_strict = abs(top_z - MUSTARD_SCANNER_CONTRACT_TOP_Z_M) <= tol
    gr_ok_strict = (
        abs(grasp_z - MUSTARD_SCANNER_CONTRACT_GRASP_TCP_Z_M) <= tol
    )
    expected_grasp_rel = expected_mustard_grasp_tcp_z_for_runtime_top(top_z)
    gr_ok_relational = abs(grasp_z - expected_grasp_rel) <= tol
    descend_delta = pre_z - grasp_z
    top_runtime_ok = (
        top_z + 1e-9 >= float(MUSTARD_EXPECTED_MIN_TOP_Z_M)
        and top_z
        <= float(MUSTARD_SCANNER_CONTRACT_TOP_Z_M)
        + float(MUSTARD_SCANNER_RUNTIME_TOP_SLACK_M)
    )
    z_match_ok = bool(
        pre_ok
        and (
            (top_ok_strict and gr_ok_strict)
            or (top_runtime_ok and gr_ok_relational)
        )
    )
    verify_mode = (
        "strict_absolute"
        if (top_ok_strict and gr_ok_strict and pre_ok)
        else ("relational_runtime_top" if z_match_ok else "fail")
    )
    fields = {
        "controller_top_z": top_z,
        "controller_pregrasp_tcp_z": pre_z,
        "controller_grasp_tcp_z": grasp_z,
        "expected_scanner_top_z": float(MUSTARD_SCANNER_CONTRACT_TOP_Z_M),
        "expected_scanner_pregrasp_tcp_z": float(
            MUSTARD_SCANNER_CONTRACT_PREGRASP_TCP_Z_M
        ),
        "expected_scanner_grasp_tcp_z": float(MUSTARD_SCANNER_CONTRACT_GRASP_TCP_Z_M),
        "expected_grasp_tcp_z_for_runtime_top": float(expected_grasp_rel),
        "descend_delta_m": float(descend_delta),
        "expected_descend_delta_m": float(MUSTARD_SCANNER_PREGRASP_MINUS_GRASP_M),
        "verify_mode": verify_mode,
        "z_match_ok": z_match_ok,
        "result": "OK" if z_match_ok else "FAIL",
    }
    return z_match_ok, fields


def format_mustard_scanner_aligned_contract_verify_log(
    fields: Dict[str, Any],
) -> str:
    return (
        "[MUSTARD_SCANNER_ALIGNED_CONTRACT_VERIFY]\n"
        "controller_top_z=%.4f\n"
        "controller_pregrasp_tcp_z=%.4f\n"
        "controller_grasp_tcp_z=%.4f\n"
        "expected_scanner_top_z=%.4f\n"
        "expected_scanner_pregrasp_tcp_z=%.4f\n"
        "expected_scanner_grasp_tcp_z=%.4f\n"
        "expected_grasp_tcp_z_for_runtime_top=%s\n"
        "descend_delta_m=%s\n"
        "verify_mode=%s\n"
        "z_match_ok=%s\n"
        "result=%s"
        % (
            float(fields["controller_top_z"]),
            float(fields["controller_pregrasp_tcp_z"]),
            float(fields["controller_grasp_tcp_z"]),
            float(fields["expected_scanner_top_z"]),
            float(fields["expected_scanner_pregrasp_tcp_z"]),
            float(fields["expected_scanner_grasp_tcp_z"]),
            "n/a"
            if fields.get("expected_grasp_tcp_z_for_runtime_top") is None
            else "%.4f"
            % float(fields["expected_grasp_tcp_z_for_runtime_top"]),
            "n/a"
            if fields.get("descend_delta_m") is None
            else "%.4f" % float(fields["descend_delta_m"]),
            str(fields.get("verify_mode") or "n/a"),
            str(bool(fields.get("z_match_ok"))).lower(),
            str(fields.get("result", "FAIL")),
        )
    )
