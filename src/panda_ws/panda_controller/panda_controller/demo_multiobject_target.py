"""Pipeline multiobjeto demo: entity, obstáculos y fuentes GT para preflight."""

from __future__ import annotations

import math
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

DEMO_KNOWN_BOX_LABELS = frozenset(
    {
        "cracker_box",
        "sugar_box",
        "gelatin_box",
    }
)
CHIPS_CAN_LABEL = "chips_can"

EXPLICIT_TOP_FACE_SOURCES = frozenset(
    {
        "runtime_gt_known_box",
        "runtime_gt_known_object",
        "runtime_gt_tall_object",
        "runtime_gt_known_cylinder",
    }
)
EXPLICIT_GRASP_CENTER_SOURCES = frozenset(
    {
        "runtime_gt_box_center",
        "runtime_gt_object_center",
        "runtime_gt_tall_object_center",
        "runtime_gt_tall_object_cap_center",
        "runtime_gt_tall_object_cap_center_sdf_offset",
        "runtime_gt_mustard_top_cap_center_geometry",
        "runtime_gt_mustard_vertical_axis_cap_center",
        "runtime_gt_mustard_mesh_local_cap_center",
        "runtime_gt_tall_object_cap_center_calibrated",
        "runtime_gt_cylinder_center",
    }
)
EXPLICIT_YAW_SOURCES = frozenset({"runtime_gt_spawn_yaw"})
EXPLICIT_CLOSING_YAW_SOURCES = frozenset(
    {
        "runtime_gt_short_axis",
        "runtime_gt_known_object_short_axis",
        "runtime_gt_yaw_free",
        "runtime_gt_cylinder_axis",
        "runtime_gt_mustard_gap_axis_normal",
        "runtime_gt_mustard_gap_axis_swap_major_minor",
        "runtime_gt_mustard_gap_axis_yaw_offset_plus_90",
        "runtime_gt_mustard_gap_axis_yaw_offset_minus_90",
    }
)

BOX_TOP = "runtime_gt_known_box"
BOX_CENTER = "runtime_gt_box_center"
CYLINDER_TOP = "runtime_gt_known_cylinder"
CYLINDER_CENTER = "runtime_gt_cylinder_center"
SPAWN_YAW = "runtime_gt_spawn_yaw"
CLOSING_BOX = "runtime_gt_short_axis"
CLOSING_CYLINDER = "runtime_gt_cylinder_axis"

# Campos de visión/GT que no debe pisar export_grasp_policy_for_executor.
PRESERVE_RUNTIME_GT_KEYS = frozenset(
    {
        "top_face_source",
        "grasp_center_source",
        "yaw_source",
        "closing_yaw_source",
        "closing_yaw_rad",
        "model_closing_yaw_rad",
        "grasp_center_base",
        "known_box_center_base",
        "entity_name",
        "gt_entity_name",
        "object_yaw_rad",
        "known_box_yaw_rad",
        "model_box_yaw_rad",
        "grasp_yaw_rad",
        "major_axis_xy",
        "minor_axis_xy",
        "model_major_axis_xy",
        "model_minor_axis_xy",
    }
)


def _entity_short(name: str) -> str:
    return str(name or "").strip().split("::")[-1]


def _xyz_from_obj(obj: Dict[str, Any], *keys: str) -> Optional[Tuple[float, float, float]]:
    for key in keys:
        raw = obj.get(key)
        if isinstance(raw, (list, tuple)) and len(raw) >= 2:
            z = float(raw[2]) if len(raw) >= 3 else 0.0
            return (float(raw[0]), float(raw[1]), z)
    return None


def _label_lower(obj: Dict[str, Any]) -> str:
    return str(obj.get("label", "")).strip().lower()


def executor_objects_from_payload(
    payload: Optional[Dict[str, Any]],
    detections: Sequence[Any],
) -> List[Dict[str, Any]]:
    if isinstance(payload, dict):
        for key in ("objects", "detections", "results", "items"):
            raw = payload.get(key)
            if isinstance(raw, list):
                return [o for o in raw if isinstance(o, dict)]
    return [d for d in detections if isinstance(d, dict)]


def resolve_target_entity_for_candidate(
    candidate: Dict[str, Any],
    scene_objects: Sequence[Dict[str, Any]],
    executor_objects: Optional[Sequence[Dict[str, Any]]] = None,
    *,
    nearest_max_dist_m: float = 0.35,
) -> Tuple[bool, str, str]:
    """Resuelve entity del target en candidate."""
    label = _label_lower(candidate)
    cand_before = _entity_short(
        str(candidate.get("entity_name") or candidate.get("gt_entity_name") or "")
    )
    gt_before = _entity_short(str(candidate.get("gt_entity_name") or ""))

    for key in ("entity_name", "gt_entity_name"):
        ent = _entity_short(str(candidate.get(key) or ""))
        if ent:
            candidate["entity_name"] = ent
            candidate["gt_entity_name"] = ent
            return True, "candidate_entity", ent

    if executor_objects:
        matches: List[str] = []
        for obj in executor_objects:
            if not isinstance(obj, dict):
                continue
            if _label_lower(obj) != label:
                continue
            ent = _entity_short(
                str(obj.get("entity_name") or obj.get("gt_entity_name") or "")
            )
            if ent:
                matches.append(ent)
        unique = sorted(set(matches))
        if len(unique) == 1:
            candidate["entity_name"] = unique[0]
            candidate["gt_entity_name"] = unique[0]
            return True, "unique_label_executor", unique[0]
        if len(unique) > 1:
            pos = _xyz_from_obj(candidate, "grasp_center_base", "position")
            if pos is not None:
                best_ent = ""
                best_d = float("inf")
                for obj in executor_objects:
                    if _label_lower(obj) != label:
                        continue
                    ent = _entity_short(
                        str(obj.get("entity_name") or obj.get("gt_entity_name") or "")
                    )
                    opos = _xyz_from_obj(obj, "grasp_center_base", "position")
                    if ent and opos is not None:
                        d = math.hypot(pos[0] - opos[0], pos[1] - opos[1])
                        if d < best_d:
                            best_d = d
                            best_ent = ent
                if best_ent and best_d <= nearest_max_dist_m:
                    candidate["entity_name"] = best_ent
                    candidate["gt_entity_name"] = best_ent
                    return True, "nearest_xy_executor", best_ent

    pos = _xyz_from_obj(candidate, "grasp_center_base", "position")
    best_ent = ""
    best_dist = float("inf")
    if pos is not None and label:
        for so in scene_objects:
            if not isinstance(so, dict):
                continue
            if _label_lower(so) != label:
                continue
            sem = _xyz_from_obj(so, "semantic_center_base", "gt_geometry_center_base")
            if sem is None:
                continue
            dist = math.hypot(pos[0] - sem[0], pos[1] - sem[1])
            if dist < best_dist:
                best_dist = dist
                best_ent = _entity_short(str(so.get("entity_name", "")))

    if best_ent and best_dist <= float(nearest_max_dist_m):
        candidate["entity_name"] = best_ent
        candidate["gt_entity_name"] = best_ent
        return True, "nearest_xy_scene", best_ent

    label_matches: List[str] = []
    for so in scene_objects:
        if not isinstance(so, dict):
            continue
        if _label_lower(so) != label:
            continue
        ent = _entity_short(str(so.get("entity_name", "")))
        if ent:
            label_matches.append(ent)
    unique_scene = sorted(set(label_matches))
    if len(unique_scene) == 1:
        candidate["entity_name"] = unique_scene[0]
        candidate["gt_entity_name"] = unique_scene[0]
        return True, "unique_label_scene", unique_scene[0]

    if gt_before:
        candidate["entity_name"] = gt_before
        candidate["gt_entity_name"] = gt_before
        return True, "gt_entity", gt_before
    if cand_before:
        candidate["entity_name"] = cand_before
        candidate["gt_entity_name"] = cand_before
        return True, "candidate_entity_legacy", cand_before

    return False, "fail", ""


def _det_is_target(
    det: Dict[str, Any],
    candidate: Dict[str, Any],
    *,
    xy_tol_m: float = 0.08,
) -> bool:
    target_label = _label_lower(candidate)
    target_entity = _entity_short(
        str(candidate.get("gt_entity_name") or candidate.get("entity_name") or "")
    )
    det_label = _label_lower(det)
    det_ent = _entity_short(str(det.get("entity_name") or det.get("gt_entity_name") or ""))
    if target_entity and det_ent and det_ent == target_entity:
        return True
    if target_label and det_label == target_label:
        if not target_entity or not det_ent or det_ent == target_entity:
            cpos = _xyz_from_obj(candidate, "grasp_center_base", "position")
            dpos = _xyz_from_obj(det, "grasp_center_base", "position")
            if cpos is None or dpos is None:
                return True
            if math.hypot(cpos[0] - dpos[0], cpos[1] - dpos[1]) <= xy_tol_m:
                return True
    return False


def filter_scene_obstacles_for_target(
    candidate: Dict[str, Any],
    scene_obstacles: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Elimina el target de la lista de obstáculos MoveIt."""
    target_entity = _entity_short(
        str(candidate.get("gt_entity_name") or candidate.get("entity_name") or "")
    )
    target_label = _label_lower(candidate)
    removed: List[str] = []
    kept: List[Dict[str, Any]] = []
    for obs in scene_obstacles:
        if not isinstance(obs, dict):
            continue
        ent = _entity_short(str(obs.get("entity_name") or ""))
        lb = _label_lower(obs)
        drop = False
        if bool(obs.get("is_target", False)):
            drop = True
        elif target_entity and ent and ent == target_entity:
            drop = True
        elif target_label and lb == target_label:
            drop = True
        if drop:
            removed.append(ent or lb or "unknown")
            continue
        kept.append(obs)
    return kept, removed


def _ensure_closing_yaw_rad(candidate: Dict[str, Any]) -> None:
    if candidate.get("closing_yaw_rad") is not None:
        return
    for key in ("model_closing_yaw_rad", "object_yaw_rad", "grasp_yaw_rad"):
        val = candidate.get(key)
        if val is not None:
            try:
                candidate["closing_yaw_rad"] = float(val)
                return
            except (TypeError, ValueError):
                pass


def normalize_demo_grasp_sources(candidate: Dict[str, Any]) -> Tuple[bool, str]:
    """Rellena closing_yaw_source para payloads GT explícitos (cajas y chips_can)."""
    label = _label_lower(candidate)
    old = str(candidate.get("closing_yaw_source", "")).strip()
    if old:
        return False, old

    _ensure_closing_yaw_rad(candidate)
    if candidate.get("closing_yaw_rad") is None:
        return False, ""

    top = str(candidate.get("top_face_source", "")).strip()
    center = str(candidate.get("grasp_center_source", "")).strip()
    yaw = str(candidate.get("yaw_source", "")).strip()

    if (
        label in DEMO_KNOWN_BOX_LABELS
        and top == BOX_TOP
        and center == BOX_CENTER
        and yaw == SPAWN_YAW
    ):
        candidate["closing_yaw_source"] = CLOSING_BOX
        return True, CLOSING_BOX

    if (
        label == CHIPS_CAN_LABEL
        and top == CYLINDER_TOP
        and center == CYLINDER_CENTER
        and yaw == SPAWN_YAW
    ):
        candidate["closing_yaw_source"] = CLOSING_CYLINDER
        return True, CLOSING_CYLINDER

    return False, ""


def demo_grasp_policy_sources_ok(candidate: Dict[str, Any]) -> bool:
    """True si las fuentes explícitas runtime GT son válidas para demo."""
    if bool(candidate.get("operational_source_fallback", False)):
        return False
    normalize_demo_grasp_sources(candidate)
    return (
        str(candidate.get("top_face_source", "")) in EXPLICIT_TOP_FACE_SOURCES
        and str(candidate.get("grasp_center_source", "")) in EXPLICIT_GRASP_CENTER_SOURCES
        and str(candidate.get("yaw_source", "")) in EXPLICIT_YAW_SOURCES
        and str(candidate.get("closing_yaw_source", "")) in EXPLICIT_CLOSING_YAW_SOURCES
    )


def _has_explicit_runtime_gt_pose_sources(entry: Dict[str, Any]) -> bool:
    return (
        str(entry.get("top_face_source", "")).strip() in EXPLICIT_TOP_FACE_SOURCES
        and str(entry.get("grasp_center_source", "")).strip()
        in EXPLICIT_GRASP_CENTER_SOURCES
        and str(entry.get("yaw_source", "")).strip() in EXPLICIT_YAW_SOURCES
    )


def merge_grasp_policy_preserving_runtime_gt(
    entry: Dict[str, Any],
    policy_fields: Dict[str, Any],
) -> Dict[str, Any]:
    """Fusiona política DB sin pisar fuentes/poses ya publicadas por visión."""
    for key, value in policy_fields.items():
        if key in PRESERVE_RUNTIME_GT_KEYS:
            existing = entry.get(key)
            if existing is not None and str(existing).strip() != "":
                continue
            if key == "closing_yaw_source" and not str(existing or "").strip():
                if entry.get("closing_yaw_rad") is not None and _has_explicit_runtime_gt_pose_sources(
                    entry
                ):
                    continue
        entry[key] = value
    return entry


def log_target_entity_resolve(
    logger: Any,
    *,
    label: str,
    candidate_entity_before: str,
    gt_entity_before: str,
    resolved_entity: str,
    method: str,
    ok: bool,
) -> None:
    if logger is None:
        return
    try:
        logger.info(
            "[TARGET_ENTITY_RESOLVE]\n"
            "label=%s\n"
            "candidate_entity_before=%s\n"
            "gt_entity_before=%s\n"
            "resolved_entity=%s\n"
            "method=%s\n"
            "result=%s"
            % (
                label,
                candidate_entity_before or "n/a",
                gt_entity_before or "n/a",
                resolved_entity or "n/a",
                method,
                "OK" if ok else "FAIL",
            )
        )
    except Exception:
        pass


def log_target_obstacle_filter(
    logger: Any,
    *,
    target_label: str,
    target_entity: str,
    obstacles_before: Sequence[str],
    obstacles_after: Sequence[str],
    removed_target: Sequence[str],
) -> None:
    if logger is None:
        return
    try:
        logger.info(
            "[TARGET_OBSTACLE_FILTER]\n"
            "target_label=%s\n"
            "target_entity=%s\n"
            "obstacles_before=%s\n"
            "obstacles_after=%s\n"
            "removed_target=%s\n"
            "result=%s"
            % (
                target_label,
                target_entity or "n/a",
                list(obstacles_before),
                list(obstacles_after),
                list(removed_target),
                "OK",
            )
        )
    except Exception:
        pass


def log_grasp_policy_source_normalize(
    logger: Any,
    *,
    label: str,
    old_closing_yaw_source: str,
    new_closing_yaw_source: str,
    top_face_source: str,
    grasp_center_source: str,
    yaw_source: str,
    applied: bool,
) -> None:
    if logger is None:
        return
    try:
        logger.info(
            "[GRASP_POLICY_SOURCE_NORMALIZE]\n"
            "label=%s\n"
            "old_closing_yaw_source=%s\n"
            "new_closing_yaw_source=%s\n"
            "top_face_source=%s\n"
            "grasp_center_source=%s\n"
            "yaw_source=%s\n"
            "result=%s"
            % (
                label,
                old_closing_yaw_source or "",
                new_closing_yaw_source or "",
                top_face_source,
                grasp_center_source,
                yaw_source,
                "APPLIED" if applied else "OK",
            )
        )
    except Exception:
        pass


def apply_demo_multiobject_target_pipeline(
    candidate: Dict[str, Any],
    *,
    payload: Optional[Dict[str, Any]],
    scene_objects: Sequence[Dict[str, Any]],
    executor_objects: Sequence[Dict[str, Any]],
    logger: Any = None,
) -> None:
    """Resuelve entity y normaliza fuentes GT en el candidato."""
    label = str(candidate.get("label", "")).strip()
    cand_before = _entity_short(
        str(candidate.get("entity_name") or candidate.get("gt_entity_name") or "")
    )
    gt_before = _entity_short(str(candidate.get("gt_entity_name") or ""))

    ok, method, resolved = resolve_target_entity_for_candidate(
        candidate,
        scene_objects,
        executor_objects,
    )
    log_target_entity_resolve(
        logger,
        label=label,
        candidate_entity_before=cand_before,
        gt_entity_before=gt_before,
        resolved_entity=resolved,
        method=method,
        ok=ok,
    )

    old_cys = str(candidate.get("closing_yaw_source", "")).strip()
    applied, new_cys = normalize_demo_grasp_sources(candidate)
    if applied or (old_cys != new_cys and new_cys):
        log_grasp_policy_source_normalize(
            logger,
            label=label,
            old_closing_yaw_source=old_cys,
            new_closing_yaw_source=str(candidate.get("closing_yaw_source", "")),
            top_face_source=str(candidate.get("top_face_source", "")),
            grasp_center_source=str(candidate.get("grasp_center_source", "")),
            yaw_source=str(candidate.get("yaw_source", "")),
            applied=applied,
        )
