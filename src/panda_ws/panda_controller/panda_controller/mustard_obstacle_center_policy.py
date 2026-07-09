"""Centro MoveIt para mustard_bottle cuando actúa como obstáculo (no target)."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

from panda_controller.mustard_xy_reachability_search import (
    DEMO_SCENE_02_MUSTARD_BODY_CENTER_XY,
)

MUSTARD_OBSTACLE_BODY_CENTER_KEYS: Tuple[str, ...] = (
    "known_object_center_base",
    "tall_object_body_center_base",
    "known_box_center_base",
    "gt_geometry_center_base",
    "semantic_box_center_base",
    "model_box_center_base",
)


def _xyz_from(obj: Dict[str, Any], key: str) -> Optional[Tuple[float, float, float]]:
    val = obj.get(key)
    if not isinstance(val, (list, tuple)) or len(val) < 3:
        return None
    try:
        return float(val[0]), float(val[1]), float(val[2])
    except (TypeError, ValueError):
        return None


def resolve_mustard_obstacle_center_base(
    scene_obj: Dict[str, Any],
) -> Tuple[Optional[Tuple[float, float, float]], str]:
    """Centro físico/cuerpo para colisión; nunca cap center operativo."""
    for key in MUSTARD_OBSTACLE_BODY_CENTER_KEYS:
        pos = _xyz_from(scene_obj, key)
        if pos is not None:
            return pos, key

    grasp = _xyz_from(scene_obj, "grasp_center_base") or _xyz_from(
        scene_obj, "chosen_target_center_base"
    )
    sem = _xyz_from(scene_obj, "semantic_center_base") or _xyz_from(scene_obj, "position")
    z_ref = (
        float(grasp[2])
        if grasp is not None
        else float(sem[2])
        if sem is not None
        else None
    )
    bx, by = (
        float(DEMO_SCENE_02_MUSTARD_BODY_CENTER_XY[0]),
        float(DEMO_SCENE_02_MUSTARD_BODY_CENTER_XY[1]),
    )
    if z_ref is not None:
        return (bx, by, float(z_ref)), "demo_scene_02_body_center_default"
    return None, "missing_body_center"


def apply_mustard_obstacle_center_for_planning_scene(
    scene_obj: Dict[str, Any],
    *,
    is_target: bool,
) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
    """Si mustard es obstáculo, reescribe position/semantic_center_base al body center."""
    label = str(scene_obj.get("label", "")).strip().lower()
    if label != "mustard_bottle" or bool(is_target):
        return scene_obj, None

    role = str(scene_obj.get("role", "")).strip().lower()
    if role == "target":
        return scene_obj, None

    center, center_source = resolve_mustard_obstacle_center_base(scene_obj)
    if center is None:
        return scene_obj, {
            "center_source": center_source,
            "obstacle_pose": None,
            "result": "FAIL",
        }

    out = dict(scene_obj)
    out["position"] = [float(center[0]), float(center[1]), float(center[2])]
    out["semantic_center_base"] = [
        float(center[0]),
        float(center[1]),
        float(center[2]),
    ]
    out["_mustard_obstacle_center_source"] = str(center_source)
    return out, {
        "center_source": "body_center",
        "position_source_key": str(center_source),
        "obstacle_pose": center,
        "result": "OK",
    }


def format_mustard_obstacle_center_policy_log(fields: Dict[str, Any]) -> str:
    pose = fields.get("obstacle_pose")
    if isinstance(pose, (list, tuple)) and len(pose) >= 3:
        pose_s = "(%.3f, %.3f, %.3f)" % (
            float(pose[0]),
            float(pose[1]),
            float(pose[2]),
        )
    else:
        pose_s = "n/a"
    return (
        "[MUSTARD_OBSTACLE_CENTER_POLICY]\n"
        "role=obstacle\n"
        "center_source=%s\n"
        "obstacle_pose=%s\n"
        "result=%s"
        % (
            str(fields.get("center_source", "body_center")),
            pose_s,
            str(fields.get("result", "FAIL")),
        )
    )
