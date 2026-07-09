"""Carga y resolución de políticas de escena demo desde YAML."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import yaml

DEFAULT_OBSTACLE_DISTURBANCE_XY_THRESHOLD_M = 0.010
DEFAULT_OBSTACLE_DISTURBANCE_Z_THRESHOLD_M = 0.020
DEFAULT_ATTACHED_TRANSPORT_SAFETY_MARGIN_TOLERANCE_M = 0.006

_POLICY_CACHE: Dict[str, Dict[str, Any]] = {}


def default_demo_scenes_dir() -> str:
    try:
        from ament_index_python.packages import get_package_share_directory

        share = get_package_share_directory("panda_controller")
        return os.path.join(share, "config", "demo_scenes")
    except Exception:
        pass
    here = Path(__file__).resolve().parent.parent / "config" / "demo_scenes"
    return str(here)


def demo_scene_policy_yaml_path(scene_id: str, *, scenes_dir: Optional[str] = None) -> str:
    sid = str(scene_id or "").strip().lower()
    base = str(scenes_dir or default_demo_scenes_dir())
    return os.path.join(base, f"{sid}.yaml")


def load_demo_scene_policy(
    scene_id: str,
    *,
    scenes_dir: Optional[str] = None,
    use_cache: bool = True,
) -> Optional[Dict[str, Any]]:
    """Carga política de escena demo. Devuelve None si no existe el YAML."""
    sid = str(scene_id or "").strip().lower()
    if not sid:
        return None
    if use_cache and sid in _POLICY_CACHE:
        return dict(_POLICY_CACHE[sid])

    path = demo_scene_policy_yaml_path(sid, scenes_dir=scenes_dir)
    if not os.path.isfile(path):
        return None

    with open(path, "r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}

    policy = _normalize_demo_scene_policy(raw, source_file=path)
    if use_cache:
        _POLICY_CACHE[sid] = dict(policy)
    return policy


def _normalize_demo_scene_policy(raw: Dict[str, Any], *, source_file: str) -> Dict[str, Any]:
    sid = str(raw.get("scene_id", "")).strip().lower()
    objects_raw = raw.get("objects") or {}
    objects: Dict[str, Dict[str, Any]] = {}
    if isinstance(objects_raw, dict):
        for label, spec in objects_raw.items():
            lb = str(label).strip().lower()
            if not lb or not isinstance(spec, dict):
                continue
            pose = spec.get("pose") or {}
            objects[lb] = {
                "role": str(spec.get("role", "target")),
                "pose": {
                    "x": float(pose.get("x", 0.0)),
                    "y": float(pose.get("y", 0.0)),
                    "yaw": float(pose.get("yaw", 0.0)),
                },
                "preferred_slot": int(spec.get("preferred_slot", 0)),
                "seed": spec.get("seed"),
            }
    pick_order = [
        str(x).strip().lower()
        for x in (raw.get("pick_order") or [])
        if str(x).strip()
    ]
    tp = dict(raw.get("transport_policy") or {})
    safety = dict(raw.get("safety") or {})
    place_policy = dict(raw.get("place_policy") or {})
    transport_phases = dict(raw.get("transport_phases") or {})
    return {
        "scene_id": sid,
        "description": str(raw.get("description", "")),
        "pick_order": pick_order,
        "objects": objects,
        "transport_policy": {
            "forbidden_waypoints_when_obstacles_remaining": list(
                tp.get("forbidden_waypoints_when_obstacles_remaining") or []
            ),
            "local_exit_candidates": list(tp.get("local_exit_candidates") or []),
            "reconfiguration_waypoints": list(tp.get("reconfiguration_waypoints") or []),
            "transport_route": list(tp.get("transport_route") or []),
            "backend": str(tp.get("backend", "direct_action")),
            "validate_attached_swept_volume": bool(
                tp.get("validate_attached_swept_volume", True)
            ),
            "defer_named_hub_to_deterministic_transport": bool(
                tp.get("defer_named_hub_to_deterministic_transport", False)
            ),
            "use_lateral_transport_corridors": bool(
                tp.get("use_lateral_transport_corridors", False)
            ),
        },
        "safety": {
            "obstacle_disturbance_xy_threshold_m": float(
                safety.get(
                    "obstacle_disturbance_xy_threshold_m",
                    DEFAULT_OBSTACLE_DISTURBANCE_XY_THRESHOLD_M,
                )
            ),
            "obstacle_disturbance_z_threshold_m": float(
                safety.get(
                    "obstacle_disturbance_z_threshold_m",
                    DEFAULT_OBSTACLE_DISTURBANCE_Z_THRESHOLD_M,
                )
            ),
            "attached_transport_safety_margin_tolerance_m": float(
                safety.get(
                    "attached_transport_safety_margin_tolerance_m",
                    DEFAULT_ATTACHED_TRANSPORT_SAFETY_MARGIN_TOLERANCE_M,
                )
            ),
            "reconfiguration_min_table_clearance_m": float(
                safety.get("reconfiguration_min_table_clearance_m", 0.200)
            ),
            "reconfiguration_min_xy_clearance_m": float(
                safety.get("reconfiguration_min_xy_clearance_m", 0.080)
            ),
            "local_exit_required_clearance_m": float(
                safety.get("local_exit_required_clearance_m", 0.050)
            ),
            "local_exit_min_table_clearance_m": float(
                safety.get("local_exit_min_table_clearance_m", 0.200)
            ),
            "reconfiguration_required_clearance_m": float(
                safety.get("reconfiguration_required_clearance_m", 0.080)
            ),
            "global_route_required_clearance_m": float(
                safety.get("global_route_required_clearance_m", 0.100)
            ),
        },
        "place_policy": place_policy,
        "transport_phases": transport_phases,
        "source_file": source_file,
    }


def preferred_slot_map_from_scene_policy(
    scene_policy: Optional[Dict[str, Any]],
) -> Dict[str, int]:
    """Índice de slot por label desde objects.*.preferred_slot del YAML de escena."""
    if not isinstance(scene_policy, dict):
        return {}
    objects = scene_policy.get("objects") or {}
    out: Dict[str, int] = {}
    if not isinstance(objects, dict):
        return out
    for label, spec in objects.items():
        if not isinstance(spec, dict):
            continue
        lb = str(label or "").strip().lower()
        if not lb:
            continue
        try:
            out[lb] = int(spec.get("preferred_slot", -1))
        except (TypeError, ValueError):
            continue
    return {k: v for k, v in out.items() if v >= 0}


DEMO_SCENE_PLACEHOLDER_TRANSPORT_WAYPOINTS = frozenset(
    {
        "carry_front_high",
        "transport_ready_unwind",
        "transport_home_high",
        "elbow_unwind_high",
    }
)


def format_demo_scene_policy_load_log(policy: Dict[str, Any]) -> str:
    return (
        "[DEMO_SCENE_POLICY_LOAD]\n"
        "scene_id=%s\n"
        "file=%s\n"
        "pick_order=%s\n"
        "result=OK"
        % (
            str(policy.get("scene_id", "")),
            str(policy.get("source_file", "")),
            list(policy.get("pick_order") or []),
        )
    )


def format_demo_scene_object_pose_logs(policy: Dict[str, Any]) -> List[str]:
    out: List[str] = []
    for label in sorted((policy.get("objects") or {}).keys()):
        spec = (policy.get("objects") or {}).get(label) or {}
        pose = spec.get("pose") or {}
        out.append(
            "[DEMO_SCENE_OBJECT_POSE]\n"
            "scene_id=%s\n"
            "label=%s\n"
            "x=%.4f\n"
            "y=%.4f\n"
            "yaw=%.4f\n"
            "preferred_slot=%d\n"
            "role=%s"
            % (
                str(policy.get("scene_id", "")),
                label,
                float(pose.get("x", 0.0)),
                float(pose.get("y", 0.0)),
                float(pose.get("yaw", 0.0)),
                int(spec.get("preferred_slot", 0)),
                str(spec.get("role", "")),
            )
        )
    return out


def demo_scene_policy_to_spawn_entries(policy: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Entradas compatibles con runtime_scene_spawner legacy spec."""
    pick_order = list(policy.get("pick_order") or [])
    objects = policy.get("objects") or {}
    entries: List[Dict[str, Any]] = []
    for idx, label in enumerate(pick_order):
        spec = objects.get(label)
        if not spec:
            continue
        pose = spec.get("pose") or {}
        entry: Dict[str, Any] = {
            "label": label,
            "x": float(pose.get("x", 0.0)),
            "y": float(pose.get("y", 0.0)),
            "yaw": float(pose.get("yaw", 0.0)),
            "order_index": idx,
        }
        seed = spec.get("seed")
        if seed is not None:
            entry["seed"] = int(seed)
        entries.append(entry)
    for label, spec in sorted(objects.items()):
        if label in pick_order:
            continue
        pose = spec.get("pose") or {}
        entry = {
            "label": label,
            "x": float(pose.get("x", 0.0)),
            "y": float(pose.get("y", 0.0)),
            "yaw": float(pose.get("yaw", 0.0)),
        }
        seed = spec.get("seed")
        if seed is not None:
            entry["seed"] = int(seed)
        entries.append(entry)
    return entries


def resolve_pick_order_from_scene_policy(
    scene_policy: Optional[Dict[str, Any]],
    *,
    fallback: Sequence[str],
) -> List[str]:
    if scene_policy and scene_policy.get("pick_order"):
        return list(scene_policy["pick_order"])
    return list(fallback)


def has_remaining_table_obstacles(
    scene_obstacles: Sequence[Dict[str, Any]],
    *,
    target_label: str = "",
) -> bool:
    tgt = str(target_label or "").strip().lower()
    for obs in scene_obstacles:
        if bool(obs.get("is_target", False)):
            continue
        lb = str(obs.get("label", "")).strip().lower()
        if lb and lb != tgt:
            return True
    return False


def filter_transport_route_for_scene(
    sequence: Sequence[str],
    scene_policy: Optional[Dict[str, Any]],
    *,
    obstacles_remaining: bool,
) -> Tuple[List[str], List[str]]:
    """Filtra waypoints prohibidos. Devuelve (sequence_after, forbidden_applied)."""
    seq = list(sequence)
    if not scene_policy or not obstacles_remaining:
        return seq, []
    forbidden = list(
        (scene_policy.get("transport_policy") or {}).get(
            "forbidden_waypoints_when_obstacles_remaining"
        )
        or []
    )
    if not forbidden:
        return seq, []
    forbidden_set = {str(x).strip() for x in forbidden if str(x).strip()}
    filtered = [wp for wp in seq if wp not in forbidden_set]
    return filtered, sorted(forbidden_set)


def format_transport_route_filter_log(
    *,
    scene_id: str,
    forbidden_waypoints: Sequence[str],
    sequence_before: Sequence[str],
    sequence_after: Sequence[str],
) -> str:
    return (
        "[TRANSPORT_ROUTE_FILTER]\n"
        "scene_id=%s\n"
        "forbidden_waypoints=%s\n"
        "sequence_before=%s\n"
        "sequence_after=%s\n"
        "result=OK"
        % (scene_id, list(forbidden_waypoints), list(sequence_before), list(sequence_after))
    )


def format_transport_policy_selected_log(scene_policy: Dict[str, Any]) -> str:
    tp = scene_policy.get("transport_policy") or {}
    return (
        "[TRANSPORT_POLICY_SELECTED]\n"
        "scene_id=%s\n"
        "local_exit_candidates=%s\n"
        "transport_route=%s\n"
        "backend=%s\n"
        "defer_named_hub_to_deterministic_transport=%s\n"
        "result=OK"
        % (
            str(scene_policy.get("scene_id", "")),
            list(tp.get("local_exit_candidates") or []),
            list(tp.get("transport_route") or []),
            str(tp.get("backend", "direct_action")),
            str(bool(tp.get("defer_named_hub_to_deterministic_transport"))).lower(),
        )
    )


def apply_scene_policy_to_carry_transport(
    base: Dict[str, Any],
    scene_policy: Optional[Dict[str, Any]],
    *,
    obstacles_remaining: bool,
) -> Dict[str, Any]:
    if not scene_policy:
        return base
    out = dict(base)
    safety = scene_policy.get("safety") or {}
    tp = scene_policy.get("transport_policy") or {}
    out["attached_transport_safety_margin_tolerance_m"] = float(
        safety.get(
            "attached_transport_safety_margin_tolerance_m",
            out.get("attached_transport_safety_margin_tolerance_m", 0.006),
        )
    )
    out["obstacle_disturbance_xy_threshold_m"] = float(
        safety.get("obstacle_disturbance_xy_threshold_m", 0.010)
    )
    out["obstacle_disturbance_z_threshold_m"] = float(
        safety.get("obstacle_disturbance_z_threshold_m", 0.020)
    )
    from panda_controller.attached_transport_phases import (
        resolve_transport_phase_clearance_thresholds,
    )

    phase_clr = resolve_transport_phase_clearance_thresholds(scene_policy, out)
    out["local_exit_required_clearance_m"] = phase_clr["local_exit_required_clearance_m"]
    out["local_exit_min_table_clearance_m"] = phase_clr["local_exit_min_table_clearance_m"]
    out["reconfiguration_required_clearance_m"] = phase_clr[
        "reconfiguration_required_clearance_m"
    ]
    out["global_route_required_clearance_m"] = phase_clr["global_route_required_clearance_m"]
    out["require_local_escape_before_global_transport"] = bool(
        tp.get("require_local_escape_before_global_transport", True)
    )
    out["transport_backend"] = str(tp.get("backend", "direct_action"))
    out["local_exit_candidates"] = list(tp.get("local_exit_candidates") or [])
    out["transport_route"] = list(tp.get("transport_route") or [])
    out["defer_entry_hub_to_deterministic_transport"] = bool(
        tp.get("defer_named_hub_to_deterministic_transport", False)
    )
    out["use_lateral_transport_corridors"] = bool(
        tp.get("use_lateral_transport_corridors", False)
    )
    if obstacles_remaining:
        forbidden = list(tp.get("forbidden_waypoints_when_obstacles_remaining") or [])
        if forbidden:
            out["forbidden_transport_waypoints"] = forbidden
            if "carry_front_high" in forbidden:
                out["post_pick_skip_carry_front_high"] = True
            reconfig = list(tp.get("reconfiguration_waypoints") or [])
            route = list(tp.get("transport_route") or [])
            hub = reconfig[-1] if reconfig else (route[0] if route else "carry_mid_high")
            out["first_transport_waypoint_after_rear_retreat"] = str(hub)
            out["allow_direct_to_carry_front_high"] = "carry_front_high" not in forbidden
    return out


def resolve_post_pick_transport_entry_target_from_scene(
    scene_policy: Optional[Dict[str, Any]],
    carry_policy: Dict[str, Any],
    *,
    default_first_waypoint: str,
    waypoints_data: Optional[Dict[str, Any]] = None,
    obstacles_remaining: bool,
) -> Dict[str, Any]:
    skip_cf = bool(
        obstacles_remaining and carry_policy.get("post_pick_skip_carry_front_high")
    )
    if not skip_cf:
        entry_wp = str(default_first_waypoint)
        allow_cf = bool(carry_policy.get("allow_direct_to_carry_front_high", True))
        return {
            "skip_carry_front_high": False,
            "entry_target_waypoint": entry_wp,
            "allow_direct_to_carry_front_high": allow_cf,
            "allow_direct_to_entry_target": entry_wp != "carry_front_high" or allow_cf,
            "allow_carry_front_high_corridors": allow_cf,
            "removed_unsafe_first_waypoint": "",
            "defer_entry_hub_to_deterministic_transport": False,
        }
    reconfig = []
    route = []
    if scene_policy:
        tp = scene_policy.get("transport_policy") or {}
        reconfig = list(tp.get("reconfiguration_waypoints") or [])
        route = list(tp.get("transport_route") or [])
    hub = str(
        carry_policy.get("first_transport_waypoint_after_rear_retreat")
        or (reconfig[-1] if reconfig else "")
        or (route[0] if route else "carry_mid_high")
    )
    if waypoints_data is not None:
        try:
            from panda_controller.tfg_motion_waypoints import waypoint_is_configured

            for wp in reconfig + route:
                if waypoint_is_configured(waypoints_data, wp):
                    hub = str(wp)
                    break
        except Exception:
            pass
    defer = bool(carry_policy.get("defer_entry_hub_to_deterministic_transport", True))
    removed = "carry_front_high"
    forbidden = list(carry_policy.get("forbidden_transport_waypoints") or [])
    if forbidden:
        removed = forbidden[0]
    return {
        "skip_carry_front_high": True,
        "entry_target_waypoint": hub,
        "allow_direct_to_carry_front_high": False,
        "allow_direct_to_entry_target": str(hub) != "carry_front_high",
        "allow_carry_front_high_corridors": False,
        "removed_unsafe_first_waypoint": removed,
        "defer_entry_hub_to_deterministic_transport": defer,
    }


def obstacle_disturbance_thresholds(
    scene_policy: Optional[Dict[str, Any]],
) -> Tuple[float, float]:
    if not scene_policy:
        return (
            DEFAULT_OBSTACLE_DISTURBANCE_XY_THRESHOLD_M,
            DEFAULT_OBSTACLE_DISTURBANCE_Z_THRESHOLD_M,
        )
    safety = scene_policy.get("safety") or {}
    return (
        float(safety.get("obstacle_disturbance_xy_threshold_m", 0.010)),
        float(safety.get("obstacle_disturbance_z_threshold_m", 0.020)),
    )
