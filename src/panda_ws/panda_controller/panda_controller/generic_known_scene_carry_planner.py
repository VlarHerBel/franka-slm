"""Planner genérico scene-aware para transporte post-pick con objeto attached."""

from __future__ import annotations

import math
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from panda_controller.demo_scene_policy import (
    apply_scene_policy_to_carry_transport,
    has_remaining_table_obstacles,
    resolve_post_pick_transport_entry_target_from_scene,
)
from panda_controller.target_collision_policy import object_collision_radius_xy

DEFAULT_MIN_PICK_LIFT_M = 0.150
DEFAULT_MIN_CARRY_HAND_Z_M = 0.700
DEFAULT_MAX_CARRY_HAND_Z_M = 0.800
DEFAULT_CARRY_CLEARANCE_ABOVE_TABLE_M = 0.120
DEFAULT_CARRY_CLEARANCE_ABOVE_OBSTACLES_M = 0.100
DEFAULT_ATTACHED_COLLISION_PADDING_M = 0.020
DEFAULT_TRANSPORT_EXIT_LANE_Y_M = -0.32
DEFAULT_HAND_TO_TCP_Z_M = 0.100
DEFAULT_OBSTACLE_CLEARANCE_MODE = "swept_volume"
DEFAULT_CARRY_HEIGHT_INCREMENT_M = 0.030
DEFAULT_ATTACHED_TRANSPORT_SAFETY_MARGIN_TOLERANCE_M = 0.006
MIN_TRANSPORT_ENTRY_DELTA_XY_M = 0.080
CARRY_SAFE_HEIGHT_RETRY_STEP_M = 0.050
ADAPTIVE_CARRY_HEIGHT_DELTAS_M = (0.0, 0.030, 0.050)

CARRY_POLICY_BY_LABEL: Dict[str, Dict[str, Any]] = {
    "cracker_box": {
        "min_pick_lift_m": 0.150,
        "min_carry_tcp_z_m": 0.700,
        "max_carry_hand_z_m": 0.800,
        "carry_clearance_above_table_m": 0.120,
        "carry_clearance_above_obstacles_m": 0.100,
        "attached_collision_padding_m": 0.020,
        "preferred_transport_corridor": "front_lane",
        "obstacle_clearance_mode": "swept_volume",
        "carry_height_increment_m": 0.030,
    },
    "chips_can": {
        "min_pick_lift_m": 0.150,
        "min_carry_tcp_z_m": 0.700,
        "max_carry_hand_z_m": 0.800,
        "carry_clearance_above_table_m": 0.120,
        "carry_clearance_above_obstacles_m": 0.100,
        "attached_collision_padding_m": 0.020,
        "preferred_transport_corridor": "front_lane",
        "obstacle_clearance_mode": "swept_volume",
        "carry_height_increment_m": 0.030,
    },
    "sugar_box": {
        "min_pick_lift_m": 0.150,
        "min_carry_tcp_z_m": 0.700,
        "max_carry_hand_z_m": 0.800,
        "carry_clearance_above_table_m": 0.120,
        "carry_clearance_above_obstacles_m": 0.100,
        "attached_collision_padding_m": 0.020,
        "preferred_transport_corridor": "front_lane",
        "obstacle_clearance_mode": "swept_volume",
        "carry_height_increment_m": 0.030,
    },
    "mustard_bottle": {
        "min_pick_lift_m": 0.150,
        "min_carry_tcp_z_m": 0.700,
        "max_carry_hand_z_m": 0.800,
        "carry_clearance_above_table_m": 0.120,
        "carry_clearance_above_obstacles_m": 0.100,
        "attached_collision_padding_m": 0.020,
        "preferred_transport_corridor": "front_lane",
        "obstacle_clearance_mode": "swept_volume",
        "carry_height_increment_m": 0.030,
    },
}


def resolve_carry_transport_policy(candidate: Dict[str, Any]) -> Dict[str, Any]:
    label = str(candidate.get("label", "")).strip().lower()
    base = dict(CARRY_POLICY_BY_LABEL.get(label, {}))
    for key in (
        "min_pick_lift_m",
        "min_carry_tcp_z_m",
        "min_carry_hand_z_m",
        "max_carry_hand_z_m",
        "carry_clearance_above_table_m",
        "carry_clearance_above_obstacles_m",
        "attached_collision_padding_m",
        "preferred_transport_corridor",
        "transport_exit_lane_y_m",
        "obstacle_clearance_mode",
        "carry_height_increment_m",
        "vertical_clearance_mode",
        "attached_transport_safety_margin_tolerance_m",
        "use_lateral_transport_corridors",
    ):
        if candidate.get(key) is not None:
            base[key] = candidate.get(key)
    scene_id = str(candidate.get("scene_id", "")).strip().lower()
    scene_policy = candidate.get("_scene_policy")
    if scene_id in ("demo_scene_02", "demo_scene_2"):
        base.setdefault("max_carry_hand_z_m", DEFAULT_MAX_CARRY_HAND_Z_M)
        base.setdefault("obstacle_clearance_mode", DEFAULT_OBSTACLE_CLEARANCE_MODE)
        base.setdefault("carry_height_increment_m", DEFAULT_CARRY_HEIGHT_INCREMENT_M)
        base.setdefault(
            "attached_transport_safety_margin_tolerance_m",
            DEFAULT_ATTACHED_TRANSPORT_SAFETY_MARGIN_TOLERANCE_M,
        )
    base.setdefault("min_pick_lift_m", DEFAULT_MIN_PICK_LIFT_M)
    base.setdefault("min_carry_tcp_z_m", DEFAULT_MIN_CARRY_HAND_Z_M)
    base.setdefault("min_carry_hand_z_m", base["min_carry_tcp_z_m"])
    base.setdefault("max_carry_hand_z_m", DEFAULT_MAX_CARRY_HAND_Z_M)
    base.setdefault("carry_clearance_above_table_m", DEFAULT_CARRY_CLEARANCE_ABOVE_TABLE_M)
    base.setdefault(
        "carry_clearance_above_obstacles_m", DEFAULT_CARRY_CLEARANCE_ABOVE_OBSTACLES_M
    )
    base.setdefault("attached_collision_padding_m", DEFAULT_ATTACHED_COLLISION_PADDING_M)
    base.setdefault("preferred_transport_corridor", "front_lane")
    base.setdefault("transport_exit_lane_y_m", DEFAULT_TRANSPORT_EXIT_LANE_Y_M)
    base.setdefault("obstacle_clearance_mode", DEFAULT_OBSTACLE_CLEARANCE_MODE)
    base.setdefault("carry_height_increment_m", DEFAULT_CARRY_HEIGHT_INCREMENT_M)
    base.setdefault("vertical_clearance_mode", "corridor_swept_volume")
    base.setdefault(
        "attached_transport_safety_margin_tolerance_m",
        DEFAULT_ATTACHED_TRANSPORT_SAFETY_MARGIN_TOLERANCE_M,
    )
    base.setdefault("use_lateral_transport_corridors", True)
    obstacles = candidate.get("scene_obstacles") or []
    remaining = has_remaining_table_obstacles(obstacles, target_label=label)
    base = apply_scene_policy_to_carry_transport(
        base,
        scene_policy if isinstance(scene_policy, dict) else None,
        obstacles_remaining=remaining,
    )
    return base


def resolve_post_pick_transport_entry_target(
    candidate: Dict[str, Any],
    policy: Dict[str, Any],
    *,
    default_first_waypoint: str,
    waypoints_data: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Resuelve el primer waypoint seguro tras transport_entry (hub vs carry_front_high)."""
    scene_policy = candidate.get("_scene_policy")
    obstacles = candidate.get("scene_obstacles") or []
    label = str(candidate.get("label", "")).strip().lower()
    remaining = has_remaining_table_obstacles(obstacles, target_label=label)
    return resolve_post_pick_transport_entry_target_from_scene(
        scene_policy if isinstance(scene_policy, dict) else None,
        policy,
        default_first_waypoint=default_first_waypoint,
        waypoints_data=waypoints_data,
        obstacles_remaining=remaining,
    )


def should_defer_transport_entry_hub_to_deterministic(
    entry_cfg: Dict[str, Any],
    selected_mode: str,
) -> bool:
    """Si True, el hub (p.ej. carry_mid_high) lo ejecuta direct_action, no MoveIt en transport_entry."""
    if not bool(entry_cfg.get("skip_carry_front_high")):
        return False
    mode = str(selected_mode or "")
    if mode == "direct_to_carry_front_high":
        return False
    if "rear_retreat" in mode:
        return True
    return bool(entry_cfg.get("defer_entry_hub_to_deterministic_transport", False))


def resolve_hand_tcp_frame(
    *,
    current_hand: Tuple[float, float, float],
    current_tcp: Optional[Tuple[float, float, float]],
    candidate: Dict[str, Any],
) -> Dict[str, float]:
    hand_z = float(current_hand[2])
    if current_tcp is not None:
        tcp_z = float(current_tcp[2])
        hand_to_tcp_z = hand_z - tcp_z
    else:
        hand_to_tcp_z = DEFAULT_HAND_TO_TCP_Z_M
        try:
            ht = candidate.get("panda_hand_to_grasp_tcp_z_m")
            if ht is not None:
                hand_to_tcp_z = float(ht)
        except (TypeError, ValueError):
            pass
        tcp_z = hand_z - hand_to_tcp_z
    return {
        "current_hand_z": hand_z,
        "current_grasp_tcp_z": tcp_z,
        "hand_to_tcp_z": hand_to_tcp_z,
    }


def hand_z_from_tcp_z(tcp_z: float, hand_to_tcp_z: float) -> float:
    return float(tcp_z) + float(hand_to_tcp_z)


def tcp_z_from_hand_z(hand_z: float, hand_to_tcp_z: float) -> float:
    return float(hand_z) - float(hand_to_tcp_z)


def _object_height_m(candidate: Dict[str, Any]) -> float:
    for key in ("effective_height_m", "object_height_m", "db_height_m"):
        try:
            val = float(candidate.get(key))
            if val > 1e-4:
                return val
        except (TypeError, ValueError):
            pass
    dims = candidate.get("dims_lwh")
    if isinstance(dims, (list, tuple)) and len(dims) >= 3:
        try:
            return float(max(float(dims[0]), float(dims[1]), float(dims[2])))
        except (TypeError, ValueError):
            pass
    return 0.210


def _resolve_carried_object_yaw_rad(candidate: Dict[str, Any]) -> float:
    for key in ("grasp_yaw_rad", "object_yaw_rad"):
        try:
            val = candidate.get(key)
            if val is not None:
                return float(val)
        except (TypeError, ValueError):
            continue
    pose = candidate.get("pose") or {}
    try:
        return float(pose.get("yaw", 0.0))
    except (TypeError, ValueError):
        return 0.0


def resolve_carried_object_center_xy(
    hand_pos: Tuple[float, float, float],
    attached_geom: Dict[str, Any],
) -> Tuple[float, float, str]:
    offset = attached_geom.get("carried_object_center_offset_xy_m")
    if isinstance(offset, (list, tuple)) and len(offset) >= 2:
        return (
            float(hand_pos[0]) + float(offset[0]),
            float(hand_pos[1]) + float(offset[1]),
            str(attached_geom.get("carried_center_source", "attached_offset_from_grasp")),
        )
    return (
        float(hand_pos[0]),
        float(hand_pos[1]),
        str(attached_geom.get("carried_center_source", "fk_hand_offset")),
    )


def carried_effective_radius_xy_toward(
    attached_geom: Dict[str, Any],
    direction_xy_unit: Tuple[float, float],
) -> float:
    pad = float(attached_geom.get("attached_collision_padding_m", 0.0))
    hm = float(attached_geom.get("carried_object_half_major_xy_m", 0.0))
    hn = float(attached_geom.get("carried_object_half_minor_xy_m", 0.0))
    if hm <= 0.0:
        return float(attached_geom.get("carried_object_radius_xy_m", 0.08))
    if hn <= 0.0:
        hn = hm
    ux, uy = float(direction_xy_unit[0]), float(direction_xy_unit[1])
    norm = math.hypot(ux, uy)
    if norm < 1e-9:
        return float(hm + pad)
    ux, uy = ux / norm, uy / norm
    yaw = float(attached_geom.get("carried_object_yaw_rad", 0.0))
    ca, sa = math.cos(yaw), math.sin(yaw)
    proj = hm * abs(ux * ca + uy * sa) + hn * abs(ux * (-sa) + uy * ca)
    return float(proj + pad)


def carried_footprint_aabb_xy(
    hand_pos: Tuple[float, float, float],
    attached_geom: Dict[str, Any],
) -> Tuple[float, float, float, float]:
    cx, cy, _src = resolve_carried_object_center_xy(hand_pos, attached_geom)
    hm = float(attached_geom.get("carried_object_half_major_xy_m", 0.0))
    hn = float(attached_geom.get("carried_object_half_minor_xy_m", 0.0))
    pad = float(attached_geom.get("attached_collision_padding_m", 0.0))
    if hm <= 0.0:
        legacy_r = float(attached_geom.get("carried_object_radius_xy_m", 0.08))
        return (cx - legacy_r, cx + legacy_r, cy - legacy_r, cy + legacy_r)
    if hn <= 0.0:
        hn = hm
    yaw = float(attached_geom.get("carried_object_yaw_rad", 0.0))
    ca, sa = math.cos(yaw), math.sin(yaw)
    corners: List[Tuple[float, float]] = []
    for sx in (-1.0, 1.0):
        for sy in (-1.0, 1.0):
            lx = sx * (hm + pad)
            ly = sy * (hn + pad)
            wx = cx + lx * ca - ly * sa
            wy = cy + lx * sa + ly * ca
            corners.append((wx, wy))
    xs = [c[0] for c in corners]
    ys = [c[1] for c in corners]
    return (min(xs), max(xs), min(ys), max(ys))


def compute_attached_object_geometry(
    candidate: Dict[str, Any],
    *,
    grasp_hand_z: Optional[float] = None,
    grasp_hand_xy: Optional[Tuple[float, float]] = None,
    table_top_z: float,
) -> Dict[str, Any]:
    policy = resolve_carry_transport_policy(candidate)
    height_m = _object_height_m(candidate)
    grasp_depth = candidate.get("recommended_grasp_depth_from_top_m")
    if grasp_depth is None:
        grasp_depth = candidate.get("insertion_depth_limit_m")
    try:
        grasp_depth_f = float(grasp_depth) if grasp_depth is not None else 0.040
    except (TypeError, ValueError):
        grasp_depth_f = 0.040
    top_z = candidate.get("top_z_m")
    try:
        top_z_f = float(top_z) if top_z is not None else None
    except (TypeError, ValueError):
        top_z_f = None
    if top_z_f is None and grasp_hand_z is not None:
        top_z_f = float(grasp_hand_z) + grasp_depth_f
    carried_below_hand = max(0.05, float(height_m) - float(grasp_depth_f) + 0.015)
    dims = candidate.get("dims_lwh")
    half_major_xy = 0.079
    half_minor_xy = 0.030
    if isinstance(dims, (list, tuple)) and len(dims) >= 2:
        try:
            half_major_xy = max(float(dims[0]), float(dims[1])) * 0.5
            half_minor_xy = min(float(dims[0]), float(dims[1])) * 0.5
        except (TypeError, ValueError):
            pass
    pad = float(policy["attached_collision_padding_m"])
    radius_xy = float(half_major_xy + pad)
    yaw_rad = _resolve_carried_object_yaw_rad(candidate)
    offset_xy = (0.0, 0.0)
    center_source = "fk_hand_offset"
    gcb = candidate.get("grasp_center_base")
    if isinstance(gcb, (list, tuple)) and len(gcb) >= 2:
        gcx, gcy = float(gcb[0]), float(gcb[1])
        if grasp_hand_xy is not None:
            offset_xy = (gcx - float(grasp_hand_xy[0]), gcy - float(grasp_hand_xy[1]))
            center_source = "attached_offset_from_grasp"
        else:
            offset_xy = (0.0, 0.0)
            center_source = "candidate_grasp_center"
    return {
        "label": str(candidate.get("label", "")),
        "attached_link": "panda_hand",
        "dims_lwh": list(dims) if isinstance(dims, (list, tuple)) else None,
        "object_height_m": float(height_m),
        "carried_object_below_hand_m": float(carried_below_hand),
        "carried_object_below_tcp_m": float(carried_below_hand),
        "carried_object_radius_xy_m": radius_xy,
        "carried_object_half_major_xy_m": float(half_major_xy),
        "carried_object_half_minor_xy_m": float(half_minor_xy),
        "carried_object_yaw_rad": float(yaw_rad),
        "carried_object_center_offset_xy_m": [float(offset_xy[0]), float(offset_xy[1])],
        "carried_center_source": center_source,
        "attached_collision_padding_m": pad,
        "grasp_depth_from_top_m": float(grasp_depth_f),
        "object_top_z_estimated": top_z_f,
        "table_top_z": float(table_top_z),
    }


def _obs_xy(obs: Dict[str, Any]) -> Optional[Tuple[float, float]]:
    pos = obs.get("position") or obs.get("grasp_center_base")
    if isinstance(pos, (list, tuple)) and len(pos) >= 2:
        return float(pos[0]), float(pos[1])
    return None


def _obstacle_height_m(obs: Dict[str, Any], table_top_z: float) -> float:
    for key in ("effective_height_m", "db_height_m", "object_height_m"):
        try:
            val = float(obs.get(key))
            if val > 1e-4:
                return val
        except (TypeError, ValueError):
            pass
    col = obs.get("collision_dims") or {}
    if isinstance(col, dict):
        if str(col.get("shape", "")).lower() == "cylinder":
            cyl = col.get("cylinder")
            if isinstance(cyl, (list, tuple)) and len(cyl) >= 2:
                try:
                    return float(cyl[1])
                except (TypeError, ValueError):
                    pass
        box = col.get("box")
        if isinstance(box, (list, tuple)) and len(box) >= 3:
            try:
                return float(max(box[0], box[1], box[2]))
            except (TypeError, ValueError):
                pass
    return 0.120


def _obstacle_top_z(obs: Dict[str, Any], table_top_z: float) -> float:
    top = obs.get("top_z_m")
    try:
        if top is not None:
            return float(top)
    except (TypeError, ValueError):
        pass
    pos = obs.get("position")
    hz = _obstacle_height_m(obs, table_top_z)
    if isinstance(pos, (list, tuple)) and len(pos) >= 3:
        return float(pos[2]) + 0.5 * hz
    return float(table_top_z) + hz


def remaining_obstacle_labels(scene_obstacles: Sequence[Dict[str, Any]]) -> List[str]:
    out: List[str] = []
    for obs in scene_obstacles:
        if not isinstance(obs, dict) or bool(obs.get("is_target", False)):
            continue
        lb = str(obs.get("label", "")).strip().lower()
        if lb:
            out.append(lb)
    return sorted(set(out))


def compute_carry_safe_tcp_z(
    *,
    policy: Dict[str, Any],
    attached_geom: Dict[str, Any],
    table_top_z: float,
    remaining_obstacles: Sequence[Dict[str, Any]],
    current_tcp_z: float,
) -> Tuple[float, Dict[str, Any]]:
    """Altura mínima del TCP de grasp (no panda_hand) para transporte seguro."""
    below = float(
        attached_geom.get("carried_object_below_tcp_m")
        or attached_geom.get("carried_object_below_hand_m")
        or 0.18
    )
    table_req = float(table_top_z) + below + float(policy["carry_clearance_above_table_m"])
    max_obs_top = float(table_top_z)
    for obs in remaining_obstacles:
        if bool(obs.get("is_target", False)):
            continue
        max_obs_top = max(max_obs_top, _obstacle_top_z(obs, table_top_z))
    obs_req = max_obs_top + below + float(policy["carry_clearance_above_obstacles_m"])
    carry_safe = max(
        float(policy.get("min_carry_tcp_z_m", 0.700)),
        table_req,
        obs_req,
        float(current_tcp_z),
    )
    detail = {
        "min_carry_tcp_z_m": float(policy.get("min_carry_tcp_z_m", 0.700)),
        "table_required_tcp_z": table_req,
        "obstacle_required_tcp_z": obs_req,
        "max_obstacle_top_z": max_obs_top,
        "current_tcp_z": float(current_tcp_z),
    }
    return float(carry_safe), detail


def compute_global_over_obstacles_hand_z(
    *,
    policy: Dict[str, Any],
    attached_geom: Dict[str, Any],
    table_top_z: float,
    remaining_obstacles: Sequence[Dict[str, Any]],
    current_hand_z: float,
) -> Tuple[float, Dict[str, Any]]:
    """Referencia: altura para pasar por encima de todos los obstáculos (no obligatoria en swept_volume)."""
    below = float(attached_geom["carried_object_below_hand_m"])
    table_req = (
        float(table_top_z) + below + float(policy["carry_clearance_above_table_m"])
    )
    max_obs_top = float(table_top_z)
    for obs in remaining_obstacles:
        if bool(obs.get("is_target", False)):
            continue
        max_obs_top = max(max_obs_top, _obstacle_top_z(obs, table_top_z))
    obs_req = max_obs_top + below + float(policy["carry_clearance_above_obstacles_m"])
    global_z = max(
        float(policy.get("min_carry_hand_z_m", policy.get("min_carry_tcp_z_m", 0.700))),
        table_req,
        obs_req,
        float(current_hand_z),
    )
    detail = {
        "table_required_hand_z": table_req,
        "obstacle_required_hand_z": obs_req,
        "max_obstacle_top_z": max_obs_top,
        "current_hand_z": float(current_hand_z),
    }
    return float(global_z), detail


def compute_carry_safe_hand_z(
    *,
    policy: Dict[str, Any],
    attached_geom: Dict[str, Any],
    table_top_z: float,
    remaining_obstacles: Sequence[Dict[str, Any]],
    current_hand_z: float,
) -> Tuple[float, Dict[str, Any]]:
    """Alias de referencia global (compatibilidad)."""
    return compute_global_over_obstacles_hand_z(
        policy=policy,
        attached_geom=attached_geom,
        table_top_z=table_top_z,
        remaining_obstacles=remaining_obstacles,
        current_hand_z=current_hand_z,
    )


def table_minimum_hand_z(
    *,
    policy: Dict[str, Any],
    attached_geom: Dict[str, Any],
    table_top_z: float,
    current_hand_z: float,
) -> float:
    """Solo clearance mesa; no eleva por encima de todos los obstáculos."""
    below = float(attached_geom["carried_object_below_hand_m"])
    table_req = (
        float(table_top_z) + below + float(policy["carry_clearance_above_table_m"]) * 0.5
    )
    return max(float(current_hand_z), table_req)


def build_adaptive_hand_z_candidates(
    current_hand_z: float,
    policy: Dict[str, Any],
    attached_geom: Dict[str, Any],
    table_top_z: float,
) -> List[float]:
    """Alturas incrementales: actual, +0.03, +0.05, luego +0.05 hasta max_carry_hand_z_m."""
    max_z = float(policy.get("max_carry_hand_z_m", DEFAULT_MAX_CARRY_HAND_Z_M))
    floor_z = table_minimum_hand_z(
        policy=policy,
        attached_geom=attached_geom,
        table_top_z=table_top_z,
        current_hand_z=current_hand_z,
    )
    increment = float(policy.get("carry_height_increment_m", DEFAULT_CARRY_HEIGHT_INCREMENT_M))
    deltas: List[float] = list(ADAPTIVE_CARRY_HEIGHT_DELTAS_M)
    z = float(current_hand_z) + float(ADAPTIVE_CARRY_HEIGHT_DELTAS_M[-1]) + increment
    while z <= max_z + 1e-6:
        deltas.append(z - float(current_hand_z))
        z += CARRY_SAFE_HEIGHT_RETRY_STEP_M
    seen: set = set()
    out: List[float] = []
    for delta in deltas:
        hz = min(max_z, max(floor_z, float(current_hand_z) + float(delta)))
        key = round(hz, 4)
        if key not in seen:
            seen.add(key)
            out.append(float(hz))
    return out


def resolve_adaptive_carry_height_policy(
    *,
    candidate: Dict[str, Any],
    current_hand: Tuple[float, float, float],
    current_tcp: Optional[Tuple[float, float, float]],
    attached_geom: Dict[str, Any],
    table_top_z: float,
    scene_obstacles: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    policy = resolve_carry_transport_policy(candidate)
    frame = resolve_hand_tcp_frame(
        current_hand=current_hand,
        current_tcp=current_tcp,
        candidate=candidate,
    )
    current_hand_z = float(frame["current_hand_z"])
    below = float(attached_geom["carried_object_below_hand_m"])
    attached_bottom_z = current_hand_z - below
    table_clearance = attached_bottom_z - float(table_top_z)
    min_table_clr = float(policy["carry_clearance_above_table_m"]) * 0.5
    req_xy_clr = float(policy["carry_clearance_above_obstacles_m"])

    global_hand_z, _global_detail = compute_global_over_obstacles_hand_z(
        policy=policy,
        attached_geom=attached_geom,
        table_top_z=float(table_top_z),
        remaining_obstacles=scene_obstacles,
        current_hand_z=current_hand_z,
    )
    max_z = float(policy.get("max_carry_hand_z_m", DEFAULT_MAX_CARRY_HAND_Z_M))
    global_hand_z = min(global_hand_z, max_z + 0.5)  # referencia sin cap estricto en log

    clearance_mode = str(policy.get("obstacle_clearance_mode", DEFAULT_OBSTACLE_CLEARANCE_MODE))
    ok_current, _checks, _metrics = validate_attached_hand_pose(
        current_hand,
        attached_geom,
        scene_obstacles,
        table_top_z=float(table_top_z),
        min_table_clearance_m=min_table_clr,
        required_xy_clearance_m=req_xy_clr,
    )
    hand_z_candidates = build_adaptive_hand_z_candidates(
        current_hand_z, policy, attached_geom, float(table_top_z)
    )
    global_height_required = clearance_mode == "over_all_obstacles"
    if global_height_required:
        selected = min(max(global_hand_z, hand_z_candidates[0]), max_z)
    elif ok_current:
        selected = current_hand_z
    else:
        selected = hand_z_candidates[0] if hand_z_candidates else current_hand_z

    return {
        "current_hand_z": current_hand_z,
        "current_tcp_z": float(frame["current_grasp_tcp_z"]),
        "attached_bottom_z": float(attached_bottom_z),
        "table_clearance": float(table_clearance),
        "obstacle_clearance_mode": clearance_mode,
        "vertical_clearance_mode": str(
            policy.get("vertical_clearance_mode", "corridor_swept_volume")
        ),
        "global_over_obstacles_height": float(global_hand_z),
        "global_height_required": bool(global_height_required),
        "preferred_hand_z": float(selected),
        "selected_hand_z": float(selected),
        "hand_z_candidates": hand_z_candidates,
        "max_carry_hand_z": float(max_z),
        "current_pose_clearance_ok": bool(ok_current),
        "result": "OK",
    }


def format_adaptive_carry_height_policy_log(adaptive: Dict[str, Any]) -> str:
    return (
        "[ADAPTIVE_CARRY_HEIGHT_POLICY]\n"
        "current_hand_z=%.3f\n"
        "current_tcp_z=%.3f\n"
        "attached_bottom_z=%.3f\n"
        "table_clearance=%.3f\n"
        "obstacle_clearance_mode=%s\n"
        "global_over_obstacles_height=%.3f\n"
        "global_height_required=%s\n"
        "selected_hand_z=%.3f\n"
        "hand_z_candidates=%s\n"
        "max_carry_hand_z=%.3f\n"
        "result=%s"
        % (
            float(adaptive.get("current_hand_z", 0.0)),
            float(adaptive.get("current_tcp_z", 0.0)),
            float(adaptive.get("attached_bottom_z", 0.0)),
            float(adaptive.get("table_clearance", 0.0)),
            str(adaptive.get("obstacle_clearance_mode", "")),
            float(adaptive.get("global_over_obstacles_height", 0.0)),
            str(bool(adaptive.get("global_height_required"))).lower(),
            float(adaptive.get("selected_hand_z", 0.0)),
            adaptive.get("hand_z_candidates", []),
            float(adaptive.get("max_carry_hand_z", 0.0)),
            str(adaptive.get("result", "OK")),
        )
    )


def attached_obstacle_clearance_3d(
    hand_pos: Tuple[float, float, float],
    attached_geom: Dict[str, Any],
    obs: Dict[str, Any],
    *,
    table_top_z: float,
    required_xy_clearance_m: float,
) -> Dict[str, Any]:
    hx, hy, hz = float(hand_pos[0]), float(hand_pos[1]), float(hand_pos[2])
    below = float(attached_geom["carried_object_below_hand_m"])
    height_m = float(attached_geom.get("object_height_m", below))
    att_bottom = hz - below
    att_top = att_bottom + height_m
    cx, cy, center_source = resolve_carried_object_center_xy(hand_pos, attached_geom)
    oxy = _obs_xy(obs)
    if oxy is None:
        return {
            "obstacle_label": str(obs.get("label", "")),
            "result": "SKIP",
            "hard_collision": False,
            "safety_margin_ok": True,
        }
    obs_r = float(object_collision_radius_xy(obs))
    obs_top = _obstacle_top_z(obs, table_top_z)
    obs_height = _obstacle_height_m(obs, table_top_z)
    obs_bottom = obs_top - obs_height
    dxy = math.hypot(cx - oxy[0], cy - oxy[1])
    if dxy > 1e-9:
        ux, uy = (float(oxy[0]) - cx) / dxy, (float(oxy[1]) - cy) / dxy
        eff_radius = carried_effective_radius_xy_toward(attached_geom, (ux, uy))
    else:
        eff_radius = carried_effective_radius_xy_toward(attached_geom, (1.0, 0.0))
    radius = float(eff_radius)
    xy_clearance = dxy - obs_r - radius
    z_clearance = att_bottom - obs_top
    z_overlap = att_bottom < obs_top + 0.010 and att_top > obs_bottom - 0.010
    xy_overlap = dxy < (obs_r + radius)
    hard_collision = bool(z_overlap and xy_overlap and z_clearance < -0.005)
    safety_margin_ok = bool(
        not hard_collision
        and (
            not xy_overlap
            or z_clearance >= float(required_xy_clearance_m)
            or xy_clearance >= float(required_xy_clearance_m)
        )
    )
    if hard_collision:
        result = "COLLISION"
    elif safety_margin_ok:
        result = "OK"
    else:
        result = "NEAR"
    object_center_z = att_bottom + 0.5 * height_m
    return {
        "obstacle_label": str(obs.get("label", "")),
        "obstacle_entity": str(obs.get("entity_name", "")),
        "obstacle_center": (float(oxy[0]), float(oxy[1]), float(obs_top)),
        "obstacle_dims": obs.get("collision_dims"),
        "obstacle_top_z": float(obs_top),
        "attached_center": (float(cx), float(cy), float(object_center_z)),
        "attached_hand": (hx, hy, hz),
        "carried_center_source": center_source,
        "attached_dims": attached_geom.get("dims_lwh"),
        "attached_bottom_z": float(att_bottom),
        "attached_top_z": float(att_top),
        "xy_center_distance": float(dxy),
        "attached_radius_xy": float(radius),
        "attached_effective_radius_xy": float(radius),
        "obstacle_radius_xy": float(obs_r),
        "required_xy_clearance": float(required_xy_clearance_m),
        "xy_clearance": float(xy_clearance),
        "z_clearance": float(z_clearance),
        "z_overlap": bool(z_overlap),
        "xy_overlap": bool(xy_overlap),
        "hard_collision": bool(hard_collision),
        "hard_collision_3d": bool(hard_collision),
        "safety_margin_ok": bool(safety_margin_ok),
        "result": result,
    }


def format_attached_transport_clearance_log(check: Dict[str, Any]) -> str:
    oc = check.get("obstacle_center")
    ac = check.get("attached_center")
    oc_str = (
        "n/a"
        if not isinstance(oc, (list, tuple))
        else "(%.3f, %.3f, %.3f)" % (float(oc[0]), float(oc[1]), float(oc[2]))
    )
    ac_str = (
        "n/a"
        if not isinstance(ac, (list, tuple))
        else "(%.3f, %.3f, %.3f)" % (float(ac[0]), float(ac[1]), float(ac[2]))
    )
    return (
        "[ATTACHED_TRANSPORT_CLEARANCE_CHECK]\n"
        "obstacle_label=%s\n"
        "obstacle_entity=%s\n"
        "obstacle_center=%s\n"
        "obstacle_dims=%s\n"
        "obstacle_top_z=%.3f\n"
        "attached_center=%s\n"
        "attached_dims=%s\n"
        "attached_bottom_z=%.3f\n"
        "attached_top_z=%.3f\n"
        "xy_center_distance=%.3f\n"
        "attached_radius_xy=%.3f\n"
        "obstacle_radius_xy=%.3f\n"
        "required_xy_clearance=%.3f\n"
        "xy_clearance=%.3f\n"
        "z_clearance=%.3f\n"
        "z_overlap=%s\n"
        "xy_overlap=%s\n"
        "hard_collision=%s\n"
        "safety_margin_ok=%s\n"
        "result=%s"
        % (
            str(check.get("obstacle_label", "")),
            str(check.get("obstacle_entity", "")),
            oc_str,
            check.get("obstacle_dims"),
            float(check.get("obstacle_top_z", 0.0)),
            ac_str,
            check.get("attached_dims"),
            float(check.get("attached_bottom_z", 0.0)),
            float(check.get("attached_top_z", 0.0)),
            float(check.get("xy_center_distance", 0.0)),
            float(check.get("attached_radius_xy", 0.0)),
            float(check.get("obstacle_radius_xy", 0.0)),
            float(check.get("required_xy_clearance", 0.0)),
            float(check.get("xy_clearance", 0.0)),
            float(check.get("z_clearance", 0.0)),
            str(bool(check.get("z_overlap"))).lower(),
            str(bool(check.get("xy_overlap"))).lower(),
            str(bool(check.get("hard_collision"))).lower(),
            str(bool(check.get("safety_margin_ok"))).lower(),
            str(check.get("result", "")),
        )
    )


def validate_attached_hand_pose(
    hand_pos: Tuple[float, float, float],
    attached_geom: Dict[str, Any],
    obstacles: Sequence[Dict[str, Any]],
    *,
    table_top_z: float,
    min_table_clearance_m: float,
    required_xy_clearance_m: float,
    safety_margin_tolerance_m: float = 0.0,
    obstacle_margin_mode: str = "3d",
) -> Tuple[bool, List[Dict[str, Any]], Dict[str, Any]]:
    below = float(attached_geom["carried_object_below_hand_m"])
    att_bottom = float(hand_pos[2]) - below
    table_clr = att_bottom - float(table_top_z)
    if table_clr + 1e-6 < float(min_table_clearance_m):
        return False, [], {
            "reason": "attached_bottom_near_table",
            "min_clearance_to_table": table_clr,
        }
    checks: List[Dict[str, Any]] = []
    hard = False
    near = False
    min_geom_xy = float("inf")
    min_safety_margin = float("inf")
    tol = float(safety_margin_tolerance_m)
    req = float(required_xy_clearance_m)
    local_xy_mode = str(obstacle_margin_mode) in (
        "xy_unless_hard_collision",
        "local_escape_post_lift",
    )
    for obs in obstacles:
        if bool(obs.get("is_target", False)):
            continue
        chk = attached_obstacle_clearance_3d(
            hand_pos,
            attached_geom,
            obs,
            table_top_z=table_top_z,
            required_xy_clearance_m=required_xy_clearance_m,
        )
        checks.append(chk)
        if local_xy_mode:
            if bool(chk.get("hard_collision")):
                hard = True
            elif chk.get("result") != "SKIP":
                xy_c = float(chk.get("xy_clearance", 0.0))
                min_geom_xy = min(min_geom_xy, xy_c)
                if bool(chk.get("xy_overlap")):
                    min_safety_margin = min(min_safety_margin, xy_c)
                    if xy_c + tol + 1e-6 < req:
                        near = True
            continue
        if chk.get("result") == "COLLISION":
            hard = True
        elif chk.get("result") == "NEAR":
            near = True
        if chk.get("result") != "SKIP":
            min_geom_xy = min(min_geom_xy, float(chk.get("xy_clearance", 0.0)))
            if bool(chk.get("xy_overlap")) or bool(chk.get("z_overlap")) or chk.get(
                "result"
            ) in ("NEAR", "COLLISION"):
                margin = min(
                    float(chk.get("xy_clearance", 0.0)),
                    float(chk.get("z_clearance", 0.0)),
                )
                min_safety_margin = min(min_safety_margin, margin)
    if local_xy_mode and min_safety_margin == float("inf"):
        min_safety_margin = min_geom_xy
    fail_metrics = {
        "min_geometric_xy_clearance_m": min_geom_xy,
        "min_safety_margin_m": min_safety_margin,
        "min_clearance_to_obstacles": min_safety_margin,
    }
    if hard:
        return False, checks, {"reason": "hard_collision", **fail_metrics}
    if near:
        if (
            tol > 0.0
            and min_safety_margin != float("inf")
            and float(min_safety_margin) >= -tol
        ):
            return True, checks, {
                "min_attached_object_bottom_z": att_bottom,
                "min_clearance_to_table": table_clr,
                "min_geometric_xy_clearance_m": min_geom_xy,
                "min_safety_margin_m": min_safety_margin,
                "min_clearance_to_obstacles": min_safety_margin,
                "borderline_margin": True,
            }
        return False, checks, {"reason": "near_obstacle_margin", **fail_metrics}
    if local_xy_mode:
        return True, checks, {
            "min_attached_object_bottom_z": att_bottom,
            "min_clearance_to_table": table_clr,
            "min_geometric_xy_clearance_m": min_geom_xy,
            "min_safety_margin_m": min_safety_margin
            if min_safety_margin != float("inf")
            else min_geom_xy,
            "min_clearance_to_obstacles": min_safety_margin
            if min_safety_margin != float("inf")
            else min_geom_xy,
        }
    return True, checks, {
        "min_attached_object_bottom_z": att_bottom,
        "min_clearance_to_table": table_clr,
        "min_geometric_xy_clearance_m": min(
            (float(c.get("xy_clearance", 0.0)) for c in checks if c.get("result") != "SKIP"),
            default=float("inf"),
        ),
        "min_safety_margin_m": min(
            (
                min(float(c.get("xy_clearance", 0.0)), float(c.get("z_clearance", 0.0)))
                for c in checks
                if c.get("result") in ("NEAR", "COLLISION")
                or bool(c.get("xy_overlap"))
                or bool(c.get("z_overlap"))
            ),
            default=float("inf"),
        ),
        "min_clearance_to_obstacles": min(
            (
                min(float(c.get("xy_clearance", 0.0)), float(c.get("z_clearance", 0.0)))
                for c in checks
                if c.get("result") in ("NEAR", "COLLISION")
                or bool(c.get("xy_overlap"))
                or bool(c.get("z_overlap"))
            ),
            default=float("inf"),
        ),
    }


def _lerp_joints(a: Sequence[float], b: Sequence[float], t: float) -> List[float]:
    n = min(len(a), len(b))
    return [float(a[i]) + float(t) * (float(b[i]) - float(a[i])) for i in range(n)]


def validate_attached_joint_segment(
    start_joints: Sequence[float],
    end_joints: Sequence[float],
    *,
    fk_hand_fn: Optional[Callable[[Any], Optional[Tuple[float, float, float]]]],
    attached_geom: Dict[str, Any],
    table_top_z: float,
    obstacles: Sequence[Dict[str, Any]],
    min_table_clearance_m: float,
    required_xy_clearance_m: float,
    safety_margin_tolerance_m: float = 0.0,
    n_samples: int = 14,
) -> Tuple[bool, Dict[str, Any], List[Dict[str, Any]]]:
    if fk_hand_fn is None:
        return False, {"reason": "fk_unavailable"}, []
    all_checks: List[Dict[str, Any]] = []
    min_table_clr = float("inf")
    min_geom_xy = float("inf")
    min_safety_margin = float("inf")
    min_bottom = float("inf")
    for i in range(max(2, int(n_samples))):
        t = float(i) / float(max(1, n_samples - 1))
        joints = _lerp_joints(start_joints, end_joints, t)
        hand = fk_hand_fn(joints)
        if hand is None:
            return False, {"reason": "fk_sample_failed", "sample": i}, all_checks
        ok, checks, metrics = validate_attached_hand_pose(
            hand,
            attached_geom,
            obstacles,
            table_top_z=table_top_z,
            min_table_clearance_m=min_table_clearance_m,
            required_xy_clearance_m=required_xy_clearance_m,
            safety_margin_tolerance_m=float(safety_margin_tolerance_m),
        )
        all_checks.extend(checks)
        min_bottom = min(min_bottom, float(metrics.get("min_attached_object_bottom_z", hand[2])))
        if metrics.get("min_clearance_to_table") is not None:
            min_table_clr = min(min_table_clr, float(metrics["min_clearance_to_table"]))
        for chk in checks:
            if chk.get("result") == "SKIP":
                continue
            min_geom_xy = min(min_geom_xy, float(chk.get("xy_clearance", 0.0)))
            if bool(chk.get("xy_overlap")) or bool(chk.get("z_overlap")) or chk.get(
                "result"
            ) in ("NEAR", "COLLISION"):
                margin = min(
                    float(chk.get("xy_clearance", 0.0)),
                    float(chk.get("z_clearance", 0.0)),
                )
                min_safety_margin = min(min_safety_margin, margin)
        if not ok:
            metrics["sample"] = i
            metrics["hand_pos"] = hand
            metrics["min_geometric_xy_clearance_m"] = min_geom_xy
            metrics["min_safety_margin_m"] = min_safety_margin
            metrics["min_clearance_to_obstacles"] = min_safety_margin
            tol = float(safety_margin_tolerance_m)
            if (
                tol > 0.0
                and min_safety_margin != float("inf")
                and float(min_safety_margin) >= -tol
                and metrics.get("reason") not in ("hard_collision", "attached_bottom_near_table")
                and not any(bool(c.get("hard_collision")) for c in checks)
            ):
                continue
            return False, metrics, all_checks
    return True, {
        "min_attached_object_bottom_z": min_bottom,
        "min_clearance_to_table": min_table_clr,
        "min_geometric_xy_clearance_m": min_geom_xy,
        "min_safety_margin_m": min_safety_margin,
        "min_clearance_to_obstacles": min_safety_margin,
    }, all_checks


def generate_transport_entry_candidates(
    current_hand_xy: Tuple[float, float],
    candidate_hand_z: float,
    policy: Dict[str, Any],
    *,
    include_lateral: bool = True,
) -> List[Dict[str, Any]]:
    x, y = float(current_hand_xy[0]), float(current_hand_xy[1])
    lane_y = float(policy.get("transport_exit_lane_y_m", DEFAULT_TRANSPORT_EXIT_LANE_Y_M))
    z_base = float(candidate_hand_z)
    raw: List[Tuple[str, Tuple[float, float, float]]] = []
    if include_lateral:
        raw.extend(
            [
                ("front_lane_y_positive", (max(x, 0.35), y + 0.14, z_base)),
                ("front_lane_y_negative", (max(x, 0.35), y - 0.14, z_base)),
                ("lateral_left", (max(0.30, x - 0.14), lane_y, z_base)),
                ("lateral_right", (max(0.30, x + 0.14), lane_y, z_base)),
                ("table_exit_lane", (0.35, lane_y, z_base)),
            ]
        )
    out: List[Dict[str, Any]] = []
    for mode, pose in raw:
        delta_xy = math.hypot(float(pose[0]) - x, float(pose[1]) - y)
        out.append(
            {
                "mode": mode,
                "candidate_hand": pose,
                "candidate_hand_z": float(z_base),
                "delta_xy_from_current": float(delta_xy),
            }
        )
    return out


def format_transport_entry_candidate_log(
    idx: int,
    cand: Dict[str, Any],
    *,
    current_hand: Tuple[float, float, float],
    metrics: Dict[str, Any],
    result: str,
) -> str:
    pose = cand.get("candidate_hand") or (0.0, 0.0, 0.0)
    return (
        "[TRANSPORT_ENTRY_CANDIDATE]\n"
        "idx=%d\n"
        "mode=%s\n"
        "current_hand=(%.3f, %.3f, %.3f)\n"
        "candidate_hand=(%.3f, %.3f, %.3f)\n"
        "candidate_hand_z=%.3f\n"
        "delta_xy_from_current=%.3f\n"
        "min_clearance_to_obstacles=%s\n"
        "hard_collision=%s\n"
        "safety_margin_ok=%s\n"
        "result=%s"
        % (
            idx,
            str(cand.get("mode", "")),
            current_hand[0],
            current_hand[1],
            current_hand[2],
            float(pose[0]),
            float(pose[1]),
            float(pose[2]),
            float(cand.get("candidate_hand_z", pose[2])),
            float(cand.get("delta_xy_from_current", 0.0)),
            metrics.get("min_clearance_to_obstacles", "n/a"),
            str(metrics.get("hard_collision", False)).lower(),
            str(metrics.get("safety_margin_ok", False)).lower(),
            result,
        )
    )


def plan_post_pick_transport_entry(
    *,
    candidate: Dict[str, Any],
    current_hand: Tuple[float, float, float],
    current_tcp: Optional[Tuple[float, float, float]],
    current_joints: Sequence[float],
    first_waypoint: str,
    first_waypoint_joints: Sequence[float],
    table_top_z: float,
    scene_obstacles: Sequence[Dict[str, Any]],
    fk_hand_fn: Optional[Callable[[Any], Optional[Tuple[float, float, float]]]],
) -> Dict[str, Any]:
    policy = resolve_carry_transport_policy(candidate)
    frame = resolve_hand_tcp_frame(
        current_hand=current_hand,
        current_tcp=current_tcp,
        candidate=candidate,
    )
    attached_geom = compute_attached_object_geometry(
        candidate,
        grasp_hand_z=float(current_hand[2]),
        table_top_z=float(table_top_z),
    )
    remaining = remaining_obstacle_labels(scene_obstacles)
    adaptive = resolve_adaptive_carry_height_policy(
        candidate=candidate,
        current_hand=current_hand,
        current_tcp=current_tcp,
        attached_geom=attached_geom,
        table_top_z=float(table_top_z),
        scene_obstacles=scene_obstacles,
    )
    global_hand_z = float(adaptive.get("global_over_obstacles_height", current_hand[2]))
    carry_safe_tcp_z = tcp_z_from_hand_z(global_hand_z, float(frame["hand_to_tcp_z"]))
    preferred_hand_z = float(adaptive.get("selected_hand_z", current_hand[2]))
    direct_ok, direct_metrics, clearance_checks = validate_attached_joint_segment(
        current_joints,
        first_waypoint_joints,
        fk_hand_fn=fk_hand_fn,
        attached_geom=attached_geom,
        table_top_z=float(table_top_z),
        obstacles=scene_obstacles,
        min_table_clearance_m=float(policy["carry_clearance_above_table_m"]) * 0.5,
        required_xy_clearance_m=float(policy["carry_clearance_above_obstacles_m"]),
    )
    candidates = generate_transport_entry_candidates(
        (float(current_hand[0]), float(current_hand[1])),
        preferred_hand_z,
        policy,
    )
    return {
        "label": str(candidate.get("label", "")),
        "attached_object_geometry": attached_geom,
        "frame": frame,
        "adaptive_carry_height": adaptive,
        "carry_safe_tcp_z": float(carry_safe_tcp_z),
        "carry_safe_hand_z": float(global_hand_z),
        "preferred_hand_z": preferred_hand_z,
        "hand_z_candidates": list(adaptive.get("hand_z_candidates") or []),
        "remaining_obstacles": remaining,
        "direct_segment_collision_free": bool(direct_ok),
        "direct_segment_metrics": direct_metrics,
        "clearance_checks": clearance_checks,
        "transport_entry_candidates": candidates,
        "entry_mode": "adaptive_corridor_swept_volume",
        "first_waypoint": first_waypoint,
    }


def format_attached_transport_preflight_log(plan: Dict[str, Any], sequence: Sequence[str]) -> str:
    geom = plan.get("attached_object_geometry") or {}
    metrics = plan.get("direct_segment_metrics") or {}
    return (
        "[ATTACHED_TRANSPORT_PREFLIGHT]\n"
        "label=%s\n"
        "attached_object_dims=%s\n"
        "carry_safe_hand_z=%.3f\n"
        "carry_safe_tcp_z=%.3f\n"
        "remaining_obstacles=%s\n"
        "direct_action_sequence=%s\n"
        "swept_collision_free=%s\n"
        "min_clearance_to_table=%s\n"
        "min_clearance_to_obstacles=%s\n"
        "result=%s\n"
        "reason=%s"
        % (
            str(plan.get("label", "")),
            geom.get("dims_lwh"),
            float(plan.get("carry_safe_hand_z", 0.0)),
            float(plan.get("carry_safe_tcp_z", 0.0)),
            plan.get("remaining_obstacles", []),
            list(sequence),
            str(bool(plan.get("direct_segment_collision_free"))).lower(),
            metrics.get("min_clearance_to_table", "n/a"),
            metrics.get("min_clearance_to_obstacles", "n/a"),
            "OK" if plan.get("direct_segment_collision_free") else "FAIL",
            str(metrics.get("reason", "n/a")),
        )
    )
