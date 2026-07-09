"""Lectura de pose real en Gazebo tras spawn y actualización de RuntimeScene GT."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import rclpy
from rclpy.node import Node
from tf2_msgs.msg import TFMessage

from panda_vision.spawn.gz_spawn_runtime import (
    gazebo_pose_info_ros_topic,
    gz_world_name_from_param,
)
from panda_vision.spawn.runtime_scene_gt import make_gt_object_entry
from panda_vision.spawn.runtime_scene_gt_geometry import (
    build_gt_fields_from_semantic_center,
    is_known_spawn_geometry_box_label,
    semantic_center_from_gazebo_model_origin,
    top_face_center_from_semantic_center,
)

Pose6 = Tuple[float, float, float, float, float, float]


def normalize_angle_rad(a: float) -> float:
    return (float(a) + math.pi) % (2.0 * math.pi) - math.pi


def quaternion_to_rpy(
    x: float, y: float, z: float, w: float
) -> Tuple[float, float, float]:
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (w * y - z * x)
    if abs(sinp) >= 1.0:
        pitch = math.copysign(math.pi / 2.0, sinp)
    else:
        pitch = math.asin(sinp)

    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)

    return float(roll), float(pitch), float(yaw)


def quaternion_to_yaw(x: float, y: float, z: float, w: float) -> float:
    return quaternion_to_rpy(x, y, z, w)[2]


def rpy_to_quaternion(
    roll: float, pitch: float, yaw: float
) -> Tuple[float, float, float, float]:
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)
    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy
    return float(qx), float(qy), float(qz), float(qw)


def update_world_pose_cache(
    msg: TFMessage,
    cache: Dict[str, Pose6],
    *,
    world_frame: str,
) -> None:
    """Actualiza ``cache`` desde tf2_msgs/TFMessage (SceneBroadcaster)."""
    world_norm = world_frame.strip().lstrip("/")

    def _matches_world(parent: str) -> bool:
        norm = parent.strip().lstrip("/")
        # SceneBroadcaster (Gazebo Sim) usa parent vacío en pose/info.
        if not norm:
            return True
        if not world_norm:
            return True
        return norm == world_norm or norm.endswith(f"/{world_norm}")

    new_map: Dict[str, Pose6] = {}
    for ts in msg.transforms:
        parent = (ts.header.frame_id or "").strip()
        if not _matches_world(parent):
            continue
        child = (ts.child_frame_id or "").strip()
        if not child:
            continue
        q = ts.transform.rotation
        t = ts.transform.translation
        try:
            roll, pitch, yaw = quaternion_to_rpy(q.x, q.y, q.z, q.w)
        except (ValueError, ZeroDivisionError):
            continue
        pose6: Pose6 = (t.x, t.y, t.z, roll, pitch, yaw)
        for key in _frame_id_cache_keys(child):
            new_map[key] = pose6
        parent_short = parent.split("::")[-1].split("/")[-1]
        if parent_short and _entity_name_like(parent_short):
            new_map[parent_short] = pose6
    if new_map:
        cache.clear()
        cache.update(new_map)


# x,y,z, qx,qy,qz,qw, roll,pitch,yaw
EntityPoseSample = Tuple[
    float, float, float,
    float, float, float, float,
    float, float, float,
]


@dataclass(frozen=True)
class PoseStabilityResult:
    stable: bool
    samples: int
    max_delta_xy_m: float
    max_delta_z_m: float
    max_delta_yaw_deg: float
    final_pose: Optional[EntityPoseSample]


@dataclass(frozen=True)
class SpawnPoseReadbackParams:
    update_runtime_scene_from_actual_gazebo_pose: bool = True
    post_spawn_settle_sec: float = 0.8
    pose_stability_timeout_sec: float = 2.0
    pose_stability_sample_count: int = 5
    pose_stability_sample_period_sec: float = 0.15
    pose_stability_xy_threshold_m: float = 0.002
    pose_stability_z_threshold_m: float = 0.001
    pose_stability_yaw_threshold_deg: float = 0.5
    reject_topdown_if_tilted: bool = True
    max_upright_roll_pitch_deg: float = 3.0
    world_frame: str = "world"


def _entity_name_like(frame_id: str) -> bool:
    short = frame_id.split("::")[-1].split("/")[-1]
    return short.startswith("runtime_ycb_")


def _frame_id_cache_keys(frame_id: str) -> List[str]:
    raw = frame_id.strip()
    if not raw:
        return []
    short = raw.split("::")[-1]
    tail = raw.split("/")[-1]
    keys = [raw, short, tail]
    if not short.startswith("model::"):
        keys.append(f"model::{short}")
    out: List[str] = []
    seen = set()
    for k in keys:
        if k and k not in seen:
            seen.add(k)
            out.append(k)
    return out


def declare_spawn_pose_readback_params(node: Node) -> None:
    node.declare_parameter("update_runtime_scene_from_actual_gazebo_pose", True)
    node.declare_parameter("gazebo_pose_info_topic", "")
    node.declare_parameter("gazebo_dynamic_pose_info_topic", "")
    node.declare_parameter("post_spawn_settle_sec", 0.8)
    node.declare_parameter("pose_stability_timeout_sec", 2.0)
    node.declare_parameter("pose_stability_sample_count", 5)
    node.declare_parameter("pose_stability_sample_period_sec", 0.15)
    node.declare_parameter("pose_stability_xy_threshold_m", 0.002)
    node.declare_parameter("pose_stability_z_threshold_m", 0.001)
    node.declare_parameter("pose_stability_yaw_threshold_deg", 0.5)
    node.declare_parameter("reject_topdown_if_tilted", True)
    node.declare_parameter("max_upright_roll_pitch_deg", 3.0)
    if not node.has_parameter("world_frame"):
        node.declare_parameter("world_frame", "world")


def readback_params_from_node(node: Node) -> SpawnPoseReadbackParams:
    def _f(name: str, default: float) -> float:
        return float(node.get_parameter(name).value)

    def _b(name: str, default: bool) -> bool:
        return bool(node.get_parameter(name).value)

    def _i(name: str, default: int) -> int:
        return int(node.get_parameter(name).value)

    return SpawnPoseReadbackParams(
        update_runtime_scene_from_actual_gazebo_pose=_b(
            "update_runtime_scene_from_actual_gazebo_pose", True
        ),
        post_spawn_settle_sec=_f("post_spawn_settle_sec", 0.8),
        pose_stability_timeout_sec=_f("pose_stability_timeout_sec", 2.0),
        pose_stability_sample_count=_i("pose_stability_sample_count", 5),
        pose_stability_sample_period_sec=_f("pose_stability_sample_period_sec", 0.15),
        pose_stability_xy_threshold_m=_f("pose_stability_xy_threshold_m", 0.002),
        pose_stability_z_threshold_m=_f("pose_stability_z_threshold_m", 0.001),
        pose_stability_yaw_threshold_deg=_f("pose_stability_yaw_threshold_deg", 0.5),
        reject_topdown_if_tilted=_b("reject_topdown_if_tilted", True),
        max_upright_roll_pitch_deg=_f("max_upright_roll_pitch_deg", 3.0),
        world_frame=str(node.get_parameter("world_frame").value).strip() or "world",
    )


def resolve_world_pose_topics(node: Node) -> List[str]:
    """Topics ROS a probar (primario pose/info, opcional dynamic_pose/info)."""
    topics: List[str] = []
    seen = set()

    def _add(topic: str) -> None:
        t = str(topic).strip()
        if t and t not in seen:
            seen.add(t)
            topics.append(t)

    custom = str(node.get_parameter("gazebo_pose_info_topic").value).strip()
    if custom:
        _add(custom)
    else:
        world_pose = ""
        if node.has_parameter("world_pose_ros_topic"):
            world_pose = str(node.get_parameter("world_pose_ros_topic").value).strip()
        if world_pose:
            _add(world_pose)
        elif node.has_parameter("world_name"):
            wn = str(node.get_parameter("world_name").value).strip()
            if wn:
                _add(gazebo_pose_info_ros_topic(wn, dynamic=False))

    dynamic = str(node.get_parameter("gazebo_dynamic_pose_info_topic").value).strip()
    if dynamic:
        _add(dynamic)

    return topics


def _entity_matches_frame(entity_name: str, frame_id: str) -> bool:
    en = entity_name.strip()
    if not en or not frame_id:
        return False
    en_short = en.split("/")[-1].split("::")[-1]
    fr_short = frame_id.strip().split("/")[-1].split("::")[-1]
    if en == frame_id or en_short == fr_short:
        return True
    if en_short in frame_id or fr_short in en:
        return True
    if en in frame_id or frame_id in en:
        return True
    return False


def log_gazebo_pose_topic_unavailable(
    logger: Any,
    *,
    topic: str,
    publisher_count: int = 0,
    node: Optional[Node] = None,
) -> None:
    if logger is None:
        return
    if node is not None and getattr(node, "_pose_bridge_hint_logged", False):
        return
    try:
        logger.warning(
            "[GAZEBO_POSE_TOPIC_UNAVAILABLE] topic=%s publisher_count=%d "
            'hint="Spawn puede seguir; GT usará pose comandada. '
            "Con el launch completo: bridge_world_pose_info:=true."
            % (topic, int(publisher_count))
        )
    except Exception:
        pass


def log_gazebo_pose_entities_seen(
    logger: Any,
    *,
    topic: str,
    cache: Dict[str, Pose6],
    max_examples: int = 12,
) -> None:
    if logger is None:
        return
    examples = sorted(cache.keys())[: max(1, int(max_examples))]
    try:
        logger.warning(
            "[GAZEBO_POSE_ENTITIES_SEEN] topic=%s count=%d examples=%s"
            % (topic, len(cache), examples)
        )
    except Exception:
        pass


def gazebo_pose_topics_ready(
    node: Node,
    topics: List[str],
    *,
    world_frame: str,
    probe_sec: float = 0.6,
    logger: Any = None,
) -> List[str]:
    """Devuelve topics con publisher ROS y al menos un TFMessage con entidades."""
    ready: List[str] = []
    for topic in topics:
        try:
            pub_count = int(node.count_publishers(topic))
        except Exception:
            pub_count = 0
        if pub_count <= 0:
            log_gazebo_pose_topic_unavailable(
                logger, topic=topic, publisher_count=pub_count, node=node
            )
            continue
        cache = sample_world_poses_once(
            node,
            world_pose_topic=topic,
            world_frame=world_frame,
            collect_sec=max(0.15, float(probe_sec)),
        )
        if cache:
            ready.append(topic)
        elif logger is not None:
            try:
                logger.warning(
                    "[GAZEBO_POSE_TOPIC_UNAVAILABLE] topic=%s publisher_count=%d "
                    "hint=\"Publisher present but no TFMessage yet; retry after spawn settle.\""
                    % (topic, pub_count)
                )
            except Exception:
                pass
    return ready


def _normalize_entity_keys(entity_name: str) -> List[str]:
    short = entity_name.split("::")[-1].strip()
    keys = [entity_name.strip(), short]
    if short and not short.startswith("model::"):
        keys.append(f"model::{short}")
    out: List[str] = []
    seen = set()
    for k in keys:
        if k and k not in seen:
            seen.add(k)
            out.append(k)
    return out


def pose6_to_entity_sample(pose6: Pose6) -> EntityPoseSample:
    x, y, z, roll, pitch, yaw = pose6
    qx, qy, qz, qw = rpy_to_quaternion(roll, pitch, yaw)
    return (
        float(x), float(y), float(z),
        float(qx), float(qy), float(qz), float(qw),
        float(roll), float(pitch), float(yaw),
    )


def max_tilt_deg_from_sample(sample: EntityPoseSample) -> float:
    roll, pitch = sample[7], sample[8]
    return max(abs(math.degrees(float(roll))), abs(math.degrees(float(pitch))))


def lookup_entity_pose(
    cache: Dict[str, Pose6],
    entity_name: str,
    *,
    logger: Any = None,
    topic: str = "",
) -> Optional[EntityPoseSample]:
    for key in _normalize_entity_keys(entity_name):
        if key in cache:
            return pose6_to_entity_sample(cache[key])

    matches: List[Tuple[str, Pose6]] = []
    for key, pose6 in cache.items():
        if _entity_matches_frame(entity_name, key):
            matches.append((key, pose6))

    if len(matches) == 1:
        return pose6_to_entity_sample(matches[0][1])

    if len(matches) > 1:
        short = entity_name.split("/")[-1].split("::")[-1]
        exact = [m for m in matches if m[0].split("::")[-1].split("/")[-1] == short]
        if len(exact) == 1:
            return pose6_to_entity_sample(exact[0][1])

    if cache and logger is not None:
        log_gazebo_pose_entities_seen(logger, topic=topic or "?", cache=cache)
    return None


def sample_world_poses_once(
    node: Node,
    *,
    world_pose_topic: str,
    world_frame: str,
    collect_sec: float,
) -> Dict[str, Pose6]:
    cache: Dict[str, Pose6] = {}

    def _cb(msg: TFMessage) -> None:
        update_world_pose_cache(msg, cache, world_frame=world_frame)

    sub = node.create_subscription(TFMessage, world_pose_topic, _cb, 10)
    t_end = time.monotonic() + max(0.05, float(collect_sec))
    try:
        while time.monotonic() < t_end and rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.05)
    finally:
        node.destroy_subscription(sub)
    return cache


def wait_for_entity_pose_stable(
    node: Node,
    *,
    entity_name: str,
    world_pose_topic: str,
    world_frame: str = "world",
    timeout_sec: float = 2.0,
    sample_count: int = 5,
    sample_period_sec: float = 0.15,
    xy_threshold_m: float = 0.002,
    z_threshold_m: float = 0.001,
    yaw_threshold_deg: float = 0.5,
    logger: Any = None,
) -> PoseStabilityResult:
    """Muestrea ``pose/info`` hasta estabilidad o timeout."""
    if not world_pose_topic:
        return PoseStabilityResult(
            stable=False,
            samples=0,
            max_delta_xy_m=float("inf"),
            max_delta_z_m=float("inf"),
            max_delta_yaw_deg=float("inf"),
            final_pose=None,
        )

    need = max(2, int(sample_count))
    period = max(0.05, float(sample_period_sec))
    collect_sec = max(period, 0.12)
    deadline = time.monotonic() + max(collect_sec, float(timeout_sec))
    history: List[EntityPoseSample] = []

    while time.monotonic() < deadline and rclpy.ok():
        cache = sample_world_poses_once(
            node,
            world_pose_topic=world_pose_topic,
            world_frame=world_frame,
            collect_sec=collect_sec,
        )
        pose = lookup_entity_pose(
            cache, entity_name, logger=logger, topic=world_pose_topic
        )
        if pose is not None:
            history.append(pose)
            if len(history) >= need:
                window = history[-need:]
                max_dxy = 0.0
                max_dz = 0.0
                max_dyaw = 0.0
                for i in range(1, len(window)):
                    a, b = window[i - 1], window[i]
                    max_dxy = max(max_dxy, math.hypot(b[0] - a[0], b[1] - a[1]))
                    max_dz = max(max_dz, abs(b[2] - a[2]))
                    dy = math.degrees(normalize_angle_rad(b[9] - a[9]))
                    max_dyaw = max(max_dyaw, abs(dy))
                stable = (
                    max_dxy <= float(xy_threshold_m)
                    and max_dz <= float(z_threshold_m)
                    and max_dyaw <= float(yaw_threshold_deg)
                )
                if stable:
                    if logger is not None:
                        try:
                            logger.info(
                                "[SPAWN_POSE_STABLE] entity=%s stable=true samples=%d "
                                "max_delta_xy=%.4f max_delta_z=%.4f max_delta_yaw_deg=%.3f"
                                % (
                                    entity_name,
                                    len(window),
                                    max_dxy,
                                    max_dz,
                                    max_dyaw,
                                )
                            )
                        except Exception:
                            pass
                    return PoseStabilityResult(
                        stable=True,
                        samples=len(window),
                        max_delta_xy_m=max_dxy,
                        max_delta_z_m=max_dz,
                        max_delta_yaw_deg=max_dyaw,
                        final_pose=window[-1],
                    )
        time.sleep(period)

    final = history[-1] if history else None
    if logger is not None:
        try:
            logger.info(
                "[SPAWN_POSE_STABLE] entity=%s stable=false samples=%d "
                "max_delta_xy=n/a (timeout)"
                % (entity_name, len(history))
            )
        except Exception:
            pass
    return PoseStabilityResult(
        stable=False,
        samples=len(history),
        max_delta_xy_m=float("inf"),
        max_delta_z_m=float("inf"),
        max_delta_yaw_deg=float("inf"),
        final_pose=final,
    )


def log_spawn_requested_pose(
    logger: Any,
    *,
    label: str,
    entity_name: str,
    position_xyz: Tuple[float, float, float],
    yaw_rad: float,
) -> None:
    if logger is None:
        return
    try:
        logger.info(
            "[SPAWN_REQUESTED_POSE] label=%s entity=%s "
            "requested_position=(%.4f,%.4f,%.4f) requested_yaw=%.3fdeg"
            % (
                label,
                entity_name,
                position_xyz[0],
                position_xyz[1],
                position_xyz[2],
                math.degrees(yaw_rad),
            )
        )
    except Exception:
        pass


def log_spawn_actual_pose(
    logger: Any,
    *,
    label: str,
    entity_name: str,
    requested_position: Tuple[float, float, float],
    requested_yaw_rad: float,
    actual: EntityPoseSample,
) -> None:
    if logger is None:
        return
    ax, ay, az = actual[0], actual[1], actual[2]
    rx, ry, rz = requested_position
    delta_xy = math.hypot(ax - rx, ay - ry)
    delta_z = abs(az - rz)
    delta_yaw = math.degrees(
        normalize_angle_rad(actual[9] - requested_yaw_rad)
    )
    try:
        logger.info(
            "[SPAWN_ACTUAL_POSE] label=%s entity=%s "
            "actual_position=(%.4f,%.4f,%.4f) actual_rpy=(%.3f,%.3f,%.3f)deg "
            "delta_xy_m=%.4f delta_z_m=%.4f delta_yaw_deg=%.3f"
            % (
                label,
                entity_name,
                ax, ay, az,
                math.degrees(actual[7]),
                math.degrees(actual[8]),
                math.degrees(actual[9]),
                delta_xy,
                delta_z,
                delta_yaw,
            )
        )
    except Exception:
        pass


def log_runtime_scene_gt_update_from_gazebo(
    logger: Any,
    *,
    label: str,
    entity_name: str,
    semantic_center: Tuple[float, float, float],
    top_face_center: Tuple[float, float, float],
    yaw_real: float,
    tilt_deg: float,
) -> None:
    if logger is None:
        return
    try:
        logger.info(
            "[RUNTIME_SCENE_GT_UPDATE_FROM_GAZEBO] label=%s entity=%s "
            "semantic_center=(%.4f,%.4f,%.4f) top_face_center=(%.4f,%.4f,%.4f) "
            "yaw_real=%.3fdeg tilt_deg=%.3f "
            "source=actual_gazebo_pose_after_settle"
            % (
                label,
                entity_name,
                semantic_center[0], semantic_center[1], semantic_center[2],
                top_face_center[0], top_face_center[1], top_face_center[2],
                math.degrees(yaw_real),
                tilt_deg,
            )
        )
    except Exception:
        pass


def log_runtime_scene_pose_tilted(
    logger: Any,
    *,
    label: str,
    entity_name: str,
    roll_deg: float,
    pitch_deg: float,
    max_deg: float,
) -> None:
    if logger is None:
        return
    try:
        logger.info(
            "[RUNTIME_SCENE_POSE_TILTED] label=%s entity=%s roll=%.3fdeg pitch=%.3fdeg "
            "max_upright_roll_pitch_deg=%.3f"
            % (label, entity_name, roll_deg, pitch_deg, max_deg)
        )
    except Exception:
        pass


def build_gt_entry_from_actual_gazebo_pose(
    *,
    entity_name: str,
    label: str,
    actual: EntityPoseSample,
    width_m: float,
    length_m: float,
    height_m: float,
    spawn_seed: Optional[int] = None,
    role: str = "unknown",
    logger: Any = None,
) -> Optional[Dict[str, Any]]:
    """GT desde pose real de Gazebo (origen modelo + orientación)."""
    lb = str(label).strip().lower()
    origin = (float(actual[0]), float(actual[1]), float(actual[2]))
    quat = (float(actual[3]), float(actual[4]), float(actual[5]), float(actual[6]))
    yaw_real = float(actual[9])

    if is_known_spawn_geometry_box_label(lb):
        sem = semantic_center_from_gazebo_model_origin(origin, quat, lb)
        top_c = top_face_center_from_semantic_center(sem, quat, lb)
        fields = build_gt_fields_from_semantic_center(
            lb, sem, yaw_real, logger=logger
        )
        entry: Dict[str, Any] = {
            "entity_name": str(entity_name),
            "label": lb,
            "world_frame": "world",
            "dims_m": [float(width_m), float(length_m), float(height_m)],
            "width_m": float(width_m),
            "length_m": float(length_m),
            "height_m": float(height_m),
            "spawn_seed": spawn_seed,
            "role": str(role),
            "gazebo_origin_world": list(origin),
            "pose_readback_source": "actual_gazebo_pose_after_settle",
            **fields,
        }
        pw = entry.setdefault("pose_world", {})
        pw.update(
            {
                "qx": quat[0],
                "qy": quat[1],
                "qz": quat[2],
                "qw": quat[3],
                "roll": float(actual[7]),
                "pitch": float(actual[8]),
            }
        )
        log_runtime_scene_gt_update_from_gazebo(
            logger,
            label=lb,
            entity_name=entity_name,
            semantic_center=sem,
            top_face_center=top_c,
            yaw_real=yaw_real,
            tilt_deg=max_tilt_deg_from_sample(actual),
        )
        from panda_vision.spawn.known_object_geometry import enrich_runtime_scene_object_fields

        return enrich_runtime_scene_object_fields(entry, role=role, logger=logger)

    return make_gt_object_entry(
        entity_name=entity_name,
        label=lb,
        x=origin[0],
        y=origin[1],
        z=origin[2],
        roll=float(actual[7]),
        pitch=float(actual[8]),
        yaw=yaw_real,
        qx=quat[0],
        qy=quat[1],
        qz=quat[2],
        qw=quat[3],
        width_m=float(width_m),
        length_m=float(length_m),
        height_m=float(height_m),
        spawn_seed=spawn_seed,
        role=role,
        logger=logger,
    )


def settle_and_build_gt_entry(
    node: Node,
    *,
    entity_name: str,
    label: str,
    requested_gazebo_xyz: Tuple[float, float, float],
    requested_yaw_rad: float,
    width_m: float,
    length_m: float,
    height_m: float,
    world_pose_topic: str = "",
    params: SpawnPoseReadbackParams,
    spawn_seed: Optional[int] = None,
    role: str = "unknown",
    logger: Any = None,
) -> Optional[Dict[str, Any]]:
    """Espera settle, lee pose estable y construye entrada GT."""
    log_spawn_requested_pose(
        logger,
        label=label,
        entity_name=entity_name,
        position_xyz=requested_gazebo_xyz,
        yaw_rad=requested_yaw_rad,
    )
    if not params.update_runtime_scene_from_actual_gazebo_pose:
        return None

    candidate_topics = resolve_world_pose_topics(node)
    if world_pose_topic.strip():
        if world_pose_topic.strip() not in candidate_topics:
            candidate_topics.insert(0, world_pose_topic.strip())

    t_end = time.monotonic() + max(0.0, float(params.post_spawn_settle_sec))
    while time.monotonic() < t_end and rclpy.ok():
        rclpy.spin_once(node, timeout_sec=0.05)

    ready_topics = gazebo_pose_topics_ready(
        node,
        candidate_topics,
        world_frame=params.world_frame,
        probe_sec=0.5,
        logger=logger,
    )
    if not ready_topics:
        if (
            logger is not None
            and candidate_topics
            and not getattr(node, "_pose_bridge_hint_logged", False)
        ):
            gz_world = ""
            if node.has_parameter("world_name"):
                gz_world = gz_world_name_from_param(
                    str(node.get_parameter("world_name").value)
                )
            try:
                logger.warning(
                    "[GAZEBO_POSE_TOPIC_UNAVAILABLE] topics=%s gz_world=%s "
                    'hint="Spawn OK; GT con pose comandada. Launch: bridge_world_pose_info:=true."'
                    % (candidate_topics, gz_world or "?")
                )
            except Exception:
                pass
        return None

    active_topic = ready_topics[0]

    stability = wait_for_entity_pose_stable(
        node,
        entity_name=entity_name,
        world_pose_topic=active_topic,
        world_frame=params.world_frame,
        timeout_sec=params.pose_stability_timeout_sec,
        sample_count=params.pose_stability_sample_count,
        sample_period_sec=params.pose_stability_sample_period_sec,
        xy_threshold_m=params.pose_stability_xy_threshold_m,
        z_threshold_m=params.pose_stability_z_threshold_m,
        yaw_threshold_deg=params.pose_stability_yaw_threshold_deg,
        logger=logger,
    )
    actual = stability.final_pose
    if actual is None:
        cache = sample_world_poses_once(
            node,
            world_pose_topic=active_topic,
            world_frame=params.world_frame,
            collect_sec=0.35,
        )
        actual = lookup_entity_pose(
            cache, entity_name, logger=logger, topic=active_topic
        )
    if actual is None:
        if logger is not None:
            try:
                logger.warning(
                    "[SPAWN_ACTUAL_POSE] entity=%s no encontrado en %s; "
                    "se omite actualización GT desde Gazebo"
                    % (entity_name, active_topic)
                )
            except Exception:
                pass
        return None

    log_spawn_actual_pose(
        logger,
        label=label,
        entity_name=entity_name,
        requested_position=requested_gazebo_xyz,
        requested_yaw_rad=requested_yaw_rad,
        actual=actual,
    )

    tilt_deg = max_tilt_deg_from_sample(actual)
    if tilt_deg > float(params.max_upright_roll_pitch_deg):
        log_runtime_scene_pose_tilted(
            logger,
            label=label,
            entity_name=entity_name,
            roll_deg=math.degrees(actual[7]),
            pitch_deg=math.degrees(actual[8]),
            max_deg=float(params.max_upright_roll_pitch_deg),
        )
        if params.reject_topdown_if_tilted:
            return None

    return build_gt_entry_from_actual_gazebo_pose(
        entity_name=entity_name,
        label=label,
        actual=actual,
        width_m=width_m,
        length_m=length_m,
        height_m=height_m,
        spawn_seed=spawn_seed,
        role=role,
        logger=logger,
    )
