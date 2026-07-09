"""Ground-truth persistente de objetos runtime YCB en ``/runtime_scene/gt_objects``.

``pose_world`` en cajas conocidas = centro semántico del cuboide (``geometry_center``).
El offset del modelo Gazebo solo se aplica en spawn, no en percepción.
"""

from __future__ import annotations

import json
import time
from typing import Any, Dict, List, Optional

import rclpy
from panda_vision_interfaces.srv import SetRuntimeSceneGt
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import String
from std_srvs.srv import Trigger

from panda_vision.spawn.runtime_scene_gt_geometry import (
    SOURCE_POSE_SEMANTICS_GEOMETRY_CENTER,
    build_gt_fields_from_semantic_center,
    enrich_gt_object_entry,
    is_known_spawn_geometry_box_label,
)

GT_OBJECTS_TOPIC = "/runtime_scene/gt_objects"
GT_SET_SERVICE = "/runtime_scene/gt/set"
GT_CLEAR_SERVICE = "/runtime_scene/gt/clear"
SCHEMA_VERSION = "1.2"
DEFAULT_GT_SERVICE_WAIT_SEC = 8.0


def gt_objects_qos() -> QoSProfile:
    return QoSProfile(
        reliability=ReliabilityPolicy.RELIABLE,
        durability=DurabilityPolicy.TRANSIENT_LOCAL,
        history=HistoryPolicy.KEEP_LAST,
        depth=1,
    )


def make_gt_object_entry(
    *,
    entity_name: str,
    label: str,
    x: float,
    y: float,
    z: float,
    roll: float,
    pitch: float,
    yaw: float,
    qx: float,
    qy: float,
    qz: float,
    qw: float,
    width_m: float,
    length_m: float,
    height_m: float,
    world_frame: str = "world",
    spawn_seed: Optional[int] = None,
    role: str = "unknown",
    logger: Any = None,
) -> Dict[str, Any]:
    """Crea entrada GT. Para cajas conocidas, (x,y,z) = centro semántico del cuboide."""
    label_l = str(label).strip().lower()
    entry: Dict[str, Any] = {
        "entity_name": str(entity_name),
        "label": label_l,
        "world_frame": str(world_frame),
        "dims_m": [float(width_m), float(length_m), float(height_m)],
        "width_m": float(width_m),
        "length_m": float(length_m),
        "height_m": float(height_m),
        "spawn_seed": spawn_seed,
    }

    if is_known_spawn_geometry_box_label(label_l):
        fields = build_gt_fields_from_semantic_center(
            label_l, (float(x), float(y), float(z)), float(yaw), logger=logger
        )
        entry.update(fields)
        pw = entry.setdefault("pose_world", {})
        pw.update(
            {
                "qx": float(qx),
                "qy": float(qy),
                "qz": float(qz),
                "qw": float(qw),
                "roll": float(roll),
                "pitch": float(pitch),
            }
        )
        entry["source_pose_semantics"] = SOURCE_POSE_SEMANTICS_GEOMETRY_CENTER
        from panda_vision.spawn.known_object_geometry import enrich_runtime_scene_object_fields

        return enrich_runtime_scene_object_fields(entry, role=role, logger=logger)

    entry["source_pose_semantics"] = "model_link_origin"
    entry["pose_world"] = {
        "x": float(x),
        "y": float(y),
        "z": float(z),
        "roll": float(roll),
        "pitch": float(pitch),
        "yaw": float(yaw),
        "qx": float(qx),
        "qy": float(qy),
        "qz": float(qz),
        "qw": float(qw),
    }
    entry["yaw_rad"] = float(yaw)
    from panda_vision.spawn.known_object_geometry import enrich_runtime_scene_object_fields

    return enrich_runtime_scene_object_fields(entry, role=role, logger=logger)


def build_gt_payload(
    objects: List[Dict[str, Any]],
    *,
    world_frame: str = "world",
    stamp_sec: Optional[float] = None,
) -> Dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "stamp_sec": float(stamp_sec if stamp_sec is not None else time.time()),
        "world_frame": str(world_frame),
        "objects": list(objects),
    }


class RuntimeSceneGtStore:
    """Estado GT en memoria (sin publicar)."""

    def __init__(self, *, world_frame: str = "world") -> None:
        self._world_frame = str(world_frame)
        self._objects: List[Dict[str, Any]] = []

    @property
    def objects(self) -> List[Dict[str, Any]]:
        return list(self._objects)

    def clear(self) -> None:
        self._objects = []

    def replace_all(self, objects: List[Dict[str, Any]]) -> None:
        self._objects = [enrich_gt_object_entry(dict(o)) for o in objects]

    def upsert(self, entry: Dict[str, Any]) -> None:
        enriched = enrich_gt_object_entry(dict(entry))
        name = str(enriched.get("entity_name", "")).strip()
        out: List[Dict[str, Any]] = []
        replaced = False
        for obj in self._objects:
            if str(obj.get("entity_name", "")).strip() == name:
                out.append(enriched)
                replaced = True
            else:
                out.append(obj)
        if not replaced:
            out.append(enriched)
        self._objects = out

    def remove_entity(self, entity_name: str) -> None:
        name = str(entity_name).strip()
        self._objects = [
            o
            for o in self._objects
            if str(o.get("entity_name", "")).strip() != name
        ]

    def payload(self) -> Dict[str, Any]:
        return build_gt_payload(self._objects, world_frame=self._world_frame)

    def payload_json(self) -> str:
        return json.dumps(self.payload())


class RuntimeSceneGtPublisher:
    """Publicador latched asociado a un ``RuntimeSceneGtStore``."""

    def __init__(
        self,
        node: Node,
        *,
        store: Optional[RuntimeSceneGtStore] = None,
        topic: str = GT_OBJECTS_TOPIC,
        world_frame: str = "world",
    ) -> None:
        self._node = node
        self._store = store or RuntimeSceneGtStore(world_frame=world_frame)
        self._pub = node.create_publisher(String, topic, gt_objects_qos())

    @property
    def store(self) -> RuntimeSceneGtStore:
        return self._store

    def clear(self) -> None:
        self._store.clear()
        self.publish(log_tag="RUNTIME_SCENE_GT_CLEAR")

    def replace_all(self, objects: List[Dict[str, Any]]) -> None:
        self._store.replace_all(objects)
        self.publish(log_tag="RUNTIME_SCENE_GT_UPDATE")

    def upsert(self, entry: Dict[str, Any]) -> None:
        self._store.upsert(entry)
        self.publish(log_tag="RUNTIME_SCENE_GT_UPDATE")

    def remove_entity(self, entity_name: str) -> None:
        self._store.remove_entity(entity_name)
        self.publish(log_tag="RUNTIME_SCENE_GT_UPDATE")

    def publish(self, *, log_tag: str = "RUNTIME_SCENE_GT_PUBLISH") -> None:
        msg = String()
        msg.data = self._store.payload_json()
        self._pub.publish(msg)
        self._node.get_logger().info(
            "[%s] n=%d topic=%s"
            % (log_tag, len(self._store.objects), GT_OBJECTS_TOPIC)
        )


class RuntimeSceneGtClient:
    """Cliente para actualizar el nodo persistente ``runtime_scene_gt_node``."""

    def __init__(
        self,
        node: Node,
        *,
        world_frame: str = "world",
        service_wait_sec: float = DEFAULT_GT_SERVICE_WAIT_SEC,
    ) -> None:
        self._node = node
        self._world_frame = str(world_frame)
        self._service_wait_sec = float(service_wait_sec)
        self._set_client = node.create_client(SetRuntimeSceneGt, GT_SET_SERVICE)
        self._clear_client = node.create_client(Trigger, GT_CLEAR_SERVICE)

    def _wait_ready(self, client, label: str) -> bool:
        if client.service_is_ready():
            return True
        if not client.wait_for_service(timeout_sec=self._service_wait_sec):
            self._node.get_logger().error(
                "[RUNTIME_SCENE_GT_UPDATE] servicio %s no disponible tras %.1fs "
                "(¿runtime_scene_gt_node en ejecución?)"
                % (label, self._service_wait_sec)
            )
            return False
        return True

    def _call_set(self, payload: Dict[str, Any]) -> bool:
        if not self._wait_ready(self._set_client, GT_SET_SERVICE):
            return False
        req = SetRuntimeSceneGt.Request()
        req.data = json.dumps(payload)
        future = self._set_client.call_async(req)
        rclpy.spin_until_future_complete(self._node, future, timeout_sec=self._service_wait_sec)
        if not future.done():
            self._node.get_logger().error(
                "[RUNTIME_SCENE_GT_UPDATE] timeout llamando %s" % GT_SET_SERVICE
            )
            return False
        result = future.result()
        if result is None or not result.success:
            msg = result.message if result else "sin respuesta"
            self._node.get_logger().error(
                "[RUNTIME_SCENE_GT_UPDATE] fallo %s: %s" % (GT_SET_SERVICE, msg)
            )
            return False
        n = len(payload.get("objects", []))
        self._node.get_logger().info(
            "[RUNTIME_SCENE_GT_UPDATE] n=%d via_client=%s"
            % (n, GT_SET_SERVICE)
        )
        return True

    def replace_all(self, objects: List[Dict[str, Any]]) -> bool:
        payload = build_gt_payload(objects, world_frame=self._world_frame)
        return self._call_set(payload)

    def clear(self) -> bool:
        if not self._wait_ready(self._clear_client, GT_CLEAR_SERVICE):
            return False
        future = self._clear_client.call_async(Trigger.Request())
        rclpy.spin_until_future_complete(
            self._node, future, timeout_sec=self._service_wait_sec
        )
        if not future.done():
            self._node.get_logger().error(
                "[RUNTIME_SCENE_GT_CLEAR] timeout llamando %s" % GT_CLEAR_SERVICE
            )
            return False
        result = future.result()
        if result is None or not result.success:
            self._node.get_logger().error(
                "[RUNTIME_SCENE_GT_CLEAR] fallo %s" % GT_CLEAR_SERVICE
            )
            return False
        self._node.get_logger().info("[RUNTIME_SCENE_GT_CLEAR] n=0")
        return True


def parse_gt_payload_from_json(raw: str) -> tuple[Optional[Dict[str, Any]], Optional[str]]:
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        return None, str(exc)
    if not isinstance(data, dict):
        return None, "payload must be a JSON object"
    return data, None


def parse_gt_payload(msg: String) -> Dict[str, Any]:
    data, _ = parse_gt_payload_from_json(msg.data)
    return data or {}


def find_gt_object_nearest_xy(
    gt_by_entity: Dict[str, Dict[str, Any]],
    xy_base: Tuple[float, float],
    *,
    label: Optional[str] = None,
    entity_prefix: str = "runtime_ycb",
    max_dist_m: float = 0.25,
) -> Optional[tuple[str, Dict[str, Any]]]:
    """Empareja entidad GT más cercana en XY (opcionalmente mismo label)."""
    import math

    best: Optional[tuple[str, Dict[str, Any], float]] = None
    p = entity_prefix.strip()
    lb = str(label).strip().lower() if label else ""
    for name, obj in gt_by_entity.items():
        short = name.split("::")[-1]
        if p and not short.startswith(p):
            continue
        if lb and str(obj.get("label", "")).strip().lower() != lb:
            continue
        sem = obj.get("semantic_box_center_base") or obj.get("gt_geometry_center_base")
        if not (isinstance(sem, (list, tuple)) and len(sem) >= 2):
            continue
        d = math.hypot(float(xy_base[0]) - float(sem[0]), float(xy_base[1]) - float(sem[1]))
        if d > float(max_dist_m):
            continue
        if best is None or d < best[2]:
            best = (short, obj, d)
    if best is None:
        return None
    return best[0], best[1]


def find_gt_object_for_label(
    gt_by_entity: Dict[str, Dict[str, Any]],
    label: str,
    *,
    entity_prefix: str = "runtime_ycb",
) -> Optional[tuple[str, Dict[str, Any]]]:
    lb = str(label).strip().lower()
    p = entity_prefix.strip()
    matches: List[tuple[str, Dict[str, Any]]] = []
    for name, obj in gt_by_entity.items():
        short = name.split("::")[-1]
        if str(obj.get("label", "")).strip().lower() != lb:
            continue
        if p and not short.startswith(p):
            continue
        matches.append((short, obj))
    if not matches:
        return None
    matches.sort(key=lambda x: x[0])
    return matches[0]
