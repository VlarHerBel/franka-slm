import json
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import rclpy
import yaml
from ament_index_python.packages import get_package_share_directory
from rclpy.node import Node
from std_msgs.msg import String


def _to_xyz(value: Any) -> Optional[list]:
    if not isinstance(value, list) or len(value) != 3:
        return None
    try:
        return [float(value[0]), float(value[1]), float(value[2])]
    except (TypeError, ValueError):
        return None


class VisionBridge(Node):
    def __init__(self):
        super().__init__("vision_bridge")

        default_params = (
            Path(get_package_share_directory("tfg_planner_slm"))
            / "config"
            / "pick_place_params.yaml"
        )
        self.declare_parameter("pick_params_path", str(default_params))
        self.declare_parameter("smoothing_alpha", 0.35)
        self.declare_parameter("stale_timeout_sec", 1.5)
        self.declare_parameter("modern_topic", "/vision_to_executor")
        self.declare_parameter("legacy_topic", "/detections_3d")
        self.declare_parameter("publish_topic", "/detected_objects")

        cfg = self._load_yaml(Path(str(self.get_parameter("pick_params_path").value)))
        vision_cfg = cfg.get("vision", {}) if isinstance(cfg, dict) else {}
        general_cfg = cfg.get("general", {}) if isinstance(cfg, dict) else {}

        self.smoothing_alpha = float(
            self.get_parameter("smoothing_alpha").value
            if "smoothing_alpha" not in vision_cfg
            else vision_cfg.get("smoothing_alpha", 0.35)
        )
        self.stale_timeout_sec = float(
            self.get_parameter("stale_timeout_sec").value
            if "stale_timeout_sec" not in vision_cfg
            else vision_cfg.get("stale_timeout_sec", 1.5)
        )
        modern_topic = str(
            vision_cfg.get(
                "preferred_input_topic", self.get_parameter("modern_topic").value
            )
        )
        legacy_topic = str(
            vision_cfg.get("legacy_input_topic", self.get_parameter("legacy_topic").value)
        )
        publish_topic = str(
            general_cfg.get(
                "detected_objects_topic", self.get_parameter("publish_topic").value
            )
        )

        self.create_subscription(String, "/color_coordinates", self._legacy_color_cb, 10)
        self.create_subscription(String, legacy_topic, self._legacy_detections_cb, 10)
        self.create_subscription(String, modern_topic, self._modern_detections_cb, 10)
        self.publisher_ = self.create_publisher(String, publish_topic, 10)

        self.detected_objects: Dict[str, Dict[str, Any]] = {}
        self.last_seen: Dict[str, float] = {}
        self.timer = self.create_timer(0.4, self.publish_state)

        self.get_logger().info(
            "Vision bridge listo. "
            f"moderno={modern_topic}, legacy={legacy_topic}, salida={publish_topic}"
        )

    @staticmethod
    def _load_yaml(path: Path) -> Dict[str, Any]:
        if not path.is_file():
            return {}
        try:
            with path.open("r", encoding="utf-8") as handle:
                loaded = yaml.safe_load(handle) or {}
                if isinstance(loaded, dict):
                    return loaded
        except Exception:
            pass
        return {}

    def _legacy_color_cb(self, msg: String) -> None:
        try:
            parts = msg.data.split(",")
            if len(parts) < 4:
                return
            tag = parts[0].strip()
            x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
            name_map = {"R": "red_cube", "G": "green_cube", "B": "blue_cube"}
            obj_name = name_map.get(tag, tag)
            self._update_object(obj_name, {"position": [x, y, z], "label": obj_name})
        except Exception as exc:
            self.get_logger().warn(f"Error traduciendo /color_coordinates: {exc}")

    def _legacy_detections_cb(self, msg: String) -> None:
        try:
            payload = json.loads(msg.data)
        except json.JSONDecodeError as exc:
            self.get_logger().warn(f"Error parseando detecciones legacy: {exc}")
            return
        detections = payload.get("detections", [])
        self._ingest_detections(detections)

    def _modern_detections_cb(self, msg: String) -> None:
        try:
            payload = json.loads(msg.data)
        except json.JSONDecodeError as exc:
            self.get_logger().warn(f"Error parseando detecciones modernas: {exc}")
            return
        detections = payload.get("objects", [])
        self._ingest_detections(detections)

    def _ingest_detections(self, detections: Iterable[Dict[str, Any]]) -> None:
        for det in detections:
            object_id = str(det.get("id") or det.get("instance_id") or det.get("label") or "").strip()
            if not object_id:
                continue
            position = _to_xyz(det.get("position")) or _to_xyz(det.get("centroid_base"))
            if position is None:
                continue
            enriched = {
                "id": object_id,
                "label": str(det.get("label") or object_id),
                "position": position,
                "score": float(det.get("score", det.get("confidence", 0.0))),
                "shape": det.get("shape"),
                "dimensions_m": _to_xyz(det.get("dimensions_m")) or [0.0, 0.0, 0.0],
                "height_m": float(det.get("height_m", 0.0)),
                "top_z_m": float(det.get("top_z_m", position[2])),
                "grasp_type": det.get("grasp_type", "top_grasp"),
                "grasp_yaw_deg": float(det.get("grasp_yaw_deg", 0.0)),
                "grasp_yaw_rad": float(det.get("grasp_yaw_rad", 0.0)),
                "approach_position": _to_xyz(det.get("approach_position")),
                "grasp_position": _to_xyz(det.get("grasp_position")),
                "color_hint": det.get("color_hint"),
            }
            self._update_object(object_id, enriched)

    def _update_object(self, object_id: str, obj_data: Dict[str, Any]) -> None:
        current_time = time.time()
        new_pos = obj_data["position"]

        if object_id in self.detected_objects:
            old = self.detected_objects[object_id]
            old_pos = old.get("position", new_pos)
            alpha = self.smoothing_alpha
            smoothed = [
                round((1.0 - alpha) * old_pos[i] + alpha * new_pos[i], 4)
                for i in range(3)
            ]
            obj_data["position"] = smoothed
        else:
            obj_data["position"] = [round(v, 4) for v in new_pos]

        self.detected_objects[object_id] = obj_data
        self.last_seen[object_id] = current_time

    def _prune_stale_objects(self) -> None:
        cutoff = time.time() - self.stale_timeout_sec
        stale_ids = [
            object_id
            for object_id, last_seen in self.last_seen.items()
            if last_seen < cutoff
        ]
        for object_id in stale_ids:
            self.last_seen.pop(object_id, None)
            self.detected_objects.pop(object_id, None)

    def publish_state(self) -> None:
        self._prune_stale_objects()
        if not self.detected_objects:
            return
        msg = String()
        msg.data = json.dumps(self.detected_objects)
        self.publisher_.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = VisionBridge()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()