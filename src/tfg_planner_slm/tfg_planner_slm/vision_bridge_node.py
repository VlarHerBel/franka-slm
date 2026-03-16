import json
import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class VisionBridge(Node):
    def __init__(self):
        super().__init__("vision_bridge")

        self.declare_parameter("smoothing_alpha", 0.35)
        self.declare_parameter("stale_timeout_sec", 1.5)

        self.smoothing_alpha = float(self.get_parameter("smoothing_alpha").value)
        self.stale_timeout_sec = float(
            self.get_parameter("stale_timeout_sec").value
        )

        self.legacy_subscription = self.create_subscription(
            String,
            "/color_coordinates",
            self.legacy_camera_callback,
            10,
        )
        self.detections_subscription = self.create_subscription(
            String,
            "/detections_3d",
            self.detections_callback,
            10,
        )
        self.publisher_ = self.create_publisher(String, "/detected_objects", 10)

        self.detected_objects = {}
        self.last_seen = {}
        self.timer = self.create_timer(0.5, self.publish_state)

        self.get_logger().info(
            "Vision bridge listo. Escuchando /detections_3d y /color_coordinates."
        )

    def legacy_camera_callback(self, msg):
        try:
            parts = msg.data.split(",")
            if len(parts) >= 4:
                tag = parts[0].strip()
                x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                name_map = {"R": "red_cube", "G": "green_cube", "B": "blue_cube"}
                obj_name = name_map.get(tag, tag)
                self.update_object(obj_name, {"position": [x, y, z]})
        except Exception as e:
            self.get_logger().warn(f"Error traduciendo coordenadas legacy: {e}")

    def detections_callback(self, msg):
        try:
            payload = json.loads(msg.data)
        except json.JSONDecodeError as exc:
            self.get_logger().warn(f"Error parseando /detections_3d: {exc}")
            return

        detections = payload.get("detections", [])
        for det in detections:
            object_id = det.get("id") or det.get("label")
            position = det.get("position")
            if not object_id or not isinstance(position, list) or len(position) != 3:
                continue

            enriched = {
                "position": [float(v) for v in position],
                "shape": det.get("shape"),
                "dimensions_m": det.get("dimensions_m", [0.0, 0.0, 0.0]),
                "height_m": float(det.get("height_m", 0.0)),
                "grasp_type": det.get("grasp_type"),
                "grasp_yaw_deg": float(det.get("grasp_yaw_deg", 0.0)),
                "color_hint": det.get("color_hint"),
            }
            self.update_object(object_id, enriched)

    def update_object(self, object_id, obj_data):
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

    def prune_stale_objects(self):
        cutoff = time.time() - self.stale_timeout_sec
        stale_ids = [
            object_id
            for object_id, last_seen in self.last_seen.items()
            if last_seen < cutoff
        ]
        for object_id in stale_ids:
            self.last_seen.pop(object_id, None)
            self.detected_objects.pop(object_id, None)

    def publish_state(self):
        self.prune_stale_objects()
        if self.detected_objects:
            msg = String()
            msg.data = json.dumps(self.detected_objects)
            self.publisher_.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = VisionBridge()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()