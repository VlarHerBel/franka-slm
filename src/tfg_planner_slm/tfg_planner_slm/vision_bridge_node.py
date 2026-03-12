import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import json
import time

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
                self.update_object(obj_name, [x, y, z])
        except Exception as e:
            self.get_logger().warn(f"Error traduciendo coordenadas legacy: {e}")

    def detections_callback(self, msg):
        try:
            payload = json.loads(msg.data)
        except json.JSONDecodeError as exc:
            self.get_logger().warn(f"Error parseando /detections_3d: {exc}")
            return

        detections = payload.get("detections", [])
        for detection in detections:
            object_id = detection.get("id") or detection.get("label")
            position = detection.get("position")
            if not object_id or not isinstance(position, list) or len(position) != 3:
                continue
            self.update_object(object_id, position)

    def update_object(self, object_id, position):
        current_time = time.time()
        new_position = [float(value) for value in position]

        if object_id in self.detected_objects:
            old_position = self.detected_objects[object_id]
            alpha = self.smoothing_alpha
            smoothed = [
                round((1.0 - alpha) * old_position[index] + alpha * new_position[index], 4)
                for index in range(3)
            ]
            self.detected_objects[object_id] = smoothed
        else:
            self.detected_objects[object_id] = [round(value, 4) for value in new_position]

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