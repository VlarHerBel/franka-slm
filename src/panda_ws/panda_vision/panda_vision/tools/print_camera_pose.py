#!/usr/bin/env python3
"""Imprime la pose de la cámara RGB-D respecto a la base del robot (TF)."""

from __future__ import annotations

import math
import sys

import rclpy
import tf2_ros
import tf_transformations
from rclpy.node import Node


class PrintCameraPose(Node):
    def __init__(self) -> None:
        super().__init__("print_camera_pose")
        self.declare_parameter("camera_frame", "camera_depth_optical_frame")
        self.declare_parameter("base_frame", "panda_link0")
        self.declare_parameter("lookup_timeout_sec", 2.0)
        self._camera_frame = str(self.get_parameter("camera_frame").value).strip()
        self._base_frame = str(self.get_parameter("base_frame").value).strip()
        self._timeout = float(self.get_parameter("lookup_timeout_sec").value)
        self._tf_buffer = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, self)

    def run_once(self) -> int:
        try:
            tf_msg = self._tf_buffer.lookup_transform(
                self._base_frame,
                self._camera_frame,
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=self._timeout),
            )
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as exc:
            self.get_logger().error(
                "[CAMERA_POSE] TF %s -> %s failed: %s"
                % (self._camera_frame, self._base_frame, exc)
            )
            return 1

        t = tf_msg.transform.translation
        q = tf_msg.transform.rotation
        mat = tf_transformations.quaternion_matrix([q.x, q.y, q.z, q.w])
        mat[0, 3] = t.x
        mat[1, 3] = t.y
        mat[2, 3] = t.z
        roll, pitch, yaw = tf_transformations.euler_from_matrix(mat)

        self.get_logger().info(
            "[CAMERA_POSE] frame=%s pose_in_%s xyz=(%.4f, %.4f, %.4f) rpy_deg=(%.2f, %.2f, %.2f)"
            % (
                self._camera_frame,
                self._base_frame,
                float(t.x),
                float(t.y),
                float(t.z),
                math.degrees(roll),
                math.degrees(pitch),
                math.degrees(yaw),
            )
        )
        return 0


def main(args=None) -> None:
    rclpy.init(args=args)
    node = PrintCameraPose()
    try:
        rclpy.spin_once(node, timeout_sec=0.5)
        code = node.run_once()
    finally:
        node.destroy_node()
        rclpy.shutdown()
    sys.exit(code)


if __name__ == "__main__":
    main()
