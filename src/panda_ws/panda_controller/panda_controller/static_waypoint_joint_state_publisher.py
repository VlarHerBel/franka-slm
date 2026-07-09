#!/usr/bin/env python3
"""Publica /joint_states fijos desde tfg_motion_waypoints.yaml (para MoveIt plan-only)."""

from __future__ import annotations

import sys
from typing import List, Optional, Sequence

from panda_controller.tfg_motion_waypoints import (
    get_waypoint_joint_positions,
    load_waypoints_file,
    resolve_waypoints_yaml_path,
)


def _finger_positions() -> List[float]:
    return [0.0399, 0.0399]


def main(argv: Optional[Sequence[str]] = None) -> int:
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import JointState

    rclpy.init()
    node = Node("static_waypoint_joint_state_publisher")
    node.declare_parameter("waypoint", "pick_workspace_ready")
    node.declare_parameter("waypoints_yaml", "")
    node.declare_parameter("publish_hz", 30.0)

    waypoint = str(node.get_parameter("waypoint").value)
    waypoints_yaml = str(node.get_parameter("waypoints_yaml").value)
    hz = float(node.get_parameter("publish_hz").value)

    wp_path = resolve_waypoints_yaml_path(waypoints_yaml)
    data = load_waypoints_file(wp_path)
    arm = get_waypoint_joint_positions(data, waypoint)
    if arm is None:
        node.get_logger().error(
            "Waypoint '%s' no encontrado en %s" % (waypoint, wp_path)
        )
        node.destroy_node()
        rclpy.shutdown()
        return 1

    pub = node.create_publisher(JointState, "/joint_states", 10)
    names = [
        "panda_joint1",
        "panda_joint2",
        "panda_joint3",
        "panda_joint4",
        "panda_joint5",
        "panda_joint6",
        "panda_joint7",
        "panda_finger_joint1",
        "panda_finger_joint2",
    ]
    positions = [float(v) for v in arm] + _finger_positions()
    period = 1.0 / max(1.0, hz)

    def _tick() -> None:
        msg = JointState()
        msg.header.stamp = node.get_clock().now().to_msg()
        msg.name = list(names)
        msg.position = list(positions)
        pub.publish(msg)

    node.create_timer(period, _tick)
    node.get_logger().info(
        "[STATIC_JOINT_STATE]\nwaypoint=%s\nyaml=%s\nhz=%.1f"
        % (waypoint, wp_path, hz)
    )
    _tick()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
