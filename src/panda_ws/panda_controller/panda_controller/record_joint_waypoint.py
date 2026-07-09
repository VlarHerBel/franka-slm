#!/usr/bin/env python3
"""Graba la pose articular actual en tfg_motion_waypoints.yaml."""

from __future__ import annotations

import sys
from typing import Dict, Optional

import rclpy
from rclpy.duration import Duration
from rclpy.node import Node
from sensor_msgs.msg import JointState

from panda_controller.tfg_motion_waypoints import (
    PANDA_ARM_JOINT_NAMES,
    ensure_template_structure,
    load_waypoints_file,
    resolve_waypoints_yaml_path,
    save_waypoints_file,
    update_waypoint_from_joint_state,
    waypoint_is_configured,
)


class RecordJointWaypointNode(Node):
    def __init__(self) -> None:
        super().__init__("record_joint_waypoint")
        self.declare_parameter("waypoint_name", "")
        self.declare_parameter("output_yaml", "")
        self.declare_parameter("joint_state_timeout_sec", 5.0)

        self._last_joint_state: Optional[JointState] = None
        self.create_subscription(JointState, "/joint_states", self._joint_cb, 10)

    def _joint_cb(self, msg: JointState) -> None:
        self._last_joint_state = msg

    def _wait_joint_state(self, timeout_sec: float) -> Optional[JointState]:
        deadline = self.get_clock().now() + Duration(seconds=float(timeout_sec))
        while rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.1)
            if self._last_joint_state is not None:
                return self._last_joint_state
            if self.get_clock().now() >= deadline:
                return None
        return None

    def _extract_arm_joints(self, msg: JointState) -> Dict[str, float]:
        name_to_pos: Dict[str, float] = {}
        for name, pos in zip(msg.name, msg.position):
            name_to_pos[str(name)] = float(pos)
        missing = [j for j in PANDA_ARM_JOINT_NAMES if j not in name_to_pos]
        if missing:
            raise RuntimeError(
                "joint_states no contiene: %s (tiene: %s)"
                % (missing, list(name_to_pos.keys())[:20])
            )
        return {j: name_to_pos[j] for j in PANDA_ARM_JOINT_NAMES}

    def run(self) -> int:
        name = str(self.get_parameter("waypoint_name").value).strip()
        if not name:
            self.get_logger().error(
                "Parámetro waypoint_name vacío. Ejemplo: -p waypoint_name:=box_high"
            )
            return 1

        out_path = resolve_waypoints_yaml_path(
            str(self.get_parameter("output_yaml").value)
        )
        if not out_path:
            self.get_logger().error("No se pudo resolver ruta de output_yaml.")
            return 1

        timeout = float(self.get_parameter("joint_state_timeout_sec").value)
        self.get_logger().info(
            "Esperando /joint_states (timeout=%.1fs)..." % timeout
        )
        js = self._wait_joint_state(timeout)
        if js is None:
            self.get_logger().error("Timeout esperando /joint_states.")
            return 1

        try:
            positions = self._extract_arm_joints(js)
        except RuntimeError as exc:
            self.get_logger().error(str(exc))
            return 1

        data = load_waypoints_file(out_path)
        if not data:
            data = ensure_template_structure({})
        data = update_waypoint_from_joint_state(data, name, positions)
        save_waypoints_file(out_path, data)

        self.get_logger().info(
            "[RECORD_JOINT_WAYPOINT] name=%s saved to %s joints=%s"
            % (
                name,
                out_path,
                [round(positions[j], 6) for j in PANDA_ARM_JOINT_NAMES],
            )
        )
        if waypoint_is_configured(data, name):
            self.get_logger().info("[RECORD_JOINT_WAYPOINT] result=OK")
            return 0
        self.get_logger().error("[RECORD_JOINT_WAYPOINT] result=FAIL")
        return 1


def main(args=None) -> None:
    rclpy.init(args=args)
    node = RecordJointWaypointNode()
    try:
        code = node.run()
    except KeyboardInterrupt:
        code = 1
    finally:
        node.destroy_node()
        rclpy.shutdown()
    sys.exit(code)


if __name__ == "__main__":
    main()
