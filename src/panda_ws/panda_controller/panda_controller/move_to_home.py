#!/usr/bin/env python3
from typing import Dict, List, Optional

import rclpy
from control_msgs.action import FollowJointTrajectory
from rclpy.action import ActionClient
from rclpy.duration import Duration
from rclpy.node import Node
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint


class MoveToHomeNode(Node):
    def __init__(self) -> None:
        super().__init__("move_to_home")
        self.declare_parameter("use_action", True)
        self.declare_parameter("action_wait_timeout_sec", 10.0)
        self.declare_parameter("result_timeout_sec", 12.0)
        self.declare_parameter("joint_state_verify_timeout_sec", 8.0)
        self.declare_parameter("home_tolerance_rad", 0.03)

        self.declare_parameter("home_joint1", -0.011715794351100829)
        self.declare_parameter("home_joint2", -0.7718978353758489)
        self.declare_parameter("home_joint3", -0.05815098072345806)
        self.declare_parameter("home_joint4", -2.264417692171618)
        self.declare_parameter("home_joint5", -0.03860755520570227)
        self.declare_parameter("home_joint6", 1.5298565212648159)
        self.declare_parameter("home_joint7", 0.7333372530280367)
        self.declare_parameter("home_motion_duration_sec", 4.0)
        self.declare_parameter("open_gripper_at_home", False)
        self.declare_parameter("close_gripper_at_home", False)
        self.declare_parameter("gripper_close_width_m", 0.0)
        self.declare_parameter("gripper_open_width_m", 0.04)
        self.declare_parameter("startup_max_attempts", 3)
        self.declare_parameter("retry_delay_sec", 3.0)

        self._arm_client = ActionClient(
            self,
            FollowJointTrajectory,
            "/arm_controller/follow_joint_trajectory",
        )
        self._gripper_client = ActionClient(
            self,
            FollowJointTrajectory,
            "/gripper_controller/follow_joint_trajectory",
        )
        self._arm_pub = self.create_publisher(
            JointTrajectory, "/arm_controller/joint_trajectory", 10
        )
        self._gripper_pub = self.create_publisher(
            JointTrajectory, "/gripper_controller/joint_trajectory", 10
        )
        self._last_joint_state: Optional[JointState] = None
        self.create_subscription(JointState, "/joint_states", self._joint_cb, 10)

    def _joint_cb(self, msg: JointState) -> None:
        self._last_joint_state = msg

    def _home_positions(self) -> List[float]:
        return [float(self.get_parameter(f"home_joint{i}").value) for i in range(1, 8)]

    def _trajectory(self, names: List[str], positions: List[float], duration_sec: float) -> JointTrajectory:
        traj = JointTrajectory()
        traj.joint_names = names
        point = JointTrajectoryPoint()
        point.positions = positions
        point.time_from_start.sec = int(duration_sec)
        point.time_from_start.nanosec = int((duration_sec - int(duration_sec)) * 1e9)
        traj.points = [point]
        return traj

    def _wait_for_joint_state(self, timeout_sec: float) -> bool:
        start = self.get_clock().now()
        while rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.1)
            if self._last_joint_state is not None:
                return True
            if (self.get_clock().now() - start) > Duration(seconds=timeout_sec):
                return False
        return False

    def _send_action_goal(
        self,
        client: ActionClient,
        action_name: str,
        trajectory: JointTrajectory,
        wait_timeout_sec: float,
        result_timeout_sec: float,
    ) -> bool:
        self.get_logger().info(f"Esperando action server {action_name}")
        if not client.wait_for_server(timeout_sec=wait_timeout_sec):
            self.get_logger().error(f"Action server no disponible: {action_name}")
            return False

        goal = FollowJointTrajectory.Goal()
        goal.trajectory = trajectory
        send_future = client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, send_future, timeout_sec=wait_timeout_sec)
        if not send_future.done():
            self.get_logger().error(f"Timeout enviando goal a {action_name}")
            return False
        goal_handle = send_future.result()
        if goal_handle is None:
            self.get_logger().error(f"Goal rechazado por {action_name} (sin goal_handle)")
            return False
        self.get_logger().info("Goal enviado")
        self.get_logger().info(f"Goal {'accepted' if goal_handle.accepted else 'rejected'}")
        if not goal_handle.accepted:
            return False

        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future, timeout_sec=result_timeout_sec)
        if not result_future.done():
            self.get_logger().error(f"Timeout esperando resultado de {action_name}")
            return False
        wrapped_result = result_future.result()
        if wrapped_result is None:
            self.get_logger().error(f"Resultado vacio de {action_name}")
            return False
        result = wrapped_result.result
        status = wrapped_result.status
        self.get_logger().info(
            f"Resultado {action_name}: status={status}, error_code={result.error_code}, error_string='{result.error_string}'"
        )
        return int(result.error_code) == 0

    def _verify_home(self) -> bool:
        tol = float(self.get_parameter("home_tolerance_rad").value)
        timeout = float(self.get_parameter("joint_state_verify_timeout_sec").value)
        if not self._wait_for_joint_state(timeout):
            self.get_logger().error("No se recibio /joint_states para verificar home.")
            return False

        js = self._last_joint_state
        assert js is not None
        name_to_pos: Dict[str, float] = {
            n: float(p) for n, p in zip(js.name, js.position)
        }
        joint_names = [f"panda_joint{i}" for i in range(1, 8)]
        target = self._home_positions()
        errors: List[float] = []
        for idx, jn in enumerate(joint_names):
            if jn not in name_to_pos:
                self.get_logger().error(f"Joint '{jn}' no encontrado en /joint_states")
                return False
            current = name_to_pos[jn]
            err = abs(current - target[idx])
            errors.append(err)
            self.get_logger().info(
                f"{jn}: target={target[idx]:.6f}, actual={current:.6f}, error={err:.6f}"
            )
        max_error = max(errors) if errors else 999.0
        self.get_logger().info(f"max_error={max_error:.6f}, tolerance={tol:.6f}")
        if max_error <= tol:
            self.get_logger().info("Home alcanzado OK")
            return True
        self.get_logger().error("Home no alcanzado (fuera de tolerancia).")
        return False

    def run(self) -> bool:
        duration = float(self.get_parameter("home_motion_duration_sec").value)
        wait_timeout = float(self.get_parameter("action_wait_timeout_sec").value)
        result_timeout = float(self.get_parameter("result_timeout_sec").value)
        use_action = bool(self.get_parameter("use_action").value)

        arm_names = [f"panda_joint{i}" for i in range(1, 8)]
        arm_traj = self._trajectory(arm_names, self._home_positions(), duration)
        self.get_logger().info(f"Target home articular: {arm_traj.points[0].positions}")

        max_attempts = max(1, int(self.get_parameter("startup_max_attempts").value))
        retry_delay = max(0.5, float(self.get_parameter("retry_delay_sec").value))
        ok_arm = False
        for attempt in range(1, max_attempts + 1):
            if use_action:
                ok_arm = self._send_action_goal(
                    self._arm_client,
                    "/arm_controller/follow_joint_trajectory",
                    arm_traj,
                    wait_timeout_sec=wait_timeout,
                    result_timeout_sec=result_timeout,
                )
            else:
                self.get_logger().warn("Publisher fallback no verifica ejecución.")
                self._arm_pub.publish(arm_traj)
                ok_arm = True
            if ok_arm:
                break
            if attempt < max_attempts:
                self.get_logger().warning(
                    "move_to_home: intento %d/%d fallido; reintento en %.1fs"
                    % (attempt, max_attempts, retry_delay)
                )
                deadline = self.get_clock().now() + Duration(seconds=retry_delay)
                while rclpy.ok() and self.get_clock().now() < deadline:
                    rclpy.spin_once(self, timeout_sec=0.1)
        if not ok_arm:
            return False

        open_gripper = bool(self.get_parameter("open_gripper_at_home").value)
        close_gripper = bool(self.get_parameter("close_gripper_at_home").value)
        gripper_width = None
        if open_gripper and close_gripper:
            self.get_logger().warn(
                "open_gripper_at_home y close_gripper_at_home activos a la vez; "
                "no se envia comando al gripper (ambiguo)."
            )
            open_gripper = False
            close_gripper = False
        if close_gripper:
            gripper_width = float(self.get_parameter("gripper_close_width_m").value)
        elif open_gripper:
            gripper_width = float(self.get_parameter("gripper_open_width_m").value)

        if gripper_width is not None:
            if close_gripper:
                self.get_logger().info("move_to_home: cerrando gripper (close_gripper_at_home=true)")
            g_positions = [gripper_width, gripper_width]
            g_traj = self._trajectory(
                ["panda_finger_joint1", "panda_finger_joint2"],
                g_positions,
                max(1.0, duration * 0.5),
            )
            if use_action:
                ok_gripper = self._send_action_goal(
                    self._gripper_client,
                    "/gripper_controller/follow_joint_trajectory",
                    g_traj,
                    wait_timeout_sec=wait_timeout,
                    result_timeout_sec=result_timeout,
                )
                if not ok_gripper:
                    self.get_logger().error("Fallo accion de gripper en home.")
                    return False
            else:
                self.get_logger().warn("Publisher fallback no verifica ejecución (gripper).")
                self._gripper_pub.publish(g_traj)
            if close_gripper:
                self.get_logger().info("Gripper cerrado en home")
            else:
                self.get_logger().info("Gripper abierto en home")
        else:
            self.get_logger().info("Gripper no modificado en home")

        return self._verify_home()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = MoveToHomeNode()
    exit_code = 0
    try:
        ok = node.run()
        if not ok:
            exit_code = 1
    finally:
        node.destroy_node()
        rclpy.shutdown()
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
