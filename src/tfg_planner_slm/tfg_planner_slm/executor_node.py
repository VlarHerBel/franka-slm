#!/usr/bin/env python3
import json
import math
import queue

import numpy as np

import rclpy
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.node import Node
from rclpy.parameter import Parameter
from std_msgs.msg import String

from pymoveit2 import MoveIt2, GripperInterface
from pymoveit2.robots import panda

# Safety margins relative to the detected top grasp point
_CLEARANCE_ABOVE_M = 0.12
_GRASP_SURFACE_MARGIN_M = 0.008


def _yaw_to_quat(yaw_rad: float):
    """Top-down gripper orientation with an optional yaw rotation.

    Base orientation points the gripper straight down (quat = [0, 1, 0, 0]).
    A yaw rotation around the approach axis (Z of EE) is composed on top.
    """
    base_quat = np.array([0.0, 1.0, 0.0, 0.0])
    if abs(yaw_rad) < 1e-4:
        return base_quat.tolist()

    cy, sy = math.cos(yaw_rad / 2.0), math.sin(yaw_rad / 2.0)
    yaw_quat = np.array([0.0, 0.0, sy, cy])

    def _qmul(q1, q2):
        w1, x1, y1, z1 = q1[3], q1[0], q1[1], q1[2]
        w2, x2, y2, z2 = q2[3], q2[0], q2[1], q2[2]
        return np.array([
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        ])

    return _qmul(base_quat, yaw_quat).tolist()


class TaskExecutor(Node):
    def __init__(self):
        super().__init__(
            "task_executor",
            parameter_overrides=[
                Parameter("use_sim_time", Parameter.Type.BOOL, True)
            ],
        )

        self.task_queue = queue.Queue()
        self.is_executing = False
        self.world_objects = {}

        self.callback_group = ReentrantCallbackGroup()

        self.moveit2 = MoveIt2(
            node=self,
            joint_names=panda.joint_names(),
            base_link_name=panda.base_link_name(),
            end_effector_name=panda.end_effector_name(),
            group_name=panda.MOVE_GROUP_ARM,
            callback_group=self.callback_group,
        )

        self.moveit2.max_velocity = 0.1
        self.moveit2.max_acceleration = 0.1

        self.gripper = GripperInterface(
            node=self,
            gripper_joint_names=panda.gripper_joint_names(),
            open_gripper_joint_positions=panda.OPEN_GRIPPER_JOINT_POSITIONS,
            closed_gripper_joint_positions=panda.CLOSED_GRIPPER_JOINT_POSITIONS,
            gripper_group_name=panda.MOVE_GROUP_GRIPPER,
            callback_group=self.callback_group,
            gripper_command_action_name="gripper_action_controller/gripper_cmd",
        )

        self.start_joints = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, math.radians(-125.0)]
        # Pose despejada para dejar libre la mayor parte de la mesa bajo la cámara.
        self.home_joints = list(self.start_joints)
        self.drop_joints = [
            math.radians(-155.0), math.radians(30.0), math.radians(-20.0),
            math.radians(-124.0), math.radians(44.0), math.radians(163.0),
            math.radians(7.0),
        ]

        self.sub_vision = self.create_subscription(
            String, "/detected_objects", self.vision_callback,
            10, callback_group=self.callback_group,
        )
        self.sub_plan = self.create_subscription(
            String, "/robot_plan", self.plan_callback,
            10, callback_group=self.callback_group,
        )
        self.status_publisher = self.create_publisher(String, "/executor_status", 10)

        self.get_logger().info("Executor sincronizado listo. Esperando plan del SLM...")
        self.publish_status("idle", "Esperando plan del SLM.")

    def go_to_start(self):
        self.get_logger().info("Moviendo a start_joints inicial...")
        self.moveit2.move_to_configuration(self.start_joints)
        self.moveit2.wait_until_executed()

    def vision_callback(self, msg):
        try:
            data = json.loads(msg.data)
            if isinstance(data, dict):
                self.world_objects.update(data)
        except json.JSONDecodeError as exc:
            self.get_logger().warn(f"Error actualizando objetos visibles: {exc}")

    def plan_callback(self, msg):
        self.get_logger().info(f"Mensaje recibido del SLM: {msg.data}")

        if self.is_executing:
            self.get_logger().warn(
                "Ya estoy ejecutando una tarea. La nueva orden se encolara."
            )

        try:
            plan = json.loads(msg.data)
            if "error" in plan:
                error_message = f"El SLM devolvio un error: {plan['error']}"
                self.get_logger().error(error_message)
                self.publish_status("error", error_message)
                return

            steps = plan.get("steps", [])
            if not steps:
                self.publish_status("error", "El plan recibido no contiene pasos.")
                return

            target_name = steps[0].get("target")
            if target_name in self.world_objects:
                obj_info = self.world_objects[target_name]

                self.get_logger().info(
                    f"Target '{target_name}' recibido. Anadiendo a la cola de trabajo."
                )
                self.task_queue.put({"target": target_name, "obj_info": obj_info})
                self.publish_status(
                    "queued",
                    f"Tarea encolada para '{target_name}'.",
                    target=target_name,
                    queue_size=self.task_queue.qsize(),
                )
            else:
                error_message = (
                    f"El SLM pidio coger '{target_name}', pero la camara no lo ve."
                )
                self.get_logger().error(error_message)
                self.publish_status("error", error_message, target=target_name)

        except Exception as e:
            error_message = f"Error procesando el plan: {e}"
            self.get_logger().error(error_message)
            self.publish_status("error", error_message)

    # ------------------------------------------------------------------
    # Adaptive pick-and-place
    # ------------------------------------------------------------------

    def _extract_grasp_params(self, obj_info):
        """Parse rich object info from the vision bridge into grasp parameters."""
        if isinstance(obj_info, list):
            return obj_info, 0.0, "top_center", 0.0, [0.0, 0.0, 0.0]

        position = obj_info.get("position", [0.0, 0.0, 0.0])
        height_m = float(obj_info.get("height_m", 0.0))
        grasp_type = obj_info.get("grasp_type", "top_center")
        grasp_yaw = float(obj_info.get("grasp_yaw_deg", 0.0))
        dims = obj_info.get("dimensions_m", [0.0, 0.0, 0.0])
        return position, height_m, grasp_type, grasp_yaw, dims

    def run_pick_and_place_sequence(self, obj_info, target_name=None):
        position, height_m, grasp_type, grasp_yaw_deg, dims = (
            self._extract_grasp_params(obj_info)
        )

        obj_x, obj_y, obj_z = position
        grasp_yaw_rad = math.radians(grasp_yaw_deg)
        quat_xyzw = _yaw_to_quat(grasp_yaw_rad)

        above_z = obj_z + _CLEARANCE_ABOVE_M
        grasp_z = obj_z + _GRASP_SURFACE_MARGIN_M

        self.get_logger().info(
            f"Grasp plan: shape_h={height_m:.3f}m, yaw={grasp_yaw_deg:.1f}deg, "
            f"type={grasp_type}, dims={dims}, above_z={above_z:.3f}, grasp_z={grasp_z:.3f}"
        )

        self.publish_status(
            "executing",
            f"Ejecutando pick and place para '{target_name or 'objeto'}'.",
            target=target_name,
        )

        pick_above = [obj_x, obj_y, above_z]
        pick_grasp = [obj_x, obj_y, grasp_z]

        self.get_logger().info("1. Moviendo a home_joints...")
        self.moveit2.move_to_configuration(self.home_joints)
        self.moveit2.wait_until_executed()

        self.get_logger().info("2. Moviendo encima del target...")
        self.moveit2.move_to_pose(position=pick_above, quat_xyzw=quat_xyzw)
        self.moveit2.wait_until_executed()

        self.get_logger().info("3. Abriendo gripper...")
        self.gripper.open()
        self.gripper.wait_until_executed()

        self.get_logger().info("4. Bajando hacia el objeto en linea recta...")
        self.moveit2.move_to_pose(
            position=pick_grasp, quat_xyzw=quat_xyzw, cartesian=True,
        )
        self.moveit2.wait_until_executed()

        self.get_logger().info("5. Cerrando gripper...")
        self.gripper.close()
        self.gripper.wait_until_executed()

        self.get_logger().info("6. Subiendo con el objeto en linea recta...")
        self.moveit2.move_to_pose(
            position=pick_above, quat_xyzw=quat_xyzw, cartesian=True,
        )
        self.moveit2.wait_until_executed()

        self.get_logger().info("7. Moviendo a home_joints...")
        self.moveit2.move_to_configuration(self.home_joints)
        self.moveit2.wait_until_executed()

        self.get_logger().info("8. Moviendo a drop_joints (Zona de soltado)...")
        self.moveit2.move_to_configuration(self.drop_joints)
        self.moveit2.wait_until_executed()

        self.get_logger().info("9. Abriendo gripper para soltar...")
        self.gripper.open()
        self.gripper.wait_until_executed()

        self.get_logger().info("10. Cerrando gripper...")
        self.gripper.close()
        self.gripper.wait_until_executed()

        self.get_logger().info("11. Volviendo a la posicion HOME...")
        self.moveit2.move_to_configuration(self.home_joints)
        self.moveit2.wait_until_executed()

        self.get_logger().info("Secuencia de Pick-and-place completada con exito!")

    def publish_status(self, status, message, target=None, queue_size=None):
        payload = {
            "status": status,
            "message": message,
            "is_executing": self.is_executing,
        }
        if target is not None:
            payload["target"] = target
        if queue_size is not None:
            payload["queue_size"] = queue_size

        msg = String()
        msg.data = json.dumps(payload)
        self.status_publisher.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = TaskExecutor()

    node.go_to_start()

    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.1)

            try:
                task = node.task_queue.get_nowait()
                obj_info = task.get("obj_info", task.get("coords"))
                target_name = task.get("target")
                task_succeeded = False

                try:
                    node.is_executing = True
                    node.publish_status(
                        "executing",
                        f"Iniciando ejecucion para '{target_name}'.",
                        target=target_name,
                        queue_size=node.task_queue.qsize(),
                    )
                    node.run_pick_and_place_sequence(
                        obj_info, target_name=target_name,
                    )
                    task_succeeded = True
                except Exception as e:
                    error_message = f"Error durante la ejecucion de la tarea: {e}"
                    node.get_logger().error(error_message)
                    node.publish_status("error", error_message, target=target_name)
                finally:
                    node.is_executing = False
                    node.get_logger().info(
                        "Executor libre y en posicion HOME. Esperando siguiente orden..."
                    )
                    if task_succeeded:
                        node.publish_status(
                            "completed",
                            f"Pick and place completado para '{target_name}'.",
                            target=target_name,
                            queue_size=node.task_queue.qsize(),
                        )
                    node.publish_status(
                        "idle",
                        "Executor libre y esperando siguiente orden.",
                        queue_size=node.task_queue.qsize(),
                    )

            except queue.Empty:
                pass

    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
