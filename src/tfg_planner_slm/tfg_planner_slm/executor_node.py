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

# panda_hand -> finger contact zone (joint offset 0.0584 + finger pad ~0.038)
_FINGER_TIP_OFFSET_M = 0.0964
# How far above the object surface to position panda_hand before descending
_CLEARANCE_ABOVE_M = 0.12
# Intermediate waypoint above object before final descent (meters above surface)
_PRE_GRASP_CLEARANCE_M = 0.045
# Minimum penetration depth below object surface for the finger pads (meters)
_MIN_GRIP_PENETRATION_M = 0.015
# Reduced penetration for very thin objects (flat_part)
_MAX_FLAT_PENETRATION_M = 0.008
# For standing cylinders/cubes, use a smaller panda_hand->finger contact offset.
# This helps MoveIt reach the panda_hand pose without changing the intended
# finger contact depth (it cancels out in grasp_z - finger_offset).
_FINGER_TIP_OFFSET_CYL_CUBE_M = 0.082


def _yaw_to_quat(yaw_rad: float, pitch_rad: float = 0.0):
    """Top-down gripper orientation with optional yaw and pitch.

    Base orientation points the gripper straight down. Yaw rotates around the
    approach axis. Pitch tilts the gripper forward for elongated/lying objects.
    """
    base_quat = np.array([0.0, 1.0, 0.0, 0.0])

    def _qmul(q1, q2):
        x1, y1, z1, w1 = q1[0], q1[1], q1[2], q1[3]
        x2, y2, z2, w2 = q2[0], q2[1], q2[2], q2[3]
        return np.array([
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        ])

    if abs(pitch_rad) > 1e-4:
        cp, sp = math.cos(pitch_rad / 2.0), math.sin(pitch_rad / 2.0)
        pitch_quat = np.array([0.0, sp, 0.0, cp])
        base_quat = _qmul(base_quat, pitch_quat)

    if abs(yaw_rad) > 1e-4:
        cy, sy = math.cos(yaw_rad / 2.0), math.sin(yaw_rad / 2.0)
        yaw_quat = np.array([0.0, 0.0, sy, cy])
        base_quat = _qmul(base_quat, yaw_quat)

    return base_quat.tolist()


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

        self.start_joints = [
            math.radians(0.0),
            math.radians(-40.0),
            math.radians(0.0),
            math.radians(-130.0),
            math.radians(0.0),
            math.radians(90.0),
            math.radians(45.0),
        ]
        self.home_joints = list(self.start_joints)
        self.drop_joints = [
            math.radians(-165.0), math.radians(25.0), math.radians(0.0),
            math.radians(-100.0), math.radians(0.0), math.radians(150.0),
            math.radians(45.0),
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
        self.get_logger().info("Esperando joint_states del controlador...")
        import time
        for attempt in range(40):
            rclpy.spin_once(self, timeout_sec=0.05)
            if self.moveit2.joint_state is not None:
                break
            time.sleep(0.05)
        if self.moveit2.joint_state is None:
            self.get_logger().warn(
                "No se recibieron joint_states tras 4s. "
                "Intentando mover a start_joints de todos modos."
            )
        else:
            self.get_logger().info("Joint states recibidos.")
        time.sleep(0.3)
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
            return obj_info, 0.0, "top_center", 0.0, [0.0, 0.0, 0.0], "unknown"

        position = obj_info.get("position", [0.0, 0.0, 0.0])
        height_m = float(obj_info.get("height_m", 0.0))
        grasp_type = obj_info.get("grasp_type", "top_center")
        grasp_yaw = float(obj_info.get("grasp_yaw_deg", 0.0))
        dims = obj_info.get("dimensions_m", [0.0, 0.0, 0.0])
        shape = obj_info.get("shape") or "unknown"
        return position, height_m, grasp_type, grasp_yaw, dims, shape

    def _move_and_check(self, position, quat_xyzw, cartesian=False, label=""):
        """Execute move_to_pose and return True only if planning+execution succeeded."""
        self.moveit2.move_to_pose(
            position=position,
            quat_xyzw=quat_xyzw,
            cartesian=cartesian,
            cartesian_fraction_threshold=0.90 if cartesian else 0.0,
        )
        ok = self.moveit2.wait_until_executed()
        if not ok:
            self.get_logger().error(
                f"Movimiento FALLIDO ({label}): pos={position}, cartesian={cartesian}"
            )
        return ok

    def _config_and_check(self, joints, label=""):
        """Execute move_to_configuration and return True only if it succeeded."""
        self.moveit2.move_to_configuration(joints)
        ok = self.moveit2.wait_until_executed()
        if not ok:
            self.get_logger().error(f"Movimiento a configuracion FALLIDO ({label})")
        return ok

    def _compute_grasp_params(self, obj_x, obj_y, obj_z, height_m, shape, grasp_type):
        """Compute above_z, pre_grasp_z, grasp_z and pitch from object properties."""
        finger_offset = _FINGER_TIP_OFFSET_M
        if shape in ("cylinder", "cube"):
            finger_offset = _FINGER_TIP_OFFSET_CYL_CUBE_M

        above_z = obj_z + _CLEARANCE_ABOVE_M + finger_offset
        pre_grasp_z = obj_z + _PRE_GRASP_CLEARANCE_M + finger_offset

        if shape == "flat_part":
            penetration = min(height_m * 0.6, _MAX_FLAT_PENETRATION_M)
        else:
            penetration = max(height_m * 0.5, _MIN_GRIP_PENETRATION_M)
        grasp_z = obj_z - penetration + finger_offset

        pitch_rad = 0.0
        if height_m < 0.06 and shape in ("elongated", "cylinder"):
            pitch_rad = math.radians(8.0)

        return above_z, pre_grasp_z, grasp_z, pitch_rad, finger_offset

    def run_pick_and_place_sequence(self, obj_info, target_name=None):
        position, height_m, grasp_type, grasp_yaw_deg, dims, shape = (
            self._extract_grasp_params(obj_info)
        )

        obj_x, obj_y, obj_z = position
        grasp_yaw_rad = math.radians(grasp_yaw_deg)
        above_z, pre_grasp_z, grasp_z, pitch_rad, finger_offset = (
            self._compute_grasp_params(
                obj_x, obj_y, obj_z, height_m, shape, grasp_type
            )
        )
        quat_xyzw = _yaw_to_quat(grasp_yaw_rad, pitch_rad)

        self.get_logger().info(
            f"Grasp plan: pos=({obj_x:.3f},{obj_y:.3f},{obj_z:.3f}), "
            f"shape={shape}, h={height_m:.3f}m, yaw={grasp_yaw_deg:.1f}deg, "
            f"pitch={math.degrees(pitch_rad):.0f}deg, type={grasp_type}, "
            f"above_z={above_z:.3f}, pre_z={pre_grasp_z:.3f}, grasp_z={grasp_z:.3f}"
        )

        self.publish_status(
            "executing",
            f"Ejecutando pick and place para '{target_name or 'objeto'}'.",
            target=target_name,
        )

        pick_above = [obj_x, obj_y, above_z]
        pick_pre = [obj_x, obj_y, pre_grasp_z]
        pick_grasp = [obj_x, obj_y, grasp_z]

        self.get_logger().info("1. Moviendo a home_joints...")
        self._config_and_check(self.home_joints, "home")

        self.get_logger().info("2. Abriendo gripper...")
        self.gripper.open()
        self.gripper.wait_until_executed()

        self.get_logger().info("3. Moviendo encima del target (con reintentos)...")
        # Try lower "above" first to keep poses inside MoveIt workspace.
        above_clearances = [0.06, 0.08, 0.10]
        # If planning fails, try slight yaw offsets to find a collision-free IK solution.
        yaw_offsets = [0.0]
        if shape not in ("cube",):
            yaw_offsets = [0.0, math.pi / 2.0, -math.pi / 2.0]

        pick_above_ok = False
        last_pick_above_try = pick_above
        last_quat_try = quat_xyzw
        moved_to_pre_directly = False
        last_pick_pre_try = pick_pre
        for yaw_off in yaw_offsets:
            quat_try = _yaw_to_quat(grasp_yaw_rad + yaw_off, pitch_rad)
            for clearance in above_clearances:
                above_z_try = obj_z + clearance + finger_offset
                pick_above_try = [obj_x, obj_y, above_z_try]
                last_pick_above_try = pick_above_try
                last_quat_try = quat_try
                if self._move_and_check(
                    pick_above_try, quat_try, label="pick_above"
                ):
                    above_z = above_z_try
                    pick_above = pick_above_try
                    quat_xyzw = quat_try
                    pick_above_ok = True
                    break
            if pick_above_ok:
                break

        if not pick_above_ok:
            # Fallback: go directly to pre-grasp (lower z) and then descend.
            self.get_logger().warn(
                "pick_above no alcanzable; intentando ir a pick_pre directamente."
            )
            pre_clearances = [0.025, 0.035, _PRE_GRASP_CLEARANCE_M]
            pre_ok = False
            for yaw_off in yaw_offsets:
                quat_try = _yaw_to_quat(grasp_yaw_rad + yaw_off, pitch_rad)
                for pre_clearance in pre_clearances:
                    pre_z_try = obj_z + pre_clearance + finger_offset
                    pick_pre_try = [obj_x, obj_y, pre_z_try]
                    last_pick_pre_try = pick_pre_try
                    if self._move_and_check(
                        pick_pre_try, quat_try, cartesian=False, label="pick_pre"
                    ):
                        quat_xyzw = quat_try
                        pick_pre = pick_pre_try
                        moved_to_pre_directly = True
                        pre_ok = True
                        break
                if pre_ok:
                    break

            if not pre_ok:
                self.get_logger().error(
                    "ABORTANDO: no se puede alcanzar pick_above/pick_pre. "
                    f"Last above pos={last_pick_above_try}, last pre pos={last_pick_pre_try}"
                )
                self._config_and_check(self.home_joints, "home_recovery")
                raise RuntimeError(
                    f"Planning failed for pick_above {last_pick_above_try} "
                    f"and pick_pre {last_pick_pre_try}"
                )

        if not moved_to_pre_directly:
            self.get_logger().info(
                "4. Bajando a pre-grasp (waypoint intermedio)..."
            )
            if not self._move_and_check(
                pick_pre, quat_xyzw, cartesian=True, label="pick_pre"
            ):
                self.get_logger().warn("Pre-grasp fallido, intentando bajar directo.")

        self.get_logger().info("5. Bajando hacia el objeto en linea recta...")
        if not self._move_and_check(pick_grasp, quat_xyzw, cartesian=True, label="pick_grasp"):
            self.get_logger().error("ABORTANDO: no se puede bajar al objeto.")
            self._config_and_check(self.home_joints, "home_recovery")
            raise RuntimeError(f"Cartesian path failed for pick_grasp {pick_grasp}")

        self.get_logger().info("6. Cerrando gripper...")
        self.gripper.close()
        self.gripper.wait_until_executed()

        self.get_logger().info("7. Subiendo con el objeto en linea recta...")
        if not self._move_and_check(pick_above, quat_xyzw, cartesian=True, label="lift"):
            self.get_logger().warn("Subida cartesiana parcial, intentando home directo.")

        self.get_logger().info("8. Moviendo a home_joints...")
        self._config_and_check(self.home_joints, "home")

        self.get_logger().info("9. Moviendo a drop_joints (Zona de soltado)...")
        self._config_and_check(self.drop_joints, "drop")

        self.get_logger().info("10. Abriendo gripper para soltar...")
        self.gripper.open()
        self.gripper.wait_until_executed()

        self.get_logger().info("11. Cerrando gripper...")
        self.gripper.close()
        self.gripper.wait_until_executed()

        self.get_logger().info("12. Volviendo a la posicion HOME...")
        self._config_and_check(self.home_joints, "home_final")

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
