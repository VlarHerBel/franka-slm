#!/usr/bin/env python3
import json
import math
import queue
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import rclpy
import yaml
from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import Pose
from moveit_msgs.msg import CollisionObject, PlanningScene
from pymoveit2 import GripperInterface, MoveIt2
from pymoveit2.robots import panda
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.node import Node
from rclpy.parameter import Parameter
from shape_msgs.msg import SolidPrimitive
from std_msgs.msg import String


def _top_grasp_quat(yaw_rad: float) -> list:
    base = [0.0, 1.0, 0.0, 0.0]
    yaw_q = [0.0, 0.0, math.sin(yaw_rad / 2.0), math.cos(yaw_rad / 2.0)]
    x1, y1, z1, w1 = base
    x2, y2, z2, w2 = yaw_q
    return [
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
    ]


class TaskExecutor(Node):
    def __init__(self):
        super().__init__(
            "task_executor",
            parameter_overrides=[Parameter("use_sim_time", Parameter.Type.BOOL, True)],
        )
        default_params = (
            Path(get_package_share_directory("tfg_planner_slm"))
            / "config"
            / "pick_place_params.yaml"
        )
        self.declare_parameter("pick_params_path", str(default_params))
        self.cfg = self._load_yaml(Path(str(self.get_parameter("pick_params_path").value)))
        self.general_cfg = self.cfg.get("general", {}) if isinstance(self.cfg, dict) else {}
        self.grasp_cfg = self.cfg.get("grasp", {}) if isinstance(self.cfg, dict) else {}
        self.place_cfg = self.cfg.get("place", {}) if isinstance(self.cfg, dict) else {}
        self.scene_cfg = self.cfg.get("planning_scene", {}) if isinstance(self.cfg, dict) else {}

        self.task_queue = queue.Queue()
        self.is_executing = False
        self.world_objects: Dict[str, Dict[str, Any]] = {}
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

        self.start_joints = [math.radians(v) for v in [0.0, -40.0, 0.0, -130.0, 0.0, 90.0, 45.0]]
        self.home_joints = list(self.start_joints)
        self.drop_joints = self._load_drop_joints()
        self.deposit_approach = list(self.place_cfg.get("deposit_approach_position_m", [-0.45, 0.0, 0.50]))
        self.deposit_place = list(self.place_cfg.get("deposit_place_position_m", [-0.45, 0.0, 0.34]))
        self.deposit_yaw_rad = math.radians(float(self.place_cfg.get("deposit_yaw_deg", 180.0)))

        detected_topic = str(self.general_cfg.get("detected_objects_topic", "/detected_objects"))
        plan_topic = str(self.general_cfg.get("robot_plan_topic", "/robot_plan"))
        status_topic = str(self.general_cfg.get("executor_status_topic", "/executor_status"))

        self.create_subscription(
            String, detected_topic, self.vision_callback, 10, callback_group=self.callback_group
        )
        self.create_subscription(
            String, plan_topic, self.plan_callback, 10, callback_group=self.callback_group
        )
        self.status_publisher = self.create_publisher(String, status_topic, 10)
        self.scene_pub = self.create_publisher(PlanningScene, "/planning_scene", 10)
        self.scene_timer = self.create_timer(1.0, self._publish_static_scene_once)
        self._scene_published = False

        self.get_logger().info("Executor listo para pick and place final.")
        self.publish_status("idle", "Esperando plan del SLM.")

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

    def _load_drop_joints(self) -> list:
        values = self.place_cfg.get("drop_joints_deg")
        if isinstance(values, list) and len(values) == 7:
            return [math.radians(float(v)) for v in values]
        return [math.radians(v) for v in [-165.0, 25.0, 0.0, -100.0, 0.0, 150.0, 45.0]]

    def _publish_static_scene_once(self) -> None:
        if self._scene_published:
            return
        table_cfg = self.scene_cfg.get("table", {})
        box_cfg = self.scene_cfg.get("deposit_box", {})
        table_obj = self._make_box_collision(
            object_id=str(table_cfg.get("id", "work_table")),
            size=table_cfg.get("size_m", [0.72, 0.48, 0.04]),
            position=table_cfg.get("position_m", [0.60, 0.00, 0.24]),
        )
        box_obj = self._make_box_collision(
            object_id=str(box_cfg.get("id", "deposit_box_collision")),
            size=box_cfg.get("size_m", [0.24, 0.20, 0.18]),
            position=box_cfg.get("position_m", [-0.45, 0.00, 0.09]),
        )
        scene = PlanningScene()
        scene.is_diff = True
        scene.world.collision_objects = [table_obj, box_obj]
        self.scene_pub.publish(scene)
        self._scene_published = True
        self.get_logger().info("Planning scene publicada con mesa + deposit_box.")

    def _make_box_collision(self, object_id: str, size: Any, position: Any) -> CollisionObject:
        primitive = SolidPrimitive()
        primitive.type = SolidPrimitive.BOX
        primitive.dimensions = [float(size[0]), float(size[1]), float(size[2])]

        pose = Pose()
        pose.position.x = float(position[0])
        pose.position.y = float(position[1])
        pose.position.z = float(position[2])
        pose.orientation.w = 1.0

        obj = CollisionObject()
        obj.header.frame_id = str(self.general_cfg.get("world_frame", "panda_link0"))
        obj.id = object_id
        obj.operation = CollisionObject.ADD
        obj.primitives = [primitive]
        obj.primitive_poses = [pose]
        return obj

    def go_to_start(self) -> None:
        import time

        self.get_logger().info("Esperando joint_states...")
        for _ in range(40):
            rclpy.spin_once(self, timeout_sec=0.05)
            if self.moveit2.joint_state is not None:
                break
            time.sleep(0.05)
        self.moveit2.move_to_configuration(self.start_joints)
        self.moveit2.wait_until_executed()

    def vision_callback(self, msg: String) -> None:
        try:
            data = json.loads(msg.data)
            if isinstance(data, dict):
                self.world_objects = data
        except json.JSONDecodeError as exc:
            self.get_logger().warn(f"Error actualizando objetos visibles: {exc}")

    def plan_callback(self, msg: String) -> None:
        try:
            plan = json.loads(msg.data)
            steps = plan.get("steps", [])
            if not steps:
                self.publish_status("error", "Plan vacío.")
                return
            target_class = str(steps[0].get("target", "")).strip()
            if not target_class:
                self.publish_status("error", "Plan sin target.")
                return
            selected = self._select_best_instance(target_class)
            if selected is None:
                self.publish_status(
                    "error",
                    f"No hay instancia visible para '{target_class}'.",
                    target=target_class,
                )
                return
            obj_id, obj_info = selected
            self.task_queue.put({"target": target_class, "object_id": obj_id, "obj_info": obj_info})
            self.publish_status(
                "queued",
                f"Tarea encolada para '{target_class}' ({obj_id}).",
                target=target_class,
                queue_size=self.task_queue.qsize(),
            )
        except Exception as exc:
            self.publish_status("error", f"Error procesando plan: {exc}")

    def _select_best_instance(self, target_class: str) -> Optional[Tuple[str, Dict[str, Any]]]:
        candidates = []
        for object_id, obj in self.world_objects.items():
            if not isinstance(obj, dict):
                continue
            label = str(obj.get("label") or object_id).strip()
            if label != target_class and object_id != target_class:
                continue
            pos = obj.get("position")
            if not isinstance(pos, list) or len(pos) != 3:
                continue
            score = float(obj.get("score", 0.0))
            candidates.append((object_id, obj, score))
        if not candidates:
            return None
        candidates.sort(key=lambda entry: (entry[2], float(entry[1]["position"][2])), reverse=True)
        return candidates[0][0], candidates[0][1]

    def _class_offsets(self, target_class: str) -> Tuple[list, list]:
        class_cfg = self.grasp_cfg.get("class_offsets", {}).get(target_class, {})
        approach = class_cfg.get(
            "approach_offset_m", self.grasp_cfg.get("approach_offset_default_m", [0.0, 0.0, 0.10])
        )
        grasp = class_cfg.get(
            "grasp_offset_m", self.grasp_cfg.get("grasp_offset_default_m", [0.0, 0.0, -0.02])
        )
        return [float(v) for v in approach], [float(v) for v in grasp]

    def _compute_pick_poses(self, target_class: str, obj: Dict[str, Any]) -> Tuple[list, list, float]:
        position = [float(v) for v in obj.get("position", [0.0, 0.0, 0.0])]
        approach = obj.get("approach_position")
        grasp = obj.get("grasp_position")
        approach_offset, grasp_offset = self._class_offsets(target_class)
        if not isinstance(approach, list) or len(approach) != 3:
            approach = [position[i] + approach_offset[i] for i in range(3)]
        else:
            approach = [float(v) for v in approach]
        if not isinstance(grasp, list) or len(grasp) != 3:
            grasp = [position[i] + grasp_offset[i] for i in range(3)]
        else:
            grasp = [float(v) for v in grasp]
        min_pick_z = float(self.grasp_cfg.get("min_pick_z_m", 0.03))
        grasp[2] = max(min_pick_z, grasp[2])
        approach[2] = max(grasp[2] + 0.03, approach[2])
        yaw_rad = float(obj.get("grasp_yaw_rad", 0.0))
        if abs(yaw_rad) < 1e-5:
            yaw_deg = float(obj.get("grasp_yaw_deg", self.grasp_cfg.get("default_grasp_yaw_deg", 0.0)))
            yaw_rad = math.radians(yaw_deg)
        return approach, grasp, yaw_rad

    def _move_pose(self, position: list, quat_xyzw: list, cartesian: bool = False, label: str = "") -> bool:
        self.moveit2.move_to_pose(
            position=position,
            quat_xyzw=quat_xyzw,
            cartesian=cartesian,
            cartesian_fraction_threshold=0.90 if cartesian else 0.0,
        )
        ok = bool(self.moveit2.wait_until_executed())
        if not ok:
            self.get_logger().error(f"Falló movimiento {label}: {position}")
        return ok

    def _move_joints(self, joints: list, label: str = "") -> bool:
        self.moveit2.move_to_configuration(joints)
        ok = bool(self.moveit2.wait_until_executed())
        if not ok:
            self.get_logger().error(f"Falló movimiento articular {label}.")
        return ok

    def run_pick_and_place_sequence(self, target_class: str, obj: Dict[str, Any]) -> None:
        approach, grasp, yaw = self._compute_pick_poses(target_class, obj)
        quat_pick = _top_grasp_quat(yaw)
        retreat_offset = [
            float(v) for v in self.grasp_cfg.get("retreat_offset_m", [0.0, 0.0, 0.12])
        ]
        retreat = [grasp[i] + retreat_offset[i] for i in range(3)]
        quat_place = _top_grasp_quat(self.deposit_yaw_rad)

        self._move_joints(self.home_joints, "home")
        self.gripper.open()
        self.gripper.wait_until_executed()

        if not self._move_pose(approach, quat_pick, cartesian=False, label="approach"):
            raise RuntimeError("No se alcanzó la pose de approach.")
        if not self._move_pose(grasp, quat_pick, cartesian=True, label="grasp"):
            raise RuntimeError("No se alcanzó la pose de grasp.")
        self.gripper.close()
        self.gripper.wait_until_executed()

        self._move_pose(retreat, quat_pick, cartesian=True, label="retreat")
        self._move_joints(self.drop_joints, "drop_joints")
        self._move_pose(self.deposit_approach, quat_place, cartesian=False, label="deposit_approach")
        self._move_pose(self.deposit_place, quat_place, cartesian=True, label="deposit_place")
        self.gripper.open()
        self.gripper.wait_until_executed()
        self._move_pose(self.deposit_approach, quat_place, cartesian=True, label="deposit_retreat")
        self._move_joints(self.home_joints, "home_final")

    def publish_status(self, status: str, message: str, target: Optional[str] = None, queue_size: Optional[int] = None):
        payload = {"status": status, "message": message, "is_executing": self.is_executing}
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
            except queue.Empty:
                continue
            target = task.get("target")
            obj_info = task.get("obj_info", {})
            try:
                node.is_executing = True
                node.publish_status("executing", f"Ejecutando pick para '{target}'.", target=target)
                node.run_pick_and_place_sequence(str(target), obj_info)
                node.publish_status("completed", f"Pick and place completado para '{target}'.", target=target)
            except Exception as exc:
                node.get_logger().error(f"Error de ejecución: {exc}")
                node.publish_status("error", f"Error en ejecución: {exc}", target=target)
            finally:
                node.is_executing = False
                node.publish_status("idle", "Executor libre.", queue_size=node.task_queue.qsize())
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
