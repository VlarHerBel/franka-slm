#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.callback_groups import ReentrantCallbackGroup
from std_msgs.msg import String

from pymoveit2 import MoveIt2, GripperInterface
from pymoveit2.robots import panda

import math
import json
import queue
from threading import Thread

class TaskExecutor(Node):
    def __init__(self):
        super().__init__('task_executor', parameter_overrides=[Parameter('use_sim_time', Parameter.Type.BOOL, True)])

        # BANDEJA DE ENTRADA (Queue) para comunicar el callback con el hilo principal de forma segura
        self.task_queue = queue.Queue()
        
        self.is_executing = False
        self.world_objects = {}

        self.callback_group = ReentrantCallbackGroup()

        # Arm MoveIt2 interface
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

        # Gripper interface
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
        self.home_joints  = [0.0, 0.0, 0.0, math.radians(-90.0), 0.0, math.radians(92.0), math.radians(50.0)]
        self.drop_joints  = [math.radians(-155.0), math.radians(30.0), math.radians(-20.0),
                             math.radians(-124.0), math.radians(44.0), math.radians(163.0), math.radians(7.0)]

        self.get_logger().info("Moviendo a start_joints inicial...")
        self.moveit2.move_to_configuration(self.start_joints)
        self.moveit2.wait_until_executed()

        self.sub_vision = self.create_subscription(String, '/detected_objects', self.vision_callback, 10, callback_group=self.callback_group)
        self.sub_plan = self.create_subscription(String, '/robot_plan', self.plan_callback, 10, callback_group=self.callback_group)

        self.get_logger().info("🦾 Executor Sincronizado LISTO. Esperando plan del SLM...")

    def vision_callback(self, msg):
        try:
            data = json.loads(msg.data)
            self.world_objects.update(data)
        except:
            pass

    def plan_callback(self, msg):
        self.get_logger().info(f"📥 MENSAJE RECIBIDO DEL SLM: {msg.data}")
        
        # Si el robot está moviéndose o ya hay un plan esperando, lo ignoramos
        # if self.is_executing or not self.task_queue.empty():
        #     self.get_logger().warn("⏳ Ya estoy ejecutando una tarea. Espera a que termine...")
        #     return  

        try:
            plan = json.loads(msg.data)
            if "error" in plan:
                self.get_logger().error(f"❌ El SLM devolvió un error: {plan['error']}")
                return

            steps = plan.get('steps', [])
            if not steps: return

            target_name = steps[0].get('target')
            
            if target_name in self.world_objects:
                coords = self.world_objects[target_name]
                
                self.get_logger().info(f"¡Target '{target_name}' recibido! Añadiendo a la cola de trabajo...")
                
                # LA MAGIA: En lugar de bloquear el hilo, metemos las coordenadas en la bandeja
                self.task_queue.put(coords)
                
            else:
                self.get_logger().error(f"El SLM pidió coger '{target_name}', pero la cámara no lo ve.")

        except Exception as e:
            self.get_logger().error(f"💥 Error procesando el plan: {e}")

    def run_pick_and_place_sequence(self, coords):
        pick_position = [coords[0], coords[1], coords[2] - 0.60]
        quat_xyzw = [0.0, 1.0, 0.0, 0.0]

        self.get_logger().info("1. Moviendo a home_joints...")
        self.moveit2.move_to_configuration(self.home_joints)
        self.moveit2.wait_until_executed()

        self.get_logger().info("2. Moviendo encima del target...")
        self.moveit2.move_to_pose(position=pick_position, quat_xyzw=quat_xyzw)
        self.moveit2.wait_until_executed()

        self.get_logger().info("3. Abriendo gripper...")
        self.gripper.open()
        self.gripper.wait_until_executed()

        self.get_logger().info("4. Bajando hacia el objeto en LÍNEA RECTA...")
        approach_position = [pick_position[0], pick_position[1], pick_position[2] - 0.31]
        self.moveit2.move_to_pose(position=approach_position, quat_xyzw=quat_xyzw, cartesian=True)
        self.moveit2.wait_until_executed()

        self.get_logger().info("5. Cerrando gripper...")
        self.gripper.close()
        self.gripper.wait_until_executed()

        self.get_logger().info("6. Subiendo con el objeto en LÍNEA RECTA...")
        self.moveit2.move_to_pose(position=pick_position, quat_xyzw=quat_xyzw, cartesian=True)
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

        self.get_logger().info("11. Volviendo a la posición HOME para preparar el siguiente Pick...")
        self.moveit2.move_to_configuration(self.home_joints)
        self.moveit2.wait_until_executed()

        self.get_logger().info("✅ ¡Secuencia de Pick-and-place completada con éxito!")


def main(args=None):
    rclpy.init(args=args)
    node = TaskExecutor()

    # Lanzamos el ROS executor en un hilo secundario SOLO para que escuche los topics
    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(node)
    spin_thread = Thread(target=executor.spin, daemon=True)
    spin_thread.start()

    # EL HILO PRINCIPAL: Bucle infinito vigilando la bandeja de entrada
    try:
        while rclpy.ok():
            try:
                # Comprueba si hay algo en la bandeja (espera 0.5 seg y si no hay nada, repite el bucle)
                coords = node.task_queue.get(timeout=0.5)
                
                # Si llega aquí, es que hay un trabajo. ¡A mover el brazo de forma segura!
                try:
                    node.is_executing = True
                    node.run_pick_and_place_sequence(coords)
                except Exception as e:
                    node.get_logger().error(f"⚠️ Error durante la ejecución de la tarea: {e}")
                finally:
                    # Aseguramos que SIEMPRE se libere el flag, haya error o no
                    node.is_executing = False
                    node.get_logger().info("🟢 Executor LIBRE y en posición HOME. Esperando siguiente orden...")
                
            except queue.Empty:
                pass # La bandeja está vacía, no pasa nada, el bucle sigue girando.
                
    except KeyboardInterrupt:
        pass
    finally:
        rclpy.shutdown()
        spin_thread.join()


if __name__ == "__main__":
    main()
