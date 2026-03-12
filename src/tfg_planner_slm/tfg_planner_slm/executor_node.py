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

        self.sub_vision = self.create_subscription(String, '/detected_objects', self.vision_callback, 10, callback_group=self.callback_group)
        self.sub_plan = self.create_subscription(String, '/robot_plan', self.plan_callback, 10, callback_group=self.callback_group)

        self.get_logger().info("🦾 Executor Sincronizado LISTO. Esperando plan del SLM...")

    def go_to_start(self):
        self.get_logger().info("Moviendo a start_joints inicial...")
        self.moveit2.move_to_configuration(self.start_joints)
        self.moveit2.wait_until_executed()

    def vision_callback(self, msg):
        try:
            data = json.loads(msg.data)
            self.world_objects.update(data)
        except:
            pass

    def plan_callback(self, msg):
        self.get_logger().info(f"📥 MENSAJE RECIBIDO DEL SLM: {msg.data}")
        
        # Si el robot está moviéndose o ya hay un plan esperando, lo ignoramos
        if self.is_executing:
            self.get_logger().warn("⏳ Ya estoy ejecutando una tarea. La pondré en cola, pero ten paciencia...")
            # No hacemos return, dejamos que se encole, pero avisamos.

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

    # --- MODELO DE EJECUCIÓN CORRECTO PARA PYMOVEIT2 ---
    # No se usa un hilo de spin en segundo plano. La librería pymoveit2 gestiona su propio
    # "spin" interno cuando se llama a una función bloqueante como `wait_until_executed`.
    # Esto entraba en conflicto con nuestro MultiThreadedExecutor, causando el bloqueo.
    #
    # El flujo correcto es:
    # 1. Dejar que las funciones de MoveIt2 bloqueen y hagan su trabajo.
    # 2. En nuestro bucle principal, llamar a `rclpy.spin_once()` para procesar
    #    manualmente los callbacks (como los mensajes del LLM) cuando el robot está inactivo.

    # 1. Mover el robot a la posición inicial. Esta llamada es bloqueante.
    node.go_to_start()

    # 2. Bucle principal: procesa callbacks y tareas de la cola.
    try:
        while rclpy.ok():
            # Procesa un callback pendiente (ej. de /robot_plan) si hay alguno.
            # Es no-bloqueante y permite que el nodo siga "vivo" para recibir órdenes.
            rclpy.spin_once(node, timeout_sec=0.1)

            try:
                # Intenta coger una tarea de la cola (de forma no-bloqueante).
                coords = node.task_queue.get_nowait()
                
                # Si llega aquí, es que hay trabajo. ¡A mover el brazo!
                try:
                    node.is_executing = True
                    # Esta función es bloqueante y usará el spinner interno de pymoveit2.
                    node.run_pick_and_place_sequence(coords)
                except Exception as e:
                    node.get_logger().error(f"⚠️ Error durante la ejecución de la tarea: {e}")
                finally:
                    node.is_executing = False
                    node.get_logger().info("🟢 Executor LIBRE y en posición HOME. Esperando siguiente orden...")
                
            except queue.Empty:
                # La cola está vacía, es el estado normal mientras espera órdenes. No hacemos nada.
                pass
                
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
