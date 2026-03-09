import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():

    # NOTA: No necesitamos generar robot_description aquí de nuevo si ya 
    # se cargó en Gazebo, pero los spawners a veces lo requieren para saber 
    # los nombres de los joints. Sin embargo, no necesitamos el nodo 
    # robot_state_publisher ni el controller_manager explícito.

    # Solo necesitamos los Spawners. 
    # Ellos hablarán automáticamente con el Controller Manager que vive DENTRO de Gazebo.

    joint_state_broadcaster_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=[
            "joint_state_broadcaster",
            "--controller-manager",
            "/controller_manager",
        ],
    )

    arm_controller_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["arm_controller", "--controller-manager", "/controller_manager"],
    )

    gripper_controller_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["gripper_controller", "--controller-manager", "/controller_manager"],
    )

    return LaunchDescription(
        [
            # robot_state_publisher_node,  <-- ELIMINADO (Ya está en gazebo.launch.py)
            # controller_manager,          <-- ELIMINADO (El plugin de Gazebo ya hace esto)
            
            joint_state_broadcaster_spawner,
            arm_controller_spawner,
            gripper_controller_spawner,
        ]
    )