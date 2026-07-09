from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

import os


def generate_launch_description():
    include_robot_stack = LaunchConfiguration("include_robot_stack")
    web_port = LaunchConfiguration("web_port")

    include_robot_stack_arg = DeclareLaunchArgument(
        "include_robot_stack",
        default_value="false",
        description="Lanza tambien Gazebo, MoveIt y vision de Panda.",
    )
    web_port_arg = DeclareLaunchArgument(
        "web_port",
        default_value="8000",
        description="Puerto HTTP para el chat web local.",
    )

    panda_bringup = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory("panda_bringup"),
                "launch",
                "pick_and_place.launch.py",
            )
        ),
        condition=IfCondition(include_robot_stack),
    )

    vision_bridge_node = Node(
        package="tfg_planner_slm",
        executable="vision_bridge_node",
        name="vision_bridge_node",
        output="screen",
    )

    llm_node = Node(
        package="tfg_planner_slm",
        executable="llm_node",
        name="llm_node",
        output="screen",
    )

    executor_node = Node(
        package="tfg_planner_slm",
        executable="executor_node",
        name="executor_node",
        output="screen",
    )

    web_bridge_node = Node(
        package="tfg_planner_slm",
        executable="web_bridge_node",
        name="web_bridge_node",
        output="screen",
        parameters=[{"port": web_port}],
    )

    return LaunchDescription(
        [
            #include_robot_stack_arg,
            web_port_arg,
            panda_bringup,
            vision_bridge_node,
            llm_node,
            executor_node,
            web_bridge_node,
        ]
    )
