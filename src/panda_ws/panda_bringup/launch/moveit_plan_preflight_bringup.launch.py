"""MoveIt + TF + joint_states estáticos para certificación plan-only (sin Gazebo)."""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import Command, LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    is_sim = LaunchConfiguration("is_sim")
    waypoint = LaunchConfiguration("waypoint")
    waypoints_yaml = LaunchConfiguration("waypoints_yaml")

    robot_description = ParameterValue(
        Command(
            [
                "xacro ",
                os.path.join(
                    get_package_share_directory("panda_description"),
                    "urdf",
                    "panda.urdf.xacro",
                ),
                " is_ignition:=False",
            ]
        ),
        value_type=str,
    )

    moveit = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory("panda_moveit"),
                "launch",
                "moveit.launch.py",
            )
        ),
        launch_arguments={
            "is_sim": is_sim,
            "is_ignition": "false",
            "with_rviz": "false",
        }.items(),
    )

    static_js = Node(
        package="panda_controller",
        executable="static_waypoint_joint_state_publisher",
        output="screen",
        parameters=[
            {"use_sim_time": is_sim},
            {"waypoint": waypoint},
            {"waypoints_yaml": waypoints_yaml},
            {"publish_hz": 30.0},
        ],
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument("is_sim", default_value="false"),
            DeclareLaunchArgument("waypoint", default_value="pick_workspace_ready"),
            DeclareLaunchArgument("waypoints_yaml", default_value=""),
            Node(
                package="robot_state_publisher",
                executable="robot_state_publisher",
                parameters=[
                    {"robot_description": robot_description},
                    {"use_sim_time": is_sim},
                ],
            ),
            TimerAction(period=3.0, actions=[moveit]),
            TimerAction(period=4.0, actions=[static_js]),
        ]
    )
