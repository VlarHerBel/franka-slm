import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch.substitutions import Command, LaunchConfiguration
from launch.conditions import UnlessCondition
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():

    is_sim = LaunchConfiguration("is_sim")
    base_x = LaunchConfiguration("base_x")
    base_y = LaunchConfiguration("base_y")
    base_z = LaunchConfiguration("base_z")
    base_roll = LaunchConfiguration("base_roll")
    base_pitch = LaunchConfiguration("base_pitch")
    base_yaw = LaunchConfiguration("base_yaw")
    
    is_sim_arg = DeclareLaunchArgument(
        "is_sim",
        default_value="True"
    )
    base_x_arg = DeclareLaunchArgument("base_x", default_value="0.0")
    base_y_arg = DeclareLaunchArgument("base_y", default_value="0.0")
    base_z_arg = DeclareLaunchArgument("base_z", default_value="0.0")
    base_roll_arg = DeclareLaunchArgument("base_roll", default_value="0.0")
    base_pitch_arg = DeclareLaunchArgument("base_pitch", default_value="0.0")
    base_yaw_arg = DeclareLaunchArgument("base_yaw", default_value="0.0")

    robot_description = ParameterValue(
        Command(
            [
                "xacro ",
                os.path.join(
                    get_package_share_directory("panda_description"),
                    "urdf",
                    "panda.urdf.xacro",
                ),
                " is_sim:=True",
                " is_ignition:=True",
                " base_x:=", base_x,
                " base_y:=", base_y,
                " base_z:=", base_z,
                " base_roll:=", base_roll,
                " base_pitch:=", base_pitch,
                " base_yaw:=", base_yaw,
            ]
        ),
        value_type=str,
    )

    robot_state_publisher_node = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        parameters=[{"robot_description": robot_description,
                     "use_sim_time": is_sim}],
        condition=UnlessCondition(is_sim),
    )

    controller_manager = Node(
        package="controller_manager",
        executable="ros2_control_node",
        parameters=[
            {"robot_description": robot_description,
             "use_sim_time": is_sim},
            os.path.join(
                get_package_share_directory("panda_controller"),
                "config",
                "panda_controllers.yaml",
            ),
        ],
        condition=UnlessCondition(is_sim),
    )

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
            is_sim_arg,
            base_x_arg,
            base_y_arg,
            base_z_arg,
            base_roll_arg,
            base_pitch_arg,
            base_yaw_arg,
            robot_state_publisher_node,
            controller_manager,
            joint_state_broadcaster_spawner,
            arm_controller_spawner,
            gripper_controller_spawner,
        ]
    )
