import os
from launch import LaunchDescription
from moveit_configs_utils import MoveItConfigsBuilder
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():

    # Arguments
    is_sim = LaunchConfiguration("is_sim")
    is_ignition = LaunchConfiguration("is_ignition")
    base_x = LaunchConfiguration("base_x")
    base_y = LaunchConfiguration("base_y")
    base_z = LaunchConfiguration("base_z")
    base_roll = LaunchConfiguration("base_roll")
    base_pitch = LaunchConfiguration("base_pitch")
    base_yaw = LaunchConfiguration("base_yaw")
    with_rviz = LaunchConfiguration("with_rviz")

    is_sim_arg = DeclareLaunchArgument(
        "is_sim", default_value="true",
        description="Use simulation time if true"
    )

    is_ignition_arg = DeclareLaunchArgument(
        "is_ignition", default_value="true",
        description="Use Ignition Gazebo if true"
    )
    base_x_arg = DeclareLaunchArgument("base_x", default_value="0.0")
    base_y_arg = DeclareLaunchArgument("base_y", default_value="0.0")
    base_z_arg = DeclareLaunchArgument("base_z", default_value="0.0")
    base_roll_arg = DeclareLaunchArgument("base_roll", default_value="0.0")
    base_pitch_arg = DeclareLaunchArgument("base_pitch", default_value="0.0")
    base_yaw_arg = DeclareLaunchArgument("base_yaw", default_value="0.0")
    with_rviz_arg = DeclareLaunchArgument(
        "with_rviz",
        default_value="false",
        description="Launch RViz2 with MoveIt config (heavy on RAM).",
    )

    # Build MoveIt config
    moveit_config = (
        MoveItConfigsBuilder("panda", package_name="panda_moveit")
        .robot_description(
            file_path=os.path.join(
                get_package_share_directory("panda_description"),
                "urdf",
                "panda.urdf.xacro"
            ),
            mappings={
                "is_ignition": is_ignition,
                "base_x": base_x,
                "base_y": base_y,
                "base_z": base_z,
                "base_roll": base_roll,
                "base_pitch": base_pitch,
                "base_yaw": base_yaw,
            }
        )
        .robot_description_semantic(file_path="config/panda.srdf")
        .trajectory_execution(file_path="config/moveit_controllers.yaml")
        .planning_pipelines(pipelines=["ompl"])
        .to_moveit_configs()
    )

    # Move Group Node
    move_group_node = Node(
        package="moveit_ros_move_group",
        executable="move_group",
        output="screen",
        parameters=[
            moveit_config.to_dict(),
            {"use_sim_time": is_sim},
            {"publish_robot_description_semantic": True}
        ],
        arguments=["--ros-args", "--log-level", "info"],
    )

    # RViz (optional; off by default in tfg_ycb_pick_place)
    rviz_config = os.path.join(
        get_package_share_directory("panda_moveit"),
        "rviz",
        "moveit.rviz",
    )

    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        output="log",
        arguments=["-d", rviz_config],
        parameters=[
            moveit_config.robot_description,
            moveit_config.robot_description_semantic,
            moveit_config.robot_description_kinematics,
            moveit_config.joint_limits,
        ],
        condition=IfCondition(with_rviz),
    )

    return LaunchDescription([
        is_sim_arg,
        is_ignition_arg,
        base_x_arg,
        base_y_arg,
        base_z_arg,
        base_roll_arg,
        base_pitch_arg,
        base_yaw_arg,
        with_rviz_arg,
        move_group_node,
        rviz_node,
    ])
