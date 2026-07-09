import os
from os import pathsep
from pathlib import Path
from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction, SetEnvironmentVariable
from launch.substitutions import (
    Command,
    EnvironmentVariable,
    LaunchConfiguration,
    PathJoinSubstitution,
    PythonExpression,
)
from launch.launch_description_sources import PythonLaunchDescriptionSource

from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def _gz_world_name(world_name: str) -> str:
    w = (world_name or "").strip()
    return w if w.endswith("_world") else f"{w}_world"


def _truthy(value: str) -> bool:
    return value.strip().lower() in ("1", "true", "yes", "on")


def _create_gz_ros2_bridge(context, *args, **kwargs):
    world_name = LaunchConfiguration("world_name").perform(context)
    gz_world = _gz_world_name(world_name)
    bridge_pose = _truthy(LaunchConfiguration("bridge_world_pose_info").perform(context))
    bridge_dynamic = _truthy(
        LaunchConfiguration("bridge_world_dynamic_pose_info").perform(context)
    )

    bridge_args = [
        "/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock",
        "/camera/camera_info@sensor_msgs/msg/CameraInfo[gz.msgs.CameraInfo",
    ]
    if bridge_pose:
        bridge_args.append(
            f"/world/{gz_world}/pose/info@tf2_msgs/msg/TFMessage[gz.msgs.Pose_V"
        )
        bridge_args.append(
            f"/world/{gz_world}/set_pose@ros_gz_interfaces/srv/SetEntityPose"
        )
    if bridge_dynamic:
        bridge_args.append(
            f"/world/{gz_world}/dynamic_pose/info@tf2_msgs/msg/TFMessage[gz.msgs.Pose_V"
        )

    return [
        Node(
            package="ros_gz_bridge",
            executable="parameter_bridge",
            arguments=bridge_args,
            output="screen",
        )
    ]


def generate_launch_description():
    panda_description = get_package_share_directory("panda_description")

    model_arg = DeclareLaunchArgument(
        name="model", default_value=os.path.join(
                panda_description, "urdf", "panda.urdf.xacro"
            ),
        description="Absolute path to robot urdf file"
    )

    world_name_arg = DeclareLaunchArgument(name="world_name", default_value="empty")
    gazebo_gui_arg = DeclareLaunchArgument(name="gazebo_gui", default_value="true")
    ycb_models_path_arg = DeclareLaunchArgument(
        name="ycb_models_path",
        default_value=str(Path.home() / "tfg_robotics_ws" / "src" / "gazebo_ycb" / "models"),
    )
    base_x_arg = DeclareLaunchArgument(name="base_x", default_value="0.0")
    base_y_arg = DeclareLaunchArgument(name="base_y", default_value="0.0")
    base_z_arg = DeclareLaunchArgument(name="base_z", default_value="0.0")
    base_roll_arg = DeclareLaunchArgument(name="base_roll", default_value="0.0")
    base_pitch_arg = DeclareLaunchArgument(name="base_pitch", default_value="0.0")
    base_yaw_arg = DeclareLaunchArgument(name="base_yaw", default_value="0.0")
    bridge_world_pose_info_arg = DeclareLaunchArgument(
        name="bridge_world_pose_info",
        default_value="false",
        description="Puente ROS de /world/<world>_world/pose/info (tf2_msgs/TFMessage).",
    )
    bridge_world_dynamic_pose_info_arg = DeclareLaunchArgument(
        name="bridge_world_dynamic_pose_info",
        default_value="false",
        description="Puente opcional de /world/<world>_world/dynamic_pose/info.",
    )

    world_path = PathJoinSubstitution([
            panda_description,
            "worlds",
            PythonExpression(expression=["'", LaunchConfiguration("world_name"), "'", " + '.world'"])
        ]
    )

    model_path = str(Path(panda_description).parent.resolve())
    model_path += pathsep + os.path.join(get_package_share_directory("panda_description"), 'models')

    gazebo_resource_path = SetEnvironmentVariable(
        "GZ_SIM_RESOURCE_PATH",
        value=[
            LaunchConfiguration("ycb_models_path"),
            pathsep,
            model_path,
            pathsep,
            EnvironmentVariable("GZ_SIM_RESOURCE_PATH", default_value=""),
        ],
    )
    ignition_resource_path = SetEnvironmentVariable(
        "IGN_GAZEBO_RESOURCE_PATH",
        value=[
            LaunchConfiguration("ycb_models_path"),
            pathsep,
            model_path,
            pathsep,
            EnvironmentVariable("IGN_GAZEBO_RESOURCE_PATH", default_value=""),
        ],
    )

    ros_distro = os.environ["ROS_DISTRO"]
    is_ignition = "True" if ros_distro == "humble" else "False"

    robot_description = ParameterValue(Command([
            "xacro ",
            LaunchConfiguration("model"),
            " is_ignition:=",
            is_ignition,
            " base_x:=",
            LaunchConfiguration("base_x"),
            " base_y:=",
            LaunchConfiguration("base_y"),
            " base_z:=",
            LaunchConfiguration("base_z"),
            " base_roll:=",
            LaunchConfiguration("base_roll"),
            " base_pitch:=",
            LaunchConfiguration("base_pitch"),
            " base_yaw:=",
            LaunchConfiguration("base_yaw"),
        ]),
        value_type=str
    )

    robot_state_publisher_node = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        parameters=[{"robot_description": robot_description,
                     "use_sim_time": True}]
    )

    gazebo = IncludeLaunchDescription(
                PythonLaunchDescriptionSource([os.path.join(
                    get_package_share_directory("ros_gz_sim"), "launch"), "/gz_sim.launch.py"]),
                launch_arguments={
                    "gz_args": PythonExpression(
                        [
                            "'",
                            world_path,
                            " -v 4 -r ' if '",
                            LaunchConfiguration("gazebo_gui"),
                            "' == 'true' else '",
                            world_path,
                            " -v 4 -r -s'",
                        ]
                    )
                }.items()
             )


    gz_spawn_entity = Node(
        package="ros_gz_sim",
        executable="create",
        output="screen",
        arguments=[
            "-topic", "robot_description",
            "-name", "panda",
        ],
    )


    ros_gz_image_bridge = Node(
        package="ros_gz_image",
        executable="image_bridge",
        arguments=["/camera/image_raw"]
    )
    ros_gz_depth_bridge = Node(
        package="ros_gz_image",
        executable="image_bridge",
        arguments=["/camera/depth_image"]
    )

    return LaunchDescription([
        model_arg,
        world_name_arg,
        gazebo_gui_arg,
        ycb_models_path_arg,
        base_x_arg,
        base_y_arg,
        base_z_arg,
        base_roll_arg,
        base_pitch_arg,
        base_yaw_arg,
        bridge_world_pose_info_arg,
        bridge_world_dynamic_pose_info_arg,
        gazebo_resource_path,
        ignition_resource_path,
        robot_state_publisher_node,
        gazebo,
        gz_spawn_entity,
        OpaqueFunction(function=_create_gz_ros2_bridge),
        ros_gz_image_bridge,
        ros_gz_depth_bridge,
    ])