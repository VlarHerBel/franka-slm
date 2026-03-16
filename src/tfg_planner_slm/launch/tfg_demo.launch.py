from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

import os


def generate_launch_description():
    world_name = LaunchConfiguration("world_name")
    with_moveit = LaunchConfiguration("with_moveit")
    with_vision = LaunchConfiguration("with_vision")
    with_executor = LaunchConfiguration("with_executor")
    with_slm = LaunchConfiguration("with_slm")
    with_web = LaunchConfiguration("with_web")
    web_port = LaunchConfiguration("web_port")

    world_name_arg = DeclareLaunchArgument(
        "world_name",
        default_value="vision_test",
        description="Mundo de Gazebo a cargar desde panda_description/worlds.",
    )
    with_moveit_arg = DeclareLaunchArgument(
        "with_moveit",
        default_value="true",
        description="Lanza controladores y MoveIt para pick and place.",
    )
    with_vision_arg = DeclareLaunchArgument(
        "with_vision",
        default_value="true",
        description="Lanza object_detector y vision_bridge_node.",
    )
    with_executor_arg = DeclareLaunchArgument(
        "with_executor",
        default_value="true",
        description="Lanza el executor del pick and place.",
    )
    with_slm_arg = DeclareLaunchArgument(
        "with_slm",
        default_value="false",
        description="Lanza el nodo conversacional LLM.",
    )
    with_web_arg = DeclareLaunchArgument(
        "with_web",
        default_value="false",
        description="Lanza la interfaz web local.",
    )
    web_port_arg = DeclareLaunchArgument(
        "web_port",
        default_value="8000",
        description="Puerto HTTP de la interfaz web.",
    )

    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory("panda_description"),
                "launch",
                "gazebo.launch.py",
            )
        ),
        launch_arguments={"world_name": world_name}.items(),
    )

    controller = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory("panda_controller"),
                "launch",
                "panda_controller.launch.py",
            )
        ),
        condition=IfCondition(with_moveit),
        launch_arguments={"is_sim": "True"}.items(),
    )

    moveit = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory("panda_moveit"),
                "launch",
                "moveit.launch.py",
            )
        ),
        condition=IfCondition(with_moveit),
        launch_arguments={"is_sim": "True"}.items(),
    )

    object_detector = Node(
        package="panda_vision",
        executable="object_detector",
        name="object_detector",
        output="screen",
        condition=IfCondition(with_vision),
    )

    vision_bridge_node = Node(
        package="tfg_planner_slm",
        executable="vision_bridge_node",
        name="vision_bridge_node",
        output="screen",
        condition=IfCondition(with_vision),
    )

    executor_node = Node(
        package="tfg_planner_slm",
        executable="executor_node",
        name="executor_node",
        output="screen",
        condition=IfCondition(with_executor),
    )

    llm_node = Node(
        package="tfg_planner_slm",
        executable="llm_node",
        name="llm_node",
        output="screen",
        condition=IfCondition(with_slm),
    )

    web_bridge_node = Node(
        package="tfg_planner_slm",
        executable="web_bridge_node",
        name="web_bridge_node",
        output="screen",
        parameters=[{"port": web_port}],
        condition=IfCondition(with_web),
    )

    return LaunchDescription(
        [
            world_name_arg,
            with_moveit_arg,
            with_vision_arg,
            with_executor_arg,
            with_slm_arg,
            with_web_arg,
            web_port_arg,
            gazebo,
            controller,
            moveit,
            object_detector,
            vision_bridge_node,
            executor_node,
            llm_node,
            web_bridge_node,
        ]
    )
