from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    model_path_arg = DeclareLaunchArgument(
        "model_path",
        default_value="yolo26n-obb.pt",
        description="Path to the Ultralytics OBB model weights.",
    )
    vision_backend_arg = DeclareLaunchArgument(
        "vision_backend",
        default_value="yolo26_obb",
        description="Vision backend id for panda_vision/perception_node.",
    )
    publish_legacy_arg = DeclareLaunchArgument(
        "publish_legacy_topic",
        default_value="false",
        description="Also publish /detections_3d for the existing vision bridge.",
    )
    confidence_arg = DeclareLaunchArgument(
        "confidence_threshold",
        default_value="0.35",
        description="Ultralytics confidence threshold.",
    )

    perception_node = Node(
        package="panda_vision",
        executable="perception_node",
        name="perception_node",
        output="screen",
        parameters=[
            {"use_sim_time": True},
            {"vision_backend": LaunchConfiguration("vision_backend")},
            {"model_path": LaunchConfiguration("model_path")},
            {"publish_legacy_topic": LaunchConfiguration("publish_legacy_topic")},
            {"legacy_detections_topic": "/detections_3d"},
            {"confidence_threshold": LaunchConfiguration("confidence_threshold")},
        ],
    )

    return LaunchDescription(
        [
            model_path_arg,
            vision_backend_arg,
            publish_legacy_arg,
            confidence_arg,
            perception_node,
        ]
    )
