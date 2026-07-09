import os

from ament_index_python.packages import get_package_prefix, get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    IncludeLaunchDescription,
    LogInfo,
    OpaqueFunction,
    TimerAction,
)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def _workspace_root() -> str:
    """Raíz del workspace; portable vía TFG_WS o ~/tfg_robotics_ws."""
    return os.environ.get("TFG_WS", os.path.expanduser("~/tfg_robotics_ws"))


# Defaults de trabajo/demo (sobrescribibles desde CLI).
DEFAULT_MODEL_PATH = os.path.join(
    _workspace_root(), "models", "vision", "yolo_obb_best.pt"
)
DEFAULT_YCB_MODELS_PATH = os.path.join(
    _workspace_root(), "src", "gazebo_ycb", "models"
)


def _truthy(value: str) -> bool:
    return str(value).strip().lower() in ("1", "true", "yes", "on")


def _log_tfg_launch_config(context, *args, **kwargs):
    """Emite [TFG_LAUNCH_CONFIG] y valida model_path al arrancar."""
    keys = [
      "scene_preset",
      "spawn_scene_on_startup_delay_sec",
      "with_controller",
      "with_moveit",
      "with_perception",
      "with_slm",
      "with_web",
      "with_rviz",
      "gazebo_gui",
      "model_path",
      "ycb_models_path",
      "bridge_world_pose_info",
      "update_runtime_scene_from_actual_gazebo_pose",
      "gazebo_world_name",
      "spawn_backend",
      "move_home_on_startup",
    ]
    actions = [LogInfo(msg="[TFG_LAUNCH_CONFIG]")]
    for key in keys:
        val = LaunchConfiguration(key).perform(context)
        actions.append(LogInfo(msg=f"  {key}={val}"))

    perception_keys = [
        ("fast_snapshot_debug", "fast_snapshot_debug"),
        ("executor_publish_hz", "executor_publish_hz"),
        ("use_runtime_scene_gt", "use_runtime_scene_gt"),
        ("debug_overlay_simplified_for_demo", "debug_overlay_simplified_for_demo"),
        ("use_spawn_geometry_for_known_boxes", "use_spawn_geometry_for_known_boxes"),
        ("plan_before_prelude", "plan_before_prelude"),
    ]
    for label, key in perception_keys:
        val = LaunchConfiguration(key).perform(context)
        actions.append(LogInfo(msg=f"  {label}={val}"))

    model_path = LaunchConfiguration("model_path").perform(context)
    if not os.path.isfile(model_path):
        actions.append(
            LogInfo(
                msg=(
                    f"[LAUNCH_MODEL_PATH_ERROR] model_path={model_path} does not exist. "
                    "Perception will fail or return empty detections until you fix the path."
                )
            )
        )
    return actions


def generate_launch_description():
    model_path = LaunchConfiguration("model_path")
    ycb_models_path = LaunchConfiguration("ycb_models_path")
    gazebo_gui = LaunchConfiguration("gazebo_gui")
    confidence = LaunchConfiguration("confidence_threshold")
    max_sync_slop_sec = LaunchConfiguration("max_sync_slop_sec")
    with_web = LaunchConfiguration("with_web")
    with_slm = LaunchConfiguration("with_slm")
    with_rviz = LaunchConfiguration("with_rviz")
    web_port = LaunchConfiguration("web_port")
    with_controller = LaunchConfiguration("with_controller")
    with_moveit = LaunchConfiguration("with_moveit")
    with_perception = LaunchConfiguration("with_perception")
    move_home_on_startup = LaunchConfiguration("move_home_on_startup")
    move_home_startup_delay_sec = LaunchConfiguration("move_home_startup_delay_sec")
    home_open_gripper_at_startup = LaunchConfiguration("home_open_gripper_at_startup")
    home_close_gripper_at_startup = LaunchConfiguration("home_close_gripper_at_startup")
    spawn_backend = LaunchConfiguration("spawn_backend")
    delete_backend = LaunchConfiguration("delete_backend")
    allow_spawn_without_clear = LaunchConfiguration("allow_spawn_without_clear")
    texture_unique_cache = LaunchConfiguration("texture_unique_cache")
    texture_cache_dir = LaunchConfiguration("texture_cache_dir")
    base_x = LaunchConfiguration("base_x")
    base_y = LaunchConfiguration("base_y")
    base_z = LaunchConfiguration("base_z")
    base_roll = LaunchConfiguration("base_roll")
    base_pitch = LaunchConfiguration("base_pitch")
    base_yaw = LaunchConfiguration("base_yaw")
    gazebo_world_name = LaunchConfiguration("gazebo_world_name")
    bridge_world_pose_info = LaunchConfiguration("bridge_world_pose_info")
    bridge_world_dynamic_pose_info = LaunchConfiguration(
        "bridge_world_dynamic_pose_info"
    )
    update_runtime_scene_from_actual_gazebo_pose = LaunchConfiguration(
        "update_runtime_scene_from_actual_gazebo_pose"
    )
    fast_snapshot_debug = LaunchConfiguration("fast_snapshot_debug")
    executor_publish_hz = LaunchConfiguration("executor_publish_hz")
    executor_publish_on_every_valid_frame = LaunchConfiguration(
        "executor_publish_on_every_valid_frame"
    )
    use_runtime_scene_gt = LaunchConfiguration("use_runtime_scene_gt")
    use_spawn_geometry_for_known_boxes = LaunchConfiguration(
        "use_spawn_geometry_for_known_boxes"
    )
    debug_overlay_simplified_for_demo = LaunchConfiguration(
        "debug_overlay_simplified_for_demo"
    )
    enable_visual_pose_gate = LaunchConfiguration("enable_visual_pose_gate")
    plan_before_prelude = LaunchConfiguration("plan_before_prelude")
    plan_before_prelude_skip_workspace_prelude = LaunchConfiguration(
        "plan_before_prelude_skip_workspace_prelude"
    )
    scene_preset = LaunchConfiguration("scene_preset")
    scene_random_seed = LaunchConfiguration("scene_random_seed")
    demo_scene_min_clearance_m = LaunchConfiguration("demo_scene_min_clearance_m")
    spawn_scene_on_startup_delay_sec = LaunchConfiguration(
        "spawn_scene_on_startup_delay_sec"
    )

    dataset_cfg = os.path.join(
        get_package_share_directory("panda_vision"),
        "config",
        "ycb_obb_dataset.yaml",
    )

    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory("panda_description"),
                "launch",
                "gazebo.launch.py",
            )
        ),
        launch_arguments={
            "world_name": gazebo_world_name,
            "gazebo_gui": gazebo_gui,
            "bridge_world_pose_info": bridge_world_pose_info,
            "bridge_world_dynamic_pose_info": bridge_world_dynamic_pose_info,
            "ycb_models_path": ycb_models_path,
            "base_x": base_x,
            "base_y": base_y,
            "base_z": base_z,
            "base_roll": base_roll,
            "base_pitch": base_pitch,
            "base_yaw": base_yaw,
        }.items(),
    )

    controller = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory("panda_controller"),
                "launch",
                "panda_controller.launch.py",
            )
        ),
        launch_arguments={
            "is_sim": "True",
            "base_x": base_x,
            "base_y": base_y,
            "base_z": base_z,
            "base_roll": base_roll,
            "base_pitch": base_pitch,
            "base_yaw": base_yaw,
        }.items(),
        condition=IfCondition(with_controller),
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
            "is_sim": "True",
            "with_rviz": with_rviz,
            "base_x": base_x,
            "base_y": base_y,
            "base_z": base_z,
            "base_roll": base_roll,
            "base_pitch": base_pitch,
            "base_yaw": base_yaw,
        }.items(),
        condition=IfCondition(with_moveit),
    )

    perception = Node(
        package="panda_vision",
        executable="perception_node",
        name="perception_node",
        output="screen",
        parameters=[
            {"use_sim_time": True},
            {"vision_backend": "yolo26_obb"},
            {"model_path": model_path},
            {"publish_legacy_topic": True},
            {"legacy_detections_topic": "/detections_3d"},
            {"confidence_threshold": confidence},
            {"max_sync_slop_sec": max_sync_slop_sec},
            {"enable_robot_occlusion_filter": True},
            {"reject_detection_if_top_z_above_m": 0.62},
            {"reject_detection_if_centroid_z_above_m": 0.62},
            {"publish_debug_image": True},
            {"draw_grasp_debug_overlay": True},
            {"debug_overlay_mode": "grasp_clean"},
            {"debug_overlay_simplified_for_demo": debug_overlay_simplified_for_demo},
            {"scene_target_label": "cracker_box"},
            {"scene_preset": scene_preset},
            {"debug_image_publish_policy": "last_rich_overlay"},
            {"publish_debug_status_image": True},
            {"debug_image_publish_every_frame": True},
            {"debug_image_heartbeat_enabled": True},
            {"debug_image_qos_reliable": True},
            {"prefer_hybrid_fit_for_boxes": True},
            {"prefer_model_top_face_fit_for_boxes": True},
            {"debug_draw_gazebo_ground_truth_top_face": False},
            {"use_runtime_scene_gt": use_runtime_scene_gt},
            {"use_spawn_geometry_for_known_boxes": use_spawn_geometry_for_known_boxes},
            {"enable_visual_pose_gate": enable_visual_pose_gate},
            {"executor_publish_hz": executor_publish_hz},
            {"executor_publish_on_every_valid_frame": executor_publish_on_every_valid_frame},
            {"fast_snapshot_debug": fast_snapshot_debug},
            {"plan_before_prelude": plan_before_prelude},
            {"plan_before_prelude_skip_workspace_prelude": plan_before_prelude_skip_workspace_prelude},
            {"publish_cached_detection": True},
            {"cached_detection_publish_hz": 5.0},
            {"enable_perception_profiling": True},
            {"max_executor_payload_age_sec": 2.0},
        ],
        condition=IfCondition(with_perception),
    )

    runtime_scene_gt = Node(
        package="panda_vision",
        executable="runtime_scene_gt_node",
        name="runtime_scene_gt_node",
        output="screen",
        parameters=[
            {"use_sim_time": True},
            {"publish_hz": 1.0},
            {"world_frame": "world"},
        ],
        condition=IfCondition(with_perception),
    )

    runtime_spawner = Node(
        package="panda_vision",
        executable="runtime_scene_spawner",
        name="runtime_scene_spawner",
        output="screen",
        parameters=[
            {"use_sim_time": True},
            {"config_path": dataset_cfg},
            {"ycb_models_path": ycb_models_path},
            {"world_name": gazebo_world_name},
            {"spawn_backend": spawn_backend},
            {"delete_backend": delete_backend},
            {"allow_spawn_without_clear": allow_spawn_without_clear},
            {"texture_unique_cache": texture_unique_cache},
            {"texture_cache_dir": texture_cache_dir},
            {"excluded_spawn_objects": ["tomato_soup_can"]},
            {"min_objects": 4},
            {"max_objects": 4},
            {"clear_all_runtime_ycb_on_spawn": True},
            {"spawn_scene_on_startup": True},
            {"scene_preset": scene_preset},
            {"scene_random_seed": scene_random_seed},
            {"demo_scene_min_clearance_m": demo_scene_min_clearance_m},
            {"spawn_scene_on_startup_delay_sec": spawn_scene_on_startup_delay_sec},
            {
                "allowed_spawn_labels": (
                    "cracker_box,sugar_box,mustard_bottle,chips_can"
                ),
            },
            {"scene_target_label": "cracker_box"},
            {"update_runtime_scene_from_actual_gazebo_pose": update_runtime_scene_from_actual_gazebo_pose},
            {"post_spawn_settle_sec": 0.8},
            {"pose_stability_timeout_sec": 2.0},
            {"pose_stability_xy_threshold_m": 0.002},
            {"pose_stability_yaw_threshold_deg": 0.5},
            {"reject_topdown_if_tilted": True},
            {"max_upright_roll_pitch_deg": 3.0},
        ],
        condition=IfCondition(with_perception),
    )

    def _create_slm_nodes(context):
        """Nodos SLM legacy (llm_node/executor). Sustituidos por web_api cuando with_web=true."""
        with_slm_enabled = _truthy(LaunchConfiguration("with_slm").perform(context))
        with_web_enabled = _truthy(LaunchConfiguration("with_web").perform(context))
        if not with_slm_enabled or with_web_enabled:
            return []
        planner_params = os.path.join(
            get_package_share_directory("tfg_planner_slm"),
            "config",
            "pick_place_params.yaml",
        )
        return [
            Node(
                package="tfg_planner_slm",
                executable="vision_bridge_node",
                name="vision_bridge_node",
                output="screen",
                parameters=[{"use_sim_time": True}, {"pick_params_path": planner_params}],
            ),
            Node(
                package="tfg_planner_slm",
                executable="llm_node",
                name="llm_node",
                output="screen",
                parameters=[{"pick_params_path": planner_params}],
            ),
            Node(
                package="tfg_planner_slm",
                executable="executor_node",
                name="executor_node",
                output="screen",
                parameters=[{"pick_params_path": planner_params}],
            ),
        ]

    def _create_web_api_process(context):
        if not _truthy(LaunchConfiguration("with_web").perform(context)):
            return []
        port = LaunchConfiguration("web_port").perform(context)
        scene_preset = LaunchConfiguration("scene_preset").perform(context).strip()
        scene_id = scene_preset if scene_preset else "two_boxes_01"
        exe = os.path.join(
            get_package_prefix("tfg_planner_slm"),
            "lib",
            "tfg_planner_slm",
            "web_api",
        )
        return [
            ExecuteProcess(
                cmd=[
                    exe,
                    "--host",
                    "0.0.0.0",
                    "--port",
                    port,
                    "--log-json",
                    "--scene-id",
                    scene_id,
                ],
                output="screen",
            )
        ]

    move_home = Node(
        package="panda_controller",
        executable="move_to_home",
        name="move_to_home_startup",
        output="screen",
        parameters=[
            {"use_sim_time": True},
            {"open_gripper_at_home": home_open_gripper_at_startup},
            {"close_gripper_at_home": home_close_gripper_at_startup},
            # Tras moveit (7s) + spawners ros2_control; antes del spawn demo (~16s).
            {"action_wait_timeout_sec": 25.0},
            {"result_timeout_sec": 45.0},
            {"joint_state_verify_timeout_sec": 15.0},
            {"startup_max_attempts": 4},
            {"retry_delay_sec": 4.0},
        ],
        condition=IfCondition(move_home_on_startup),
    )

    launch_args = [
        DeclareLaunchArgument("model_path", default_value=DEFAULT_MODEL_PATH),
        DeclareLaunchArgument("ycb_models_path", default_value=DEFAULT_YCB_MODELS_PATH),
        DeclareLaunchArgument("confidence_threshold", default_value="0.35"),
        DeclareLaunchArgument("max_sync_slop_sec", default_value="0.30"),
        DeclareLaunchArgument("gazebo_gui", default_value="true"),
        DeclareLaunchArgument("with_slm", default_value="false"),
        DeclareLaunchArgument(
            "with_web",
            default_value="false",
            description=(
                "UI web + SLM v1.1 (web_api) en el mismo launch. "
                "Recomendado: false y lanzar tfg_web_api.launch.py en otra terminal."
            ),
        ),
        DeclareLaunchArgument("with_rviz", default_value="false"),
        DeclareLaunchArgument("web_port", default_value="8000"),
        DeclareLaunchArgument("with_controller", default_value="true"),
        DeclareLaunchArgument("with_moveit", default_value="true"),
        DeclareLaunchArgument("with_perception", default_value="true"),
        DeclareLaunchArgument("move_home_on_startup", default_value="true"),
        DeclareLaunchArgument(
            "move_home_startup_delay_sec",
            default_value="25.0",
            description=(
                "Segundos desde el launch hasta move_to_home (tras spawn demo ~22s "
                "y asentamiento físico)."
            ),
        ),
        DeclareLaunchArgument("home_open_gripper_at_startup", default_value="false"),
        DeclareLaunchArgument("home_close_gripper_at_startup", default_value="false"),
        DeclareLaunchArgument("spawn_backend", default_value="ros_gz_create_cli"),
        DeclareLaunchArgument("delete_backend", default_value="gz_service_cli"),
        DeclareLaunchArgument("allow_spawn_without_clear", default_value="false"),
        DeclareLaunchArgument("texture_unique_cache", default_value="true"),
        DeclareLaunchArgument("texture_cache_dir", default_value=""),
        DeclareLaunchArgument("base_x", default_value="0.0"),
        DeclareLaunchArgument("base_y", default_value="0.0"),
        DeclareLaunchArgument("base_z", default_value="0.0"),
        DeclareLaunchArgument("base_roll", default_value="0.0"),
        DeclareLaunchArgument("base_pitch", default_value="0.0"),
        DeclareLaunchArgument("base_yaw", default_value="0.0"),
        DeclareLaunchArgument("gazebo_world_name", default_value="vision_test_ycb"),
        DeclareLaunchArgument(
            "update_runtime_scene_from_actual_gazebo_pose",
            default_value="true",
        ),
        DeclareLaunchArgument(
            "bridge_world_pose_info",
            default_value="true",
            description=(
                "Puente ros_gz /world/<world>_world/pose/info para readback de pose real."
            ),
        ),
        DeclareLaunchArgument(
            "bridge_world_dynamic_pose_info",
            default_value="false",
            description="Puente opcional de dynamic_pose/info como fallback.",
        ),
        DeclareLaunchArgument("fast_snapshot_debug", default_value="true"),
        DeclareLaunchArgument("executor_publish_hz", default_value="3.0"),
        DeclareLaunchArgument(
            "executor_publish_on_every_valid_frame", default_value="true"
        ),
        DeclareLaunchArgument("use_runtime_scene_gt", default_value="true"),
        DeclareLaunchArgument("use_spawn_geometry_for_known_boxes", default_value="true"),
        DeclareLaunchArgument(
            "debug_overlay_simplified_for_demo", default_value="true"
        ),
        DeclareLaunchArgument("enable_visual_pose_gate", default_value="false"),
        DeclareLaunchArgument("plan_before_prelude", default_value="true"),
        DeclareLaunchArgument(
            "plan_before_prelude_skip_workspace_prelude", default_value="true"
        ),
        DeclareLaunchArgument(
            "scene_preset",
            default_value="",
            description=(
                "Preset de escena runtime: two_boxes_01/02/03 (cracker+sugar), "
                "chips_mustard_01 (chips+mustard, aleatorio), "
                "chips_mustard_02 (chips+mustard, layout fácil; goldens propios tras benchmark), "
                "demo_scene_01/02/03 (4 objetos), "
                "demo_scene_01_3obj/02_3obj/03_3obj (3 objetos, sin mostaza), "
                "demo_scene_02_remaining_sugar_mustard (sugar+mustard), "
                "deposit_02_cracker_chips (cracker+chips en cajón 0-1), "
                "deposit_03_mustard_only (cracker+chips+sugar en cajón 0-2; solo mustard en mesa). "
                "Escenas *_03 / chips_mustard_01 usan spawn aleatorio. Vacío = spawn aleatorio."
            ),
        ),
        DeclareLaunchArgument(
            "scene_random_seed",
            default_value="0",
            description=(
                "Semilla para escenas spawn_mode=random "
                "(two_boxes_03, chips_mustard_01; 0 = aleatorio cada arranque)."
            ),
        ),
        DeclareLaunchArgument(
            "demo_scene_min_clearance_m",
            default_value="0.03",
            description="Separación mínima extra entre objetos en presets demo (m).",
        ),
        DeclareLaunchArgument(
            "spawn_scene_on_startup_delay_sec",
            default_value="12.0",
            description=(
                "Segundos tras arrancar runtime_scene_spawner antes del spawn automático "
                "(activo si scene_preset no está vacío)."
            ),
        ),
    ]

    return LaunchDescription(
        launch_args
        + [
            OpaqueFunction(function=_log_tfg_launch_config),
            gazebo,
            TimerAction(period=6.0, actions=[controller]),
            TimerAction(period=9.0, actions=[moveit]),
            TimerAction(
                period=12.0,
                actions=[perception, runtime_scene_gt, runtime_spawner],
            ),
            TimerAction(
                period=15.0,
                actions=[
                    OpaqueFunction(function=_create_slm_nodes),
                    OpaqueFunction(function=_create_web_api_process),
                ],
            ),
            TimerAction(
                period=move_home_startup_delay_sec,
                actions=[move_home],
            ),
        ]
    )
