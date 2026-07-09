#!/usr/bin/env python3
"""Modular perception: vision Strategy, Open3D top plane, grasp service clients, JSON out."""

from __future__ import annotations

# transforms3d (via tf_transformations) expects np.float on older stacks.
import numpy as np

if not hasattr(np, "float"):
    setattr(np, "float", float)

import json
import math
import threading
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import cv2
import rclpy
import tf2_ros
import tf_transformations
from cv_bridge import CvBridge
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.clock import Clock, ClockType
from rclpy.duration import Duration
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import String
from std_srvs.srv import Trigger
from tf2_msgs.msg import TFMessage

from panda_vision.backends.base import VisionBackend
from panda_vision.backends.factory import create_vision_backend
from panda_vision.geometry.camera_projection import (
    depth_mask_to_points_camera,
    scaled_intrinsics_from_camera_info,
)
from panda_vision.geometry.open3d_top_surface import (
    Open3DTopSurfaceConfig,
    estimate_top_surface_plane_centroid,
)
from panda_vision.grasp.anygrasp_client import AnyGraspClient
from panda_vision.grasp.foundation_pose_client import FoundationPoseClient
from panda_vision.grasp.none_client import NoGraspClient
from panda_vision.grasp.mustard_cap_center import apply_mustard_cap_center_calibration
from panda_vision.grasp.mustard_bottle_axis_semantics import (
    apply_mustard_bottle_axis_semantics,
    log_mustard_overlay_axis_debug,
)
from panda_vision.grasp.object_grasp_policy import (
    OBJECT_DB,
    TALL_OBJECT_CAP_CENTER_SDF_OFFSET_SOURCE,
    TALL_OBJECT_CAP_CENTER_SOURCE,
    apply_tall_object_topdown_grasp_center_offset,
    export_grasp_policy_for_executor,
    get_collision_dimensions,
    get_grasp_policy,
    normalize_label,
    resolve_tall_object_top_z_m,
)
from panda_vision.spawn.known_object_geometry import (
    DEFAULT_MUSTARD_CAP_CENTER_MODE,
    MUSTARD_TOP_CAP_CENTER_GEOMETRY_SOURCE,
    MUSTARD_VERTICAL_AXIS_CAP_CENTER_SOURCE,
    ROLE_OBSTACLE,
    ROLE_TARGET,
    RUNTIME_GT_TALL_CAP_CENTER_SOURCES,
    apply_tall_object_sdf_geometry_correction,
    is_runtime_gt_tall_cap_center_source,
)
from panda_vision.grasp.known_object_pose_fit import (
    fit_known_top_rectangle_pose,
    log_pose_fit_summary,
    try_hybrid_top_face_known_dims_fit,
)
from panda_vision.grasp.gazebo_gt_top_face import (
    build_gt_top_face_corners_base,
    build_gt_top_face_from_spawner_entry,
    check_model_dim_convention,
    compute_top_face_gt_metrics,
    evaluate_visual_pose_gate,
    find_runtime_entity_for_label,
    log_model_dim_convention_check,
    log_pose_gate_visual,
    log_top_face_gt_compare,
    model_top_face_mask_coherence,
    update_world_pose_cache,
)
from panda_vision.spawn.runtime_scene_gt import (
    GT_OBJECTS_TOPIC,
    find_gt_object_for_label,
    find_gt_object_nearest_xy,
    gt_objects_qos,
    parse_gt_payload,
)
from panda_vision.spawn.runtime_scene_gt_geometry import (
    compute_synthetic_operational_top_face_base,
    enrich_gt_object_entry_base,
    get_known_box_gt_spec,
    is_known_spawn_geometry_box_label,
    log_synthetic_top_face_overlay_projection,
    resolve_runtime_gt_spawn_axes,
)
from panda_vision.grasp.model_box_top_face_fit import (
    fit_model_cuboid_top_face,
    is_box_like_known_shape,
    log_model_top_face_summary,
    merge_hybrid_as_model_source,
    observed_vs_model_corner_error_m,
)
from panda_vision.spawn.demo_scene_presets import (
    is_consolidated_demo_scene_objects,
    is_demo_scene_preset,
    log_demo_scene_vision_labels,
    runtime_labels_from_scene_objects,
)
from panda_vision.spawn.gz_spawn_runtime import gz_world_name_from_param
from panda_vision.grasp.top_face_extractor import (
    extract_top_face_points,
    log_top_face_summary,
)
from panda_vision.point_cloud_ros import numpy_xyz_to_pointcloud2
from panda_vision.types import PerceptionTelemetry, pose_to_dict


_EDGE_GRASP_OFFSET_M = 0.015
DEMO_KNOWN_OBJECTS: Set[str] = {
    "cracker_box",
    "sugar_box",
    "gelatin_box",
    "mustard_bottle",
    "bleach_cleanser",
    "chips_can",
}
KNOWN_BOX_LABELS: Set[str] = {
    "cracker_box",
    "sugar_box",
    "gelatin_box",
    "pudding_box",
}
TALL_KNOWN_OBJECT_LABELS: Set[str] = {
    "mustard_bottle",
    "bleach_cleanser",
}
CYLINDER_KNOWN_LABELS: Set[str] = {"chips_can"}


def _mustard_gap_closing_source_ok(closing_yaw_source: Any) -> bool:
    return str(closing_yaw_source or "").strip().startswith(
        "runtime_gt_mustard_gap_axis_"
    )


def _resolve_tall_cap_offset_local_xy(
    entry: Dict[str, Any],
    tall_dbg: Optional[Dict[str, Any]] = None,
) -> Tuple[Tuple[float, float], str]:
    """Offset local XY para logs; nunca KeyError por claves opcionales del RuntimeScene."""
    dbg = tall_dbg if isinstance(tall_dbg, dict) else {}
    raw = dbg.get("offset_local_xy")
    if isinstance(raw, (list, tuple)) and len(raw) >= 2:
        return (float(raw[0]), float(raw[1])), "offset_local_xy"
    for key, src in (
        (
            "mustard_top_cap_center_offset_local_xyz",
            "mustard_top_cap_center_offset_local_xyz",
        ),
        (
            "tall_object_sdf_cap_center_offset_local",
            "tall_object_sdf_cap_center_offset_local",
        ),
        (
            "geometry_center_to_cap_center_offset_local_xyz",
            "geometry_center_to_cap_center_offset_local_xyz",
        ),
    ):
        val = entry.get(key)
        if isinstance(val, (list, tuple)) and len(val) >= 2:
            return (float(val[0]), float(val[1])), src
    return (0.0, 0.0), "fallback_zero"

_DEBUG_IMAGE_POLICIES = (
    "last_rich_overlay",
    "live_rgb_status",
    "heartbeat_only",
)


def _safe_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _is_edge_strategy(strategy: str) -> bool:
    return "edge" in str(strategy).lower() or "push_to_edge" in str(strategy).lower()


class PerceptionNode(Node):
    def __init__(self) -> None:
        super().__init__("perception_node")

        self.declare_parameter("target_frame", "panda_link0")
        self.declare_parameter("camera_optical_frame", "camera_link_optical")
        self.declare_parameter("vision_backend", "yolo26_obb")
        self.declare_parameter("model_path", "yolo26n-obb.pt")
        self.declare_parameter("grounded_model_name", "stub")
        self.declare_parameter("confidence_threshold", 0.35)
        self.declare_parameter("min_mask_pixels", 200)
        self.declare_parameter("max_sync_slop_sec", 0.30)
        self.declare_parameter("text_prompt", "object")
        self.declare_parameter("text_prompt_topic", "/vision/text_prompt")
        self.declare_parameter("grasp_backend", "none")
        self.declare_parameter("foundation_pose_service", "/foundation_pose/estimate")
        self.declare_parameter("anygrasp_service", "/anygrasp/grasps")
        self.declare_parameter("grasp_service_timeout_sec", 5.0)
        self.declare_parameter("vision_executor_topic", "/vision_to_executor")
        self.declare_parameter("publish_legacy_topic", False)
        self.declare_parameter("legacy_detections_topic", "/detections_3d")
        self.declare_parameter("publish_debug_image", True)
        self.declare_parameter("debug_image_topic", "/vision/debug_image")
        self.declare_parameter("debug_image_qos_reliable", True)
        self.declare_parameter("debug_image_publish_every_frame", True)
        self.declare_parameter("debug_image_heartbeat_enabled", True)
        self.declare_parameter("debug_image_heartbeat_hz", 1.0)
        self.declare_parameter("debug_image_save_to_disk", False)
        self.declare_parameter("debug_image_save_dir", "/tmp/tfg_debug_images")
        self.declare_parameter("debug_image_log_every_n_frames", 30)
        self.declare_parameter("debug_image_publish_policy", "last_rich_overlay")
        self.declare_parameter("debug_draw_raw_yolo_detections", True)
        self.declare_parameter("publish_debug_status_image", True)
        self.declare_parameter(
            "debug_status_image_topic", "/vision/debug_status_image"
        )
        # Open3D
        self.declare_parameter("open3d_voxel_size_m", 0.003)
        self.declare_parameter("open3d_normal_nn_max_nn", 30)
        self.declare_parameter("open3d_normal_radius_m", 0.02)
        self.declare_parameter("open3d_max_normal_angle_deg", 35.0)
        self.declare_parameter("open3d_ransac_distance_m", 0.004)
        self.declare_parameter("open3d_ransac_n", 3)
        self.declare_parameter("open3d_ransac_iterations", 1000)
        self.declare_parameter("open3d_min_inliers", 50)
        self.declare_parameter("open3d_z_min_m", 0.05)
        self.declare_parameter("top_grasp_approach_offset_m", [0.0, 0.0, 0.10])
        self.declare_parameter("top_grasp_offset_m", [0.0, 0.0, -0.02])
        self.declare_parameter("enable_tf_diagnostics", True)
        self.declare_parameter("tf_debug_every_n_frames", 30)
        self.declare_parameter("warn_base_z_min_m", 0.15)
        self.declare_parameter("warn_base_z_max_m", 0.60)
        self.declare_parameter("warn_base_x_min_m", 0.20)
        self.declare_parameter("warn_base_x_max_m", 0.90)
        self.declare_parameter("center_method", "top_surface_center")
        self.declare_parameter("top_surface_band_m", 0.02)
        self.declare_parameter("min_top_surface_points", 30)
        self.declare_parameter("use_measured_dimensions_for_policy", False)
        self.declare_parameter("table_z_m", 0.27)
        self.declare_parameter("enable_robot_occlusion_filter", True)
        self.declare_parameter("reject_detection_if_top_z_above_m", 0.62)
        self.declare_parameter("reject_detection_if_centroid_z_above_m", 0.62)
        self.declare_parameter("reject_detection_if_top_z_below_m", 0.20)
        self.declare_parameter("reject_out_of_table_before_fit", True)
        self.declare_parameter("target_label_filter", "")
        self.declare_parameter("scene_preset", "")
        self.declare_parameter("publish_cached_detection", True)
        self.declare_parameter("cached_detection_publish_hz", 5.0)
        self.declare_parameter("cache_ttl_s", 2.0)
        self.declare_parameter("max_executor_payload_age_sec", 2.0)
        self.declare_parameter("prefer_hybrid_fit_for_boxes", True)
        self.declare_parameter("prefer_model_top_face_fit_for_boxes", True)
        self.declare_parameter("debug_global_rectangle_search", False)
        self.declare_parameter("draw_grasp_debug_overlay", True)
        self.declare_parameter("debug_overlay_mode", "grasp_clean")
        self.declare_parameter("debug_overlay_simplified_for_demo", True)
        self.declare_parameter("debug_draw_gazebo_ground_truth_top_face", False)
        self.declare_parameter("use_runtime_scene_gt", True)
        self.declare_parameter("use_spawn_geometry_for_known_boxes", True)
        self.declare_parameter("mustard_bottle_axis_mapping", "normal")
        self.declare_parameter("mustard_cap_center_offset_candidate_index", -1)
        self.declare_parameter("mustard_cap_center_offset_long_m", 0.0)
        self.declare_parameter("mustard_cap_center_offset_short_m", 0.0)
        self.declare_parameter(
            "mustard_cap_center_mode",
            DEFAULT_MUSTARD_CAP_CENTER_MODE,
        )
        self.declare_parameter("allow_operational_source_fallback_for_debug", False)
        self.declare_parameter("enable_visual_pose_gate", False)
        self.declare_parameter("visual_pose_gate_max_model_vs_gt_center_xy_m", 0.015)
        self.declare_parameter("visual_pose_gate_max_observed_vs_model_center_xy_m", 0.025)
        self.declare_parameter("visual_pose_gate_require_mask_coherence", True)
        self.declare_parameter("debug_gt_entity_prefix", "runtime_ycb")
        self.declare_parameter("debug_gt_use_only_for_visualization", True)
        self.declare_parameter("debug_gt_world_frame", "world")
        self.declare_parameter("debug_gt_gazebo_world_name", "vision_test_ycb")
        self.declare_parameter("debug_gt_world_pose_topic", "")
        self.declare_parameter("enable_perception_profiling", True)
        self.declare_parameter("fast_snapshot_debug", False)
        self.declare_parameter("executor_publish_hz", 3.0)
        self.declare_parameter("executor_publish_on_every_valid_frame", True)
        self.declare_parameter("max_first_snapshot_wait_sec", 5.0)
        self.declare_parameter("perception_timing_log_every_n_frames", 30)

        self._target_frame = str(self.get_parameter("target_frame").value)
        self._camera_optical_frame = str(
            self.get_parameter("camera_optical_frame").value
        )
        self._max_slop = float(self.get_parameter("max_sync_slop_sec").value)
        self._min_mask_pixels = int(self.get_parameter("min_mask_pixels").value)
        self._param_prompt = str(self.get_parameter("text_prompt").value)
        self._grasp_timeout = float(
            self.get_parameter("grasp_service_timeout_sec").value
        )
        self._publish_debug = bool(self.get_parameter("publish_debug_image").value)
        self._debug_draw_raw_yolo_detections = bool(
            self.get_parameter("debug_draw_raw_yolo_detections").value
        )
        self._debug_image_qos_reliable = bool(
            self.get_parameter("debug_image_qos_reliable").value
        )
        self._debug_image_log_every_n = max(
            1, int(self.get_parameter("debug_image_log_every_n_frames").value)
        )
        self._debug_publish_every_frame = bool(
            self.get_parameter("debug_image_publish_every_frame").value
        )
        self._debug_heartbeat_enabled = bool(
            self.get_parameter("debug_image_heartbeat_enabled").value
        )
        self._debug_heartbeat_hz = max(
            0.2, float(self.get_parameter("debug_image_heartbeat_hz").value)
        )
        self._debug_image_save_to_disk = bool(
            self.get_parameter("debug_image_save_to_disk").value
        )
        self._debug_image_save_dir = Path(
            str(self.get_parameter("debug_image_save_dir").value)
        ).expanduser()
        self._debug_publish_count = 0
        self._debug_heartbeat_publish_count = 0
        self._debug_image_last_publish_monotonic = 0.0
        self._last_rich_debug_bgr: Optional[np.ndarray] = None
        self._last_rich_debug_stamp_monotonic = 0.0
        self._last_rich_debug_label = ""
        self._last_rich_debug_reason = ""
        self._last_rich_debug_has_overlays = False
        self._debug_frame_serial = 0
        policy_raw = str(self.get_parameter("debug_image_publish_policy").value).strip().lower()
        if policy_raw not in _DEBUG_IMAGE_POLICIES:
            self.get_logger().warn(
                "[DEBUG_IMAGE] debug_image_publish_policy='%s' invalid; using last_rich_overlay"
                % policy_raw
            )
            policy_raw = "last_rich_overlay"
        self._debug_publish_policy = policy_raw
        self._publish_debug_status_image = bool(
            self.get_parameter("publish_debug_status_image").value
        )
        self._debug_status_image_topic = str(
            self.get_parameter("debug_status_image_topic").value
        ).strip()
        self._use_sim_time_flag = bool(getattr(self, "use_sim_time", False))
        self._debug_heartbeat_stop: Optional[threading.Event] = None
        self._debug_heartbeat_thread: Optional[threading.Thread] = None
        self._debug_image_heartbeat_timer = None
        self._draw_debug_overlay = bool(self.get_parameter("draw_grasp_debug_overlay").value)
        _dom = str(self.get_parameter("debug_overlay_mode").value).strip().lower()
        if _dom not in ("full", "grasp_clean"):
            self.get_logger().warn(
                "debug_overlay_mode='%s' invalido; usando grasp_clean" % _dom
            )
            _dom = "grasp_clean"
        self._debug_overlay_mode = _dom
        self._debug_overlay_simplified_for_demo = bool(
            self.get_parameter("debug_overlay_simplified_for_demo").value
        )
        self._debug_draw_gazebo_gt = bool(
            self.get_parameter("debug_draw_gazebo_ground_truth_top_face").value
        )
        if self._debug_overlay_simplified_for_demo:
            self._debug_draw_gazebo_gt = False
        self._use_runtime_scene_gt = bool(
            self.get_parameter("use_runtime_scene_gt").value
        )
        self._use_spawn_geometry_for_known_boxes = bool(
            self.get_parameter("use_spawn_geometry_for_known_boxes").value
        )
        _mustard_map = str(self.get_parameter("mustard_bottle_axis_mapping").value).strip().lower()
        if _mustard_map not in (
            "normal",
            "swap_major_minor",
            "yaw_offset_plus_90",
            "yaw_offset_minus_90",
        ):
            self.get_logger().warn(
                "mustard_bottle_axis_mapping='%s' invalido; usando normal" % _mustard_map
            )
            _mustard_map = "normal"
        self._mustard_bottle_axis_mapping = _mustard_map
        try:
            self._mustard_cap_offset_candidate_index = int(
                self.get_parameter("mustard_cap_center_offset_candidate_index").value
            )
        except (TypeError, ValueError):
            self._mustard_cap_offset_candidate_index = -1
        try:
            self._mustard_cap_offset_long_m = float(
                self.get_parameter("mustard_cap_center_offset_long_m").value
            )
        except (TypeError, ValueError):
            self._mustard_cap_offset_long_m = 0.0
        try:
            self._mustard_cap_offset_short_m = float(
                self.get_parameter("mustard_cap_center_offset_short_m").value
            )
        except (TypeError, ValueError):
            self._mustard_cap_offset_short_m = 0.0
        _mustard_cap_mode = str(
            self.get_parameter("mustard_cap_center_mode").value
        ).strip().lower()
        if _mustard_cap_mode not in (
            "vertical_axis_from_footprint",
            "sdf_offset",
            "mesh_local_cap_center",
        ):
            self.get_logger().warn(
                "mustard_cap_center_mode='%s' invalido; usando %s"
                % (_mustard_cap_mode, DEFAULT_MUSTARD_CAP_CENTER_MODE)
            )
            _mustard_cap_mode = DEFAULT_MUSTARD_CAP_CENTER_MODE
        self._mustard_cap_center_mode = _mustard_cap_mode
        self._allow_operational_source_fallback_for_debug = bool(
            self.get_parameter("allow_operational_source_fallback_for_debug").value
        )
        self._enable_visual_pose_gate = bool(
            self.get_parameter("enable_visual_pose_gate").value
        )
        self._visual_gate_max_model_gt_xy = float(
            self.get_parameter("visual_pose_gate_max_model_vs_gt_center_xy_m").value
        )
        self._visual_gate_max_obs_model_xy = float(
            self.get_parameter("visual_pose_gate_max_observed_vs_model_center_xy_m").value
        )
        self._visual_gate_require_mask = bool(
            self.get_parameter("visual_pose_gate_require_mask_coherence").value
        )
        self._gt_spawner_by_entity: Dict[str, Dict[str, Any]] = {}
        self._runtime_scene_stamp_sec: float = 0.0
        self._runtime_scene_source: str = "runtime_scene_gt"
        self._debug_gt_entity_prefix = str(
            self.get_parameter("debug_gt_entity_prefix").value
        ).strip()
        self._debug_gt_viz_only = bool(
            self.get_parameter("debug_gt_use_only_for_visualization").value
        )
        self._debug_gt_world_frame = str(
            self.get_parameter("debug_gt_world_frame").value
        ).strip() or "world"
        _gz_w = str(self.get_parameter("debug_gt_gazebo_world_name").value).strip()
        self._debug_gt_gz_world = gz_world_name_from_param(_gz_w or "vision_test_ycb")
        _gt_topic_in = str(self.get_parameter("debug_gt_world_pose_topic").value).strip()
        self._debug_gt_pose_topic = (
            _gt_topic_in
            if _gt_topic_in
            else f"/world/{self._debug_gt_gz_world}/pose/info"
        )
        self._gt_world_poses: Dict[str, Tuple[float, float, float, float, float, float]] = {}
        self._gt_compare_log_serial = 0
        self._publish_legacy = bool(self.get_parameter("publish_legacy_topic").value)
        self._enable_robot_occlusion_filter = bool(
            self.get_parameter("enable_robot_occlusion_filter").value
        )
        self._reject_top_z_above = float(
            self.get_parameter("reject_detection_if_top_z_above_m").value
        )
        self._reject_centroid_z_above = float(
            self.get_parameter("reject_detection_if_centroid_z_above_m").value
        )
        self._reject_top_z_below = float(
            self.get_parameter("reject_detection_if_top_z_below_m").value
        )
        self._reject_out_of_table_before_fit = bool(
            self.get_parameter("reject_out_of_table_before_fit").value
        )
        self._target_label_filter = str(
            self.get_parameter("target_label_filter").value
        ).strip().lower()
        self._scene_preset = str(self.get_parameter("scene_preset").value).strip()
        self._publish_cached_detection = bool(
            self.get_parameter("publish_cached_detection").value
        )
        self._cache_ttl_s = float(self.get_parameter("cache_ttl_s").value)
        self._max_executor_payload_age_sec = float(
            self.get_parameter("max_executor_payload_age_sec").value
        )
        self._prefer_hybrid_fit_for_boxes = bool(
            self.get_parameter("prefer_hybrid_fit_for_boxes").value
        )
        self._prefer_model_top_face_fit_for_boxes = bool(
            self.get_parameter("prefer_model_top_face_fit_for_boxes").value
        )
        self._debug_global_rectangle_search = bool(
            self.get_parameter("debug_global_rectangle_search").value
        )
        self._enable_perception_profiling = bool(
            self.get_parameter("enable_perception_profiling").value
        )
        self._fast_snapshot_debug = bool(self.get_parameter("fast_snapshot_debug").value)
        self._executor_publish_hz = max(
            0.5, float(self.get_parameter("executor_publish_hz").value)
        )
        self._executor_publish_on_every_valid = bool(
            self.get_parameter("executor_publish_on_every_valid_frame").value
        )
        self._max_first_snapshot_wait_sec = max(
            0.5, float(self.get_parameter("max_first_snapshot_wait_sec").value)
        )
        self._timing_log_every_n = max(
            1, int(self.get_parameter("perception_timing_log_every_n_frames").value)
        )
        if self._fast_snapshot_debug:
            self._debug_draw_gazebo_gt = False
            self._debug_overlay_mode = "grasp_clean"
            self._debug_overlay_simplified_for_demo = True
            self._executor_publish_on_every_valid = True
            self._timing_log_every_n = min(self._timing_log_every_n, 5)
            self.get_logger().info(
                "[FAST_SNAPSHOT_DEBUG] GT off, grasp_clean overlay, executor publish on valid frame"
            )
        self._rgb_topic = "/camera/image_raw"
        self._depth_topic = "/camera/depth_image"
        self._camera_info_topic = "/camera/camera_info"
        self._vision_executor_topic = str(
            self.get_parameter("vision_executor_topic").value
        )
        self._legacy_detections_topic = str(
            self.get_parameter("legacy_detections_topic").value
        )
        self._debug_image_topic = str(self.get_parameter("debug_image_topic").value)
        self._model_path = str(self.get_parameter("model_path").value)
        self._vision_backend = str(self.get_parameter("vision_backend").value)
        self._confidence_threshold = float(
            self.get_parameter("confidence_threshold").value
        )
        self._top_grasp_approach_offset = [
            float(v) for v in self.get_parameter("top_grasp_approach_offset_m").value
        ]
        self._top_grasp_offset = [
            float(v) for v in self.get_parameter("top_grasp_offset_m").value
        ]
        self._enable_tf_diagnostics = bool(
            self.get_parameter("enable_tf_diagnostics").value
        )
        self._tf_debug_every_n_frames = max(
            1, int(self.get_parameter("tf_debug_every_n_frames").value)
        )
        self._warn_base_z_min = float(self.get_parameter("warn_base_z_min_m").value)
        self._warn_base_z_max = float(self.get_parameter("warn_base_z_max_m").value)
        self._warn_base_x_min = float(self.get_parameter("warn_base_x_min_m").value)
        self._warn_base_x_max = float(self.get_parameter("warn_base_x_max_m").value)
        self._center_method = str(self.get_parameter("center_method").value).strip()
        self._top_surface_band_m = float(self.get_parameter("top_surface_band_m").value)
        self._min_top_surface_points = int(
            self.get_parameter("min_top_surface_points").value
        )
        self._use_measured_dimensions_for_policy = bool(
            self.get_parameter("use_measured_dimensions_for_policy").value
        )
        self._table_z_m = float(self.get_parameter("table_z_m").value)

        self._prompt_lock = threading.Lock()
        self._topic_prompt_override: Optional[str] = None

        qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, depth=10)
        self._cb_group = ReentrantCallbackGroup()
        self._service_cb_group = ReentrantCallbackGroup()
        self._frame_process_lock = threading.Lock()

        self._bridge = CvBridge()
        self._tf_buffer = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, self)

        self.create_subscription(
            Image,
            self._rgb_topic,
            self._image_cb,
            qos,
            callback_group=self._cb_group,
        )
        self.create_subscription(
            Image,
            self._depth_topic,
            self._depth_cb,
            qos,
            callback_group=self._cb_group,
        )
        self.create_subscription(
            CameraInfo,
            self._camera_info_topic,
            self._cam_info_cb,
            qos,
            callback_group=self._cb_group,
        )
        tp_topic = str(self.get_parameter("text_prompt_topic").value)
        if self._use_runtime_scene_gt:
            self.create_subscription(
                String,
                GT_OBJECTS_TOPIC,
                self._runtime_scene_gt_cb,
                gt_objects_qos(),
                callback_group=self._cb_group,
            )
            self.get_logger().info(
                "[GT_TOP_FACE] runtime_scene GT topic=%s (solo visualización/métricas)"
                % GT_OBJECTS_TOPIC
            )
        if self._debug_draw_gazebo_gt and not self._use_runtime_scene_gt:
            self.create_subscription(
                TFMessage,
                self._debug_gt_pose_topic,
                self._world_pose_tf_cb,
                10,
                callback_group=self._cb_group,
            )
            self.get_logger().info(
                "[GT_TOP_FACE] overlay Gazebo pose/info topic=%s world=%s"
                % (self._debug_gt_pose_topic, self._debug_gt_world_frame)
            )
        self.create_subscription(
            String,
            tp_topic,
            self._text_prompt_cb,
            10,
            callback_group=self._cb_group,
        )

        self._pub = self.create_publisher(String, self._vision_executor_topic, 10)
        self._legacy_pub = self.create_publisher(
            String, self._legacy_detections_topic, 10
        )
        if self._debug_image_qos_reliable:
            debug_qos = QoSProfile(
                reliability=ReliabilityPolicy.RELIABLE,
                durability=DurabilityPolicy.VOLATILE,
                history=HistoryPolicy.KEEP_LAST,
                depth=10,
            )
        else:
            debug_qos = QoSProfile(
                reliability=ReliabilityPolicy.BEST_EFFORT,
                durability=DurabilityPolicy.VOLATILE,
                history=HistoryPolicy.KEEP_LAST,
                depth=10,
            )
        self.debug_image_pub = self.create_publisher(
            Image, self._debug_image_topic, debug_qos
        )
        self._debug_pub = self.debug_image_pub
        self.debug_status_pub = None
        if self._publish_debug and self._publish_debug_status_image:
            self.debug_status_pub = self.create_publisher(
                Image, self._debug_status_image_topic, debug_qos
            )

        self._camera_info: Optional[CameraInfo] = None
        self._last_image: Optional[Image] = None
        self._last_depth: Optional[Image] = None
        self._last_pair_key: Optional[tuple] = None
        self._first_sync_logged = False
        now_monotonic = time.monotonic()
        self._last_rgb_received_monotonic = now_monotonic
        self._last_depth_received_monotonic: Optional[float] = None
        self._last_camera_info_received_monotonic = now_monotonic
        self._last_depth_warning_monotonic = 0.0
        self._processed_frame_count = 0
        self._processing_paused = False
        self._pause_lock = threading.Lock()
        self._last_valid_payload: Optional[Dict[str, Any]] = None
        self._last_valid_payload_monotonic: float = 0.0
        self._last_executor_payload: Optional[Dict[str, Any]] = None
        self._last_executor_payload_stamp: float = 0.0
        self._last_executor_valid_objects_count: int = 0
        self._last_executor_payload_monotonic: float = 0.0
        self._node_start_monotonic = time.monotonic()
        self._executor_last_publish_monotonic = 0.0
        self._executor_skip_log_serial = 0
        self._warned_first_snapshot_slow = False
        self._rgb_rx_count = 0
        self._depth_rx_count = 0
        self._rate_window_start = time.monotonic()
        self._rgb_hz = 0.0
        self._depth_hz = 0.0
        self._timing_log_serial = 0

        self._vision = self._build_vision_backend()
        self._grasp = self._build_grasp_client()
        self.create_timer(1.0, self._health_check_cb, callback_group=self._cb_group)
        if self._publish_debug and self._debug_heartbeat_enabled:
            self._start_debug_image_heartbeat()
        if self._debug_image_save_to_disk:
            try:
                self._debug_image_save_dir.mkdir(parents=True, exist_ok=True)
            except OSError as exc:
                self.get_logger().warn(
                    "[DEBUG_IMAGE] cannot create save dir %s: %s"
                    % (self._debug_image_save_dir, exc)
                )
        self._pause_processing_service_name = "/perception_node/pause_processing"
        self._resume_processing_service_name = "/perception_node/resume_processing"
        self.create_service(
            Trigger,
            self._pause_processing_service_name,
            self._pause_processing_cb,
            callback_group=self._service_cb_group,
        )
        self.create_service(
            Trigger,
            self._resume_processing_service_name,
            self._resume_processing_cb,
            callback_group=self._service_cb_group,
        )
        self.get_logger().info(
            "[PERCEPTION_SERVICE] created %s"
            % self._pause_processing_service_name
        )
        self.get_logger().info(
            "[PERCEPTION_SERVICE] created %s"
            % self._resume_processing_service_name
        )
        if self._publish_cached_detection:
            cache_hz = max(0.5, float(self.get_parameter("cached_detection_publish_hz").value))
            self.create_timer(
                1.0 / cache_hz,
                self._cached_detection_publish_cb,
                callback_group=self._cb_group,
            )
        self.create_timer(
            1.0 / self._executor_publish_hz,
            self._executor_periodic_publish_cb,
            callback_group=self._cb_group,
        )
        self._log_startup_configuration()

        self.get_logger().info(
            f"perception_node: vision={self._vision.backend_id}, grasp={self._grasp.backend_id}"
        )

    def _build_open3d_config(self) -> Open3DTopSurfaceConfig:
        return Open3DTopSurfaceConfig(
            voxel_size_m=float(self.get_parameter("open3d_voxel_size_m").value),
            normal_nn_max_nn=int(
                self.get_parameter("open3d_normal_nn_max_nn").value
            ),
            normal_radius_m=float(
                self.get_parameter("open3d_normal_radius_m").value
            ),
            max_normal_angle_deg=float(
                self.get_parameter("open3d_max_normal_angle_deg").value
            ),
            ransac_distance_threshold_m=float(
                self.get_parameter("open3d_ransac_distance_m").value
            ),
            ransac_n=int(self.get_parameter("open3d_ransac_n").value),
            ransac_num_iterations=int(
                self.get_parameter("open3d_ransac_iterations").value
            ),
            min_inliers=int(self.get_parameter("open3d_min_inliers").value),
            z_min_depth=float(self.get_parameter("open3d_z_min_m").value),
        )

    def _build_vision_backend(self) -> VisionBackend:
        vb = str(self.get_parameter("vision_backend").value)
        model_path = str(self.get_parameter("model_path").value)
        conf = float(self.get_parameter("confidence_threshold").value)
        gname = str(self.get_parameter("grounded_model_name").value)
        try:
            return create_vision_backend(
                vb,
                model_path=model_path,
                confidence=conf,
                min_mask_pixels=self._min_mask_pixels,
                grounded_model_name=gname,
                grounded_segment_fn=None,
            )
        except Exception as exc:
            self.get_logger().error(f"Vision backend failed to load: {exc}")
            raise

    def _build_grasp_client(self):
        gb = str(self.get_parameter("grasp_backend").value).strip().lower()
        if gb == "foundation_pose":
            return FoundationPoseClient(
                self,
                str(self.get_parameter("foundation_pose_service").value),
                callback_group=self._cb_group,
            )
        if gb == "anygrasp":
            return AnyGraspClient(
                self,
                str(self.get_parameter("anygrasp_service").value),
                callback_group=self._cb_group,
            )
        return NoGraspClient()

    def _text_prompt_cb(self, msg: String) -> None:
        text = msg.data.strip()
        with self._prompt_lock:
            self._topic_prompt_override = text if text else None

    def _effective_text_prompt(self) -> str:
        with self._prompt_lock:
            if self._topic_prompt_override is not None:
                return self._topic_prompt_override
        return self._param_prompt

    def _cam_info_cb(self, msg: CameraInfo) -> None:
        self._camera_info = msg
        self._last_camera_info_received_monotonic = time.monotonic()

    def _image_cb(self, msg: Image) -> None:
        self._last_image = msg
        now = time.monotonic()
        self._last_rgb_received_monotonic = now
        self._rgb_rx_count += 1
        self._update_stream_hz(now)
        if self._publish_debug and self._debug_publish_every_frame:
            self._maybe_publish_live_rgb_status(
                msg,
                banner="waiting rgb-d sync",
            )
        self._try_process()

    def _depth_cb(self, msg: Image) -> None:
        self._last_depth = msg
        now = time.monotonic()
        self._last_depth_received_monotonic = now
        self._depth_rx_count += 1
        self._update_stream_hz(now)
        self._try_process()

    @staticmethod
    def _stamp_key(a: Image, b: Image) -> tuple:
        return (
            a.header.stamp.sec,
            a.header.stamp.nanosec,
            b.header.stamp.sec,
            b.header.stamp.nanosec,
        )

    def _pause_processing_cb(self, _request: Trigger.Request, response: Trigger.Response) -> Trigger.Response:
        with self._pause_lock:
            self._processing_paused = True
        self.get_logger().info("[PERCEPTION_PAUSE] processing paused")
        response.success = True
        response.message = "perception processing paused"
        return response

    def _resume_processing_cb(self, _request: Trigger.Request, response: Trigger.Response) -> Trigger.Response:
        with self._pause_lock:
            self._processing_paused = False
        self.get_logger().info("[PERCEPTION_RESUME] processing resumed")
        response.success = True
        response.message = "perception processing resumed"
        return response

    def _is_processing_paused(self) -> bool:
        with self._pause_lock:
            return self._processing_paused

    def _update_stream_hz(self, now: float) -> None:
        dt = now - self._rate_window_start
        if dt < 1.0:
            return
        self._rgb_hz = float(self._rgb_rx_count) / dt
        self._depth_hz = float(self._depth_rx_count) / dt
        self._rgb_rx_count = 0
        self._depth_rx_count = 0
        self._rate_window_start = now

    def _age_since_last_executor_publish(self) -> float:
        if self._executor_last_publish_monotonic <= 0.0:
            return time.monotonic() - self._node_start_monotonic
        return time.monotonic() - self._executor_last_publish_monotonic

    def _is_frame_processing_busy(self) -> bool:
        return self._frame_process_lock.locked()

    def _executor_cache_age_ms(self) -> float:
        if self._last_executor_payload_monotonic <= 0.0:
            return -1.0
        return (time.monotonic() - self._last_executor_payload_monotonic) * 1000.0

    @staticmethod
    def _executor_object_summary(payload: Optional[Dict[str, Any]]) -> Tuple[List[str], List[str], List[str]]:
        if not isinstance(payload, dict):
            return [], [], []
        labels: List[str] = []
        top_sources: List[str] = []
        grasp_sources: List[str] = []
        for obj in payload.get("objects") or []:
            if not isinstance(obj, dict):
                continue
            labels.append(str(obj.get("label") or "?"))
            top_sources.append(str(obj.get("top_face_source") or ""))
            grasp_sources.append(str(obj.get("grasp_center_source") or ""))
        return labels, top_sources, grasp_sources

    def _log_executor_skip(
        self,
        reason: str,
        *,
        valid_objects_count: Optional[int] = None,
        bypass_busy_check: bool = False,
        publish_source: str = "",
        label: str = "",
        top_face_source: str = "",
        grasp_center_source: str = "",
        yaw_source: str = "",
        closing_yaw_source: str = "",
    ) -> None:
        self._executor_skip_log_serial += 1
        age = self._age_since_last_executor_publish()
        has_payload = self._last_executor_payload is not None
        if valid_objects_count is None:
            if has_payload:
                valid_objects_count = int(self._last_executor_valid_objects_count)
            else:
                valid_objects_count = 0
        if (
            self._executor_last_publish_monotonic <= 0.0
            and age >= self._max_first_snapshot_wait_sec
            and not self._warned_first_snapshot_slow
        ):
            self._warned_first_snapshot_slow = True
            self.get_logger().warn(
                "[VISION_EXECUTOR_SKIP] still no publish after %.1fs reason=%s "
                "valid_objects_count=%d has_payload=%s processing_busy=%s "
                "cache_age_ms=%.1f bypass_busy_check=%s source=%s "
                "(check rgb/depth sync, YOLO load, valid detections)"
                % (
                    age,
                    reason,
                    int(valid_objects_count),
                    str(has_payload).lower(),
                    str(self._is_frame_processing_busy()).lower(),
                    self._executor_cache_age_ms(),
                    str(bypass_busy_check).lower(),
                    publish_source,
                )
            )
            return
        if self._executor_skip_log_serial % 25 != 1:
            return
        self.get_logger().info(
            "[VISION_EXECUTOR_SKIP] reason=%s valid_objects_count=%d has_payload=%s "
            "processing_busy=%s cache_age_ms=%.1f bypass_busy_check=%s source=%s "
            "age_since_last_publish=%.3f label=%s top_face_source=%s "
            "grasp_center_source=%s yaw_source=%s closing_yaw_source=%s"
            % (
                reason,
                int(valid_objects_count),
                str(has_payload).lower(),
                str(self._is_frame_processing_busy()).lower(),
                self._executor_cache_age_ms(),
                str(bypass_busy_check).lower(),
                publish_source,
                age,
                label,
                top_face_source,
                grasp_center_source,
                yaw_source,
                closing_yaw_source,
            )
        )

    def _runtime_scene_entity_short_names(self) -> Set[str]:
        out: Set[str] = set()
        for name in self._gt_spawner_by_entity:
            short = str(name).split("::")[-1].strip()
            if short:
                out.add(short)
        return out

    def _invalidate_executor_cache(self, reason: str) -> None:
        self._last_executor_payload = None
        self._last_executor_payload_stamp = 0.0
        self._last_executor_valid_objects_count = 0
        self._last_executor_payload_monotonic = 0.0
        self._last_valid_payload = None
        self._last_valid_payload_monotonic = 0.0
        self.get_logger().info(
            "[VISION_EXECUTOR_CACHE_INVALIDATE] reason=%s" % str(reason)
        )

    @staticmethod
    def _normalize_operational_source_contract(obj: Dict[str, Any]) -> None:
        """Fuerza contrato explícito perception→controller por label/shape."""
        label = str(obj.get("label") or obj.get("id") or "").strip()
        lb = normalize_label(label)
        top = str(obj.get("top_face_source") or "").strip()
        gcs_old = str(obj.get("grasp_center_source") or "").strip()

        if lb in CYLINDER_KNOWN_LABELS or top == "runtime_gt_known_cylinder":
            obj["top_face_source"] = "runtime_gt_known_cylinder"
            obj["grasp_center_source"] = "runtime_gt_cylinder_center"
            obj["yaw_source"] = "runtime_gt_spawn_yaw"
            cys = str(obj.get("closing_yaw_source") or "").strip()
            if cys not in ("runtime_gt_yaw_free", "runtime_gt_cylinder_axis"):
                obj["closing_yaw_source"] = "runtime_gt_yaw_free"
        elif (
            lb in KNOWN_BOX_LABELS
            or top == "runtime_gt_known_box"
            or is_known_spawn_geometry_box_label(lb)
        ):
            obj["top_face_source"] = "runtime_gt_known_box"
            if gcs_old == "runtime_gt_object_center":
                obj["grasp_center_source"] = "runtime_gt_box_center"
            else:
                obj["grasp_center_source"] = "runtime_gt_box_center"
            obj["yaw_source"] = "runtime_gt_spawn_yaw"
            obj["closing_yaw_source"] = str(
                obj.get("closing_yaw_source") or "runtime_gt_short_axis"
            )
        elif lb in TALL_KNOWN_OBJECT_LABELS or top in (
            "runtime_gt_known_object",
            "runtime_gt_tall_object",
        ):
            if top not in ("runtime_gt_known_object", "runtime_gt_tall_object"):
                obj["top_face_source"] = "runtime_gt_tall_object"
            else:
                obj["top_face_source"] = top
            if not is_runtime_gt_tall_cap_center_source(
                obj.get("grasp_center_source")
            ) and str(obj.get("grasp_center_source") or "").strip() not in (
                "runtime_gt_object_center",
                "runtime_gt_tall_object_center",
            ):
                obj["grasp_center_source"] = "runtime_gt_tall_object_center"
            obj["yaw_source"] = "runtime_gt_spawn_yaw"
            cys = str(obj.get("closing_yaw_source") or "").strip()
            if _mustard_gap_closing_source_ok(cys):
                pass
            elif cys not in (
                "runtime_gt_short_axis",
                "runtime_gt_known_object_short_axis",
            ):
                obj["closing_yaw_source"] = "runtime_gt_short_axis"

        if str(obj.get("top_face_source", "")).startswith("runtime_gt_"):
            obj["operational_source_fallback"] = False

    def _log_operational_source_contract(self, obj: Dict[str, Any]) -> None:
        self._normalize_operational_source_contract(obj)
        shape = str(obj.get("shape") or obj.get("collision_shape") or "")
        self.get_logger().info(
            "[OPERATIONAL_SOURCE_CONTRACT] label=%s shape=%s top_face_source=%s "
            "grasp_center_source=%s yaw_source=%s closing_yaw_source=%s "
            "operational_source_fallback=%s"
            % (
                str(obj.get("label", "")),
                shape or "n/a",
                str(obj.get("top_face_source", "")),
                str(obj.get("grasp_center_source", "")),
                str(obj.get("yaw_source", "")),
                str(obj.get("closing_yaw_source", "")),
                str(bool(obj.get("operational_source_fallback", False))).lower(),
            )
        )

    def _annotate_object_runtime_scene_fields(self, obj: Dict[str, Any]) -> None:
        ent = str(obj.get("entity_name") or obj.get("gt_entity_name") or "").strip()
        if ent:
            obj["entity_name"] = ent.split("::")[-1]
        obj["runtime_scene_stamp"] = float(self._runtime_scene_stamp_sec)
        obj["runtime_scene_source"] = str(self._runtime_scene_source)
        obj["top_face_source"] = str(obj.get("top_face_source") or "")
        obj["grasp_center_source"] = str(obj.get("grasp_center_source") or "")

    def _log_vision_grasp_policy_export(self, obj: Dict[str, Any]) -> None:
        """Trazabilidad de política publicada en /vision_to_executor."""
        label = str(obj.get("label", "")).strip()
        if not label:
            return
        self.get_logger().info(
                "[VISION_GRASP_POLICY_EXPORT]\n"
                "label=%s\n"
                "strategy=%s\n"
                "required_grasp_width_m=%s\n"
                "effective_required_grasp_width_m=%s\n"
                "db_required_width_m=%s\n"
                "required_width_source=%s\n"
                "measured_required_width_m=%s\n"
                "recommended_grasp_depth_from_top_m=%s\n"
                "recommended_open_joint_m=%s\n"
                "recommended_close_joint_m=%s\n"
                "edge_grasp_requested=%s\n"
                "edge_offset_m=%s\n"
                "dims_lwh=%s"
                % (
                    label,
                    str(obj.get("grasp_strategy", "")),
                    obj.get("required_grasp_width_m"),
                    obj.get("effective_required_grasp_width_m"),
                    obj.get("db_required_width_m"),
                    str(obj.get("required_width_source", "")),
                    obj.get("measured_required_width_m"),
                    obj.get("recommended_grasp_depth_from_top_m"),
                    obj.get("recommended_open_joint_m"),
                    obj.get("recommended_close_joint_m"),
                    str(obj.get("edge_grasp_requested", "")).lower(),
                    obj.get("edge_offset_m"),
                    obj.get("dims_lwh"),
                )
        )

    def _enrich_executor_payload_runtime_metadata(self, payload: Dict[str, Any]) -> None:
        payload["runtime_scene_stamp_sec"] = float(self._runtime_scene_stamp_sec)
        payload["runtime_scene_source"] = str(self._runtime_scene_source)
        for obj in payload.get("objects") or []:
            if isinstance(obj, dict):
                self._annotate_object_runtime_scene_fields(obj)
        tc = payload.get("target_candidate")
        if isinstance(tc, dict):
            self._annotate_object_runtime_scene_fields(tc)

    def _entity_in_runtime_scene(self, entity_name: str) -> bool:
        ent = str(entity_name or "").strip()
        if not ent:
            return False
        short = ent.split("::")[-1]
        scene = self._runtime_scene_entity_short_names()
        if short in scene or ent in scene:
            return True
        for name in scene:
            if short in name or name in short:
                return True
        return False

    def _validate_executor_publish(
        self,
        payload: Dict[str, Any],
        *,
        publish_source: str,
    ) -> Tuple[bool, str]:
        _n_obj, n_valid = self._count_valid_objects(payload)
        if n_valid <= 0:
            return False, "no_valid_objects"
        if self._use_runtime_scene_gt and not self._gt_spawner_by_entity:
            return False, "runtime_scene_empty"
        operational = [
            o
            for o in (payload.get("objects") or [])
            if isinstance(o, dict) and self._is_operational_detection(o)
        ]
        for obj in operational:
            ent = str(obj.get("entity_name") or obj.get("gt_entity_name") or "").strip()
            if self._use_runtime_scene_gt and not self._entity_in_runtime_scene(ent):
                return False, "entity_not_in_runtime_scene"
        if publish_source in ("timer_cache", "periodic_cache"):
            cache_age = time.monotonic() - self._last_executor_payload_monotonic
            if cache_age > self._max_executor_payload_age_sec:
                return False, "stale_payload"
            if self._use_runtime_scene_gt:
                payload_stamp = _safe_float(payload.get("runtime_scene_stamp_sec"), 0.0)
                if (
                    payload_stamp is not None
                    and self._runtime_scene_stamp_sec > 0.0
                    and abs(float(payload_stamp) - self._runtime_scene_stamp_sec) > 1e-4
                ):
                    return False, "stale_payload"
                for obj in operational:
                    obj_stamp = _safe_float(obj.get("runtime_scene_stamp"), 0.0)
                    if (
                        obj_stamp is not None
                        and self._runtime_scene_stamp_sec > 0.0
                        and abs(float(obj_stamp) - self._runtime_scene_stamp_sec) > 1e-4
                    ):
                        return False, "stale_payload"
        return True, ""

    def _count_valid_objects(self, payload: Dict[str, Any]) -> Tuple[int, int]:
        objects = payload.get("objects")
        if not isinstance(objects, list):
            return 0, 0
        valid = [o for o in objects if isinstance(o, dict) and self._is_operational_detection(o)]
        return len(objects), len(valid)

    def _first_demo_source_mismatch(
        self, payload: Dict[str, Any]
    ) -> Optional[Dict[str, str]]:
        objects = payload.get("objects")
        if not isinstance(objects, list):
            return None
        for obj in objects:
            if not isinstance(obj, dict):
                continue
            label = str(obj.get("label") or "").strip().lower()
            if label not in DEMO_KNOWN_OBJECTS:
                continue
            tfs = str(obj.get("top_face_source") or "").strip()
            gcs = str(obj.get("grasp_center_source") or "").strip()
            ys = str(obj.get("yaw_source") or "").strip()
            cys = str(obj.get("closing_yaw_source") or "").strip()
            top_ok = tfs in (
                "runtime_gt_known_box",
                "runtime_gt_known_object",
                "runtime_gt_tall_object",
                "runtime_gt_known_cylinder",
            )
            center_ok = gcs in (
                "runtime_gt_box_center",
                "runtime_gt_object_center",
                "runtime_gt_tall_object_center",
                *RUNTIME_GT_TALL_CAP_CENTER_SOURCES,
                "runtime_gt_cylinder_center",
            )
            yaw_ok = ys == "runtime_gt_spawn_yaw"
            close_ok = cys in (
                "runtime_gt_short_axis",
                "runtime_gt_known_object_short_axis",
                "runtime_gt_yaw_free",
                "runtime_gt_cylinder_axis",
            ) or _mustard_gap_closing_source_ok(cys)
            if not (top_ok and center_ok and yaw_ok and close_ok):
                return {
                    "label": label,
                    "top_face_source": tfs,
                    "grasp_center_source": gcs,
                    "yaw_source": ys,
                    "closing_yaw_source": cys,
                }
        return None

    def _cache_executor_payload(self, payload: Dict[str, Any]) -> int:
        """Guarda payload operativo para publicación directa y timer."""
        n_obj, n_valid = self._count_valid_objects(payload)
        if n_valid == 0:
            return 0
        if self._use_runtime_scene_gt and not self._gt_spawner_by_entity:
            self._log_executor_skip(
                "runtime_scene_empty",
                valid_objects_count=0,
                publish_source="cache_store",
            )
            return 0
        operational = [
            o
            for o in (payload.get("objects") or [])
            if isinstance(o, dict) and self._is_operational_detection(o)
        ]
        for obj in operational:
            ent = str(obj.get("entity_name") or obj.get("gt_entity_name") or "").strip()
            if self._use_runtime_scene_gt and not self._entity_in_runtime_scene(ent):
                self._log_executor_skip(
                    "entity_not_in_runtime_scene",
                    valid_objects_count=n_valid,
                    publish_source="cache_store",
                )
                return 0
        cached = json.loads(json.dumps(payload))
        for obj in operational:
            self._normalize_operational_source_contract(obj)
        cached["objects"] = operational
        self._enrich_executor_payload_runtime_metadata(cached)
        telem = dict(cached.get("telemetry") or {})
        telem["cached"] = False
        telem["cache_age_s"] = 0.0
        cached["telemetry"] = telem
        stamp = _safe_float(payload.get("stamp_sec"), time.monotonic()) or time.monotonic()
        self._last_executor_payload = cached
        self._last_executor_payload_stamp = float(stamp)
        self._last_executor_valid_objects_count = int(n_valid)
        self._last_executor_payload_monotonic = time.monotonic()
        self._last_valid_payload = cached
        self._last_valid_payload_monotonic = self._last_executor_payload_monotonic
        labels, top_sources, grasp_sources = self._executor_object_summary(cached)
        self.get_logger().info(
            "[VISION_EXECUTOR_PAYLOAD_READY] valid_objects_count=%d objects=%d "
            "labels=%s top_face_sources=%s grasp_center_sources=%s cache_updated=true"
            % (
                int(n_valid),
                int(n_obj),
                ",".join(labels) if labels else "none",
                ",".join(top_sources) if top_sources else "none",
                ",".join(grasp_sources) if grasp_sources else "none",
            )
        )
        if self._debug_image_save_to_disk and self._last_rich_debug_bgr is not None:
            for obj in operational:
                label = str(obj.get("label") or "object").strip().replace(" ", "_")
                self._save_debug_bgr_to_disk(
                    self._last_rich_debug_bgr,
                    prefix=f"stable_snapshot_{label}",
                )
        return int(n_valid)

    def _maybe_log_demo_scene_vision_labels(
        self,
        payload: Dict[str, Any],
        operational: List[Dict[str, Any]],
    ) -> None:
        scene_objects = payload.get("scene_objects")
        if not isinstance(scene_objects, list) or not is_consolidated_demo_scene_objects(
            scene_objects
        ):
            return
        preset = self._scene_preset
        if not preset or not is_demo_scene_preset(preset):
            preset = preset or "demo_scene_consolidated"
        runtime_labels = runtime_labels_from_scene_objects(scene_objects)
        vision_labels = [
            str(o.get("label", "")).strip().lower()
            for o in operational
            if isinstance(o, dict) and str(o.get("label", "")).strip()
        ]
        log_demo_scene_vision_labels(
            self.get_logger(),
            scene_preset=preset,
            runtime_labels=runtime_labels,
            vision_labels=vision_labels,
        )

    def _publish_vision_to_executor_payload(
        self,
        payload: Dict[str, Any],
        reason: str,
        *,
        publish_source: str = "direct_frame",
        bypass_busy_check: bool = False,
    ) -> bool:
        has_payload = isinstance(payload, dict) and bool(payload.get("objects"))
        n_obj, n_valid = self._count_valid_objects(payload)
        busy = self._is_frame_processing_busy()
        self.get_logger().info(
            "[VISION_EXECUTOR_PUBLISH_ATTEMPT] source=%s valid_objects_count=%d "
            "processing_busy=%s bypass_busy_check=%s has_payload=%s reason=%s"
            % (
                publish_source,
                int(n_valid),
                str(busy).lower(),
                str(bypass_busy_check).lower(),
                str(has_payload).lower(),
                reason,
            )
        )
        if self._is_processing_paused():
            self._log_executor_skip(
                "paused",
                valid_objects_count=n_valid,
                bypass_busy_check=bypass_busy_check,
                publish_source=publish_source,
            )
            return False
        if not bypass_busy_check and busy:
            self._log_executor_skip(
                "processing_busy",
                valid_objects_count=n_valid,
                bypass_busy_check=False,
                publish_source=publish_source,
            )
            return False
        if n_valid == 0:
            mismatch = self._first_demo_source_mismatch(payload)
            if mismatch is not None:
                self._log_executor_skip(
                    "source_not_whitelisted",
                    valid_objects_count=0,
                    bypass_busy_check=bypass_busy_check,
                    publish_source=publish_source,
                    label=mismatch.get("label", ""),
                    top_face_source=mismatch.get("top_face_source", ""),
                    grasp_center_source=mismatch.get("grasp_center_source", ""),
                    yaw_source=mismatch.get("yaw_source", ""),
                    closing_yaw_source=mismatch.get("closing_yaw_source", ""),
                )
                return False
            self._log_executor_skip(
                "no_valid_objects",
                valid_objects_count=0,
                bypass_busy_check=bypass_busy_check,
                publish_source=publish_source,
            )
            return False
        publish_payload = json.loads(json.dumps(payload))
        operational = [
            o
            for o in (publish_payload.get("objects") or [])
            if isinstance(o, dict) and self._is_operational_detection(o)
        ]
        publish_payload["objects"] = operational
        self._enrich_executor_payload_runtime_metadata(publish_payload)
        ok, skip_reason = self._validate_executor_publish(
            publish_payload, publish_source=publish_source
        )
        if not ok:
            self._log_executor_skip(
                skip_reason,
                valid_objects_count=n_valid,
                bypass_busy_check=bypass_busy_check,
                publish_source=publish_source,
            )
            return False
        for obj in operational:
            self._log_operational_source_contract(obj)
            self._log_vision_grasp_policy_export(obj)
        self._maybe_log_demo_scene_vision_labels(publish_payload, operational)
        tc_pub = publish_payload.get("target_candidate")
        if isinstance(tc_pub, dict):
            self._log_vision_grasp_policy_export(tc_pub)
        now = time.monotonic()
        elapsed = self._age_since_last_executor_publish()
        stamp = _safe_float(publish_payload.get("stamp_sec"), 0.0) or 0.0
        label = "none"
        top_face_source = ""
        grasp_center_source = ""
        yaw_source = ""
        for obj in operational:
            label = str(obj.get("label") or "object")
            top_face_source = str(obj.get("top_face_source") or "")
            grasp_center_source = str(obj.get("grasp_center_source") or "")
            yaw_source = str(obj.get("yaw_source") or "")
            break
        msg = String()
        body = json.dumps(publish_payload)
        msg.data = body
        t_pub0 = time.perf_counter()
        self._pub.publish(msg)
        pub_ms = (time.perf_counter() - t_pub0) * 1000.0
        self._executor_last_publish_monotonic = now
        self.get_logger().info(
            "[VISION_EXECUTOR_PUBLISH] source=%s bytes=%d stamp=%.3f objects=%d "
            "valid_objects=%d label=%s top_face_source=%s grasp_center_source=%s "
            "yaw_source=%s elapsed_since_last_publish=%.3f reason=%s publish_ms=%.2f"
            % (
                publish_source,
                len(body.encode("utf-8")),
                float(stamp),
                n_obj,
                n_valid,
                label,
                top_face_source,
                grasp_center_source,
                yaw_source,
                float(elapsed),
                reason,
                pub_ms,
            )
        )
        return True

    def _executor_periodic_publish_cb(self) -> None:
        if self._is_processing_paused():
            self._log_executor_skip(
                "paused",
                publish_source="timer_cache",
            )
            return
        if self._last_executor_payload is None:
            self._log_executor_skip(
                "no_cached_payload",
                publish_source="timer_cache",
            )
            return
        age = time.monotonic() - self._last_executor_payload_monotonic
        if age > self._cache_ttl_s:
            self._log_executor_skip(
                "cache_expired",
                valid_objects_count=self._last_executor_valid_objects_count,
                publish_source="timer_cache",
            )
            return
        payload = json.loads(json.dumps(self._last_executor_payload))
        telem = dict(payload.get("telemetry") or {})
        telem["cached"] = True
        telem["cache_age_s"] = float(age)
        payload["telemetry"] = telem
        self._publish_vision_to_executor_payload(
            payload,
            reason="periodic_cache",
            publish_source="timer_cache",
            bypass_busy_check=True,
        )

    def _maybe_log_perception_timing(self, prof: Dict[str, float]) -> None:
        self._timing_log_serial += 1
        if self._timing_log_serial % self._timing_log_every_n != 0:
            return
        model_fit = prof.get("model_cuboid", 0.0) + prof.get("hybrid", 0.0) + prof.get(
            "global_search", 0.0
        )
        self.get_logger().info(
            "[PERCEPTION_TIMING] rgb_hz=%.1f depth_hz=%.1f sync_wait_ms=%.1f yolo_ms=%.1f "
            "pointcloud_ms=%.1f top_face_ms=%.1f model_fit_ms=%.1f debug_overlay_ms=%.1f "
            "executor_publish_ms=%.1f total_ms=%.1f"
            % (
                self._rgb_hz,
                self._depth_hz,
                prof.get("sync_wait", 0.0),
                prof.get("yolo", 0.0),
                prof.get("rgbd", 0.0),
                prof.get("top_face", 0.0),
                model_fit,
                prof.get("overlay", 0.0),
                prof.get("publish", 0.0),
                prof.get("total", 0.0),
            )
        )

    def _detection_passes_label_filter(self, label: str) -> bool:
        if not self._target_label_filter:
            return True
        return str(label or "").strip().lower() == self._target_label_filter

    def _reject_detection_z_out_of_range(
        self,
        label: str,
        centroid_base: Optional[List[float]],
        top_z_m: float,
    ) -> bool:
        if not self._enable_robot_occlusion_filter or not self._reject_out_of_table_before_fit:
            return False
        reasons: List[str] = []
        if top_z_m > self._reject_top_z_above:
            reasons.append(f"top_z={top_z_m:.3f}>{self._reject_top_z_above:.3f}")
        if top_z_m < self._reject_top_z_below:
            reasons.append(f"top_z={top_z_m:.3f}<{self._reject_top_z_below:.3f}")
        if centroid_base is not None and len(centroid_base) >= 3:
            cz = float(centroid_base[2])
            if cz > self._reject_centroid_z_above:
                reasons.append(
                    f"centroid_z={cz:.3f}>{self._reject_centroid_z_above:.3f}"
                )
        if not reasons:
            return False
        self.get_logger().info(
            "[PERCEPTION_FILTER] rejected label=%s reason=z_out_of_table_range %s"
            % (label, ", ".join(reasons))
        )
        return True

    def _is_operational_detection(self, obj: Dict[str, Any]) -> bool:
        if self._enable_visual_pose_gate and obj.get("visual_pose_gate_passed") is False:
            return False
        label = str(obj.get("label") or "").strip().lower()
        gcs = str(obj.get("grasp_center_source") or "").strip()
        tfs = str(obj.get("top_face_source") or "").strip()
        ys = str(obj.get("yaw_source") or "").strip()
        cys = str(obj.get("closing_yaw_source") or "").strip()

        explicit_top_ok = tfs in (
            "runtime_gt_known_box",
            "runtime_gt_known_object",
            "runtime_gt_tall_object",
            "runtime_gt_known_cylinder",
        )
        explicit_center_ok = gcs in (
            "runtime_gt_box_center",
            "runtime_gt_object_center",
            "runtime_gt_tall_object_center",
            *RUNTIME_GT_TALL_CAP_CENTER_SOURCES,
            "runtime_gt_cylinder_center",
        )
        explicit_yaw_ok = ys == "runtime_gt_spawn_yaw"
        explicit_close_ok = cys in (
            "runtime_gt_short_axis",
            "runtime_gt_known_object_short_axis",
            "runtime_gt_yaw_free",
            "runtime_gt_cylinder_axis",
        ) or _mustard_gap_closing_source_ok(cys)
        if explicit_top_ok and explicit_center_ok and explicit_yaw_ok and explicit_close_ok:
            self.get_logger().info(
                "[OPERATIONAL_SOURCE_ACCEPT] label=%s top_face_source=%s grasp_center_source=%s "
                "yaw_source=%s closing_yaw_source=%s operational_source_fallback=false"
                % (label, tfs, gcs, ys, cys)
            )
            return True
        if label in DEMO_KNOWN_OBJECTS:
            self.get_logger().info(
                "[OPERATIONAL_SOURCE_REJECT] label=%s reason=source_not_whitelisted "
                "top_face_source=%s grasp_center_source=%s yaw_source=%s closing_yaw_source=%s"
                % (label, tfs, gcs, ys, cys)
            )
            return False

        if tfs in (
            "runtime_gt_known_box",
            "runtime_gt_known_object",
            "runtime_gt_tall_object",
        ):
            return bool(obj.get("pose_fit_success")) or gcs in (
                "runtime_gt_box_center",
                "runtime_gt_object_center",
                "runtime_gt_tall_object_center",
                *RUNTIME_GT_TALL_CAP_CENTER_SOURCES,
            )
        if bool(obj.get("hybrid_fit_success")) and ys == "hybrid_top_face_known_dims":
            return True
        if bool(obj.get("pose_fit_success")) and ys in (
            "known_rectangle_fit",
            "hybrid_top_face_known_dims",
        ):
            return True
        if gcs == "model_box_center" and bool(obj.get("model_top_face_success")):
            return True
        # Fallback defensivo para debug (no camino principal de demo).
        ent = str(obj.get("entity_name") or obj.get("gt_entity_name") or "").strip()
        in_runtime_scene = self._entity_in_runtime_scene(ent)
        grasp_center = obj.get("grasp_center_base") or obj.get("position")
        has_center = isinstance(grasp_center, (list, tuple)) and len(grasp_center) >= 3
        top_center = obj.get("top_face_center_base")
        top_corners = obj.get("top_face_corners_base")
        has_top_face = (
            isinstance(top_center, (list, tuple))
            and len(top_center) >= 3
            or isinstance(top_corners, (list, tuple))
            and len(top_corners) >= 4
        )
        if in_runtime_scene and has_center and has_top_face:
            self.get_logger().info(
                "[OPERATIONAL_FALLBACK_CANDIDATE] label=%s top_face_source=%s grasp_center_source=%s "
                "yaw_source=%s closing_yaw_source=%s in_runtime_scene=%s"
                % (label, tfs, gcs, ys, cys, str(in_runtime_scene).lower())
            )
            if label in DEMO_KNOWN_OBJECTS and not self._allow_operational_source_fallback_for_debug:
                self.get_logger().info(
                    "[OPERATIONAL_SOURCE_REJECT] label=%s reason=source_not_whitelisted "
                    "top_face_source=%s grasp_center_source=%s yaw_source=%s closing_yaw_source=%s"
                    % (label, tfs, gcs, ys, cys)
                )
                return False
            if not self._allow_operational_source_fallback_for_debug:
                self.get_logger().info(
                    "[OPERATIONAL_SOURCE_REJECT] label=%s reason=fallback_disabled "
                    "top_face_source=%s grasp_center_source=%s yaw_source=%s closing_yaw_source=%s"
                    % (label, tfs, gcs, ys, cys)
                )
                return False
            self.get_logger().info(
                "[OPERATIONAL_SOURCE_ACCEPT] label=%s top_face_source=%s grasp_center_source=%s "
                "yaw_source=%s closing_yaw_source=%s operational_source_fallback=true"
                % (label, tfs, gcs, ys, cys)
            )
            return True
        return False

    def _update_valid_detection_cache(self, payload: Dict[str, Any]) -> None:
        self._cache_executor_payload(payload)

    def _cached_detection_publish_cb(self) -> None:
        if self._is_processing_paused():
            return
        if not self._publish_cached_detection or self._last_executor_payload is None:
            return
        age = time.monotonic() - self._last_executor_payload_monotonic
        if age > self._cache_ttl_s:
            return
        payload = json.loads(json.dumps(self._last_executor_payload))
        telem = dict(payload.get("telemetry") or {})
        telem["cached"] = True
        telem["cache_age_s"] = float(age)
        payload["telemetry"] = telem
        self._publish_vision_to_executor_payload(
            payload,
            reason="periodic_cache",
            publish_source="timer_cache",
            bypass_busy_check=True,
        )

    def _try_process(self) -> None:
        if self._is_processing_paused():
            self._log_executor_skip("paused")
            if self._publish_debug:
                self._publish_debug_status_image_bgr(
                    self._make_fallback_debug_bgr("processing paused by executor"),
                    self._camera_optical_frame,
                    banner="processing paused by executor",
                )
            return
        if self._last_image is None:
            self._log_executor_skip("no_rgbd_sync")
            if self._publish_debug:
                self._publish_debug_status_image_bgr(
                    self._make_fallback_debug_bgr("debug fallback: no rgb frame"),
                    self._camera_optical_frame,
                    banner="debug fallback: no rgb frame",
                )
            return
        if self._camera_info is None or self._last_depth is None:
            reason = (
                "waiting depth"
                if self._camera_info is not None
                else "waiting camera_info"
            )
            self._log_executor_skip("no_rgbd_sync")
            self._publish_debug_while_waiting_sync(reason)
            return
        im = self._last_image
        dp = self._last_depth
        ta = float(im.header.stamp.sec) + float(im.header.stamp.nanosec) * 1e-9
        tb = float(dp.header.stamp.sec) + float(dp.header.stamp.nanosec) * 1e-9
        if abs(ta - tb) > self._max_slop:
            self._log_executor_skip("no_rgbd_sync")
            return
        key = self._stamp_key(im, dp)
        if key == self._last_pair_key:
            return
        if not self._frame_process_lock.acquire(blocking=False):
            if self._processed_frame_count % 25 == 0:
                self.get_logger().debug(
                    "[PERCEPTION_FRAME_SKIP] reason=processing_busy "
                    "(executor publish uses in-frame direct path when valid)"
                )
            return
        self._last_pair_key = key
        if not self._first_sync_logged:
            self.get_logger().info("Primera sincronizacion RGB-D recibida.")
            self._first_sync_logged = True
        try:
            self._process_frame(im, dp)
        except Exception as exc:
            self.get_logger().error(
                "[PERCEPTION] frame processing error: %s" % exc
            )
            self._log_executor_skip("processing_error")
        finally:
            self._frame_process_lock.release()

    def _log_startup_configuration(self) -> None:
        self.get_logger().info(
            "Perception topics/config: "
            f"rgb={self._rgb_topic}, depth={self._depth_topic}, "
            f"camera_info={self._camera_info_topic}, debug={self._debug_image_topic}, "
            f"detections={self._legacy_detections_topic}, out={self._vision_executor_topic}, "
            f"model_path={self._model_path}, vision_backend={self._vision_backend}, "
            f"confidence_threshold={self._confidence_threshold:.2f}, max_sync_slop_sec={self._max_slop:.2f}"
        )
        self.get_logger().info(
            "[DEBUG_IMAGE_CONFIG] publish_debug_image=%s heartbeat=%s heartbeat_hz=%.3f "
            "publish_every_frame=%s policy=%s status_topic=%s use_sim_time=%s"
            % (
                str(self._publish_debug).lower(),
                str(self._debug_heartbeat_enabled).lower(),
                self._debug_heartbeat_hz,
                str(self._debug_publish_every_frame).lower(),
                self._debug_publish_policy,
                self._debug_status_image_topic,
                str(self._use_sim_time_flag).lower(),
            )
        )
        self.get_logger().info(
            "[DEBUG_IMAGE_CONFIG] overlay_simplified_for_demo=%s overlay_mode=%s "
            "draw_gt_overlay=%s"
            % (
                str(self._debug_overlay_simplified_for_demo).lower(),
                self._debug_overlay_mode,
                str(self._debug_draw_gazebo_gt).lower(),
            )
        )

    def _debug_heartbeat_period_sec(self) -> float:
        return 1.0 / max(0.1, float(self._debug_heartbeat_hz))

    def _start_debug_image_heartbeat(self) -> None:
        period = self._debug_heartbeat_period_sec()
        self._debug_wall_clock = Clock(clock_type=ClockType.STEADY_TIME)
        try:
            self._debug_image_heartbeat_timer = self.create_timer(
                period,
                self._publish_debug_image_heartbeat,
                clock=self._debug_wall_clock,
                callback_group=self._cb_group,
            )
            self.get_logger().info(
                "[DEBUG_IMAGE] heartbeat timer created with STEADY_TIME period=%.3fs"
                % period
            )
            self._publish_debug_image_heartbeat()
        except (TypeError, ValueError) as exc:
            self.get_logger().warn(
                "[DEBUG_IMAGE] STEADY_TIME create_timer failed (%s); using monotonic thread"
                % exc
            )
            self._start_debug_image_heartbeat_thread(period)

    def _start_debug_image_heartbeat_thread(self, period: float) -> None:
        self._debug_heartbeat_stop = threading.Event()

        def _loop() -> None:
            while rclpy.ok() and not self._debug_heartbeat_stop.is_set():
                try:
                    self._publish_debug_image_heartbeat()
                except Exception as loop_exc:
                    self.get_logger().warn(
                        "[DEBUG_IMAGE] heartbeat thread error: %s" % loop_exc
                    )
                if self._debug_heartbeat_stop.wait(period):
                    break

        self._debug_heartbeat_thread = threading.Thread(
            target=_loop, name="debug_image_heartbeat", daemon=True
        )
        self._debug_heartbeat_thread.start()
        self.get_logger().info(
            "[DEBUG_IMAGE] heartbeat thread created with monotonic wall time period=%.3fs"
            % period
        )
        self._publish_debug_image_heartbeat()

    def destroy_node(self) -> None:
        if self._debug_heartbeat_stop is not None:
            self._debug_heartbeat_stop.set()
        if self._debug_image_heartbeat_timer is not None:
            try:
                self.destroy_timer(self._debug_image_heartbeat_timer)
            except Exception:
                pass
        super().destroy_node()

    def _health_check_cb(self) -> None:
        now = time.monotonic()
        warn_interval_sec = 5.0
        no_depth_timeout_sec = 4.0
        if (
            self._last_depth_received_monotonic is None
            and now - self._last_depth_warning_monotonic >= warn_interval_sec
        ):
            self.get_logger().warn(
                f"No se recibe {self._depth_topic}; percepcion RGB-D no puede ejecutarse."
            )
            self._last_depth_warning_monotonic = now
            return

        if (
            self._last_depth_received_monotonic is not None
            and (now - self._last_depth_received_monotonic) > no_depth_timeout_sec
            and now - self._last_depth_warning_monotonic >= warn_interval_sec
        ):
            self.get_logger().warn(
                f"{self._depth_topic} se ha detenido; percepcion RGB-D no puede ejecutarse."
            )
            self._last_depth_warning_monotonic = now
            return

        if (
            self._last_depth_received_monotonic is None
            and self._last_image is not None
            and self._camera_info is not None
            and now - self._last_depth_warning_monotonic >= warn_interval_sec
        ):
            self.get_logger().warn(
                f"Se reciben {self._rgb_topic} y {self._camera_info_topic}, pero no {self._depth_topic}."
            )
            self._last_depth_warning_monotonic = now

    def _depth_to_meters(self, depth: np.ndarray, encoding: str) -> np.ndarray:
        if encoding == "16UC1":
            return depth.astype(np.float32) / 1000.0
        return depth.astype(np.float32)

    def _lookup_transform(self, source_frame: str):
        try:
            return self._tf_buffer.lookup_transform(
                self._target_frame,
                source_frame,
                rclpy.time.Time(),
                timeout=Duration(seconds=0.5),
            )
        except (
            tf2_ros.LookupException,
            tf2_ros.ConnectivityException,
            tf2_ros.ExtrapolationException,
        ) as exc:
            self.get_logger().warn(
                f"TF {source_frame}->{self._target_frame}: {exc}"
            )
            return None

    def _lookup_world_to_base_matrix(self) -> Optional[np.ndarray]:
        tf_msg = self._lookup_transform(self._debug_gt_world_frame)
        if tf_msg is None:
            return None
        return self._build_transform_matrix(tf_msg)

    def _world_pose_tf_cb(self, msg: TFMessage) -> None:
        if not self._debug_draw_gazebo_gt:
            return
        update_world_pose_cache(
            msg,
            self._gt_world_poses,
            world_frame=self._debug_gt_world_frame,
        )

    def _runtime_scene_gt_cb(self, msg: String) -> None:
        payload = parse_gt_payload(msg)
        objs = payload.get("objects")
        if not isinstance(objs, list):
            objs = []
        old_entities = sorted(self._runtime_scene_entity_short_names())
        prev_stamp = float(self._runtime_scene_stamp_sec)
        stamp = _safe_float(payload.get("stamp_sec"), 0.0)
        new_stamp = float(stamp) if stamp is not None else prev_stamp

        new_map: Dict[str, Dict[str, Any]] = {}
        for obj in objs:
            if not isinstance(obj, dict):
                continue
            name = str(obj.get("entity_name", "")).strip()
            if name:
                new_map[name] = obj

        new_entities = sorted(
            {str(k).split("::")[-1].strip() for k in new_map if str(k).strip()}
        )

        if not new_map:
            self.get_logger().info("[RUNTIME_SCENE_EMPTY]")
            if old_entities or self._last_executor_payload is not None:
                self._invalidate_executor_cache("runtime_scene_empty")
            self._gt_spawner_by_entity = {}
            return

        stamp_changed = (
            prev_stamp > 0.0
            and new_stamp > 0.0
            and abs(new_stamp - prev_stamp) > 1e-4
        )
        if set(old_entities) != set(new_entities):
            self.get_logger().info(
                "[RUNTIME_SCENE_ENTITY_CHANGED] old_entities=%s new_entities=%s"
                % (old_entities, new_entities)
            )
            self._invalidate_executor_cache("entity_changed")
        elif stamp_changed and new_entities:
            self.get_logger().info(
                "[RUNTIME_SCENE_ENTITY_CHANGED] old_entities=%s new_entities=%s "
                "runtime_scene_stamp=%.3f->%.3f (same names, new GT)"
                % (old_entities, new_entities, prev_stamp, new_stamp)
            )
            self._invalidate_executor_cache("entity_changed")

        self._runtime_scene_stamp_sec = new_stamp
        self._gt_spawner_by_entity = new_map
        twb = self._lookup_world_to_base_matrix()
        if twb is not None:
            for obj in self._gt_spawner_by_entity.values():
                enrich_gt_object_entry_base(obj, twb)

    @staticmethod
    def _gt_entry_to_scene_object(entity_name: str, entry: Dict[str, Any]) -> Dict[str, Any]:
        sem_b = entry.get("semantic_box_center_base") or entry.get("gt_geometry_center_base")
        top_c = entry.get("gt_top_face_center_base")
        return {
            "entity_name": str(entity_name),
            "label": str(entry.get("label", "")),
            "role": str(entry.get("role", "unknown")),
            "semantic_center_base": list(sem_b) if isinstance(sem_b, (list, tuple)) else None,
            "semantic_center_world": entry.get("semantic_box_center_world"),
            "yaw_rad": _safe_float(entry.get("gt_yaw_rad", entry.get("yaw_rad"))),
            "dims_lwh": entry.get("dims_used_lwh") or entry.get("dims_lwh"),
            "top_face_center_base": list(top_c) if isinstance(top_c, (list, tuple)) else None,
            "top_face_corners_base": entry.get("gt_top_face_corners_base"),
            "length_axis_world": entry.get("gt_length_axis_world")
            or entry.get("length_axis_world"),
            "width_axis_world": entry.get("gt_width_axis_world")
            or entry.get("width_axis_world"),
            "closing_axis_world": entry.get("gt_closing_axis_world")
            or entry.get("closing_axis_world"),
            "required_gripper_width": entry.get("required_gripper_width"),
            "collision_shape": entry.get("collision_shape"),
            "collision_dims": entry.get("collision_dims"),
            "collision_dims_inflated": entry.get("collision_dims_inflated"),
            "collision_margin": entry.get("collision_margin"),
            "collision_box_pose": entry.get("collision_box_pose"),
            "collision_dims_moveit": entry.get("collision_dims_moveit"),
            "grasp_policy": entry.get("grasp_policy"),
            "top_face_source": "runtime_gt_known_box",
            "grasp_center_source": "runtime_gt_box_center",
            "yaw_source": "runtime_gt_spawn_yaw",
        }

    def _build_runtime_scene_executor_context(
        self,
        objects_out: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Escena geométrica multi-objeto desde GT runtime + detección YOLO."""
        scene_objects: List[Dict[str, Any]] = []
        if self._use_runtime_scene_gt and self._gt_spawner_by_entity:
            twb = self._lookup_world_to_base_matrix()
            for ent_name, entry in sorted(self._gt_spawner_by_entity.items()):
                if twb is not None:
                    enrich_gt_object_entry_base(entry, twb, logger=self.get_logger())
                short = ent_name.split("::")[-1]
                scene_objects.append(self._gt_entry_to_scene_object(short, entry))

        target_candidate: Optional[Dict[str, Any]] = None
        obstacles: List[Dict[str, Any]] = []
        targets_gt = [o for o in scene_objects if o.get("role") == ROLE_TARGET]
        obstacles_gt = [o for o in scene_objects if o.get("role") == ROLE_OBSTACLE]

        best_det: Optional[Dict[str, Any]] = None
        best_score = -1.0
        for obj in objects_out:
            if not isinstance(obj, dict) or not self._is_operational_detection(obj):
                continue
            sc = _safe_float(obj.get("score"), 0.0) or 0.0
            if sc > best_score:
                best_score = sc
                best_det = obj

        if best_det is not None:
            pos = best_det.get("grasp_center_base") or best_det.get("position")
            lbl = str(best_det.get("label", ""))
            matched_entity = str(best_det.get("gt_entity_name") or "")
            if (
                not matched_entity
                and isinstance(pos, (list, tuple))
                and len(pos) >= 2
                and self._gt_spawner_by_entity
            ):
                found = find_gt_object_nearest_xy(
                    self._gt_spawner_by_entity,
                    (float(pos[0]), float(pos[1])),
                    label=lbl,
                    entity_prefix=self._debug_gt_entity_prefix,
                    max_dist_m=0.20,
                )
                if found is not None:
                    matched_entity, _entry = found
                    best_det["gt_entity_name"] = matched_entity
            target_candidate = dict(best_det)
            target_candidate["entity_name"] = matched_entity
            target_candidate["role"] = ROLE_TARGET
            self.get_logger().info(
                "[TARGET_OBJECT_SELECTED] label=%s entity=%s score=%.3f "
                "top_face_source=%s grasp_center_source=%s yaw_source=%s"
                % (
                    lbl,
                    matched_entity or "n/a",
                    float(best_det.get("score", 0.0)),
                    str(best_det.get("top_face_source", "")),
                    str(best_det.get("grasp_center_source", "")),
                    str(best_det.get("yaw_source", "")),
                )
            )
            if targets_gt:
                for tg in targets_gt:
                    if matched_entity and tg.get("entity_name") == matched_entity:
                        target_candidate["runtime_scene_object"] = tg
                        break
                else:
                    target_candidate["runtime_scene_object"] = targets_gt[0]
        elif targets_gt:
            target_candidate = {"runtime_scene_object": targets_gt[0], "role": ROLE_TARGET}

        for obs in obstacles_gt:
            obstacles.append(obs)
        for obj in scene_objects:
            if obj.get("role") == ROLE_OBSTACLE:
                continue
            ent = str(obj.get("entity_name", ""))
            if target_candidate and ent == str(target_candidate.get("entity_name", "")):
                continue
            if obj.get("role") not in (ROLE_TARGET, ROLE_OBSTACLE):
                obstacles.append(obj)

        return {
            "scene_objects": scene_objects,
            "obstacles": obstacles,
            "target_candidate": target_candidate,
        }

    def _resolve_gt_top_face_for_label(
        self, label: str
    ) -> Tuple[Optional[List[List[float]]], str, Optional[Dict[str, Any]]]:
        twb = self._lookup_world_to_base_matrix()
        if twb is None:
            return None, "", None
        if self._use_runtime_scene_gt and self._gt_spawner_by_entity:
            found = find_gt_object_for_label(
                self._gt_spawner_by_entity,
                label,
                entity_prefix=self._debug_gt_entity_prefix,
            )
            if found is not None:
                entity_name, entry = found
                enrich_gt_object_entry_base(entry, twb)
                corners = entry.get("gt_top_face_corners_base")
                if not (isinstance(corners, list) and len(corners) >= 4):
                    corners = build_gt_top_face_from_spawner_entry(entry, twb)
                if corners:
                    return corners, entity_name, entry
        if self._debug_draw_gazebo_gt and self._gt_world_poses:
            found_gz = find_runtime_entity_for_label(
                self._gt_world_poses,
                label,
                entity_prefix=self._debug_gt_entity_prefix,
            )
            if found_gz is not None:
                entity_name, pose6 = found_gz
                corners = build_gt_top_face_corners_base(pose6, label, twb)
                if corners:
                    return corners, entity_name, None
        return None, "", None

    def _apply_visual_gt_metrics_and_gate(
        self,
        det: Any,
        pose_meta: Dict[str, Any],
        *,
        model_corners: Optional[List[List[float]]],
        observed_corners: Optional[List[List[float]]],
        fx: float,
        z_est_m: float,
        project_uv_fn: Any,
        obb_center_uv: Optional[List[float]] = None,
    ) -> Dict[str, Any]:
        gt_corners, gt_entity, _gt_entry = self._resolve_gt_top_face_for_label(
            str(det.label)
        )
        has_gt = bool(gt_corners and len(gt_corners) >= 4)
        if not has_gt:
            pose_meta["visual_pose_gate_passed"] = True
            pose_meta["visual_pose_gate_reason"] = "no_gt_available"
            return pose_meta
        metrics = compute_top_face_gt_metrics(
            observed_corners_base=observed_corners,
            model_corners_base=model_corners,
            gt_corners_base=gt_corners,
            fx=float(fx),
            z_est_m=max(float(z_est_m), 0.25),
        )
        pose_meta.update(metrics)
        mask_ok, mask_reason, _ = model_top_face_mask_coherence(
            model_corners or [],
            det,
            project_uv_fn,
            obb_center_uv=obb_center_uv,
        )
        gate_on = self._enable_visual_pose_gate and (
            self._use_runtime_scene_gt or self._debug_draw_gazebo_gt
        )
        passed, reason = evaluate_visual_pose_gate(
            metrics,
            mask_ok,
            mask_reason,
            enable=gate_on,
            max_model_vs_gt_center_xy_m=self._visual_gate_max_model_gt_xy,
            max_observed_vs_model_center_xy_m=self._visual_gate_max_obs_model_xy,
            require_mask_coherence=self._visual_gate_require_mask,
        )
        pose_meta["visual_pose_gate_passed"] = bool(passed)
        pose_meta["visual_pose_gate_reason"] = str(reason)
        pose_meta["gt_entity_name"] = gt_entity
        self._gt_compare_log_serial += 1
        if self._gt_compare_log_serial % self._debug_image_log_every_n == 1:
            log_top_face_gt_compare(
                self.get_logger(),
                label=str(det.label),
                entity=gt_entity,
                metrics=metrics,
                pose_meta=pose_meta,
            )
            log_pose_gate_visual(
                self.get_logger(),
                label=str(det.label),
                accepted=passed,
                reason=reason,
                metrics=metrics,
            )
        return pose_meta

    @staticmethod
    def _transform_point(point: np.ndarray, transform) -> np.ndarray:
        t = np.array(
            [
                transform.transform.translation.x,
                transform.transform.translation.y,
                transform.transform.translation.z,
            ]
        )
        q = [
            transform.transform.rotation.x,
            transform.transform.rotation.y,
            transform.transform.rotation.z,
            transform.transform.rotation.w,
        ]
        m = tf_transformations.quaternion_matrix(q)
        m[:3, 3] = t
        h = np.array([point[0], point[1], point[2], 1.0])
        return (m @ h)[:3]

    @staticmethod
    def _transform_points(points: np.ndarray, transform) -> np.ndarray:
        if points.size == 0:
            return points.reshape((-1, 3))
        t = np.array(
            [
                transform.transform.translation.x,
                transform.transform.translation.y,
                transform.transform.translation.z,
            ]
        )
        q = [
            transform.transform.rotation.x,
            transform.transform.rotation.y,
            transform.transform.rotation.z,
            transform.transform.rotation.w,
        ]
        m = tf_transformations.quaternion_matrix(q)
        m[:3, 3] = t
        homog = np.hstack((points, np.ones((points.shape[0], 1), dtype=np.float64)))
        return (m @ homog.T).T[:, :3]

    @staticmethod
    def _build_transform_matrix(transform) -> np.ndarray:
        t = np.array(
            [
                transform.transform.translation.x,
                transform.transform.translation.y,
                transform.transform.translation.z,
            ],
            dtype=np.float64,
        )
        q = [
            transform.transform.rotation.x,
            transform.transform.rotation.y,
            transform.transform.rotation.z,
            transform.transform.rotation.w,
        ]
        m = tf_transformations.quaternion_matrix(q)
        m[:3, 3] = t
        return m

    @staticmethod
    def _policy_shape_supports_known_grasp_center(shape: str) -> bool:
        s = str(shape or "").strip().lower()
        return s in ("box", "low_box", "low_box_wide", "curved_long")

    @staticmethod
    def _known_rectangle_fit_valid_for_operational_center(
        policy: Dict[str, Any], pose_meta: Dict[str, Any]
    ) -> bool:
        if not PerceptionNode._policy_shape_supports_known_grasp_center(
            str(policy.get("shape", ""))
        ):
            return False
        ys = str(pose_meta.get("yaw_source", "")).strip()
        if ys not in ("known_rectangle_fit", "hybrid_top_face_known_dims"):
            return False
        if ys == "known_rectangle_fit":
            if not bool(pose_meta.get("pose_fit_success", False)):
                return False
        elif not bool(pose_meta.get("hybrid_fit_success", False)):
            return False
        try:
            if float(pose_meta.get("yaw_confidence", 0.0)) < 0.65:
                return False
        except (TypeError, ValueError):
            return False
        pfe = pose_meta.get("pose_fit_error")
        if pfe is None:
            return False
        try:
            pfe_lim = 0.10 if ys == "hybrid_top_face_known_dims" else 0.055
            if float(pfe) > pfe_lim:
                return False
        except (TypeError, ValueError):
            return False
        ir = pose_meta.get("inlier_ratio")
        if ir is not None:
            try:
                if float(ir) < 0.76:
                    return False
            except (TypeError, ValueError):
                pass
        if ys == "known_rectangle_fit":
            ym = pose_meta.get("yaw_margin_score")
            if ym is not None:
                try:
                    if float(ym) < 0.00035:
                        return False
                except (TypeError, ValueError):
                    pass
        kbc = pose_meta.get("known_box_center_base")
        return isinstance(kbc, (list, tuple)) and len(kbc) >= 3

    @staticmethod
    def _hybrid_fit_has_operational_center(pose_meta: Dict[str, Any]) -> bool:
        if not bool(pose_meta.get("hybrid_fit_success", False)):
            return False
        if str(pose_meta.get("yaw_source", "")).strip() != "hybrid_top_face_known_dims":
            return False
        kbc = pose_meta.get("known_box_center_base")
        return isinstance(kbc, (list, tuple)) and len(kbc) >= 3

    @staticmethod
    def _model_fit_has_operational_center(pose_meta: Dict[str, Any]) -> bool:
        if not bool(pose_meta.get("model_top_face_success", False)):
            return False
        src = str(pose_meta.get("top_face_source", "")).strip()
        if src not in ("known_model", "hybrid_known_model"):
            return False
        mbc = pose_meta.get("model_box_center_base")
        return isinstance(mbc, (list, tuple)) and len(mbc) >= 3

    @staticmethod
    def _runtime_gt_spawn_has_operational_center(pose_meta: Dict[str, Any]) -> bool:
        if str(pose_meta.get("top_face_source", "")).strip() not in (
            "runtime_gt_known_box",
            "runtime_gt_known_object",
            "runtime_gt_tall_object",
            "runtime_gt_known_cylinder",
        ):
            return False
        kbc = (
            pose_meta.get("known_box_center_base")
            or pose_meta.get("known_object_center_base")
            or pose_meta.get("known_cylinder_center_base")
        )
        return isinstance(kbc, (list, tuple)) and len(kbc) >= 3

    @staticmethod
    def _normalize_closing_yaw_rad(yaw_rad: float) -> float:
        return float((float(yaw_rad) + math.pi) % (2.0 * math.pi) - math.pi)

    def _resolve_closing_yaw_and_source(
        self,
        label: str,
        pose_meta: Dict[str, Any],
        policy: Dict[str, Any],
        yaw_value: float,
    ) -> Tuple[Optional[float], Optional[str], str]:
        """Resuelve closing_yaw/closing_yaw_source sin depender de orden de logs/overlay."""
        lb = str(label).strip().lower()
        top_face_source = str(pose_meta.get("top_face_source", "")).strip()
        closing_yaw: Optional[float] = None
        closing_yaw_source: Optional[str] = None
        stage = "init"

        if self._runtime_gt_spawn_has_operational_center(pose_meta):
            stage = "runtime_gt"
            closing_yaw_rad = _safe_float(pose_meta.get("model_closing_yaw_rad"))
            if closing_yaw_rad is None:
                closing_yaw_rad = _safe_float(pose_meta.get("closing_yaw_rad"))
            if closing_yaw_rad is not None:
                closing_yaw = float(closing_yaw_rad)
                if top_face_source == "runtime_gt_known_cylinder":
                    closing_yaw_source = str(
                        pose_meta.get("closing_yaw_source", "runtime_gt_yaw_free")
                    )
                elif top_face_source in (
                    "runtime_gt_known_object",
                    "runtime_gt_tall_object",
                ):
                    closing_yaw_source = str(
                        pose_meta.get(
                            "closing_yaw_source", "runtime_gt_known_object_short_axis"
                        )
                    )
                elif top_face_source in (
                    "runtime_gt_known_box",
                    "runtime_scene_gt_known_box",
                ) or lb in KNOWN_BOX_LABELS:
                    closing_yaw_source = str(
                        pose_meta.get("closing_yaw_source", "runtime_gt_short_axis")
                    )
                elif top_face_source in (
                    "runtime_gt_tall_object",
                    "runtime_gt_known_object",
                ) or lb in TALL_KNOWN_OBJECT_LABELS:
                    closing_yaw_source = str(
                        pose_meta.get("closing_yaw_source", "runtime_gt_short_axis")
                    )
                else:
                    closing_yaw_source = str(
                        pose_meta.get("closing_yaw_source", "runtime_gt_short_axis")
                    )

        if closing_yaw is None and self._model_fit_has_operational_center(pose_meta):
            stage = "model_fit"
            cy = _safe_float(pose_meta.get("model_closing_yaw_rad"), None)
            if cy is not None:
                closing_yaw = float(cy)
                closing_yaw_source = str(
                    pose_meta.get("closing_yaw_source", "model_fit_axis")
                )

        if closing_yaw is None:
            stage = "yaw_policy_derived"
            preferred_axis = str(policy.get("preferred_closing_axis", "short_axis"))
            if preferred_axis in ("short_axis", "perpendicular_to_long_axis"):
                closing_yaw = float(yaw_value) + math.pi / 2.0
            else:
                closing_yaw = float(yaw_value)
            closing_yaw_source = str(
                pose_meta.get("closing_yaw_source", "derived_from_yaw_policy")
            )

        if closing_yaw is not None:
            closing_yaw = self._normalize_closing_yaw_rad(closing_yaw)
            return closing_yaw, closing_yaw_source, stage

        stage = "fallback_pose_meta"
        cy_meta = _safe_float(pose_meta.get("model_closing_yaw_rad"))
        if cy_meta is None:
            cy_meta = _safe_float(pose_meta.get("closing_yaw_rad"))
        if cy_meta is not None:
            closing_yaw = self._normalize_closing_yaw_rad(float(cy_meta))
            closing_yaw_source = str(
                pose_meta.get("closing_yaw_source", "normalized_pose_meta")
            )
            return closing_yaw, closing_yaw_source, stage

        if lb in DEMO_KNOWN_OBJECTS:
            return None, None, "missing_closing_yaw_demo"

        stage = "active_yaw_fallback"
        closing_yaw = self._normalize_closing_yaw_rad(float(yaw_value))
        closing_yaw_source = "active_yaw_fallback"
        return closing_yaw, closing_yaw_source, stage

    def _resolve_runtime_gt_operational_face(
        self, label: str
    ) -> Optional[Dict[str, Any]]:
        """Cara superior operativa desde GT runtime (centro semántico + top_z del spawn)."""
        if not self._use_runtime_scene_gt or not self._gt_spawner_by_entity:
            return None
        found = find_gt_object_for_label(
            self._gt_spawner_by_entity,
            label,
            entity_prefix=self._debug_gt_entity_prefix,
        )
        if found is None:
            return None
        entity_name, entry = found
        twb = self._lookup_world_to_base_matrix()
        if twb is None:
            return None

        entry["mustard_cap_center_mode"] = str(self._mustard_cap_center_mode)
        apply_tall_object_sdf_geometry_correction(entry, logger=self.get_logger())
        enrich_gt_object_entry_base(entry, twb, logger=self.get_logger())

        sdf_applied = bool(entry.get("tall_object_sdf_offset_applied"))
        model_origin_base = entry.get("model_origin_pose_base")

        corners = entry.get("gt_top_face_corners_base")
        geom_center = (
            entry.get("gt_geometry_center_base")
            or entry.get("semantic_box_center_base")
        )
        if not isinstance(geom_center, (list, tuple)) or len(geom_center) < 3:
            pw = entry.get("pose_world")
            if isinstance(pw, dict):
                xyz_w = [
                    _safe_float(pw.get("x"), 0.0),
                    _safe_float(pw.get("y"), 0.0),
                    _safe_float(pw.get("z"), 0.0),
                ]
                hom = np.array(
                    [[float(xyz_w[0]), float(xyz_w[1]), float(xyz_w[2]), 1.0]],
                    dtype=np.float64,
                )
                out = (twb @ hom.T).T[0, :3]
                geom_center = [float(out[0]), float(out[1]), float(out[2])]
        top_face_center = entry.get("gt_top_face_center_base")
        if not (isinstance(geom_center, (list, tuple)) and len(geom_center) >= 3):
            return None

        gx, gy, gz_geom = float(geom_center[0]), float(geom_center[1]), float(geom_center[2])
        dims_lwh = entry.get("dims_used_lwh") or entry.get("dims_lwh") or entry.get("dims_m") or []
        if not (isinstance(corners, list) and len(corners) >= 4):
            if not (isinstance(dims_lwh, (list, tuple)) and len(dims_lwh) >= 3):
                return None
            l = float(dims_lwh[0])
            w = float(dims_lwh[1])
            h = float(dims_lwh[2])
            yaw_for_corners = float(entry.get("gt_yaw_rad", entry.get("yaw_rad", 0.0)))
            hz = 0.5 * h
            z_top = gz_geom + hz
            lax = str(entry.get("local_length_axis", "x"))
            wax = str(entry.get("local_width_axis", "y"))
            axes_fb = resolve_runtime_gt_spawn_axes(
                yaw_for_corners,
                local_length_axis=lax,
                local_width_axis=wax,
            )
            lx, ly = axes_fb["long_axis_xy"]
            sx, sy = axes_fb["short_axis_xy"]
            ex = np.array([float(lx), float(ly), 0.0], dtype=np.float64)
            ey = np.array([float(sx), float(sy), 0.0], dtype=np.float64)
            center_top = np.array([gx, gy, z_top], dtype=np.float64)
            hl = 0.5 * l
            hw = 0.5 * w
            corners = [
                list((center_top + hl * ex + hw * ey).astype(float)),
                list((center_top + hl * ex - hw * ey).astype(float)),
                list((center_top - hl * ex - hw * ey).astype(float)),
                list((center_top - hl * ex + hw * ey).astype(float)),
            ]
            top_face_center = [float(center_top[0]), float(center_top[1]), float(center_top[2])]
        payload_top_z_before = _safe_float(entry.get("top_z_m"))
        if payload_top_z_before is None and isinstance(top_face_center, (list, tuple)) and len(top_face_center) >= 3:
            payload_top_z_before = float(top_face_center[2])

        dims_lwh_pre = entry.get("dims_used_lwh") or entry.get("dims_used") or entry.get("dims_lwh") or entry.get("dims_m") or []
        height_m_pre = None
        if isinstance(dims_lwh_pre, (list, tuple)) and len(dims_lwh_pre) >= 3:
            height_m_pre = float(dims_lwh_pre[2])

        lb_pre = normalize_label(label)
        policy_pre = get_grasp_policy(lb_pre)
        use_cylinder_top_z = lb_pre in CYLINDER_KNOWN_LABELS
        use_tall_top_z = (
            str(policy_pre.get("primary_strategy", "")) == "tall_object_topdown"
            or lb_pre in TALL_KNOWN_OBJECT_LABELS
            or use_cylinder_top_z
        )
        tall_z_dbg: Dict[str, Any] = {}
        if use_tall_top_z:
            if sdf_applied and payload_top_z_before is not None:
                top_z_m = float(payload_top_z_before)
                tall_z_dbg = {
                    "label": lb_pre,
                    "geometry_center_z": float(gz_geom),
                    "height_m": height_m_pre,
                    "payload_top_z_before": payload_top_z_before,
                    "payload_top_z_after": top_z_m,
                    "computed_top_z": top_z_m,
                    "source": "sdf_preserved_top_z_m",
                }
            else:
                top_z_m, tall_z_dbg = resolve_tall_object_top_z_m(
                    lb_pre,
                    float(gz_geom),
                    height_m=height_m_pre,
                    payload_top_z_before=payload_top_z_before,
                )
            self.get_logger().info(
                "[TALL_OBJECT_Z_RESOLVE]\n"
                "label=%s\n"
                "geometry_center_z=%.4f\n"
                "height_m=%s\n"
                "computed_top_z=%.4f\n"
                "payload_top_z_before=%s\n"
                "payload_top_z_after=%.4f\n"
                "source=%s"
                % (
                    lb_pre,
                    float(tall_z_dbg.get("geometry_center_z", gz_geom)),
                    "%.4f" % float(tall_z_dbg["height_m"])
                    if tall_z_dbg.get("height_m") is not None
                    else "n/a",
                    float(tall_z_dbg.get("computed_top_z", top_z_m)),
                    "n/a"
                    if tall_z_dbg.get("payload_top_z_before") is None
                    else "%.4f" % float(tall_z_dbg["payload_top_z_before"]),
                    float(top_z_m),
                    str(tall_z_dbg.get("source", "known_geometry_height")),
                )
            )
        else:
            top_z_m = payload_top_z_before
            if top_z_m is None:
                if height_m_pre is not None:
                    top_z_m = float(gz_geom) + 0.5 * float(height_m_pre)
                else:
                    top_z_m = float(gz_geom)
            top_z_m = float(top_z_m)

        if isinstance(top_face_center, (list, tuple)) and len(top_face_center) >= 3:
            top_center_base = [
                float(top_face_center[0]),
                float(top_face_center[1]),
                float(top_z_m),
            ]
        else:
            top_center_base = [gx, gy, top_z_m]

        if use_cylinder_top_z:
            grasp_center_base = [gx, gy, float(gz_geom)]
        else:
            grasp_center_base = [gx, gy, top_z_m]

        gt_yaw = float(entry.get("gt_yaw_rad", entry.get("yaw_rad", 0.0)))

        length_ax = entry.get("gt_length_axis_world")
        width_ax = entry.get("gt_width_axis_world")
        if isinstance(length_ax, (list, tuple)) and len(length_ax) >= 2:
            long_xy = [float(length_ax[0]), float(length_ax[1])]
        else:
            long_xy = [float(math.cos(gt_yaw)), float(math.sin(gt_yaw))]
        if isinstance(width_ax, (list, tuple)) and len(width_ax) >= 2:
            short_xy = [float(width_ax[0]), float(width_ax[1])]
        else:
            short_xy = [
                float(math.cos(gt_yaw + math.pi / 2.0)),
                float(math.sin(gt_yaw + math.pi / 2.0)),
            ]

        closing_ax = entry.get("gt_closing_axis_world") or entry.get("gt_width_axis_world")
        if isinstance(closing_ax, (list, tuple)) and len(closing_ax) >= 2:
            closing_yaw_rad = float(
                math.atan2(float(closing_ax[1]), float(closing_ax[0]))
            )
        else:
            closing_yaw_rad = float(
                math.atan2(float(short_xy[1]), float(short_xy[0]))
            )

        dims_lwh = entry.get("dims_used_lwh") or entry.get("dims_used") or entry.get("dims_lwh") or entry.get("dims_m") or []
        shape = str(entry.get("shape", "")).strip().lower()
        lb = normalize_label(label)
        if lb in CYLINDER_KNOWN_LABELS or "cylinder" in shape:
            top_src = "runtime_gt_known_cylinder"
            center_src = "runtime_gt_cylinder_center"
            closing_yaw_source = "runtime_gt_yaw_free"
        elif (
            lb in KNOWN_BOX_LABELS
            or is_known_spawn_geometry_box_label(label)
        ):
            top_src = "runtime_gt_known_box"
            center_src = "runtime_gt_box_center"
            closing_yaw_source = "runtime_gt_short_axis"
        elif lb in TALL_KNOWN_OBJECT_LABELS or shape in (
            "tall_box_like",
            "bottle",
        ):
            top_src = "runtime_gt_tall_object"
            center_src = "runtime_gt_tall_object_center"
            closing_yaw_source = "runtime_gt_short_axis"
        else:
            top_src = "runtime_gt_known_object"
            center_src = "runtime_gt_object_center"
            closing_yaw_source = "runtime_gt_short_axis"

        tall_body_center_base = [gx, gy, float(gz_geom)]
        old_grasp_xy = (
            (float(model_origin_base[0]), float(model_origin_base[1]))
            if isinstance(model_origin_base, (list, tuple)) and len(model_origin_base) >= 2
            else (gx, gy)
        )
        used_direct_runtime_gt = False
        offset_local_xy_source = ""
        entry_gcs = str(entry.get("grasp_center_source") or "").strip()
        entry_gcb = entry.get("grasp_center_base")
        if (
            lb == "mustard_bottle"
            and entry_gcs
            in (
                MUSTARD_TOP_CAP_CENTER_GEOMETRY_SOURCE,
                MUSTARD_VERTICAL_AXIS_CAP_CENTER_SOURCE,
            )
            and isinstance(entry_gcb, (list, tuple))
            and len(entry_gcb) >= 3
        ):
            grasp_center_base = [
                float(entry_gcb[0]),
                float(entry_gcb[1]),
                float(entry_gcb[2]),
            ]
            center_src = entry_gcs
            top_z_m = float(entry_gcb[2])
            top_center_base = list(grasp_center_base)
            used_direct_runtime_gt = True
            off_local_xy, offset_local_xy_source = _resolve_tall_cap_offset_local_xy(
                entry, {}
            )
            off_base = entry.get("offset_base_xy")
            if not (isinstance(off_base, (list, tuple)) and len(off_base) >= 2):
                off_base = (0.0, 0.0)
            tall_dbg = {
                "applied": True,
                "label": lb,
                "strategy": "runtime_scene_cap_center",
                "body_center_xy": (gx, gy),
                "topdown_center_xy": (
                    float(grasp_center_base[0]),
                    float(grasp_center_base[1]),
                ),
                "source": center_src,
                "sdf_offset_applied": bool(sdf_applied),
                "offset_local_xy": off_local_xy,
                "offset_base_xy": (
                    float(off_base[0]),
                    float(off_base[1]),
                ),
            }
        elif sdf_applied:
            cap_b = entry.get("gt_top_face_center_base") or entry.get("grasp_center_base")
            if isinstance(cap_b, (list, tuple)) and len(cap_b) >= 2:
                grasp_center_base = [
                    float(cap_b[0]),
                    float(cap_b[1]),
                    float(top_z_m),
                ]
                if (
                    isinstance(cap_b, (list, tuple))
                    and len(cap_b) >= 3
                    and str(entry.get("grasp_center_source") or "").strip()
                    in (
                        MUSTARD_TOP_CAP_CENTER_GEOMETRY_SOURCE,
                        MUSTARD_VERTICAL_AXIS_CAP_CENTER_SOURCE,
                    )
                ):
                    grasp_center_base = [
                        float(cap_b[0]),
                        float(cap_b[1]),
                        float(cap_b[2]),
                    ]
                    top_z_m = float(cap_b[2])
            else:
                grasp_center_base = [gx, gy, float(top_z_m)]
            center_src = str(
                entry.get("grasp_center_source")
                or TALL_OBJECT_CAP_CENTER_SDF_OFFSET_SOURCE
            )
            if lb == "mustard_bottle" and center_src == TALL_OBJECT_CAP_CENTER_SDF_OFFSET_SOURCE:
                center_src = MUSTARD_TOP_CAP_CENTER_GEOMETRY_SOURCE
            off_local_xy, offset_local_xy_source = _resolve_tall_cap_offset_local_xy(
                entry, {}
            )
            off_base = entry.get("offset_base_xy")
            if not (isinstance(off_base, (list, tuple)) and len(off_base) >= 2):
                off_base = (0.0, 0.0)
            tall_dbg = {
                "applied": True,
                "label": lb,
                "strategy": "tall_object_topdown",
                "body_center_xy": (gx, gy),
                "topdown_center_xy": (
                    float(grasp_center_base[0]),
                    float(grasp_center_base[1]),
                ),
                "source": center_src,
                "sdf_offset_applied": True,
                "offset_local_xy": off_local_xy,
                "offset_base_xy": (
                    float(off_base[0]),
                    float(off_base[1]),
                ),
            }
        else:
            grasp_center_base, center_src, tall_dbg = apply_tall_object_topdown_grasp_center_offset(
                lb,
                (gx, gy),
                gt_yaw,
                top_z_m,
                default_center_source=center_src,
            )
        if center_src in (
            TALL_OBJECT_CAP_CENTER_SOURCE,
            TALL_OBJECT_CAP_CENTER_SDF_OFFSET_SOURCE,
        ) or tall_dbg.get("applied"):
            cap_x = float(grasp_center_base[0])
            cap_y = float(grasp_center_base[1])
            top_z_m = float(top_z_m)
            grasp_center_base = [cap_x, cap_y, top_z_m]
            top_center_base = [cap_x, cap_y, top_z_m]
        if lb == "mustard_bottle":
            off_local_xy, off_src = _resolve_tall_cap_offset_local_xy(entry, tall_dbg)
            if not offset_local_xy_source:
                offset_local_xy_source = off_src
            self.get_logger().info(
                "[MUSTARD_GT_CAP_CENTER_CONSUME]\n"
                "entry_grasp_center_base=%s\n"
                "entry_grasp_center_source=%s\n"
                "used_grasp_center_base=%s\n"
                "used_direct_runtime_gt=%s\n"
                "grasp_center_from_runtime=%s\n"
                "offset_local_xy_source=%s\n"
                "offset_local_xy=(%.4f, %.4f)\n"
                "result=OK"
                % (
                    str(entry_gcb),
                    entry_gcs or "n/a",
                    str(grasp_center_base),
                    str(used_direct_runtime_gt).lower(),
                    str(used_direct_runtime_gt).lower(),
                    offset_local_xy_source or off_src,
                    float(off_local_xy[0]),
                    float(off_local_xy[1]),
                )
            )
        if tall_dbg.get("applied") and center_src != TALL_OBJECT_CAP_CENTER_SDF_OFFSET_SOURCE:
            off_local_xy, _ = _resolve_tall_cap_offset_local_xy(entry, tall_dbg)
            body_xy = tall_dbg.get("body_center_xy", (gx, gy))
            topdown_xy = tall_dbg.get(
                "topdown_center_xy",
                (float(grasp_center_base[0]), float(grasp_center_base[1])),
            )
            off_base_xy = tall_dbg.get("offset_base_xy", (0.0, 0.0))
            self.get_logger().info(
                "[TALL_OBJECT_GRASP_CENTER]\n"
                "label=%s\n"
                "strategy=%s\n"
                "body_center=(%.4f, %.4f)\n"
                "topdown_center=(%.4f, %.4f)\n"
                "offset_local_xy=(%.4f, %.4f)\n"
                "offset_base_xy=(%.4f, %.4f)\n"
                "yaw=%.4f\n"
                "source=%s"
                % (
                    str(tall_dbg.get("label", lb)),
                    str(tall_dbg.get("strategy", "tall_object_topdown")),
                    float(body_xy[0]),
                    float(body_xy[1]),
                    float(topdown_xy[0]),
                    float(topdown_xy[1]),
                    float(off_local_xy[0]),
                    float(off_local_xy[1]),
                    float(off_base_xy[0]),
                    float(off_base_xy[1]),
                    float(tall_dbg.get("yaw_rad", gt_yaw)),
                    str(tall_dbg.get("source", center_src)),
                )
            )
            if lb == "bleach_cleanser":
                offset_local = tall_dbg.get("offset_local_xy", (0.0, 0.0))
                offset_base = tall_dbg.get("offset_base_xy", (0.0, 0.0))
                self.get_logger().info(
                    "[BLEACH_CAP_CENTER_OFFSET]\n"
                    "label=bleach_cleanser\n"
                    "old_grasp_center_xy=(%.4f, %.4f)\n"
                    "major_axis_xy=(%.4f, %.4f)\n"
                    "minor_axis_xy=(%.4f, %.4f)\n"
                    "offset_long_m=%.4f\n"
                    "offset_short_m=%.4f\n"
                    "offset_world_xy=(%.4f, %.4f)\n"
                    "new_grasp_center_xy=(%.4f, %.4f)\n"
                    "source=topdown_grasp_center_offset\n"
                    "result=APPLIED"
                    % (
                        float(tall_dbg.get("body_center_xy", (gx, gy))[0]),
                        float(tall_dbg.get("body_center_xy", (gx, gy))[1]),
                        float(long_xy[0]),
                        float(long_xy[1]),
                        float(short_xy[0]),
                        float(short_xy[1]),
                        float(offset_local[0]),
                        float(offset_local[1]),
                        float(offset_base[0]),
                        float(offset_base[1]),
                        float(
                            tall_dbg.get(
                                "topdown_center_xy",
                                (grasp_center_base[0], grasp_center_base[1]),
                            )[0]
                        ),
                        float(
                            tall_dbg.get(
                                "topdown_center_xy",
                                (grasp_center_base[0], grasp_center_base[1]),
                            )[1]
                        ),
                    )
                )

        return {
            "gt_entity_name": entity_name,
            "top_center_base": top_center_base,
            "top_z_m": top_z_m,
            "top_face_corners_base": [
                [float(c[0]), float(c[1]), float(c[2])] for c in corners[:4]
            ],
            "grasp_center_base": grasp_center_base,
            "tall_object_body_center_base": tall_body_center_base,
            "tall_object_grasp_center_debug": tall_dbg,
            "tall_object_top_z_resolve": tall_z_dbg,
            "tall_object_sdf_offset_applied": sdf_applied,
            "model_origin_pose_base": list(model_origin_base)
            if isinstance(model_origin_base, (list, tuple))
            else None,
            "top_face_source": top_src,
            "grasp_center_source": center_src,
            "yaw_source": "runtime_gt_spawn_yaw",
            "closing_yaw_source": closing_yaw_source,
            "closing_yaw_rad": closing_yaw_rad,
            "gt_yaw_rad": gt_yaw,
            "long_axis_xy": long_xy,
            "short_axis_xy": short_xy,
            "dims_used_lwh": list(dims_lwh) if isinstance(dims_lwh, (list, tuple)) else [],
            "bottom_corners_base": list(entry.get("gt_bottom_face_corners_base") or []),
            "mustard_old_offset_cap_center_base": entry.get(
                "mustard_old_offset_cap_center_base"
            ),
            "mustard_vertical_axis_cap_center_base": entry.get(
                "mustard_vertical_axis_cap_center_base"
            ),
            "mustard_mesh_local_cap_center_base": entry.get(
                "mustard_mesh_local_cap_center_base"
            ),
            "mustard_footprint_center_source": entry.get(
                "mustard_footprint_center_source"
            ),
        }

    @staticmethod
    def _apply_operational_face_to_pose_meta(
        meta: Dict[str, Any],
        grasp_fields: Dict[str, Any],
        center_info: Dict[str, Any],
        ops: Dict[str, Any],
    ) -> None:
        """Copia campos operativos resueltos a meta / grasp_fields / center_info."""
        top_z_m = float(ops["top_z_m"])
        gt_yaw = float(ops["gt_yaw_rad"])
        long_xy = list(ops["long_axis_xy"])
        short_xy = list(ops["short_axis_xy"])
        grasp_center = list(ops["grasp_center_base"])
        top_center = list(ops["top_center_base"])
        corners = list(ops["top_face_corners_base"])
        dims_lwh = ops.get("dims_used_lwh") or []

        meta["runtime_gt_geometry_applied"] = True
        meta["gt_entity_name"] = ops.get("gt_entity_name", "")
        meta["top_face_source"] = str(ops["top_face_source"])
        meta["grasp_center_source"] = str(ops["grasp_center_source"])
        meta["top_corners_base"] = None
        meta["runtime_gt_top_face_corners_base"] = corners
        meta["gt_top_face_center_base"] = top_center
        meta["bottom_corners_base"] = list(ops.get("bottom_corners_base") or [])
        body_b = ops.get("tall_object_body_center_base")
        if isinstance(body_b, (list, tuple)) and len(body_b) >= 2:
            meta["tall_object_body_center_base"] = [
                float(body_b[0]),
                float(body_b[1]),
                float(body_b[2]) if len(body_b) >= 3 else float(top_z_m),
            ]
            meta["known_object_center_base"] = list(meta["tall_object_body_center_base"])
        else:
            meta["known_object_center_base"] = list(grasp_center)
        if ops.get("tall_object_sdf_offset_applied"):
            mob = ops.get("model_origin_pose_base")
            if isinstance(mob, (list, tuple)) and len(mob) >= 3:
                meta["mustard_sdf_model_origin_base"] = [
                    float(mob[0]),
                    float(mob[1]),
                    float(mob[2]),
                ]
            meta["mustard_sdf_cap_center_base"] = list(grasp_center)
            meta["mustard_sdf_correction_applied"] = True
        old_off = ops.get("mustard_old_offset_cap_center_base")
        if isinstance(old_off, (list, tuple)) and len(old_off) >= 3:
            meta["mustard_old_offset_cap_center_base"] = [
                float(old_off[0]),
                float(old_off[1]),
                float(old_off[2]),
            ]
        vert_cap = ops.get("mustard_vertical_axis_cap_center_base")
        if isinstance(vert_cap, (list, tuple)) and len(vert_cap) >= 3:
            meta["mustard_vertical_axis_cap_center_base"] = list(vert_cap)
        mesh_cap = ops.get("mustard_mesh_local_cap_center_base")
        if isinstance(mesh_cap, (list, tuple)) and len(mesh_cap) >= 3:
            meta["mustard_mesh_local_cap_center_base"] = list(mesh_cap)
        if ops.get("mustard_footprint_center_source"):
            meta["mustard_footprint_center_source"] = str(
                ops.get("mustard_footprint_center_source")
            )
        meta["known_box_center_base"] = list(grasp_center)
        meta["known_cylinder_center_base"] = list(grasp_center)
        meta["top_surface_center_base"] = [
            float(grasp_center[0]),
            float(grasp_center[1]),
            float(top_z_m),
        ]
        meta["known_box_yaw_rad"] = gt_yaw
        meta["model_box_yaw_rad"] = gt_yaw
        meta["model_closing_yaw_rad"] = float(ops["closing_yaw_rad"])
        meta["closing_yaw_source"] = str(ops.get("closing_yaw_source", "runtime_gt_short_axis"))
        meta["long_axis_xy"] = long_xy
        meta["short_axis_xy"] = short_xy
        meta["yaw_source"] = str(ops["yaw_source"])
        meta["yaw_confidence"] = 1.0
        meta["center_method"] = "runtime_gt_spawn_geometry"
        meta["yaw_fit_method"] = "runtime_gt_spawn_geometry"
        meta["top_face_success"] = True
        meta["pose_fit_success"] = True
        meta["pose_fit_error"] = 0.0
        meta["top_z_estimated"] = top_z_m
        if isinstance(dims_lwh, (list, tuple)) and len(dims_lwh) >= 3:
            meta["db_length_m"] = float(dims_lwh[0])
            meta["db_width_m"] = float(dims_lwh[1])
            meta["projected_extent_length_m"] = float(dims_lwh[0])
            meta["projected_extent_width_m"] = float(dims_lwh[1])

        grasp_fields["top_z_m"] = top_z_m
        grasp_fields["grasp_yaw_rad"] = gt_yaw
        grasp_fields["grasp_yaw_deg"] = float(math.degrees(gt_yaw))
        grasp_fields["major_axis_xy"] = long_xy
        grasp_fields["minor_axis_xy"] = short_xy
        if isinstance(dims_lwh, (list, tuple)) and len(dims_lwh) >= 2:
            grasp_fields["footprint_major_m"] = float(dims_lwh[0])
            grasp_fields["footprint_minor_m"] = float(dims_lwh[1])

        center_info["chosen_target_center_base"] = grasp_center
        center_info["top_surface_center_base"] = list(grasp_center)
        center_info["target_center_method"] = "runtime_gt_top_face_center"
        center_info["grasp_center_source"] = str(ops["grasp_center_source"])

    def _resolve_operational_face_from_pose_meta(
        self,
        label: str,
        pose_meta: Dict[str, Any],
        grasp_fields: Dict[str, Any],
        *,
        top_face_observed_z: Optional[float] = None,
    ) -> Optional[Dict[str, Any]]:
        """Fallback observed/model/hybrid cuando no hay GT runtime."""
        corners = pose_meta.get("runtime_gt_top_face_corners_base")
        if not (isinstance(corners, list) and len(corners) >= 4):
            corners = pose_meta.get("model_top_face_corners_base")
        if not (isinstance(corners, list) and len(corners) >= 4):
            corners = pose_meta.get("top_corners_base")
        if not (isinstance(corners, list) and len(corners) >= 4):
            return None

        top_z_m = _safe_float(pose_meta.get("top_z_estimated"))
        if top_z_m is None:
            top_z_m = _safe_float(grasp_fields.get("top_z_m"))
        if top_z_m is None and top_face_observed_z is not None:
            top_z_m = float(top_face_observed_z)
        if top_z_m is None:
            try:
                top_z_m = float(np.mean([float(c[2]) for c in corners[:4]]))
            except (TypeError, ValueError):
                return None

        cx = float(np.mean([float(c[0]) for c in corners[:4]]))
        cy = float(np.mean([float(c[1]) for c in corners[:4]]))
        top_center_base = [cx, cy, float(top_z_m)]

        grasp_center = pose_meta.get("known_box_center_base") or pose_meta.get(
            "model_box_center_base"
        )
        if isinstance(grasp_center, (list, tuple)) and len(grasp_center) >= 3:
            grasp_center_base = [float(grasp_center[0]), float(grasp_center[1]), float(top_z_m)]
        else:
            grasp_center_base = [cx, cy, float(top_z_m)]

        gt_yaw = _safe_float(
            pose_meta.get("known_box_yaw_rad", pose_meta.get("model_box_yaw_rad"))
        )
        if gt_yaw is None:
            gt_yaw = _safe_float(grasp_fields.get("grasp_yaw_rad"), 0.0) or 0.0

        closing_yaw = _safe_float(pose_meta.get("model_closing_yaw_rad"), gt_yaw) or gt_yaw
        short_xy = pose_meta.get("short_axis_xy") or grasp_fields.get("minor_axis_xy")
        long_xy = pose_meta.get("long_axis_xy") or grasp_fields.get("major_axis_xy")
        if not isinstance(short_xy, (list, tuple)) or len(short_xy) < 2:
            short_xy = [float(math.cos(closing_yaw + math.pi / 2.0)), float(math.sin(closing_yaw + math.pi / 2.0))]
        if not isinstance(long_xy, (list, tuple)) or len(long_xy) < 2:
            long_xy = [float(math.cos(gt_yaw)), float(math.sin(gt_yaw))]

        return {
            "gt_entity_name": pose_meta.get("gt_entity_name", ""),
            "top_center_base": top_center_base,
            "top_z_m": float(top_z_m),
            "top_face_corners_base": [
                [float(c[0]), float(c[1]), float(c[2])] for c in corners[:4]
            ],
            "grasp_center_base": grasp_center_base,
            "top_face_source": str(pose_meta.get("top_face_source", "observed")),
            "grasp_center_source": str(
                pose_meta.get("grasp_center_source", "chosen_target_center")
            ),
            "yaw_source": str(pose_meta.get("yaw_source", "pca_raw")),
            "closing_yaw_rad": float(closing_yaw),
            "gt_yaw_rad": float(gt_yaw),
            "long_axis_xy": [float(long_xy[0]), float(long_xy[1])],
            "short_axis_xy": [float(short_xy[0]), float(short_xy[1])],
            "dims_used_lwh": [],
            "bottom_corners_base": list(pose_meta.get("bottom_corners_base") or []),
        }

    def _resolve_operational_top_face(
        self,
        label: str,
        pose_meta: Dict[str, Any],
        grasp_fields: Dict[str, Any],
        *,
        top_face_observed_z: Optional[float] = None,
    ) -> Optional[Dict[str, Any]]:
        """Resuelve cara superior operativa (GT runtime o fallback fit)."""
        if self._use_runtime_scene_gt or self._use_spawn_geometry_for_known_boxes:
            gt_ops = self._resolve_runtime_gt_operational_face(label)
            if gt_ops is not None:
                return gt_ops
        return self._resolve_operational_face_from_pose_meta(
            label,
            pose_meta,
            grasp_fields,
            top_face_observed_z=top_face_observed_z,
        )

    def _apply_runtime_known_box_spawn_geometry(
        self,
        label: str,
        meta: Dict[str, Any],
        grasp_fields: Dict[str, Any],
        center_info: Dict[str, Any],
        policy: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """Top face + grasp operativos desde spawn GT (sin fit RGB-D)."""
        ops = self._resolve_runtime_gt_operational_face(label)
        if ops is None:
            return None
        self._apply_operational_face_to_pose_meta(meta, grasp_fields, center_info, ops)
        req_w = float(policy.get("required_grasp_width_m", 0.0) or 0.0)
        try:
            top_src = str(ops.get("top_face_source", "runtime_gt_known_object"))
            center_src = str(ops.get("grasp_center_source", "runtime_gt_object_center"))
            self.get_logger().info(
                "[KNOWN_OBJECT_GT_GRASP] label=%s top_face_source=%s "
            "grasp_center_source=%s yaw_source=runtime_gt_spawn_yaw "
            "closing_yaw_source=%s closing_yaw_rad=%.4f required_width=%.4f top_z_m=%.4f "
            "dims_lwh=%s entity=%s operational_source_fallback=false"
                % (
                    label,
                    top_src,
                    center_src,
                str(ops.get("closing_yaw_source", "runtime_gt_short_axis")),
                    float(ops["closing_yaw_rad"]),
                    req_w,
                    float(ops["top_z_m"]),
                    ops.get("dims_used_lwh", []),
                    ops.get("gt_entity_name", ""),
                )
            )
        except Exception:
            pass
        return meta

    @staticmethod
    def _project_base_xyz_to_uv(
        xyz: List[float],
        tf_base_to_cam: np.ndarray,
        fx: float,
        fy: float,
        cx: float,
        cy: float,
        w: int,
        h: int,
    ) -> Optional[Tuple[int, int]]:
        hpb = np.array(
            [float(xyz[0]), float(xyz[1]), float(xyz[2]), 1.0], dtype=np.float64
        )
        pc = (tf_base_to_cam @ hpb)[:3]
        if float(pc[2]) <= 1e-6:
            return None
        u = int(np.clip((float(pc[0]) * fx / float(pc[2])) + cx, 0, w - 1))
        v = int(np.clip((float(pc[1]) * fy / float(pc[2])) + cy, 0, h - 1))
        return (u, v)

    def _project_base_points_to_uv(
        self,
        pts_base: np.ndarray,
        tf_base_to_cam: np.ndarray,
        fx: float,
        fy: float,
        cx: float,
        cy: float,
        w: int,
        h: int,
        max_points: int = 4000,
    ) -> np.ndarray:
        if pts_base.size == 0:
            return np.empty((0, 2), dtype=np.int32)
        pts = np.asarray(pts_base, dtype=np.float64).reshape(-1, 3)
        n = int(pts.shape[0])
        if n > max_points:
            idx = np.linspace(0, n - 1, num=max_points, dtype=np.int64)
            pts = pts[idx]
        hom = np.hstack((pts, np.ones((pts.shape[0], 1), dtype=np.float64)))
        cam = (tf_base_to_cam @ hom.T).T[:, :3]
        z = cam[:, 2]
        valid = z > 1e-6
        if not np.any(valid):
            return np.empty((0, 2), dtype=np.int32)
        cam = cam[valid]
        u = (cam[:, 0] * fx / cam[:, 2]) + cx
        v = (cam[:, 1] * fy / cam[:, 2]) + cy
        u = np.clip(u, 0, w - 1).astype(np.int32)
        v = np.clip(v, 0, h - 1).astype(np.int32)
        return np.column_stack((u, v))

    @staticmethod
    def _draw_tag_bgr(
        img: np.ndarray,
        text: str,
        uv: Tuple[int, int],
        color_bgr: Tuple[int, int, int],
        dy: int = -8,
        font_scale: float = 0.42,
    ) -> None:
        x = int(np.clip(uv[0], 0, img.shape[1] - 1))
        y = int(np.clip(uv[1] + dy, 12, img.shape[0] - 4))
        cv2.putText(
            img,
            text,
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (0, 0, 0),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            img,
            text,
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            color_bgr,
            1,
            cv2.LINE_AA,
        )

    @staticmethod
    def _make_fallback_debug_bgr(
        text: str,
        width: int = 640,
        height: int = 480,
        extra_lines: Optional[List[str]] = None,
    ) -> np.ndarray:
        img = np.zeros((int(height), int(width), 3), dtype=np.uint8)
        img[:] = (32, 32, 32)
        lines = [str(text)]
        if extra_lines:
            lines.extend(str(ln) for ln in extra_lines)
        y = 36
        for line in lines:
            cv2.putText(
                img,
                line,
                (12, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.58,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )
            y += 28
        return img

    def _make_heartbeat_fallback_bgr(self) -> np.ndarray:
        return self._make_fallback_debug_bgr(
            "debug heartbeat: perception_node alive",
            extra_lines=[
                f"use_sim_time={str(self._use_sim_time_flag).lower()}",
                "waiting for rgb-d / detections",
            ],
        )

    def _rich_debug_age_sec(self) -> float:
        if self._last_rich_debug_stamp_monotonic <= 0.0:
            return 0.0
        return max(0.0, time.monotonic() - self._last_rich_debug_stamp_monotonic)

    def _cache_rich_debug_bgr(
        self,
        debug_bgr: np.ndarray,
        *,
        label: str,
        reason: str,
        has_overlays: bool,
    ) -> None:
        self._last_rich_debug_bgr = debug_bgr.copy()
        self._last_rich_debug_stamp_monotonic = time.monotonic()
        self._last_rich_debug_label = str(label)
        self._last_rich_debug_reason = str(reason)
        self._last_rich_debug_has_overlays = bool(has_overlays)

    def _cache_and_publish_rich_debug(
        self,
        debug_bgr: np.ndarray,
        camera_frame_id: str,
        num_detections: int,
        valid_objects: int,
        *,
        label: str,
        reason: str,
        has_overlays: bool,
        banner: Optional[str] = None,
    ) -> None:
        self._cache_rich_debug_bgr(
            debug_bgr,
            label=label,
            reason=reason,
            has_overlays=has_overlays,
        )
        self.get_logger().info(
            "[DEBUG_IMAGE_RICH] cached label=%s overlays=%s top_face=%s reason=%s "
            "detections=%d valid_objects=%d"
            % (
                label,
                str(has_overlays).lower(),
                str(has_overlays).lower(),
                reason,
                int(num_detections),
                int(valid_objects),
            )
        )
        self._publish_debug_image_bgr(
            debug_bgr,
            camera_frame_id,
            num_detections,
            valid_objects=valid_objects,
            banner=banner,
            frame_available=True,
            publisher=self.debug_image_pub,
        )

    def _publish_debug_while_waiting_sync(self, reason: str) -> None:
        if not self._publish_debug:
            return
        status_banner = f"waiting rgb-d sync ({reason})"
        if self._debug_publish_every_frame:
            self._maybe_publish_live_rgb_status(
                self._last_image,
                banner=status_banner,
            )
        if self._debug_publish_policy != "last_rich_overlay":
            return
        if self._last_rich_debug_bgr is not None:
            since_pub = time.monotonic() - self._debug_image_last_publish_monotonic
            if since_pub >= 0.5:
                age = self._rich_debug_age_sec()
                self._publish_debug_image_bgr(
                    self._last_rich_debug_bgr.copy(),
                    self._last_image.header.frame_id
                    if self._last_image
                    else self._camera_optical_frame,
                    0,
                    valid_objects=0,
                    banner=f"stale overlay: waiting rgb-d sync age={age:.1f}s",
                    log_call=False,
                    save_disk=False,
                    publisher=self.debug_image_pub,
                )
        else:
            self._publish_debug_status_image_bgr(
                self._make_fallback_debug_bgr(
                    "stale overlay: waiting rgb-d sync",
                    extra_lines=[f"reason={reason}", "no rich cache yet"],
                ),
                self._camera_optical_frame,
                banner=status_banner,
            )

    def _maybe_publish_live_rgb_status(
        self,
        image_msg: Image,
        *,
        banner: str,
    ) -> None:
        if not self._publish_debug:
            return
        if self._debug_publish_policy == "last_rich_overlay":
            try:
                bgr = self._bridge.imgmsg_to_cv2(image_msg, desired_encoding="bgr8")
            except Exception as exc:
                self.get_logger().warn(
                    "[DEBUG_IMAGE] status rgb decode failed: %s" % exc
                )
                bgr = self._make_fallback_debug_bgr("status: rgb decode failed")
            self._publish_debug_status_image_bgr(
                bgr,
                image_msg.header.frame_id or self._camera_optical_frame,
                banner=banner,
            )
            return
        if self._debug_publish_policy == "live_rgb_status":
            self._publish_debug_from_rgb_message(
                image_msg,
                num_detections=0,
                valid_objects=0,
                banner=banner,
                publisher=self.debug_image_pub,
            )

    def _publish_debug_status_image_bgr(
        self,
        debug_bgr: np.ndarray,
        camera_frame_id: str,
        *,
        banner: Optional[str] = None,
    ) -> None:
        if not self._publish_debug or self.debug_status_pub is None:
            return
        self._publish_debug_image_bgr(
            debug_bgr,
            camera_frame_id,
            0,
            valid_objects=0,
            banner=banner,
            frame_available=False,
            log_call=False,
            save_disk=False,
            publisher=self.debug_status_pub,
        )

    def _draw_debug_banner(self, img: np.ndarray, banner: str, y0: int = 28) -> None:
        cv2.putText(
            img,
            str(banner),
            (10, int(y0)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )

    def _save_debug_bgr_to_disk(
        self,
        debug_bgr: np.ndarray,
        *,
        prefix: str = "debug",
    ) -> None:
        if not self._debug_image_save_to_disk:
            return
        try:
            self._debug_image_save_dir.mkdir(parents=True, exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            if prefix.startswith("debug"):
                self._debug_frame_serial += 1
                fname = f"{prefix}_{ts}_frame{self._debug_frame_serial:04d}.png"
            else:
                fname = f"{prefix}_{ts}.png"
            path = self._debug_image_save_dir / fname
            cv2.imwrite(str(path), debug_bgr)
            self.get_logger().info("[DEBUG_IMAGE] saved %s" % path)
        except Exception as exc:
            self.get_logger().warn("[DEBUG_IMAGE] save failed: %s" % exc)

    def _publish_debug_from_rgb_message(
        self,
        image_msg: Image,
        *,
        num_detections: int,
        valid_objects: int,
        banner: Optional[str] = None,
        publisher=None,
    ) -> None:
        try:
            bgr = self._bridge.imgmsg_to_cv2(image_msg, desired_encoding="bgr8")
        except Exception as exc:
            self.get_logger().warn(
                "[DEBUG_IMAGE] failed to publish: rgb decode %s" % exc
            )
            pub = publisher if publisher is not None else self.debug_image_pub
            self._publish_debug_image_bgr(
                self._make_fallback_debug_bgr("debug fallback: rgb decode failed"),
                image_msg.header.frame_id or self._camera_optical_frame,
                0,
                valid_objects=0,
                banner="debug fallback: rgb decode failed",
                publisher=pub,
            )
            return
        frame_id = image_msg.header.frame_id or self._camera_optical_frame
        pub = publisher if publisher is not None else self.debug_image_pub
        self._publish_debug_image_bgr(
            bgr,
            frame_id,
            num_detections,
            valid_objects=valid_objects,
            banner=banner,
            publisher=pub,
        )

    def _publish_debug_image_heartbeat(self) -> None:
        """Heartbeat wall-time: mantiene vivo el topic sin pisar overlays ricos en /vision/debug_image."""
        if not self._publish_debug or not self._debug_heartbeat_enabled:
            return
        policy = self._debug_publish_policy
        since_last_pub = time.monotonic() - self._debug_image_last_publish_monotonic
        refresh_main = since_last_pub >= max(
            0.85, 0.5 / max(0.1, self._debug_heartbeat_hz)
        )
        if self._last_rich_debug_bgr is not None:
            age = self._rich_debug_age_sec()
            if refresh_main and policy in ("last_rich_overlay", "heartbeat_only"):
                self.get_logger().info(
                    "[DEBUG_IMAGE_HEARTBEAT] publishing last rich overlay age=%.1fs"
                    % age
                )
                self._publish_debug_image_bgr(
                    self._last_rich_debug_bgr.copy(),
                    self._camera_optical_frame,
                    0,
                    valid_objects=0,
                    banner=f"stale overlay age={age:.1f}s",
                    log_call=False,
                    save_disk=False,
                    is_heartbeat=True,
                    publisher=self.debug_image_pub,
                )
            elif self._debug_heartbeat_publish_count % 10 == 0:
                self.get_logger().info(
                    "[DEBUG_IMAGE_HEARTBEAT] skip main refresh (recent rich publish %.2fs ago)"
                    % since_last_pub
                )
            self._publish_debug_status_image_bgr(
                self._make_heartbeat_fallback_bgr(),
                self._camera_optical_frame,
                banner=f"heartbeat alive age={age:.1f}s",
            )
            return

        self.get_logger().info(
            "[DEBUG_IMAGE_HEARTBEAT] publishing fallback no rich overlay yet"
        )
        fallback = self._make_fallback_debug_bgr(
            "debug heartbeat: no rich debug frame yet",
            extra_lines=[
                f"use_sim_time={str(self._use_sim_time_flag).lower()}",
                "waiting for rgb-d / detections",
            ],
        )
        self._publish_debug_status_image_bgr(
            fallback,
            self._camera_optical_frame,
            banner="debug heartbeat: no rich debug frame yet",
        )
        if policy in ("live_rgb_status", "heartbeat_only"):
            self._publish_debug_image_bgr(
                fallback,
                self._camera_optical_frame,
                0,
                valid_objects=0,
                log_call=False,
                save_disk=False,
                is_heartbeat=True,
                publisher=self.debug_image_pub,
            )

    def _publish_debug_image_bgr(
        self,
        debug_bgr: np.ndarray,
        camera_frame_id: str,
        num_detections: int,
        *,
        valid_objects: int = 0,
        banner: Optional[str] = None,
        frame_available: bool = True,
        log_call: bool = True,
        save_disk: bool = True,
        is_heartbeat: bool = False,
        publisher=None,
    ) -> None:
        if not self._publish_debug:
            return
        pub = publisher if publisher is not None else self.debug_image_pub
        if log_call:
            self.get_logger().info(
                "[DEBUG_IMAGE_CALL] about_to_publish frame_available=%s detections=%d valid_objects=%d"
                % (
                    str(frame_available).lower(),
                    int(num_detections),
                    int(valid_objects),
                )
            )
        try:
            if debug_bgr is None or debug_bgr.size == 0:
                self.get_logger().warn(
                    "[DEBUG_IMAGE] failed to publish: empty image buffer"
                )
                out = self._make_fallback_debug_bgr("debug fallback: empty buffer")
            elif debug_bgr.ndim != 3 or debug_bgr.shape[2] != 3:
                self.get_logger().warn(
                    "[DEBUG_IMAGE] failed to publish: expected BGR HxWx3 got shape=%s"
                    % (str(debug_bgr.shape),)
                )
                out = self._make_fallback_debug_bgr("debug fallback: invalid shape")
            else:
                out = debug_bgr.copy()
            h, w = int(out.shape[0]), int(out.shape[1])
            if h < 1 or w < 1:
                self.get_logger().warn(
                    "[DEBUG_IMAGE] failed to publish: invalid size %dx%d" % (w, h)
                )
                out = self._make_fallback_debug_bgr("debug fallback: invalid size")
                h, w = int(out.shape[0]), int(out.shape[1])
            if banner:
                self._draw_debug_banner(out, banner)
            debug_msg = self._bridge.cv2_to_imgmsg(out, encoding="bgr8")
            debug_msg.header.stamp = self.get_clock().now().to_msg()
            debug_msg.header.frame_id = (
                str(camera_frame_id or self._camera_optical_frame).strip()
                or self._camera_optical_frame
            )
            pub.publish(debug_msg)
            self._debug_publish_count += 1
            self._debug_image_last_publish_monotonic = time.monotonic()
            if save_disk:
                self._save_debug_bgr_to_disk(out, prefix="debug")
            if is_heartbeat:
                self._debug_heartbeat_publish_count += 1
                if self._debug_heartbeat_publish_count % 10 == 0:
                    self.get_logger().info(
                        "[DEBUG_IMAGE_HEARTBEAT] published fallback count=%d hz=%.3f steady_time=true"
                        % (
                            self._debug_heartbeat_publish_count,
                            self._debug_heartbeat_hz,
                        )
                    )
            elif self._debug_publish_count % self._debug_image_log_every_n == 0:
                self.get_logger().info(
                    "[DEBUG_IMAGE] published count=%d topic=%s encoding=bgr8 "
                    "size=%dx%d detections=%d valid_objects=%d"
                    % (
                        self._debug_publish_count,
                        self._debug_image_topic,
                        w,
                        h,
                        int(num_detections),
                        int(valid_objects),
                    )
                )
        except Exception as exc:
            self.get_logger().warn("[DEBUG_IMAGE] failed to publish: %s" % exc)

    def _resolve_synthetic_operational_geometry_for_overlay(
        self,
        label: str,
        pose_meta: Dict[str, Any],
        grasp_fields: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """Top face para overlay: dims_lwh reales (sin inflado); semántica + yaw + ejes locales."""
        gt_entry: Optional[Dict[str, Any]] = None
        if (
            self._use_runtime_scene_gt
            and self._gt_spawner_by_entity
            and is_known_spawn_geometry_box_label(label)
        ):
            found = find_gt_object_for_label(
                self._gt_spawner_by_entity,
                label,
                entity_prefix=self._debug_gt_entity_prefix,
            )
            if found is not None:
                _entity, gt_entry = found
                twb = self._lookup_world_to_base_matrix()
                if twb is not None:
                    enrich_gt_object_entry_base(gt_entry, twb, logger=self.get_logger())
                sem = (
                    gt_entry.get("semantic_box_center_base")
                    or gt_entry.get("gt_geometry_center_base")
                )
                yaw = gt_entry.get("gt_yaw_rad", gt_entry.get("yaw_rad"))
                if (
                    isinstance(sem, (list, tuple))
                    and len(sem) >= 3
                    and yaw is not None
                ):
                    return compute_synthetic_operational_top_face_base(
                        (float(sem[0]), float(sem[1]), float(sem[2])),
                        float(yaw),
                        label=str(label),
                        apply_yaw_offset=False,
                        for_overlay=True,
                        logger=self.get_logger(),
                    )

        yaw = _safe_float(
            pose_meta.get("known_box_yaw_rad", pose_meta.get("model_box_yaw_rad"))
        )
        if yaw is None:
            yaw = _safe_float(grasp_fields.get("grasp_yaw_rad"), 0.0) or 0.0

        dims_lwh: Optional[Tuple[float, float, float]] = None
        dl = _safe_float(pose_meta.get("db_length_m"))
        dw = _safe_float(pose_meta.get("db_width_m"))
        dh = _safe_float(grasp_fields.get("measured_height_m"))
        if dl is not None and dw is not None:
            if dh is None or dh <= 0.0:
                dims_gf = grasp_fields.get("dimensions_m")
                if isinstance(dims_gf, (list, tuple)) and len(dims_gf) >= 3:
                    dh = float(dims_gf[2])
            if dh is not None and dh > 0.0:
                dims_lwh = (float(dl), float(dw), float(dh))
        if dims_lwh is None:
            spec = get_known_box_gt_spec(label)
            if spec is not None:
                dims_lwh = spec.dims_lwh_m
        if dims_lwh is None:
            return None

        sem: Optional[List[float]] = None
        for key in ("semantic_box_center_base", "gt_geometry_center_base"):
            v = pose_meta.get(key)
            if isinstance(v, (list, tuple)) and len(v) >= 3:
                sem = [float(v[0]), float(v[1]), float(v[2])]
                break
        if sem is None:
            kbc = pose_meta.get("known_box_center_base")
            top_z = _safe_float(pose_meta.get("top_z_estimated"), grasp_fields.get("top_z_m"))
            if (
                isinstance(kbc, (list, tuple))
                and len(kbc) >= 3
                and top_z is not None
            ):
                sem = [
                    float(kbc[0]),
                    float(kbc[1]),
                    float(top_z) - 0.5 * float(dims_lwh[2]),
                ]
        if sem is None:
            return None

        return compute_synthetic_operational_top_face_base(
            (float(sem[0]), float(sem[1]), float(sem[2])),
            float(yaw),
            dims_lwh,
            label=str(label),
            apply_yaw_offset=not is_known_spawn_geometry_box_label(label),
            for_overlay=True,
            logger=self.get_logger(),
        )

    def _draw_detection_yolo_demo(self, debug: np.ndarray, det: Any) -> None:
        """YOLO bbox/OBB + etiqueta (solo demostración de detección)."""
        h, w = debug.shape[:2]
        col_bbox = (0, 255, 255)
        col_obb = (0, 165, 255)
        anchor_uv: Optional[Tuple[int, int]] = None
        if det.bbox_xyxy is not None:
            x1, y1, x2, y2 = [int(v) for v in det.bbox_xyxy]
            x1 = int(np.clip(x1, 0, w - 1))
            x2 = int(np.clip(x2, 0, w - 1))
            y1 = int(np.clip(y1, 0, h - 1))
            y2 = int(np.clip(y2, 0, h - 1))
            cv2.rectangle(debug, (x1, y1), (x2, y2), col_bbox, 1, cv2.LINE_AA)
            anchor_uv = (x1, max(18, y1))
        if det.obb_polygon_uv is not None:
            poly = np.round(det.obb_polygon_uv).astype(np.int32)
            cv2.polylines(debug, [poly], True, col_obb, 2, cv2.LINE_AA)
            ou = int(np.clip(np.min(det.obb_polygon_uv[:, 0]), 0, w - 1))
            ov = int(np.clip(np.min(det.obb_polygon_uv[:, 1]), 0, h - 1))
            anchor_uv = (ou, max(18, ov))
        if anchor_uv is not None:
            self._draw_tag_bgr(
                debug,
                "YOLO_RAW %s %.2f" % (str(det.label), float(det.score)),
                anchor_uv,
                col_obb,
                dy=-6,
                font_scale=0.5,
            )

    def _draw_grasp_debug_overlay_demo_simplified(
        self,
        debug: np.ndarray,
        det: Any,
        tf_msg: Any,
        fx: float,
        fy: float,
        cx: float,
        cy: float,
        w: int,
        h: int,
        pose_meta: Dict[str, Any],
        grasp_fields: Dict[str, Any],
        closing_yaw_rad: Optional[float],
    ) -> None:
        """Demo: YOLO + top face sintética operativa + centro + eje de cierre."""
        col_oper_top = (0, 255, 0)
        col_oper_ctr = (0, 255, 255)
        col_close = (0, 140, 255)

        self._draw_detection_yolo_demo(debug, det)

        tf_base_to_cam: Optional[np.ndarray] = None
        if tf_msg is not None:
            try:
                tf_cam_to_base = self._build_transform_matrix(tf_msg)
                tf_base_to_cam = np.linalg.inv(tf_cam_to_base)
            except np.linalg.LinAlgError:
                tf_base_to_cam = None

        synth = self._resolve_synthetic_operational_geometry_for_overlay(
            str(det.label), pose_meta, grasp_fields
        )
        if synth is None or tf_base_to_cam is None:
            if tf_base_to_cam is None:
                self._draw_tag_bgr(
                    debug,
                    "TF unavailable",
                    (12, 40),
                    (0, 128, 255),
                    dy=0,
                    font_scale=0.45,
                )
            return

        corners = synth.get("top_face_corners_base")
        grasp_cb = synth.get("grasp_center_base")
        close_yaw = _safe_float(synth.get("closing_yaw_rad"), closing_yaw_rad)

        # Cuadrilátero verde: 4 esquinas 3D en base → TF a cámara → pinhole (CameraInfo).
        if isinstance(corners, list) and len(corners) >= 4:
            oper_uv: List[Tuple[int, int]] = []
            for c in corners[:4]:
                if not isinstance(c, (list, tuple)) or len(c) < 3:
                    continue
                puv = self._project_base_xyz_to_uv(
                    [float(c[0]), float(c[1]), float(c[2])],
                    tf_base_to_cam,
                    fx,
                    fy,
                    cx,
                    cy,
                    w,
                    h,
                )
                if puv is not None:
                    oper_uv.append(puv)
            if len(oper_uv) >= 3:
                poly = np.array(oper_uv, dtype=np.int32).reshape(-1, 1, 2)
                cv2.polylines(debug, [poly], True, col_oper_top, 3, cv2.LINE_AA)
            if oper_uv:
                log_synthetic_top_face_overlay_projection(
                    self.get_logger(),
                    label=str(det.label),
                    synth=synth,
                    projected_pixels=oper_uv,
                )

        if isinstance(grasp_cb, (list, tuple)) and len(grasp_cb) >= 3:
            ctr_uv = self._project_base_xyz_to_uv(
                [float(grasp_cb[0]), float(grasp_cb[1]), float(grasp_cb[2])],
                tf_base_to_cam,
                fx,
                fy,
                cx,
                cy,
                w,
                h,
            )
            if ctr_uv is not None:
                cv2.circle(debug, ctr_uv, 7, col_oper_ctr, 2, cv2.LINE_AA)
                self._draw_grasp_gap_and_pad_axes_overlay(
                    debug,
                    label=str(det.label),
                    pose_meta=pose_meta,
                    axis_anchor_b=[
                        float(grasp_cb[0]),
                        float(grasp_cb[1]),
                        float(grasp_cb[2]),
                    ],
                    tf_base_to_cam=tf_base_to_cam,
                    fx=fx,
                    fy=fy,
                    cx=cx,
                    cy=cy,
                    w=w,
                    h=h,
                    closing_yaw_rad=close_yaw,
                    col_pad=(0, 140, 255),
                    col_gap=(255, 220, 60),
                )

    def _draw_detection_uv_preview(
        self,
        debug: np.ndarray,
        det: Any,
        det_index: int,
    ) -> None:
        """Overlays minimos en UV (YOLO/OBB/centros) sin depender de TF."""
        h, w = debug.shape[:2]
        col_yolo = (0, 255, 255)
        col_obb = (0, 165, 255)
        if det.bbox_xyxy is not None:
            x1, y1, x2, y2 = [int(v) for v in det.bbox_xyxy]
            x1 = int(np.clip(x1, 0, w - 1))
            x2 = int(np.clip(x2, 0, w - 1))
            y1 = int(np.clip(y1, 0, h - 1))
            y2 = int(np.clip(y2, 0, h - 1))
            cv2.rectangle(debug, (x1, y1), (x2, y2), col_yolo, 2, cv2.LINE_AA)
            self._draw_tag_bgr(debug, "YOLO_RAW", (x1, max(18, y1)), col_yolo, dy=-8)
        if det.obb_polygon_uv is not None:
            poly = np.round(det.obb_polygon_uv).astype(np.int32)
            cv2.polylines(debug, [poly], True, col_obb, 2, cv2.LINE_AA)
            ou = int(np.clip(np.mean(det.obb_polygon_uv[:, 0]), 0, w - 1))
            ov = int(np.clip(np.mean(det.obb_polygon_uv[:, 1]), 0, h - 1))
            cv2.circle(debug, (ou, ov), 5, col_obb, 2, cv2.LINE_AA)
            self._draw_tag_bgr(debug, "YOLO_RAW obb_center", (ou, ov), col_obb, dy=-8)

    def _draw_detection_uv_clean(self, debug: np.ndarray, det: Any) -> None:
        """YOLO bbox + OBB sin etiquetas (modo grasp_clean)."""
        h, w = debug.shape[:2]
        col_yolo = (0, 255, 255)
        col_obb = (0, 165, 255)
        if det.bbox_xyxy is not None:
            x1, y1, x2, y2 = [int(v) for v in det.bbox_xyxy]
            x1 = int(np.clip(x1, 0, w - 1))
            x2 = int(np.clip(x2, 0, w - 1))
            y1 = int(np.clip(y1, 0, h - 1))
            y2 = int(np.clip(y2, 0, h - 1))
            cv2.rectangle(debug, (x1, y1), (x2, y2), col_yolo, 2, cv2.LINE_AA)
        if det.obb_polygon_uv is not None:
            poly = np.round(det.obb_polygon_uv).astype(np.int32)
            cv2.polylines(debug, [poly], True, col_obb, 2, cv2.LINE_AA)

    def _draw_grasp_debug_overlay_clean(
        self,
        debug: np.ndarray,
        det: Any,
        tf_msg: Any,
        fx: float,
        fy: float,
        cx: float,
        cy: float,
        w: int,
        h: int,
        grasp_fields: Dict[str, Any],
        center_info: Dict[str, Any],
        pose_meta: Dict[str, Any],
        closing_yaw_rad: Optional[float],
        pts_obj_base: np.ndarray,
        grasp_center_base: Optional[List[float]],
        *,
        chosen_center_uv: Optional[List[float]] = None,
        grasp_center_uv: Optional[List[float]] = None,
        obb_center_uv: Optional[List[float]] = None,
    ) -> None:
        """Overlay minimo: mesa/caja/ejes/cierre legibles; texto compacto en esquina."""
        col_observed_top = (200, 200, 180)
        col_model_top = (0, 255, 0)
        col_gt_top = (255, 0, 255)
        col_major = (255, 255, 255)
        col_gap_body = (255, 220, 60)
        col_pad_arrow = (0, 140, 255)
        col_gap_arrow = (255, 220, 60)
        col_grasp = (0, 220, 0)
        col_model_ctr = (255, 0, 255)
        gt_cmp_line = ""

        self._draw_detection_uv_clean(debug, det)

        tf_base_to_cam: Optional[np.ndarray] = None
        if tf_msg is not None:
            try:
                tf_cam_to_base = self._build_transform_matrix(tf_msg)
                tf_base_to_cam = np.linalg.inv(tf_cam_to_base)
            except np.linalg.LinAlgError:
                tf_base_to_cam = None

        if (
            tf_base_to_cam is not None
            and pts_obj_base.size > 0
            and bool(pose_meta.get("observed_top_face_success", False))
        ):
            tf_top = extract_top_face_points(
                pts_obj_base,
                top_z_m=float(grasp_fields.get("top_z_m") or 0.0),
            )
            tp = tf_top.get("top_points_base")
            if isinstance(tp, np.ndarray) and tp.size > 0:
                uvs = self._project_base_points_to_uv(
                    tp, tf_base_to_cam, fx, fy, cx, cy, w, h, max_points=2000
                )
                if uvs.shape[0] >= 3:
                    hull = cv2.convexHull(uvs.astype(np.float32))
                    cv2.polylines(
                        debug,
                        [hull.astype(np.int32)],
                        True,
                        col_observed_top,
                        1,
                        cv2.LINE_AA,
                    )
                    if len(hull) >= 1:
                        p0 = tuple(hull[0].ravel().astype(int))
                        self._draw_tag_bgr(debug, "observed", p0, col_observed_top, dy=-10)

        model_tc = pose_meta.get("model_top_face_corners_base")
        if not (isinstance(model_tc, list) and len(model_tc) >= 4):
            if str(pose_meta.get("top_face_source", "")) in (
                "known_model",
                "hybrid_known_model",
            ):
                model_tc = pose_meta.get("top_corners_base")
        if tf_base_to_cam is not None and isinstance(model_tc, list) and len(model_tc) >= 4:
            model_uv: List[Tuple[int, int]] = []
            for c in model_tc[:4]:
                if not isinstance(c, (list, tuple)) or len(c) < 3:
                    continue
                puv = self._project_base_xyz_to_uv(
                    [float(c[0]), float(c[1]), float(c[2])],
                    tf_base_to_cam,
                    fx,
                    fy,
                    cx,
                    cy,
                    w,
                    h,
                )
                if puv is not None:
                    model_uv.append(puv)
            if len(model_uv) >= 3:
                poly_m = np.array(model_uv, dtype=np.int32).reshape(-1, 1, 2)
                cv2.polylines(
                    debug, [poly_m], True, col_model_top, 3, cv2.LINE_AA
                )
                self._draw_tag_bgr(debug, "model", model_uv[0], col_model_top, dy=-10)

        col_obs_ctr = (200, 220, 255)
        col_gt_ctr = (255, 0, 255)

        def _draw_center_from_corners(
            corners: Optional[List[List[float]]],
            color: Tuple[int, int, int],
            tag: str,
        ) -> None:
            if tf_base_to_cam is None or not corners or len(corners) < 4:
                return
            cxys = [
                [float(c[0]), float(c[1]), float(c[2])]
                for c in corners[:4]
                if isinstance(c, (list, tuple)) and len(c) >= 3
            ]
            if len(cxys) < 4:
                return
            center_b = [
                float(np.mean([p[0] for p in cxys])),
                float(np.mean([p[1] for p in cxys])),
                float(np.mean([p[2] for p in cxys])),
            ]
            uv = self._project_base_xyz_to_uv(
                center_b, tf_base_to_cam, fx, fy, cx, cy, w, h
            )
            if uv is not None:
                cv2.circle(debug, uv, 7, color, 2, cv2.LINE_AA)
                self._draw_tag_bgr(debug, tag, uv, color, dy=16)

        obs_tc_ov = pose_meta.get("top_corners_base")
        if str(pose_meta.get("top_face_source", "")).strip() == "runtime_gt_known_box":
            obs_tc_ov = None
        if isinstance(obs_tc_ov, list) and len(obs_tc_ov) >= 4:
            _draw_center_from_corners(obs_tc_ov, col_obs_ctr, "obs_ctr")

        if isinstance(model_tc, list) and len(model_tc) >= 4:
            _draw_center_from_corners(model_tc, col_model_ctr, "model_ctr")

        gt_corners_base: Optional[List[List[float]]] = None
        gt_entity_name = ""
        if self._use_runtime_scene_gt or self._debug_draw_gazebo_gt:
            gt_corners_base, gt_entity_name, _ = self._resolve_gt_top_face_for_label(
                str(det.label)
            )
            if (
                gt_corners_base
                and tf_base_to_cam is not None
                and len(gt_corners_base) >= 4
            ):
                gt_uv: List[Tuple[int, int]] = []
                for c in gt_corners_base[:4]:
                    puv = self._project_base_xyz_to_uv(
                        c,
                        tf_base_to_cam,
                        fx,
                        fy,
                        cx,
                        cy,
                        w,
                        h,
                    )
                    if puv is not None:
                        gt_uv.append(puv)
                if len(gt_uv) >= 3:
                    poly_gt = np.array(gt_uv, dtype=np.int32).reshape(-1, 1, 2)
                    cv2.polylines(
                        debug,
                        [poly_gt],
                        True,
                        col_gt_top,
                        2,
                        cv2.LINE_AA,
                    )
                    self._draw_tag_bgr(debug, "RUNTIME_GT", gt_uv[0], col_gt_top, dy=14)
                _draw_center_from_corners(gt_corners_base, col_gt_ctr, "gt_ctr")

        if pose_meta.get("model_vs_gt_center_error_xy_m") is not None:
            gt_cmp_line = (
                "gt_cmp: yaw=%.1fdeg ctr=%.3fm gate=%s"
                % (
                    float(pose_meta.get("model_vs_gt_yaw_error_deg", float("nan"))),
                    float(pose_meta.get("model_vs_gt_center_error_xy_m", float("nan"))),
                    str(pose_meta.get("visual_pose_gate_passed", "n/a")),
                )
            )

        axis_anchor_b: Optional[List[float]] = None
        if isinstance(grasp_center_base, (list, tuple)) and len(grasp_center_base) >= 3:
            axis_anchor_b = [
                float(grasp_center_base[0]),
                float(grasp_center_base[1]),
                float(grasp_center_base[2]),
            ]
        else:
            ct = center_info.get("chosen_target_center_base")
            if isinstance(ct, (list, tuple)) and len(ct) >= 3:
                axis_anchor_b = [float(ct[0]), float(ct[1]), float(ct[2])]

        if tf_base_to_cam is not None and axis_anchor_b is not None:
            _top_src_ax = str(pose_meta.get("top_face_source", "")).strip()
            if _top_src_ax in (
                "runtime_gt_known_box",
                "runtime_gt_tall_object",
                "runtime_gt_known_object",
            ):
                maj = pose_meta.get("long_axis_xy") or grasp_fields.get("major_axis_xy")
                minr = pose_meta.get("short_axis_xy") or grasp_fields.get("minor_axis_xy")
            elif bool(pose_meta.get("model_top_face_success", False)):
                maj = pose_meta.get("model_major_axis_xy") or grasp_fields.get(
                    "major_axis_xy"
                )
                minr = pose_meta.get("model_minor_axis_xy") or grasp_fields.get(
                    "minor_axis_xy"
                )
            else:
                maj = grasp_fields.get("major_axis_xy")
                minr = grasp_fields.get("minor_axis_xy")
            axis_len_m = 0.10
            cb = [float(axis_anchor_b[0]), float(axis_anchor_b[1]), float(axis_anchor_b[2])]
            if isinstance(maj, (list, tuple)) and len(maj) >= 2:
                mx, my = float(maj[0]), float(maj[1])
                nrm = math.hypot(mx, my)
                if nrm > 1e-9:
                    mx, my = mx / nrm, my / nrm
                    p0 = self._project_base_xyz_to_uv(
                        cb, tf_base_to_cam, fx, fy, cx, cy, w, h
                    )
                    p1b = [cb[0] + axis_len_m * mx, cb[1] + axis_len_m * my, cb[2]]
                    p1 = self._project_base_xyz_to_uv(
                        p1b, tf_base_to_cam, fx, fy, cx, cy, w, h
                    )
                    if p0 is not None and p1 is not None:
                        cv2.line(debug, p0, p1, col_major, 2, cv2.LINE_AA)
                        self._draw_tag_bgr(debug, "body_long", p1, col_major, dy=-6)
            if isinstance(minr, (list, tuple)) and len(minr) >= 2:
                sx, sy = float(minr[0]), float(minr[1])
                nrm = math.hypot(sx, sy)
                if nrm > 1e-9:
                    sx, sy = sx / nrm, sy / nrm
                    p0 = self._project_base_xyz_to_uv(
                        cb, tf_base_to_cam, fx, fy, cx, cy, w, h
                    )
                    p1b = [cb[0] + axis_len_m * sx, cb[1] + axis_len_m * sy, cb[2]]
                    p1 = self._project_base_xyz_to_uv(
                        p1b, tf_base_to_cam, fx, fy, cx, cy, w, h
                    )
                    if p0 is not None and p1 is not None:
                        cv2.line(debug, p0, p1, col_gap_body, 1, cv2.LINE_AA)

            mbc = pose_meta.get("model_box_center_base")
            if not (isinstance(mbc, (list, tuple)) and len(mbc) >= 3):
                mbc = pose_meta.get("known_box_center_base")
            if isinstance(mbc, (list, tuple)) and len(mbc) >= 3:
                muv = self._project_base_xyz_to_uv(
                    [float(mbc[0]), float(mbc[1]), float(mbc[2])],
                    tf_base_to_cam,
                    fx,
                    fy,
                    cx,
                    cy,
                    w,
                    h,
                )
                if muv is not None:
                    cv2.circle(debug, muv, 6, col_model_ctr, 2, cv2.LINE_AA)

            self._draw_grasp_gap_and_pad_axes_overlay(
                debug,
                label=str(det.label),
                pose_meta=pose_meta,
                axis_anchor_b=cb,
                tf_base_to_cam=tf_base_to_cam,
                fx=fx,
                fy=fy,
                cx=cx,
                cy=cy,
                w=w,
                h=h,
                closing_yaw_rad=closing_yaw_rad,
                col_pad=col_pad_arrow,
                col_gap=col_gap_arrow,
            )

        col_body_ctr = (200, 200, 120)
        body_ctr_b = pose_meta.get("tall_object_body_center_base")
        if (
            tf_base_to_cam is not None
            and isinstance(body_ctr_b, (list, tuple))
            and len(body_ctr_b) >= 3
        ):
            body_uv = self._project_base_xyz_to_uv(
                [float(body_ctr_b[0]), float(body_ctr_b[1]), float(body_ctr_b[2])],
                tf_base_to_cam,
                fx,
                fy,
                cx,
                cy,
                w,
                h,
            )
            if body_uv is not None:
                cv2.circle(debug, body_uv, 6, col_body_ctr, 2, cv2.LINE_AA)
                self._draw_tag_bgr(debug, "body_ctr", body_uv, col_body_ctr, dy=-12)

        if tf_base_to_cam is not None:
            old_off_b = pose_meta.get("mustard_old_offset_cap_center_base")
            vert_cap_b = pose_meta.get("mustard_vertical_axis_cap_center_base")
            if isinstance(old_off_b, (list, tuple)) and len(old_off_b) >= 3:
                ouv_old = self._project_base_xyz_to_uv(
                    [float(old_off_b[0]), float(old_off_b[1]), float(old_off_b[2])],
                    tf_base_to_cam,
                    fx,
                    fy,
                    cx,
                    cy,
                    w,
                    h,
                )
                if ouv_old is not None:
                    cv2.circle(debug, ouv_old, 10, (0, 220, 255), 2, cv2.LINE_AA)
                    self._draw_tag_bgr(
                        debug,
                        "cap_offset_old",
                        ouv_old,
                        (0, 220, 255),
                        dy=-24,
                        font_scale=0.38,
                    )
            if isinstance(vert_cap_b, (list, tuple)) and len(vert_cap_b) >= 3:
                vuv = self._project_base_xyz_to_uv(
                    [float(vert_cap_b[0]), float(vert_cap_b[1]), float(vert_cap_b[2])],
                    tf_base_to_cam,
                    fx,
                    fy,
                    cx,
                    cy,
                    w,
                    h,
                )
                if vuv is not None:
                    cv2.circle(debug, vuv, 12, (0, 0, 255), 2, cv2.LINE_AA)
                    cv2.drawMarker(
                        debug,
                        vuv,
                        (0, 0, 255),
                        markerType=cv2.MARKER_CROSS,
                        markerSize=18,
                        thickness=2,
                    )
                    self._draw_tag_bgr(
                        debug,
                        "cap_vertical_axis",
                        vuv,
                        (0, 0, 255),
                        dy=-24,
                        font_scale=0.38,
                    )
            mesh_cap_b = pose_meta.get("mustard_mesh_local_cap_center_base")
            if isinstance(mesh_cap_b, (list, tuple)) and len(mesh_cap_b) >= 3:
                muv = self._project_base_xyz_to_uv(
                    [float(mesh_cap_b[0]), float(mesh_cap_b[1]), float(mesh_cap_b[2])],
                    tf_base_to_cam,
                    fx,
                    fy,
                    cx,
                    cy,
                    w,
                    h,
                )
                if muv is not None:
                    cv2.circle(debug, muv, 11, (255, 0, 255), 2, cv2.LINE_AA)
                    cv2.drawMarker(
                        debug,
                        muv,
                        (255, 0, 255),
                        markerType=cv2.MARKER_TILTED_CROSS,
                        markerSize=16,
                        thickness=2,
                    )
                    self._draw_tag_bgr(
                        debug,
                        "cap_mesh_local",
                        muv,
                        (255, 0, 255),
                        dy=22,
                        font_scale=0.38,
                    )
            gcb_txt = grasp_center_base
            gcs_ov_cap = str(pose_meta.get("grasp_center_source", "")).strip()
            if (
                isinstance(gcb_txt, (list, tuple))
                and len(gcb_txt) >= 3
                and gcs_ov_cap
            ):
                guv_cap = self._project_base_xyz_to_uv(
                    [float(gcb_txt[0]), float(gcb_txt[1]), float(gcb_txt[2])],
                    tf_base_to_cam,
                    fx,
                    fy,
                    cx,
                    cy,
                    w,
                    h,
                )
                if guv_cap is not None:
                    self._draw_tag_bgr(
                        debug,
                        "%s (%.3f,%.3f,%.3f)"
                        % (
                            gcs_ov_cap,
                            float(gcb_txt[0]),
                            float(gcb_txt[1]),
                            float(gcb_txt[2]),
                        ),
                        guv_cap,
                        (220, 220, 220),
                        dy=40,
                        font_scale=0.36,
                    )
        if (
            tf_base_to_cam is not None
            and bool(pose_meta.get("mustard_sdf_correction_applied"))
        ):
            mob = pose_meta.get("mustard_sdf_model_origin_base")
            mcb = pose_meta.get("mustard_sdf_cap_center_base")
            col_origin = (180, 180, 60)
            col_cap = (60, 220, 255)
            ouv = None
            if isinstance(mob, (list, tuple)) and len(mob) >= 3:
                ouv = self._project_base_xyz_to_uv(
                    [float(mob[0]), float(mob[1]), float(mob[2])],
                    tf_base_to_cam,
                    fx,
                    fy,
                    cx,
                    cy,
                    w,
                    h,
                )
                if ouv is not None:
                    cv2.circle(debug, ouv, 5, col_origin, 1, cv2.LINE_AA)
                    self._draw_tag_bgr(
                        debug, "runtime origin", ouv, col_origin, dy=-20, font_scale=0.38
                    )
            if isinstance(mcb, (list, tuple)) and len(mcb) >= 3:
                cuv = self._project_base_xyz_to_uv(
                    [float(mcb[0]), float(mcb[1]), float(mcb[2])],
                    tf_base_to_cam,
                    fx,
                    fy,
                    cx,
                    cy,
                    w,
                    h,
                )
                if cuv is not None:
                    cv2.circle(debug, cuv, 9, col_cap, 2, cv2.LINE_AA)
                    self._draw_tag_bgr(
                        debug,
                        "mustard_sdf_cap_center",
                        cuv,
                        col_cap,
                        dy=22,
                        font_scale=0.40,
                    )
                    if ouv is not None:
                        cv2.arrowedLine(
                            debug,
                            ouv,
                            cuv,
                            col_cap,
                            2,
                            tipLength=0.25,
                            line_type=cv2.LINE_AA,
                        )

        if grasp_center_uv is not None:
            gu = int(np.clip(float(grasp_center_uv[0]), 0, w - 1))
            gv = int(np.clip(float(grasp_center_uv[1]), 0, h - 1))
            cv2.circle(debug, (gu, gv), 8, col_grasp, 2, cv2.LINE_AA)
            gcs_ov = str(pose_meta.get("grasp_center_source", "")).strip()
            if gcs_ov:
                self._draw_tag_bgr(debug, gcs_ov, (gu, gv), col_grasp, dy=14)
        elif (
            tf_base_to_cam is not None
            and isinstance(grasp_center_base, (list, tuple))
            and len(grasp_center_base) >= 3
        ):
            guv = self._project_base_xyz_to_uv(
                [float(grasp_center_base[0]), float(grasp_center_base[1]), float(grasp_center_base[2])],
                tf_base_to_cam,
                fx,
                fy,
                cx,
                cy,
                w,
                h,
            )
            if guv is not None:
                cv2.circle(debug, guv, 8, col_grasp, 2, cv2.LINE_AA)
                gcs_ov = str(pose_meta.get("grasp_center_source", "")).strip()
                if gcs_ov:
                    self._draw_tag_bgr(debug, gcs_ov, guv, col_grasp, dy=14)

        mfe = pose_meta.get("model_fit_error")
        mfe_s = "n/a" if mfe is None else f"{float(mfe):.4f}"
        close_deg = (
            "n/a"
            if closing_yaw_rad is None
            else f"{math.degrees(float(closing_yaw_rad)):.0f}"
        )
        lines = [
            f"{det.label} {float(det.score):.2f}",
            f"top_src={pose_meta.get('top_face_source', '')}",
            f"yaw={pose_meta.get('yaw_source', '')} conf={float(pose_meta.get('yaw_confidence', 0.0)):.2f}",
            f"model_err={mfe_s}",
            f"close_yaw={close_deg}",
        ]
        if gt_cmp_line:
            lines.append(gt_cmp_line)
        x0, y0 = 8, 22
        for i, line in enumerate(lines[:6]):
            y = y0 + i * 18
            cv2.putText(
                debug,
                line,
                (x0, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                (0, 0, 0),
                3,
                cv2.LINE_AA,
            )
            cv2.putText(
                debug,
                line,
                (x0, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

        if self._debug_draw_gazebo_gt:
            leg_x = max(8, w - 168)
            leg_y = h - 78
            legend = [
                ("green=model", (0, 255, 0)),
                ("purple=GT", col_gt_top),
                ("gray=observed", col_observed_top),
                ("orange=pad/finger", (0, 140, 255)),
                ("cyan=gap/close", (255, 220, 60)),
            ]
            for li, (txt, col) in enumerate(legend):
                ly = leg_y + li * 16
                cv2.putText(
                    debug,
                    txt,
                    (leg_x, ly),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.42,
                    (0, 0, 0),
                    2,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    debug,
                    txt,
                    (leg_x, ly),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.42,
                    col,
                    1,
                    cv2.LINE_AA,
                )

    def _resolve_overlay_pad_axis_xy(
        self, pose_meta: Dict[str, Any]
    ) -> Tuple[Optional[List[float]], str]:
        for key, src in (
            ("finger_pad_axis_xy", "finger_pad_axis_xy"),
            ("overlay_pad_axis_xy", "overlay_pad_axis_xy"),
            ("long_axis_xy", "long_axis_xy"),
            ("major_axis_xy", "major_axis_xy"),
        ):
            v = pose_meta.get(key)
            if isinstance(v, (list, tuple)) and len(v) >= 2:
                return [float(v[0]), float(v[1])], src
        pyaw = pose_meta.get("finger_pad_yaw_rad")
        if pyaw is not None:
            try:
                y = float(pyaw)
                return [float(math.cos(y)), float(math.sin(y))], "finger_pad_yaw_rad"
            except (TypeError, ValueError):
                pass
        return None, "missing"

    def _resolve_overlay_gap_axis_xy(
        self, pose_meta: Dict[str, Any], closing_yaw_rad: Optional[float]
    ) -> Tuple[Optional[List[float]], str]:
        for key, src in (
            ("grasp_gap_axis_xy", "grasp_gap_axis_xy"),
            ("overlay_gap_axis_xy", "overlay_gap_axis_xy"),
            ("short_axis_xy", "short_axis_xy"),
            ("minor_axis_xy", "minor_axis_xy"),
        ):
            v = pose_meta.get(key)
            if isinstance(v, (list, tuple)) and len(v) >= 2:
                return [float(v[0]), float(v[1])], src
        gyaw = pose_meta.get("grasp_gap_yaw_rad")
        if gyaw is not None:
            try:
                y = float(gyaw)
                return [float(math.cos(y)), float(math.sin(y))], "grasp_gap_yaw_rad"
            except (TypeError, ValueError):
                pass
        if closing_yaw_rad is not None:
            y = float(closing_yaw_rad)
            return [float(math.cos(y)), float(math.sin(y))], "closing_yaw_rad_fallback"
        return None, "missing"

    def _draw_grasp_gap_and_pad_axes_overlay(
        self,
        debug: np.ndarray,
        *,
        label: str = "",
        pose_meta: Dict[str, Any],
        axis_anchor_b: List[float],
        tf_base_to_cam: np.ndarray,
        fx: float,
        fy: float,
        cx: float,
        cy: float,
        w: int,
        h: int,
        closing_yaw_rad: Optional[float],
        col_pad: Tuple[int, int, int],
        col_gap: Tuple[int, int, int],
    ) -> None:
        """Naranja = dedos/pads (eje largo); cian = gap/cierre (ancho corto)."""
        cb = [float(axis_anchor_b[0]), float(axis_anchor_b[1]), float(axis_anchor_b[2])]
        p0 = self._project_base_xyz_to_uv(cb, tf_base_to_cam, fx, fy, cx, cy, w, h)
        if p0 is None:
            return
        L = 0.09

        pad_axis, pad_src = self._resolve_overlay_pad_axis_xy(pose_meta)
        gap_axis, gap_src = self._resolve_overlay_gap_axis_xy(
            pose_meta, closing_yaw_rad
        )

        if isinstance(pad_axis, (list, tuple)) and len(pad_axis) >= 2:
            px, py = float(pad_axis[0]), float(pad_axis[1])
            pn = math.hypot(px, py)
            if pn > 1e-9:
                px, py = px / pn, py / pn
                p1b_pad = [cb[0] + L * px, cb[1] + L * py, cb[2]]
                p1_pad = self._project_base_xyz_to_uv(
                    p1b_pad, tf_base_to_cam, fx, fy, cx, cy, w, h
                )
                if p1_pad is not None:
                    cv2.arrowedLine(
                        debug,
                        p0,
                        p1_pad,
                        col_pad,
                        3,
                        tipLength=0.22,
                        line_type=cv2.LINE_AA,
                    )
                    self._draw_tag_bgr(
                        debug, "orange=pad/finger", p1_pad, col_pad, dy=-10, font_scale=0.40
                    )

        if isinstance(gap_axis, (list, tuple)) and len(gap_axis) >= 2:
            gx, gy = float(gap_axis[0]), float(gap_axis[1])
            gn = math.hypot(gx, gy)
            if gn > 1e-9:
                gx, gy = gx / gn, gy / gn
                p1b_gap = [cb[0] + L * gx, cb[1] + L * gy, cb[2]]
                p1_gap = self._project_base_xyz_to_uv(
                    p1b_gap, tf_base_to_cam, fx, fy, cx, cy, w, h
                )
                if p1_gap is not None:
                    cv2.arrowedLine(
                        debug,
                        p0,
                        p1_gap,
                        col_gap,
                        2,
                        tipLength=0.22,
                        line_type=cv2.LINE_AA,
                    )
                    self._draw_tag_bgr(
                        debug, "cyan=gap/close", p1_gap, col_gap, dy=22, font_scale=0.40
                    )

        if normalize_label(label) == "mustard_bottle":
            log_mustard_overlay_axis_debug(
                pose_meta,
                orange_axis_xy=pad_axis,
                cyan_axis_xy=gap_axis,
                orange_source=pad_src,
                cyan_source=gap_src,
                logger=self.get_logger(),
            )

    @staticmethod
    def _draw_uv_marker(
        debug: np.ndarray,
        uv: Optional[List[float]],
        color: Tuple[int, int, int],
        label: str,
        radius: int = 6,
    ) -> None:
        if uv is None or len(uv) < 2:
            return
        h, w = debug.shape[:2]
        u = int(np.clip(float(uv[0]), 0, w - 1))
        v = int(np.clip(float(uv[1]), 0, h - 1))
        cv2.circle(debug, (u, v), radius, color, 2, cv2.LINE_AA)
        PerceptionNode._draw_tag_bgr(debug, label, (u, v), color, dy=12)

    def _draw_top_face_diagnostic_panel(
        self,
        debug: np.ndarray,
        det_index: int,
        det: Any,
        pose_meta: Dict[str, Any],
        closing_yaw_rad: Optional[float],
    ) -> None:
        """Panel compacto de diagnostico top-face (esquina inferior derecha)."""
        h, w = debug.shape[:2]
        x0 = max(8, w - 360)
        y0 = max(60, h - 200 - det_index * 12)
        close_deg = "n/a"
        if closing_yaw_rad is not None:
            close_deg = f"{math.degrees(float(closing_yaw_rad)):.1f}"
        obj_yaw = pose_meta.get("object_yaw_deg")
        if obj_yaw is None:
            obj_yaw = pose_meta.get("selected_yaw_deg")
        obj_yaw_s = "n/a" if obj_yaw is None else f"{float(obj_yaw):.1f}"
        ir_v = _safe_float(pose_meta.get("inlier_ratio"))
        rmse_v = _safe_float(pose_meta.get("open3d_ransac_rmse"))
        fit_v = _safe_float(pose_meta.get("pose_fit_error"))
        lines = [
            f"TOP_FACE [{det.label}]",
            f"top_face_success={str(bool(pose_meta.get('top_face_success'))).lower()}",
            f"top_pts={int(pose_meta.get('top_face_num_points', 0))}",
            f"ratio={float(pose_meta.get('top_face_point_ratio', 0.0)):.3f}",
            f"inlier={'n/a' if ir_v is None else f'{ir_v:.3f}'}",
            f"rmse={'n/a' if rmse_v is None else f'{rmse_v:.4f}'}",
            f"fit_err={'n/a' if fit_v is None else f'{fit_v:.5f}'}",
            f"selected_yaw={obj_yaw_s}",
            f"close_yaw={close_deg}",
            f"yaw_src={pose_meta.get('yaw_source', '')}",
        ]
        for li, line in enumerate(lines):
            y = y0 + li * 15
            cv2.putText(
                debug,
                line,
                (x0, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.38,
                (0, 0, 0),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                debug,
                line,
                (x0, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.38,
                (180, 255, 180),
                1,
                cv2.LINE_AA,
            )

    def _draw_grasp_debug_overlay(
        self,
        debug: np.ndarray,
        det_index: int,
        det: Any,
        tf_msg: Any,
        fx: float,
        fy: float,
        cx: float,
        cy: float,
        w: int,
        h: int,
        grasp_fields: Dict[str, Any],
        center_info: Dict[str, Any],
        pose_meta: Dict[str, Any],
        closing_yaw_rad: Optional[float],
        pts_obj_base: np.ndarray,
        grasp_center_base: Optional[List[float]],
        *,
        chosen_center_uv: Optional[List[float]] = None,
        grasp_center_uv: Optional[List[float]] = None,
        obb_center_uv: Optional[List[float]] = None,
    ) -> None:
        """Capas de depuracion sobre BGR (no altera logica de grasp)."""
        if self._debug_overlay_simplified_for_demo:
            self._draw_grasp_debug_overlay_demo_simplified(
                debug,
                det,
                tf_msg,
                fx,
                fy,
                cx,
                cy,
                w,
                h,
                pose_meta,
                grasp_fields,
                closing_yaw_rad,
            )
            return
        if self._debug_overlay_mode == "grasp_clean":
            self._draw_grasp_debug_overlay_clean(
                debug,
                det,
                tf_msg,
                fx,
                fy,
                cx,
                cy,
                w,
                h,
                grasp_fields,
                center_info,
                pose_meta,
                closing_yaw_rad,
                pts_obj_base,
                grasp_center_base,
                chosen_center_uv=chosen_center_uv,
                grasp_center_uv=grasp_center_uv,
                obb_center_uv=obb_center_uv,
            )
            return

        col_yolo = (0, 255, 255)
        col_obb = (0, 165, 255)
        col_observed_top = (160, 160, 160)
        col_model_top = (0, 255, 0)
        col_known = (0, 255, 0)
        col_chosen = (255, 255, 0)
        col_known_ctr = (255, 0, 255)
        col_major = (255, 255, 255)
        col_minor = (0, 0, 255)
        col_close = (0, 200, 255)
        col_grasp_used = (0, 220, 0)

        if self._debug_draw_raw_yolo_detections:
            self._draw_detection_uv_preview(debug, det, det_index)
        self._draw_uv_marker(debug, chosen_center_uv, col_chosen, "chosen_center_uv")
        self._draw_uv_marker(debug, grasp_center_uv, col_grasp_used, "grasp_center_uv", 8)
        if obb_center_uv is not None:
            self._draw_uv_marker(debug, obb_center_uv, col_obb, "obb_center_uv", 5)

        tf_base_to_cam: Optional[np.ndarray] = None
        if tf_msg is not None:
            tf_cam_to_base = self._build_transform_matrix(tf_msg)
            try:
                tf_base_to_cam = np.linalg.inv(tf_cam_to_base)
            except np.linalg.LinAlgError:
                self.get_logger().warn(
                    "[DEBUG_IMAGE] TF invert failed for overlay label=%s"
                    % str(det.label)
                )
        else:
            self._draw_tag_bgr(
                debug,
                "TF unavailable (UV-only overlay)",
                (12, 52 + det_index * 18),
                (0, 128, 255),
                dy=0,
                font_scale=0.45,
            )

        top_contour_drawn = False

        # --- 3) Top face observada (gris tenue): solo diagnóstico ---
        if (
            tf_base_to_cam is not None
            and pts_obj_base.size > 0
            and bool(pose_meta.get("observed_top_face_success", False))
        ):
            tf_top = extract_top_face_points(
                pts_obj_base,
                top_z_m=float(grasp_fields.get("top_z_m") or 0.0),
            )
            tp = tf_top.get("top_points_base")
            if isinstance(tp, np.ndarray) and tp.size > 0:
                uvs = self._project_base_points_to_uv(
                    tp, tf_base_to_cam, fx, fy, cx, cy, w, h, max_points=3500
                )
                if uvs.shape[0] >= 3:
                    hull = cv2.convexHull(uvs.astype(np.float32))
                    cv2.polylines(
                        debug,
                        [hull.astype(np.int32)],
                        True,
                        col_observed_top,
                        1,
                        cv2.LINE_AA,
                    )
                    mc = np.mean(hull.reshape(-1, 2), axis=0)
                    self._draw_tag_bgr(
                        debug,
                        "observed top face",
                        (int(mc[0]), int(mc[1])),
                        col_observed_top,
                        dy=-22,
                        font_scale=0.38,
                    )
                    top_contour_drawn = True

        # --- 4) Top face modelo (verde fuerte): fuente preferente para grasp ---
        model_tc = pose_meta.get("model_top_face_corners_base")
        if not (isinstance(model_tc, list) and len(model_tc) >= 4):
            src_tf = str(pose_meta.get("top_face_source", ""))
            if src_tf in ("known_model", "hybrid_known_model"):
                model_tc = pose_meta.get("top_corners_base")
        if tf_base_to_cam is not None and isinstance(model_tc, list) and len(model_tc) >= 4:
            model_uv: List[Tuple[int, int]] = []
            for c in model_tc[:4]:
                if not isinstance(c, (list, tuple)) or len(c) < 3:
                    continue
                puv = self._project_base_xyz_to_uv(
                    [float(c[0]), float(c[1]), float(c[2])],
                    tf_base_to_cam,
                    fx,
                    fy,
                    cx,
                    cy,
                    w,
                    h,
                )
                if puv is not None:
                    model_uv.append(puv)
            if len(model_uv) >= 3:
                poly_m = np.array(model_uv, dtype=np.int32).reshape(-1, 1, 2)
                cv2.polylines(
                    debug, [poly_m], True, col_model_top, 3, cv2.LINE_AA
                )
                self._draw_tag_bgr(
                    debug,
                    "model top face",
                    model_uv[0],
                    col_model_top,
                    dy=-10,
                )
                top_contour_drawn = True

        if not top_contour_drawn:
            self._draw_tag_bgr(
                debug,
                "no top contour",
                (12, 72 + det_index * 18),
                (0, 0, 255),
                dy=0,
                font_scale=0.45,
            )

        # --- 7) / 8) Ejes major (blanco) y minor (rojo) ---
        axis_anchor_b: Optional[List[float]] = None
        if (
            isinstance(grasp_center_base, (list, tuple))
            and len(grasp_center_base) >= 3
        ):
            axis_anchor_b = [
                float(grasp_center_base[0]),
                float(grasp_center_base[1]),
                float(grasp_center_base[2]),
            ]
        else:
            ct = center_info.get("chosen_target_center_base")
            if isinstance(ct, (list, tuple)) and len(ct) >= 3:
                axis_anchor_b = [float(ct[0]), float(ct[1]), float(ct[2])]

        chosen_b = center_info.get("chosen_target_center_base")
        if tf_base_to_cam is not None and axis_anchor_b is not None:
            _top_src_ax = str(pose_meta.get("top_face_source", "")).strip()
            if _top_src_ax in (
                "runtime_gt_known_box",
                "runtime_gt_tall_object",
                "runtime_gt_known_object",
            ):
                maj = pose_meta.get("long_axis_xy") or grasp_fields.get("major_axis_xy")
                minr = pose_meta.get("short_axis_xy") or grasp_fields.get("minor_axis_xy")
            elif bool(pose_meta.get("model_top_face_success", False)):
                maj = pose_meta.get("model_major_axis_xy") or grasp_fields.get(
                    "major_axis_xy"
                )
                minr = pose_meta.get("model_minor_axis_xy") or grasp_fields.get(
                    "minor_axis_xy"
                )
            else:
                maj = grasp_fields.get("major_axis_xy")
                minr = grasp_fields.get("minor_axis_xy")
            axis_len_m = 0.10
            cb = [float(axis_anchor_b[0]), float(axis_anchor_b[1]), float(axis_anchor_b[2])]
            if isinstance(maj, (list, tuple)) and len(maj) >= 2:
                mx, my = float(maj[0]), float(maj[1])
                nrm = math.hypot(mx, my)
                if nrm > 1e-9:
                    mx, my = mx / nrm, my / nrm
                    p0 = self._project_base_xyz_to_uv(cb, tf_base_to_cam, fx, fy, cx, cy, w, h)
                    p1b = [
                        cb[0] + axis_len_m * mx,
                        cb[1] + axis_len_m * my,
                        cb[2],
                    ]
                    p1 = self._project_base_xyz_to_uv(
                        p1b, tf_base_to_cam, fx, fy, cx, cy, w, h
                    )
                    if p0 is not None and p1 is not None:
                        cv2.line(debug, p0, p1, col_major, 2, cv2.LINE_AA)
                        self._draw_tag_bgr(debug, "body_long", p1, col_major, dy=-6)
            if isinstance(minr, (list, tuple)) and len(minr) >= 2:
                sx, sy = float(minr[0]), float(minr[1])
                nrm = math.hypot(sx, sy)
                if nrm > 1e-9:
                    sx, sy = sx / nrm, sy / nrm
                    p0 = self._project_base_xyz_to_uv(cb, tf_base_to_cam, fx, fy, cx, cy, w, h)
                    p1b = [
                        cb[0] + axis_len_m * sx,
                        cb[1] + axis_len_m * sy,
                        cb[2],
                    ]
                    p1 = self._project_base_xyz_to_uv(
                        p1b, tf_base_to_cam, fx, fy, cx, cy, w, h
                    )
                    if p0 is not None and p1 is not None:
                        cv2.line(debug, p0, p1, col_minor, 2, cv2.LINE_AA)
                        self._draw_tag_bgr(
                            debug,
                            "body_short",
                            p1,
                            col_minor,
                            dy=-18,
                            font_scale=0.36,
                        )

        # --- 5) chosen center (cian): se dibuja despues de (6) para quedar encima ---
        # --- 6) known box center (magenta) ---
        kbc = pose_meta.get("model_box_center_base")
        if not (isinstance(kbc, (list, tuple)) and len(kbc) >= 3):
            kbc = pose_meta.get("known_box_center_base")
        if tf_base_to_cam is not None and isinstance(kbc, (list, tuple)) and len(kbc) >= 3:
            kuv = self._project_base_xyz_to_uv(
                [float(kbc[0]), float(kbc[1]), float(kbc[2])],
                tf_base_to_cam,
                fx,
                fy,
                cx,
                cy,
                w,
                h,
            )
            if kuv is not None:
                cv2.circle(debug, kuv, 6, col_known_ctr, 2, cv2.LINE_AA)
                self._draw_tag_bgr(
                    debug, "known box center", kuv, col_known_ctr, dy=-8
                )

        if tf_base_to_cam is not None and isinstance(chosen_b, (list, tuple)) and len(chosen_b) >= 3:
            cuv = self._project_base_xyz_to_uv(
                [float(chosen_b[0]), float(chosen_b[1]), float(chosen_b[2])],
                tf_base_to_cam,
                fx,
                fy,
                cx,
                cy,
                w,
                h,
            )
            if cuv is not None:
                cv2.circle(debug, cuv, 7, col_chosen, 2, cv2.LINE_AA)
                self._draw_tag_bgr(debug, "chosen center", cuv, col_chosen, dy=16)

        # --- 9) Naranja=pads, cian=gap (no usar closing_yaw como orientación de dedos) ---
        if tf_base_to_cam is not None and axis_anchor_b is not None:
            self._draw_grasp_gap_and_pad_axes_overlay(
                debug,
                label=str(det.label),
                pose_meta=pose_meta,
                axis_anchor_b=[
                    float(axis_anchor_b[0]),
                    float(axis_anchor_b[1]),
                    float(axis_anchor_b[2]),
                ],
                tf_base_to_cam=tf_base_to_cam,
                fx=fx,
                fy=fy,
                cx=cx,
                cy=cy,
                w=w,
                h=h,
                closing_yaw_rad=closing_yaw_rad,
                col_pad=(0, 140, 255),
                col_gap=(255, 220, 60),
            )

        # Centro operativo de grasp (JSON / controlador)
        if (
            isinstance(grasp_center_base, (list, tuple))
            and len(grasp_center_base) >= 3
        ):
            guv = self._project_base_xyz_to_uv(
                [float(grasp_center_base[0]), float(grasp_center_base[1]), float(grasp_center_base[2])],
                tf_base_to_cam,
                fx,
                fy,
                cx,
                cy,
                w,
                h,
            )
            if guv is not None:
                cv2.circle(debug, guv, 12, col_grasp_used, 3, cv2.LINE_AA)
                self._draw_tag_bgr(
                    debug,
                    "OPERATIONAL grasp center (used)",
                    guv,
                    col_grasp_used,
                    dy=24,
                    font_scale=0.44,
                )

        # --- Texto fijo (metricas) ---
        pfe = pose_meta.get("pose_fit_error")
        pfe_s = "n/a" if pfe is None else f"{float(pfe):.5f}"
        mfe = pose_meta.get("model_fit_error")
        mfe_s = "n/a" if mfe is None else f"{float(mfe):.5f}"
        tfs = str(pose_meta.get("top_face_source", "observed"))
        yfm = pose_meta.get("yaw_fit_method", "")
        ir = pose_meta.get("inlier_ratio")
        ir_s = "n/a" if ir is None else f"{float(ir):.3f}"
        oem = pose_meta.get("outside_error_m")
        oem_s = "n/a" if oem is None else f"{float(oem):.4f}"
        pl = pose_meta.get("projected_extent_length_m")
        pw = pose_meta.get("projected_extent_width_m")
        plw_s = "n/a"
        if pl is not None and pw is not None:
            plw_s = f"{float(pl):.3f}/{float(pw):.3f}"
        yms = pose_meta.get("yaw_margin_score")
        yms_s = "n/a" if yms is None else f"{float(yms):.4f}"
        act_yaw = float(grasp_fields.get("grasp_yaw_deg", 0.0))
        obj_yaw = pose_meta.get("object_yaw_deg")
        if obj_yaw is None:
            try:
                obj_yaw = math.degrees(float(pose_meta.get("object_yaw_rad", 0.0)))
            except (TypeError, ValueError):
                obj_yaw = act_yaw
        close_yaw_deg = (
            math.degrees(float(closing_yaw_rad))
            if closing_yaw_rad is not None
            else float("nan")
        )
        close_yaw_s = (
            "n/a" if closing_yaw_rad is None else f"{close_yaw_deg:.1f}"
        )
        fit_yaw = pose_meta.get("selected_yaw_deg")
        fit_yaw_s = "n/a" if fit_yaw is None else f"{float(fit_yaw):.1f}"
        partial = pose_meta.get("partial_top_face_detected", False)
        cfm = pose_meta.get("center_fit_method", "") or center_info.get(
            "target_center_method", ""
        )
        gcs = center_info.get("grasp_center_source", "")
        col_m = pose_meta.get("center_offset_long_m")
        cos_m = pose_meta.get("center_offset_short_m")
        off_s = "n/a"
        if col_m is not None and cos_m is not None:
            off_s = f"L={float(col_m):.3f} S={float(cos_m):.3f}"
        db_l = pose_meta.get("db_length_m")
        db_w = pose_meta.get("db_width_m")
        obs_l = pose_meta.get("observed_extent_length_m")
        obs_w = pose_meta.get("observed_extent_width_m")
        db_obs_s = "n/a"
        if db_l is not None and obs_l is not None:
            db_obs_s = f"L {float(obs_l):.3f}/{float(db_l):.3f} W {float(obs_w or 0):.3f}/{float(db_w or 0):.3f}"
        lines = [
            f"{det.label} score={float(det.score):.3f}",
            f"top_face_source: {tfs}",
            f"observed_top_face_success: {str(bool(pose_meta.get('observed_top_face_success'))).lower()}",
            f"model_top_face_success: {str(bool(pose_meta.get('model_top_face_success'))).lower()}",
            f"model_fit_error: {mfe_s}",
            f"yaw_source: {pose_meta.get('yaw_source', '')}",
            f"object_yaw_deg: {float(obj_yaw):.1f}",
            f"closing_yaw_deg: {close_yaw_s}",
            f"active_yaw_deg: {act_yaw:.1f}",
            f"fit_candidate_yaw_deg: {fit_yaw_s}",
            f"center_method: {cfm}",
            f"grasp_center_source: {gcs}",
            f"top_face_success: {str(bool(pose_meta.get('top_face_success'))).lower()}",
            f"yaw_fit_method: {yfm}",
            f"partial_top_face: {str(bool(partial)).lower()}",
            f"center_fit_method: {cfm}",
            f"center_offset: {off_s}",
            f"obs/db L/W: {db_obs_s}",
            f"yaw_confidence: {float(pose_meta.get('yaw_confidence', 0.0)):.3f}",
            f"pose_fit_error: {pfe_s}",
            f"inlier_ratio: {ir_s}",
            f"outside_err_m: {oem_s}",
            f"proj_L/W_m: {plw_s}",
            f"yaw_margin: {yms_s}",
            f"observed_top_face_pts: {int(pose_meta.get('observed_top_face_num_points', 0))}",
            f"observed_top_face_ratio: {float(pose_meta.get('observed_top_face_ratio', 0.0)):.3f}",
            f"top_face_num_points: {int(pose_meta.get('top_face_num_points', 0))}",
            f"top_face_point_ratio: {float(pose_meta.get('top_face_point_ratio', 0.0)):.3f}",
        ]
        # Centro observado top_face vs centro DB/híbrido
        tp_obs = pose_meta.get("top_face_observed_center_base")
        kbc_final = pose_meta.get("known_box_center_base")
        if tf_base_to_cam is not None and isinstance(tp_obs, (list, tuple)) and len(tp_obs) >= 3:
            ouv = self._project_base_xyz_to_uv(
                [float(tp_obs[0]), float(tp_obs[1]), float(tp_obs[2])],
                tf_base_to_cam,
                fx,
                fy,
                cx,
                cy,
                w,
                h,
            )
            if ouv is not None:
                cv2.circle(debug, ouv, 5, (255, 128, 0), 2, cv2.LINE_AA)
                self._draw_tag_bgr(
                    debug, "top_face observed ctr", ouv, (255, 128, 0), dy=-8
                )
        if (
            tf_base_to_cam is not None
            and isinstance(kbc_final, (list, tuple))
            and len(kbc_final) >= 3
            and isinstance(tp_obs, (list, tuple))
            and len(tp_obs) >= 3
        ):
            fuv = self._project_base_xyz_to_uv(
                [float(kbc_final[0]), float(kbc_final[1]), float(kbc_final[2])],
                tf_base_to_cam,
                fx,
                fy,
                cx,
                cy,
                w,
                h,
            )
            ouv2 = self._project_base_xyz_to_uv(
                [float(tp_obs[0]), float(tp_obs[1]), float(tp_obs[2])],
                tf_base_to_cam,
                fx,
                fy,
                cx,
                cy,
                w,
                h,
            )
            if fuv is not None and ouv2 is not None:
                cv2.arrowedLine(
                    debug,
                    ouv2,
                    fuv,
                    (0, 255, 255),
                    2,
                    tipLength=0.25,
                    line_type=cv2.LINE_AA,
                )
                self._draw_tag_bgr(
                    debug, "center correction", fuv, (0, 255, 255), dy=-8
                )

        x0 = 8
        y0 = 18 + det_index * 240
        lh = 16
        for li, line in enumerate(lines):
            y = y0 + li * lh
            cv2.putText(
                debug,
                line,
                (x0, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.46,
                (0, 0, 0),
                3,
                cv2.LINE_AA,
            )
            cv2.putText(
                debug,
                line,
                (x0, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.46,
                (240, 240, 240),
                1,
                cv2.LINE_AA,
            )

        self._draw_top_face_diagnostic_panel(
            debug, det_index, det, pose_meta, closing_yaw_rad
        )

    def _choose_target_center(
        self,
        centroid_base: Optional[List[float]],
        points_base: np.ndarray,
        top_z_m: float,
    ) -> Dict[str, Any]:
        pointcloud_centroid = None
        pointcloud_median = None
        top_surface_center = None
        chosen = centroid_base
        method = "current"
        if points_base.size > 0:
            pointcloud_centroid = [
                float(v) for v in np.mean(points_base, axis=0).tolist()
            ]
            pointcloud_median = [
                float(v) for v in np.median(points_base, axis=0).tolist()
            ]
            top_mask = points_base[:, 2] >= (top_z_m - self._top_surface_band_m)
            top_points = points_base[top_mask]
            if top_points.shape[0] >= self._min_top_surface_points:
                top_surface_center = [
                    float(np.median(top_points[:, 0])),
                    float(np.median(top_points[:, 1])),
                    float(np.median(top_points[:, 2])),
                ]
            elif pointcloud_median is not None:
                top_surface_center = list(pointcloud_median)

        if self._center_method == "pointcloud_centroid" and pointcloud_centroid is not None:
            chosen = pointcloud_centroid
            method = "pointcloud_centroid"
        elif self._center_method == "pointcloud_median" and pointcloud_median is not None:
            chosen = pointcloud_median
            method = "pointcloud_median"
        elif self._center_method == "top_surface_center" and top_surface_center is not None:
            chosen = top_surface_center
            method = "top_surface_center"
        elif centroid_base is not None:
            chosen = centroid_base
            method = "current"

        return {
            "pointcloud_centroid_base": pointcloud_centroid,
            "pointcloud_median_base": pointcloud_median,
            "top_surface_center_base": top_surface_center,
            "chosen_target_center_base": chosen,
            "target_center_method": method,
        }

    def _compute_top_grasp_fields(
        self, centroid_base: Optional[List[float]], points_base: np.ndarray
    ) -> Dict[str, Any]:
        if centroid_base is None or points_base.size == 0:
            return {
                "dimensions_m": [0.0, 0.0, 0.0],
                "top_z_m": 0.0,
                "grasp_yaw_rad": 0.0,
                "grasp_yaw_deg": 0.0,
                "approach_position": None,
                "grasp_position": None,
                "footprint_major_m": 0.0,
                "footprint_minor_m": 0.0,
                "measured_height_m": 0.0,
                "major_axis_xy": [1.0, 0.0],
                "minor_axis_xy": [0.0, 1.0],
            }
        mins = np.percentile(points_base, 5.0, axis=0)
        maxs = np.percentile(points_base, 95.0, axis=0)
        dims = np.maximum(maxs - mins, 0.0)
        xy = points_base[:, :2]
        cov = np.cov(xy.T) if xy.shape[0] >= 3 else np.eye(2)
        eigvals, eigvecs = np.linalg.eigh(cov)
        major_idx = int(np.argmax(eigvals))
        minor_idx = 1 - major_idx
        major_axis = eigvecs[:, major_idx]
        minor_axis = eigvecs[:, minor_idx]
        yaw = float(np.arctan2(major_axis[1], major_axis[0]))
        top_z = float(maxs[2])
        if xy.shape[0] >= 2:
            proj_major = xy @ major_axis
            proj_minor = xy @ minor_axis
            footprint_major_m = float(
                np.percentile(proj_major, 95.0) - np.percentile(proj_major, 5.0)
            )
            footprint_minor_m = float(
                np.percentile(proj_minor, 95.0) - np.percentile(proj_minor, 5.0)
            )
        else:
            footprint_major_m = float(dims[0])
            footprint_minor_m = float(dims[1])
        footprint_major_m = max(footprint_major_m, 0.0)
        footprint_minor_m = max(footprint_minor_m, 0.0)
        measured_height_m = float(max(dims[2], 0.0))
        grasp_pos = [
            float(centroid_base[0] + self._top_grasp_offset[0]),
            float(centroid_base[1] + self._top_grasp_offset[1]),
            float(top_z + self._top_grasp_offset[2]),
        ]
        approach_pos = [
            float(grasp_pos[0] + self._top_grasp_approach_offset[0]),
            float(grasp_pos[1] + self._top_grasp_approach_offset[1]),
            float(grasp_pos[2] + self._top_grasp_approach_offset[2]),
        ]
        return {
            "dimensions_m": [float(dims[0]), float(dims[1]), float(dims[2])],
            "top_z_m": top_z,
            "grasp_yaw_rad": yaw,
            "grasp_yaw_deg": float(np.degrees(yaw)),
            "approach_position": approach_pos,
            "grasp_position": grasp_pos,
            "footprint_major_m": footprint_major_m,
            "footprint_minor_m": footprint_minor_m,
            "measured_height_m": measured_height_m,
            "major_axis_xy": [float(major_axis[0]), float(major_axis[1])],
            "minor_axis_xy": [float(minor_axis[0]), float(minor_axis[1])],
        }

    def _apply_hybrid_fit_to_meta(
        self,
        meta: Dict[str, Any],
        grasp_fields: Dict[str, Any],
        center_info: Dict[str, Any],
        hybrid: Dict[str, Any],
        top_face_result: Dict[str, Any],
        pca_yaw: float,
    ) -> Dict[str, Any]:
        maj = grasp_fields.get("major_axis_xy") or [1.0, 0.0]
        meta["hybrid_fit_success"] = True
        meta["pose_fit_success"] = True
        meta["pose_fit_error"] = _safe_float(hybrid.get("fit_error"), 0.0)
        meta["partial_top_face_detected"] = bool(
            hybrid.get("partial_top_face_detected", False)
        )
        meta["yaw_fit_method"] = hybrid.get("yaw_fit_method")
        meta["center_fit_method"] = hybrid.get("center_fit_method")
        meta["center_offset_long_m"] = hybrid.get("center_offset_long_m")
        meta["center_offset_short_m"] = hybrid.get("center_offset_short_m")
        meta["known_box_yaw_rad"] = _safe_float(hybrid.get("yaw_rad"), pca_yaw)
        meta["known_box_center_base"] = list(hybrid.get("center_xyz", [0, 0, 0]))
        meta["top_corners_base"] = hybrid.get("top_corners_base")
        meta["bottom_corners_base"] = hybrid.get("bottom_corners_base")
        meta["long_axis_xy"] = list(hybrid.get("long_axis_xy", maj))
        meta["short_axis_xy"] = list(hybrid.get("short_axis_xy", grasp_fields.get("minor_axis_xy") or [0.0, 1.0]))
        meta["selected_yaw_deg"] = hybrid.get("selected_yaw_deg")
        for k in (
            "projected_extent_length_m",
            "projected_extent_width_m",
            "length_error_m",
            "width_error_m",
            "outside_error_m",
            "inlier_ratio",
            "edge_support_score",
            "observed_extent_length_m",
            "observed_extent_width_m",
            "db_length_m",
            "db_width_m",
        ):
            if hybrid.get(k) is not None:
                meta[k] = hybrid.get(k)
        tp_obs = top_face_result.get("top_points_base")
        if isinstance(tp_obs, np.ndarray) and tp_obs.size:
            meta["top_face_observed_center_base"] = [
                float(np.median(tp_obs[:, 0])),
                float(np.median(tp_obs[:, 1])),
                float(np.median(tp_obs[:, 2])),
            ]
        yaw_h = _safe_float(hybrid.get("yaw_rad"), pca_yaw) or pca_yaw
        grasp_fields["grasp_yaw_rad"] = yaw_h
        grasp_fields["grasp_yaw_deg"] = float(np.degrees(yaw_h))
        grasp_fields["major_axis_xy"] = list(meta["long_axis_xy"])
        grasp_fields["minor_axis_xy"] = list(meta["short_axis_xy"])
        grasp_fields["footprint_major_m"] = _safe_float(hybrid.get("long_dim_m"), 0.0) or 0.0
        grasp_fields["footprint_minor_m"] = _safe_float(hybrid.get("short_dim_m"), 0.0) or 0.0
        tz = _safe_float(meta.get("top_z_estimated"), grasp_fields.get("top_z_m")) or 0.0
        grasp_fields["top_z_m"] = tz
        cx, cy, _ = [float(x) for x in meta["known_box_center_base"]]
        center_info["chosen_target_center_base"] = [cx, cy, tz]
        center_info["target_center_method"] = "hybrid_known_box_center"
        meta["yaw_source"] = "hybrid_top_face_known_dims"
        meta["yaw_confidence"] = _safe_float(hybrid.get("yaw_confidence"), 0.0) or 0.0
        meta["pca_object_yaw_rad"] = pca_yaw
        return meta

    def _apply_model_fit_to_meta(
        self,
        meta: Dict[str, Any],
        grasp_fields: Dict[str, Any],
        center_info: Dict[str, Any],
        model: Dict[str, Any],
        top_face_result: Dict[str, Any],
        pca_yaw: float,
    ) -> Dict[str, Any]:
        """Aplica top face / yaw / centro desde cuboide DB (known_model o hybrid_known_model)."""
        src = str(model.get("top_face_source", "known_model"))
        meta["top_face_source"] = src
        meta["model_top_face_success"] = bool(model.get("model_top_face_success", True))
        meta["model_fit_error"] = _safe_float(model.get("model_fit_error"), 0.0)
        meta["model_box_yaw_rad"] = _safe_float(model.get("model_box_yaw_rad"), pca_yaw)
        meta["model_closing_yaw_rad"] = _safe_float(
            model.get("model_closing_yaw_rad"),
            _safe_float(meta["model_box_yaw_rad"], pca_yaw) or pca_yaw,
        )
        meta["model_box_center_base"] = list(
            model.get("model_box_center_base", model.get("center_xyz", [0, 0, 0]))
        )
        meta["model_top_face_corners_base"] = list(
            model.get("model_top_face_corners_base", model.get("top_corners_base", []))
        )
        meta["model_major_axis_xy"] = list(
            model.get("model_major_axis_xy", model.get("long_axis_xy", [1.0, 0.0]))
        )
        meta["model_minor_axis_xy"] = list(
            model.get("model_minor_axis_xy", model.get("short_axis_xy", [0.0, 1.0]))
        )
        meta["hybrid_fit_success"] = src == "hybrid_known_model"
        meta["pose_fit_success"] = True
        meta["pose_fit_error"] = meta["model_fit_error"]
        meta["top_corners_base"] = meta["model_top_face_corners_base"]
        meta["bottom_corners_base"] = list(
            model.get("model_bottom_corners_base", model.get("bottom_corners_base", []))
        )
        meta["known_box_yaw_rad"] = meta["model_box_yaw_rad"]
        meta["known_box_center_base"] = meta["model_box_center_base"]
        meta["long_axis_xy"] = meta["model_major_axis_xy"]
        meta["short_axis_xy"] = meta["model_minor_axis_xy"]
        meta["yaw_fit_method"] = model.get("yaw_fit_method", "model_cuboid")
        meta["center_method"] = model.get("center_method", "model_cuboid_top_slab")
        for k in (
            "projected_extent_length_m",
            "projected_extent_width_m",
            "length_error_m",
            "width_error_m",
            "outside_error_m",
            "inlier_ratio",
            "edge_support_score",
            "observed_extent_length_m",
            "observed_extent_width_m",
            "db_length_m",
            "db_width_m",
            "selected_yaw_deg",
            "partial_top_face_detected",
            "num_top_slab_points",
            "num_segmented_points",
        ):
            if model.get(k) is not None:
                meta[k] = model.get(k)

        tp_obs = top_face_result.get("top_points_base")
        if isinstance(tp_obs, np.ndarray) and tp_obs.size:
            meta["top_face_observed_center_base"] = [
                float(np.median(tp_obs[:, 0])),
                float(np.median(tp_obs[:, 1])),
                float(np.median(tp_obs[:, 2])),
            ]
            meta["observed_top_face_success"] = bool(top_face_result.get("success", False))
            meta["observed_top_face_num_points"] = int(top_face_result.get("num_top_points", 0))
            meta["observed_top_face_ratio"] = float(
                top_face_result.get("top_point_ratio", 0.0)
            )
            corners_m = meta.get("model_top_face_corners_base")
            if isinstance(corners_m, list) and len(corners_m) >= 3:
                c_err, yaw_deg = observed_vs_model_corner_error_m(tp_obs, corners_m)
                meta["observed_vs_model_corner_error_m"] = c_err
                meta["observed_vs_model_yaw_deg"] = yaw_deg

        yaw_m = _safe_float(meta["model_box_yaw_rad"], pca_yaw) or pca_yaw
        grasp_fields["grasp_yaw_rad"] = yaw_m
        grasp_fields["grasp_yaw_deg"] = float(np.degrees(yaw_m))
        grasp_fields["major_axis_xy"] = list(meta["model_major_axis_xy"])
        grasp_fields["minor_axis_xy"] = list(meta["model_minor_axis_xy"])
        grasp_fields["footprint_major_m"] = _safe_float(model.get("long_dim_m"), 0.0) or 0.0
        grasp_fields["footprint_minor_m"] = _safe_float(model.get("short_dim_m"), 0.0) or 0.0
        tz = _safe_float(model.get("top_z_model_m"), grasp_fields.get("top_z_m")) or 0.0
        grasp_fields["top_z_m"] = tz
        meta["top_z_estimated"] = tz
        cx, cy, _ = [float(x) for x in meta["model_box_center_base"]]
        center_info["chosen_target_center_base"] = [cx, cy, tz]
        center_info["target_center_method"] = "model_box_center"
        center_info["grasp_center_source"] = "model_box_center"
        meta["yaw_source"] = (
            "hybrid_top_face_known_dims"
            if src == "hybrid_known_model"
            else "known_model_top_face"
        )
        meta["yaw_confidence"] = _safe_float(model.get("yaw_confidence"), 0.0) or 0.0
        meta["top_face_success"] = True
        meta["pca_object_yaw_rad"] = pca_yaw
        return meta

    def _enrich_grasp_pose_top_face(
        self,
        label: str,
        grasp_fields: Dict[str, Any],
        center_info: Dict[str, Any],
        pts_obj_base: np.ndarray,
        policy: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Top face + ajuste rectangular DB; actualiza grasp_fields y center_info in-place."""
        meta: Dict[str, Any] = {
            "label": label,
            "top_face_success": False,
            "top_face_method": "none",
            "top_face_num_points": 0,
            "top_face_point_ratio": 0.0,
            "top_z_estimated": float(grasp_fields.get("top_z_m", 0.0)),
            "yaw_source": "pca_raw",
            "yaw_confidence": 0.45,
            "pose_fit_success": False,
            "pose_fit_error": None,
            "top_corners_base": None,
            "bottom_corners_base": None,
            "known_box_center_base": None,
            "known_box_yaw_rad": None,
            "long_axis_xy": grasp_fields.get("major_axis_xy"),
            "short_axis_xy": grasp_fields.get("minor_axis_xy"),
            "yaw_fit_method": None,
            "center_method": None,
            "center_shift_from_median_m": None,
            "projected_extent_length_m": None,
            "projected_extent_width_m": None,
            "length_error_m": None,
            "width_error_m": None,
            "outside_error_m": None,
            "inlier_ratio": None,
            "edge_support_score": None,
            "yaw_margin_score": None,
            "best_score": None,
            "second_best_score": None,
            "num_yaw_candidates": None,
            "selected_yaw_deg": None,
            "partial_top_face_detected": False,
            "hybrid_fit_success": False,
            "center_fit_method": None,
            "center_offset_long_m": None,
            "center_offset_short_m": None,
            "observed_extent_length_m": None,
            "observed_extent_width_m": None,
            "db_length_m": None,
            "db_width_m": None,
            "fit_reject_reason": None,
            "top_face_observed_center_base": None,
            "pca_object_yaw_rad": None,
            "top_face_source": "observed",
            "observed_top_face_success": False,
            "observed_top_face_num_points": 0,
            "observed_top_face_ratio": 0.0,
            "model_top_face_success": False,
            "model_fit_error": None,
            "model_box_center_base": None,
            "model_box_yaw_rad": None,
            "model_closing_yaw_rad": None,
            "model_top_face_corners_base": None,
            "model_major_axis_xy": None,
            "model_minor_axis_xy": None,
        }
        xy = pts_obj_base[:, :2] if pts_obj_base.size else np.zeros((0, 2))
        if xy.shape[0] >= 5:
            cov = np.cov(xy.T)
            ev = np.linalg.eigvalsh(cov)
            ev.sort()
            if float(ev[0]) > 1e-10:
                meta["yaw_confidence"] = float(
                    min(1.0, max(0.05, (float(ev[1]) / float(ev[0]) - 1.0) / 4.0))
                )

        shape = str(policy.get("shape", ""))
        cyl_like = shape in ("cylinder", "low_cylinder", "cylinder_wide", "sphere_like")
        box_like = shape in ("box", "low_box", "low_box_wide", "curved_long")

        if pts_obj_base.size == 0:
            return meta

        t_tf0 = time.perf_counter()
        tf = extract_top_face_points(
            pts_obj_base,
            top_z_m=float(grasp_fields.get("top_z_m") or 0.0),
        )
        top_face_ms = (time.perf_counter() - t_tf0) * 1000.0
        log_top_face_summary(self.get_logger(), label, tf)
        meta["top_face_method"] = str(tf.get("method", "none"))
        meta["top_face_num_points"] = int(tf.get("num_top_points", 0))
        meta["top_face_point_ratio"] = float(tf.get("top_point_ratio", 0.0))
        meta["top_z_estimated"] = float(tf.get("top_z_estimated", meta["top_z_estimated"]))
        meta["top_face_success"] = bool(tf.get("success", False))
        meta["observed_top_face_success"] = meta["top_face_success"]
        meta["observed_top_face_num_points"] = meta["top_face_num_points"]
        meta["observed_top_face_ratio"] = meta["top_face_point_ratio"]
        if meta["observed_top_face_success"]:
            tp_diag = tf.get("top_points_base")
            if isinstance(tp_diag, np.ndarray) and tp_diag.size:
                meta["top_face_observed_center_base"] = [
                    float(np.median(tp_diag[:, 0])),
                    float(np.median(tp_diag[:, 1])),
                    float(np.median(tp_diag[:, 2])),
                ]

        if cyl_like:
            meta["yaw_source"] = "cylinder_yaw_irrelevant"
            meta["yaw_confidence"] = 1.0
            if meta["top_face_success"]:
                tp = tf["top_points_base"]
                if isinstance(tp, np.ndarray) and tp.size:
                    cx = float(np.median(tp[:, 0]))
                    cy = float(np.median(tp[:, 1]))
                    cz = float(np.median(tp[:, 2]))
                    center_info["chosen_target_center_base"] = [cx, cy, cz]
                    center_info["target_center_method"] = "top_face_median"
            return meta

        if not box_like:
            if shape == "bottle":
                meta["yaw_source"] = "bottle_pca"
                if meta["top_face_success"]:
                    tp = tf["top_points_base"]
                    if isinstance(tp, np.ndarray) and tp.size:
                        cx = float(np.median(tp[:, 0]))
                        cy = float(np.median(tp[:, 1]))
                        cz = float(np.median(tp[:, 2]))
                        center_info["chosen_target_center_base"] = [cx, cy, cz]
                        center_info["target_center_method"] = "top_face_median"
            return meta

        db_entry = OBJECT_DB.get(normalize_label(label))
        db_dims = None
        if db_entry is not None and "dims" in db_entry:
            d = db_entry["dims"]
            if len(d) == 3:
                db_dims = (float(d[0]), float(d[1]), float(d[2]))
        if db_dims is None:
            return meta

        if (
            self._use_runtime_scene_gt
            and normalize_label(label) in TALL_KNOWN_OBJECT_LABELS
        ):
            gt_tall = self._resolve_runtime_gt_operational_face(label)
            if gt_tall is not None:
                self._apply_operational_face_to_pose_meta(
                    meta, grasp_fields, center_info, gt_tall
                )
                top_pts = tf.get("top_points_base")
                if isinstance(top_pts, np.ndarray) and top_pts.size:
                    meta["top_face_observed_center_base"] = [
                        float(np.median(top_pts[:, 0])),
                        float(np.median(top_pts[:, 1])),
                        float(np.median(top_pts[:, 2])),
                    ]
                meta["_profile_ms"] = {"top_face": top_face_ms}
                return meta

        pca_yaw = _safe_float(grasp_fields.get("grasp_yaw_rad"))
        if pca_yaw is None:
            self.get_logger().warn(
                "[PERCEPTION] skipping detection label=%s reason=pca_yaw_none" % label
            )
            meta["yaw_source"] = "unavailable"
            meta["pose_fit_success"] = False
            meta["fit_reject_reason"] = "pca_yaw_none"
            return meta

        meta["pca_object_yaw_rad"] = pca_yaw
        maj = grasp_fields.get("major_axis_xy") or [1.0, 0.0]
        minr = grasp_fields.get("minor_axis_xy") or [0.0, 1.0]
        top_z_est = _safe_float(meta.get("top_z_estimated"), 0.0) or 0.0
        hybrid_attempted = False
        profile_ms: Dict[str, float] = {}

        # --- 0) GT determinista desde spawn (cajas conocidas runtime) ---
        if (
            self._use_spawn_geometry_for_known_boxes
            and is_known_spawn_geometry_box_label(label)
        ):
            rt_meta = self._apply_runtime_known_box_spawn_geometry(
                label, meta, grasp_fields, center_info, policy
            )
            if rt_meta is not None:
                top_pts = tf.get("top_points_base")
                if not isinstance(top_pts, np.ndarray) or top_pts.size == 0:
                    top_pts = np.empty((0, 3), dtype=np.float64)
                if isinstance(top_pts, np.ndarray) and top_pts.size:
                    rt_meta["top_face_observed_center_base"] = [
                        float(np.median(top_pts[:, 0])),
                        float(np.median(top_pts[:, 1])),
                        float(np.median(top_pts[:, 2])),
                    ]
                model_corners_dbg: Optional[List[List[float]]] = None
                if is_box_like_known_shape(shape) and self._prefer_model_top_face_fit_for_boxes:
                    t_m0 = time.perf_counter()
                    model_result = fit_model_cuboid_top_face(
                        pts_obj_base,
                        label,
                        db_dims,
                        self._table_z_m,
                        yaw_hint_rad=pca_yaw,
                        top_z_hint_m=top_z_est,
                    )
                    profile_ms["model_cuboid"] = (time.perf_counter() - t_m0) * 1000.0
                    log_model_top_face_summary(self.get_logger(), label, model_result)
                    rt_meta["model_top_face_success"] = bool(
                        model_result.get("model_top_face_success", False)
                    )
                    rt_meta["model_fit_error"] = _safe_float(
                        model_result.get("model_fit_error")
                    )
                    mcorners = model_result.get("model_top_face_corners_base")
                    if isinstance(mcorners, list) and len(mcorners) >= 4:
                        rt_meta["model_top_face_corners_base"] = list(mcorners)
                        model_corners_dbg = list(mcorners)
                    rt_meta["model_box_center_base"] = model_result.get(
                        "model_box_center_base"
                    )
                    rt_meta["model_box_yaw_rad"] = model_result.get("model_box_yaw_rad")
                gt_corners_dbg = rt_meta.get("runtime_gt_top_face_corners_base")
                tz_rt = _safe_float(grasp_fields.get("top_z_m"), top_z_est) or top_z_est
                if (
                    isinstance(model_corners_dbg, list)
                    and len(model_corners_dbg) >= 4
                    and isinstance(gt_corners_dbg, list)
                    and len(gt_corners_dbg) >= 4
                ):
                    metrics = compute_top_face_gt_metrics(
                        observed_corners_base=None,
                        model_corners_base=model_corners_dbg,
                        gt_corners_base=gt_corners_dbg,
                        fx=600.0,
                        z_est_m=max(float(tz_rt), 0.25),
                    )
                    rt_meta.update(metrics)
                    log_top_face_gt_compare(
                        self.get_logger(),
                        label=label,
                        entity=str(rt_meta.get("gt_entity_name", "")),
                        metrics=metrics,
                        pose_meta=rt_meta,
                    )
                profile_ms["top_face"] = top_face_ms
                rt_meta["_profile_ms"] = profile_ms
                return rt_meta

        # --- 1) Model-based cuboid (nube segmentada completa; franja superior interna) ---
        if is_box_like_known_shape(shape) and self._prefer_model_top_face_fit_for_boxes:
            t_m0 = time.perf_counter()
            model_result = fit_model_cuboid_top_face(
                pts_obj_base,
                label,
                db_dims,
                self._table_z_m,
                yaw_hint_rad=pca_yaw,
                top_z_hint_m=top_z_est,
            )
            profile_ms["model_cuboid"] = (time.perf_counter() - t_m0) * 1000.0
            log_model_top_face_summary(self.get_logger(), label, model_result)
            meta["model_top_face_success"] = bool(
                model_result.get("model_top_face_success", False)
            )
            meta["model_fit_error"] = _safe_float(model_result.get("model_fit_error"))
            if model_result.get("success", False):
                profile_ms["top_face"] = top_face_ms
                meta["_profile_ms"] = profile_ms
                return self._apply_model_fit_to_meta(
                    meta, grasp_fields, center_info, model_result, tf, pca_yaw
                )
            meta["fit_reject_reason"] = str(
                model_result.get("message", "model_fit_failed")
            )

        top_pts = tf.get("top_points_base")
        if not isinstance(top_pts, np.ndarray) or top_pts.size == 0:
            top_pts = np.empty((0, 3), dtype=np.float64)

        # --- 2) Fallback híbrido / rectangle sobre top face observada (diagnóstico / respaldo) ---
        if self._prefer_hybrid_fit_for_boxes and top_pts.size > 0:
            t_h0 = time.perf_counter()
            hybrid_first = try_hybrid_top_face_known_dims_fit(
                top_pts,
                db_dims,
                pca_yaw,
                (float(maj[0]), float(maj[1])),
                (float(minr[0]), float(minr[1])),
                self._table_z_m,
                top_z_est,
                rectangle_fit=None,
            )
            profile_ms["hybrid"] = (time.perf_counter() - t_h0) * 1000.0
            hybrid_attempted = True
            if hybrid_first.get("success", False):
                profile_ms["top_face"] = top_face_ms
                meta["_profile_ms"] = profile_ms
                model_hybrid = merge_hybrid_as_model_source(hybrid_first)
                return self._apply_model_fit_to_meta(
                    meta, grasp_fields, center_info, model_hybrid, tf, pca_yaw
                )

        run_rectangle = (
            self._debug_global_rectangle_search
            or not self._prefer_hybrid_fit_for_boxes
            or hybrid_attempted
        )
        fit: Dict[str, Any] = {"success": False, "message": "skipped"}
        if run_rectangle and top_pts.size > 0:
            t_r0 = time.perf_counter()
            fit = fit_known_top_rectangle_pose(
                top_pts,
                label,
                db_dims,
                self._table_z_m,
                top_z_est,
                yaw_initial_rad=pca_yaw,
                top_face_point_ratio=float(meta.get("top_face_point_ratio", 0.0)),
            )
            profile_ms["global_search"] = (time.perf_counter() - t_r0) * 1000.0
            log_pose_fit_summary(self.get_logger(), label, fit)
            meta["pose_fit_success"] = bool(fit.get("success", False))
            meta["pose_fit_error"] = _safe_float(fit.get("fit_error"), 0.0)
            meta["top_corners_base"] = fit.get("top_corners_base")
            meta["bottom_corners_base"] = fit.get("bottom_corners_base")
            meta["known_box_yaw_rad"] = _safe_float(fit.get("yaw_rad"))
            meta["known_box_center_base"] = list(fit.get("center_xyz", [0, 0, 0]))
            meta["yaw_fit_method"] = fit.get("yaw_fit_method")
            meta["center_method"] = fit.get("center_method")
            meta["center_shift_from_median_m"] = fit.get("center_shift_from_median_m")
            meta["projected_extent_length_m"] = fit.get("projected_extent_length_m")
            meta["projected_extent_width_m"] = fit.get("projected_extent_width_m")
            meta["length_error_m"] = fit.get("length_error_m")
            meta["width_error_m"] = fit.get("width_error_m")
            meta["outside_error_m"] = fit.get("outside_error_m")
            meta["inlier_ratio"] = fit.get("inlier_ratio")
            meta["edge_support_score"] = fit.get("edge_support_score")
            meta["yaw_margin_score"] = fit.get("yaw_margin_score")
            meta["best_score"] = fit.get("best_score")
            meta["second_best_score"] = fit.get("second_best_score")
            meta["num_yaw_candidates"] = fit.get("num_yaw_candidates")
            meta["selected_yaw_deg"] = fit.get("selected_yaw_deg")
            meta["partial_top_face_detected"] = bool(
                fit.get("partial_top_face_detected", False)
            )
            meta["observed_extent_length_m"] = fit.get("observed_extent_length_m")
            meta["observed_extent_width_m"] = fit.get("observed_extent_width_m")
            meta["db_length_m"] = fit.get("db_length_m")
            meta["db_width_m"] = fit.get("db_width_m")

        if fit.get("success", False):
            yaw_obj = _safe_float(fit.get("yaw_rad"))
            if yaw_obj is None:
                meta["yaw_source"] = "unavailable"
                meta["pose_fit_success"] = False
                return meta
            meta["top_face_source"] = "observed"
            grasp_fields["grasp_yaw_rad"] = yaw_obj
            grasp_fields["grasp_yaw_deg"] = float(np.degrees(yaw_obj))
            grasp_fields["major_axis_xy"] = list(fit["long_axis_xy"])
            grasp_fields["minor_axis_xy"] = list(fit["short_axis_xy"])
            grasp_fields["footprint_major_m"] = _safe_float(fit.get("long_dim_m"), 0.0) or 0.0
            grasp_fields["footprint_minor_m"] = _safe_float(fit.get("short_dim_m"), 0.0) or 0.0
            grasp_fields["top_z_m"] = top_z_est
            cx, cy, _ = [float(x) for x in fit["center_xyz"]]
            center_info["chosen_target_center_base"] = [cx, cy, top_z_est]
            center_info["target_center_method"] = "known_rectangle_fit"
            meta["yaw_source"] = "known_rectangle_fit"
            meta["yaw_confidence"] = _safe_float(fit.get("yaw_confidence"), 0.0) or 0.0
            meta["long_axis_xy"] = list(fit["long_axis_xy"])
            meta["short_axis_xy"] = list(fit["short_axis_xy"])
            profile_ms["top_face"] = top_face_ms
            meta["_profile_ms"] = profile_ms
            return meta

        if top_pts.size > 0 and (
            not hybrid_attempted or fit.get("message") != "skipped"
        ):
            t_h1 = time.perf_counter()
            hybrid = try_hybrid_top_face_known_dims_fit(
                top_pts,
                db_dims,
                pca_yaw,
                (float(maj[0]), float(maj[1])),
                (float(minr[0]), float(minr[1])),
                self._table_z_m,
                top_z_est,
                rectangle_fit=fit if run_rectangle else None,
            )
            profile_ms["hybrid"] = profile_ms.get("hybrid", 0.0) + (
                (time.perf_counter() - t_h1) * 1000.0
            )
            if hybrid.get("success", False):
                profile_ms["top_face"] = top_face_ms
                meta["_profile_ms"] = profile_ms
                model_hybrid = merge_hybrid_as_model_source(hybrid)
                return self._apply_model_fit_to_meta(
                    meta, grasp_fields, center_info, model_hybrid, tf, pca_yaw
                )

        meta["top_face_source"] = "observed"
        meta["yaw_source"] = "pca_raw"
        meta["fit_reject_reason"] = str(fit.get("message", "rectangle_fit_failed"))
        try:
            meta["yaw_confidence"] = float(
                min(float(meta.get("yaw_confidence", 0.45)), 0.55)
            )
        except (TypeError, ValueError):
            meta["yaw_confidence"] = 0.45
        profile_ms["top_face"] = top_face_ms
        meta["_profile_ms"] = profile_ms
        return meta

    def _process_frame(self, image_msg: Image, depth_msg: Image) -> None:
        t_frame0 = time.perf_counter()
        t_sync0 = time.perf_counter()
        prof: Dict[str, float] = {
            "sync_wait": 0.0,
            "yolo": 0.0,
            "rgbd": 0.0,
            "top_face": 0.0,
            "hybrid": 0.0,
            "global_search": 0.0,
            "model_cuboid": 0.0,
            "overlay": 0.0,
            "publish": 0.0,
            "total": 0.0,
        }
        prof["sync_wait"] = (time.perf_counter() - t_sync0) * 1000.0
        deferred_overlays: List[Dict[str, Any]] = []
        cam_frame = self._camera_optical_frame
        try:
            bgr = self._bridge.imgmsg_to_cv2(image_msg, desired_encoding="bgr8")
            depth_raw = self._bridge.imgmsg_to_cv2(
                depth_msg, desired_encoding="passthrough"
            )
        except Exception as exc:
            self.get_logger().error(f"cv_bridge: {exc}")
            if self._publish_debug:
                self._publish_debug_status_image_bgr(
                    self._make_fallback_debug_bgr("debug fallback: cv_bridge failed"),
                    cam_frame,
                    banner="debug fallback: cv_bridge failed",
                )
            return

        depth_m = self._depth_to_meters(depth_raw, depth_msg.encoding)
        ci = self._camera_info
        assert ci is not None
        h, w = depth_m.shape[:2]
        fx, fy, cx, cy = scaled_intrinsics_from_camera_info(ci, h, w)
        cam_frame = ci.header.frame_id or self._camera_optical_frame

        debug = bgr.copy()

        prompt = self._effective_text_prompt()
        t_vis0 = time.perf_counter()
        try:
            detections = self._vision.detect(bgr, depth_m, text_prompt=prompt)
        except Exception as exc:
            self.get_logger().error(f"Vision detect failed: {exc}")
            if self._publish_debug:
                self._cache_and_publish_rich_debug(
                    debug,
                    cam_frame,
                    0,
                    0,
                    label="none",
                    reason="vision_detect_failed",
                    has_overlays=False,
                    banner="vision detect failed",
                )
            return
        vision_total_ms = (time.perf_counter() - t_vis0) * 1000.0
        prof["yolo"] = vision_total_ms

        def _det_center_preview(det_obj: Any) -> List[float]:
            if getattr(det_obj, "obb_polygon_uv", None) is not None:
                poly = det_obj.obb_polygon_uv
                return [float(np.mean(poly[:, 0])), float(np.mean(poly[:, 1]))]
            if getattr(det_obj, "bbox_xyxy", None) is not None:
                x1, y1, x2, y2 = [float(v) for v in det_obj.bbox_xyxy]
                return [0.5 * (x1 + x2), 0.5 * (y1 + y2)]
            return [float("nan"), float("nan")]

        raw_labels = [str(getattr(d, "label", "")) for d in detections]
        raw_scores = [float(getattr(d, "score", 0.0)) for d in detections]
        raw_classes = [
            getattr(d, "class_id", getattr(d, "class_idx", getattr(d, "cls", None)))
            for d in detections
        ]
        raw_centers = [_det_center_preview(d) for d in detections]
        self.get_logger().info(
            "[RAW_YOLO_DETECTIONS]\n"
            "frame_id=%s\n"
            "num_raw=%d\n"
            "labels=%s\n"
            "scores=%s\n"
            "classes=%s\n"
            "boxes_or_obb_centers=%s\n"
            "image_shape=%dx%d"
            % (
                str(cam_frame),
                int(len(detections)),
                str(raw_labels),
                str([round(v, 4) for v in raw_scores]),
                str(raw_classes),
                str(raw_centers),
                int(h),
                int(w),
            )
        )
        for det in detections:
            if str(getattr(det, "label", "")).strip().lower() == "bleach_cleanser":
                bbox_uv = None
                if getattr(det, "bbox_xyxy", None) is not None:
                    bbox_uv = [float(v) for v in det.bbox_xyxy]
                obb_uv = _det_center_preview(det)
                self.get_logger().info(
                    "[RAW_YOLO_BLEACH]\n"
                    "label=bleach_cleanser\n"
                    "score=%.4f\n"
                    "bbox_uv=%s\n"
                    "obb_center_uv=%s\n"
                    "result=RAW_DETECTED"
                    % (
                        float(getattr(det, "score", 0.0)),
                        str(bbox_uv),
                        str(obb_uv),
                    )
                )

        raw_yolo_count = int(len(detections))
        after_confidence_count = 0
        after_depth_count = 0
        after_table_count = 0
        after_geometry_count = 0

        o3d_cfg = self._build_open3d_config()
        objects_out: List[Dict[str, Any]] = []
        o3d_ms_acc = 0.0
        inlier_ratios: List[float] = []
        rmses: List[float] = []
        grasp_confs: List[float] = []
        grasp_latencies: List[float] = []

        self._processed_frame_count += 1
        if len(detections) == 0:
            cv2.putText(
                debug,
                "no detections",
                (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )
        should_log_tf = self._enable_tf_diagnostics and (
            self._processed_frame_count % self._tf_debug_every_n_frames == 0
        )
        for i_det, det in enumerate(detections):
            if (
                self._publish_debug
                and self._draw_debug_overlay
                and self._debug_draw_raw_yolo_detections
            ):
                self._draw_detection_uv_preview(debug, det, i_det)
            closing_yaw: Optional[float] = None
            closing_yaw_source: Optional[str] = None
            build_stage = "init"
            pose_meta_build: Optional[Dict[str, Any]] = None
            grasp_center_source_build = ""
            try:
                det_label = str(det.label or "")
                det_score = float(getattr(det, "score", 0.0))
                conf_ok = det_score >= float(self._confidence_threshold)
                self.get_logger().info(
                    "[YOLO_FILTER_DECISION] label=%s score=%.4f stage=confidence passed=%s reason=%s"
                    % (
                        det_label,
                        det_score,
                        str(conf_ok).lower(),
                        "score_gte_threshold" if conf_ok else "score_below_threshold",
                    )
                )
                if not conf_ok:
                    continue
                after_confidence_count += 1

                if not self._detection_passes_label_filter(det.label):
                    self.get_logger().info(
                        "[YOLO_FILTER_DECISION] label=%s score=%.4f stage=operational passed=false reason=target_label_filter_mismatch"
                        % (det_label, det_score)
                    )
                    continue
                t_o3 = time.perf_counter()
                ts = estimate_top_surface_plane_centroid(
                    det.mask, depth_m, fx, fy, cx, cy, o3d_cfg
                )
                o3d_ms_acc += (time.perf_counter() - t_o3) * 1000.0
                prof["rgbd"] += (time.perf_counter() - t_o3) * 1000.0

                if not ts.success:
                    self.get_logger().info(
                        "[YOLO_FILTER_DECISION] label=%s score=%.4f stage=depth passed=false reason=%s"
                        % (
                            det_label,
                            det_score,
                            str(ts.message or "top_surface_estimation_failed"),
                        )
                    )
                    continue
                after_depth_count += 1
                self.get_logger().info(
                    "[YOLO_FILTER_DECISION] label=%s score=%.4f stage=depth passed=true reason=top_surface_estimation_ok"
                    % (det_label, det_score)
                )

                if ts.success:
                    inlier_ratios.append(ts.inlier_ratio)
                    if ts.ransac_rmse > 0:
                        rmses.append(ts.ransac_rmse)

                    cc = ts.centroid_camera
                    tf_msg = self._lookup_transform(cam_frame)
                    centroid_base: Optional[List[float]] = None
                    pts_obj_base = np.empty((0, 3), dtype=np.float64)
                    if tf_msg is not None:
                        pb = self._transform_point(cc, tf_msg)
                        centroid_base = [float(pb[0]), float(pb[1]), float(pb[2])]
                        if should_log_tf:
                            t = tf_msg.transform.translation
                            q = tf_msg.transform.rotation
                            self.get_logger().info(
                                "TF diagnostico %s->%s: t=(%.3f, %.3f, %.3f), q=(%.4f, %.4f, %.4f, %.4f)"
                                % (
                                    cam_frame,
                                    self._target_frame,
                                    t.x,
                                    t.y,
                                    t.z,
                                    q.x,
                                    q.y,
                                    q.z,
                                    q.w,
                                )
                            )

                    pts_obj = depth_mask_to_points_camera(
                        det.mask, depth_m, fx, fy, cx, cy, z_min=o3d_cfg.z_min_depth
                    )
                    if tf_msg is not None and pts_obj.size > 0:
                        pts_obj_base = self._transform_points(
                            pts_obj.astype(np.float64), tf_msg
                        )
                    grasp_fields = self._compute_top_grasp_fields(centroid_base, pts_obj_base)
                    if self._reject_detection_z_out_of_range(
                        str(det.label), centroid_base, float(grasp_fields["top_z_m"])
                    ):
                        self.get_logger().info(
                            "[YOLO_FILTER_DECISION] label=%s score=%.4f stage=table passed=false reason=z_out_of_table_range"
                            % (det_label, det_score)
                        )
                        continue
                    after_table_count += 1
                    self.get_logger().info(
                        "[YOLO_FILTER_DECISION] label=%s score=%.4f stage=table passed=true reason=z_within_table_range"
                        % (det_label, det_score)
                    )
                    center_info = self._choose_target_center(
                    centroid_base=centroid_base,
                    points_base=pts_obj_base,
                    top_z_m=float(grasp_fields["top_z_m"]),
                    )
                    if should_log_tf:
                        self.get_logger().info(
                            "Objeto='%s' frame_camera='%s' frame_target='%s' centroid_camera=%s centroid_base=%s pointcloud_centroid=%s pointcloud_median=%s top_surface_center=%s chosen_center=%s method=%s top_z_m=%.3f approach=%s grasp=%s"
                            % (
                                det.label,
                                cam_frame,
                                self._target_frame,
                                [float(cc[0]), float(cc[1]), float(cc[2])],
                                centroid_base,
                                center_info["pointcloud_centroid_base"],
                                center_info["pointcloud_median_base"],
                                center_info["top_surface_center_base"],
                                center_info["chosen_target_center_base"],
                                center_info["target_center_method"],
                                float(grasp_fields["top_z_m"]),
                                grasp_fields["approach_position"],
                                grasp_fields["grasp_position"],
                            )
                    )
                    if centroid_base is not None:
                        if centroid_base[2] < self._warn_base_z_min or centroid_base[2] > self._warn_base_z_max:
                            self.get_logger().warn(
                                "Centroid_base z sospechosa para objeto sobre mesa: z=%.3f fuera de [%.2f, %.2f] (label=%s)"
                                % (
                                    centroid_base[2],
                                    self._warn_base_z_min,
                                    self._warn_base_z_max,
                                    det.label,
                                )
                            )
                    if centroid_base[0] < self._warn_base_x_min or centroid_base[0] > self._warn_base_x_max:
                        self.get_logger().warn(
                            "Centroid_base x posiblemente fuera de alcance: x=%.3f fuera de [%.2f, %.2f] (label=%s)"
                            % (
                                centroid_base[0],
                                self._warn_base_x_min,
                                self._warn_base_x_max,
                                det.label,
                            )
                        )

                    cloud = numpy_xyz_to_pointcloud2(pts_obj, cam_frame, image_msg.header.stamp)
                    hyps, gconf, glat = self._grasp.query(cloud, cam_frame, self._grasp_timeout)
                    grasp_confs.append(gconf)
                    grasp_latencies.append(glat)

                    grasp_pose_dict: Optional[Dict[str, Any]] = None
                    candidates_dict: List[Dict[str, Any]] = []
                    if hyps:
                        grasp_pose_dict = pose_to_dict(hyps[0].position, hyps[0].orientation_xyzw)
                    for h in hyps:
                        candidates_dict.append(
                            pose_to_dict(h.position, h.orientation_xyzw)
                            | {"confidence": h.confidence}
                        )

                    obb_list: Optional[List[List[float]]] = None
                    if det.obb_polygon_uv is not None:
                        obb_list = det.obb_polygon_uv.astype(float).tolist()

                    proj_u = int(np.clip((cc[0] * fx / max(cc[2], 1e-6)) + cx, 0, w - 1))
                    proj_v = int(np.clip((cc[1] * fy / max(cc[2], 1e-6)) + cy, 0, h - 1))
                    chosen_uv = None

                    footprint_major_m = float(grasp_fields.get("footprint_major_m", 0.0))
                    footprint_minor_m = float(grasp_fields.get("footprint_minor_m", 0.0))
                    measured_height_m = float(grasp_fields.get("measured_height_m", 0.0))

                    _maj = footprint_major_m if footprint_major_m > 0.0 else None
                    _min = footprint_minor_m if footprint_minor_m > 0.0 else None
                    _hm = measured_height_m if measured_height_m > 0.0 else None
                    policy = get_grasp_policy(
                        det.label,
                        measured_footprint_major_m=_maj,
                        measured_footprint_minor_m=_min,
                        measured_height_m=_hm,
                        use_measured_dimensions=self._use_measured_dimensions_for_policy,
                    )
                    policy_exec = export_grasp_policy_for_executor(
                        det.label,
                        measured_footprint_major_m=_maj,
                        measured_footprint_minor_m=_min,
                        measured_height_m=_hm,
                        use_measured_dimensions=self._use_measured_dimensions_for_policy,
                        edge_offset_m=float(_EDGE_GRASP_OFFSET_M),
                    )

                    pose_meta = self._enrich_grasp_pose_top_face(
                        str(det.label), grasp_fields, center_info, pts_obj_base, policy
                    )
                    pose_meta_build = pose_meta
                    if not self._runtime_gt_spawn_has_operational_center(pose_meta):
                        ops_face = self._resolve_operational_top_face(
                            str(det.label),
                            pose_meta,
                            grasp_fields,
                            top_face_observed_z=_safe_float(
                                pose_meta.get("top_z_estimated")
                            ),
                        )
                        if ops_face is not None:
                            self._apply_operational_face_to_pose_meta(
                                pose_meta, grasp_fields, center_info, ops_face
                            )
                            if (
                                str(ops_face.get("top_face_source", "")).strip()
                                in ("runtime_gt_known_box", "runtime_gt_known_object", "runtime_gt_tall_object")
                            ):
                                req_w = float(policy.get("required_grasp_width_m", 0.0) or 0.0)
                                dims_lwh = ops_face.get("dims_used_lwh") or []
                                self.get_logger().info(
                                    "[KNOWN_OBJECT_GT_GRASP] label=%s "
                                    "top_face_source=%s "
                                    "grasp_center_source=%s "
                                    "yaw_source=runtime_gt_spawn_yaw "
                                    "closing_yaw_source=%s closing_yaw_rad=%.4f required_width=%.4f "
                                    "top_z_m=%.4f dims_lwh=%s entity=%s operational_source_fallback=false"
                                    % (
                                        str(det.label),
                                        str(ops_face.get("top_face_source", "")),
                                        str(ops_face.get("grasp_center_source", "")),
                                        str(ops_face.get("closing_yaw_source", "runtime_gt_short_axis")),
                                        float(ops_face["closing_yaw_rad"]),
                                        req_w,
                                        float(ops_face["top_z_m"]),
                                        dims_lwh,
                                        ops_face.get("gt_entity_name", ""),
                                    )
                                )
                                _lb = str(det.label).strip().lower()
                                if _lb == "mustard_bottle":
                                    self.get_logger().info("[MUSTARD_GRASP_GEOMETRY] top_z_m=%.4f required_width=%.4f" % (float(ops_face["top_z_m"]), req_w))
                                elif _lb == "gelatin_box":
                                    self.get_logger().info("[GELATIN_GRASP_GEOMETRY] top_z_m=%.4f required_width=%.4f" % (float(ops_face["top_z_m"]), req_w))
                                elif _lb == "bleach_cleanser":
                                    self.get_logger().info("[BLEACH_GRASP_GEOMETRY] top_z_m=%.4f required_width=%.4f" % (float(ops_face["top_z_m"]), req_w))
                                elif _lb == "chips_can":
                                    self.get_logger().info("[CHIPS_GRASP_GEOMETRY] top_z_m=%.4f required_width=%.4f" % (float(ops_face["top_z_m"]), req_w))
                    pmeta_prof = pose_meta.pop("_profile_ms", None)
                    if isinstance(pmeta_prof, dict):
                        prof["top_face"] += pmeta_prof.get("top_face", 0.0)
                        prof["hybrid"] += pmeta_prof.get("hybrid", 0.0)
                        prof["global_search"] += pmeta_prof.get("global_search", 0.0)
                        prof["model_cuboid"] = pmeta_prof.get("model_cuboid", 0.0)

                    if str(pose_meta.get("yaw_source", "")).strip() == "unavailable":
                        self.get_logger().info(
                            "[YOLO_FILTER_DECISION] label=%s score=%.4f stage=geometry passed=false reason=yaw_unavailable"
                            % (det_label, det_score)
                        )
                        continue

                    tz = _safe_float(grasp_fields.get("top_z_m"), 0.0) or 0.0
                    grasp_center_base: Optional[List[float]] = None
                    grasp_center_source = "chosen_target_center"
                    grasp_center_source_build = grasp_center_source
                    kbc = pose_meta.get("known_box_center_base")
                    if self._runtime_gt_spawn_has_operational_center(pose_meta):
                        ch_ops = center_info.get("chosen_target_center_base")
                        if (
                            isinstance(ch_ops, (list, tuple))
                            and len(ch_ops) >= 2
                        ):
                            grasp_center_base = [
                                float(ch_ops[0]),
                                float(ch_ops[1]),
                                float(tz),
                            ]
                        else:
                            grasp_center_base = [float(kbc[0]), float(kbc[1]), tz]
                        grasp_center_source = str(
                            pose_meta.get("grasp_center_source")
                            or center_info.get("grasp_center_source")
                            or (
                                "runtime_gt_box_center"
                                if str(pose_meta.get("top_face_source", "")).strip()
                                == "runtime_gt_known_box"
                                else "runtime_gt_object_center"
                            )
                        )
                        grasp_center_source_build = grasp_center_source
                    elif self._model_fit_has_operational_center(pose_meta):
                        mbc = pose_meta.get("model_box_center_base")
                        grasp_center_base = [float(mbc[0]), float(mbc[1]), tz]
                        grasp_center_source = "model_box_center"
                        grasp_center_source_build = grasp_center_source
                    elif self._hybrid_fit_has_operational_center(pose_meta):
                        grasp_center_base = [float(kbc[0]), float(kbc[1]), tz]
                        grasp_center_source = "hybrid_known_center"
                        grasp_center_source_build = grasp_center_source
                    elif self._known_rectangle_fit_valid_for_operational_center(
                        policy, pose_meta
                    ):
                        if isinstance(kbc, (list, tuple)) and len(kbc) >= 3:
                            grasp_center_base = [float(kbc[0]), float(kbc[1]), tz]
                            grasp_center_source = "known_box_center"
                            grasp_center_source_build = grasp_center_source
                    if grasp_center_base is None:
                        ch = center_info.get("chosen_target_center_base")
                        if isinstance(ch, (list, tuple)) and len(ch) >= 3:
                            grasp_center_base = [float(ch[0]), float(ch[1]), tz]
                        elif centroid_base is not None:
                            grasp_center_base = [
                                float(centroid_base[0]),
                                float(centroid_base[1]),
                                tz,
                            ]

                    if grasp_center_base is not None:
                        grasp_fields["grasp_position"] = [
                            float(grasp_center_base[0] + self._top_grasp_offset[0]),
                            float(grasp_center_base[1] + self._top_grasp_offset[1]),
                            float(tz + self._top_grasp_offset[2]),
                    ]
                    grasp_fields["approach_position"] = [
                        float(grasp_fields["grasp_position"][0] + self._top_grasp_approach_offset[0]),
                        float(grasp_fields["grasp_position"][1] + self._top_grasp_approach_offset[1]),
                        float(grasp_fields["grasp_position"][2] + self._top_grasp_approach_offset[2]),
                    ]

                    yaw_value = _safe_float(grasp_fields.get("grasp_yaw_rad"), 0.0) or 0.0
                    pca_yaw_value = _safe_float(
                        pose_meta.get("pca_object_yaw_rad", grasp_fields.get("grasp_yaw_rad")),
                        yaw_value,
                    ) or yaw_value
                    footprint_major_m = float(grasp_fields.get("footprint_major_m", 0.0))
                    footprint_minor_m = float(grasp_fields.get("footprint_minor_m", 0.0))

                    build_stage = "closing_yaw_resolve"
                    closing_yaw, closing_yaw_source, closing_yaw_stage = (
                        self._resolve_closing_yaw_and_source(
                            str(det.label), pose_meta, policy, yaw_value
                        )
                    )
                    if closing_yaw is None:
                        self.get_logger().error(
                            "[PERCEPTION_OBJECT_BUILD_ERROR] label=%s stage=%s "
                            "reason=missing_closing_yaw closing_yaw_defined=false "
                            "closing_yaw_rad=%s closing_yaw_source=%s top_face_source=%s "
                            "grasp_center_source=%s yaw_source=%s"
                            % (
                                str(det.label),
                                closing_yaw_stage,
                                pose_meta.get("model_closing_yaw_rad"),
                                str(pose_meta.get("closing_yaw_source", "")),
                                str(pose_meta.get("top_face_source", "")),
                                str(
                                    pose_meta.get("grasp_center_source", grasp_center_source)
                                ),
                                str(pose_meta.get("yaw_source", "")),
                            )
                        )
                        continue

                    if str(det.label).strip().lower() == "mustard_bottle":
                        mustard_axis = apply_mustard_bottle_axis_semantics(
                            label=str(det.label),
                            mapping=self._mustard_bottle_axis_mapping,
                            pose_meta=pose_meta,
                            grasp_fields=grasp_fields,
                            logger=self.get_logger(),
                        )
                        closing_yaw = _safe_float(
                            pose_meta.get("closing_yaw_rad"), closing_yaw
                        )
                        closing_yaw_source = str(
                            pose_meta.get("closing_yaw_source") or closing_yaw_source
                        )
                        if not mustard_axis.get("publish_allowed", True):
                            self.get_logger().error(
                                "[PERCEPTION_OBJECT_BUILD_ERROR] label=mustard_bottle "
                                "stage=mustard_axis_semantics "
                                "axis_debug_result=%s width_sanity_result=%s "
                                "reason=mustard_perception_axis_invalid"
                                % (
                                    str(mustard_axis.get("axis_debug_result", "FAIL")),
                                    str(mustard_axis.get("width_sanity_result", "FAIL")),
                                )
                            )
                            continue
                        yaw_value = _safe_float(
                            pose_meta.get("finger_pad_yaw_rad"),
                            _safe_float(grasp_fields.get("grasp_yaw_rad"), yaw_value),
                        ) or yaw_value
                        pose_meta.setdefault("label", str(det.label))
                        _cap_idx = int(self._mustard_cap_offset_candidate_index)
                        _geom_cap = str(
                            pose_meta.get("grasp_center_source") or ""
                        ) in (
                            MUSTARD_TOP_CAP_CENTER_GEOMETRY_SOURCE,
                            MUSTARD_VERTICAL_AXIS_CAP_CENTER_SOURCE,
                        )
                        _manual_cap_tune = (
                            _cap_idx >= 0
                            or abs(self._mustard_cap_offset_long_m) > 1e-6
                            or abs(self._mustard_cap_offset_short_m) > 1e-6
                        )
                        if _manual_cap_tune and not _geom_cap:
                            cap_cal = apply_mustard_cap_center_calibration(
                                pose_meta=pose_meta,
                                grasp_fields=grasp_fields,
                                center_info=center_info,
                                candidate_index=_cap_idx,
                                offset_long_m=(
                                    None
                                    if _cap_idx >= 0
                                    else self._mustard_cap_offset_long_m
                                ),
                                offset_short_m=(
                                    None
                                    if _cap_idx >= 0
                                    else self._mustard_cap_offset_short_m
                                ),
                                logger=self.get_logger(),
                            )
                        else:
                            cap_cal = {"applied": False, "result": "SKIP_GEOMETRY_CAP"}
                        if cap_cal.get("applied") and cap_cal.get("result") == "OK":
                            gcb = cap_cal.get("grasp_center_base")
                            if isinstance(gcb, (list, tuple)) and len(gcb) >= 3:
                                grasp_center_base = [
                                    float(gcb[0]),
                                    float(gcb[1]),
                                    float(gcb[2]),
                                ]
                                grasp_center_source = str(
                                    cap_cal.get("cap_center_source", grasp_center_source)
                                )
                                grasp_center_source_build = grasp_center_source
                                if grasp_center_base is not None:
                                    grasp_fields["grasp_position"] = [
                                        float(grasp_center_base[0] + self._top_grasp_offset[0]),
                                        float(grasp_center_base[1] + self._top_grasp_offset[1]),
                                        float(grasp_center_base[2] + self._top_grasp_offset[2]),
                                    ]
                                    grasp_fields["approach_position"] = [
                                        float(
                                            grasp_fields["grasp_position"][0]
                                            + self._top_grasp_approach_offset[0]
                                        ),
                                        float(
                                            grasp_fields["grasp_position"][1]
                                            + self._top_grasp_approach_offset[1]
                                        ),
                                        float(
                                            grasp_fields["grasp_position"][2]
                                            + self._top_grasp_approach_offset[2]
                                        ),
                                    ]

                    build_stage = "uv_projection"

                    chosen_center_base = center_info["chosen_target_center_base"]
                    chosen_uv = None
                    grasp_center_uv = None
                    obb_center_uv = None
                    if det.obb_polygon_uv is not None:
                        obb_center_uv = [
                            float(np.mean(det.obb_polygon_uv[:, 0])),
                            float(np.mean(det.obb_polygon_uv[:, 1])),
                        ]
                    if tf_msg is not None:
                        tf_cam_to_base = self._build_transform_matrix(tf_msg)
                        tf_base_to_cam = np.linalg.inv(tf_cam_to_base)
                    if tf_msg is not None and chosen_center_base is not None:
                        h_chosen = np.array(
                            [
                                chosen_center_base[0],
                                chosen_center_base[1],
                                chosen_center_base[2],
                                1.0,
                            ],
                            dtype=np.float64,
                        )
                        chosen_cam = (tf_base_to_cam @ h_chosen)[:3]
                        if chosen_cam[2] > 1e-6:
                            cu = int(
                                np.clip((chosen_cam[0] * fx / chosen_cam[2]) + cx, 0, w - 1)
                            )
                            cv = int(
                                np.clip((chosen_cam[1] * fy / chosen_cam[2]) + cy, 0, h - 1)
                            )
                            chosen_uv = [float(cu), float(cv)]
                    if tf_msg is not None and grasp_center_base is not None:
                        hg = np.array(
                            [
                                grasp_center_base[0],
                                grasp_center_base[1],
                                grasp_center_base[2],
                                1.0,
                            ],
                            dtype=np.float64,
                        )
                        gcam = (tf_base_to_cam @ hg)[:3]
                        if gcam[2] > 1e-6:
                            gu = int(np.clip((gcam[0] * fx / gcam[2]) + cx, 0, w - 1))
                            gv = int(np.clip((gcam[1] * fy / gcam[2]) + cy, 0, h - 1))
                            grasp_center_uv = [float(gu), float(gv)]
                    if self._runtime_gt_spawn_has_operational_center(pose_meta):
                        self.get_logger().info(
                            "[KNOWN_OBJECT_OVERLAY_DEBUG] label=%s projected_center_px=%s projected_top_face_px=%s "
                            "closing_yaw_rad=%.4f closing_yaw_source=%s source=%s"
                            % (
                                str(det.label),
                                str(grasp_center_uv),
                                str(chosen_uv),
                                float(closing_yaw),
                                str(closing_yaw_source or ""),
                                str(pose_meta.get("top_face_source", "")),
                            )
                        )

                    c_err_m = pose_meta.get("observed_vs_model_corner_error_m")
                    if (
                        self._model_fit_has_operational_center(pose_meta)
                        and c_err_m is not None
                        and math.isfinite(float(c_err_m))
                        and float(c_err_m) > 0.025
                        and float(pose_meta.get("yaw_confidence", 0.0)) >= 0.65
                    ):
                        z_est = max(float(tz), 0.25)
                        err_px = float(c_err_m) * float(fx) / z_est
                        self.get_logger().warn(
                            "[TOP_FACE_MODEL_DISAGREEMENT] label=%s "
                            "observed_vs_model_corner_error_px=%.1f "
                            "observed_vs_model_yaw_deg=%.2f action=use_known_model"
                            % (
                                str(det.label),
                                err_px,
                                float(pose_meta.get("observed_vs_model_yaw_deg", 0.0)),
                            )
                        )

                    model_tc_gate = pose_meta.get("model_top_face_corners_base")
                    if not (
                        isinstance(model_tc_gate, list) and len(model_tc_gate) >= 4
                    ):
                        model_tc_gate = pose_meta.get("top_corners_base")
                    obs_tc_gate = pose_meta.get("top_corners_base")
                    if (
                        (self._use_runtime_scene_gt or self._debug_draw_gazebo_gt)
                        and tf_msg is not None
                    ):
                        try:
                            _tf_cam = self._build_transform_matrix(tf_msg)
                            _tf_b2c = np.linalg.inv(_tf_cam)

                            def _proj_uv_gate(c: List[float]) -> Optional[Tuple[int, int]]:
                                return self._project_base_xyz_to_uv(
                                    c,
                                    _tf_b2c,
                                    float(fx),
                                    float(fy),
                                    float(cx),
                                    float(cy),
                                    w,
                                    h,
                                )

                            pose_meta = self._apply_visual_gt_metrics_and_gate(
                                det,
                                pose_meta,
                                model_corners=model_tc_gate
                                if isinstance(model_tc_gate, list)
                                else None,
                                observed_corners=obs_tc_gate
                                if isinstance(obs_tc_gate, list)
                                else None,
                                fx=float(fx),
                                z_est_m=float(grasp_fields.get("top_z_m") or 0.35),
                                project_uv_fn=_proj_uv_gate,
                                obb_center_uv=obb_center_uv,
                            )
                        except np.linalg.LinAlgError:
                            pass

                    if self._publish_debug and self._draw_debug_overlay:
                        center_info_overlay = dict(center_info)
                        center_info_overlay["grasp_center_source"] = grasp_center_source
                        deferred_overlays.append(
                            {
                                "debug": debug,
                                "det_index": i_det,
                                "det": det,
                                "tf_msg": tf_msg,
                                "fx": float(fx),
                                "fy": float(fy),
                                "cx": float(cx),
                                "cy": float(cy),
                                "w": w,
                                "h": h,
                                "grasp_fields": grasp_fields,
                                "center_info": center_info_overlay,
                                "pose_meta": pose_meta,
                                "closing_yaw_rad": closing_yaw,
                                "pts_obj_base": pts_obj_base,
                                "grasp_center_base": grasp_center_base,
                                "chosen_center_uv": chosen_uv,
                                "grasp_center_uv": grasp_center_uv,
                                "obb_center_uv": obb_center_uv,
                            }
                        )

                    if (
                        str(det.label).strip().lower() == "cracker_box"
                        and str(pose_meta.get("yaw_source")) == "pca_raw"
                        and float(pose_meta.get("yaw_confidence", 0.0)) < 0.8
                    ):
                        self.get_logger().warn(
                            "cracker_box requires accurate yaw; pca_raw yaw is not reliable enough"
                        )

                    edge_requested = (
                        _is_edge_strategy(policy.get("primary_strategy", ""))
                        or bool(policy.get("prefer_edge", False))
                        or bool(policy.get("prefer_push_to_edge", False))
                    )

                    col_dims = get_collision_dimensions(det.label, padding_m=0.0)
                    collision_dims_json: Optional[Dict[str, Any]] = None
                    if col_dims is not None:
                        collision_dims_json = {
                            "shape": col_dims.get("shape"),
                            "box": [float(v) for v in col_dims["box"]] if "box" in col_dims else None,
                            "cylinder": (
                                [float(v) for v in col_dims["cylinder"]]
                                if "cylinder" in col_dims
                                else None
                            ),
                            "box_fallback": [float(v) for v in col_dims["box_fallback"]],
                            "db_dims": [float(v) for v in col_dims["db_dims"]],
                        }

                    build_stage = "object_ready"
                    _is_mustard_pub = str(det.label).strip().lower() == "mustard_bottle"
                    _finger_pad_yaw = _safe_float(pose_meta.get("finger_pad_yaw_rad"))
                    _grasp_gap_yaw = _safe_float(
                        pose_meta.get("grasp_gap_yaw_rad"), closing_yaw
                    )
                    obj_entry: Dict[str, Any] = {
                        "id": det.label,
                        "label": det.label,
                        "score": det.score,
                        "inference_ms": det.inference_ms,
                        "centroid_camera": [float(cc[0]), float(cc[1]), float(cc[2])],
                        "centroid_base": centroid_base,
                        "pointcloud_centroid_base": center_info["pointcloud_centroid_base"],
                        "pointcloud_median_base": center_info["pointcloud_median_base"],
                        "top_surface_center_base": center_info["top_surface_center_base"],
                        "chosen_target_center_base": center_info["chosen_target_center_base"],
                        "target_center_method": center_info["target_center_method"],
                        "grasp_center_base": grasp_center_base,
                        "grasp_center_source": grasp_center_source,
                        "position": grasp_center_base
                        if grasp_center_base is not None
                        else (center_info["chosen_target_center_base"] or centroid_base),
                        "bbox_center_uv": [float(proj_u), float(proj_v)],
                        "chosen_center_uv": chosen_uv,
                        "grasp_center_uv": grasp_center_uv,
                        "obb_center_uv": None if det.obb_polygon_uv is None else [float(np.mean(det.obb_polygon_uv[:, 0])), float(np.mean(det.obb_polygon_uv[:, 1]))],
                        "dimensions_m": grasp_fields["dimensions_m"],
                        "top_z_m": grasp_fields["top_z_m"],
                        "grasp_yaw_rad": (
                            float(_finger_pad_yaw)
                            if _is_mustard_pub and _finger_pad_yaw is not None
                            else grasp_fields["grasp_yaw_rad"]
                        ),
                        "grasp_yaw_deg": (
                            float(math.degrees(_finger_pad_yaw))
                            if _is_mustard_pub and _finger_pad_yaw is not None
                            else grasp_fields["grasp_yaw_deg"]
                        ),
                        "object_yaw_rad": (
                            float(_finger_pad_yaw)
                            if _is_mustard_pub and _finger_pad_yaw is not None
                            else pca_yaw_value
                        ),
                        "active_yaw_rad": (
                            float(_finger_pad_yaw)
                            if _is_mustard_pub and _finger_pad_yaw is not None
                            else yaw_value
                        ),
                        "closing_yaw_rad": (
                            float(_grasp_gap_yaw)
                            if _is_mustard_pub and _grasp_gap_yaw is not None
                            else closing_yaw
                        ),
                        "closing_yaw_source": closing_yaw_source,
                        "grasp_gap_axis_xy": pose_meta.get("grasp_gap_axis_xy"),
                        "finger_pad_axis_xy": pose_meta.get("finger_pad_axis_xy"),
                        "grasp_gap_yaw_rad": pose_meta.get("grasp_gap_yaw_rad"),
                        "finger_pad_yaw_rad": pose_meta.get("finger_pad_yaw_rad"),
                        "closing_yaw_semantics": pose_meta.get("closing_yaw_semantics"),
                        "commanded_tcp_yaw_rad_hint": pose_meta.get(
                            "commanded_tcp_yaw_rad_hint"
                        ),
                        "approach_position": grasp_fields["approach_position"],
                        "grasp_position": grasp_fields["grasp_position"],
                        "height_m": grasp_fields["dimensions_m"][2],
                        "footprint_major_m": footprint_major_m,
                        "footprint_minor_m": footprint_minor_m,
                        "measured_height_m": measured_height_m,
                        "major_axis_xy": grasp_fields.get("major_axis_xy"),
                        "minor_axis_xy": grasp_fields.get("minor_axis_xy"),
                        "top_face_success": bool(pose_meta.get("top_face_success")),
                        "top_face_method": str(pose_meta.get("top_face_method", "none")),
                        "top_face_num_points": int(pose_meta.get("top_face_num_points", 0)),
                        "top_face_point_ratio": float(pose_meta.get("top_face_point_ratio", 0.0)),
                        "top_z_estimated": float(pose_meta.get("top_z_estimated", grasp_fields["top_z_m"])),
                        "yaw_source": str(pose_meta.get("yaw_source", "pca_raw")),
                        "yaw_confidence": float(pose_meta.get("yaw_confidence", 0.0)),
                        "pose_fit_success": bool(pose_meta.get("pose_fit_success")),
                        "pose_fit_error": pose_meta.get("pose_fit_error"),
                        "yaw_fit_method": pose_meta.get("yaw_fit_method"),
                        "center_method": pose_meta.get("center_method"),
                        "center_shift_from_median_m": pose_meta.get("center_shift_from_median_m"),
                        "projected_extent_length_m": pose_meta.get("projected_extent_length_m"),
                        "projected_extent_width_m": pose_meta.get("projected_extent_width_m"),
                        "length_error_m": pose_meta.get("length_error_m"),
                        "width_error_m": pose_meta.get("width_error_m"),
                        "outside_error_m": pose_meta.get("outside_error_m"),
                        "inlier_ratio": pose_meta.get("inlier_ratio"),
                        "edge_support_score": pose_meta.get("edge_support_score"),
                        "yaw_margin_score": pose_meta.get("yaw_margin_score"),
                        "best_score": pose_meta.get("best_score"),
                        "second_best_score": pose_meta.get("second_best_score"),
                        "num_yaw_candidates": pose_meta.get("num_yaw_candidates"),
                        "selected_yaw_deg": pose_meta.get("selected_yaw_deg"),
                        "top_corners_base": pose_meta.get("top_corners_base"),
                        "bottom_corners_base": pose_meta.get("bottom_corners_base"),
                        "known_box_center_base": pose_meta.get("known_box_center_base"),
                        "known_box_yaw_rad": pose_meta.get("known_box_yaw_rad"),
                        "long_axis_xy": pose_meta.get("long_axis_xy") or grasp_fields.get("major_axis_xy"),
                        "short_axis_xy": pose_meta.get("short_axis_xy") or grasp_fields.get("minor_axis_xy"),
                        "partial_top_face_detected": pose_meta.get("partial_top_face_detected"),
                        "hybrid_fit_success": pose_meta.get("hybrid_fit_success"),
                        "center_fit_method": pose_meta.get("center_fit_method"),
                        "center_offset_long_m": pose_meta.get("center_offset_long_m"),
                        "center_offset_short_m": pose_meta.get("center_offset_short_m"),
                        "observed_extent_length_m": pose_meta.get("observed_extent_length_m"),
                        "observed_extent_width_m": pose_meta.get("observed_extent_width_m"),
                        "db_length_m": pose_meta.get("db_length_m"),
                        "db_width_m": pose_meta.get("db_width_m"),
                        "fit_reject_reason": pose_meta.get("fit_reject_reason"),
                        "top_face_observed_center_base": pose_meta.get(
                            "top_face_observed_center_base"
                        ),
                        "top_face_source": pose_meta.get("top_face_source"),
                        "operational_source_fallback": bool(
                            pose_meta.get("runtime_gt_geometry_applied", False)
                        )
                        is False,
                        "observed_top_face_success": pose_meta.get(
                            "observed_top_face_success"
                        ),
                        "observed_top_face_num_points": pose_meta.get(
                            "observed_top_face_num_points"
                        ),
                        "observed_top_face_ratio": pose_meta.get(
                            "observed_top_face_ratio"
                        ),
                        "model_top_face_success": pose_meta.get(
                            "model_top_face_success"
                        ),
                        "model_fit_error": pose_meta.get("model_fit_error"),
                        "model_top_face_corners_base": pose_meta.get(
                            "model_top_face_corners_base"
                        ),
                        "model_box_center_base": pose_meta.get("model_box_center_base"),
                        "model_box_yaw_rad": pose_meta.get("model_box_yaw_rad"),
                        "model_closing_yaw_rad": pose_meta.get("model_closing_yaw_rad"),
                        "model_major_axis_xy": pose_meta.get("model_major_axis_xy"),
                        "model_minor_axis_xy": pose_meta.get("model_minor_axis_xy"),
                        "observed_vs_model_corner_error_m": pose_meta.get(
                            "observed_vs_model_corner_error_m"
                        ),
                        "observed_vs_model_yaw_deg": pose_meta.get(
                            "observed_vs_model_yaw_deg"
                        ),
                        "visual_pose_gate_passed": pose_meta.get(
                            "visual_pose_gate_passed"
                        ),
                        "visual_pose_gate_reason": pose_meta.get(
                            "visual_pose_gate_reason"
                        ),
                        "model_vs_gt_center_error_xy_m": pose_meta.get(
                            "model_vs_gt_center_error_xy_m"
                        ),
                        "model_vs_gt_yaw_error_deg": pose_meta.get(
                            "model_vs_gt_yaw_error_deg"
                        ),
                        "model_vs_gt_corner_error_m": pose_meta.get(
                            "model_vs_gt_corner_error_m"
                        ),
                        "observed_vs_gt_center_error_xy_m": pose_meta.get(
                            "observed_vs_gt_center_error_xy_m"
                        ),
                        "observed_vs_model_center_error_xy_m": pose_meta.get(
                            "observed_vs_model_center_error_xy_m"
                        ),
                        "entity_name": (
                            str(pose_meta.get("gt_entity_name") or "").split("::")[-1]
                            if pose_meta.get("gt_entity_name")
                            else ""
                        ),
                        "gt_entity_name": pose_meta.get("gt_entity_name"),
                        "fit_candidate_yaw_deg": pose_meta.get("selected_yaw_deg"),
                        "active_yaw_deg": float(np.degrees(yaw_value)),
                        "plane_normal": [
                            float(ts.plane_normal[0]),
                            float(ts.plane_normal[1]),
                            float(ts.plane_normal[2]),
                        ],
                        "top_surface_success": ts.success,
                        "top_surface_message": ts.message,
                        "open3d_inlier_ratio": ts.inlier_ratio,
                        "open3d_ransac_rmse": ts.ransac_rmse,
                        "obb_polygon_uv": obb_list,
                        "grasp_pose": grasp_pose_dict,
                        "grasp_candidates": candidates_dict,
                        **policy_exec,
                        "collision_dims": collision_dims_json,
                    }
                    if _is_mustard_pub:
                        for _mk in (
                            "mustard_old_offset_cap_center_base",
                            "mustard_vertical_axis_cap_center_base",
                            "mustard_mesh_local_cap_center_base",
                            "tall_object_body_center_base",
                            "mustard_cap_center_mode",
                        ):
                            if pose_meta.get(_mk) is not None:
                                obj_entry[_mk] = pose_meta[_mk]
                        apply_mustard_bottle_axis_semantics(
                            label=str(det.label),
                            mapping=self._mustard_bottle_axis_mapping,
                            pose_meta=pose_meta,
                            grasp_fields=grasp_fields,
                            obj_entry=obj_entry,
                            logger=self.get_logger(),
                        )
                    self._normalize_operational_source_contract(obj_entry)
                    self.get_logger().info(
                        "[YOLO_FILTER_DECISION] label=%s score=%.4f stage=geometry passed=true reason=object_ready"
                        % (det_label, det_score)
                    )
                    after_geometry_count += 1
                    self.get_logger().info(
                        "[PERCEPTION_OBJECT_READY] label=%s closing_yaw=%.4f "
                        "closing_yaw_source=%s top_face_source=%s grasp_center_source=%s "
                        "yaw_source=%s operational_source_fallback=%s"
                        % (
                            str(det.label),
                            float(closing_yaw),
                            str(obj_entry.get("closing_yaw_source") or ""),
                            str(obj_entry.get("top_face_source", "")),
                            str(obj_entry.get("grasp_center_source", "")),
                            str(obj_entry.get("yaw_source", "")),
                            str(
                                bool(obj_entry.get("operational_source_fallback", False))
                            ).lower(),
                        )
                    )
                    objects_out.append(obj_entry)

                    pms = pose_meta.get("_profile_ms")
                    if isinstance(pms, dict):
                        prof["top_face"] += float(pms.get("top_face", 0.0))
                        prof["model_cuboid"] += float(pms.get("model_cuboid", 0.0))
                        prof["hybrid"] += float(pms.get("hybrid", 0.0))
                        prof["global_search"] += float(pms.get("global_search", 0.0))
                    if should_log_tf or policy["risk_level"] == "high":
                        self.get_logger().info(
                            "[GRASP_POLICY] label=%s strategy=%s risk=%s src=%s yaw=%.3f closing_yaw=%.3f "
                            "req_width=%.3f db_req=%.3f meas_req=%s "
                            "open_joint=%.4f close_joint=%.4f depth_from_top=%.3f "
                            "footprint=(%.3f, %.3f) height=%.3f db_h=%.3f eff_h=%.3f edge=%s"
                            % (
                                det.label,
                                policy["primary_strategy"],
                                policy["risk_level"],
                                policy.get("dimension_source", "?"),
                                yaw_value,
                                closing_yaw,
                                policy["required_grasp_width_m"],
                                float(policy.get("db_required_width_m") or 0.0),
                                ("n/a" if policy.get("measured_required_width_m") is None
                                 else ("%.3f" % float(policy["measured_required_width_m"]))),
                                policy["recommended_open_joint_m"],
                                policy["recommended_close_joint_m"],
                                policy["recommended_grasp_depth_from_top_m"],
                                footprint_major_m,
                                footprint_minor_m,
                                measured_height_m,
                                float(policy.get("db_height_m") or 0.0),
                                float(policy.get("effective_height_m") or 0.0),
                                str(edge_requested).lower(),
                            )
                        )
                    if should_log_tf:
                        self.get_logger().info(
                            "Centro objetivo: method=%s bbox_center_uv=%s obb_center_uv=%s chosen_center_base=%s chosen_center_uv=%s"
                            % (
                                center_info["target_center_method"],
                                [float(proj_u), float(proj_v)],
                                None
                                if det.obb_polygon_uv is None
                                else [
                                    float(np.mean(det.obb_polygon_uv[:, 0])),
                                    float(np.mean(det.obb_polygon_uv[:, 1])),
                                ],
                                center_info["chosen_target_center_base"],
                                chosen_uv,
                            )
                        )
            except Exception as exc:
                det_label = str(getattr(det, "label", "?"))
                self.get_logger().error(
                    "[PERCEPTION_OBJECT_BUILD_ERROR] label=%s stage=%s error=%s "
                    "closing_yaw_defined=%s closing_yaw_rad=%s closing_yaw_source=%s "
                    "top_face_source=%s grasp_center_source=%s yaw_source=%s\n%s"
                    % (
                        det_label,
                        build_stage,
                        exc,
                        str(closing_yaw is not None).lower(),
                        closing_yaw,
                        str(closing_yaw_source or ""),
                        str(
                            (pose_meta_build or {}).get("top_face_source", "")
                        ),
                        str(grasp_center_source_build),
                        str((pose_meta_build or {}).get("yaw_source", "")),
                        traceback.format_exc(),
                    )
                )
                self.get_logger().error(
                    "[PERCEPTION] detection skipped due to exception label=%s error=%s"
                    % (det_label, exc)
                )

        mean_inlier = float(np.mean(inlier_ratios)) if inlier_ratios else 0.0
        mean_rmse = float(np.mean(rmses)) if rmses else 0.0
        max_gconf = float(max(grasp_confs)) if grasp_confs else 0.0
        max_glat = float(max(grasp_latencies)) if grasp_latencies else 0.0

        telem = PerceptionTelemetry(
            vision_backend=self._vision.backend_id,
            vision_model_name=self._vision.model_name,
            vision_inference_ms_total=vision_total_ms,
            vision_inference_ms_per_detection=[d.inference_ms for d in detections],
            text_prompt_used=prompt,
            open3d_ms_total=o3d_ms_acc,
            open3d_plane_inlier_ratio=mean_inlier,
            open3d_ransac_rmse=mean_rmse,
            grasp_backend=self._grasp.backend_id,
            grasp_confidence=max_gconf,
            grasp_service_latency_ms=max_glat,
        )

        scene_ctx = self._build_runtime_scene_executor_context(objects_out)
        after_runtime_association_count = 0
        after_operational_count = 0
        for obj in objects_out:
            label = str(obj.get("label") or "").strip().lower()
            score = _safe_float(obj.get("score"), 0.0) or 0.0
            ent = str(obj.get("entity_name") or obj.get("gt_entity_name") or "").strip()
            runtime_ok = True
            runtime_reason = "runtime_scene_not_required"
            if self._use_runtime_scene_gt:
                runtime_ok = bool(ent) and self._entity_in_runtime_scene(ent)
                runtime_reason = (
                    "entity_in_runtime_scene" if runtime_ok else "entity_not_in_runtime_scene"
                )
            if runtime_ok:
                after_runtime_association_count += 1
            self.get_logger().info(
                "[YOLO_FILTER_DECISION] label=%s score=%.4f stage=runtime_association passed=%s reason=%s"
                % (label, score, str(runtime_ok).lower(), runtime_reason)
            )
            operational_ok = runtime_ok and self._is_operational_detection(obj)
            if operational_ok:
                after_operational_count += 1
            self.get_logger().info(
                "[YOLO_FILTER_DECISION] label=%s score=%.4f stage=operational passed=%s reason=%s"
                % (
                    label,
                    score,
                    str(operational_ok).lower(),
                    "operational_contract_ok" if operational_ok else "operational_contract_reject",
                )
            )

        if len(objects_out) == 0:
            reason = "unknown"
            if raw_yolo_count == 0:
                reason = "no_raw_yolo"
            elif after_confidence_count == 0:
                reason = "confidence_filter"
            elif after_depth_count == 0:
                reason = "depth_filter"
            elif after_table_count == 0:
                reason = "table_filter"
            elif after_runtime_association_count == 0:
                reason = "runtime_association_filter"
            elif after_operational_count == 0:
                reason = "operational_filter"
            self.get_logger().info(
                "[DETECTIONS_3D_EMPTY]\n"
                "raw_yolo_count=%d\n"
                "after_confidence_count=%d\n"
                "after_depth_count=%d\n"
                "after_table_count=%d\n"
                "after_runtime_association_count=%d\n"
                "after_operational_count=%d\n"
                "reason=%s"
                % (
                    raw_yolo_count,
                    after_confidence_count,
                    after_depth_count,
                    after_table_count,
                    after_runtime_association_count,
                    after_operational_count,
                    reason,
                )
            )

        payload: Dict[str, Any] = {
            "schema_version": "1.1",
            "stamp_sec": float(image_msg.header.stamp.sec)
            + float(image_msg.header.stamp.nanosec) * 1e-9,
            "frame_id": self._target_frame,
            "camera_frame_id": cam_frame,
            "telemetry": telem.to_dict(),
            "objects": objects_out,
            "scene_objects": scene_ctx.get("scene_objects") or [],
            "obstacles": scene_ctx.get("obstacles") or [],
            "target_candidate": scene_ctx.get("target_candidate"),
        }

        n_executor_valid = self._cache_executor_payload(payload)

        if self._is_processing_paused():
            self._log_executor_skip(
                "paused",
                valid_objects_count=n_executor_valid,
                publish_source="direct_frame",
            )
            if self._publish_debug:
                self._publish_debug_status_image_bgr(
                    self._make_fallback_debug_bgr("processing paused by executor"),
                    cam_frame,
                    banner="processing paused by executor",
                )
            return

        if (
            self._executor_publish_on_every_valid
            and n_executor_valid > 0
        ):
            t_pub0 = time.perf_counter()
            self._publish_vision_to_executor_payload(
                payload,
                reason="valid_frame",
                publish_source="direct_frame",
                bypass_busy_check=True,
            )
            prof["publish"] += (time.perf_counter() - t_pub0) * 1000.0
        elif n_executor_valid == 0:
            self._log_executor_skip(
                "no_valid_objects",
                valid_objects_count=0,
                bypass_busy_check=True,
                publish_source="direct_frame",
            )

        if len(detections) == 0:
            self._log_executor_skip(
                "no_detections",
                valid_objects_count=n_executor_valid,
                bypass_busy_check=True,
                publish_source="direct_frame",
            )

        prof["total"] = (time.perf_counter() - t_frame0) * 1000.0
        self._maybe_log_perception_timing(prof)
        if self._enable_perception_profiling:
            self.get_logger().info(
                "[PERCEPTION_PROFILE] total=%.1f yolo=%.1f rgbd=%.1f top_face=%.1f "
                "hybrid=%.1f global_search=%.1f overlay=%.1f publish=%.1f objects=%d"
                % (
                    prof["total"],
                    prof["yolo"],
                    prof["rgbd"],
                    prof["top_face"],
                    prof["hybrid"],
                    prof["global_search"],
                    prof["overlay"],
                    prof["publish"],
                    len(objects_out),
                )
            )

        if self._publish_legacy:
            legacy = {
                "backend": self._vision.backend_id,
                "frame_id": self._target_frame,
                "detections": objects_out,
            }
            lm = String()
            lm.data = json.dumps(legacy)
            self._legacy_pub.publish(lm)

        banner = None
        if len(detections) == 0:
            banner = "no detections"
        elif len(objects_out) == 0:
            banner = "no valid objects"
        rich_label = "none"
        if objects_out:
            rich_label = str(objects_out[0].get("label") or "object")
        elif detections:
            rich_label = str(detections[0].label)
        rich_reason = (
            "processed"
            if objects_out
            else ("no_valid_objects" if detections else "no_detections")
        )
        has_overlays = bool(
            self._draw_debug_overlay and (len(detections) > 0 or len(objects_out) > 0)
        )
        if self._publish_debug and deferred_overlays:
            for ov in deferred_overlays:
                t_ov0 = time.perf_counter()
                self._draw_grasp_debug_overlay(
                    ov["debug"],
                    ov["det_index"],
                    ov["det"],
                    ov["tf_msg"],
                    ov["fx"],
                    ov["fy"],
                    ov["cx"],
                    ov["cy"],
                    ov["w"],
                    ov["h"],
                    ov["grasp_fields"],
                    ov["center_info"],
                    ov["pose_meta"],
                    ov["closing_yaw_rad"],
                    ov["pts_obj_base"],
                    ov["grasp_center_base"],
                    chosen_center_uv=ov.get("chosen_center_uv"),
                    grasp_center_uv=ov.get("grasp_center_uv"),
                    obb_center_uv=ov.get("obb_center_uv"),
                )
                prof["overlay"] += (time.perf_counter() - t_ov0) * 1000.0
        if self._publish_debug:
            self._cache_and_publish_rich_debug(
                debug,
                cam_frame,
                len(detections),
                len(objects_out),
                label=rich_label,
                reason=rich_reason,
                has_overlays=has_overlays,
                banner=banner,
            )


def main(args: Optional[List[str]] = None) -> None:
    rclpy.init(args=args)
    node = PerceptionNode()
    exe = MultiThreadedExecutor(num_threads=6)
    exe.add_node(node)
    try:
        exe.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        try:
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    main()
