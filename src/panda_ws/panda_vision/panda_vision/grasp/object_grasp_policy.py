"""Per-object grasp policy database for the Franka Panda parallel gripper.

This module never advocates for changing the end-effector. The Franka Hand
(parallel jaw, ~80 mm max opening) is assumed at all times.

For objects that are larger than the gripper opening, the policy keeps
``parallel_jaw_allowed = True`` and falls back to edge-style strategies
(edge_grasp, push_to_edge_then_grasp, partial_grasp_best_effort) so the
execution layer can still attempt a grasp without changing the EEF.
"""

from __future__ import annotations

import math
import logging
from typing import Any, Dict, FrozenSet, List, Optional, Tuple


MAX_GRIPPER_WIDTH_M = 0.080
WIDTH_MARGIN_M = 0.004
SAFE_MAX_WIDTH_M = MAX_GRIPPER_WIDTH_M - 2 * WIDTH_MARGIN_M

MAX_FINGER_JOINT_M = 0.040
MIN_FINGER_JOINT_M = 0.000

LOW_OBJECT_HEIGHT_M = 0.040
VERY_LOW_OBJECT_HEIGHT_M = 0.030


_RISK_LOW = "low"
_RISK_MEDIUM = "medium"
_RISK_HIGH = "high"


_AXIS_SHORT = "short_axis"
_AXIS_DIAMETER = "diameter"
_AXIS_PERP_LONG = "perpendicular_to_long_axis"
_AXIS_EDGE = "edge"


OBJECT_DB: Dict[str, Dict[str, Any]] = {
    "cracker_box": {
        "shape": "box",
        "dims": (0.060, 0.158, 0.210),
        "primary_strategy": "top_down_short_axis",
        "fallback_strategies": ["oblique_short_axis", "lateral_short_axis"],
        "required_width": 0.060,
        "grasp_depth_from_top": 0.050,
        "risk": _RISK_LOW,
        "preferred_closing_axis": _AXIS_SHORT,
        "close_squeeze_m": 0.000,
        "min_pick_lift_m": 0.150,
        "min_carry_tcp_z_m": 0.700,
        "carry_clearance_above_table_m": 0.120,
        "carry_clearance_above_obstacles_m": 0.100,
        "attached_collision_padding_m": 0.020,
        "preferred_transport_corridor": "front_lane",
        "notes": "Pinza cierra por el lado corto (60 mm); cierre sin apretar.",
    },
    "sugar_box": {
        "shape": "box",
        "dims": (0.038, 0.089, 0.175),
        "primary_strategy": "top_down_short_axis",
        "fallback_strategies": ["lateral_short_axis", "oblique_short_axis"],
        "required_width": 0.038,
        "grasp_depth_from_top": 0.055,
        "risk": _RISK_LOW,
        "preferred_closing_axis": _AXIS_SHORT,
        "min_pick_lift_m": 0.150,
        "min_carry_tcp_z_m": 0.700,
        "carry_clearance_above_table_m": 0.120,
        "carry_clearance_above_obstacles_m": 0.100,
        "attached_collision_padding_m": 0.020,
        "preferred_transport_corridor": "front_lane",
        "notes": "Caja estrecha; cierre por eje corto (38 mm).",
    },
    "mustard_bottle": {
        "shape": "tall_box_like",
        "dims": (0.0577, 0.0953, 0.1909),
        "primary_strategy": "tall_object_topdown",
        "fallback_strategies": ["oblique_body_grasp", "side_grasp_tall_object"],
        "required_width": 0.058,
        "grasp_depth_from_top": 0.035,
        "risk": _RISK_MEDIUM,
        "preferred_closing_axis": _AXIS_SHORT,
        "min_pick_lift_m": 0.150,
        "min_carry_tcp_z_m": 0.700,
        "carry_clearance_above_table_m": 0.120,
        "carry_clearance_above_obstacles_m": 0.100,
        "attached_collision_padding_m": 0.020,
        "preferred_transport_corridor": "front_lane",
        "notes": "Cuerpo de la botella, no el tapon.",
    },
    "chips_can": {
        "shape": "cylinder",
        "diameter": 0.075,
        "height": 0.250,
        "primary_strategy": "cylinder_topdown",
        "fallback_strategies": ["oblique_cylinder_grasp", "lateral_cylinder_grasp"],
        "required_width": 0.075,
        "grasp_depth_from_top": 0.035,
        "risk": _RISK_MEDIUM,
        "preferred_closing_axis": _AXIS_DIAMETER,
        "close_squeeze_m": 0.001,
        "min_pick_lift_m": 0.150,
        "min_carry_tcp_z_m": 0.700,
        "carry_clearance_above_table_m": 0.120,
        "carry_clearance_above_obstacles_m": 0.100,
        "attached_collision_padding_m": 0.020,
        "preferred_transport_corridor": "front_lane",
        "notes": "Cilindro alto; yaw indiferente, cierre por diametro.",
    },
    "bleach_cleanser": {
        "shape": "tall_box_like",
        "dims": (0.1021, 0.0666, 0.2505),
        "primary_strategy": "tall_object_topdown",
        "fallback_strategies": ["oblique_body_grasp", "lateral_body_grasp"],
        "required_width": 0.067,
        "grasp_depth_from_top": 0.040,
        "risk": _RISK_MEDIUM,
        "preferred_closing_axis": _AXIS_SHORT,
        "close_squeeze_m": 0.001,
        "notes": "Botella alta; agarrar por el eje corto del cuerpo, cierre suave.",
    },
    "apple": {
        "shape": "sphere_like",
        "diameter": 0.075,
        "height": 0.075,
        "primary_strategy": "oblique_center_grasp",
        "fallback_strategies": ["top_down_center_grasp", "lateral_center_grasp"],
        "required_width": 0.075,
        "grasp_depth_from_top": 0.030,
        "risk": _RISK_MEDIUM,
        "preferred_closing_axis": _AXIS_DIAMETER,
        "close_squeeze_m": 0.001,
        "notes": "Esferico; yaw indiferente, cierre suave.",
    },
    "banana": {
        "shape": "curved_long",
        "dims": (0.198, 0.075, 0.037),
        "primary_strategy": "oblique_perpendicular_to_long_axis",
        "fallback_strategies": ["lateral_edge_grasp", "push_to_edge_then_grasp"],
        "required_width": 0.075,
        "grasp_depth_from_top": 0.020,
        "risk": _RISK_HIGH,
        "preferred_closing_axis": _AXIS_PERP_LONG,
        "close_squeeze_m": 0.000,
        "notes": "Objeto curvo y bajo; agarrar perpendicular al eje largo, sin apretar.",
    },
    "tuna_fish_can": {
        "shape": "low_cylinder",
        "diameter": 0.085,
        "height": 0.033,
        "primary_strategy": "edge_grasp",
        "fallback_strategies": [
            "oblique_edge_grasp",
            "push_to_edge_then_grasp",
            "partial_grasp_best_effort",
        ],
        "required_width": 0.085,
        "grasp_depth_from_top": 0.015,
        "risk": _RISK_HIGH,
        "preferred_closing_axis": _AXIS_EDGE,
        "notes": "Diametro 85 mm > 76 mm utiles: edge grasp por borde.",
    },
    "potted_meat_can": {
        "shape": "box",
        "dims": (0.050, 0.097, 0.082),
        "primary_strategy": "top_down_short_axis",
        "fallback_strategies": ["oblique_short_axis", "lateral_short_axis"],
        "required_width": 0.050,
        "grasp_depth_from_top": 0.040,
        "risk": _RISK_LOW,
        "preferred_closing_axis": _AXIS_SHORT,
        "notes": "Lata rectangular pequena; cierre por eje corto (50 mm).",
    },
    "gelatin_box": {
        "shape": "low_box",
        "dims": (0.073, 0.085, 0.028),
        "primary_strategy": "low_box_topdown",
        "fallback_strategies": [
            "edge_grasp",
            "top_down_short_axis_if_collision_free",
        ],
        "required_width": 0.073,
        "grasp_depth_from_top": 0.003,
        "risk": _RISK_MEDIUM,
        "preferred_closing_axis": _AXIS_SHORT,
        "close_squeeze_m": 0.001,
        "notes": "Caja baja (28 mm); oblicuo o edge para no chocar con mesa.",
    },
    "pudding_box": {
        "shape": "low_box_wide",
        "dims": (0.110, 0.089, 0.035),
        "primary_strategy": "edge_grasp",
        "fallback_strategies": [
            "oblique_edge_grasp",
            "top_down_short_axis_if_collision_free",
            "partial_grasp_best_effort",
        ],
        "required_width": 0.089,
        "grasp_depth_from_top": 0.016,
        "risk": _RISK_MEDIUM,
        "preferred_closing_axis": _AXIS_SHORT,
        "notes": "Caja baja: cierre efectivo por eje corto/altura (edge), no por 89 mm de ancho.",
    },
    "master_chef_can": {
        "shape": "cylinder_wide",
        "diameter": 0.102,
        "height": 0.139,
        "primary_strategy": "edge_grasp",
        "fallback_strategies": [
            "oblique_edge_grasp",
            "lateral_partial_grasp",
            "push_to_edge_then_grasp",
        ],
        "required_width": 0.102,
        "grasp_depth_from_top": 0.060,
        "risk": _RISK_HIGH,
        "preferred_closing_axis": _AXIS_EDGE,
        "notes": "Cilindro 102 mm: imposible diametro completo, edge grasp.",
    },
}


# Perfil extendido (parámetros de ejecución / candidatos). Se fusiona en ``get_grasp_policy``.
_PROFILE_OVERRIDES: Dict[str, Dict[str, Any]] = {
    "cracker_box": {
        "recommended_open_joint_m": 0.040,
        "recommended_close_joint_m": 0.027,
        "recommended_grasp_depth_from_top_m": 0.033,
        "insertion_depth_limit_m": 0.036,
        "palm_clearance_above_top_m": 0.005,
        "depth_candidates_from_top_m": [0.030, 0.033, 0.036],
        "pregrasp_clearance_above_top_m": 0.080,
        "max_cartesian_descend_m": 0.068,
        "descend_velocity_scaling": 0.028,
        "descend_acceleration_scaling": 0.028,
        "min_tcp_clearance_above_table_m": 0.012,
        "object_high_clearance_above_top_m": 0.100,
        "safe_pregrasp_clearance_above_top_m": 0.140,
        "safe_pregrasp_extra_above_pregrasp_m": 0.10,
        "center_offset_candidates_m": [0.0, 0.006, -0.006, 0.010, -0.010],
        "center_offset_axes": ["major_axis", "minor_axis"],
        "yaw_candidate_offsets_rad": [0.0, 3.1416, 1.5708, -1.5708],
        "approach_distance_min_m": 0.120,
        "use_target_collision_until_pregrasp": True,
        "remove_target_collision_before_descend": True,
        "contact_required": True,
        "min_contact_margin_m": 0.003,
        "supported_for_parallel_jaw_topdown": True,
        "max_attempts": 6,
        "requires_known_box_center": True,
        "min_yaw_confidence": 0.80,
        "max_pose_fit_error": 0.015,
        "allow_center_fallback": False,
        "min_gripper_total_margin_m": 0.008,
        "side_grasp_tall_object_enabled": False,
        "yaw_policy": "align_short_axis",
        "object_safe_above_clearance_m": 0.150,
        "preferred_cartesian_descend_m": 0.075,
        "min_cartesian_descend_m": 0.060,
        "max_cartesian_descend_m": 0.095,
        "contact_width_tolerance_m": 0.011,
        "min_closure_delta_m": 0.010,
        "post_lift_min_delta_z_m": 0.040,
        "release_z_m": 0.425,
        "supported_for_demo": True,
        "demo_priority": "robust_main",
    },
    "sugar_box": {
        "recommended_open_joint_m": 0.040,
        "recommended_close_joint_m": 0.017,
        "recommended_grasp_depth_from_top_m": 0.028,
        "insertion_depth_limit_m": 0.032,
        "palm_clearance_above_top_m": 0.005,
        "edge_offset_m": 0.012,
        "depth_candidates_from_top_m": [0.025, 0.028, 0.032],
        "pregrasp_clearance_above_top_m": 0.070,
        "max_cartesian_descend_m": 0.055,
        "descend_velocity_scaling": 0.022,
        "descend_acceleration_scaling": 0.022,
        "min_tcp_clearance_above_table_m": 0.012,
        "object_high_clearance_above_top_m": 0.120,
        "safe_pregrasp_clearance_above_top_m": 0.130,
        "center_offset_candidates_m": [0.0, 0.006, -0.006],
        "center_offset_axes": ["major_axis"],
        "yaw_candidate_offsets_rad": [0.0, 3.1416],
        "approach_distance_min_m": 0.120,
        "use_target_collision_until_pregrasp": True,
        "remove_target_collision_before_descend": True,
        "contact_required": True,
        "min_contact_margin_m": 0.004,
        "supported_for_parallel_jaw_topdown": True,
        "max_attempts": 4,
        "requires_known_box_center": True,
        "min_yaw_confidence": 0.80,
        "max_pose_fit_error": 0.015,
        "allow_center_fallback": False,
        "min_gripper_total_margin_m": 0.008,
        "side_grasp_tall_object_enabled": False,
        "post_lift_min_delta_z_m": 0.040,
        "release_z_m": 0.375,
        "supported_for_demo": True,
        "demo_priority": "robust_main",
    },
    "potted_meat_can": {
        "recommended_open_joint_m": 0.040,
        "recommended_close_joint_m": 0.023,
        "recommended_grasp_depth_from_top_m": 0.040,
        "depth_candidates_from_top_m": [0.035, 0.040, 0.045],
        "pregrasp_clearance_above_top_m": 0.080,
        "safe_pregrasp_clearance_above_top_m": 0.130,
        "center_offset_candidates_m": [0.0, 0.006, -0.006],
        "center_offset_axes": ["major_axis"],
        "yaw_candidate_offsets_rad": [0.0, 3.1416],
        "contact_required": True,
        "supported_for_parallel_jaw_topdown": True,
        "max_attempts": 4,
        "use_target_collision_until_pregrasp": True,
        "remove_target_collision_before_descend": True,
        "requires_known_box_center": True,
        "min_yaw_confidence": 0.80,
        "max_pose_fit_error": 0.015,
        "allow_center_fallback": False,
        "min_gripper_total_margin_m": 0.008,
        "side_grasp_tall_object_enabled": False,
    },
    "mustard_bottle": {
        "recommended_open_joint_m": 0.040,
        "recommended_close_joint_m": 0.027,
        "recommended_grasp_depth_from_top_m": 0.035,
        "depth_candidates_from_top_m": [0.030, 0.035, 0.040],
        "pregrasp_clearance_above_top_m": 0.100,
        "safe_pregrasp_clearance_above_top_m": 0.160,
        "center_offset_candidates_m": [0.0, 0.008, -0.008],
        "center_offset_axes": ["major_axis", "minor_axis"],
        "yaw_candidate_offsets_rad": [0.0, 3.1416],
        "contact_required": True,
        "use_target_collision_until_pregrasp": True,
        "remove_target_collision_before_descend": True,
        "supported_for_parallel_jaw_topdown": True,
        "max_attempts": 4,
        "requires_known_box_center": False,
        "min_yaw_confidence": 0.65,
        "max_pose_fit_error": 0.030,
        "allow_center_fallback": False,
        "min_gripper_total_margin_m": 0.008,
        "side_grasp_tall_object_enabled": False,
        "yaw_policy": "align_short_axis",
        "object_safe_above_clearance_m": 0.150,
        "preferred_cartesian_descend_m": 0.075,
        "min_cartesian_descend_m": 0.050,
        "max_cartesian_descend_m": 0.090,
        "descend_velocity_scaling": 0.025,
        "descend_acceleration_scaling": 0.025,
        "min_contact_margin_m": 0.003,
        "contact_width_tolerance_m": 0.014,
        "min_closure_delta_m": 0.008,
        "post_lift_min_delta_z_m": 0.040,
        "release_z_m": 0.480,
        "topdown_grasp_center_offset_long_m": 0.0,
        "topdown_grasp_center_offset_short_m": 0.0,
        "mustard_cap_center_offset_candidates": [
            [0.0, 0.0],
            [0.005, 0.0],
            [-0.005, 0.0],
            [0.0, 0.005],
            [0.0, -0.005],
            [0.008, 0.0],
            [-0.008, 0.0],
        ],
        "min_top_z_m": 0.42,
        "min_pregrasp_tcp_z_m": 0.49,
        "use_palm_bridge_z_constraint": True,
        "palm_bridge_clearance_above_top_m": 0.003,
        "palm_bridge_below_panda_hand_m": 0.063,
        "panda_hand_to_grasp_tcp_z_m": 0.100,
        "supported_for_demo": True,
        "demo_priority": "demo_high_object",
    },
    "bleach_cleanser": {
        "recommended_open_joint_m": 0.040,
        "recommended_close_joint_m": 0.031,
        "recommended_grasp_depth_from_top_m": 0.038,
        "depth_candidates_from_top_m": [0.035, 0.038, 0.042],
        "pregrasp_clearance_above_top_m": 0.120,
        "safe_pregrasp_clearance_above_top_m": 0.180,
        "center_offset_candidates_m": [0.0, 0.010, -0.010],
        "center_offset_axes": ["major_axis", "minor_axis"],
        "yaw_candidate_offsets_rad": [0.0, 3.1416],
        "contact_required": True,
        "use_target_collision_until_pregrasp": True,
        "remove_target_collision_before_descend": True,
        "supported_for_parallel_jaw_topdown": True,
        "max_attempts": 4,
        "requires_known_box_center": False,
        "min_yaw_confidence": 0.70,
        "max_pose_fit_error": 0.020,
        "allow_center_fallback": True,
        "min_gripper_total_margin_m": 0.008,
        "side_grasp_tall_object_enabled": False,
        "yaw_policy": "align_short_axis",
        "object_safe_above_clearance_m": 0.160,
        "preferred_cartesian_descend_m": 0.080,
        "min_cartesian_descend_m": 0.055,
        "max_cartesian_descend_m": 0.100,
        "descend_velocity_scaling": 0.020,
        "descend_acceleration_scaling": 0.020,
        "min_contact_margin_m": 0.0020,
        "contact_width_tolerance_m": 0.012,
        "min_closure_delta_m": 0.010,
        "post_lift_min_delta_z_m": 0.040,
        "release_z_m": 0.520,
        "use_palm_bridge_z_constraint": True,
        "palm_bridge_clearance_above_top_m": 0.003,
        "palm_bridge_below_panda_hand_m": 0.063,
        "panda_hand_to_grasp_tcp_z_m": 0.100,
        "topdown_grasp_center_offset_long_m": 0.0085,
        "topdown_grasp_center_offset_short_m": -0.0004,
        "topdown_grasp_center_offset_local_xy_m": [0.0085, -0.0004],
        "supported_for_demo": True,
        "demo_priority": "demo_high_object_secondary",
    },
    "chips_can": {
        "recommended_open_joint_m": 0.040,
        "recommended_close_joint_m": 0.0365,
        "recommended_grasp_depth_from_top_m": 0.035,
        "depth_candidates_from_top_m": [0.030, 0.035, 0.040],
        "pregrasp_clearance_above_top_m": 0.120,
        "safe_pregrasp_clearance_above_top_m": 0.180,
        "center_offset_candidates_m": [0.0, 0.004, -0.004],
        "center_offset_axes": ["major_axis", "minor_axis"],
        "yaw_candidate_offsets_rad": [0.0],
        "contact_required": True,
        "use_target_collision_until_pregrasp": True,
        "remove_target_collision_before_descend": True,
        "supported_for_parallel_jaw_topdown": True,
        "max_attempts": 4,
        "requires_known_box_center": False,
        "min_yaw_confidence": 0.0,
        "max_pose_fit_error": 1.0,
        "allow_center_fallback": False,
        "min_gripper_total_margin_m": 0.003,
        "side_grasp_tall_object_enabled": False,
        "yaw_policy": "yaw_free",
        "object_safe_above_clearance_m": 0.160,
        "preferred_cartesian_descend_m": 0.060,
        "min_cartesian_descend_m": 0.040,
        "max_cartesian_descend_m": 0.080,
        "descend_velocity_scaling": 0.020,
        "descend_acceleration_scaling": 0.020,
        "contact_width_tolerance_m": 0.006,
        "min_closure_delta_m": 0.006,
        "post_lift_min_delta_z_m": 0.040,
        "release_z_m": 0.520,
        "supported_for_demo": "experimental_tight",
        "demo_priority": "optional_cylinder_high",
    },
    "apple": {
        "recommended_open_joint_m": 0.040,
        "recommended_close_joint_m": 0.0365,
        "supported_for_parallel_jaw_topdown": "experimental",
        "depth_candidates_from_top_m": [0.025, 0.030, 0.035],
        "center_offset_candidates_m": [0.0],
        "center_offset_axes": ["major_axis"],
        "yaw_candidate_offsets_rad": [0.0, 1.5708],
        "max_attempts": 4,
    },
    "banana": {
        "supported_for_parallel_jaw_topdown": "experimental",
        "depth_candidates_from_top_m": [0.015, 0.020, 0.025],
        "center_offset_candidates_m": [0.0, 0.005, -0.005],
        "center_offset_axes": ["major_axis"],
        "yaw_candidate_offsets_rad": [0.0, 3.1416],
        "max_attempts": 4,
    },
    "gelatin_box": {
        "supported_for_parallel_jaw_topdown": True,
        "recommended_open_joint_m": 0.040,
        "recommended_close_joint_m": 0.014,
        "close_joint_candidates_m": [0.014, 0.013, 0.012],
        "use_above_top_grasp_z": True,
        "grasp_tcp_above_top_m": 0.002,
        "grasp_tcp_above_top_candidates_m": [0.002],
        "min_grasp_tcp_above_top_m": 0.002,
        "micro_descend_from_pregrasp_enabled": True,
        "micro_descend_before_close_m": 0.035,
        "max_micro_descend_before_close_m": 0.035,
        "disable_normal_cartesian_descend": True,
        "single_close_attempt": True,
        "enable_z_candidates_after_contact_fail": False,
        "recommended_grasp_depth_from_top_m": 0.003,
        "insertion_depth_limit_m": 0.005,
        "palm_clearance_above_top_m": 0.005,
        "edge_offset_m": 0.010,
        "pregrasp_clearance_above_top_m": 0.060,
        "preferred_cartesian_descend_m": 0.040,
        "min_cartesian_descend_m": 0.025,
        "max_cartesian_descend_m": 0.055,
        "descend_velocity_scaling": 0.015,
        "descend_acceleration_scaling": 0.015,
        "min_tcp_clearance_above_table_m": 0.018,
        "object_high_clearance_above_top_m": 0.120,
        "approach_distance_min_m": 0.100,
        "depth_candidates_from_top_m": [0.002, 0.003, 0.004, 0.005],
        "center_offset_candidates_m": [0.0, 0.004, -0.004],
        "center_offset_axes": ["major_axis", "minor_axis"],
        "yaw_candidate_offsets_rad": [0.0, 1.5708],
        "max_attempts": 4,
        "requires_known_box_center": True,
        "min_yaw_confidence": 0.75,
        "max_pose_fit_error": 0.020,
        "allow_center_fallback": False,
        "min_gripper_total_margin_m": 0.006,
        "side_grasp_tall_object_enabled": False,
        "yaw_policy": "align_short_axis_strict",
        "object_safe_above_clearance_m": 0.120,
        "max_expected_width_m": 0.080,
        "contact_width_tolerance_m": 0.006,
        "thin_edge_contact_width_tolerance_m": 0.007,
        "max_allowed_thin_edge_compression_m": 0.007,
        "min_closure_delta_m": 0.0030,
        "min_contact_margin_m": 0.0015,
        "post_lift_min_delta_z_m": 0.025,
        "release_z_m": 0.315,
        "supported_for_demo": "experimental",
        "demo_priority": "experimental_small_box",
        "visual_dims_estimated_m": [0.075, 0.090, 0.030],
    },
    "tuna_fish_can": {
        "supported_for_parallel_jaw_topdown": False,
        "notes": "Object not robustly graspable with current Franka Hand top-down policy",
        "max_attempts": 2,
    },
    "pudding_box": {
        "supported_for_parallel_jaw_topdown": True,
        "recommended_open_joint_m": 0.040,
        "recommended_close_joint_m": 0.016,
        "recommended_grasp_depth_from_top_m": 0.008,
        "insertion_depth_limit_m": 0.010,
        "palm_clearance_above_top_m": 0.005,
        "edge_offset_m": 0.015,
        "pregrasp_clearance_above_top_m": 0.055,
        "max_cartesian_descend_m": 0.038,
        "descend_velocity_scaling": 0.017,
        "descend_acceleration_scaling": 0.017,
        "min_tcp_clearance_above_table_m": 0.010,
        "object_high_clearance_above_top_m": 0.125,
        "approach_distance_min_m": 0.095,
        "depth_candidates_from_top_m": [0.006, 0.008, 0.010],
        "center_offset_candidates_m": [0.0, 0.004],
        "center_offset_axes": ["major_axis", "minor_axis"],
        "yaw_candidate_offsets_rad": [0.0, 1.5708],
        "max_attempts": 4,
        "requires_known_box_center": True,
        "min_yaw_confidence": 0.70,
        "max_pose_fit_error": 0.025,
        "allow_center_fallback": True,
        "min_gripper_total_margin_m": 0.004,
        "side_grasp_tall_object_enabled": False,
        "notes": "Demo: edge/top-down con anchura efectiva por eje corto o altura.",
        "yaw_policy": "align_short_axis_strict",
        "supported_for_demo": False,
        "demo_priority": "disabled_geometry_mismatch",
        "visual_dims_estimated_m": [0.111, 0.081, 0.090],
    },
    "master_chef_can": {
        "supported_for_parallel_jaw_topdown": False,
        "notes": "Object not robustly graspable with current Franka Hand top-down policy",
        "max_attempts": 2,
    },
}


def _default_profile_extension_fields(
    entry: Dict[str, Any],
    label: str,
    primary: str,
    fallbacks: List[str],
    risk_level: str,
    required_width: float,
    db_required_width: float,
    measured_required_width_out: Optional[float],
    object_height_out: float,
    db_height_f: float,
    effective_height: float,
    open_joint: float,
    close_joint: float,
    depth_from_top: float,
    dimension_source: str,
    notes: str,
    preferred_closing_axis: str,
    flags: Dict[str, bool],
) -> Dict[str, Any]:
    dft = float(depth_from_top)
    depth_cand = [round(dft * 0.92, 4), round(dft, 4), round(dft * 1.08, 4)]
    key = normalize_label(label)
    ov = dict(_PROFILE_OVERRIDES.get(key, {}))
    if "depth_candidates_from_top_m" not in ov:
        ov["depth_candidates_from_top_m"] = depth_cand
    if "center_offset_candidates_m" not in ov:
        ov["center_offset_candidates_m"] = [0.0, 0.006, -0.006]
    if "center_offset_axes" not in ov:
        ov["center_offset_axes"] = ["major_axis"]
    if "yaw_candidate_offsets_rad" not in ov:
        ov["yaw_candidate_offsets_rad"] = [0.0, 3.1416]
    if "pregrasp_clearance_above_top_m" not in ov:
        ov["pregrasp_clearance_above_top_m"] = 0.09 if risk_level == _RISK_LOW else 0.10
    if "safe_pregrasp_clearance_above_top_m" not in ov:
        ov["safe_pregrasp_clearance_above_top_m"] = float(ov["pregrasp_clearance_above_top_m"]) + 0.05
    if "safe_pregrasp_extra_above_pregrasp_m" not in ov:
        ov["safe_pregrasp_extra_above_pregrasp_m"] = 0.10
    if "approach_distance_min_m" not in ov:
        ov["approach_distance_min_m"] = 0.12
    if "use_target_collision_until_pregrasp" not in ov:
        ov["use_target_collision_until_pregrasp"] = True
    if "remove_target_collision_before_descend" not in ov:
        ov["remove_target_collision_before_descend"] = True
    if "contact_required" not in ov:
        ov["contact_required"] = True
    if "min_contact_margin_m" not in ov:
        ov["min_contact_margin_m"] = 0.004
    if "supported_for_parallel_jaw_topdown" not in ov:
        ov["supported_for_parallel_jaw_topdown"] = True
    if "max_attempts" not in ov:
        ov["max_attempts"] = 4
    if "requires_known_box_center" not in ov:
        ov["requires_known_box_center"] = False
    if "min_yaw_confidence" not in ov:
        ov["min_yaw_confidence"] = 0.0
    if "max_pose_fit_error" not in ov:
        ov["max_pose_fit_error"] = 1.0
    if "allow_center_fallback" not in ov:
        ov["allow_center_fallback"] = True
    if "min_gripper_total_margin_m" not in ov:
        ov["min_gripper_total_margin_m"] = 0.008
    if "side_grasp_tall_object_enabled" not in ov:
        ov["side_grasp_tall_object_enabled"] = False
    if "preferred_cartesian_descend_m" not in ov:
        ov["preferred_cartesian_descend_m"] = 0.070
    if "min_cartesian_descend_m" not in ov:
        ov["min_cartesian_descend_m"] = 0.040
    if "yaw_policy" not in ov:
        ov["yaw_policy"] = "align_short_axis"
    if "object_safe_above_clearance_m" not in ov:
        ov["object_safe_above_clearance_m"] = 0.120
    if "min_closure_delta_m" not in ov:
        ov["min_closure_delta_m"] = 0.010
    if "contact_width_tolerance_m" not in ov:
        ov["contact_width_tolerance_m"] = 0.012
    if "post_lift_min_delta_z_m" not in ov:
        ov["post_lift_min_delta_z_m"] = 0.040
    if "supported_for_demo" not in ov:
        ov["supported_for_demo"] = True
    if "demo_priority" not in ov:
        ov["demo_priority"] = "default"
    ro = float(ov.get("recommended_open_joint_m", open_joint))
    rc = float(ov.get("recommended_close_joint_m", close_joint))
    rd = float(ov.get("recommended_grasp_depth_from_top_m", dft))
    edge_offset = ov.get("edge_offset_m")
    palm_clearance = ov.get("palm_clearance_above_top_m")
    return {
        "label": label,
        "shape": str(entry.get("shape", "")),
        "primary_strategy": primary,
        "fallback_strategies": list(fallbacks),
        "risk_level": risk_level,
        "dimension_source": dimension_source,
        "required_grasp_width_m": float(required_width),
        "db_required_width_m": float(db_required_width),
        "measured_required_width_m": (
            float(measured_required_width_out)
            if measured_required_width_out is not None
            else None
        ),
        "object_height_m": float(object_height_out),
        "db_height_m": float(db_height_f),
        "effective_height_m": float(effective_height),
        "recommended_open_joint_m": ro,
        "recommended_close_joint_m": rc,
        "recommended_grasp_depth_from_top_m": rd,
        "edge_offset_m": float(edge_offset) if edge_offset is not None else None,
        "palm_clearance_above_top_m": (
            float(palm_clearance) if palm_clearance is not None else None
        ),
        "depth_candidates_from_top_m": list(ov["depth_candidates_from_top_m"]),
        "pregrasp_clearance_above_top_m": float(ov["pregrasp_clearance_above_top_m"]),
        "safe_pregrasp_clearance_above_top_m": float(ov["safe_pregrasp_clearance_above_top_m"]),
        "safe_pregrasp_extra_above_pregrasp_m": float(ov["safe_pregrasp_extra_above_pregrasp_m"]),
        "center_offset_candidates_m": [float(x) for x in ov["center_offset_candidates_m"]],
        "center_offset_axes": [str(x) for x in ov["center_offset_axes"]],
        "yaw_candidate_offsets_rad": [float(x) for x in ov["yaw_candidate_offsets_rad"]],
        "approach_distance_min_m": float(ov["approach_distance_min_m"]),
        "use_target_collision_until_pregrasp": bool(ov["use_target_collision_until_pregrasp"]),
        "remove_target_collision_before_descend": bool(ov["remove_target_collision_before_descend"]),
        "contact_required": bool(ov["contact_required"]),
        "min_contact_margin_m": float(ov["min_contact_margin_m"]),
        "supported_for_parallel_jaw_topdown": ov["supported_for_parallel_jaw_topdown"],
        "max_attempts": int(ov["max_attempts"]),
        "requires_known_box_center": bool(ov["requires_known_box_center"]),
        "min_yaw_confidence": float(ov["min_yaw_confidence"]),
        "max_pose_fit_error": float(ov["max_pose_fit_error"]),
        "allow_center_fallback": bool(ov["allow_center_fallback"]),
        "min_gripper_total_margin_m": float(ov["min_gripper_total_margin_m"]),
        "side_grasp_tall_object_enabled": bool(ov["side_grasp_tall_object_enabled"]),
        "preferred_closing_axis": preferred_closing_axis,
        "prefer_oblique": flags["prefer_oblique"],
        "prefer_lateral": flags["prefer_lateral"],
        "prefer_edge": flags["prefer_edge"],
        "prefer_push_to_edge": flags["prefer_push_to_edge"],
        "notes": str(ov.get("notes", notes)),
        "max_cartesian_descend_m": ov.get("max_cartesian_descend_m"),
        "descend_velocity_scaling": ov.get("descend_velocity_scaling"),
        "descend_acceleration_scaling": ov.get("descend_acceleration_scaling"),
        "insertion_depth_limit_m": ov.get("insertion_depth_limit_m"),
        "min_tcp_clearance_above_table_m": ov.get("min_tcp_clearance_above_table_m"),
        "object_high_clearance_above_top_m": ov.get("object_high_clearance_above_top_m"),
        "preferred_cartesian_descend_m": ov.get("preferred_cartesian_descend_m"),
        "min_cartesian_descend_m": ov.get("min_cartesian_descend_m"),
        "yaw_policy": ov.get("yaw_policy"),
        "object_safe_above_clearance_m": ov.get("object_safe_above_clearance_m"),
        "max_expected_width_m": ov.get("max_expected_width_m"),
        "min_closure_delta_m": ov.get("min_closure_delta_m"),
        "contact_width_tolerance_m": ov.get("contact_width_tolerance_m"),
        "post_lift_min_delta_z_m": ov.get("post_lift_min_delta_z_m"),
        "release_z_m": ov.get("release_z_m"),
        "topdown_grasp_center_offset_long_m": ov.get("topdown_grasp_center_offset_long_m"),
        "topdown_grasp_center_offset_short_m": ov.get("topdown_grasp_center_offset_short_m"),
        "topdown_grasp_center_offset_local_xy_m": ov.get(
            "topdown_grasp_center_offset_local_xy_m"
        ),
        "min_top_z_m": ov.get("min_top_z_m"),
        "min_pregrasp_tcp_z_m": ov.get("min_pregrasp_tcp_z_m"),
        "use_palm_bridge_z_constraint": ov.get("use_palm_bridge_z_constraint"),
        "palm_bridge_clearance_above_top_m": ov.get("palm_bridge_clearance_above_top_m"),
        "palm_bridge_below_panda_hand_m": ov.get("palm_bridge_below_panda_hand_m"),
        "panda_hand_to_grasp_tcp_z_m": ov.get("panda_hand_to_grasp_tcp_z_m"),
        "supported_for_demo": ov.get("supported_for_demo"),
        "demo_priority": ov.get("demo_priority"),
        "visual_dims_estimated_m": ov.get("visual_dims_estimated_m"),
    }


_EDGE_STRATEGIES = {
    "edge_grasp",
    "oblique_edge_grasp",
    "lateral_edge_grasp",
    "push_to_edge_then_grasp",
    "partial_grasp_best_effort",
    "lateral_partial_grasp",
}

_OBLIQUE_STRATEGIES = {
    "oblique_short_axis",
    "oblique_body_grasp",
    "oblique_cylinder_grasp",
    "oblique_center_grasp",
    "oblique_perpendicular_to_long_axis",
    "oblique_edge_grasp",
}

LATERAL_GRASP_STRATEGIES = {
    "lateral_short_axis",
    "lateral_body_grasp",
    "lateral_cylinder_grasp",
    "lateral_center_grasp",
    "lateral_edge_grasp",
    "lateral_partial_grasp",
}

# Estrategia lateral futura (no activar en el flujo principal hasta implementar poses laterales).
SIDE_GRASP_TALL_STRATEGIES: FrozenSet[str] = frozenset({"side_grasp_tall_object"})

_PUSH_STRATEGIES = {
    "push_to_edge_then_grasp",
}

TALL_OBJECT_CAP_CENTER_SOURCE = "runtime_gt_tall_object_cap_center"
TALL_OBJECT_CAP_CENTER_SDF_OFFSET_SOURCE = (
    "runtime_gt_tall_object_cap_center_sdf_offset"
)
TALL_OBJECT_CAP_CENTER_CALIBRATED_SOURCE = (
    "runtime_gt_tall_object_cap_center_calibrated"
)
_LOG = logging.getLogger("panda_vision.object_grasp_policy")


def resolve_tall_object_top_z_m(
    label: str,
    geometry_center_z: float,
    *,
    height_m: Optional[float] = None,
    payload_top_z_before: Optional[float] = None,
) -> Tuple[float, Dict[str, Any]]:
    """Cara superior Z = centro geométrico + altura/2 (no confundir con centro del cuerpo)."""
    policy = get_grasp_policy(label)
    lb = normalize_label(label)
    h = height_m
    if h is None:
        dims = policy.get("dims")
        if isinstance(dims, (list, tuple)) and len(dims) >= 3:
            h = float(dims[2])
        else:
            h = policy.get("db_height_m") or policy.get("object_height_m")
    try:
        h_f = float(h) if h is not None else None
    except (TypeError, ValueError):
        h_f = None
    gz = float(geometry_center_z)
    before = _to_optional_float(payload_top_z_before)
    dbg: Dict[str, Any] = {
        "label": lb,
        "geometry_center_z": gz,
        "height_m": h_f,
        "payload_top_z_before": before,
        "source": "known_geometry_height",
    }
    if h_f is None or h_f <= 1e-6:
        dbg["computed_top_z"] = before if before is not None else gz
        dbg["payload_top_z_after"] = dbg["computed_top_z"]
        dbg["source"] = "fallback_no_height"
        return float(dbg["computed_top_z"]), dbg

    computed = gz + 0.5 * h_f
    after = computed
    if before is not None:
        # Payload con Z de centro geométrico (típico tras cap_center mal etiquetado).
        if abs(before - gz) < 0.025 or before < computed - 0.02:
            after = computed
            dbg["source"] = "known_geometry_height"
        else:
            after = float(before)
            dbg["source"] = "payload_top_z_kept"
    dbg["computed_top_z"] = computed
    dbg["payload_top_z_after"] = after
    return float(after), dbg


def _to_optional_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_float_with_default(
    value: Any, *, default: float, label: str, field: str
) -> float:
    if value is None:
        _LOG.info(
            "[TALL_OBJECT_OFFSET_DEFAULT] label=%s field=%s old_value=None new_value=%.1f reason=none_policy_offset",
            label,
            field,
            float(default),
        )
        return float(default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def tall_object_topdown_cap_offset_configured(policy: Dict[str, Any]) -> bool:
    if str(policy.get("primary_strategy", "")) != "tall_object_topdown":
        return False
    if policy.get("topdown_grasp_center_offset_local_xy_m") is not None:
        return True
    return (
        "topdown_grasp_center_offset_long_m" in policy
        or "topdown_grasp_center_offset_short_m" in policy
    )


def apply_tall_object_topdown_grasp_center_offset(
    label: str,
    body_xy: Tuple[float, float],
    yaw_rad: float,
    top_z_m: float,
    *,
    default_center_source: str = "runtime_gt_tall_object_center",
) -> Tuple[List[float], str, Dict[str, Any]]:
    """Desplaza el centro operativo top-down (p. ej. tapón) respecto al centro GT del cuerpo."""
    policy = get_grasp_policy(label)
    lb = normalize_label(label)
    strategy = str(policy.get("primary_strategy", ""))
    base_dbg: Dict[str, Any] = {
        "applied": False,
        "label": lb,
        "strategy": strategy,
    }
    if not tall_object_topdown_cap_offset_configured(policy):
        return (
            [float(body_xy[0]), float(body_xy[1]), float(top_z_m)],
            str(default_center_source),
            base_dbg,
        )

    offset_long = 0.0
    offset_short = 0.0
    local = policy.get("topdown_grasp_center_offset_local_xy_m")
    if local is None:
        offset_long = 0.0
        offset_short = 0.0
    elif isinstance(local, (list, tuple)) and len(local) >= 2:
        offset_long = _safe_float_with_default(
            local[0],
            default=0.0,
            label=lb,
            field="topdown_grasp_center_offset_local_xy_m[0]",
        )
        offset_short = _safe_float_with_default(
            local[1],
            default=0.0,
            label=lb,
            field="topdown_grasp_center_offset_local_xy_m[1]",
        )
    else:
        offset_long = _safe_float_with_default(
            policy.get("topdown_grasp_center_offset_long_m"),
            default=0.0,
            label=lb,
            field="topdown_grasp_center_offset_long_m",
        )
        offset_short = _safe_float_with_default(
            policy.get("topdown_grasp_center_offset_short_m"),
            default=0.0,
            label=lb,
            field="topdown_grasp_center_offset_short_m",
        )

    yaw = float(yaw_rad)
    c = math.cos(yaw)
    s = math.sin(yaw)
    dx = offset_long * c - offset_short * s
    dy = offset_long * s + offset_short * c
    cap_x = float(body_xy[0]) + dx
    cap_y = float(body_xy[1]) + dy
    dbg = {
        "applied": True,
        "label": lb,
        "strategy": "tall_object_topdown",
        "body_center_xy": (float(body_xy[0]), float(body_xy[1])),
        "topdown_center_xy": (cap_x, cap_y),
        "offset_local_xy": (offset_long, offset_short),
        "offset_base_xy": (dx, dy),
        "yaw_rad": yaw,
        "source": TALL_OBJECT_CAP_CENTER_SOURCE,
    }
    return (
        [cap_x, cap_y, float(top_z_m)],
        TALL_OBJECT_CAP_CENTER_SOURCE,
        dbg,
    )


def normalize_label(label: Optional[str]) -> str:
    """Normalize an incoming YOLO label to the canonical key in OBJECT_DB.

    Accepts case variations like ``Mustard_bottle`` / ``mustard_bottle``.
    Also strips whitespace and replaces spaces with underscores.
    """
    if not label:
        return ""
    norm = str(label).strip().lower().replace(" ", "_").replace("-", "_")
    return norm


def finger_joint_from_total_width(total_width_m: float) -> float:
    """Convert a desired total gripper aperture to per-finger joint value.

    The Franka Hand exposes two finger joints, each in ``[0.0, 0.04]`` m,
    and the total opening between fingers equals ``2 * finger_joint``.
    Output is clamped to the valid joint range.
    """
    try:
        total = float(total_width_m)
    except (TypeError, ValueError):
        total = 0.0
    if total < 0.0:
        total = 0.0
    joint = total / 2.0
    if joint < MIN_FINGER_JOINT_M:
        joint = MIN_FINGER_JOINT_M
    if joint > MAX_FINGER_JOINT_M:
        joint = MAX_FINGER_JOINT_M
    return float(joint)


def _default_close_squeeze_for_risk(risk_level: str) -> float:
    """Per-risk-level default for the close 'squeeze' margin (m)."""
    if risk_level == _RISK_HIGH:
        return 0.001
    if risk_level == _RISK_MEDIUM:
        return 0.002
    return 0.004


def compute_open_close_joints(
    required_width_m: float,
    risk_level: str,
    close_squeeze_m: Optional[float] = None,
) -> Tuple[float, float]:
    """Compute recommended per-finger open / close joint values.

    ``required_width_m`` is the total aperture (between fingers) needed to
    enclose the object. The returned values are PER FINGER joint positions
    (``total = 2 * joint``).

    ``close_squeeze_m`` is the margin subtracted from ``required_width_m`` to
    obtain the total closing aperture::

        close_total = required_width_m - close_squeeze_m

    When ``None``, a per-risk default is used (``0.004`` low, ``0.002`` medium,
    ``0.001`` high). Smaller values close less aggressively (less risk of
    pushing the object); larger values bite tighter.

    Open joint is always commanded to ``MAX_FINGER_JOINT_M = 0.040`` (gripper
    fully open within mechanical limits).
    """
    try:
        req = float(required_width_m)
    except (TypeError, ValueError):
        req = 0.0
    if req < 0.0:
        req = 0.0

    if close_squeeze_m is None:
        squeeze = _default_close_squeeze_for_risk(risk_level)
    else:
        try:
            squeeze = float(close_squeeze_m)
        except (TypeError, ValueError):
            squeeze = _default_close_squeeze_for_risk(risk_level)
        if squeeze < 0.0:
            squeeze = 0.0

    open_total = 2.0 * MAX_FINGER_JOINT_M
    if risk_level == _RISK_HIGH:
        close_total = min(req, MAX_GRIPPER_WIDTH_M)
        close_total = max(close_total - squeeze, 0.020)
    else:
        close_total = max(req - squeeze, 0.010)

    if close_total > open_total:
        close_total = open_total

    open_joint = finger_joint_from_total_width(open_total)
    close_joint = finger_joint_from_total_width(close_total)
    return open_joint, close_joint


def _entry_dimensions(entry: Dict[str, Any]) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """Return (major, minor, height) in metres from a DB entry, if defined."""
    if "dims" in entry:
        dims = entry["dims"]
        if len(dims) == 3:
            sorted_xy = sorted([float(dims[0]), float(dims[1])])
            minor = sorted_xy[0]
            major = sorted_xy[1]
            height = float(dims[2])
            return major, minor, height
    if "diameter" in entry and "height" in entry:
        d = float(entry["diameter"])
        h = float(entry["height"])
        return d, d, h
    return None, None, None


def _use_measured_if_plausible(
    measured: Optional[float],
    db_value: Optional[float],
    tolerance_ratio: float = 0.25,
) -> Optional[float]:
    """Return ``measured`` only if it is within ``tolerance_ratio`` of ``db_value``.

    The DB value remains the trusted source; the measured dimension is only
    accepted when it stays close to the known YCB ground-truth.
    """
    if measured is None:
        return db_value
    try:
        meas_f = float(measured)
    except (TypeError, ValueError):
        return db_value
    if meas_f <= 0.0:
        return db_value
    if db_value is None:
        return meas_f
    try:
        db_f = float(db_value)
    except (TypeError, ValueError):
        return db_value
    if db_f <= 0.0:
        return meas_f
    if meas_f < db_f * (1.0 - tolerance_ratio):
        return db_f
    if meas_f > db_f * (1.0 + tolerance_ratio):
        return db_f
    return meas_f


def _required_width_for_box(
    entry_required: float,
    measured_major: Optional[float],
    measured_minor: Optional[float],
) -> float:
    if measured_minor is not None and measured_minor > 0.0:
        return float(measured_minor)
    if measured_major is not None and measured_minor is None and measured_major > 0.0:
        return float(measured_major)
    return float(entry_required)


def _required_width_for_cylinder(
    entry_required: float,
    measured_major: Optional[float],
    measured_minor: Optional[float],
) -> float:
    candidates = [entry_required]
    if measured_minor is not None and measured_minor > 0.0:
        candidates.append(float(measured_minor))
    if measured_major is not None and measured_major > 0.0:
        candidates.append(float(measured_major))
    sensible = [c for c in candidates if 0.020 <= c <= 0.150]
    if sensible:
        return float(max(sensible))
    return float(entry_required)


def _required_width_for_banana(
    entry_required: float,
    measured_major: Optional[float],
    measured_minor: Optional[float],
) -> float:
    if measured_minor is not None and 0.020 <= measured_minor <= 0.090:
        return float(measured_minor)
    return float(entry_required)


def _required_width_for_apple(
    entry_required: float,
    measured_major: Optional[float],
    measured_minor: Optional[float],
) -> float:
    candidates = [entry_required]
    if measured_minor is not None and measured_minor > 0.0:
        candidates.append(float(measured_minor))
    if measured_major is not None and measured_major > 0.0:
        candidates.append(float(measured_major))
    sensible = [c for c in candidates if 0.030 <= c <= 0.120]
    if sensible:
        return float(min(sensible))
    return float(entry_required)


def _adjusted_required_width(
    entry: Dict[str, Any],
    measured_major: Optional[float],
    measured_minor: Optional[float],
) -> float:
    shape = entry.get("shape", "")
    entry_required = float(entry.get("required_width", SAFE_MAX_WIDTH_M))
    if shape in ("box", "low_box", "low_box_wide", "bottle"):
        return _required_width_for_box(entry_required, measured_major, measured_minor)
    if shape in ("cylinder", "low_cylinder", "cylinder_wide"):
        return _required_width_for_cylinder(entry_required, measured_major, measured_minor)
    if shape == "curved_long":
        return _required_width_for_banana(entry_required, measured_major, measured_minor)
    if shape == "sphere_like":
        return _required_width_for_apple(entry_required, measured_major, measured_minor)
    return entry_required


def resolve_effective_required_grasp_width(
    entry: Dict[str, Any],
    primary: str,
    fallbacks: List[str],
    db_required_width: float,
) -> Tuple[float, str]:
    """Anchura efectiva para comprobación de margen de pinza (no siempre el eje largo)."""
    shape = str(entry.get("shape", ""))
    _, db_minor, db_height = _entry_dimensions(entry)
    width = float(db_required_width)
    flags = _strategy_flags(primary, fallbacks)

    if shape in ("low_box", "low_box_wide"):
        dims = entry.get("dims")
        minor_db = None
        height_db = None
        if isinstance(dims, (list, tuple)) and len(dims) >= 3:
            sorted_xy = sorted([float(dims[0]), float(dims[1])])
            minor_db = sorted_xy[0]
            height_db = float(dims[2])
        candidates: List[float] = [width]
        if db_minor is not None:
            candidates.append(float(db_minor))
        if minor_db is not None:
            candidates.append(float(minor_db))
        if db_height is not None:
            candidates.append(float(db_height))
        if height_db is not None:
            candidates.append(float(height_db))

        if primary in _EDGE_STRATEGIES or flags["prefer_edge"]:
            edge_eff = min(c for c in candidates if c > 0.0)
            if edge_eff <= SAFE_MAX_WIDTH_M:
                return edge_eff, "edge_short_axis_or_height"

        if db_minor is not None and float(db_minor) <= SAFE_MAX_WIDTH_M:
            return float(db_minor), "db_minor_axis"
        if minor_db is not None and float(minor_db) <= SAFE_MAX_WIDTH_M:
            return float(minor_db), "dims_minor_axis"

    return width, "db_required_width"


def _strategy_flags(primary: str, fallbacks: List[str]) -> Dict[str, bool]:
    all_strategies = [primary] + list(fallbacks)
    return {
        "prefer_oblique": any(s in _OBLIQUE_STRATEGIES for s in all_strategies)
        or primary in _OBLIQUE_STRATEGIES,
        "prefer_lateral": any(s in LATERAL_GRASP_STRATEGIES for s in all_strategies)
        or primary in LATERAL_GRASP_STRATEGIES,
        "prefer_edge": any(s in _EDGE_STRATEGIES for s in all_strategies)
        or primary in _EDGE_STRATEGIES,
        "prefer_push_to_edge": any(s in _PUSH_STRATEGIES for s in all_strategies)
        or primary in _PUSH_STRATEGIES,
    }


def _default_policy_for_unknown(
    label: str,
    measured_footprint_major_m: Optional[float],
    measured_footprint_minor_m: Optional[float],
    measured_height_m: Optional[float],
) -> Dict[str, Any]:
    """Return a conservative policy for labels not in the curated DB."""
    primary = "top_down_short_axis"
    fallbacks = [
        "oblique_short_axis",
        "edge_grasp",
        "partial_grasp_best_effort",
    ]
    db_required_width = SAFE_MAX_WIDTH_M
    measured_required_width: Optional[float] = None
    if measured_footprint_minor_m is not None and measured_footprint_minor_m > 0.0:
        measured_required_width = float(measured_footprint_minor_m)
    elif measured_footprint_major_m is not None and measured_footprint_major_m > 0.0:
        measured_required_width = float(measured_footprint_major_m)
    required_width = db_required_width
    db_height = 0.080
    object_height = db_height
    effective_height = db_height
    if measured_height_m is not None and measured_height_m > 0.0:
        object_height = float(measured_height_m)
    open_joint, close_joint = compute_open_close_joints(required_width, _RISK_MEDIUM)
    flags = _strategy_flags(primary, fallbacks)
    notes_unknown = "Etiqueta no esta en la base de datos; politica conservadora por defecto."
    depth_from_top = 0.040
    entry_unknown: Dict[str, Any] = {"shape": "unknown", "notes": notes_unknown}
    base: Dict[str, Any] = {
        "label": label,
        "shape": "unknown",
        "primary_strategy": primary,
        "fallback_strategies": fallbacks,
        "parallel_jaw_allowed": True,
        "risk_level": _RISK_MEDIUM,
        "required_grasp_width_m": float(required_width),
        "db_required_width_m": float(db_required_width),
        "measured_required_width_m": (
            float(measured_required_width)
            if measured_required_width is not None
            else None
        ),
        "object_height_m": float(object_height),
        "db_height_m": float(db_height),
        "effective_height_m": float(effective_height),
        "recommended_open_joint_m": float(open_joint),
        "recommended_close_joint_m": float(close_joint),
        "close_squeeze_m": float(_default_close_squeeze_for_risk(_RISK_MEDIUM)),
        "recommended_grasp_depth_from_top_m": depth_from_top,
        "preferred_closing_axis": _AXIS_SHORT,
        "prefer_oblique": flags["prefer_oblique"],
        "prefer_lateral": flags["prefer_lateral"],
        "prefer_edge": flags["prefer_edge"],
        "prefer_push_to_edge": flags["prefer_push_to_edge"],
        "dimension_source": "db",
        "notes": notes_unknown,
    }
    ext = _default_profile_extension_fields(
        entry_unknown,
        label,
        primary,
        fallbacks,
        _RISK_MEDIUM,
        float(required_width),
        float(db_required_width),
        measured_required_width,
        float(object_height),
        float(db_height),
        float(effective_height),
        open_joint,
        close_joint,
        depth_from_top,
        "db",
        notes_unknown,
        _AXIS_SHORT,
        flags,
    )
    base.update(ext)
    return base


def get_grasp_policy(
    label: str,
    measured_footprint_major_m: Optional[float] = None,
    measured_footprint_minor_m: Optional[float] = None,
    measured_height_m: Optional[float] = None,
    use_measured_dimensions: bool = False,
) -> Dict[str, Any]:
    """Return the per-object grasp policy for ``label``.

    By default (``use_measured_dimensions=False``) the YCB database is the
    authoritative source for grasp width, height and depth. The measured
    dimensions are kept solely as diagnostic fields
    (``measured_required_width_m``, ``object_height_m``).

    If ``use_measured_dimensions=True``, the measured values are accepted
    only when they stay within ``+/- 25%%`` of the DB value (see
    ``_use_measured_if_plausible``); otherwise the DB value is restored.
    """
    key = normalize_label(label)
    entry = OBJECT_DB.get(key)
    if entry is None:
        return _default_policy_for_unknown(
            label,
            measured_footprint_major_m,
            measured_footprint_minor_m,
            measured_height_m,
        )

    db_major, db_minor, db_height = _entry_dimensions(entry)
    db_required_width = float(entry.get("required_width", db_minor if db_minor is not None else SAFE_MAX_WIDTH_M))

    measured_required_width = _adjusted_required_width(
        entry, measured_footprint_major_m, measured_footprint_minor_m
    )
    if (
        measured_footprint_major_m is None
        and measured_footprint_minor_m is None
    ):
        measured_required_width_out: Optional[float] = None
    else:
        measured_required_width_out = float(measured_required_width)

    if use_measured_dimensions:
        candidate_width = _use_measured_if_plausible(
            measured_required_width_out, db_required_width, tolerance_ratio=0.25
        )
        required_width = float(
            candidate_width if candidate_width is not None else db_required_width
        )
        width_used_measured = (
            measured_required_width_out is not None
            and candidate_width is not None
            and abs(candidate_width - db_required_width) > 1e-9
        )
    else:
        required_width = float(db_required_width)
        width_used_measured = False

    primary = str(entry.get("primary_strategy", "top_down_short_axis"))
    fallbacks = list(entry.get("fallback_strategies", []))

    required_width, required_width_source = resolve_effective_required_grasp_width(
        entry, primary, fallbacks, required_width
    )

    db_height_f = float(db_height if db_height is not None else 0.080)
    if use_measured_dimensions:
        eff_height_candidate = _use_measured_if_plausible(
            measured_height_m, db_height_f, tolerance_ratio=0.25
        )
        effective_height = float(
            eff_height_candidate if eff_height_candidate is not None else db_height_f
        )
        height_used_measured = (
            measured_height_m is not None
            and eff_height_candidate is not None
            and abs(eff_height_candidate - db_height_f) > 1e-9
        )
    else:
        effective_height = db_height_f
        height_used_measured = False

    object_height_out = float(
        measured_height_m
        if measured_height_m is not None and measured_height_m > 0.0
        else db_height_f
    )

    if width_used_measured and height_used_measured:
        dimension_source = "measured"
    elif width_used_measured or height_used_measured:
        dimension_source = "mixed"
    else:
        dimension_source = "db"

    risk_level = str(entry.get("risk", _RISK_MEDIUM))

    escalation_threshold = MAX_GRIPPER_WIDTH_M - WIDTH_MARGIN_M
    if required_width > escalation_threshold and risk_level != _RISK_HIGH:
        risk_level = _RISK_HIGH
        if primary not in _EDGE_STRATEGIES and primary not in _PUSH_STRATEGIES:
            fallbacks = [primary] + [f for f in fallbacks if f != "edge_grasp"]
            primary = "edge_grasp"
        for extra in ("oblique_edge_grasp", "partial_grasp_best_effort"):
            if extra not in fallbacks:
                fallbacks.append(extra)
        required_width, required_width_source = resolve_effective_required_grasp_width(
            entry, primary, fallbacks, required_width
        )

    if "close_squeeze_m" in entry:
        try:
            squeeze_override = float(entry["close_squeeze_m"])
        except (TypeError, ValueError):
            squeeze_override = None
    else:
        squeeze_override = None
    open_joint, close_joint = compute_open_close_joints(
        required_width, risk_level, close_squeeze_m=squeeze_override
    )
    effective_close_squeeze = (
        squeeze_override
        if squeeze_override is not None
        else _default_close_squeeze_for_risk(risk_level)
    )

    flags = _strategy_flags(primary, fallbacks)

    notes = str(entry.get("notes", ""))
    shape = str(entry.get("shape", ""))
    if shape in ("cylinder", "low_cylinder", "cylinder_wide", "sphere_like"):
        extra_note = " Yaw es indiferente para esta forma; se calcula igualmente."
        if extra_note.strip() not in notes:
            notes = (notes + extra_note).strip()

    depth_from_top = float(entry.get("grasp_depth_from_top", 0.040))
    base: Dict[str, Any] = {
        "label": label,
        "shape": shape,
        "primary_strategy": primary,
        "fallback_strategies": fallbacks,
        "parallel_jaw_allowed": True,
        "risk_level": risk_level,
        "required_grasp_width_m": float(required_width),
        "required_width_source": required_width_source,
        "db_required_width_m": float(db_required_width),
        "footprint_major_m": float(db_major) if db_major is not None else None,
        "footprint_minor_m": float(db_minor) if db_minor is not None else None,
        "measured_required_width_m": (
            float(measured_required_width_out)
            if measured_required_width_out is not None
            else None
        ),
        "object_height_m": float(object_height_out),
        "db_height_m": float(db_height_f),
        "effective_height_m": float(effective_height),
        "recommended_open_joint_m": float(open_joint),
        "recommended_close_joint_m": float(close_joint),
        "close_squeeze_m": float(effective_close_squeeze),
        "recommended_grasp_depth_from_top_m": depth_from_top,
        "preferred_closing_axis": str(entry.get("preferred_closing_axis", _AXIS_SHORT)),
        "prefer_oblique": flags["prefer_oblique"],
        "prefer_lateral": flags["prefer_lateral"],
        "prefer_edge": flags["prefer_edge"],
        "prefer_push_to_edge": flags["prefer_push_to_edge"],
        "dimension_source": dimension_source,
        "notes": notes,
    }
    ext = _default_profile_extension_fields(
        entry,
        label,
        primary,
        fallbacks,
        risk_level,
        required_width,
        db_required_width,
        measured_required_width_out,
        object_height_out,
        db_height_f,
        effective_height,
        open_joint,
        close_joint,
        depth_from_top,
        dimension_source,
        notes,
        str(entry.get("preferred_closing_axis", _AXIS_SHORT)),
        flags,
    )
    base.update(ext)
    return base


def get_collision_dimensions(label: str, padding_m: float = 0.0) -> Optional[Dict[str, Any]]:
    """Return collision-object dimensions derived purely from the YCB DB.

    The returned dict has the shape::

        {
            "shape": "box" | "cylinder",
            "box": (sx, sy, sz)        # if shape == "box"
            "cylinder": (radius, height)  # if shape == "cylinder"
            "box_fallback": (sx, sy, sz)  # always present
            "db_dims": (major, minor, height)
        }

    ``padding_m`` is added to every dimension. Returns ``None`` for unknown
    labels.
    """
    key = normalize_label(label)
    entry = OBJECT_DB.get(key)
    if entry is None:
        return None
    major, minor, height = _entry_dimensions(entry)
    if major is None or minor is None or height is None:
        return None
    shape = str(entry.get("shape", ""))
    pad = max(float(padding_m), 0.0)
    is_cylinder = shape in ("cylinder", "low_cylinder", "cylinder_wide")
    is_sphere = shape == "sphere_like"
    result: Dict[str, Any] = {
        "db_dims": (float(major), float(minor), float(height)),
        "box_fallback": (
            float(major) + pad,
            float(minor) + pad,
            float(height) + pad,
        ),
    }
    if is_cylinder or is_sphere:
        radius = float(entry.get("diameter", major)) / 2.0
        result["shape"] = "cylinder" if is_cylinder else "sphere"
        result["cylinder"] = (float(radius) + pad / 2.0, float(height) + pad)
    else:
        result["shape"] = "box"
        result["box"] = (
            float(major) + pad,
            float(minor) + pad,
            float(height) + pad,
        )
    return result


def export_grasp_policy_for_executor(
    label: str,
    *,
    measured_footprint_major_m: Optional[float] = None,
    measured_footprint_minor_m: Optional[float] = None,
    measured_height_m: Optional[float] = None,
    use_measured_dimensions: bool = False,
    edge_offset_m: Optional[float] = None,
) -> Dict[str, Any]:
    """Campos canónicos para /vision_to_executor y perception_to_pregrasp_test."""
    policy = get_grasp_policy(
        label,
        measured_footprint_major_m=measured_footprint_major_m,
        measured_footprint_minor_m=measured_footprint_minor_m,
        measured_height_m=measured_height_m,
        use_measured_dimensions=use_measured_dimensions,
    )
    lb = normalize_label(label)
    dims_lwh: Optional[List[float]] = None
    footprint_major: Optional[float] = policy.get("footprint_major_m")
    footprint_minor: Optional[float] = policy.get("footprint_minor_m")
    try:
        from panda_vision.spawn.runtime_scene_gt_geometry import get_known_box_gt_spec

        spec = get_known_box_gt_spec(lb)
        if spec is not None:
            l, w, h = spec.dims_lwh_m
            dims_lwh = [float(l), float(w), float(h)]
            footprint_major = max(float(l), float(w))
            footprint_minor = min(float(l), float(w))
    except Exception:
        pass

    primary = str(policy.get("primary_strategy", ""))
    fallbacks = list(policy.get("fallback_strategies") or [])
    flags = _strategy_flags(primary, fallbacks)
    edge_grasp_requested = bool(
        primary in _EDGE_STRATEGIES or flags.get("prefer_edge", False)
    )
    eff_w = float(policy["required_grasp_width_m"])
    db_w = float(policy.get("db_required_width_m") or eff_w)

    out: Dict[str, Any] = {
        "grasp_strategy": primary,
        "fallback_grasp_strategies": fallbacks,
        "parallel_jaw_allowed": policy.get("parallel_jaw_allowed", True),
        "grasp_risk_level": policy.get("risk_level", _RISK_MEDIUM),
        "required_grasp_width_m": eff_w,
        "effective_required_grasp_width_m": eff_w,
        "db_required_width_m": db_w,
        "required_width_source": policy.get("required_width_source", "db_required_width"),
        "measured_required_width_m": policy.get("measured_required_width_m"),
        "object_height_m": policy.get("object_height_m"),
        "db_height_m": policy.get("db_height_m"),
        "effective_height_m": policy.get("effective_height_m"),
        "dimension_source": policy.get("dimension_source"),
        "recommended_open_joint_m": policy.get("recommended_open_joint_m"),
        "recommended_close_joint_m": policy.get("recommended_close_joint_m"),
        "recommended_grasp_depth_from_top_m": policy.get("recommended_grasp_depth_from_top_m"),
        "depth_candidates_from_top_m": policy.get("depth_candidates_from_top_m"),
        "pregrasp_clearance_above_top_m": policy.get("pregrasp_clearance_above_top_m"),
        "safe_pregrasp_clearance_above_top_m": policy.get(
            "safe_pregrasp_clearance_above_top_m"
        ),
        "safe_pregrasp_extra_above_pregrasp_m": policy.get(
            "safe_pregrasp_extra_above_pregrasp_m"
        ),
        "approach_distance_min_m": policy.get("approach_distance_min_m"),
        "use_target_collision_until_pregrasp": policy.get(
            "use_target_collision_until_pregrasp"
        ),
        "remove_target_collision_before_descend": policy.get(
            "remove_target_collision_before_descend"
        ),
        "contact_required": policy.get("contact_required"),
        "min_contact_margin_m": policy.get("min_contact_margin_m"),
        "supported_for_parallel_jaw_topdown": policy.get(
            "supported_for_parallel_jaw_topdown"
        ),
        "max_attempts": policy.get("max_attempts"),
        "requires_known_box_center": policy.get("requires_known_box_center"),
        "min_yaw_confidence": policy.get("min_yaw_confidence"),
        "max_pose_fit_error": policy.get("max_pose_fit_error"),
        "allow_center_fallback": policy.get("allow_center_fallback"),
        "min_gripper_total_margin_m": policy.get("min_gripper_total_margin_m"),
        "side_grasp_tall_object_enabled": policy.get("side_grasp_tall_object_enabled"),
        "preferred_closing_axis": policy.get("preferred_closing_axis"),
        "prefer_oblique": policy.get("prefer_oblique"),
        "prefer_lateral": policy.get("prefer_lateral"),
        "prefer_edge": policy.get("prefer_edge"),
        "prefer_push_to_edge": policy.get("prefer_push_to_edge"),
        "grasp_policy_notes": policy.get("notes", ""),
        "footprint_major_m": footprint_major,
        "footprint_minor_m": footprint_minor,
        "dims_lwh": dims_lwh,
        "edge_grasp_requested": edge_grasp_requested,
        "edge_offset_m": (
            float(edge_offset_m)
            if edge_offset_m is not None
            else policy.get("edge_offset_m")
        ),
        "palm_clearance_above_top_m": policy.get("palm_clearance_above_top_m"),
        "max_cartesian_descend_m": policy.get("max_cartesian_descend_m"),
        "descend_velocity_scaling": policy.get("descend_velocity_scaling"),
        "descend_acceleration_scaling": policy.get("descend_acceleration_scaling"),
        "insertion_depth_limit_m": policy.get("insertion_depth_limit_m"),
        "min_tcp_clearance_above_table_m": policy.get("min_tcp_clearance_above_table_m"),
        "object_high_clearance_above_top_m": policy.get(
            "object_high_clearance_above_top_m"
        ),
        "preferred_cartesian_descend_m": policy.get("preferred_cartesian_descend_m"),
        "min_cartesian_descend_m": policy.get("min_cartesian_descend_m"),
        "yaw_policy": policy.get("yaw_policy"),
        "object_safe_above_clearance_m": policy.get("object_safe_above_clearance_m"),
        "max_expected_width_m": policy.get("max_expected_width_m"),
        "min_closure_delta_m": policy.get("min_closure_delta_m"),
        "contact_width_tolerance_m": policy.get("contact_width_tolerance_m"),
        "thin_edge_contact_width_tolerance_m": policy.get(
            "thin_edge_contact_width_tolerance_m"
        ),
        "max_allowed_thin_edge_compression_m": policy.get(
            "max_allowed_thin_edge_compression_m"
        ),
        "post_lift_min_delta_z_m": policy.get("post_lift_min_delta_z_m"),
        "min_pick_lift_m": policy.get("min_pick_lift_m"),
        "min_carry_tcp_z_m": policy.get("min_carry_tcp_z_m"),
        "carry_clearance_above_table_m": policy.get("carry_clearance_above_table_m"),
        "carry_clearance_above_obstacles_m": policy.get(
            "carry_clearance_above_obstacles_m"
        ),
        "attached_collision_padding_m": policy.get("attached_collision_padding_m"),
        "preferred_transport_corridor": policy.get("preferred_transport_corridor"),
        "release_z_m": policy.get("release_z_m"),
        "topdown_grasp_center_offset_long_m": policy.get("topdown_grasp_center_offset_long_m"),
        "topdown_grasp_center_offset_short_m": policy.get(
            "topdown_grasp_center_offset_short_m"
        ),
        "topdown_grasp_center_offset_local_xy_m": policy.get(
            "topdown_grasp_center_offset_local_xy_m"
        ),
        "min_top_z_m": policy.get("min_top_z_m"),
        "min_pregrasp_tcp_z_m": policy.get("min_pregrasp_tcp_z_m"),
        "use_palm_bridge_z_constraint": policy.get("use_palm_bridge_z_constraint"),
        "palm_bridge_clearance_above_top_m": policy.get("palm_bridge_clearance_above_top_m"),
        "palm_bridge_below_panda_hand_m": policy.get("palm_bridge_below_panda_hand_m"),
        "panda_hand_to_grasp_tcp_z_m": policy.get("panda_hand_to_grasp_tcp_z_m"),
        "micro_descend_from_pregrasp_enabled": policy.get(
            "micro_descend_from_pregrasp_enabled"
        ),
        "micro_descend_before_close_m": policy.get("micro_descend_before_close_m"),
        "max_micro_descend_before_close_m": policy.get("max_micro_descend_before_close_m"),
        "disable_normal_cartesian_descend": policy.get("disable_normal_cartesian_descend"),
        "single_close_attempt": policy.get("single_close_attempt"),
        "enable_z_candidates_after_contact_fail": policy.get(
            "enable_z_candidates_after_contact_fail"
        ),
        "supported_for_demo": policy.get("supported_for_demo"),
        "demo_priority": policy.get("demo_priority"),
        "visual_dims_estimated_m": policy.get("visual_dims_estimated_m"),
    }
    return out


__all__ = [
    "MAX_GRIPPER_WIDTH_M",
    "WIDTH_MARGIN_M",
    "SAFE_MAX_WIDTH_M",
    "MAX_FINGER_JOINT_M",
    "MIN_FINGER_JOINT_M",
    "LOW_OBJECT_HEIGHT_M",
    "VERY_LOW_OBJECT_HEIGHT_M",
    "LATERAL_GRASP_STRATEGIES",
    "SIDE_GRASP_TALL_STRATEGIES",
    "normalize_label",
    "finger_joint_from_total_width",
    "compute_open_close_joints",
    "get_grasp_policy",
    "apply_tall_object_topdown_grasp_center_offset",
    "resolve_tall_object_top_z_m",
    "export_grasp_policy_for_executor",
    "TALL_OBJECT_CAP_CENTER_SOURCE",
    "TALL_OBJECT_CAP_CENTER_CALIBRATED_SOURCE",
    "tall_object_topdown_cap_offset_configured",
    "resolve_effective_required_grasp_width",
    "get_collision_dimensions",
]
