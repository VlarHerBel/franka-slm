"""Construcción del comando ROS pick_place (perfil validado, sin ejecutar)."""

from __future__ import annotations

import shlex
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

# Escena por defecto para validación SLM (cracker + sugar); debe coincidir con scene_preset del launch.
DEFAULT_PICK_PLACE_SCENE_ID = "two_boxes_01"

# Parámetros del perfil pick_place validado (defaults de perception_to_pregrasp_test).
VALIDATED_PROFILE_PARAMS: Tuple[Tuple[str, str], ...] = (
    ("execution_mode", "pick_place"),
    ("execution_cycle_mode", "snapshot_execute_home"),
    ("execute_once", "true"),
    ("plan_before_prelude", "true"),
    ("plan_before_prelude_skip_workspace_prelude", "false"),
    ("plan_before_motion_require_grasp_ik_from_home", "false"),
    ("enable_safe_pregrasp_stage", "false"),
    ("verify_gripper_gap_axis_after_pregrasp", "true"),
    ("enable_gripper_axis_in_place_correction", "true"),
    ("gripper_physical_yaw_correction_rad", "0.0"),
    ("gripper_axis_correction_backend", "joint7_direct"),
    ("gripper_gap_verify_target_angle_deg", "5.0"),
    ("gripper_gap_verify_max_angle_deg", "10.0"),
    ("gripper_axis_search_step_rad", "0.15"),
    ("gripper_axis_search_max_steps", "12"),
    ("gripper_axis_joint7_sign", "auto"),
    ("gripper_axis_search_direction", "auto"),
    ("gripper_centering_verify_enabled", "true"),
    ("gripper_centering_enable_xy_correction", "true"),
    ("use_shifted_grasp_candidates", "false"),
    ("lock_lift_orientation_to_current_tf", "true"),
    ("lift_use_current_xy_from_tf", "true"),
    ("lock_lift_orientation_allow_legacy_on_tf_fail", "false"),
    ("require_gripper_axis_target_for_pick", "true"),
    ("deterministic_transport_time_policy", "distance_based"),
    ("use_place_slots", "true"),
    ("enable_place_slots", "true"),
    ("place_slot_mode", "ordered_near_to_far"),
    ("demo_reset_completed_state_on_start", "false"),
    ("deposit_box_wall_top_z_m", "0.17"),
    ("place_slot_approach_tcp_z_m", "0.65"),
    ("place_slot_retreat_tcp_z_m", "0.65"),
    ("deposit_safety_margin_xy_m", "0.03"),
    ("return_home_after_place", "true"),
    ("return_home_after_place_via_safe_pose", "true"),
    ("return_home_after_execution", "true"),
    ("demo_authoritative_scene", "false"),
    ("clear_table_manual_step", "false"),
    ("paired_grid_max_runtime_sec", "120.0"),
    ("scene_id", DEFAULT_PICK_PLACE_SCENE_ID),
)

# Perfil demo_scene_02_clear_table: golden pick + transport acotado, un objeto por proceso.
CLEAR_TABLE_STEP_PARAMS: Tuple[Tuple[str, str], ...] = (
    ("execution_mode", "clear_table"),
    ("execution_cycle_mode", "snapshot_execute_home"),
    ("execute_once", "true"),
    ("plan_before_prelude", "true"),
    ("plan_before_prelude_skip_workspace_prelude", "false"),
    ("enable_safe_pregrasp_stage", "false"),
    ("verify_gripper_gap_axis_after_pregrasp", "true"),
    ("enable_gripper_axis_in_place_correction", "true"),
    ("gripper_axis_correction_backend", "joint7_direct"),
    ("demo_authoritative_scene", "true"),
    ("clear_table_manual_step", "true"),
    ("scene_id", "demo_scene_02"),
    ("demo_persist_completed_objects", "true"),
    ("demo_completed_state_file", "/tmp/tfg_demo_completed_objects.json"),
    ("execution_profile", "demo"),
    ("demo_fast_mode", "true"),
    ("demo_golden_pick_fast_execute", "true"),
    ("demo_require_full_pick_route_prevalidation", "false"),
    ("require_full_pick_route_preplanned_before_prelude", "false"),
    ("paired_grid_search_mode", "prioritized_or_cached"),
    ("paired_grid_max_candidates_runtime", "36"),
    ("paired_grid_max_runtime_sec", "120.0"),
    ("paired_grid_accept_first_valid", "true"),
    ("paired_joint7_offline_fast_mode", "true"),
    ("use_golden_execution_candidate", "false"),
    ("require_golden_execution_candidate", "false"),
    ("deterministic_transport_time_policy", "distance_based"),
    ("use_place_slots", "true"),
    ("enable_place_slots", "true"),
    ("place_slot_mode", "ordered_near_to_far"),
    ("return_home_after_place", "true"),
    ("return_home_after_execution", "true"),
    ("chips_can_use_legacy_successful_pick_policy", "true"),
    ("chips_can_use_actual_tf_descend_delta", "true"),
    ("chips_can_fast_joint7_axis_correction_enabled", "true"),
    ("chips_can_fast_joint7_error_threshold_deg", "25.0"),
    ("chips_can_fast_joint7_final_target_deg", "8.0"),
    ("chips_can_fast_joint7_hard_max_deg", "12.0"),
    ("mustard_fast_joint7_axis_correction_enabled", "true"),
    ("mustard_fast_joint7_error_threshold_deg", "20.0"),
    ("mustard_fast_joint7_final_target_deg", "8.0"),
    ("mustard_fast_joint7_hard_max_deg", "12.0"),
    ("sugar_box_final_descend_avoid_collisions", "false"),
    ("sugar_box_geometric_fallback_descend_fraction_threshold", "0.80"),
)

# Grasp/descend mostaza: close_joint_m baja = dedos más cerrados (0≈cerrado, ~0.04≈abierto).
MUSTARD_PICK_GRASP_PARAMS: Tuple[Tuple[str, str], ...] = (
    ("mustard_close_joint_m", "0.018"),
    ("mustard_min_required_depth_from_top_m", "0.046"),
    ("mustard_extra_micro_descend_after_cartesian_m", "0.022"),
    ("mustard_extra_micro_descend_max_m", "0.028"),
    ("mustard_min_bridge_clearance_after_microdescend_m", "0.0005"),
    ("mustard_palm_bridge_clearance_above_top_m", "0.0025"),
    ("mustard_post_descend_palm_bridge_tolerance_m", "0.0010"),
    ("mustard_post_descend_palm_bridge_min_at_grasp_m", "-0.005"),
    ("mustard_auto_extra_micro_descend_on_shortfall", "true"),
    ("mustard_grasp_tcp_z_tolerance_m", "0.004"),
    ("mustard_post_descend_depth_tolerance_m", "0.002"),
    ("post_grasp_pause_sec", "2.0"),
    ("mustard_operational_single_close_attempt", "true"),
    ("mustard_operational_hold_gripper_after_close", "true"),
    ("mustard_operational_skip_squeeze_after_close", "false"),
    ("mustard_lift_fallback_enabled", "true"),
    ("mustard_pregrasp_ik_joint_goal", "false"),
)

# Movimiento suave (comida): transporte y place más lentos en chips_mustard_*.
MUSTARD_FOOD_SAFE_MOTION_PARAMS: Tuple[Tuple[str, str], ...] = (
    ("deterministic_transport_time_scale", "2.2"),
    ("deterministic_transport_nominal_joint_speed_rad_s", "0.28"),
    ("deterministic_transport_min_segment_time_s", "1.8"),
    ("deterministic_transport_first_segment_min_time_s", "4.5"),
    ("deterministic_transport_segment_padding_s", "0.55"),
    ("place_approach_velocity_scaling", "0.035"),
    ("place_release_velocity_scaling", "0.02"),
    ("place_retreat_velocity_scaling", "0.03"),
    ("lift_velocity_scaling", "0.06"),
    ("deterministic_transport_time_scale_return_home", "1.0"),
    ("post_place_return_home_joint_waypoint_velocity_scaling", "0.35"),
    ("post_place_return_home_joint_waypoint_acceleration_scaling", "0.35"),
)

CLEAR_TABLE_PICK_ORDER: Tuple[str, ...] = (
    "cracker_box",
    "chips_can",
    "sugar_box",
    "mustard_bottle",
)

TWO_BOXES_CLEAR_TABLE_PICK_ORDER: Tuple[str, ...] = (
    "cracker_box",
    "sugar_box",
)

CHIPS_MUSTARD_CLEAR_TABLE_PICK_ORDER: Tuple[str, ...] = (
    "chips_can",
    "mustard_bottle",
)

DEMO_SCENE_3OBJ_CLEAR_TABLE_PICK_ORDER: Tuple[str, ...] = (
    "cracker_box",
    "chips_can",
    "sugar_box",
)

CLEAR_TABLE_MAX_STEPS = len(CLEAR_TABLE_PICK_ORDER)


@dataclass(frozen=True)
class ExecutionProgressConfig:
    pick_order: Tuple[str, ...]
    total_objects: int
    progress_index_offset: int


def resolve_execution_progress_config(
    scene_id: Optional[str] = None,
) -> ExecutionProgressConfig:
    """Total x/N de la UI y pasos reales de mesa (depósito precargado suma al contador)."""
    sid = str(scene_id or DEFAULT_PICK_PLACE_SCENE_ID).strip().lower()
    if sid.startswith("deposit_"):
        from .deposit_scene_loader import (
            load_initial_deposit_slots,
            resolve_table_pick_order,
        )

        table_order = resolve_table_pick_order(sid)
        deposit_count = len(load_initial_deposit_slots(sid))
        return ExecutionProgressConfig(
            pick_order=table_order,
            total_objects=deposit_count + len(table_order),
            progress_index_offset=deposit_count,
        )
    pick_order = resolve_clear_table_pick_order(scene_id)
    return ExecutionProgressConfig(
        pick_order=pick_order,
        total_objects=len(pick_order),
        progress_index_offset=0,
    )


def clear_table_max_steps(scene_id: Optional[str] = None) -> int:
    return resolve_execution_progress_config(scene_id).total_objects


def _is_demo_multiobject_scene_id(scene_id: Optional[str]) -> bool:
    sid = str(scene_id or "").strip().lower()
    if sid.endswith("_nogolden"):
        sid = sid.removesuffix("_nogolden")
    if sid in ("demo_scene_01", "demo_scene_02", "demo_scene_03"):
        return True
    return (
        sid.endswith("_3obj")
        and sid.removesuffix("_3obj") in ("demo_scene_01", "demo_scene_02", "demo_scene_03")
    )


def scene_uses_demo_golden_fast_execute(
    scene_id: Optional[str],
    *,
    target_label: Optional[str] = None,
) -> bool:
    """Golden fast execute: demo multiobjeto; sugar solo tras cracker/chips o escenas debug."""
    label = str(target_label or "").strip().lower()
    sid = str(scene_id or "").strip().lower()
    if sid.startswith("deposit_"):
        from panda_vision.spawn.demo_scene_presets import (
            demo_scene_policy_scene_id_for_preset,
        )

        if demo_scene_policy_scene_id_for_preset(sid) != "demo_scene_02":
            return False
        from .deposit_scene_loader import resolve_table_pick_order

        return label in resolve_table_pick_order(sid)
    if sid == "chips_mustard_02":
        from panda_controller.demo_golden_pick_candidate import golden_fast_execute_available

        return golden_fast_execute_available(sid, label)
    if label == "sugar_box":
        from panda_controller.sugar_box_safe_entry import sugar_box_scene_allows_golden

        if sugar_box_scene_allows_golden(sid):
            return True
        if sid in ("demo_scene_02", "demo_scene_02_clear_table"):
            return True
        return False
    return _is_demo_multiobject_scene_id(sid) or sid in (
        "demo_scene_01",
        "demo_scene_02",
        "demo_scene_03",
    )


def _apply_golden_fast_execute_override(
    merged: Dict[str, str],
    scene_id: Optional[str],
    *,
    target_label: Optional[str] = None,
) -> None:
    if not scene_uses_demo_golden_fast_execute(scene_id, target_label=target_label):
        merged["demo_golden_pick_fast_execute"] = "false"


def demo_multiobject_pick_place_overrides(
    scene_id: Optional[str],
) -> Tuple[Tuple[str, str], ...]:
    """Escenas demo multiobjeto: obstáculos autoritativos + perfil demo en pick_place suelto."""
    if not _is_demo_multiobject_scene_id(scene_id):
        if not str(scene_id or "").strip().lower().startswith("deposit_"):
            return ()
    return (
        ("demo_authoritative_scene", "true"),
        ("execution_profile", "demo"),
    )


def resolve_clear_table_pick_order(scene_id: Optional[str] = None) -> Tuple[str, ...]:
    """Orden de clear_table según escena."""
    sid = str(scene_id or DEFAULT_PICK_PLACE_SCENE_ID).strip().lower()
    if sid.startswith("deposit_"):
        from .deposit_scene_loader import resolve_table_pick_order

        table_order = resolve_table_pick_order(sid)
        if table_order:
            return table_order
    if sid.endswith("_nogolden"):
        sid = sid.removesuffix("_nogolden")
    if sid.startswith("two_boxes"):
        return TWO_BOXES_CLEAR_TABLE_PICK_ORDER
    if sid.startswith("chips_mustard"):
        return CHIPS_MUSTARD_CLEAR_TABLE_PICK_ORDER
    if sid.endswith("_3obj") and sid.removesuffix("_3obj") in (
        "demo_scene_01",
        "demo_scene_02",
        "demo_scene_03",
    ):
        return DEMO_SCENE_3OBJ_CLEAR_TABLE_PICK_ORDER
    return CLEAR_TABLE_PICK_ORDER


# Perfil clear_table genérico (two_boxes, sin golden demo_scene_02).
GENERIC_CLEAR_TABLE_EXTRA_PARAMS: Tuple[Tuple[str, str], ...] = (
    ("execution_mode", "clear_table"),
    ("clear_table_manual_step", "true"),
    ("demo_persist_completed_objects", "true"),
    ("demo_completed_state_file", "/tmp/tfg_demo_completed_objects.json"),
    ("demo_reset_completed_state_on_start", "false"),
    ("require_full_pick_route_preplanned_before_prelude", "false"),
    ("paired_grid_search_mode", "prioritized_or_cached"),
    ("paired_grid_max_candidates_runtime", "36"),
    ("paired_grid_accept_first_valid", "true"),
    ("paired_joint7_offline_fast_mode", "true"),
    ("sugar_box_final_descend_avoid_collisions", "false"),
    ("sugar_box_geometric_fallback_descend_fraction_threshold", "0.80"),
)

# Parámetros de ejecución (grid joint7, prevalidación relajada) sin atar a demo_scene_02.
_DEMO_SCENE_BINDING_KEYS: frozenset = frozenset(
    {
        "execution_mode",
        "clear_table_manual_step",
        "demo_authoritative_scene",
        "scene_id",
        "demo_persist_completed_objects",
        "demo_completed_state_file",
    }
)

PICK_PLACE_EXECUTION_HELPER_PARAMS: Tuple[Tuple[str, str], ...] = tuple(
    (name, value)
    for name, value in CLEAR_TABLE_STEP_PARAMS
    if name not in _DEMO_SCENE_BINDING_KEYS
)

# Parámetros demo compartidos (pick_place suelto y clear_table). Excluir solo lo propio de clear_table.
_PICK_PLACE_OMIT_FROM_DEMO: frozenset = frozenset(
    {
        "execution_mode",
        "clear_table_manual_step",
    }
)


def demo_pick_params_for_target(target_label: str) -> Tuple[Tuple[str, str], ...]:
    """Perfil demo_scene_02 para un objeto YCB concreto (pick_place directo)."""
    label = str(target_label or "").strip().lower()
    if label not in CLEAR_TABLE_PICK_ORDER:
        return ()
    return tuple(
        (name, value)
        for name, value in CLEAR_TABLE_STEP_PARAMS
        if name not in _PICK_PLACE_OMIT_FROM_DEMO
    )


def _ros_param_args(name: str, value: str) -> List[str]:
    return ["-p", "%s:=%s" % (name, value)]


def build_pick_place_ros2_args(
    *,
    dry_run: bool,
    target_label: str,
    slot_index: Optional[int] = None,
    slot_name: Optional[str] = None,
    slot_user_specified: bool = False,
    demo_profile: bool = False,
    scene_id: Optional[str] = None,
) -> List[str]:
    """Argv para `ros2 run panda_controller perception_to_pregrasp_test`."""
    argv: List[str] = [
        "ros2",
        "run",
        "panda_controller",
        "perception_to_pregrasp_test",
        "--ros-args",
    ]
    argv.extend(_ros_param_args("dry_run", str(bool(dry_run)).lower()))
    argv.extend(_ros_param_args("target_label", target_label))
    if slot_user_specified and slot_index is not None:
        argv.extend(_ros_param_args("place_slot_user_specified", "true"))
        argv.extend(_ros_param_args("place_slot_index", str(int(slot_index))))
        if slot_name:
            argv.extend(_ros_param_args("place_slot_name", str(slot_name)))
    merged_params = dict(VALIDATED_PROFILE_PARAMS)
    merged_params.update(dict(PICK_PLACE_EXECUTION_HELPER_PARAMS))
    if demo_profile:
        merged_params.update(dict(demo_pick_params_for_target(target_label)))
    label = str(target_label or "").strip().lower()
    if label == "mustard_bottle":
        merged_params.update(dict(MUSTARD_PICK_GRASP_PARAMS))
    sid = str(scene_id or "").strip()
    if sid and not demo_profile:
        merged_params["scene_id"] = sid
    merged_params.update(dict(demo_multiobject_pick_place_overrides(sid or merged_params.get("scene_id"))))
    if sid == "chips_mustard_01":
        merged_params["demo_golden_pick_fast_execute"] = "false"
    _apply_golden_fast_execute_override(
        merged_params, sid or merged_params.get("scene_id"), target_label=label
    )
    if (
        sid in ("chips_mustard_01", "chips_mustard_02")
        and label == "mustard_bottle"
        and merged_params.get("demo_golden_pick_fast_execute") == "false"
    ):
        merged_params.update(dict(MUSTARD_FOOD_SAFE_MOTION_PARAMS))
    for name, value in merged_params.items():
        argv.extend(_ros_param_args(name, value))
    return argv


def build_clear_table_step_ros2_args(
    *,
    dry_run: bool,
    reset_completed_state: bool = False,
    step_label: Optional[str] = None,
    scene_id: Optional[str] = None,
    demo_profile: bool = False,
    slot_index: Optional[int] = None,
    slot_user_specified: bool = False,
) -> List[str]:
    """Un paso de clear_table (un pick+place); el orden lo gestiona el controlador."""
    argv: List[str] = [
        "ros2",
        "run",
        "panda_controller",
        "perception_to_pregrasp_test",
        "--ros-args",
    ]
    argv.extend(_ros_param_args("dry_run", str(bool(dry_run)).lower()))
    argv.extend(
        _ros_param_args(
            "target_label",
            str(step_label).strip().lower() if step_label else "clear_table",
        )
    )
    argv.extend(
        _ros_param_args(
            "demo_reset_completed_state_on_start",
            str(bool(reset_completed_state)).lower(),
        )
    )
    if slot_user_specified and slot_index is not None:
        argv.extend(_ros_param_args("place_slot_user_specified", "true"))
        argv.extend(_ros_param_args("place_slot_index", str(int(slot_index))))
    if demo_profile:
        merged = dict(CLEAR_TABLE_STEP_PARAMS)
        sid = str(scene_id or merged.get("scene_id", "demo_scene_02")).strip()
        merged["scene_id"] = sid
        step = str(step_label or "").strip().lower()
        _apply_golden_fast_execute_override(merged, sid, target_label=step or None)
        for name, value in merged.items():
            argv.extend(_ros_param_args(name, value))
        return argv
    sid = str(scene_id or DEFAULT_PICK_PLACE_SCENE_ID).strip()
    merged = dict(VALIDATED_PROFILE_PARAMS)
    merged.update(dict(PICK_PLACE_EXECUTION_HELPER_PARAMS))
    merged.update(dict(GENERIC_CLEAR_TABLE_EXTRA_PARAMS))
    merged["scene_id"] = sid
    merged.update(dict(demo_multiobject_pick_place_overrides(sid)))
    step = str(step_label or "").strip().lower()
    if step == "mustard_bottle" and not scene_uses_demo_golden_fast_execute(
        sid, target_label=step
    ):
        merged.update(dict(MUSTARD_PICK_GRASP_PARAMS))
    _apply_golden_fast_execute_override(merged, sid, target_label=step or None)
    for name, value in merged.items():
        argv.extend(_ros_param_args(name, value))
    return argv


def build_clear_table_ros2_command_string(
    *,
    dry_run: bool,
    reset_completed_state: bool = True,
    scene_id: Optional[str] = None,
    demo_profile: bool = False,
) -> str:
    return shlex.join(
        build_clear_table_step_ros2_args(
            dry_run=dry_run,
            reset_completed_state=reset_completed_state,
            scene_id=scene_id,
            demo_profile=demo_profile,
        )
    )


def build_pick_place_ros2_command_string(
    *,
    dry_run: bool,
    target_label: str,
    slot_index: Optional[int] = None,
    slot_name: Optional[str] = None,
    slot_user_specified: bool = False,
    demo_profile: bool = False,
    scene_id: Optional[str] = None,
) -> str:
    return shlex.join(
        build_pick_place_ros2_args(
            dry_run=dry_run,
            target_label=target_label,
            slot_index=slot_index,
            slot_name=slot_name,
            slot_user_specified=slot_user_specified,
            demo_profile=demo_profile,
            scene_id=scene_id,
        )
    )
