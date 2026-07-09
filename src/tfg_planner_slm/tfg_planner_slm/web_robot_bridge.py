"""Puente web → ROS: ejecuta pick_place y clear_table en Gazebo sin confirmación interactiva."""

from __future__ import annotations

from typing import Callable, Optional, Tuple

from .command_dispatcher import InternalAction
from .execution_progress import ExecutionProgressTracker
from .ros_pick_place_cmd import (
    resolve_clear_table_pick_order,
    resolve_execution_progress_config,
)
from .ros_command_executor import (
    ExecutionResult,
    execute_clear_table_action,
    execute_pick_place_action,
)
from .ros_preflight import run_gazebo_preflight_checks
from .slm_backend_session import SlmBackendSession

GAZEBO_MISSING_MESSAGE = "Falta simulación con Gazebo."


def attempt_robot_action_in_gazebo(
    session: SlmBackendSession,
    action: InternalAction,
    *,
    ros_timeout_sec: float = 300.0,
    clear_table_ros_timeout_sec: Optional[float] = None,
    progress_callback: Optional[Callable[[dict], None]] = None,
    scene_id: Optional[str] = None,
) -> Tuple[Optional[ExecutionResult], bool]:
    """Intenta ejecutar pick_place o clear_table en Gazebo.

    Returns:
        (execution_result, simulation_unavailable)
    """
    if not action.execution_supported:
        return None, False

    if action.intent not in ("pick_place", "clear_table"):
        return None, False

    print("[WEB_API] comprobando Gazebo/ROS antes de ejecutar...", flush=True)
    preflight = run_gazebo_preflight_checks(timeout_s=10.0)
    if not preflight.ok:
        print("[WEB_API] preflight falló → %s" % GAZEBO_MISSING_MESSAGE, flush=True)
        for err in preflight.blocking_errors:
            print("[WEB_API] preflight: %s" % err, flush=True)
        return None, True

    if action.intent == "pick_place":
        print("[WEB_API] ejecutando pick_place en Gazebo (sin confirmación)...", flush=True)
        exec_result = execute_pick_place_action(
            action,
            slot_occupancy=session.slot_occupancy,
            execute_sim=True,
            assume_yes=True,
            i_understand_this_moves_gazebo_robot=True,
            ros_timeout_sec=ros_timeout_sec,
            progress_callback=progress_callback,
            scene_id=scene_id,
        )
        if (
            exec_result.started
            and exec_result.success
            and action.slot_index is not None
            and action.target_label
        ):
            session.slot_occupancy.mark_occupied(
                int(action.slot_index), str(action.target_label)
            )
            print(
                "[WEB_API] slot %d marcado ocupado por %s"
                % (int(action.slot_index), action.target_label),
                flush=True,
            )
        return exec_result, False

    step_timeout = float(clear_table_ros_timeout_sec or ros_timeout_sec)
    pick_order = resolve_clear_table_pick_order(scene_id)
    progress_cfg = resolve_execution_progress_config(scene_id)
    print(
        "[WEB_API] ejecutando clear_table en Gazebo (%d pasos mesa %s, total progreso %d, timeout/paso=%.0fs)..."
        % (
            len(pick_order),
            ",".join(pick_order),
            progress_cfg.total_objects,
            step_timeout,
        ),
        flush=True,
    )
    tracker = ExecutionProgressTracker(
        total_objects=progress_cfg.total_objects,
        progress_index_offset=progress_cfg.progress_index_offset,
        pick_order=pick_order,
    )
    tracker.mark_interpretation_done()
    if progress_callback is not None:
        tracker.set_callback(progress_callback)
        progress_callback(tracker.to_public_dict())
    exec_result = execute_clear_table_action(
        action,
        slot_occupancy=session.slot_occupancy,
        execute_sim=True,
        assume_yes=True,
        i_understand_this_moves_gazebo_robot=True,
        ros_timeout_sec=step_timeout,
        progress=tracker,
        progress_callback=progress_callback,
        scene_id=scene_id,
    )
    if exec_result.progress is None:
        exec_result.progress = tracker.to_public_dict()
    return exec_result, False


def attempt_pick_place_in_gazebo(
    session: SlmBackendSession,
    action: InternalAction,
    *,
    ros_timeout_sec: float = 300.0,
) -> Tuple[Optional[ExecutionResult], bool]:
    """Compatibilidad: delega en attempt_robot_action_in_gazebo."""
    return attempt_robot_action_in_gazebo(
        session, action, ros_timeout_sec=ros_timeout_sec
    )
