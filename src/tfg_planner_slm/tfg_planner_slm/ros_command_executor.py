"""Ejecución segura de comandos ROS para pick_place (sin shell=True).

Por defecto no ejecuta nada: la CLI debe pasar flags explícitos.
En Fase 1 se permite solo dry_run:=true.
En Fase 2 se permite dry_run:=false solo con confirmación fuerte.

El perfil de parámetros explícitos coincide con los defaults validados de
perception_to_pregrasp_test.py (trazabilidad aunque el nodo ya los tenga por defecto).
"""

from __future__ import annotations

import shlex
import subprocess
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

from .command_dispatcher import InternalAction
from .execution_progress import ExecutionProgressTracker
from .ros_pick_place_cmd import (
    build_clear_table_step_ros2_args,
    build_pick_place_ros2_args,
    resolve_clear_table_pick_order,
    resolve_execution_progress_config,
)
from .slot_state import SlotOccupancy

_FAILURE_MARKERS: Tuple[str, ...] = (
    "Rejecting grasp",
    "gripper margin too small",
    "[GEOMETRIC_GRASP_CHECK] Rejecting",
    "all candidates failed",
    "aborting pick",
    "Fallo etapa",
    "Planning failed! Error code: FAILURE",
    "descend orientation lock FAIL",
    "[CHIPS_CAN_PRE_DESCEND_POSE_GATE] result=FAIL",
    "target_collision_not_removed_before_descend",
    "chips_can descend depth plan FAIL",
    "chips_can legacy actual tf descend failed",
)

_TERMINAL_FAIL_PATTERNS: Tuple[str, ...] = (
    "[CLEAR_TABLE_MANUAL_STEP]",
    "[CLEAR_TABLE_COMPLETE]",
    "[POST_PRELUDE_PREGRASP_REPLAN]",
    "[POST_PRELUDE_PICK_ROUTE_VALIDATE]",
)

_PICK_PLACE_SUCCESS_MARKERS: Tuple[str, ...] = (
    "[PLACE] deterministic sequence completed successfully",
    "[MODE] execution_mode='pick_place' completado.",
    "[PLACE_CANDIDATE] idx=0 result=OK",
    "[PLACE_RELEASE_SELECTED]",
    "[DEMO_TARGET_DONE]",
)

_CLEAR_TABLE_COMPLETE_MARKER = "[CLEAR_TABLE_COMPLETE]"
_CLEAR_TABLE_STEP_DONE_MARKER = "[CLEAR_TABLE_TARGET_DONE]"


@dataclass
class ExecutionResult:
    started: bool
    success: bool
    returncode: int | None
    stdout: str
    stderr: str
    duration_s: float | None
    skipped_reason: str = ""
    failure_reason: str = ""
    steps_completed: int = 0
    progress: Optional[dict] = None


def _terminal_failure_in_output(combined: str) -> Optional[str]:
    """Fallos terminales (no logs intermedios de reintento con result=FAIL)."""
    if _CLEAR_TABLE_STEP_DONE_MARKER in combined:
        return None
    if _clear_table_sequence_complete(combined):
        return None
    if any(m in combined for m in _PICK_PLACE_SUCCESS_MARKERS):
        return None
    lower = combined.lower()
    for marker in _FAILURE_MARKERS:
        if marker.lower() in lower:
            return "failure_marker:%s" % marker
    tail = combined[-8000:]
    if "result=FAIL" in tail:
        for pat in _TERMINAL_FAIL_PATTERNS:
            if pat in tail and "result=FAIL" in tail.split(pat)[-1][:400]:
                return "failure_marker:terminal_result=FAIL"
    return None


def analyze_controller_output(
    stdout: str,
    stderr: str,
    returncode: int | None,
    *,
    require_pick_place_success_markers: bool,
    require_clear_table_complete: bool = False,
) -> Tuple[bool, str]:
    """Interpreta salida de perception_to_pregrasp_test (returncode no basta)."""
    combined = "%s\n%s" % (stdout or "", stderr or "")

    if require_clear_table_complete:
        if _clear_table_sequence_complete(combined):
            return True, "clear_table_complete"
        fail = _terminal_failure_in_output(combined)
        if fail:
            return False, fail
        return False, "missing_clear_table_complete"

    if require_pick_place_success_markers:
        if any(m in combined for m in _PICK_PLACE_SUCCESS_MARKERS):
            return True, "pick_place_success_markers"
        if _CLEAR_TABLE_STEP_DONE_MARKER in combined:
            return True, "clear_table_step_done"
        fail = _terminal_failure_in_output(combined)
        if fail:
            return False, fail
        return False, "missing_pick_place_success_markers"

    fail = _terminal_failure_in_output(combined)
    if fail:
        return False, fail

    if returncode is not None and returncode != 0:
        return False, "nonzero_returncode"

    return True, "returncode_ok"


def _stream_subprocess_output(
    proc: subprocess.Popen,
    *,
    on_line: Optional[Callable[[str, str], None]] = None,
) -> Tuple[str, str]:
    stdout_chunks: List[str] = []
    stderr_chunks: List[str] = []

    def _reader(stream, name: str, out_list: List[str]) -> None:
        for line in iter(stream.readline, ""):
            out_list.append(line)
            if on_line is not None:
                on_line(line, name)

    threads = [
        threading.Thread(target=_reader, args=(proc.stdout, "stdout", stdout_chunks), daemon=True),
        threading.Thread(target=_reader, args=(proc.stderr, "stderr", stderr_chunks), daemon=True),
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return "".join(stdout_chunks), "".join(stderr_chunks)


def _clear_table_sequence_complete(combined: str) -> bool:
    if _CLEAR_TABLE_COMPLETE_MARKER not in combined:
        return False
    for line in combined.splitlines():
        if _CLEAR_TABLE_COMPLETE_MARKER in line and "result=OK" in line:
            return True
    return "result=OK" in combined.split(_CLEAR_TABLE_COMPLETE_MARKER, 1)[-1][:200]


_FAILURE_CONTEXT_PATTERNS: Tuple[str, ...] = (
    "[GRASP_CANDIDATE]",
    "[CHIPS_CAN",
    "[PICK_ABORT",
    "PICK_ABORT_SAFE",
    "[CHIPS_CAN_TOP_CLEARANCE",
    "[CHIPS_CAN_LEGACY",
    "[CHIPS_CAN_DESCEND",
    "[CHIPS_CAN_PRE_DESCEND",
    "[CHIPS_CAN_FINAL_DESCEND",
    "[CHIPS_CAN_CARTESIAN",
    "[LOW_OBJECT",
    "[GRIPPER_CENTERING]",
    "target_collision_not_removed",
    "all candidates failed",
    "[HOME_FAILED]",
    "[ATTACH]",
    "[MOVE_HOME]",
    "object_high_pregrasp",
    "high_to_low",
)


def _extract_failure_context_lines(combined: str, *, max_lines: int = 40) -> List[str]:
    """Líneas de diagnóstico del fallo real (no solo reintentos MOVE_HOME al final)."""
    lines = [ln for ln in (combined or "").splitlines() if ln.strip()]
    if not lines:
        return []
    matched: List[str] = []
    for ln in lines:
        if any(pat in ln for pat in _FAILURE_CONTEXT_PATTERNS):
            matched.append(ln)
    if not matched:
        return lines[-max_lines:]
    return matched[-max_lines:]


def _log_output_tail(combined: str, *, reason: str, max_lines: int = 60) -> None:
    lines = [ln for ln in (combined or "").splitlines() if ln.strip()]
    if not lines:
        return
    context = _extract_failure_context_lines(combined, max_lines=max_lines)
    tail = lines[-max_lines:]
    print(
        "[ROS_EXECUTOR] output tail (%s, last %d lines):" % (reason, len(tail)),
        flush=True,
    )
    for ln in tail:
        print("[ROS_EXECUTOR][tail] %s" % ln, flush=True)
    if context and context != tail[-len(context) :]:
        print(
            "[ROS_EXECUTOR] failure context (%s, %d lines):"
            % (reason, len(context)),
            flush=True,
        )
        for ln in context:
            print("[ROS_EXECUTOR][context] %s" % ln, flush=True)


def _run_controller_subprocess(
    argv: List[str],
    *,
    ros_timeout_sec: float,
    require_pick_place_success_markers: bool,
    require_clear_table_complete: bool = False,
    log_label: str = "",
    progress: Optional[ExecutionProgressTracker] = None,
) -> ExecutionResult:
    print("[ROS_EXECUTOR] starting %s" % (log_label or "command"), flush=True)
    print("[ROS_EXECUTOR] argv=%s" % shlex.join(argv), flush=True)

    t0 = time.perf_counter()

    def _on_line(line: str, stream: str) -> None:
        if progress is not None:
            progress.on_log_line(line, stream=stream)

    try:
        proc = subprocess.Popen(
            argv,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        try:
            stdout, stderr = _stream_subprocess_output(proc, on_line=_on_line)
            returncode = proc.wait(timeout=float(ros_timeout_sec))
        except subprocess.TimeoutExpired:
            proc.kill()
            stdout, stderr = _stream_subprocess_output(proc)
            duration_s = time.perf_counter() - t0
            print(
                "[ROS_EXECUTOR] failed returncode=timeout duration_s=%.2f" % duration_s,
                flush=True,
            )
            return ExecutionResult(
                started=True,
                success=False,
                returncode=None,
                stdout=stdout,
                stderr=stderr,
                duration_s=duration_s,
                skipped_reason="timeout",
                failure_reason="timeout",
                progress=progress.to_public_dict() if progress else None,
            )

        duration_s = time.perf_counter() - t0
        ok, reason = analyze_controller_output(
            stdout,
            stderr,
            returncode,
            require_pick_place_success_markers=require_pick_place_success_markers,
            require_clear_table_complete=require_clear_table_complete,
        )
        if not ok:
            if returncode == 0 and require_pick_place_success_markers:
                print(
                    "[ROS_EXECUTOR] returncode=0 but output indicates failure (%s)"
                    % reason,
                    flush=True,
                )
            _log_output_tail("%s\n%s" % (stdout or "", stderr or ""), reason=reason)
        print(
            "[ROS_EXECUTOR] %s returncode=%s duration_s=%.2f reason=%s"
            % (
                "success" if ok else "failed",
                str(returncode),
                duration_s,
                reason,
            ),
            flush=True,
        )
        return ExecutionResult(
            started=True,
            success=ok,
            returncode=returncode,
            stdout=stdout,
            stderr=stderr,
            duration_s=duration_s,
            skipped_reason="" if ok else reason,
            failure_reason="" if ok else reason,
            progress=progress.to_public_dict() if progress else None,
        )
    except OSError as exc:
        duration_s = time.perf_counter() - t0
        print("[ROS_EXECUTOR] failed reason=os_error:%s" % exc, flush=True)
        return ExecutionResult(
            started=False,
            success=False,
            returncode=None,
            stdout="",
            stderr=str(exc),
            duration_s=duration_s,
            skipped_reason="os_error",
            failure_reason="os_error",
            progress=progress.to_public_dict() if progress else None,
        )


def _confirm(question: str) -> bool:
    try:
        ans = input(question).strip().lower()
    except (EOFError, KeyboardInterrupt):
        return False
    return ans in ("s", "si", "sí", "y", "yes")


def execute_pick_place_action(
    action: InternalAction,
    *,
    slot_occupancy: Optional[SlotOccupancy] = None,
    execute: bool = False,
    execute_sim: bool = False,
    assume_yes: bool = False,
    i_understand_this_moves_gazebo_robot: bool = False,
    ros_timeout_sec: float = 300.0,
    progress: Optional[ExecutionProgressTracker] = None,
    progress_callback: Optional[Callable[[dict], None]] = None,
    demo_profile: bool = False,
    scene_id: Optional[str] = None,
) -> ExecutionResult:
    """Ejecuta (si procede) una acción pick_place segura vía subprocess.run()."""
    if not execute and not execute_sim:
        return ExecutionResult(
            started=False,
            success=False,
            returncode=None,
            stdout="",
            stderr="",
            duration_s=None,
            skipped_reason="not_requested",
        )

    if action.intent != "pick_place" or not action.execution_supported:
        return ExecutionResult(
            started=False,
            success=False,
            returncode=None,
            stdout="",
            stderr="",
            duration_s=None,
            skipped_reason="not_execution_supported",
        )

    if action.target_label is None or action.slot_index is None:
        return ExecutionResult(
            started=False,
            success=False,
            returncode=None,
            stdout="",
            stderr="",
            duration_s=None,
            skipped_reason="missing_target_or_slot",
        )

    occupancy = slot_occupancy if slot_occupancy is not None else SlotOccupancy()
    can, reason = occupancy.can_place(int(action.slot_index), str(action.target_label))
    if not can:
        return ExecutionResult(
            started=False,
            success=False,
            returncode=None,
            stdout="",
            stderr="",
            duration_s=None,
            skipped_reason="slot_blocked:%s" % reason,
        )

    if execute and execute_sim:
        return ExecutionResult(
            started=False,
            success=False,
            returncode=None,
            stdout="",
            stderr="",
            duration_s=None,
            skipped_reason="conflicting_flags_execute_and_execute_sim",
        )

    dry_run = True if execute else False
    require_success_markers = bool(execute_sim)

    if execute_sim and not i_understand_this_moves_gazebo_robot:
        return ExecutionResult(
            started=False,
            success=False,
            returncode=None,
            stdout="",
            stderr="",
            duration_s=None,
            skipped_reason="missing_i_understand_flag",
        )

    if execute_sim and not assume_yes:
        question = (
            "Esto moverá el robot en Gazebo. Acción: pick_place %s -> slot %d. ¿Continuar? [s/N] "
            % (action.target_label, int(action.slot_index))
        )
        if not _confirm(question):
            return ExecutionResult(
                started=False,
                success=False,
                returncode=None,
                stdout="",
                stderr="",
                duration_s=None,
                skipped_reason="user_declined",
            )

    if execute and not assume_yes:
        question = (
            "¿Ejecutar dry_run de pick_place de %s al slot %d? [s/N] "
            % (action.target_label, int(action.slot_index))
        )
        if not _confirm(question):
            return ExecutionResult(
                started=False,
                success=False,
                returncode=None,
                stdout="",
                stderr="",
                duration_s=None,
                skipped_reason="user_declined",
            )

    argv = build_pick_place_ros2_args(
        dry_run=dry_run,
        target_label=str(action.target_label),
        slot_index=int(action.slot_index),
        slot_user_specified=True,
        demo_profile=demo_profile,
        scene_id=scene_id,
    )
    print(
        "[ROS_EXECUTOR] starting command dry_run=%s target=%s slot=%d"
        % (str(dry_run).lower(), action.target_label, int(action.slot_index)),
        flush=True,
    )
    print("[ROS_EXECUTOR] argv=%s" % shlex.join(argv), flush=True)

    if progress is None:
        progress = ExecutionProgressTracker()
    progress.mark_interpretation_done()
    if action.target_label:
        progress.on_step_start(1, str(action.target_label))
    if progress_callback is not None:
        progress.set_callback(progress_callback)

    result = _run_controller_subprocess(
        argv,
        ros_timeout_sec=ros_timeout_sec,
        require_pick_place_success_markers=require_success_markers,
        log_label="pick_place dry_run=%s target=%s slot=%d"
        % (str(dry_run).lower(), action.target_label, int(action.slot_index)),
        progress=progress,
    )
    if action.target_label and result.duration_s is not None:
        progress.on_step_finished(
            1,
            str(action.target_label),
            float(result.duration_s),
            bool(result.started and result.success),
        )
    result.progress = progress.to_public_dict()
    return result


def execute_clear_table_action(
    action: InternalAction,
    *,
    slot_occupancy: Optional[SlotOccupancy] = None,
    execute_sim: bool = False,
    assume_yes: bool = False,
    i_understand_this_moves_gazebo_robot: bool = False,
    ros_timeout_sec: float = 300.0,
    progress: Optional[ExecutionProgressTracker] = None,
    progress_callback: Optional[Callable[[dict], None]] = None,
    scene_id: Optional[str] = None,
) -> ExecutionResult:
    """Ejecuta clear_table como secuencia de pasos (un subprocess por objeto)."""
    if not execute_sim:
        return ExecutionResult(
            started=False,
            success=False,
            returncode=None,
            stdout="",
            stderr="",
            duration_s=None,
            skipped_reason="not_requested",
        )

    if action.intent != "clear_table" or not action.execution_supported:
        return ExecutionResult(
            started=False,
            success=False,
            returncode=None,
            stdout="",
            stderr="",
            duration_s=None,
            skipped_reason="not_execution_supported",
        )

    if execute_sim and not i_understand_this_moves_gazebo_robot:
        return ExecutionResult(
            started=False,
            success=False,
            returncode=None,
            stdout="",
            stderr="",
            duration_s=None,
            skipped_reason="missing_i_understand_flag",
        )

    if execute_sim and not assume_yes:
        question = (
            "Esto moverá el robot en Gazebo para recoger toda la mesa. ¿Continuar? [s/N] "
        )
        if not _confirm(question):
            return ExecutionResult(
                started=False,
                success=False,
                returncode=None,
                stdout="",
                stderr="",
                duration_s=None,
                skipped_reason="user_declined",
            )

    from .deposit_scene_loader import seed_slot_occupancy_from_scene

    occupancy = slot_occupancy if slot_occupancy is not None else SlotOccupancy()
    occupancy.reset()
    seed_slot_occupancy_from_scene(occupancy, str(scene_id or ""))

    progress_cfg = resolve_execution_progress_config(scene_id)
    pick_order = list(progress_cfg.pick_order)
    max_steps = len(pick_order)
    if progress is None:
        progress = ExecutionProgressTracker(
            total_objects=progress_cfg.total_objects,
            progress_index_offset=progress_cfg.progress_index_offset,
            pick_order=tuple(pick_order),
        )
    progress.mark_interpretation_done()
    if progress_callback is not None:
        progress.set_callback(progress_callback)

    combined_stdout: List[str] = []
    combined_stderr: List[str] = []
    total_duration = 0.0
    steps_done = 0

    for step_idx in range(max_steps):
        label = pick_order[step_idx]
        if progress is not None:
            progress.on_step_start(step_idx + 1, label)

        free_slot = occupancy.first_free_slot()
        if free_slot is None:
            return ExecutionResult(
                started=True,
                success=False,
                returncode=None,
                stdout="\n".join(combined_stdout),
                stderr="\n".join(combined_stderr),
                duration_s=total_duration,
                skipped_reason="clear_table_no_free_slot",
                failure_reason="clear_table_no_free_slot",
                steps_completed=steps_done,
                progress=progress.to_public_dict() if progress else None,
            )

        argv = build_clear_table_step_ros2_args(
            dry_run=False,
            reset_completed_state=(step_idx == 0),
            step_label=label,
            scene_id=scene_id,
            slot_index=int(free_slot),
            slot_user_specified=True,
        )
        print(
            "[ROS_EXECUTOR] clear_table deposit slot=%d (first free) label=%s"
            % (int(free_slot), label),
            flush=True,
        )
        print(
            "[ROS_EXECUTOR] clear_table step=%d/%d label=%s scene_id=%s"
            % (
                step_idx + 1,
                max_steps,
                label,
                str(scene_id or "default"),
            ),
            flush=True,
        )
        step_result = _run_controller_subprocess(
            argv,
            ros_timeout_sec=ros_timeout_sec,
            require_pick_place_success_markers=True,
            log_label="clear_table step %d" % (step_idx + 1),
            progress=progress,
        )
        combined_stdout.append(step_result.stdout or "")
        combined_stderr.append(step_result.stderr or "")
        if step_result.duration_s is not None:
            total_duration += float(step_result.duration_s)

        if progress is not None:
            progress.on_step_finished(
                step_idx + 1,
                label,
                float(step_result.duration_s or 0.0),
                bool(step_result.started and step_result.success),
            )

        if not step_result.started or not step_result.success:
            return ExecutionResult(
                started=True,
                success=False,
                returncode=step_result.returncode,
                stdout="\n".join(combined_stdout),
                stderr="\n".join(combined_stderr),
                duration_s=total_duration,
                skipped_reason=step_result.failure_reason or "step_failed",
                failure_reason=step_result.failure_reason or "step_failed",
                steps_completed=steps_done,
                progress=progress.to_public_dict() if progress else None,
            )

        steps_done += 1
        occupancy.mark_occupied(int(free_slot), label, overwrite=True)

        combined = "\n".join(combined_stdout + combined_stderr)
        if _clear_table_sequence_complete(combined):
            break

    combined = "\n".join(combined_stdout + combined_stderr)
    if not _clear_table_sequence_complete(combined) and steps_done < max_steps:
        return ExecutionResult(
            started=True,
            success=False,
            returncode=0,
            stdout="\n".join(combined_stdout),
            stderr="\n".join(combined_stderr),
            duration_s=total_duration,
            skipped_reason="missing_clear_table_complete",
            failure_reason="missing_clear_table_complete",
            steps_completed=steps_done,
        )

    print(
        "[ROS_EXECUTOR] clear_table sequence success steps=%d duration_s=%.2f"
        % (steps_done, total_duration),
        flush=True,
    )
    return ExecutionResult(
        started=True,
        success=True,
        returncode=0,
        stdout="\n".join(combined_stdout),
        stderr="\n".join(combined_stderr),
        duration_s=total_duration,
        steps_completed=steps_done,
        progress=progress.to_public_dict() if progress else None,
    )
