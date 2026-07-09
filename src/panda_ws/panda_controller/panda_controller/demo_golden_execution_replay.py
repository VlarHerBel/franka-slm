"""Golden execution replay executor (v2 YAML -> motion con guards físicos)."""

from __future__ import annotations

import math
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

from panda_controller.demo_golden_execution_candidate import (
    format_golden_execution_abort_log,
    format_golden_execution_done_log,
    format_golden_execution_phase_done_log,
    format_golden_execution_phase_start_log,
)
from panda_controller.tfg_motion_waypoints import PANDA_ARM_JOINT_NAMES

TCP_XY_TOL_M = 0.015
TCP_Z_TOL_M = 0.012
MIN_DESCEND_DELTA_Z_M = 0.010
MIN_LIFT_DELTA_Z_M = 0.015
DURATION_SHORT_FAIL_THRESHOLD_S = 0.1
DURATION_EXPECTED_MIN_S = 0.5

CRITICAL_MOTION_PHASES = frozenset(
    {
        "cartesian_descend_to_grasp",
        "cartesian_lift",
        "place_approach",
        "place_release",
        "place_retreat",
    }
)

ATTACHED_GUARD_PHASES = frozenset(
    {
        "post_lift_local_escape",
        "transport_entry_to_safe_hub",
        "deterministic_transport",
    }
)


def phase_tcp_goal(phase: Dict[str, Any]) -> Optional[List[float]]:
    for key in ("goal_tcp", "tcp_goal"):
        val = phase.get(key)
        if isinstance(val, (list, tuple)) and len(val) >= 3:
            return [float(val[0]), float(val[1]), float(val[2])]
    return None


def phase_start_tcp(phase: Dict[str, Any]) -> Optional[List[float]]:
    val = phase.get("start_tcp")
    if isinstance(val, (list, tuple)) and len(val) >= 3:
        return [float(val[0]), float(val[1]), float(val[2])]
    return None


def phase_motion_fields(
    phase: Dict[str, Any],
) -> Tuple[bool, bool, bool, bool, bool]:
    has_goal_js = isinstance(phase.get("goal_js"), (list, tuple)) and bool(
        phase.get("goal_js")
    )
    has_points = bool(phase.get("points"))
    has_cartesian = phase_tcp_goal(phase) is not None
    has_waypoint = bool(phase.get("goal_waypoint") or phase.get("target_waypoint"))
    has_j7 = phase.get("joint7_target") is not None
    has_sequence = bool(phase.get("sequence"))
    executable = (
        has_goal_js
        or has_points
        or has_cartesian
        or has_waypoint
        or has_j7
        or has_sequence
    )
    return executable, has_goal_js, has_points, has_cartesian, has_waypoint


def tcp_distance_xyz(a: Sequence[float], b: Sequence[float]) -> float:
    return math.sqrt(
        sum((float(a[i]) - float(b[i])) ** 2 for i in range(min(3, len(a), len(b))))
    )


def tcp_xy_distance(a: Sequence[float], b: Sequence[float]) -> float:
    return math.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1]))


def motion_required_for_goal(
    goal_tcp: Optional[Sequence[float]],
    current_tcp: Optional[Sequence[float]],
    *,
    xy_tol: float = TCP_XY_TOL_M,
    z_tol: float = TCP_Z_TOL_M,
) -> Tuple[bool, float, bool]:
    if goal_tcp is None or current_tcp is None:
        return True, float("inf"), False
    dist = tcp_distance_xyz(goal_tcp, current_tcp)
    xy_ok = tcp_xy_distance(goal_tcp, current_tcp) <= xy_tol
    z_ok = abs(float(goal_tcp[2]) - float(current_tcp[2])) <= z_tol
    at_goal = xy_ok and z_ok
    return not at_goal, dist, at_goal


def evaluate_replay_duration(
    *,
    expected_s: float,
    real_s: float,
    motion_required_flag: bool,
    motion_executed: bool,
    at_goal: bool,
) -> Tuple[bool, str]:
    if motion_executed:
        return True, "motion_executed"
    if not motion_required_flag and at_goal:
        return True, "already_at_goal_safe_noop"
    if expected_s <= DURATION_EXPECTED_MIN_S:
        return True, "short_expected_duration"
    if real_s >= DURATION_SHORT_FAIL_THRESHOLD_S:
        return True, "real_duration_ok"
    if motion_required_flag:
        return False, "zero_duration_with_motion_required"
    return True, "already_at_goal_safe_noop"


def format_golden_replay_motion_phase_guard_log(fields: Dict[str, Any]) -> str:
    lines = [
        "[GOLDEN_REPLAY_MOTION_PHASE_GUARD]",
        "phase_idx=%s" % fields.get("phase_idx", ""),
        "phase_name=%s" % fields.get("phase_name", ""),
        "phase_type=%s" % fields.get("phase_type", ""),
    ]
    for key in (
        "current_tcp",
        "goal_tcp",
        "distance_to_goal",
        "motion_required",
        "has_goal_js",
        "has_points",
        "has_cartesian_goal",
        "execution_attempted",
        "motion_executed",
        "result",
        "reason",
    ):
        if key in fields and fields.get(key) not in (None, ""):
            lines.append("%s=%s" % (key, fields.get(key)))
    if "result" not in fields:
        lines.append("result=FAIL")
    return "\n".join(lines)


def format_golden_replay_attached_state_guard_log(fields: Dict[str, Any]) -> str:
    return (
        "[GOLDEN_REPLAY_ATTACHED_STATE_GUARD]\n"
        "phase_name=%s\n"
        "object_attached=%s\n"
        "required=%s\n"
        "cartesian_lift_ok=%s\n"
        "result=%s\n"
        "reason=%s"
        % (
            fields.get("phase_name", ""),
            fields.get("object_attached", "unknown"),
            fields.get("required", "true"),
            fields.get("cartesian_lift_ok", "unknown"),
            fields.get("result", "FAIL"),
            fields.get("reason", ""),
        )
    )


class GoldenExecutionReplayer:
    """Ejecuta fases golden v2 con verificación física (no no-ops silenciosos)."""

    def __init__(self, node: Any) -> None:
        self._n = node
        self._ctx: Dict[str, Any] = {}

    def run(self, candidate: Dict[str, Any]) -> bool:
        golden = candidate.get("_golden_execution_v2")
        if not isinstance(golden, dict):
            self._n.get_logger().error(
                "[GOLDEN_EXECUTION_ABORT]\nphase_name=init\nreason=missing_golden_v2"
            )
            return False
        if bool(self._n._dry_run):
            self._n.get_logger().info(
                "[GOLDEN_EXECUTION_REPLAY] dry_run=true; skipping motion"
            )
            return True

        self._n._reload_motion_waypoints()
        phases = golden.get("phases") or []
        quat = self._resolve_quat(candidate, golden)
        self._n._active_move_quat = quat
        self._ctx = {
            "phase_results": {},
            "quat": quat,
            "descend_ok": False,
            "lift_ok": False,
            "place_release_ok": False,
            "descend_start_z": None,
            "lift_start_z": None,
        }

        t0 = time.monotonic()
        total_expected = sum(float(p.get("duration_s", 0.0)) for p in phases)
        label = str(candidate.get("label") or self._n._target_label or "")

        target_collision_id = self._setup_collision(candidate)
        self._n._current_target_collision_id = target_collision_id

        for phase_idx, phase in enumerate(phases):
            if not isinstance(phase, dict):
                continue
            ok, reason = self._execute_phase(phase_idx, phase, candidate, golden)
            if not ok:
                self._n.get_logger().error(
                    format_golden_execution_abort_log(
                        {
                            "phase_name": str(phase.get("name", "")),
                            "reason": reason or "phase_fail",
                            "object_attached": str(
                                bool(self._n._current_target_attached)
                            ).lower(),
                            "safe_state_action": "hold_position",
                        }
                    )
                )
                return False

        total_real = time.monotonic() - t0
        self._n.get_logger().info(
            format_golden_execution_done_log(
                {
                    "scene_id": str(self._n._scene_id or ""),
                    "target_label": label,
                    "slot_index": int(self._n._place_slot_index),
                    "total_expected_time_s": total_expected,
                    "total_real_time_s": total_real,
                    "result": "OK",
                }
            )
        )
        self._n._already_returned_home_after_place = True
        self._n._home_return_ok_after_place = True
        return True

    def _setup_collision(self, candidate: Dict[str, Any]) -> Optional[str]:
        scene_obstacles = candidate.get("scene_obstacles") or []
        if not (
            self._n._add_detected_objects_to_scene and scene_obstacles
        ):
            return None
        return self._n._add_detected_objects_to_planning_scene(
            scene_obstacles,
            include_target=self._n._include_target_collision(candidate),
            candidate=candidate,
        )

    def _resolve_quat(
        self, candidate: Dict[str, Any], golden: Dict[str, Any]
    ) -> Tuple[float, float, float, float]:
        yaw = None
        for src in (
            candidate.get("_final_tcp_yaw_rad"),
            (golden.get("candidate") or {}).get("commanded_tcp_yaw_rad"),
            (golden.get("object_pose") or {}).get("yaw_rad"),
        ):
            try:
                if src is not None:
                    yaw = float(src)
                    break
            except (TypeError, ValueError):
                continue
        if yaw is None:
            yaw = 0.0
        return self._n._downward_yaw_quaternion(float(yaw))

    def _current_tcp(
        self, quat: Tuple[float, float, float, float]
    ) -> Optional[List[float]]:
        _hand, tcp = self._n._read_current_hand_and_tcp(quat)
        if tcp is None:
            pos = self._n._tcp_position_in_planning_frame()
            if pos is not None:
                return [float(pos[0]), float(pos[1]), float(pos[2])]
            return None
        return [float(tcp[0]), float(tcp[1]), float(tcp[2])]

    def _log_motion_guard(
        self,
        *,
        phase_idx: int,
        phase: Dict[str, Any],
        current_tcp: Optional[List[float]],
        goal_tcp: Optional[List[float]],
        distance: float,
        motion_required_flag: bool,
        has_goal_js: bool,
        has_points: bool,
        has_cartesian: bool,
        execution_attempted: bool,
        motion_executed: bool,
        result: str,
        reason: str,
        when: str,
    ) -> None:
        self._n.get_logger().info(
            format_golden_replay_motion_phase_guard_log(
                {
                    "phase_idx": phase_idx,
                    "phase_name": phase.get("name", ""),
                    "phase_type": phase.get("type", ""),
                    "when": when,
                    "current_tcp": current_tcp,
                    "goal_tcp": goal_tcp,
                    "distance_to_goal": "%.4f" % distance if math.isfinite(distance) else "n/a",
                    "motion_required": str(motion_required_flag).lower(),
                    "has_goal_js": str(has_goal_js).lower(),
                    "has_points": str(has_points).lower(),
                    "has_cartesian_goal": str(has_cartesian).lower(),
                    "execution_attempted": str(execution_attempted).lower(),
                    "motion_executed": str(motion_executed).lower(),
                    "result": result,
                    "reason": reason,
                }
            )
        )

    def _log_phase_done(
        self,
        *,
        phase_idx: int,
        phase_name: str,
        real_dur: float,
        expected_dur: float,
        ok: bool,
        reason: str,
        motion_executed: bool,
    ) -> None:
        self._n.get_logger().info(
            format_golden_execution_phase_done_log(
                {
                    "phase_idx": phase_idx,
                    "phase_name": phase_name,
                    "duration_real_s": real_dur,
                    "expected_duration_s": expected_dur,
                    "result": "OK" if ok else "FAIL",
                    "reason": reason,
                    "motion_executed": str(motion_executed).lower(),
                    "object_attached": str(
                        bool(self._n._current_target_attached)
                    ).lower(),
                }
            )
        )

    def _attached_guard(self, phase_name: str) -> Tuple[bool, str]:
        attached = bool(self._n._current_target_attached)
        lift_ok = bool(self._ctx.get("lift_ok"))
        ok = attached and lift_ok
        reason = "OK" if ok else "attached_or_lift_not_ok"
        self._n.get_logger().info(
            format_golden_replay_attached_state_guard_log(
                {
                    "phase_name": phase_name,
                    "object_attached": str(attached).lower(),
                    "required": "true",
                    "cartesian_lift_ok": str(lift_ok).lower(),
                    "result": "OK" if ok else "FAIL",
                    "reason": reason,
                }
            )
        )
        return ok, reason

    def _execute_phase(
        self,
        phase_idx: int,
        phase: Dict[str, Any],
        candidate: Dict[str, Any],
        golden: Dict[str, Any],
    ) -> Tuple[bool, str]:
        phase_name = str(phase.get("name", "")).strip().lower()
        phase_type = str(phase.get("type", "")).strip()
        expected_dur = float(phase.get("duration_s", 0.0))
        phase_t0 = time.monotonic()
        quat = self._ctx["quat"]
        current_tcp = self._current_tcp(quat)
        goal_tcp = phase_tcp_goal(phase)
        executable, has_goal_js, has_points, has_cartesian, _has_wp = phase_motion_fields(
            phase
        )
        motion_req, dist, at_goal = motion_required_for_goal(goal_tcp, current_tcp)

        self._n.get_logger().info(
            format_golden_execution_phase_start_log(
                {
                    "phase_idx": phase_idx,
                    "phase_name": phase_name,
                    "type": phase_type,
                }
            )
        )
        self._log_motion_guard(
            phase_idx=phase_idx,
            phase=phase,
            current_tcp=current_tcp,
            goal_tcp=goal_tcp,
            distance=dist,
            motion_required_flag=motion_req,
            has_goal_js=has_goal_js,
            has_points=has_points,
            has_cartesian=has_cartesian,
            execution_attempted=False,
            motion_executed=False,
            result="PENDING",
            reason="before_execute",
            when="before",
        )

        if phase_name in ATTACHED_GUARD_PHASES:
            guard_ok, guard_reason = self._attached_guard(phase_name)
            if not guard_ok:
                self._finish_phase(
                    phase_idx,
                    phase_name,
                    phase_t0,
                    expected_dur,
                    False,
                    guard_reason,
                    False,
                    phase,
                )
                return False, guard_reason

        ok = True
        reason = "OK"
        motion_executed = False
        execution_attempted = False

        if phase_name == "close_gripper":
            if not self._ctx.get("descend_ok"):
                reason = "cartesian_descend_not_physical_ok"
                ok = False
            else:
                execution_attempted = True
                ok = self._n._close_gripper_on_grasp_if_needed(dry_run=False)
                reason = "close_gripper_fail" if not ok else "OK"
                motion_executed = ok
        elif phase_name == "attach_and_verify":
            execution_attempted = True
            ok, reason = self._replay_attach(candidate)
            motion_executed = ok
        elif phase_name == "cartesian_descend_to_grasp":
            execution_attempted = True
            ok, reason, motion_executed = self._replay_descend(phase, quat)
            self._ctx["descend_ok"] = bool(ok)
            self._ctx["phase_results"][phase_name] = "OK" if ok else "FAIL"
        elif phase_name == "cartesian_lift":
            execution_attempted = True
            ok, reason, motion_executed = self._replay_lift(phase, quat, candidate)
            self._ctx["lift_ok"] = bool(ok)
            self._ctx["phase_results"][phase_name] = "OK" if ok else "FAIL"
        elif phase_name == "place_release":
            execution_attempted = True
            ok, reason, motion_executed = self._replay_place_cartesian(
                phase, quat, stage="golden_place_release"
            )
            if ok:
                rel_z = phase.get("release_tcp_z")
                cur = self._current_tcp(quat)
                if rel_z is not None and cur is not None:
                    if abs(float(cur[2]) - float(rel_z)) > TCP_Z_TOL_M:
                        ok = False
                        reason = "place_release_z_not_at_goal"
                        motion_executed = False
                if ok:
                    self._ctx["place_release_ok"] = True
                    if rel_z is not None:
                        self._ctx["place_release_z"] = float(rel_z)
        elif phase_name == "open_detach":
            ok, reason = self._replay_open_detach(phase, quat, candidate)
            execution_attempted = True
            motion_executed = ok
        elif phase_name in CRITICAL_MOTION_PHASES:
            execution_attempted = True
            ok, reason, motion_executed = self._replay_place_cartesian(
                phase,
                quat,
                stage="golden_%s" % phase_name,
            )
        elif phase_type == "attach":
            execution_attempted = True
            ok, reason = self._replay_attach(candidate)
            motion_executed = ok
        elif phase_type == "gripper_and_detach":
            execution_attempted = True
            ok, reason = self._replay_open_detach(phase, quat, candidate)
            motion_executed = ok
        elif phase_type in ("joint_trajectory", "direct_action_or_joint_trajectory"):
            execution_attempted = True
            ok = self._n._execute_golden_phase_joint_trajectory(
                phase, waypoints_data=self._n._motion_waypoints_data
            )
            motion_executed = ok
            reason = "joint_trajectory_fail" if not ok else "OK"
        elif phase_type == "waypoint_sequence":
            execution_attempted = True
            seq = [str(x) for x in (phase.get("sequence") or [])]
            ok = self._n._execute_joint_trajectory_direct(
                seq, self._n._motion_waypoints_data
            )
            motion_executed = ok
            reason = "waypoint_sequence_fail" if not ok else "OK"
        elif phase_type == "gripper_command":
            execution_attempted = True
            if self._n._golden_phase_is_pregrasp_open(phase):
                ok = self._n._open_gripper_at_pregrasp_if_needed(dry_run=False)
                reason = "open_gripper_at_pregrasp_fail" if not ok else "OK"
            else:
                ok = self._n._close_gripper_on_grasp_if_needed(dry_run=False)
                reason = "close_gripper_fail" if not ok else "OK"
            motion_executed = ok
        elif phase_type == "joint_adjustment":
            execution_attempted = True
            ok, reason, motion_executed = self._replay_joint7_adjustment(phase)
        elif phase_type in ("cartesian_or_joint_trajectory", "trajectory"):
            execution_attempted = True
            ok, reason, motion_executed = self._replay_cartesian_or_joint(phase, quat)
        else:
            if phase_name in CRITICAL_MOTION_PHASES or (
                phase_name in ("post_lift_local_escape", "transport_entry_to_safe_hub")
            ):
                ok = False
                reason = "unsupported_critical_phase_type"
            else:
                ok = True
                reason = "noop_phase_type"

        if ok and phase_name in CRITICAL_MOTION_PHASES:
            dur_ok, dur_reason = evaluate_replay_duration(
                expected_s=expected_dur,
                real_s=time.monotonic() - phase_t0,
                motion_required_flag=motion_req,
                motion_executed=motion_executed,
                at_goal=at_goal,
            )
            if not dur_ok:
                ok = False
                reason = dur_reason

        self._log_motion_guard(
            phase_idx=phase_idx,
            phase=phase,
            current_tcp=self._current_tcp(quat),
            goal_tcp=goal_tcp,
            distance=dist,
            motion_required_flag=motion_req,
            has_goal_js=has_goal_js,
            has_points=has_points,
            has_cartesian=has_cartesian,
            execution_attempted=execution_attempted,
            motion_executed=motion_executed,
            result="OK" if ok else "FAIL",
            reason=reason,
            when="after",
        )
        self._finish_phase(
            phase_idx,
            phase_name,
            phase_t0,
            expected_dur,
            ok,
            reason,
            motion_executed,
            phase,
        )
        self._ctx["phase_results"][phase_name] = "OK" if ok else "FAIL"
        return ok, reason

    def _finish_phase(
        self,
        phase_idx: int,
        phase_name: str,
        phase_t0: float,
        expected_dur: float,
        ok: bool,
        reason: str,
        motion_executed: bool,
        phase: Dict[str, Any],
    ) -> None:
        real_dur = time.monotonic() - phase_t0
        self._log_phase_done(
            phase_idx=phase_idx,
            phase_name=phase_name,
            real_dur=real_dur,
            expected_dur=expected_dur,
            ok=ok,
            reason=reason,
            motion_executed=motion_executed,
        )
        if self._n._golden_execution_recorder is not None:
            self._n._golden_execution_record_phase(
                phase_name,
                str(phase.get("type", "")),
                duration_s=real_dur,
                payload=dict(phase),
            )

    def _replay_joint7_adjustment(
        self, phase: Dict[str, Any]
    ) -> Tuple[bool, str, bool]:
        j7_target = phase.get("joint7_target")
        if j7_target is None:
            return True, "noop_no_joint7_target", False
        js = self._n._current_arm_joint_positions_list()
        if js is None or len(js) < 7:
            return False, "missing_arm_joints", False
        positions = [float(v) for v in js]
        j_idx = PANDA_ARM_JOINT_NAMES.index("panda_joint7")
        positions[j_idx] = float(j7_target)
        from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

        traj = JointTrajectory()
        traj.joint_names = list(PANDA_ARM_JOINT_NAMES)
        pt = JointTrajectoryPoint()
        pt.positions = positions
        pt.time_from_start = self._n._duration_from_seconds(
            float(phase.get("duration_s", 1.5))
        )
        traj.points = [pt]
        ok, _, _, _ = self._n._send_arm_follow_joint_trajectory(traj)
        return bool(ok), "joint7_adjust_fail" if not ok else "OK", bool(ok)

    def _replay_cartesian_or_joint(
        self,
        phase: Dict[str, Any],
        quat: Tuple[float, float, float, float],
    ) -> Tuple[bool, str, bool]:
        if isinstance(phase.get("goal_js"), (list, tuple)) and phase.get("goal_js"):
            ok = self._n._execute_golden_phase_joint_trajectory(
                phase, waypoints_data=self._n._motion_waypoints_data
            )
            return bool(ok), "joint_goal_fail" if not ok else "OK", bool(ok)
        if phase.get("points"):
            ok = self._n._execute_golden_phase_joint_trajectory(
                phase, waypoints_data=self._n._motion_waypoints_data
            )
            return bool(ok), "joint_points_fail" if not ok else "OK", bool(ok)
        wp = str(phase.get("target_waypoint") or phase.get("goal_waypoint") or "")
        if wp:
            ok = self._n._execute_golden_phase_joint_trajectory(
                phase, waypoints_data=self._n._motion_waypoints_data
            )
            return bool(ok), "waypoint_fail" if not ok else "OK", bool(ok)
        goal = phase_tcp_goal(phase)
        if goal is not None:
            return self._replay_place_cartesian(
                phase, quat, stage="golden_%s" % phase.get("name", "motion")
            )
        return False, "missing_executable_motion", False

    def _replay_descend(
        self,
        phase: Dict[str, Any],
        quat: Tuple[float, float, float, float],
    ) -> Tuple[bool, str, bool]:
        quat = self._ctx["quat"]
        start_tcp = phase_start_tcp(phase) or self._current_tcp(quat)
        if start_tcp is not None:
            self._ctx["descend_start_z"] = float(start_tcp[2])
        if phase_tcp_goal(phase) is not None:
            ok, reason, motion_executed = self._replay_place_cartesian(
                phase, quat, stage="golden_cartesian_descend"
            )
        else:
            ok, reason, motion_executed = self._replay_cartesian_or_joint(phase, quat)
        if not ok:
            if reason in (
                "missing_executable_motion",
                "place_missing_executable_motion",
            ):
                return False, "cartesian_descend_missing_executable_motion", False
            return False, reason, motion_executed
        goal = phase_tcp_goal(phase)
        cur = self._current_tcp(quat)
        if goal is None or cur is None:
            return False, "cartesian_descend_verify_no_tcp", motion_executed
        if tcp_xy_distance(goal, cur) > TCP_XY_TOL_M:
            return False, "cartesian_descend_xy_not_at_goal", motion_executed
        if abs(float(goal[2]) - float(cur[2])) > TCP_Z_TOL_M:
            return False, "cartesian_descend_z_not_at_goal", motion_executed
        start_z = self._ctx.get("descend_start_z")
        if start_z is not None:
            delta = float(start_z) - float(cur[2])
            if delta + 1e-6 < MIN_DESCEND_DELTA_Z_M:
                return False, "cartesian_descend_insufficient_delta_z", motion_executed
        return True, "OK", motion_executed

    def _replay_lift(
        self,
        phase: Dict[str, Any],
        quat: Tuple[float, float, float, float],
        candidate: Dict[str, Any],
    ) -> Tuple[bool, str, bool]:
        if not self._ctx.get("descend_ok"):
            return False, "lift_before_descend_ok", False
        cur_before = self._current_tcp(quat)
        if cur_before is not None:
            self._ctx["lift_start_z"] = float(cur_before[2])
        if phase_tcp_goal(phase) is not None:
            ok, reason, motion_executed = self._replay_place_cartesian(
                phase, quat, stage="golden_cartesian_lift"
            )
        else:
            ok, reason, motion_executed = self._replay_cartesian_or_joint(phase, quat)
        if not ok:
            if reason in (
                "missing_executable_motion",
                "place_missing_executable_motion",
            ):
                return False, "cartesian_lift_missing_executable_motion", False
            return False, reason, motion_executed
        cur = self._current_tcp(quat)
        goal = phase_tcp_goal(phase)
        if goal is not None and cur is not None:
            if abs(float(goal[2]) - float(cur[2])) > TCP_Z_TOL_M:
                return False, "cartesian_lift_z_not_at_goal", motion_executed
        start_z = self._ctx.get("lift_start_z")
        if start_z is not None and cur is not None:
            if float(cur[2]) - float(start_z) + 1e-6 < MIN_LIFT_DELTA_Z_M:
                return False, "cartesian_lift_insufficient_delta_z", motion_executed
        if not bool(self._n._current_target_attached):
            return False, "lift_without_attached_object", motion_executed
        lift_lock = self._n._lock_lift_orientation_from_current_tf(quat)
        if lift_lock is not None:
            attached_id = (
                self._n._current_attached_object_id
                or self._n._current_target_collision_id
            )
            if not self._n._post_lift_verify(lift_lock, quat, attached_id, candidate):
                return False, "gazebo_object_lift_verify_fail", motion_executed
        return True, "OK", motion_executed

    def _replay_place_cartesian(
        self,
        phase: Dict[str, Any],
        quat: Tuple[float, float, float, float],
        *,
        stage: str,
    ) -> Tuple[bool, str, bool]:
        goal = phase_tcp_goal(phase)
        if goal is None:
            if isinstance(phase.get("goal_js"), (list, tuple)) and phase.get("goal_js"):
                ok = self._n._execute_golden_phase_joint_trajectory(
                    phase, waypoints_data=self._n._motion_waypoints_data
                )
                return bool(ok), "place_joint_fail" if not ok else "OK", bool(ok)
            return False, "place_missing_executable_motion", False
        cur = self._current_tcp(quat)
        motion_req, _dist, at_goal = motion_required_for_goal(goal, cur)
        if not motion_req:
            return True, "already_at_goal_safe_noop", False
        moveit_pose = self._n._tcp_pose_to_moveit_pose(
            (float(goal[0]), float(goal[1]), float(goal[2])), quat
        )
        vel = float(getattr(self._n, "_place_approach_vel", 0.05))
        acc = float(getattr(self._n, "_place_approach_acc", 0.05))
        if "release" in stage:
            vel = float(getattr(self._n, "_place_release_vel", 0.03))
            acc = float(getattr(self._n, "_place_release_acc", 0.03))
        elif "retreat" in stage:
            vel = float(getattr(self._n, "_place_retreat_vel", 0.04))
            acc = float(getattr(self._n, "_place_retreat_acc", 0.04))
        elif "lift" in stage:
            vel = float(getattr(self._n, "_lift_vel", 0.05))
            acc = float(getattr(self._n, "_lift_acc", 0.05))
        elif "descend" in stage:
            vel = float(getattr(self._n, "_pregrasp_vel", 0.05))
            acc = float(getattr(self._n, "_pregrasp_acc", 0.05))
        ok = self._n._cartesian_place_move_checked(
            stage,
            moveit_pose,
            quat,
            vel,
            acc,
            tcp_pose=(float(goal[0]), float(goal[1]), float(goal[2])),
        )
        if not ok:
            return False, "%s_cartesian_fail" % stage, True
        cur_after = self._current_tcp(quat)
        if cur_after is not None:
            _mr, _d, at_goal_after = motion_required_for_goal(goal, cur_after)
            if _mr:
                return False, "%s_not_at_goal_after_motion" % stage, True
        return True, "OK", True

    def _replay_attach(self, candidate: Dict[str, Any]) -> Tuple[bool, str]:
        if not self._ctx.get("descend_ok"):
            return False, "attach_before_descend_ok"
        target_id = self._n._current_target_collision_id
        ok = self._n._attach_target_object_after_grasp(target_id, candidate)
        if not ok:
            return False, "attach_fail"
        if not bool(self._n._last_attach_executed):
            return False, "attach_not_executed"
        if not bool(self._n._last_planning_scene_attach_ok):
            return False, "planning_scene_attach_not_ok"
        if not bool(self._n._current_target_attached):
            return False, "attached_flag_false_after_attach"
        entity = self._n._resolve_target_entity_name(candidate)
        gz_required = self._n._gazebo_physical_attach_required(entity, candidate)
        if gz_required and not bool(self._n._last_gazebo_physical_attach_ok):
            return False, "gazebo_physical_attach_not_ok"
        if not self._n._post_attach_verify(candidate):
            return False, "attach_verify_not_physical_ok"
        return True, "OK"

    def _replay_open_detach(
        self,
        phase: Dict[str, Any],
        quat: Tuple[float, float, float, float],
        candidate: Dict[str, Any],
    ) -> Tuple[bool, str]:
        if not bool(self._n._current_target_attached):
            return False, "open_detach_object_not_attached"
        if not self._ctx.get("place_release_ok"):
            return False, "open_detach_before_place_release_ok"
        rel_z = phase.get("release_tcp_z") or self._ctx.get("place_release_z")
        goal = phase_tcp_goal(phase)
        cur = self._current_tcp(quat)
        if rel_z is not None and cur is not None:
            if abs(float(cur[2]) - float(rel_z)) > TCP_Z_TOL_M * 2.0:
                return False, "open_detach_not_at_release_pose"
        elif goal is not None and cur is not None:
            _mr, _d, at_goal = motion_required_for_goal(goal, cur)
            if _mr:
                return False, "open_detach_not_at_release_pose"
        ok_open = self._n._open_gripper_at_place_if_needed(dry_run=False)
        if not ok_open:
            return False, "open_detach_fail"
        ok_detach = self._n._detach_target_object_after_place()
        if not ok_detach:
            return False, "open_detach_fail"
        return True, "OK"
