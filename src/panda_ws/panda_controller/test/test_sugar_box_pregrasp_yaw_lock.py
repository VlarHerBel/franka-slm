"""Regresión: sugar_box direct pregrasp conserva yaw validado en plan-before."""

import logging
import math

from panda_controller.perception_to_pregrasp_test import PerceptionToPregraspTest


class _YawLockStub(PerceptionToPregraspTest):
    def __init__(self) -> None:
        self._use_safe_pregrasp = False
        self._enable_safe_pregrasp_stage = False
        self._gripper_physical_yaw_correction_rad = 0.0
        self._moveit2 = object()

    def get_logger(self) -> logging.Logger:
        return logging.getLogger("test_sugar_box_pregrasp_yaw_lock")

    def _tcp_pose_to_moveit_pose(self, tcp_pose, quat):  # type: ignore[no-untyped-def]
        return (float(tcp_pose[0]), float(tcp_pose[1]), float(tcp_pose[2]) + 0.1)

    def _ranked_downward_yaw_variants_for_pose(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        raise AssertionError("no debe re-ranking yaw cuando plan-before está bloqueado")

    def _current_arm_joint_state(self):  # type: ignore[no-untyped-def]
        return object()


def _validated_candidate(*, locked_flags: bool) -> dict:
    validated_yaw = 1.6964889803845846
    quat = PerceptionToPregraspTest._downward_yaw_quaternion(validated_yaw)
    cand = {
        "label": "sugar_box",
        "_plan_before_motion_validated": {
            "ok": True,
            "mode": "direct_pregrasp",
            "variant_name": "top_down_yaw",
            "quat": list(quat),
            "final_yaw_rad": validated_yaw,
        },
        "_full_pick_route_prevalidated": True,
        "_commanded_tcp_yaw_rad": validated_yaw,
        "_execution_quaternion_override": list(quat),
    }
    if locked_flags:
        cand["_post_prelude_pregrasp_locked"] = True
        cand["_pick_route_execute_cached"] = True
        cand["_sugar_box_direct_pregrasp_cached_traj"] = object()
    return cand


def test_direct_pregrasp_yaw_lock_uses_plan_before_quat() -> None:
    stub = _YawLockStub()
    candidate = _validated_candidate(locked_flags=True)
    fallback = (0.0, 1.0, 0.0, 0.0)
    validated_quat = tuple(candidate["_plan_before_motion_validated"]["quat"])
    sel_quat, sel_yaw = stub._select_pregrasp_execution_quaternion(
        candidate,
        (0.63, -0.175, 0.472),
        (0.63, -0.175, 0.572),
        fallback,
    )
    assert sel_quat == validated_quat
    assert math.isclose(sel_yaw, 1.6964889803845846, abs_tol=1e-6)
    assert candidate.get("_direct_pregrasp_yaw_execution_locked") is True


def test_direct_pregrasp_yaw_lock_requires_validated_mode() -> None:
    stub = _YawLockStub()
    candidate = _validated_candidate(locked_flags=True)
    candidate["_plan_before_motion_validated"]["mode"] = "safe_pregrasp"
    assert stub._direct_pregrasp_yaw_execution_is_locked(candidate) is False


def test_full_pick_route_prevalidated_also_locks_yaw() -> None:
    stub = _YawLockStub()
    candidate = _validated_candidate(locked_flags=False)
    candidate.pop("_post_prelude_pregrasp_locked", None)
    candidate.pop("_pick_route_execute_cached", None)
    candidate.pop("_sugar_box_direct_pregrasp_cached_traj", None)
    assert stub._direct_pregrasp_yaw_execution_is_locked(candidate) is True
