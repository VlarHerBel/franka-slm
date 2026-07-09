"""Regresión: apertura en pregrasp (no waypoint) y yaw_free descend lock."""

import logging

import numpy as np

from panda_controller.perception_to_pregrasp_test import PerceptionToPregraspTest


class _OpenPolicyStub(PerceptionToPregraspTest):
    def __init__(self) -> None:
        self._chips_can_open_gripper_before_pregrasp_motion = False
        self._chips_can_open_gripper_at_direct_pregrasp_before_descend = True
        self._enable_gripper = True
        self._gripper_sent: list[tuple[float, str]] = []

    def get_logger(self) -> logging.Logger:
        return logging.getLogger("test_chips_can_open_policy")

    def _is_chips_can_cylinder_topdown_candidate(self, candidate):  # type: ignore[override]
        return True

    def _raw_effective_open_joint(self) -> float:
        return 0.04

    def _cap_open_joint_value(self, value, _ctx):  # type: ignore[no-untyped-def]
        return float(value)

    def _gripper_fingers_near_width(self, _target):  # type: ignore[no-untyped-def]
        return False

    def _effective_verify_open(self) -> bool:
        return True

    def _send_gripper_goal(self, open_joint, label, verify=True):  # type: ignore[no-untyped-def]
        self._gripper_sent.append((float(open_joint), str(label)))
        return True

    def _open_gripper_at_pregrasp_if_needed(self, dry_run=False):  # type: ignore[no-untyped-def]
        raise AssertionError("chips_can no debe usar open_gripper_at_pregrasp genérico")


def test_open_at_direct_pregrasp_not_waypoint() -> None:
    stub = _OpenPolicyStub()
    candidate = {"label": "chips_can", "grasp_strategy": "cylinder_topdown"}
    assert stub._open_chips_can_gripper_before_pregrasp_motion(candidate) is True
    assert stub._gripper_sent == []
    ok = stub._open_chips_can_gripper_at_direct_pregrasp_before_descend(candidate)
    assert ok is True
    assert stub._gripper_sent == [(0.04, "chips_can_apertura_en_pregrasp")]
    assert candidate.get("_chips_can_gripper_opened_at_pregrasp") is True


class _YawFreeDescendStub(PerceptionToPregraspTest):
    def __init__(self) -> None:
        self._chips_can_skip_gripper_gap_alignment_when_gt = True
        self._moveit_target_link = "panda_hand"
        self._use_grasp_tcp = False
        self._chips_can_gt_cylinder_centering_max_error_xy_m = 0.006

    def get_logger(self) -> logging.Logger:
        return logging.getLogger("test_chips_can_yaw_free_descend")

    def _chips_can_skip_gripper_gap_alignment(self, candidate):  # type: ignore[no-untyped-def]
        return True

    def _is_chips_can_gt_cylinder_candidate(self, candidate):  # type: ignore[no-untyped-def]
        return True

    def _link_pose_in_planning_frame(self, _link):  # type: ignore[no-untyped-def]
        q = PerceptionToPregraspTest._downward_yaw_quaternion(0.0)
        return {
            "position": (0.55, 0.19, 0.64),
            "quat": q,
        }

    def _compute_chips_can_gt_cylinder_centering_metrics(self, candidate):  # type: ignore[no-untyped-def]
        return {
            "target_center": np.array([0.55, 0.19]),
            "centering_target_source": "runtime_gt_cylinder_axis",
            "actual_finger_midpoint_xy": np.array([0.55, 0.19]),
            "error_xy_vec": np.zeros(2),
            "error_xy_m": 0.001,
            "cylinder_dims": "radius=0.0375 height=0.2500",
        }

    def _compute_gripper_centering_metrics(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        return None

    def _chips_can_gt_cylinder_centering_within_thresholds(self, metrics):  # type: ignore[no-untyped-def]
        return float(metrics["error_xy_m"]) <= 0.006

    def _gripper_fingers_near_width(self, _target):  # type: ignore[no-untyped-def]
        return True

    def _cap_open_joint_value(self, value, _ctx):  # type: ignore[no-untyped-def]
        return float(value)

    def _raw_effective_open_joint(self) -> float:
        return 0.04

    def _quaternion_angle_deg(self, a, b):  # type: ignore[no-untyped-def]
        return 0.0


def test_yaw_free_descend_orientation_lock_skips_gap_axis() -> None:
    stub = _YawFreeDescendStub()
    candidate = {
        "label": "chips_can",
        "grasp_strategy": "cylinder_topdown",
        "grasp_center_source": "runtime_gt_cylinder_center",
        "top_face_source": "runtime_gt_known_cylinder",
    }
    q = PerceptionToPregraspTest._downward_yaw_quaternion(0.5)
    locked = stub._lock_descend_orientation_from_current_tf(candidate, q)
    assert locked is not None
    assert "locked_hand_quat" in locked
