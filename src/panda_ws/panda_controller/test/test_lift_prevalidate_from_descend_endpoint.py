"""Lift prevalidate debe usar endpoint JS del descenso, no IK directa del grasp."""

import logging

from panda_controller.paired_joint7_offline_sim import joint_state_from_positions
from panda_controller.perception_to_pregrasp_test import PerceptionToPregraspTest


class _LiftPrevalidateStub(PerceptionToPregraspTest):
    def __init__(self) -> None:
        self._cartesian_fraction_threshold = 0.95
        self._moveit_target_link = "panda_hand"
        self._moveit2 = self
        self._ik_calls = 0
        self._lift_fraction = 1.0
        self._lift_pts = 4

    def get_logger(self) -> logging.Logger:
        return logging.getLogger("test_lift_prevalidate_from_descend_endpoint")

    def compute_ik(self, **kwargs):  # type: ignore[no-untyped-def]
        self._ik_calls += 1
        return None

    def _tcp_to_moveit_hand_pose(self, tcp, quat):  # type: ignore[no-untyped-def]
        return (float(tcp[0]), float(tcp[1]), float(tcp[2]) - 0.1034), quat

    def _joint_values_7d_from_any(self, js, context=""):  # type: ignore[no-untyped-def]
        if hasattr(js, "position"):
            return list(js.position)[:7]
        return list(js)[:7]

    def _fk_tcp_position_from_joint_state(self, js):  # type: ignore[no-untyped-def]
        return (0.455, 0.115, 0.432)

    def _plan_cartesian_fraction_from_joint_state(  # type: ignore[no-untyped-def]
        self, start_js, lift_moveit, quat, **kwargs
    ):
        return self._lift_fraction, self._lift_pts

    def _call_get_cartesian_path_from_joint_state(  # type: ignore[no-untyped-def]
        self, start_js, lift_moveit, quat, **kwargs
    ):
        audit = {
            "fraction": self._lift_fraction,
            "returned_last_point_js": [0.1, -0.5, 0.0, -2.0, 0.0, 1.5, 0.2],
        }
        return self._lift_fraction, self._lift_pts, audit

    def _log_full_pick_route_segment(self, segment, result, **fields):  # type: ignore[no-untyped-def]
        return None

    def _prepare_planning_scene_for_lift_prevalidation(  # type: ignore[no-untyped-def]
        self, candidate, target_collision_id, scene_obstacles, **kwargs
    ):
        return True

    def _restore_planning_scene_after_lift_prevalidation(  # type: ignore[no-untyped-def]
        self, candidate, scene_obstacles=None
    ):
        return None


def test_lift_uses_prevalidated_descend_endpoint_without_grasp_ik() -> None:
    stub = _LiftPrevalidateStub()
    candidate = {
        "label": "cracker_box",
        "prevalidated_grasp_js_from_descend": joint_state_from_positions(
            [0.1, -0.5, 0.0, -2.0, 0.0, 1.5, 0.2]
        ),
        "grasp_state_source": "staged_descend_endpoint",
        "lift_start_state_source": "staged_descend_endpoint",
    }
    pregrasp_js = joint_state_from_positions([0.0] * 7)
    ok, frac = stub._paired_validate_vertical_lift_from_pregrasp_js(
        candidate,
        (0.455, 0.115, 0.432),
        (1.0, 0.0, 0.0, 0.0),
        pregrasp_js,
        candidate_idx=35,
    )
    assert ok is True
    assert frac == 1.0
    assert stub._ik_calls == 1
    assert candidate.get("lift_prevalidated") is True
    assert (
        candidate.get("lift_scene_policy") == "target_removed_for_lift_prevalidation"
    )


def test_lift_scene_prepare_called_for_descend_endpoint() -> None:
    stub = _LiftPrevalidateStub()
    stub._scene_prepare_calls = 0
    stub._scene_restore_calls = 0

    def _prepare(*args, **kwargs):  # type: ignore[no-untyped-def]
        stub._scene_prepare_calls += 1
        return True

    def _restore(*args, **kwargs):  # type: ignore[no-untyped-def]
        stub._scene_restore_calls += 1

    stub._prepare_planning_scene_for_lift_prevalidation = _prepare
    stub._restore_planning_scene_after_lift_prevalidation = _restore
    candidate = {
        "label": "cracker_box",
        "prevalidated_grasp_js_from_descend": joint_state_from_positions(
            [0.1, -0.5, 0.0, -2.0, 0.0, 1.5, 0.2]
        ),
        "grasp_state_source": "collision_off_descend_endpoint",
        "lift_start_state_source": "collision_off_descend_endpoint",
        "scene_obstacles": [{"label": "chips_can", "position": [0.5, 0.0, 0.45]}],
    }
    ok, _ = stub._paired_validate_vertical_lift_from_pregrasp_js(
        candidate,
        (0.455, 0.115, 0.432),
        (1.0, 0.0, 0.0, 0.0),
        joint_state_from_positions([0.0] * 7),
        candidate_idx=35,
        scene_obstacles=candidate["scene_obstacles"],
        target_collision_id="target_cracker_box",
    )
    assert ok is True
    assert stub._scene_prepare_calls == 1
    assert stub._scene_restore_calls == 0


def test_lift_sugar_geometric_fallback_uses_pregrasp_proxy_without_grasp_ik() -> None:
    stub = _LiftPrevalidateStub()
    pregrasp_js = joint_state_from_positions([0.1, -0.5, 0.0, -2.0, 0.0, 1.5, 0.2])
    candidate = {
        "label": "sugar_box",
        "_cartesian_descend_prevalidation_source": "geometric_fallback",
        "_simple_direct_pick_route": True,
        "scene_obstacles": [],
    }
    ok, frac = stub._paired_validate_vertical_lift_from_pregrasp_js(
        candidate,
        (0.630, -0.175, 0.407),
        (1.0, 0.0, 0.0, 0.0),
        pregrasp_js,
        candidate_idx=0,
    )
    assert ok is True
    assert frac == 1.0
    assert stub._ik_calls >= 1
    assert candidate.get("lift_prevalidated") is True


def test_lift_sugar_geometric_proxy_with_distant_mustard_obstacle() -> None:
    stub = _LiftPrevalidateStub()
    pregrasp_js = joint_state_from_positions([0.1, -0.5, 0.0, -2.0, 0.0, 1.5, 0.2])
    candidate = {
        "label": "sugar_box",
        "_cartesian_descend_prevalidation_source": "geometric_fallback",
        "_simple_direct_pick_route": True,
        "pregrasp_tcp": [0.630, -0.175, 0.472],
        "scene_obstacles": [
            {
                "label": "mustard_bottle",
                "position": (0.660, 0.060, 0.368),
                "collision_dims": {
                    "shape": "box",
                    "box": [0.1003, 0.0627, 0.1959],
                },
            }
        ],
    }
    ok, frac = stub._paired_validate_vertical_lift_from_pregrasp_js(
        candidate,
        (0.630, -0.175, 0.407),
        (1.0, 0.0, 0.0, 0.0),
        pregrasp_js,
        candidate_idx=0,
    )
    assert ok is True
    assert frac == 1.0
    assert candidate.get("lift_prevalidated") is True
