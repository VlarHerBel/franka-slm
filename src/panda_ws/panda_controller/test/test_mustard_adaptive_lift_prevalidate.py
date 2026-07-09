"""Lift adaptativo mustard_bottle demo_scene_02 en prevalidación."""

import logging

from panda_controller.mustard_depth_search import (
    MUSTARD_DEMO_SCENE_02_ADAPTIVE_LIFT_M,
    mustard_adaptive_lift_prevalidate_active,
)
from panda_controller.paired_joint7_offline_sim import joint_state_from_positions
from panda_controller.perception_to_pregrasp_test import PerceptionToPregraspTest


class _AdaptiveLiftStub(PerceptionToPregraspTest):
    def __init__(self) -> None:
        self._cartesian_fraction_threshold = 0.95
        self._lift_distance_m = 0.150
        self._moveit_target_link = "panda_hand"
        self._moveit2 = self
        self._scene_id = "demo_scene_02"
        self._lift_fraction_by_m: dict[float, float] = {}

    def get_logger(self) -> logging.Logger:
        return logging.getLogger("test_mustard_adaptive_lift_prevalidate")

    def compute_ik(self, **kwargs):  # type: ignore[no-untyped-def]
        return None

    def _tcp_to_moveit_hand_pose(self, tcp, quat):  # type: ignore[no-untyped-def]
        return (float(tcp[0]), float(tcp[1]), float(tcp[2]) - 0.1034), quat

    def _joint_values_7d_from_any(self, js, context=""):  # type: ignore[no-untyped-def]
        if hasattr(js, "position"):
            return list(js.position)[:7]
        return list(js)[:7]

    def _fk_tcp_position_from_joint_state(self, js):  # type: ignore[no-untyped-def]
        return (0.455, 0.115, 0.4269)

    def _call_get_cartesian_path_from_joint_state(  # type: ignore[no-untyped-def]
        self, start_js, lift_moveit, quat, **kwargs
    ):
        lift_z = float(lift_moveit[2])
        lift_m = round(lift_z - (0.4269 - 0.1034), 3)
        fraction = float(self._lift_fraction_by_m.get(lift_m, 0.5))
        audit = {
            "fraction": fraction,
            "returned_last_point_js": [0.1, -0.5, 0.0, -2.0, 0.0, 1.5, 0.2],
        }
        return fraction, 4, audit

    def _log_full_pick_route_segment(self, segment, result, **fields):  # type: ignore[no-untyped-def]
        return None

    def _prepare_planning_scene_for_lift_prevalidation(  # type: ignore[no-untyped-def]
        self, candidate, target_collision_id, scene_obstacles, **kwargs
    ):
        return True


def test_mustard_adaptive_lift_constants() -> None:
    assert mustard_adaptive_lift_prevalidate_active("mustard_bottle", "demo_scene_02")
    assert mustard_adaptive_lift_prevalidate_active("mustard_bottle", "chips_mustard_01")
    assert not mustard_adaptive_lift_prevalidate_active("sugar_box", "demo_scene_02")
    assert MUSTARD_DEMO_SCENE_02_ADAPTIVE_LIFT_M == (
        0.060,
        0.080,
        0.100,
        0.120,
        0.150,
    )


def test_adaptive_lift_accepts_first_sufficient_fraction() -> None:
    stub = _AdaptiveLiftStub()
    stub._lift_fraction_by_m = {
        0.060: 0.70,
        0.080: 0.96,
        0.100: 1.0,
    }
    candidate = {
        "label": "mustard_bottle",
        "scene_id": "demo_scene_02",
        "prevalidated_grasp_js_from_descend": joint_state_from_positions(
            [0.1, -0.5, 0.0, -2.0, 0.0, 1.5, 0.2]
        ),
        "grasp_state_source": "cartesian_descend_endpoint",
        "lift_start_state_source": "cartesian_descend_endpoint",
    }
    ok, frac = stub._paired_validate_vertical_lift_from_pregrasp_js(
        candidate,
        (0.455, 0.115, 0.4269),
        (1.0, 0.0, 0.0, 0.0),
        joint_state_from_positions([0.0] * 7),
        candidate_idx=0,
    )
    assert ok is True
    assert frac >= 0.95
    assert candidate["selected_pick_lift_m"] == 0.080
    assert candidate["mustard_adaptive_lift_m"] == 0.080


def test_adaptive_lift_uses_descend_endpoint_not_ik() -> None:
    stub = _AdaptiveLiftStub()
    stub._lift_fraction_by_m = {0.060: 1.0}
    stub._ik_calls = 0

    def compute_ik(**kwargs):  # type: ignore[no-untyped-def]
        stub._ik_calls += 1
        return None

    stub.compute_ik = compute_ik
    candidate = {
        "label": "mustard_bottle",
        "scene_id": "demo_scene_02",
        "prevalidated_grasp_js_from_descend": joint_state_from_positions(
            [0.1, -0.5, 0.0, -2.0, 0.0, 1.5, 0.2]
        ),
        "grasp_state_source": "cartesian_descend_endpoint",
        "lift_start_state_source": "cartesian_descend_endpoint",
    }
    ok, _ = stub._paired_validate_vertical_lift_from_pregrasp_js(
        candidate,
        (0.455, 0.115, 0.4269),
        (1.0, 0.0, 0.0, 0.0),
        joint_state_from_positions([0.0] * 7),
    )
    assert ok is True
    assert stub._ik_calls == 1
