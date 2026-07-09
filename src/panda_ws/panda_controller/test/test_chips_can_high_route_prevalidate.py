"""Tests prevalidación ruta chips_can object_high."""

from __future__ import annotations

import logging
from types import SimpleNamespace
from typing import Any, Dict, Optional, Tuple

from panda_controller.chips_can_high_route_prevalidate import (
    CHIPS_CAN_HIGH_ROUTE_PENDING_PREFLIGHT_SOURCE,
    CHIPS_CAN_HIGH_ROUTE_PREFLIGHT_SOURCE,
    CHIPS_CAN_HIGH_TO_LOW_ACTUAL_SOURCE,
    OK_CHIPS_HIGH_ROUTE_PENDING_FINAL_DESCEND_VALIDATE,
    OK_CHIPS_HIGH_ROUTE_PREVALIDATED,
    chips_can_high_route_pending_preflight_accepts,
    chips_can_high_route_preflight_accepts,
    chips_can_high_route_yaw_passes,
    evaluate_chips_can_high_to_low_pregrasp_verify,
    select_chips_can_high_route_pending_descend_variant,
    select_chips_can_high_route_yaw_variant,
    summarize_chips_can_high_route_yaw_variants,
)
from panda_controller.demo_pick_route_preflight import (
    OK_FULL_PICK_ROUTE_RESULTS,
    pick_route_preflight_allows_motion,
)
from panda_controller.perception_to_pregrasp_test import PerceptionToPregraspTest
from panda_controller.tfg_motion_waypoints import PANDA_ARM_JOINT_NAMES


class _PreflightStub(PerceptionToPregraspTest):
    def __init__(self) -> None:
        self._chips_can_force_high_pregrasp_stage = True
        self._plan_before_prelude = True
        self._dry_run = False
        self._moveit2 = object()
        self._demo_authoritative_scene = True
        self._scene_id = "demo_scene_02"
        self._cartesian_fraction_threshold = 0.95

    def get_logger(self) -> logging.Logger:
        return logging.getLogger("test_chips_can_high_route_prevalidate")

    def _grasp_ik_required_before_pregrasp_from_home(self) -> bool:
        return False

    def _demo_full_pick_route_prevalidation_required(self, candidate: dict) -> bool:
        return True

    def _paired_pregrasp_validation_required(self, candidate: dict) -> bool:
        return True


def _chips_high_route_candidate() -> dict:
    return {
        "label": "chips_can",
        "grasp_strategy": "cylinder_topdown",
        "uses_low_object_high_approach_stage": True,
        "_chips_can_high_route_prevalidated": True,
        "_cartesian_descend_prevalidated": True,
        "_plan_before_motion_validated": {
            "ok": True,
            "mode": "high_pregrasp",
        },
        "_pregrasp_plan_preflight": {
            "ik_pregrasp": "SKIP_HIGH_ENTRY",
            "plan_pregrasp": "OK_OBJECT_HIGH",
            "ik_grasp": "FAIL",
            "plan_before_result": OK_CHIPS_HIGH_ROUTE_PREVALIDATED,
            "grasp_ik_missing_allowed": True,
        },
    }


def test_ok_chips_high_route_in_full_pick_results() -> None:
    assert OK_CHIPS_HIGH_ROUTE_PREVALIDATED in OK_FULL_PICK_ROUTE_RESULTS


def test_chips_can_high_route_yaw_passes_requires_both_cartesian_segments() -> None:
    assert chips_can_high_route_yaw_passes(
        object_high_plan_ok=True,
        object_high_to_low_fraction=1.0,
        low_to_grasp_fraction=0.96,
        fraction_threshold=0.95,
    )
    assert not chips_can_high_route_yaw_passes(
        object_high_plan_ok=True,
        object_high_to_low_fraction=1.0,
        low_to_grasp_fraction=0.49,
        fraction_threshold=0.95,
    )


def test_select_yaw_variant_prefers_full_route_ok_then_joint_dist() -> None:
    variants = [
        {
            "variant_name": "top_down_yaw_pi",
            "joint_dist": 0.5,
            "object_high_plan_ok": True,
            "object_high_to_low_fraction": 1.0,
            "low_to_grasp_fraction": 0.49,
        },
        {
            "variant_name": "top_down_yaw_zero",
            "joint_dist": 2.0,
            "object_high_plan_ok": True,
            "object_high_to_low_fraction": 1.0,
            "low_to_grasp_fraction": 0.98,
        },
        {
            "variant_name": "commanded_yaw",
            "joint_dist": 1.0,
            "object_high_plan_ok": True,
            "object_high_to_low_fraction": 1.0,
            "low_to_grasp_fraction": 0.99,
        },
    ]
    selected = select_chips_can_high_route_yaw_variant(
        variants, fraction_threshold=0.95
    )
    assert selected is not None
    assert selected["variant_name"] == "commanded_yaw"


def test_summarize_yaw_variants_tracks_best_fractions() -> None:
    summary = summarize_chips_can_high_route_yaw_variants(
        [
            {
                "object_high_to_low_fraction": 1.0,
                "low_to_grasp_fraction": 0.49,
            },
            {
                "object_high_to_low_fraction": 0.88,
                "low_to_grasp_fraction": 0.72,
            },
        ]
    )
    assert summary["best_object_high_to_low_fraction"] == 1.0
    assert summary["best_low_to_grasp_fraction"] == 0.72


def test_select_pending_descend_variant_requires_high_to_low_only() -> None:
    variants = [
        {
            "variant_name": "top_down_yaw_pi",
            "joint_dist": 0.5,
            "object_high_plan_ok": True,
            "object_high_to_low_fraction": 1.0,
            "low_to_grasp_fraction": 0.5,
            "entry_tcp": (0.52, -0.095, 0.610),
        },
        {
            "variant_name": "top_down_yaw_zero",
            "joint_dist": 2.0,
            "object_high_plan_ok": True,
            "object_high_to_low_fraction": 0.8,
            "low_to_grasp_fraction": 0.99,
            "entry_tcp": (0.52, -0.095, 0.660),
        },
        {
            "variant_name": "commanded_yaw",
            "joint_dist": 1.0,
            "object_high_plan_ok": True,
            "object_high_to_low_fraction": 1.0,
            "low_to_grasp_fraction": 0.99,
            "entry_tcp": (0.52, -0.095, 0.660),
        },
    ]
    selected = select_chips_can_high_route_pending_descend_variant(
        variants, fraction_threshold=0.95, low_pregrasp_tcp_z=0.610
    )
    assert selected is not None
    assert selected["variant_name"] == "commanded_yaw"


def test_chips_can_high_route_pending_preflight_accepts() -> None:
    assert chips_can_high_route_pending_preflight_accepts(
        plan_before_result=OK_CHIPS_HIGH_ROUTE_PENDING_FINAL_DESCEND_VALIDATE,
        preflight_source=CHIPS_CAN_HIGH_ROUTE_PENDING_PREFLIGHT_SOURCE,
        object_high_plan_ok=True,
        selected_entry_target="object_high_pregrasp",
        chips_can_high_route_pending_final_descend=True,
    )


def test_chips_can_high_route_preflight_accepts() -> None:
    assert chips_can_high_route_preflight_accepts(
        plan_before_result=OK_CHIPS_HIGH_ROUTE_PREVALIDATED,
        cartesian_prevalidated=True,
        preflight_source=CHIPS_CAN_HIGH_ROUTE_PREFLIGHT_SOURCE,
        object_high_plan_ok=True,
        selected_entry_target="object_high_pregrasp",
        chips_can_high_route_prevalidated=True,
    )


def test_pick_route_preflight_allows_motion_with_cartesian_prevalidated() -> None:
    ok, reason = pick_route_preflight_allows_motion(
        plan_before_result=OK_CHIPS_HIGH_ROUTE_PREVALIDATED,
        cartesian_descend_prevalidated=True,
        full_route_required=True,
        cartesian_fraction=1.0,
        fraction_threshold=0.95,
        cartesian_descend_prevalidation_source="moveit",
        paired_validation_required=True,
    )
    assert ok is True
    assert reason == "ok"


def test_high_route_preflight_status_ok() -> None:
    stub = _PreflightStub()
    status = stub._resolve_pick_route_preflight_plan_status(_chips_high_route_candidate())
    assert status["object_high_plan_ok"] is True
    assert status["route_entry_plan_ok"] is True
    assert status["pregrasp_plan_ok"] is True
    assert status["selected_entry_target"] == "object_high_pregrasp"
    assert status["preflight_source"] == CHIPS_CAN_HIGH_ROUTE_PREFLIGHT_SOURCE
    assert status["cartesian_prevalidated"] is True
    assert status["route_preflight_ok"] is True


def test_legacy_ok_chips_high_pregrasp_still_blocked_without_cartesian() -> None:
    stub = _PreflightStub()
    cand = _chips_high_route_candidate()
    cand["_cartesian_descend_prevalidated"] = False
    cand["_chips_can_high_route_prevalidated"] = False
    cand["_pregrasp_plan_preflight"]["plan_before_result"] = "OK_CHIPS_HIGH_PREGRASP"
    status = stub._resolve_pick_route_preflight_plan_status(cand)
    assert status["route_preflight_ok"] is False
    assert status["pregrasp_plan_ok"] is False


class _ChipsHighRoutePrevalidateStub(PerceptionToPregraspTest):
    """Stub mínimo para llegar al tramo low->grasp en prevalidación chips_can."""

    def __init__(self) -> None:
        self._current_target_collision_id = "target_chips_can"
        self._enable_pick_workspace_prelude = True
        self._pick_workspace_prelude_waypoint = "pick_workspace_ready"
        self._motion_waypoints_data = {
            "pick_workspace_ready": {
                "joints": {name: 0.0 for name in PANDA_ARM_JOINT_NAMES},
            }
        }
        self._planning_frame = "world"
        self._moveit_target_link = "panda_link8"
        self._plan_before_prelude_orientation_tolerance = 0.1
        self._cartesian_fraction_threshold = 0.95
        self._add_detected_objects_to_scene = True
        self._planning_scene_object_ids = {"target_chips_can"}
        self._ensure_remove_calls: list[tuple[Any, ...]] = []
        self._moveit2 = object()
        self._last_get_cartesian_path_audit: dict = {}

    def get_logger(self) -> logging.Logger:
        return logging.getLogger("test_chips_can_high_route_prevalidate_stub")

    def _include_target_collision(self, candidate: dict) -> bool:
        return True

    def _target_collision_present_in_scene(self, target_id: str) -> bool:
        return True

    def _validate_plan_to_joint_waypoint_from_start(
        self, start_js: Any, waypoint: str, *, start_state_label: str = ""
    ) -> bool:
        return True

    def _joint_state_for_waypoint(self, waypoint: str) -> Any:
        return SimpleNamespace(name="pick_workspace_ready", position=[0.0] * 7)

    def _tcp_pose_to_moveit_pose(
        self, tcp: Tuple[float, float, float], quat: Tuple[float, float, float, float]
    ) -> list:
        return list(tcp)

    def _joint_state_from_plan_trajectory(self, traj: Any) -> Any:
        return SimpleNamespace(name="object_high", position=[0.1] * 7)

    def _apply_table_collision_if_needed(self) -> None:
        return None

    def _add_detected_objects_to_planning_scene(
        self,
        scene_obstacles: list,
        *,
        include_target: bool,
        candidate: dict,
    ) -> None:
        return None

    def _paired_pregrasp_validation_required(self, candidate: dict) -> bool:
        return True

    def _prevalidate_cartesian_descend_virtual(
        self,
        candidate: dict,
        start_tcp: Tuple[float, float, float],
        end_tcp: Tuple[float, float, float],
        quat: Tuple[float, float, float, float],
        start_js: Any,
        *,
        stage_label: str,
        target_collision_removed: bool,
        object_safe_above_to_pregrasp_ok: bool,
    ) -> tuple[bool, float]:
        self._last_get_cartesian_path_audit = {
            "fraction": 1.0,
            "joint_values_end": [0.2] * 7,
        }
        return True, 1.0

    def _joint_state_from_cartesian_audit(
        self, min_fraction: float = 0.0
    ) -> Any:
        return SimpleNamespace(name="low_pregrasp", position=[0.2] * 7)

    def _ensure_target_collision_removed_for_final_descend(
        self,
        target_collision_id: Optional[str],
        candidate: Dict[str, Any],
        *,
        timeout_sec: float = 1.0,
    ) -> bool:
        self._ensure_remove_calls.append((target_collision_id, candidate))
        return True

    def _store_plan_before_motion_validated_pregrasp(self, *args: Any, **kwargs: Any) -> None:
        return None

    def _store_pregrasp_plan_preflight_status(self, *args: Any, **kwargs: Any) -> None:
        return None

    def _store_preplanned_pick_route(self, *args: Any, **kwargs: Any) -> None:
        return None


def _chips_prevalidate_inputs() -> dict:
    candidate = {
        "label": "chips_can",
        "grasp_strategy": "cylinder_topdown",
        "uses_low_object_high_approach_stage": True,
        "scene_obstacles": [
            {
                "label": "mustard_bottle",
                "position": [0.45, 0.05, 0.45],
                "shape": "box",
                "size": [0.08, 0.08, 0.20],
            }
        ],
    }
    pick = {
        "variant_name": "top_down_yaw_pi",
        "quat": (0.0, 1.0, 0.0, 0.0),
        "commanded_yaw_rad": 3.14159,
        "ik_grasp": "FAIL",
    }
    pre_plan = (0.520, -0.095, 0.610)
    gr_plan = (0.520, -0.095, 0.475)
    high_plan = (0.520, -0.095, 0.660)
    return {
        "candidate": candidate,
        "plan_targets": {},
        "source_frame": "world",
        "snap_str": "0.520,-0.095",
        "pre_plan": pre_plan,
        "gr_plan": gr_plan,
        "high_plan": high_plan,
        "start_js": SimpleNamespace(name="home", position=[0.0] * 7),
        "start_state_label": "home",
        "pick": pick,
    }


def test_prevalidate_chips_can_calls_ensure_remove_with_target_id() -> None:
    """Regresión: _ensure_target_collision_removed_for_final_descend(target_id, candidate)."""
    stub = _ChipsHighRoutePrevalidateStub()
    inputs = _chips_prevalidate_inputs()
    candidate = inputs["candidate"]

    class _FakeTraj:
        points = [object()]

    stub._moveit2 = SimpleNamespace(  # type: ignore[assignment]
        plan=lambda **kwargs: _FakeTraj()
    )

    ok = stub._prevalidate_chips_can_high_route_before_motion(
        candidate,
        plan_targets=inputs["plan_targets"],
        source_frame=inputs["source_frame"],
        snap_str=inputs["snap_str"],
        pre_plan=inputs["pre_plan"],
        gr_plan=inputs["gr_plan"],
        high_plan=inputs["high_plan"],
        start_js=inputs["start_js"],
        start_state_label=inputs["start_state_label"],
        pick=inputs["pick"],
    )

    assert ok is True
    assert len(stub._ensure_remove_calls) == 1
    target_id, cand_arg = stub._ensure_remove_calls[0]
    assert target_id == "target_chips_can"
    assert cand_arg is candidate
    assert candidate.get("_chips_can_high_route_prevalidated") is True
    assert candidate.get("cartesian_prevalidated") is True
    pf = candidate.get("_pick_route_preflight_status") or {}
    assert pf.get("plan_before_result") == OK_CHIPS_HIGH_ROUTE_PREVALIDATED
    assert pf.get("preflight_source") == CHIPS_CAN_HIGH_ROUTE_PREFLIGHT_SOURCE


def test_search_adaptive_entry_selects_yaw_with_full_route_ok() -> None:
    """Regresión: no elegir yaw solo por joint_dist si falla low→grasp cartesiano."""
    stub = _ChipsHighRoutePrevalidateStub()
    stub._chips_can_entry_clearance_search_order = lambda: [0.15]  # type: ignore[method-assign]
    stub._chips_can_high_pregrasp_clearance_above_top_m = 0.15
    stub._chips_can_min_non_contact_clearance_above_top_m = 0.10
    stub._max_target_z = 1.0
    stub._transform_target_to_planning_frame = (  # type: ignore[method-assign]
        lambda tcp, _frame, stage="": ((tcp[0], tcp[1], tcp[2]), True)
    )

    def _fake_ranked(candidate, base_yaw, entry_plan, start_js):  # type: ignore[no-untyped-def]
        return [
            ("top_down_yaw_pi", (0.0, 1.0, 0.0, 0.0), 3.14159, object(), 0.5),
            ("top_down_yaw_zero", (0.0, 1.0, 0.0, 0.0), 0.0, object(), 2.0),
        ]

    stub._ranked_pregrasp_yaw_variants_for_pose = _fake_ranked  # type: ignore[method-assign]

    probe_calls: list[str] = []

    def _fake_probe(candidate, **kwargs):  # type: ignore[no-untyped-def]
        name = str(kwargs.get("variant_name", ""))
        probe_calls.append(name)
        if name == "top_down_yaw_pi":
            return {
                "variant_name": name,
                "quat": kwargs["quat"],
                "commanded_yaw_rad": kwargs["commanded_yaw_rad"],
                "joint_dist": kwargs["joint_dist"],
                "object_high_plan_ok": True,
                "object_high_to_low_fraction": 1.0,
                "low_to_grasp_fraction": 0.49,
                "full_route_ok": False,
                "reject_reason": "low_to_grasp_fraction_below_threshold",
                "high_js": SimpleNamespace(name="high", position=[0.1] * 7),
            }
        return {
            "variant_name": name,
            "quat": kwargs["quat"],
            "commanded_yaw_rad": kwargs["commanded_yaw_rad"],
            "joint_dist": kwargs["joint_dist"],
            "object_high_plan_ok": True,
            "object_high_to_low_fraction": 1.0,
            "low_to_grasp_fraction": 0.98,
            "full_route_ok": True,
            "reject_reason": "",
            "high_js": SimpleNamespace(name="high", position=[0.2] * 7),
        }

    stub._probe_chips_can_high_route_yaw_variant = _fake_probe  # type: ignore[method-assign]
    stub._log_chips_can_entry_z_search = lambda *args, **kwargs: None  # type: ignore[method-assign]
    stub._moveit2 = SimpleNamespace(compute_ik=lambda **kwargs: object())

    candidate = {"label": "chips_can", "grasp_strategy": "cylinder_topdown"}
    pick = stub._search_chips_can_adaptive_entry_pregrasp(
        candidate,
        top_z=0.51,
        grasp_xy=(0.52, -0.095),
        gr_plan=(0.52, -0.095, 0.475),
        pre_plan=(0.52, -0.095, 0.610),
        start_js=SimpleNamespace(name="home", position=[0.0] * 7),
        base_yaw=0.0,
        source_frame="world",
    )
    assert pick is not None
    assert pick["variant_name"] == "top_down_yaw_zero"
    assert float(pick["low_to_grasp_fraction"]) >= 0.95
    assert probe_calls[0] == "top_down_yaw_pi"


def test_pending_preflight_status_ok() -> None:
    stub = _PreflightStub()
    cand = _chips_high_route_candidate()
    cand["_chips_can_high_route_pending_final_descend"] = True
    cand["_cartesian_descend_prevalidated"] = False
    cand["_chips_can_high_route_prevalidated"] = False
    cand["_pregrasp_plan_preflight"]["plan_before_result"] = (
        OK_CHIPS_HIGH_ROUTE_PENDING_FINAL_DESCEND_VALIDATE
    )
    status = stub._resolve_pick_route_preflight_plan_status(cand)
    assert status["preflight_source"] == CHIPS_CAN_HIGH_ROUTE_PENDING_PREFLIGHT_SOURCE
    assert status["object_high_plan_ok"] is True
    assert status["route_preflight_ok"] is True
    assert status["cartesian_prevalidated"] is False


class _ChipsCacheStub(PerceptionToPregraspTest):
    def __init__(self) -> None:
        self._current_js = SimpleNamespace(
            name=[f"panda_joint{i}" for i in range(1, 8)],
            position=[0.1 * i for i in range(1, 8)],
        )
        self._current_tcp = (0.520, -0.095, 0.610)

    def get_logger(self) -> logging.Logger:
        return logging.getLogger("test_chips_can_cache")

    def _current_arm_joint_state(self) -> Any:
        return self._current_js

    def _tcp_position_in_planning_frame(self) -> Optional[Tuple[float, float, float]]:
        return self._current_tcp

    def _joint_state_weighted_distance(self, a: Any, b: Any) -> float:
        return 0.0

    def _tcp_xyz_error_m(self, a: Any, b: Tuple[float, float, float]) -> float:
        return 0.001


def test_verify_chips_can_cached_low_pregrasp_state_ok() -> None:
    stub = _ChipsCacheStub()
    candidate: dict = {
        "label": "chips_can",
        "validated_pregrasp_source": CHIPS_CAN_HIGH_TO_LOW_ACTUAL_SOURCE,
        "validated_pregrasp_js_after_cartesian": stub._current_js,
        "_chips_can_gt_centering_verified_at_pregrasp": True,
        "_chips_can_gripper_centering_ok_at_pregrasp": True,
        "_chips_can_object_disturbance_ok_after_open": True,
    }
    assert stub._verify_chips_can_cached_low_pregrasp_state_before_descend(
        candidate,
        (0.520, -0.095, 0.610),
    )


def test_verify_accepts_tcp_ok_despite_js_mismatch_after_gripper_open() -> None:
    ok, reason = evaluate_chips_can_high_to_low_pregrasp_verify(
        tcp_error_m=0.0005,
        tcp_threshold_m=0.015,
        js_distance=0.1325,
        js_threshold=0.08,
        disturbance_ok=True,
        centering_ok=True,
    )
    assert ok is True
    assert reason == "chips_can_high_to_low_tcp_pose_ok_js_refreshed"
