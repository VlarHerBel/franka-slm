"""Preflight de ruta pick demo multiobjeto (sin ROS)."""

from __future__ import annotations

from typing import Any, Dict, Iterable, Optional, Sequence, Set, Tuple

CHIPS_CAN_LABEL = "chips_can"
SUGAR_BOX_LABEL = "sugar_box"

OK_FULL_PICK_ROUTE_RESULTS = frozenset(
    {
        "OK",
        "OK_FULL_PICK_ROUTE",
        "OK_FULL_PICK_ROUTE_PREVALIDATED",
        "OK_SAFE_PREGRASP",
        "OK_CHIPS_HIGH_PREGRASP",
        "OK_CHIPS_HIGH_ROUTE_PREVALIDATED",
        "OK_CHIPS_HIGH_ROUTE_PENDING_FINAL_DESCEND_VALIDATE",
        "OK_CHIPS_LEGACY_SUCCESS_POLICY",
        "OK_CHIPS_LEGACY_PENDING_ACTUAL_TF_DESCEND",
        "OK_CHIPS_ADAPTIVE_ENTRY_PREGRASP",
        "OK_HIGH_STAGE_ONLY",
        "PREPLANNED_OK",
    }
)

REJECTED_DEFERRED_PICK_ROUTE_RESULTS = frozenset(
    {
        "OK_OBJECT_SAFE_ABOVE_PRELUDE_PHASE1",
        "OK_OBJECT_SAFE_ABOVE_DEFERRED",
        "OK_PRELUDE_PHASE1",
    }
)

# Fase 1 sugar_box: solo HOME -> pick_workspace_ready; ruta completa tras preludio.
OK_PRELUDE_PHASE1_RESULTS = frozenset({"OK_PRELUDE_PHASE1"})

# Descenso cartesiano se valida en pregrasp real (no desde HOME/current).
OK_PREGRASP_PENDING_DESCEND_RESULTS = frozenset(
    {"OK_PREGRASP_PENDING_DESCEND_VALIDATE"}
)


def _label_lower(candidate: Dict[str, Any]) -> str:
    return str(candidate.get("label", "")).strip().lower()


def active_table_obstacle_count(scene_obstacles: Optional[Sequence[Dict[str, Any]]]) -> int:
    if not scene_obstacles:
        return 0
    n = 0
    for obs in scene_obstacles:
        if not isinstance(obs, dict):
            continue
        if bool(obs.get("is_target", False)):
            continue
        n += 1
    return n


def scene_obstacle_labels_for_log(
    scene_obstacles: Optional[Sequence[Dict[str, Any]]],
) -> list:
    out: list = []
    if not scene_obstacles:
        return out
    for obs in scene_obstacles:
        if not isinstance(obs, dict) or bool(obs.get("is_target", False)):
            continue
        lb = str(obs.get("label", "")).strip()
        if lb:
            out.append(lb)
    return out


def demo_full_pick_route_prevalidation_required(
    *,
    candidate: Dict[str, Any],
    demo_fast_mode: bool,
    demo_motion_profile_active: bool,
    require_param: bool,
    chips_can_candidate: bool,
    demo_authoritative_scene: bool = False,
    scene_id: str = "",
) -> bool:
    from panda_controller.paired_pregrasp_descend_validation import (
        paired_pregrasp_descend_validation_required,
    )

    if paired_pregrasp_descend_validation_required(
        candidate=candidate,
        demo_authoritative_scene=demo_authoritative_scene,
        scene_id=scene_id,
    ):
        return True
    if chips_can_candidate:
        return False
    if active_table_obstacle_count(candidate.get("scene_obstacles")) < 1:
        return False
    if bool(demo_fast_mode) or bool(demo_motion_profile_active):
        return True
    return bool(require_param)


def sugar_box_force_safe_pregrasp_when_obstacles(
    *,
    candidate: Dict[str, Any],
    param_enabled: bool,
    min_obstacles: int = 1,
) -> bool:
    if not param_enabled:
        return False
    if _label_lower(candidate) != SUGAR_BOX_LABEL:
        return False
    if bool(candidate.get("_sugar_box_demo_scene_02_remaining_equivalent")):
        return False
    return active_table_obstacle_count(candidate.get("scene_obstacles")) >= int(
        min_obstacles
    )


def executor_object_marked_placed_or_completed(obj: Dict[str, Any]) -> bool:
    for key in ("placed", "pick_completed", "grasp_completed", "deposited"):
        val = obj.get(key)
        if isinstance(val, bool) and val:
            return True
    for key in ("status", "state", "lifecycle", "operational_state"):
        val = obj.get(key)
        if isinstance(val, str):
            norm = val.strip().lower()
            if norm in (
                "placed",
                "completed",
                "deposited",
                "in_place",
                "done",
                "removed_from_table",
            ):
                return True
    return False


def executor_object_excluded_from_table_obstacles(
    obj: Dict[str, Any],
    *,
    completed_entities: Set[str],
    completed_labels: Set[str],
) -> bool:
    if executor_object_marked_placed_or_completed(obj):
        return True
    ent = str(obj.get("entity_name") or obj.get("gt_entity_name") or "").strip()
    if ent:
        short = ent.split("::")[-1]
        if short in completed_entities or ent in completed_entities:
            return True
    lb = str(obj.get("label", "")).strip().lower()
    if lb and lb in completed_labels:
        return True
    return False


def pick_route_preflight_allows_motion(
    *,
    plan_before_result: str,
    cartesian_descend_prevalidated: bool,
    full_route_required: bool,
    cartesian_fraction: Optional[float],
    fraction_threshold: float,
    sugar_two_phase_deferred: bool = False,
    cartesian_descend_pending_at_pregrasp: bool = False,
    object_safe_above_deferred: bool = False,
    cartesian_descend_prevalidation_source: str = "moveit",
    paired_validation_required: bool = False,
    label: str = "",
    simple_direct_route: bool = False,
) -> Tuple[bool, str]:
    result = str(plan_before_result or "").strip()
    if sugar_two_phase_deferred and result in OK_PRELUDE_PHASE1_RESULTS:
        return True, "deferred_full_route_after_pick_workspace_ready"
    if result in REJECTED_DEFERRED_PICK_ROUTE_RESULTS:
        return False, "deferred_or_incomplete_pick_route"
    if object_safe_above_deferred:
        return False, "object_safe_above_route_not_fully_prevalidated"
    if (
        cartesian_descend_pending_at_pregrasp
        and result in OK_PREGRASP_PENDING_DESCEND_RESULTS
    ):
        return True, "descend_validate_at_pregrasp"
    if not full_route_required:
        if result == "OK_PREGRASP_ONLY":
            return True, "legacy_pregrasp_only"
        return result in OK_FULL_PICK_ROUTE_RESULTS or result == "OK_PREGRASP_ONLY", (
            "ok" if result else "missing_plan_before_result"
        )
    if not cartesian_descend_prevalidated:
        return False, "cartesian_descend_not_prevalidated"
    if result == "OK_PREGRASP_ONLY":
        return False, "cartesian_descend_not_prevalidated"
    if result not in OK_FULL_PICK_ROUTE_RESULTS and result != "OK_FULL_PICK_ROUTE":
        return False, "plan_before_result_not_full_route"
    if paired_validation_required:
        from panda_controller.paired_pregrasp_descend_validation import (
            paired_prevalidation_source_acceptable,
        )

        if not paired_prevalidation_source_acceptable(
            label=str(label or ""),
            prevalidation_source=str(cartesian_descend_prevalidation_source or ""),
            simple_direct_route=bool(simple_direct_route),
        ):
            src = str(cartesian_descend_prevalidation_source or "").strip()
            if src == "geometric_fallback":
                return (
                    False,
                    "geometric_fallback_not_allowed_in_paired_pregrasp_validation",
                )
            return False, "cartesian_descend_not_moveit_prevalidated"
    if result == "OK_FULL_PICK_ROUTE_PREVALIDATED":
        return True, "full_pick_route_prevalidated"
    if cartesian_fraction is not None:
        if (
            str(cartesian_descend_prevalidation_source or "").strip()
            != "geometric_fallback"
            and float(cartesian_fraction) + 1e-6 < float(fraction_threshold)
        ):
            return False, "cartesian_fraction_below_threshold"
    return True, "ok"
