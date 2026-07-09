"""Política de ejecución conservadora demo_scene_02 + cracker_box."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

from panda_controller.demo_cracker_box_cartesian_prevalidate import (
    demo_scene_02_cracker_box_policy_active,
)

MAX_SANE_GRIPPER_CENTERING_ERROR_XY_M = 0.050
DEFAULT_XY_CORRECTION_MAX_STEP_M = 0.006


def lock_demo_cracker_pregrasp_on_candidate(
    candidate: Dict[str, Any],
    *,
    pregrasp_tcp: Sequence[float],
    object_safe_above_tcp: Sequence[float],
) -> None:
    """Fija pregrasp demo en candidate tras aplicar DEMO_SCENE_02_CRACKER_PREGRASP_POLICY."""
    candidate["demo_pregrasp_policy_locked"] = True
    candidate["demo_pregrasp_tcp"] = [
        float(pregrasp_tcp[0]),
        float(pregrasp_tcp[1]),
        float(pregrasp_tcp[2]),
    ]
    candidate["demo_pregrasp_tcp_z"] = float(pregrasp_tcp[2])
    candidate["demo_object_safe_above_tcp"] = [
        float(object_safe_above_tcp[0]),
        float(object_safe_above_tcp[1]),
        float(object_safe_above_tcp[2]),
    ]
    candidate["demo_object_safe_above_tcp_z"] = float(object_safe_above_tcp[2])
    candidate["demo_conservative_pregrasp_execution"] = True


def demo_pregrasp_policy_locked(candidate: Dict[str, Any]) -> bool:
    return bool(candidate.get("demo_pregrasp_policy_locked"))


def demo_conservative_pregrasp_execution_active(
    candidate: Dict[str, Any],
    *,
    demo_authoritative_scene: bool,
    scene_id: str,
) -> bool:
    label = str(candidate.get("label", ""))
    return demo_scene_02_cracker_box_policy_active(
        label=label,
        demo_authoritative_scene=demo_authoritative_scene,
        scene_id=scene_id,
    ) and demo_pregrasp_policy_locked(candidate)


def demo_pregrasp_xy_correction_cartesian_only(
    *,
    demo_authoritative_scene: bool,
    scene_id: str,
    label: str,
) -> bool:
    return demo_scene_02_cracker_box_policy_active(
        label=label,
        demo_authoritative_scene=demo_authoritative_scene,
        scene_id=scene_id,
    )


def demo_locked_pregrasp_floor_z(candidate: Dict[str, Any]) -> Optional[float]:
    if not demo_pregrasp_policy_locked(candidate):
        return None
    raw = candidate.get("demo_pregrasp_tcp_z")
    if raw is None:
        demo_pre = candidate.get("demo_pregrasp_tcp")
        if isinstance(demo_pre, (list, tuple)) and len(demo_pre) >= 3:
            raw = demo_pre[2]
    try:
        return float(raw) if raw is not None else None
    except (TypeError, ValueError):
        return None


def apply_demo_locked_pregrasp_z_floor_to_reachability(
    candidate: Dict[str, Any],
    *,
    min_pre_z: float,
    desired_pre_z: float,
    start_z: float,
    z_values: List[float],
) -> Tuple[float, float, float, List[float], Optional[float]]:
    """Impide búsqueda de pregrasp por debajo del piso demo."""
    floor_z = demo_locked_pregrasp_floor_z(candidate)
    if floor_z is None:
        return min_pre_z, desired_pre_z, start_z, z_values, None
    min_pre_z = max(float(min_pre_z), float(floor_z))
    desired_pre_z = max(float(desired_pre_z), float(floor_z))
    start_z = max(float(start_z), float(floor_z))
    filtered = [float(z) for z in z_values if float(z) >= float(floor_z) - 1e-6]
    if not filtered:
        filtered = [float(floor_z)]
    return min_pre_z, desired_pre_z, start_z, filtered, float(floor_z)


def enforce_demo_locked_pregrasp_plan_targets(
    candidate: Dict[str, Any],
    plan_targets: Dict[str, Any],
) -> Tuple[bool, str, Optional[float], Optional[float]]:
    """Restaura pregrasp/object_safe demo; falla si quedó por debajo del piso."""
    if not demo_pregrasp_policy_locked(candidate):
        return True, "not_locked", None, None

    demo_pre = candidate.get("demo_pregrasp_tcp")
    demo_safe = candidate.get("demo_object_safe_above_tcp")
    if not isinstance(demo_pre, (list, tuple)) or len(demo_pre) < 3:
        return False, "missing_demo_pregrasp_tcp", None, None

    demo_z = float(demo_pre[2])
    pre_tcp = plan_targets.get("pregrasp_tcp")
    selected_z: Optional[float] = None
    if isinstance(pre_tcp, (list, tuple)) and len(pre_tcp) >= 3:
        selected_z = float(pre_tcp[2])
        if selected_z < demo_z - 1e-6:
            return (
                False,
                "pregrasp_lowered_after_demo_policy",
                selected_z,
                demo_z,
            )
        if selected_z < demo_z + 1e-6:
            plan_targets["pregrasp_tcp"] = (
                float(demo_pre[0]),
                float(demo_pre[1]),
                demo_z,
            )
            selected_z = demo_z

    if isinstance(demo_safe, (list, tuple)) and len(demo_safe) >= 3:
        candidate["object_safe_above_tcp"] = [
            float(demo_safe[0]),
            float(demo_safe[1]),
            float(demo_safe[2]),
        ]
        plan_targets["object_safe_above_tcp"] = (
            float(demo_safe[0]),
            float(demo_safe[1]),
            float(demo_safe[2]),
        )

    final_descend = float(plan_targets["pregrasp_tcp"][2]) - float(
        plan_targets["grasp_tcp"][2]
    )
    candidate["vertical_descend_tcp_m"] = final_descend
    candidate["effective_approach_distance_m"] = final_descend
    return True, "locked", selected_z, demo_z


def gripper_centering_error_sane(
    error_xy_m: float,
    *,
    max_sane_error_xy_m: float = MAX_SANE_GRIPPER_CENTERING_ERROR_XY_M,
) -> bool:
    try:
        err = float(error_xy_m)
    except (TypeError, ValueError):
        return False
    if not math.isfinite(err):
        return False
    return err <= float(max_sane_error_xy_m)


def gripper_centering_correction_coherent(
    *,
    before_error_xy_m: float,
    after_error_xy_m: float,
    step_m: float,
    max_step_m: float = DEFAULT_XY_CORRECTION_MAX_STEP_M,
    max_sane_before_m: float = MAX_SANE_GRIPPER_CENTERING_ERROR_XY_M,
    tf_fresh_validated: bool = False,
) -> Tuple[bool, str]:
    """Rechaza correcciones pequeñas que aparentan arreglar errores imposibles."""
    try:
        before = float(before_error_xy_m)
        after = float(after_error_xy_m)
        step = float(step_m)
    except (TypeError, ValueError):
        return False, "invalid_metrics"
    if not math.isfinite(before) or not math.isfinite(after) or not math.isfinite(step):
        return False, "non_finite_metrics"
    if before <= max_sane_before_m + 1e-6:
        return True, "ok"
    if step > max_step_m + 1e-6:
        return True, "ok_large_step"
    if tf_fresh_validated:
        return True, "ok_fresh_tf"
    if after + 1e-6 < before * 0.5:
        return False, "incoherent_correction_without_fresh_tf"
    return True, "ok"
