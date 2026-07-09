"""Búsqueda depth/descend para sugar_box demo_scene_02 multiobjeto (sin ROS)."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

SUGAR_BOX_DEPTH_FROM_TOP_M: Tuple[float, ...] = (
    0.006,
    0.008,
    0.010,
    0.012,
    0.014,
    0.016,
    0.018,
    0.020,
    0.022,
    0.024,
)

SUGAR_BOX_DESCEND_CANDIDATES_M: Tuple[float, ...] = (
    0.040,
    0.045,
    0.050,
    0.055,
)

DEFAULT_SUGAR_BOX_MIN_PREGRASP_CLEARANCE_ABOVE_TOP_M = 0.020

# Grid fijo demo_scene_02 sugar_box (4 depths × 3 descends × 6 yaw = 72 máx.)
SUGAR_BOX_DEMO_DEPTH_FROM_TOP_M: Tuple[float, ...] = (
    0.018,
    0.020,
    0.022,
    0.024,
)

SUGAR_BOX_DEMO_DESCEND_CANDIDATES_M: Tuple[float, ...] = (
    0.045,
    0.050,
    0.055,
)

SUGAR_BOX_DEMO_YAW_VARIANT_PRIORITY: Tuple[str, ...] = (
    "top_down_yaw_pi",
    "top_down_yaw_neg_pi_over_2",
    "commanded_yaw_pi",
    "commanded_yaw",
    "top_down_yaw_pi_over_2",
    "top_down_yaw_zero",
)

SUGAR_BOX_DEMO_MAX_CANDIDATES = 72

# Golden reachability scanner (demo_scene_02 sugar_box pose 0.630,-0.175,-3.0159).
DEMO_SCENE_02_SUGAR_BOX_SCANNER_ALIGNED_PREGRASP_Z = 0.5000
DEMO_SCENE_02_SUGAR_BOX_SCANNER_ALIGNED_GRASP_Z = 0.4200
DEMO_SCENE_02_SUGAR_BOX_SCANNER_ALIGNED_DEPTH_FROM_TOP_M = 0.025
DEMO_SCENE_02_SUGAR_BOX_SCANNER_ALIGNED_COMMANDED_YAW_RAD = -1.445104
DEMO_SCENE_02_SUGAR_BOX_SCANNER_ALIGNED_REFERENCE_TOP_Z = 0.445
DEMO_SCENE_02_SUGAR_BOX_SCANNER_ALIGNED_SEED = "pick_workspace_ready"


def build_sugar_box_scanner_aligned_depth_spec(
    *,
    xy: Tuple[float, float],
    top_z_m: float,
    min_pregrasp_clearance_above_top_m: float = DEFAULT_SUGAR_BOX_MIN_PREGRASP_CLEARANCE_ABOVE_TOP_M,
) -> Optional[Dict[str, Any]]:
    """Candidato prioritario alineado con reachability scanner (pregrasp 0.500 / grasp 0.420)."""
    gr_z = float(DEMO_SCENE_02_SUGAR_BOX_SCANNER_ALIGNED_GRASP_Z)
    pre_z = float(DEMO_SCENE_02_SUGAR_BOX_SCANNER_ALIGNED_PREGRASP_Z)
    clearance = float(pre_z) - float(top_z_m)
    if clearance + 1e-6 < float(min_pregrasp_clearance_above_top_m):
        return None
    if pre_z <= gr_z + 1e-6:
        return None
    depth = float(top_z_m) - gr_z
    return {
        "depth_from_top_m": float(depth),
        "descend_m": float(pre_z) - gr_z,
        "requested_descend_m": float(pre_z) - gr_z,
        "grasp_tcp_z": gr_z,
        "pregrasp_tcp_z": pre_z,
        "effective_descend_m": float(pre_z) - gr_z,
        "pregrasp_clearance_above_top": clearance,
        "clamped": False,
        "pre_plan": (float(xy[0]), float(xy[1]), pre_z),
        "gr_plan": (float(xy[0]), float(xy[1]), gr_z),
        "scanner_aligned": True,
        "scanner_reference_top_z": float(
            DEMO_SCENE_02_SUGAR_BOX_SCANNER_ALIGNED_REFERENCE_TOP_Z
        ),
        "scanner_commanded_yaw_rad": float(
            DEMO_SCENE_02_SUGAR_BOX_SCANNER_ALIGNED_COMMANDED_YAW_RAD
        ),
        "scanner_seed": DEMO_SCENE_02_SUGAR_BOX_SCANNER_ALIGNED_SEED,
    }


def format_sugar_box_scanner_aligned_candidate_log(fields: Dict[str, Any]) -> str:
    return (
        "[SUGAR_BOX_SCANNER_ALIGNED_CANDIDATE]\n"
        "scanner_pregrasp_tcp_z=%.4f\n"
        "scanner_grasp_tcp_z=%.4f\n"
        "controller_pregrasp_tcp_z=%.4f\n"
        "controller_grasp_tcp_z=%.4f\n"
        "selected=%s\n"
        "seed=%s"
        % (
            float(DEMO_SCENE_02_SUGAR_BOX_SCANNER_ALIGNED_PREGRASP_Z),
            float(DEMO_SCENE_02_SUGAR_BOX_SCANNER_ALIGNED_GRASP_Z),
            float(fields.get("controller_pregrasp_tcp_z", 0.0)),
            float(fields.get("controller_grasp_tcp_z", 0.0)),
            str(bool(fields.get("selected", False))).lower(),
            str(fields.get("seed") or DEMO_SCENE_02_SUGAR_BOX_SCANNER_ALIGNED_SEED),
        )
    )


def prioritize_sugar_box_scanner_aligned_yaw_variants(
    ranked: Sequence[Tuple[Any, ...]],
    *,
    scanner_yaw_rad: float = DEMO_SCENE_02_SUGAR_BOX_SCANNER_ALIGNED_COMMANDED_YAW_RAD,
) -> List[Tuple[Any, ...]]:
    """Prioriza yaw más cercano al commanded_yaw del scanner."""
    items = list(ranked)
    if not items:
        return []
    return sorted(
        items,
        key=lambda item: abs(float(item[2]) - float(scanner_yaw_rad)),
    )


def build_sugar_box_demo_depth_descend_tcp_specs(
    *,
    xy: Tuple[float, float],
    top_z_m: float,
    min_pregrasp_clearance_above_top_m: float = DEFAULT_SUGAR_BOX_MIN_PREGRASP_CLEARANCE_ABOVE_TOP_M,
) -> List[Dict[str, Any]]:
    specs: List[Dict[str, Any]] = []
    scanner = build_sugar_box_scanner_aligned_depth_spec(
        xy=xy,
        top_z_m=float(top_z_m),
        min_pregrasp_clearance_above_top_m=float(min_pregrasp_clearance_above_top_m),
    )
    if scanner is not None:
        specs.append(scanner)
    specs.extend(
        build_sugar_box_depth_descend_tcp_specs(
            xy=xy,
            top_z_m=float(top_z_m),
            depth_candidates_m=SUGAR_BOX_DEMO_DEPTH_FROM_TOP_M,
            descend_candidates_m=SUGAR_BOX_DEMO_DESCEND_CANDIDATES_M,
            min_pregrasp_clearance_above_top_m=float(min_pregrasp_clearance_above_top_m),
        )
    )
    return specs


def prioritize_sugar_box_demo_yaw_variants(
    ranked: Sequence[Tuple[Any, ...]],
    *,
    priority: Sequence[str] = SUGAR_BOX_DEMO_YAW_VARIANT_PRIORITY,
) -> List[Tuple[Any, ...]]:
    order = {str(name): idx for idx, name in enumerate(priority)}
    filtered = [item for item in ranked if str(item[0]) in order]
    filtered.sort(key=lambda item: order[str(item[0])])
    return filtered


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return float(default)
        if isinstance(value, str) and value.strip().lower() in ("", "n/a", "none", "nan"):
            return float(default)
        v = float(value)
        if not math.isfinite(v):
            return float(default)
        return v
    except (TypeError, ValueError):
        return float(default)


def _variant_result_ok(item: Dict[str, Any]) -> bool:
    res = str(item.get("result", "")).strip().upper()
    if res:
        return res == "OK"
    return bool(item.get("ok", False))


def _variant_reject_reason(item: Dict[str, Any]) -> str:
    return str(item.get("reject_reason", "") or "").strip()


def _is_valid_deferred_pregrasp_variant(item: Dict[str, Any]) -> bool:
    if not _variant_result_ok(item):
        return False
    if _variant_reject_reason(item):
        return False
    if not bool(item.get("defer_final_descend", True)):
        return False
    return True


def _deferred_variant_sort_key(item: Dict[str, Any]) -> Tuple[float, float, float, float]:
    return (
        _safe_float(item.get("depth_from_top_m"), 0.0),
        _safe_float(item.get("effective_descend_m"), 0.0),
        _safe_float(item.get("pregrasp_clearance_above_top"), 0.0),
        _safe_float(item.get("object_safe_above_to_pregrasp_fraction"), 1.0),
    )


def sugar_box_grasp_tcp_z_from_depth(*, top_z_m: float, depth_from_top_m: float) -> float:
    return float(top_z_m) - float(depth_from_top_m)


def sugar_box_pregrasp_tcp_z_from_grasp_and_descend(
    *,
    grasp_tcp_z: float,
    descend_m: float,
) -> float:
    """pregrasp_tcp_z = grasp_tcp_z + descend_m (sin clamp duro por min_clearance)."""
    return float(grasp_tcp_z) + float(descend_m)


def sugar_box_depth_tcp_spec(
    *,
    xy: Tuple[float, float],
    top_z_m: float,
    depth_from_top_m: float,
    descend_m: float,
    min_pregrasp_clearance_above_top_m: float = DEFAULT_SUGAR_BOX_MIN_PREGRASP_CLEARANCE_ABOVE_TOP_M,
) -> Optional[Dict[str, Any]]:
    gr_z = sugar_box_grasp_tcp_z_from_depth(
        top_z_m=float(top_z_m), depth_from_top_m=float(depth_from_top_m)
    )
    pre_z = sugar_box_pregrasp_tcp_z_from_grasp_and_descend(
        grasp_tcp_z=gr_z, descend_m=float(descend_m)
    )
    clearance = float(pre_z) - float(top_z_m)
    if clearance + 1e-6 < float(min_pregrasp_clearance_above_top_m):
        return None
    if pre_z <= gr_z + 1e-6:
        return None
    effective_descend = float(pre_z) - float(gr_z)
    return {
        "depth_from_top_m": float(depth_from_top_m),
        "descend_m": float(descend_m),
        "requested_descend_m": float(descend_m),
        "grasp_tcp_z": float(gr_z),
        "pregrasp_tcp_z": float(pre_z),
        "effective_descend_m": float(effective_descend),
        "pregrasp_clearance_above_top": float(clearance),
        "clamped": False,
        "pre_plan": (float(xy[0]), float(xy[1]), float(pre_z)),
        "gr_plan": (float(xy[0]), float(xy[1]), float(gr_z)),
    }


def build_sugar_box_depth_descend_tcp_specs(
    *,
    xy: Tuple[float, float],
    top_z_m: float,
    depth_candidates_m: Sequence[float] = SUGAR_BOX_DEPTH_FROM_TOP_M,
    descend_candidates_m: Sequence[float] = SUGAR_BOX_DESCEND_CANDIDATES_M,
    min_pregrasp_clearance_above_top_m: float = DEFAULT_SUGAR_BOX_MIN_PREGRASP_CLEARANCE_ABOVE_TOP_M,
) -> List[Dict[str, Any]]:
    specs: List[Dict[str, Any]] = []
    for depth in depth_candidates_m:
        for descend in descend_candidates_m:
            spec = sugar_box_depth_tcp_spec(
                xy=xy,
                top_z_m=float(top_z_m),
                depth_from_top_m=float(depth),
                descend_m=float(descend),
                min_pregrasp_clearance_above_top_m=float(
                    min_pregrasp_clearance_above_top_m
                ),
            )
            if spec is not None:
                specs.append(spec)
    return specs


SUGAR_PRIORITIZED_IK_SEEDS: Tuple[str, ...] = (
    "pick_workspace_ready",
    "object_safe_above",
)

SUGAR_FULL_GRID_IK_SEEDS: Tuple[str, ...] = (
    "pick_workspace_ready",
    "object_safe_above",
    "home",
    "joint7_near_zero",
)


def _sugar_depth_spec_to_paired_grid_spec(
    depth_spec: Dict[str, Any],
    *,
    grid_idx: int,
    yaw_name: str,
    yaw_rad: float,
    ik_seed_label: str,
    priority: str = "fallback",
) -> Dict[str, Any]:
    pre_plan = depth_spec["pre_plan"]
    gr_plan = depth_spec["gr_plan"]
    return {
        "grid_idx": int(grid_idx),
        "yaw_name": str(yaw_name),
        "yaw_rad": float(yaw_rad),
        "yaw_deg": math.degrees(float(yaw_rad)),
        "pregrasp_tcp_z": float(pre_plan[2]),
        "grasp_tcp_z": float(gr_plan[2]),
        "depth_from_top_m": float(depth_spec["depth_from_top_m"]),
        "pre_plan": pre_plan,
        "gr_plan": gr_plan,
        "ik_seed_label": str(ik_seed_label),
        "priority": str(priority),
    }


def iter_sugar_paired_prioritized_grid_specs(
    *,
    xy: Tuple[float, float],
    top_z: float,
    commanded_yaw_rad: float,
    selected_pregrasp_z: float,
    recommended_depth_from_top: float,
    max_candidates: int = 72,
    ik_seed_labels: Sequence[str] = SUGAR_PRIORITIZED_IK_SEEDS,
) -> List[Dict[str, Any]]:
    """Grid priorizado sugar_box: depth×descend × yaw × IK seed (sin golden)."""
    from panda_controller.paired_cracker_box_candidate_grid import wrap_to_pi

    depth_specs = build_sugar_box_depth_descend_tcp_specs(
        xy=xy,
        top_z_m=float(top_z),
        depth_candidates_m=SUGAR_BOX_DEMO_DEPTH_FROM_TOP_M,
        descend_candidates_m=SUGAR_BOX_DEMO_DESCEND_CANDIDATES_M,
    )
    if not depth_specs:
        return []

    depth_specs = sorted(
        depth_specs,
        key=lambda spec: (
            abs(float(spec["depth_from_top_m"]) - float(recommended_depth_from_top)),
            abs(float(spec["pregrasp_tcp_z"]) - float(selected_pregrasp_z)),
        ),
    )

    specs: List[Dict[str, Any]] = []
    seen: set = set()

    def _append(
        depth_spec: Dict[str, Any],
        *,
        yaw_name: str,
        yaw_rad: float,
        seed: str,
        priority: str,
    ) -> None:
        if len(specs) >= int(max_candidates):
            return
        key = (
            round(float(yaw_rad), 5),
            round(float(depth_spec["pregrasp_tcp_z"]), 4),
            round(float(depth_spec["depth_from_top_m"]), 4),
            str(seed),
        )
        if key in seen:
            return
        seen.add(key)
        specs.append(
            _sugar_depth_spec_to_paired_grid_spec(
                depth_spec,
                grid_idx=len(specs),
                yaw_name=yaw_name,
                yaw_rad=float(yaw_rad),
                ik_seed_label=str(seed),
                priority=str(priority),
            )
        )

    cmd_yaw = wrap_to_pi(float(commanded_yaw_rad))
    primary_seed = str(ik_seed_labels[0]) if ik_seed_labels else "pick_workspace_ready"
    _append(
        depth_specs[0],
        yaw_name="canonical_commanded_yaw",
        yaw_rad=cmd_yaw,
        seed=primary_seed,
        priority="canonical",
    )

    yaw_roots = (cmd_yaw, wrap_to_pi(cmd_yaw + math.pi))
    seeds = tuple(str(s) for s in ik_seed_labels[:2]) or SUGAR_PRIORITIZED_IK_SEEDS
    for i, yaw_rad in enumerate(yaw_roots):
        yaw_name = "commanded_yaw" if i == 0 else "commanded_yaw_pi"
        for depth_spec in depth_specs:
            for seed in seeds:
                _append(
                    depth_spec,
                    yaw_name=yaw_name,
                    yaw_rad=float(yaw_rad),
                    seed=seed,
                    priority="fallback",
                )
                if len(specs) >= int(max_candidates):
                    return specs[: int(max_candidates)]
    return specs[: int(max_candidates)]


def iter_sugar_paired_full_grid_specs(
    *,
    xy: Tuple[float, float],
    top_z: float,
    base_yaw_rad: float,
    depth_candidates_m: Sequence[float] = SUGAR_BOX_DEPTH_FROM_TOP_M,
    descend_candidates_m: Sequence[float] = SUGAR_BOX_DESCEND_CANDIDATES_M,
    ik_seed_labels: Sequence[str] = SUGAR_FULL_GRID_IK_SEEDS,
) -> List[Dict[str, Any]]:
    """Grid exhaustivo sugar_box (modo debug)."""
    from panda_controller.paired_cracker_box_candidate_grid import (
        build_cracker_paired_yaw_variants,
    )

    depth_specs = build_sugar_box_depth_descend_tcp_specs(
        xy=xy,
        top_z_m=float(top_z),
        depth_candidates_m=depth_candidates_m,
        descend_candidates_m=descend_candidates_m,
    )
    specs: List[Dict[str, Any]] = []
    idx = 0
    for depth_spec in depth_specs:
        for yaw_name, yaw_rad in build_cracker_paired_yaw_variants(base_yaw_rad):
            for seed in ik_seed_labels:
                specs.append(
                    _sugar_depth_spec_to_paired_grid_spec(
                        depth_spec,
                        grid_idx=idx,
                        yaw_name=yaw_name,
                        yaw_rad=float(yaw_rad),
                        ik_seed_label=str(seed),
                    )
                )
                idx += 1
    return specs


def select_sugar_box_depth_z_variant(
    variants: Sequence[Dict[str, Any]],
    *,
    fraction_threshold: float,
    defer_final_descend: bool = False,
) -> Optional[Dict[str, Any]]:
    """Elige variante; con defer=True basta pregrasp OK (sin fraction cartesiana)."""
    if defer_final_descend:
        valid = [
            item
            for item in variants
            if _is_valid_deferred_pregrasp_variant(item)
        ]
        if not valid:
            return None
        return max(valid, key=_deferred_variant_sort_key)

    passing: List[Dict[str, Any]] = []
    threshold = float(fraction_threshold)
    for item in variants:
        if not bool(item.get("ok", False)):
            continue
        frac = item.get("cartesian_fraction")
        if frac is None:
            continue
        if _safe_float(frac, -1.0) + 1e-6 < threshold:
            continue
        passing.append(item)
    if not passing:
        return None
    return max(
        passing,
        key=lambda item: (
            _safe_float(item.get("depth_from_top_m"), 0.0),
            _safe_float(item.get("cartesian_fraction"), 0.0),
        ),
    )


def format_sugar_box_depth_z_effective_log(fields: Dict[str, Any]) -> str:
    return (
        "[SUGAR_BOX_DEPTH_Z_EFFECTIVE]\n"
        "top_z=%.4f\n"
        "depth_from_top=%.4f\n"
        "grasp_tcp_z=%.4f\n"
        "requested_descend_m=%.4f\n"
        "effective_descend_m=%.4f\n"
        "pregrasp_clearance_above_top=%.4f\n"
        "clamped=%s\n"
        "result=%s"
        % (
            float(fields.get("top_z", 0.0)),
            float(fields.get("depth_from_top_m", 0.0)),
            float(fields.get("grasp_tcp_z", 0.0)),
            float(fields.get("requested_descend_m", 0.0)),
            float(fields.get("effective_descend_m", 0.0)),
            float(fields.get("pregrasp_clearance_above_top", 0.0)),
            str(bool(fields.get("clamped", False))).lower(),
            fields.get("result", "OK"),
        )
    )


SUGAR_BOX_DEFER_FINAL_DESCEND_PREFLIGHT_SOURCE = "sugar_box_defer_final_descend"


def format_sugar_box_prevalidate_defer_final_descend_log() -> str:
    return (
        "[SUGAR_BOX_PREVALIDATE_DEFER_FINAL_DESCEND]\n"
        "reason=requires_actual_joint7_aligned_state\n"
        "prevalidated_until=pregrasp\n"
        "result=OK_DEFER_DESCEND_TO_ACTUAL_PREGRASP"
    )


def format_sugar_box_deferred_prevalidate_authorization_log(
    fields: Dict[str, Any],
) -> str:
    return (
        "[SUGAR_BOX_DEFERRED_PREVALIDATE_AUTHORIZATION]\n"
        "plan_before_result=%s\n"
        "selected_entry_target=%s\n"
        "cartesian_descend_pending_at_pregrasp=%s\n"
        "object_safe_above_plan_ok=%s\n"
        "pregrasp_plan_ok=%s\n"
        "motion_authorized=%s\n"
        "result=%s"
        % (
            str(fields.get("plan_before_result", "n/a")),
            str(fields.get("selected_entry_target", "n/a")),
            str(bool(fields.get("cartesian_descend_pending_at_pregrasp", False))).lower(),
            str(bool(fields.get("object_safe_above_plan_ok", False))).lower(),
            str(bool(fields.get("pregrasp_plan_ok", False))).lower(),
            str(bool(fields.get("motion_authorized", False))).lower(),
            str(fields.get("result", "FAIL")),
        )
    )


def format_sugar_box_depth_z_search_log(fields: Dict[str, Any]) -> str:
    frac = fields.get("cartesian_fraction")
    frac_s = "n/a" if frac is None else "%.5f" % float(frac)
    return (
        "[SUGAR_BOX_DEPTH_Z_SEARCH]\n"
        "depth_from_top=%.4f\n"
        "grasp_tcp_z=%.4f\n"
        "pregrasp_tcp_z=%.4f\n"
        "descend_m=%.4f\n"
        "yaw_variant=%s\n"
        "cartesian_fraction=%s\n"
        "grasp_ik_ok=%s\n"
        "lift_prevalidate_ok=%s\n"
        "result=%s\n"
        "reject_reason=%s"
        % (
            float(fields.get("depth_from_top_m", 0.0)),
            float(fields.get("grasp_tcp_z", 0.0)),
            float(fields.get("pregrasp_tcp_z", 0.0)),
            float(fields.get("descend_m", 0.0)),
            str(fields.get("yaw_variant", "n/a")),
            frac_s,
            str(bool(fields.get("grasp_ik_ok", False))).lower(),
            str(bool(fields.get("lift_prevalidate_ok", False))).lower(),
            fields.get("result", "FAIL"),
            fields.get("reject_reason", ""),
        )
    )


def format_sugar_box_depth_z_selected_log(fields: Dict[str, Any]) -> str:
    if str(fields.get("result", "OK")).upper() == "FAIL":
        return (
            "[SUGAR_BOX_DEPTH_Z_SELECTED]\n"
            "result=FAIL\n"
            "reason=%s"
            % str(fields.get("reason", "no_valid_deferred_pregrasp_variant"))
        )
    defer = bool(fields.get("defer_final_descend", False))
    frac = fields.get("cartesian_fraction")
    frac_s = "deferred" if defer else "%.5f" % _safe_float(frac, 0.0)
    return (
        "[SUGAR_BOX_DEPTH_Z_SELECTED]\n"
        "depth_from_top=%.4f\n"
        "grasp_tcp_z=%.4f\n"
        "pregrasp_tcp_z=%.4f\n"
        "descend_m=%.4f\n"
        "yaw_variant=%s\n"
        "cartesian_fraction=%s\n"
        "defer_final_descend=%s\n"
        "result=OK"
        % (
            _safe_float(fields.get("depth_from_top_m"), 0.0),
            _safe_float(fields.get("grasp_tcp_z"), 0.0),
            _safe_float(fields.get("pregrasp_tcp_z"), 0.0),
            _safe_float(fields.get("descend_m"), 0.0),
            str(fields.get("yaw_variant", "n/a")),
            frac_s,
            str(defer).lower(),
        )
    )


def format_sugar_box_depth_z_selected_fail_log(
    reason: str = "no_valid_deferred_pregrasp_variant",
) -> str:
    return format_sugar_box_depth_z_selected_log(
        {"result": "FAIL", "reason": str(reason)}
    )
