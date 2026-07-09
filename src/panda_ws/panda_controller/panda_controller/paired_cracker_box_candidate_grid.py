"""Grid de candidatos paired pregrasp+descenso para demo_scene_02 cracker_box."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Sequence, Tuple

CRACKER_PREGRASP_Z_VARIANTS_M: Tuple[float, ...] = (0.555, 0.575, 0.595, 0.620)
CRACKER_DEPTH_FROM_TOP_VARIANTS_M: Tuple[float, ...] = (0.020, 0.025, 0.030, 0.033)
CRACKER_YAW_OFFSETS_DEG: Tuple[float, ...] = (0.0, -5.0, 5.0, -10.0, 10.0)
CRACKER_IK_SEED_LABELS: Tuple[str, ...] = (
    "pick_workspace_ready",
    "object_safe_above",
    "home",
    "joint7_near_zero",
)
CRACKER_PRIORITIZED_IK_SEEDS: Tuple[str, ...] = (
    "pick_workspace_ready",
    "object_safe_above",
)

PAIRED_GRID_MODE_PRIORITIZED = "prioritized"
PAIRED_GRID_MODE_PRIORITIZED_OR_CACHED = "prioritized_or_cached"
PAIRED_GRID_MODE_FULL_DEBUG = "full_debug"
PAIRED_GRID_MAX_PRIORITIZED_CANDIDATES = 36

PAIRED_SAFE_GEOMETRIC_SOURCE = "paired_safe_geometric"
CRACKER_RECORDED_WAYPOINT = "demo_scene_02_cracker_box_above"


def wrap_to_pi(angle: float) -> float:
    a = float(angle)
    while a > math.pi:
        a -= 2.0 * math.pi
    while a < -math.pi:
        a += 2.0 * math.pi
    return a


def build_cracker_paired_yaw_variants(base_yaw_rad: float) -> List[Tuple[str, float]]:
    """yaw, yaw+pi, y sus ±5°/±10°."""
    base = wrap_to_pi(float(base_yaw_rad))
    out: List[Tuple[str, float]] = []
    seen: set = set()

    def _add(name: str, yaw_rad: float) -> None:
        y = wrap_to_pi(yaw_rad)
        key = round(y, 5)
        if key in seen:
            return
        seen.add(key)
        out.append((str(name), y))

    for pi_add in (0.0, math.pi):
        root = wrap_to_pi(base + pi_add)
        tag = "yaw" if abs(pi_add) < 1e-6 else "yaw_pi"
        _add("%s" % tag, root)
        for off_deg in CRACKER_YAW_OFFSETS_DEG:
            if abs(off_deg) < 1e-9:
                continue
            off = math.radians(float(off_deg))
            _add("%s_%+.0fdeg" % (tag, off_deg), root + off)
    return out


def cracker_tcp_targets_for_grid(
    *,
    xy: Tuple[float, float],
    top_z: float,
    pregrasp_tcp_z: float,
    depth_from_top_m: float,
) -> Tuple[Tuple[float, float, float], Tuple[float, float, float], float]:
    grasp_z = float(top_z) - float(depth_from_top_m)
    pre = (float(xy[0]), float(xy[1]), float(pregrasp_tcp_z))
    gr = (float(xy[0]), float(xy[1]), float(grasp_z))
    return pre, gr, float(depth_from_top_m)


def iter_cracker_paired_grid_specs(
    *,
    xy: Tuple[float, float],
    top_z: float,
    base_yaw_rad: float,
    pregrasp_z_variants: Sequence[float] = CRACKER_PREGRASP_Z_VARIANTS_M,
    depth_variants: Sequence[float] = CRACKER_DEPTH_FROM_TOP_VARIANTS_M,
    ik_seed_labels: Sequence[str] = CRACKER_IK_SEED_LABELS,
) -> List[Dict[str, Any]]:
    specs: List[Dict[str, Any]] = []
    idx = 0
    yaws = build_cracker_paired_yaw_variants(base_yaw_rad)
    for pre_z in pregrasp_z_variants:
        for depth in depth_variants:
            pre_plan, gr_plan, depth_val = cracker_tcp_targets_for_grid(
                xy=xy,
                top_z=top_z,
                pregrasp_tcp_z=float(pre_z),
                depth_from_top_m=float(depth),
            )
            for yaw_name, yaw_rad in yaws:
                for seed_label in ik_seed_labels:
                    specs.append(
                        {
                            "grid_idx": int(idx),
                            "yaw_name": str(yaw_name),
                            "yaw_rad": float(yaw_rad),
                            "yaw_deg": math.degrees(float(yaw_rad)),
                            "pregrasp_tcp_z": float(pre_z),
                            "grasp_tcp_z": float(gr_plan[2]),
                            "depth_from_top_m": float(depth_val),
                            "pre_plan": pre_plan,
                            "gr_plan": gr_plan,
                            "ik_seed_label": str(seed_label),
                        }
                    )
                    idx += 1
    return specs


def _cracker_spec_entry(
    *,
    grid_idx: int,
    xy: Tuple[float, float],
    top_z: float,
    yaw_name: str,
    yaw_rad: float,
    pregrasp_tcp_z: float,
    depth_from_top_m: float,
    ik_seed_label: str,
    priority: str = "fallback",
) -> Dict[str, Any]:
    pre_plan, gr_plan, depth_val = cracker_tcp_targets_for_grid(
        xy=xy,
        top_z=float(top_z),
        pregrasp_tcp_z=float(pregrasp_tcp_z),
        depth_from_top_m=float(depth_from_top_m),
    )
    return {
        "grid_idx": int(grid_idx),
        "yaw_name": str(yaw_name),
        "yaw_rad": float(yaw_rad),
        "yaw_deg": math.degrees(float(yaw_rad)),
        "pregrasp_tcp_z": float(pregrasp_tcp_z),
        "grasp_tcp_z": float(gr_plan[2]),
        "depth_from_top_m": float(depth_val),
        "pre_plan": pre_plan,
        "gr_plan": gr_plan,
        "ik_seed_label": str(ik_seed_label),
        "priority": str(priority),
    }


def iter_cracker_paired_prioritized_grid_specs(
    *,
    xy: Tuple[float, float],
    top_z: float,
    commanded_yaw_rad: float,
    selected_pregrasp_z: float,
    recommended_depth_from_top: float,
    max_candidates: int = PAIRED_GRID_MAX_PRIORITIZED_CANDIDATES,
    ik_seed_labels: Sequence[str] = CRACKER_PRIORITIZED_IK_SEEDS,
) -> List[Dict[str, Any]]:
    """Grid priorizado: canónico + fallback pequeño (máx 36 por defecto)."""
    specs: List[Dict[str, Any]] = []
    seen: set = set()

    def _key(
        yaw_rad: float,
        pre_z: float,
        depth: float,
        seed: str,
    ) -> Tuple[float, float, float, str]:
        return (
            round(float(yaw_rad), 5),
            round(float(pre_z), 4),
            round(float(depth), 4),
            str(seed),
        )

    def _append(
        *,
        yaw_name: str,
        yaw_rad: float,
        pre_z: float,
        depth: float,
        seed: str,
        priority: str,
    ) -> None:
        if len(specs) >= int(max_candidates):
            return
        k = _key(yaw_rad, pre_z, depth, seed)
        if k in seen:
            return
        seen.add(k)
        specs.append(
            _cracker_spec_entry(
                grid_idx=len(specs),
                xy=xy,
                top_z=float(top_z),
                yaw_name=yaw_name,
                yaw_rad=float(yaw_rad),
                pregrasp_tcp_z=float(pre_z),
                depth_from_top_m=float(depth),
                ik_seed_label=str(seed),
                priority=str(priority),
            )
        )

    cmd_yaw = wrap_to_pi(float(commanded_yaw_rad))
    sel_pre = float(selected_pregrasp_z)
    rec_depth = float(recommended_depth_from_top)
    primary_seed = str(ik_seed_labels[0]) if ik_seed_labels else "pick_workspace_ready"

    _append(
        yaw_name="canonical_commanded_yaw",
        yaw_rad=cmd_yaw,
        pre_z=sel_pre,
        depth=rec_depth,
        seed=primary_seed,
        priority="canonical",
    )

    yaw_roots = (cmd_yaw, wrap_to_pi(cmd_yaw + math.pi))
    pre_zs = (sel_pre, sel_pre + 0.015, sel_pre + 0.030)
    depths = (rec_depth, rec_depth - 0.005, rec_depth + 0.005)
    seeds = tuple(str(s) for s in ik_seed_labels[:2]) or CRACKER_PRIORITIZED_IK_SEEDS

    for i, yaw_rad in enumerate(yaw_roots):
        yaw_name = "commanded_yaw" if i == 0 else "commanded_yaw_pi"
        for pre_z in pre_zs:
            for depth in depths:
                for seed in seeds:
                    _append(
                        yaw_name=yaw_name,
                        yaw_rad=yaw_rad,
                        pre_z=float(pre_z),
                        depth=float(depth),
                        seed=seed,
                        priority="fallback",
                    )
                    if len(specs) >= int(max_candidates):
                        return specs[: int(max_candidates)]
    return specs[: int(max_candidates)]


def iter_known_box_paired_grid_specs(
    *,
    label: str,
    xy: Tuple[float, float],
    top_z: float,
    commanded_yaw_rad: float,
    selected_pregrasp_z: float,
    recommended_depth_from_top: float,
    grid_mode: str,
    max_candidates: int,
) -> List[Dict[str, Any]]:
    """Delega al iterador de grid paired según label (cracker_box / sugar_box)."""
    lb = str(label or "").strip().lower()
    if lb == "sugar_box":
        from panda_controller.sugar_box_depth_search import (
            iter_sugar_paired_full_grid_specs,
            iter_sugar_paired_prioritized_grid_specs,
        )

        if grid_mode == PAIRED_GRID_MODE_FULL_DEBUG:
            return iter_sugar_paired_full_grid_specs(
                xy=xy,
                top_z=float(top_z),
                base_yaw_rad=float(commanded_yaw_rad),
            )
        return iter_sugar_paired_prioritized_grid_specs(
            xy=xy,
            top_z=float(top_z),
            commanded_yaw_rad=float(commanded_yaw_rad),
            selected_pregrasp_z=float(selected_pregrasp_z),
            recommended_depth_from_top=float(recommended_depth_from_top),
            max_candidates=int(max_candidates),
        )
    if grid_mode == PAIRED_GRID_MODE_FULL_DEBUG:
        return iter_cracker_paired_grid_specs(
            xy=xy, top_z=float(top_z), base_yaw_rad=float(commanded_yaw_rad)
        )
    return iter_cracker_paired_prioritized_grid_specs(
        xy=xy,
        top_z=float(top_z),
        commanded_yaw_rad=float(commanded_yaw_rad),
        selected_pregrasp_z=float(selected_pregrasp_z),
        recommended_depth_from_top=float(recommended_depth_from_top),
        max_candidates=int(max_candidates),
    )


def resolve_paired_grid_search_mode(
    *,
    mode_param: str,
    enable_full_640_debug: bool,
) -> str:
    mode = str(mode_param or PAIRED_GRID_MODE_PRIORITIZED_OR_CACHED).strip().lower()
    if bool(enable_full_640_debug):
        return PAIRED_GRID_MODE_FULL_DEBUG
    if mode in (PAIRED_GRID_MODE_FULL_DEBUG, "full", "640", "debug"):
        return PAIRED_GRID_MODE_FULL_DEBUG
    if mode in (
        PAIRED_GRID_MODE_PRIORITIZED_OR_CACHED,
        "cached",
        "prioritized_or_cached",
        "prioritized+cached",
    ):
        return PAIRED_GRID_MODE_PRIORITIZED_OR_CACHED
    if mode in (PAIRED_GRID_MODE_PRIORITIZED, "prioritized"):
        return PAIRED_GRID_MODE_PRIORITIZED
    return PAIRED_GRID_MODE_PRIORITIZED_OR_CACHED


def format_paired_candidate_grid_try_log(fields: Dict[str, Any]) -> str:
    return (
        "[PAIRED_CANDIDATE_GRID_TRY]\n"
        "label=%s\n"
        "candidate_idx=%s\n"
        "yaw_deg=%s\n"
        "pregrasp_tcp_z=%s\n"
        "grasp_tcp_z=%s\n"
        "depth_from_top=%s\n"
        "ik_seed=%s\n"
        "ik_pregrasp_ok=%s\n"
        "plan_to_pregrasp_ok=%s\n"
        "fk_contract_ok=%s\n"
        "cartesian_fraction=%s\n"
        "endpoint_ik_ok=%s\n"
        "collision_ok=%s\n"
        "joint_limit_margin=%s\n"
        "result=%s\n"
        "reject_reason=%s"
        % (
            fields.get("label", "n/a"),
            fields.get("candidate_idx", "n/a"),
            fields.get("yaw_deg", "n/a"),
            fields.get("pregrasp_tcp_z", "n/a"),
            fields.get("grasp_tcp_z", "n/a"),
            fields.get("depth_from_top", "n/a"),
            fields.get("ik_seed", "n/a"),
            str(bool(fields.get("ik_pregrasp_ok"))).lower(),
            str(bool(fields.get("plan_to_pregrasp_ok"))).lower(),
            str(bool(fields.get("fk_contract_ok"))).lower(),
            fields.get("cartesian_fraction", "n/a"),
            str(bool(fields.get("endpoint_ik_ok"))).lower(),
            str(bool(fields.get("collision_ok"))).lower(),
            "n/a"
            if fields.get("joint_limit_margin") is None
            else "%.4f" % float(fields["joint_limit_margin"]),
            str(fields.get("result", "REJECT")),
            str(fields.get("reject_reason", "")),
        )
    )


def format_cartesian_descend_fail_diag_extended(fields: Dict[str, Any]) -> str:
    return (
        "[CARTESIAN_DESCEND_FAIL_DIAG]\n"
        "label=%s\n"
        "fraction=%s\n"
        "failed_at_step=%s\n"
        "failed_tcp_z=%s\n"
        "target_grasp_tcp_z=%s\n"
        "endpoint_ik_ok=%s\n"
        "collision_contact=%s\n"
        "joint_limit_near=%s\n"
        "offline_obstacle_probes=%s\n"
        "likely_blocking_obstacle=%s\n"
        "reason=%s"
        % (
            fields.get("label", "n/a"),
            fields.get("fraction", "n/a"),
            fields.get("failed_at_step", "n/a"),
            fields.get("failed_tcp_z", "n/a"),
            fields.get("target_grasp_tcp_z", "n/a"),
            str(bool(fields.get("endpoint_ik_ok"))).lower(),
            fields.get("collision_contact", "n/a"),
            str(bool(fields.get("joint_limit_near"))).lower(),
            fields.get("offline_obstacle_probes", "n/a"),
            fields.get("likely_blocking_obstacle", "n/a"),
            fields.get("reason", "unknown"),
        )
    )
