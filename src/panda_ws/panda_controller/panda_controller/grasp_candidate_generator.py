"""Generación y puntuación de candidatos de grasp (offsets de centro, profundidad, yaw)."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class GraspCandidate:
    idx: int
    label: str
    strategy: str
    center_xyz: Tuple[float, float, float]
    tcp_grasp_xyz: Tuple[float, float, float]
    tcp_pregrasp_xyz: Tuple[float, float, float]
    tcp_safe_pregrasp_xyz: Tuple[float, float, float]
    final_tcp_yaw_rad: float
    desired_closing_yaw_rad: float
    open_joint_m: float
    close_joint_m: float
    depth_from_top_m: float
    center_offset_m: float
    center_offset_axis: str
    yaw_offset_rad: float
    min_contact_margin_m: float
    score: float
    notes: str


def _axis_unit(detection: Dict[str, Any], axis: str) -> Optional[Tuple[float, float]]:
    key = "major_axis_xy" if axis.strip().lower() == "major_axis" else "minor_axis_xy"
    raw = detection.get(key)
    if not isinstance(raw, (list, tuple)) or len(raw) != 2:
        return None
    try:
        x, y = float(raw[0]), float(raw[1])
    except (TypeError, ValueError):
        return None
    n = math.hypot(x, y)
    if n < 1e-9:
        return None
    return (x / n, y / n)


def generate_grasp_candidates(
    detection: Dict[str, Any],
    profile: Dict[str, Any],
    gripper_jaw_axis_offset_rad: float = 0.0,
    approach_distance_m: Optional[float] = None,
    lift_distance_m: float = 0.15,
) -> List[GraspCandidate]:
    """Genera candidatos ordenados por score (mejor primero)."""
    label = str(detection.get("label", "") or "unknown")
    strategy = str(profile.get("primary_strategy", "top_down_short_axis"))
    top_z = detection.get("top_z_m")
    if top_z is None:
        return []
    try:
        top_z_f = float(top_z)
    except (TypeError, ValueError):
        return []

    pos = detection.get("chosen_target_center_base") or detection.get("position")
    if not isinstance(pos, (list, tuple)) or len(pos) < 2:
        return []
    cx0, cy0 = float(pos[0]), float(pos[1])

    base_closing = detection.get("closing_yaw_rad")
    if base_closing is None:
        base_closing = detection.get("grasp_yaw_rad")
    try:
        base_closing_f = float(base_closing) if base_closing is not None else 0.0
    except (TypeError, ValueError):
        base_closing_f = 0.0

    open_j = float(profile.get("recommended_open_joint_m") or 0.04)
    close_j = float(profile.get("recommended_close_joint_m") or 0.02)
    min_margin = float(profile.get("min_contact_margin_m") or 0.003)

    depths = profile.get("depth_candidates_from_top_m") or [
        profile.get("recommended_grasp_depth_from_top_m", 0.05)
    ]
    depths_f: List[float] = []
    for d in depths:
        try:
            depths_f.append(float(d))
        except (TypeError, ValueError):
            continue
    if not depths_f:
        try:
            depths_f = [float(profile.get("recommended_grasp_depth_from_top_m", 0.05))]
        except (TypeError, ValueError):
            depths_f = [0.05]

    center_offsets = profile.get("center_offset_candidates_m") or [0.0]
    center_axes = profile.get("center_offset_axes") or ["major_axis"]
    yaw_offs = profile.get("yaw_candidate_offsets_rad") or [0.0]

    pre_clr = float(profile.get("pregrasp_clearance_above_top_m") or 0.08)
    safe_clr = float(profile.get("safe_pregrasp_clearance_above_top_m") or 0.13)
    safe_extra = float(profile.get("safe_pregrasp_extra_above_pregrasp_m", 0.10))
    approach = (
        float(approach_distance_m)
        if approach_distance_m is not None
        else float(profile.get("approach_distance_min_m") or 0.12)
    )

    candidates: List[GraspCandidate] = []
    idx = 0
    for yaw_off in yaw_offs:
        try:
            yaw_off_f = float(yaw_off)
        except (TypeError, ValueError):
            continue
        desired_closing = base_closing_f + yaw_off_f
        desired_closing = math.atan2(math.sin(desired_closing), math.cos(desired_closing))
        final_tcp_yaw = desired_closing + float(gripper_jaw_axis_offset_rad)
        final_tcp_yaw = math.atan2(math.sin(final_tcp_yaw), math.cos(final_tcp_yaw))

        for depth in depths_f:
            grasp_z = top_z_f - depth
            for axis in center_axes:
                axis_u = _axis_unit(detection, str(axis))
                if axis_u is None:
                    continue
                for off in center_offsets:
                    try:
                        off_f = float(off)
                    except (TypeError, ValueError):
                        continue
                    cx = cx0 + axis_u[0] * off_f
                    cy = cy0 + axis_u[1] * off_f
                    tcp_grasp = (cx, cy, grasp_z)
                    pre_z = max(grasp_z + approach, top_z_f + pre_clr)
                    safe_z = max(top_z_f + safe_clr, pre_z + safe_extra)
                    tcp_pre = (cx, cy, pre_z)
                    tcp_safe = (cx, cy, safe_z)
                    center_xyz = (cx, cy, top_z_f)

                    score = 0.0
                    score += 5.0 if abs(yaw_off_f) < 1e-6 else 0.0
                    score += 4.0 if abs(off_f) < 1e-6 else max(0.0, 3.0 - abs(off_f) * 200.0)
                    med = float(np.median(np.array(depths_f)))
                    score += 3.0 - min(abs(depth - med) * 80.0, 3.0)
                    score += 1.0 if str(axis) == "major_axis" else 0.5

                    notes = "yaw_off=%.3f off=%.4f axis=%s depth=%.3f" % (
                        yaw_off_f,
                        off_f,
                        axis,
                        depth,
                    )
                    candidates.append(
                        GraspCandidate(
                            idx=idx,
                            label=label,
                            strategy=strategy,
                            center_xyz=center_xyz,
                            tcp_grasp_xyz=tcp_grasp,
                            tcp_pregrasp_xyz=tcp_pre,
                            tcp_safe_pregrasp_xyz=tcp_safe,
                            final_tcp_yaw_rad=final_tcp_yaw,
                            desired_closing_yaw_rad=desired_closing,
                            open_joint_m=open_j,
                            close_joint_m=close_j,
                            depth_from_top_m=depth,
                            center_offset_m=off_f,
                            center_offset_axis=str(axis),
                            yaw_offset_rad=yaw_off_f,
                            min_contact_margin_m=min_margin,
                            score=float(score),
                            notes=notes,
                        )
                    )
                    idx += 1

    candidates.sort(key=lambda c: c.score, reverse=True)
    for i, c in enumerate(candidates):
        c.idx = i
    return candidates
