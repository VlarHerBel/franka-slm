"""Centro XY del tapón para mustard_bottle (tall_object_topdown).

El offset se aplica sobre ejes corregidos de semántica mustard:
- long_axis_xy = finger_pad_axis (eje largo visible)
- short_axis_xy = grasp_gap_axis (eje corto / apertura)
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

from panda_vision.grasp.object_grasp_policy import (
    TALL_OBJECT_CAP_CENTER_CALIBRATED_SOURCE,
    _PROFILE_OVERRIDES,
    normalize_label,
)

CAP_CENTER_SOURCE = TALL_OBJECT_CAP_CENTER_CALIBRATED_SOURCE


def _unit_xy(v: Any) -> Optional[Tuple[float, float]]:
    if not isinstance(v, (list, tuple)) or len(v) < 2:
        return None
    x, y = float(v[0]), float(v[1])
    n = math.hypot(x, y)
    if n < 1e-9:
        return None
    return (x / n, y / n)


def _xy2(v: Any) -> Optional[Tuple[float, float]]:
    if not isinstance(v, (list, tuple)) or len(v) < 2:
        return None
    return (float(v[0]), float(v[1]))


def resolve_mustard_cap_center_offsets(
    *,
    candidate_index: int = -1,
    offset_long_m: Optional[float] = None,
    offset_short_m: Optional[float] = None,
) -> Tuple[float, float, str]:
    """Resuelve (offset_long, offset_short) desde política o candidatos."""
    policy = _PROFILE_OVERRIDES.get("mustard_bottle", {})
    idx = int(candidate_index)
    candidates = policy.get("mustard_cap_center_offset_candidates")
    if idx >= 0 and isinstance(candidates, list) and idx < len(candidates):
        pair = candidates[idx]
        if isinstance(pair, (list, tuple)) and len(pair) >= 2:
            return float(pair[0]), float(pair[1]), "mustard_cap_center_offset_candidates[%d]" % idx
    if candidate_index < 0 and offset_long_m is not None and offset_short_m is not None:
        return float(offset_long_m), float(offset_short_m), "mustard_cap_center_offset_params"
    off_l = float(policy.get("topdown_grasp_center_offset_long_m") or 0.0)
    off_s = float(policy.get("topdown_grasp_center_offset_short_m") or 0.0)
    return off_l, off_s, "topdown_grasp_center_offset_policy"


def compute_mustard_cap_center_xy(
    body_center_xy: Tuple[float, float],
    long_axis_xy: Tuple[float, float],
    short_axis_xy: Tuple[float, float],
    offset_long_m: float,
    offset_short_m: float,
) -> Tuple[float, float]:
    """cap_xy = body + long*u_long*off_long + short*u_short*off_short."""
    lu = _unit_xy(long_axis_xy)
    su = _unit_xy(short_axis_xy)
    if lu is None or su is None:
        return (float(body_center_xy[0]), float(body_center_xy[1]))
    bx, by = float(body_center_xy[0]), float(body_center_xy[1])
    cap_x = bx + lu[0] * float(offset_long_m) + su[0] * float(offset_short_m)
    cap_y = by + lu[1] * float(offset_long_m) + su[1] * float(offset_short_m)
    return (cap_x, cap_y)


def _resolve_body_center_xy(meta: Dict[str, Any]) -> Optional[Tuple[float, float]]:
    for key in (
        "tall_object_body_center_base",
        "known_object_center_base",
        "semantic_box_center_base",
        "known_box_center_base",
    ):
        xy = _xy2(meta.get(key))
        if xy is not None:
            return xy
    return None


def _resolve_pad_and_gap_axes(meta: Dict[str, Any]) -> Tuple[Optional[Tuple[float, float]], Optional[Tuple[float, float]]]:
    pad = _unit_xy(
        meta.get("finger_pad_axis_xy")
        or meta.get("body_long_axis_xy")
        or meta.get("long_axis_xy")
        or meta.get("major_axis_xy")
    )
    gap = _unit_xy(
        meta.get("grasp_gap_axis_xy")
        or meta.get("body_short_axis_xy")
        or meta.get("short_axis_xy")
        or meta.get("minor_axis_xy")
    )
    return pad, gap


def apply_mustard_cap_center_calibration(
    *,
    pose_meta: Dict[str, Any],
    grasp_fields: Optional[Dict[str, Any]] = None,
    center_info: Optional[Dict[str, Any]] = None,
    candidate_index: int = -1,
    offset_long_m: Optional[float] = None,
    offset_short_m: Optional[float] = None,
    logger: Any = None,
) -> Dict[str, Any]:
    """Actualiza grasp_center sobre el tapón calibrado; no modifica ejes/yaw."""
    result: Dict[str, Any] = {
        "applied": False,
        "result": "SKIP",
        "cap_center_source": CAP_CENTER_SOURCE,
    }
    lb = normalize_label(str(pose_meta.get("label", "mustard_bottle")))
    if lb != "mustard_bottle":
        return result

    body_xy = _resolve_body_center_xy(pose_meta)
    if body_xy is None:
        old_g = _xy2(pose_meta.get("grasp_center_base"))
        if old_g is None:
            result["result"] = "FAIL"
            result["reason"] = "missing_body_center"
            return result
        body_xy = old_g

    pad_u, gap_u = _resolve_pad_and_gap_axes(pose_meta)
    if pad_u is None or gap_u is None:
        result["result"] = "FAIL"
        result["reason"] = "missing_pad_or_gap_axis"
        return result

    top_z = pose_meta.get("top_z_m") or pose_meta.get("top_z_estimated")
    if top_z is None and isinstance(grasp_fields, dict):
        top_z = grasp_fields.get("top_z_m")
    try:
        top_z_m = float(top_z if top_z is not None else 0.0)
    except (TypeError, ValueError):
        top_z_m = 0.0

    old_grasp_xy = _xy2(pose_meta.get("grasp_center_base"))
    if old_grasp_xy is None and isinstance(center_info, dict):
        old_grasp_xy = _xy2(center_info.get("chosen_target_center_base"))
    if old_grasp_xy is None:
        old_grasp_xy = body_xy

    off_l, off_s, off_src = resolve_mustard_cap_center_offsets(
        candidate_index=candidate_index,
        offset_long_m=offset_long_m,
        offset_short_m=offset_short_m,
    )
    cap_xy = compute_mustard_cap_center_xy(
        body_xy, pad_u, gap_u, off_l, off_s
    )
    grasp_center = [float(cap_xy[0]), float(cap_xy[1]), float(top_z_m)]

    targets: List[Dict[str, Any]] = [pose_meta]
    if isinstance(grasp_fields, dict):
        targets.append(grasp_fields)
    if isinstance(center_info, dict):
        targets.append(center_info)

    for tgt in targets:
        tgt["grasp_center_base"] = list(grasp_center)
        tgt["grasp_center_source"] = CAP_CENTER_SOURCE
        tgt["top_surface_center_base"] = list(grasp_center)
        if "chosen_target_center_base" in tgt or tgt is center_info:
            tgt["chosen_target_center_base"] = list(grasp_center)

    pose_meta["mustard_cap_center_offset_long_m"] = float(off_l)
    pose_meta["mustard_cap_center_offset_short_m"] = float(off_s)
    pose_meta["mustard_cap_center_offset_source"] = off_src

    result.update(
        {
            "applied": True,
            "result": "OK",
            "body_center_xy": body_xy,
            "old_grasp_center_xy": old_grasp_xy,
            "new_cap_center_xy": cap_xy,
            "long_axis_xy": list(pad_u),
            "short_axis_xy": list(gap_u),
            "offset_long_m": float(off_l),
            "offset_short_m": float(off_s),
            "offset_source": off_src,
            "grasp_center_base": grasp_center,
        }
    )

    if logger is not None:
        try:
            logger.info(
                "[MUSTARD_CAP_CENTER_OFFSET]\n"
                "body_center_xy=(%.4f, %.4f)\n"
                "old_grasp_center_xy=(%.4f, %.4f)\n"
                "new_cap_center_xy=(%.4f, %.4f)\n"
                "long_axis_xy=(%.4f, %.4f)\n"
                "short_axis_xy=(%.4f, %.4f)\n"
                "offset_long_m=%.4f\n"
                "offset_short_m=%.4f\n"
                "offset_source=%s\n"
                "cap_center_source=%s\n"
                "projected_cap_uv=n/a\n"
                "result=OK"
                % (
                    float(body_xy[0]),
                    float(body_xy[1]),
                    float(old_grasp_xy[0]),
                    float(old_grasp_xy[1]),
                    float(cap_xy[0]),
                    float(cap_xy[1]),
                    float(pad_u[0]),
                    float(pad_u[1]),
                    float(gap_u[0]),
                    float(gap_u[1]),
                    float(off_l),
                    float(off_s),
                    off_src,
                    CAP_CENTER_SOURCE,
                )
            )
        except Exception:
            pass

    return result
