"""Semántica gap vs pads para mustard_bottle (pinza paralela).

Opción A:
- closing_yaw_rad = yaw del eje gap (apertura/cierre sobre ancho corto)
- finger_pad_yaw_rad = yaw longitudinal de dedos (paralelo al eje largo)
- object_yaw_rad / grasp_yaw_rad = finger_pad_yaw_rad (no closing_yaw)
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

from panda_vision.grasp.object_grasp_policy import MAX_GRIPPER_WIDTH_M, normalize_label
from panda_vision.spawn.known_object_geometry import is_runtime_gt_tall_cap_center_source

MUSTARD_SHORT_WIDTH_M = 0.058
MUSTARD_LONG_WIDTH_M = 0.095
MUSTARD_AXIS_DOT_SHORT_MIN = 0.98
MUSTARD_AXIS_DOT_LONG_MAX = 0.25
CLOSING_YAW_SEMANTICS_GAP_AXIS = "gap_axis"

def _runtime_gt_axes_trusted(pose_meta: Dict[str, Any]) -> bool:
    """Ejes largo/corto del spawn GT; no exigir alineación con PCA de la máscara."""
    if bool(pose_meta.get("runtime_gt_geometry_applied")):
        return True
    if is_runtime_gt_tall_cap_center_source(pose_meta.get("grasp_center_source")):
        return True
    tfs = str(pose_meta.get("top_face_source") or "").strip()
    return tfs in ("runtime_gt_tall_object", "runtime_gt_known_object")


_VALID_MAPPINGS = frozenset(
    {
        "normal",
        "swap_major_minor",
        "yaw_offset_plus_90",
        "yaw_offset_minus_90",
    }
)


def _wrap_pi(yaw: float) -> float:
    return float((float(yaw) + math.pi) % (2.0 * math.pi) - math.pi)


def _unit_xy(v: Any) -> Optional[Tuple[float, float]]:
    if not isinstance(v, (list, tuple)) or len(v) < 2:
        return None
    x, y = float(v[0]), float(v[1])
    n = math.hypot(x, y)
    if n < 1e-9:
        return None
    return (x / n, y / n)


def _rotate_xy_90(axis: Tuple[float, float], *, sign: int) -> Tuple[float, float]:
    x, y = axis
    if int(sign) >= 0:
        return (-y, x)
    return (y, -x)


def _dot(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return abs(a[0] * b[0] + a[1] * b[1])


def _yaw_from_xy(axis: Tuple[float, float]) -> float:
    return float(math.atan2(axis[1], axis[0]))


def _apply_axis_mapping(
    major_xy: Optional[Tuple[float, float]],
    minor_xy: Optional[Tuple[float, float]],
    mapping: str,
) -> Optional[Dict[str, Any]]:
    if minor_xy is None:
        return None
    mapping = str(mapping).strip().lower()
    if mapping not in _VALID_MAPPINGS:
        mapping = "normal"
    if mapping == "normal":
        gap = minor_xy
        pad = major_xy if major_xy is not None else _rotate_xy_90(minor_xy, sign=1)
        rejected = major_xy
    elif mapping == "swap_major_minor":
        if major_xy is None:
            return None
        gap = major_xy
        pad = minor_xy
        rejected = minor_xy
    elif mapping == "yaw_offset_plus_90":
        gap = _rotate_xy_90(minor_xy, sign=1)
        pad = minor_xy
        rejected = major_xy
    else:
        gap = _rotate_xy_90(minor_xy, sign=-1)
        pad = minor_xy
        rejected = major_xy
    return {
        "mapping": mapping,
        "body_long_axis_xy": list(pad),
        "body_short_axis_xy": list(gap),
        "grasp_gap_axis_xy": list(gap),
        "finger_pad_axis_xy": list(pad),
        "rejected_gap_axis_xy": list(rejected) if rejected is not None else None,
    }


def _write_axes_to_targets(
    targets: List[Dict[str, Any]],
    *,
    long_xy: List[float],
    short_xy: List[float],
    grasp_gap_yaw_rad: float,
    finger_pad_yaw_rad: float,
    closing_yaw_source: str,
    gap_xy: List[float],
    pad_xy: List[float],
    commanded_tcp_yaw_rad: float,
) -> None:
    for tgt in targets:
        if not isinstance(tgt, dict):
            continue
        tgt["major_axis_xy"] = list(long_xy)
        tgt["minor_axis_xy"] = list(short_xy)
        tgt["long_axis_xy"] = list(long_xy)
        tgt["short_axis_xy"] = list(short_xy)
        tgt["model_major_axis_xy"] = list(long_xy)
        tgt["model_minor_axis_xy"] = list(short_xy)
        tgt["grasp_gap_axis_xy"] = list(gap_xy)
        tgt["finger_pad_axis_xy"] = list(pad_xy)
        tgt["body_long_axis_xy"] = list(long_xy)
        tgt["body_short_axis_xy"] = list(short_xy)
        tgt["grasp_gap_yaw_rad"] = float(grasp_gap_yaw_rad)
        tgt["finger_pad_yaw_rad"] = float(finger_pad_yaw_rad)
        tgt["closing_yaw_rad"] = float(grasp_gap_yaw_rad)
        tgt["model_closing_yaw_rad"] = float(grasp_gap_yaw_rad)
        tgt["closing_yaw_semantics"] = CLOSING_YAW_SEMANTICS_GAP_AXIS
        tgt["closing_yaw_source"] = str(closing_yaw_source)
        tgt["grasp_yaw_rad"] = float(finger_pad_yaw_rad)
        if "grasp_yaw_deg" in tgt:
            tgt["grasp_yaw_deg"] = float(math.degrees(finger_pad_yaw_rad))
        tgt["object_yaw_rad"] = float(finger_pad_yaw_rad)
        tgt["commanded_tcp_yaw_rad_hint"] = float(commanded_tcp_yaw_rad)
        fp_maj = float(MUSTARD_LONG_WIDTH_M)
        fp_min = float(MUSTARD_SHORT_WIDTH_M)
        tgt["footprint_major_m"] = fp_maj
        tgt["footprint_minor_m"] = fp_min
        tgt["required_grasp_width_m"] = fp_min
        if "effective_required_grasp_width_m" in tgt:
            tgt["effective_required_grasp_width_m"] = fp_min


def apply_mustard_bottle_axis_semantics(
    *,
    label: str,
    mapping: str,
    pose_meta: Dict[str, Any],
    grasp_fields: Optional[Dict[str, Any]] = None,
    obj_entry: Optional[Dict[str, Any]] = None,
    logger: Any = None,
    local_gap_axis_angle_rad: float = math.pi / 2.0,
) -> Dict[str, Any]:
    """Publica gap/pad separados; closing_yaw = gap, object_yaw = finger_pad."""
    lb = normalize_label(label)
    result: Dict[str, Any] = {
        "label": lb,
        "applied": False,
        "axis_debug_result": "SKIP",
        "width_sanity_result": "SKIP",
        "publish_allowed": True,
    }
    if lb != "mustard_bottle":
        return result

    major_u = _unit_xy(
        pose_meta.get("long_axis_xy")
        or pose_meta.get("major_axis_xy")
        or (grasp_fields or {}).get("major_axis_xy")
    )
    minor_u = _unit_xy(
        pose_meta.get("short_axis_xy")
        or pose_meta.get("minor_axis_xy")
        or (grasp_fields or {}).get("minor_axis_xy")
    )
    mapped = _apply_axis_mapping(major_u, minor_u, mapping)
    if mapped is None:
        result["axis_debug_result"] = "FAIL"
        result["width_sanity_result"] = "FAIL"
        result["publish_allowed"] = False
        result["reason"] = "missing_internal_axes"
        return result

    gap_xy = mapped["grasp_gap_axis_xy"]
    pad_xy = mapped["finger_pad_axis_xy"]
    long_xy = mapped["body_long_axis_xy"]
    short_xy = mapped["body_short_axis_xy"]
    grasp_gap_yaw = _yaw_from_xy((float(gap_xy[0]), float(gap_xy[1])))
    finger_pad_yaw = _yaw_from_xy((float(pad_xy[0]), float(pad_xy[1])))
    commanded_tcp_yaw = _wrap_pi(
        float(grasp_gap_yaw) - float(local_gap_axis_angle_rad)
    )
    closing_source = "runtime_gt_mustard_gap_axis_%s" % str(mapped["mapping"])

    gap_u = _unit_xy(gap_xy)
    dot_short = _dot(gap_u, minor_u) if gap_u and minor_u else 0.0
    dot_long = _dot(gap_u, major_u) if gap_u and major_u else 0.0
    axis_ok = dot_short >= MUSTARD_AXIS_DOT_SHORT_MIN and dot_long <= MUSTARD_AXIS_DOT_LONG_MAX

    pad_u = _unit_xy(pad_xy)
    dot_pad_long = _dot(pad_u, major_u) if pad_u and major_u else 0.0
    dot_pad_short = _dot(pad_u, minor_u) if pad_u and minor_u else 0.0
    pad_ok = dot_pad_long >= MUSTARD_AXIS_DOT_SHORT_MIN and dot_pad_short <= MUSTARD_AXIS_DOT_LONG_MAX
    if _runtime_gt_axes_trusted(pose_meta):
        axis_ok = True
        pad_ok = True

    selected_axis = "short"
    selected_required = float(MUSTARD_SHORT_WIDTH_M)
    if dot_short < MUSTARD_AXIS_DOT_SHORT_MIN and dot_long > MUSTARD_AXIS_DOT_LONG_MAX:
        selected_axis = "long"
        selected_required = float(MUSTARD_LONG_WIDTH_M)
    width_ok = selected_required <= float(MAX_GRIPPER_WIDTH_M) + 1e-6

    targets: List[Dict[str, Any]] = [pose_meta]
    if isinstance(grasp_fields, dict):
        targets.append(grasp_fields)
    if isinstance(obj_entry, dict):
        targets.append(obj_entry)
    _write_axes_to_targets(
        targets,
        long_xy=long_xy,
        short_xy=short_xy,
        grasp_gap_yaw_rad=grasp_gap_yaw,
        finger_pad_yaw_rad=finger_pad_yaw,
        closing_yaw_source=closing_source,
        gap_xy=gap_xy,
        pad_xy=pad_xy,
        commanded_tcp_yaw_rad=commanded_tcp_yaw,
    )
    pose_meta["mustard_axis_mapping"] = str(mapped["mapping"])
    pose_meta["overlay_pad_axis_xy"] = list(pad_xy)
    pose_meta["overlay_gap_axis_xy"] = list(gap_xy)

    publish_ok = bool(axis_ok and pad_ok and width_ok)
    result.update(
        {
            "applied": True,
            "mapping": str(mapped["mapping"]),
            "axis_debug_result": "OK" if axis_ok and pad_ok else "FAIL",
            "width_sanity_result": "OK" if width_ok else "FAIL",
            "publish_allowed": publish_ok,
            "grasp_gap_yaw_rad": grasp_gap_yaw,
            "finger_pad_yaw_rad": finger_pad_yaw,
            "commanded_tcp_yaw_rad_hint": commanded_tcp_yaw,
            "closing_axis_dot_short": dot_short,
            "closing_axis_dot_long": dot_long,
            "pad_axis_dot_long": dot_pad_long,
            "pad_axis_dot_short": dot_pad_short,
            "selected_axis": selected_axis,
            "selected_required_width_m": selected_required,
        }
    )

    if logger is not None:
        try:
            logger.info(
                "[MUSTARD_PERCEPTION_AXIS_DEBUG]\n"
                "closing_yaw_semantics=%s\n"
                "grasp_gap_axis_xy=%s\n"
                "finger_pad_axis_xy=%s\n"
                "grasp_gap_yaw_deg=%.1f\n"
                "finger_pad_yaw_deg=%.1f\n"
                "closing_yaw_rad=%.4f (gap)\n"
                "object_yaw_rad=%.4f (finger_pad)\n"
                "commanded_tcp_yaw_rad_hint=%.4f\n"
                "closing_axis_dot_short=%.4f\n"
                "closing_axis_dot_long=%.4f\n"
                "pad_axis_dot_long=%.4f\n"
                "pad_axis_dot_short=%.4f\n"
                "mustard_bottle_axis_mapping=%s\n"
                "result=%s"
                % (
                    CLOSING_YAW_SEMANTICS_GAP_AXIS,
                    gap_xy,
                    pad_xy,
                    math.degrees(grasp_gap_yaw),
                    math.degrees(finger_pad_yaw),
                    grasp_gap_yaw,
                    finger_pad_yaw,
                    commanded_tcp_yaw,
                    dot_short,
                    dot_long,
                    dot_pad_long,
                    dot_pad_short,
                    str(mapped["mapping"]),
                    result["axis_debug_result"],
                )
            )
            logger.info(
                "[MUSTARD_AXIS_SEMANTICS_FINAL]\n"
                "long_axis_yaw_deg=%.2f\n"
                "short_axis_yaw_deg=%.2f\n"
                "grasp_gap_yaw_deg=%.2f\n"
                "finger_pad_yaw_deg=%.2f\n"
                "closing_yaw_semantics=%s\n"
                "grasp_gap_axis_xy=%s\n"
                "finger_pad_axis_xy=%s\n"
                "result=%s"
                % (
                    math.degrees(_yaw_from_xy(_unit_xy(long_xy) or (0.0, 1.0))),
                    math.degrees(_yaw_from_xy(_unit_xy(short_xy) or (1.0, 0.0))),
                    math.degrees(grasp_gap_yaw),
                    math.degrees(finger_pad_yaw),
                    CLOSING_YAW_SEMANTICS_GAP_AXIS,
                    gap_xy,
                    pad_xy,
                    "OK" if publish_ok else "FAIL",
                )
            )
            width_reason = ""
            if not width_ok:
                width_reason = "selected_axis_too_wide_for_gripper"
            logger.info(
                "[MUSTARD_WIDTH_AXIS_SANITY]\n"
                "selected_axis=%s\n"
                "selected_required_width_m=%.4f\n"
                "max_gripper_width_m=%.4f\n"
                "result=%s\n"
                "reason=%s"
                % (
                    selected_axis,
                    selected_required,
                    float(MAX_GRIPPER_WIDTH_M),
                    result["width_sanity_result"],
                    width_reason,
                )
            )
        except Exception:
            pass
    return result


def log_mustard_overlay_axis_debug(
    pose_meta: Dict[str, Any],
    *,
    orange_axis_xy: Optional[List[float]],
    cyan_axis_xy: Optional[List[float]],
    orange_source: str,
    cyan_source: str,
    logger: Any,
) -> Dict[str, Any]:
    """Valida que naranja=pads y cian=gap en /vision/debug_image."""
    pad_u = _unit_xy(
        pose_meta.get("finger_pad_axis_xy") or pose_meta.get("overlay_pad_axis_xy")
    )
    gap_u = _unit_xy(
        pose_meta.get("grasp_gap_axis_xy") or pose_meta.get("overlay_gap_axis_xy")
    )
    orange_u = _unit_xy(orange_axis_xy)
    cyan_u = _unit_xy(cyan_axis_xy)
    dot_orange_pad = _dot(orange_u, pad_u) if orange_u and pad_u else 0.0
    dot_cyan_gap = _dot(cyan_u, gap_u) if cyan_u and gap_u else 0.0
    ok = (
        dot_orange_pad >= MUSTARD_AXIS_DOT_SHORT_MIN
        and dot_cyan_gap >= MUSTARD_AXIS_DOT_SHORT_MIN
    )
    orange_yaw = (
        math.degrees(_yaw_from_xy(orange_u)) if orange_u is not None else float("nan")
    )
    cyan_yaw = math.degrees(_yaw_from_xy(cyan_u)) if cyan_u is not None else float("nan")
    fp_yaw = pose_meta.get("finger_pad_yaw_rad")
    gg_yaw = pose_meta.get("grasp_gap_yaw_rad")
    out = {
        "result": "OK" if ok else "FAIL",
        "dot_orange_vs_finger_pad": dot_orange_pad,
        "dot_cyan_vs_gap": dot_cyan_gap,
    }
    if logger is not None:
        try:
            logger.info(
                "[MUSTARD_OVERLAY_AXIS_DEBUG]\n"
                "orange_axis_source=%s\n"
                "orange_yaw_deg=%.2f\n"
                "cyan_axis_source=%s\n"
                "cyan_yaw_deg=%.2f\n"
                "payload_finger_pad_yaw_deg=%s\n"
                "payload_grasp_gap_yaw_deg=%s\n"
                "dot_orange_vs_finger_pad=%.4f\n"
                "dot_cyan_vs_gap=%.4f\n"
                "result=%s"
                % (
                    str(orange_source),
                    float(orange_yaw),
                    str(cyan_source),
                    float(cyan_yaw),
                    _fmt(fp_yaw, nd=2),
                    _fmt(gg_yaw, nd=2),
                    dot_orange_pad,
                    dot_cyan_gap,
                    out["result"],
                )
            )
        except Exception:
            pass
    return out
