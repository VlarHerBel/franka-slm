"""Búsqueda acotada XY+Z+yaw para mustard_bottle demo_scene_02 (sin ROS)."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

from panda_controller.grasp_centering_target import resolve_object_geometric_center_xy
from panda_controller.mustard_depth_search import (
    DEMO_SCENE_02_MUSTARD_MIN_GRASP_Z_M,
    DEMO_SCENE_02_MUSTARD_SCANNER_ALIGNED_COMMANDED_YAW_RAD,
    DEMO_SCENE_02_MUSTARD_SCANNER_ALIGNED_GRASP_Z,
    DEMO_SCENE_02_MUSTARD_SCANNER_ALIGNED_PREGRASP_Z,
    build_mustard_scanner_aligned_descend_spec,
    _wrap_to_pi,
)
from panda_controller.mustard_top_z_policy import (
    expected_mustard_grasp_tcp_z_for_runtime_top,
)

# Golden scanner body/scene center (demo_scene_02 spawn pose).
DEMO_SCENE_02_MUSTARD_BODY_CENTER_XY = (0.6600, 0.0600)

MUSTARD_DEMO_PREGRASP_Z_CANDIDATES: Tuple[float, ...] = (
    0.4909,
    0.5000,
    0.5220,
)

MUSTARD_DEMO_GRASP_Z_CANDIDATES: Tuple[float, ...] = (
    0.4269,
    0.4320,
    0.4420,
    0.4520,
)

MUSTARD_AXIS_OFFSETS_M: Tuple[float, ...] = (0.005, 0.010, 0.015)
MUSTARD_INTERPOLATION_FRACTIONS: Tuple[float, ...] = (0.25, 0.50, 0.75)

MUSTARD_XY_SOURCE_PRIORITY: Dict[str, int] = {
    "cap_center": 0,
    "body_center": 1,
    "interpolated": 2,
    "axis_offset_major": 3,
    "axis_offset_minor": 4,
}


def _xy_tuple(raw: Any) -> Optional[Tuple[float, float]]:
    if not isinstance(raw, (list, tuple)) or len(raw) < 2:
        return None
    try:
        return (float(raw[0]), float(raw[1]))
    except (TypeError, ValueError):
        return None


def _axis_unit(raw: Any) -> Optional[Tuple[float, float]]:
    xy = _xy_tuple(raw)
    if xy is None:
        return None
    norm = math.hypot(float(xy[0]), float(xy[1]))
    if norm < 1e-9:
        return None
    return (float(xy[0]) / norm, float(xy[1]) / norm)


def resolve_mustard_cap_center_xy(
    candidate: Dict[str, Any],
    *,
    controller_grasp_xy: Tuple[float, float],
) -> Tuple[float, float]:
    for key in (
        "grasp_center_base",
        "chosen_target_center_base",
        "_runtime_grasp_tcp_xy",
    ):
        xy = _xy_tuple(candidate.get(key))
        if xy is not None:
            return xy
    return (float(controller_grasp_xy[0]), float(controller_grasp_xy[1]))


def resolve_mustard_body_center_xy(
    candidate: Dict[str, Any],
) -> Tuple[float, float]:
    geom = resolve_object_geometric_center_xy(candidate)
    if geom is not None:
        return (float(geom[0]), float(geom[1]))
    for key in ("known_box_center_base", "position", "model_box_center_base"):
        xy = _xy_tuple(candidate.get(key))
        if xy is not None:
            return xy
    return (
        float(DEMO_SCENE_02_MUSTARD_BODY_CENTER_XY[0]),
        float(DEMO_SCENE_02_MUSTARD_BODY_CENTER_XY[1]),
    )


def resolve_mustard_axis_units(
    candidate: Dict[str, Any],
) -> Tuple[Optional[Tuple[float, float]], Optional[Tuple[float, float]]]:
    major = None
    minor = None
    for key in ("major_axis_xy", "model_major_axis_xy", "long_axis_xy"):
        major = _axis_unit(candidate.get(key))
        if major is not None:
            break
    for key in ("minor_axis_xy", "model_minor_axis_xy", "short_axis_xy"):
        minor = _axis_unit(candidate.get(key))
        if minor is not None:
            break
    return major, minor


def mustard_xy_error_to_cap_m(
    grasp_xy: Tuple[float, float],
    cap_center_xy: Tuple[float, float],
) -> float:
    return float(
        math.hypot(
            float(grasp_xy[0]) - float(cap_center_xy[0]),
            float(grasp_xy[1]) - float(cap_center_xy[1]),
        )
    )


def mustard_xy_error_acceptable(
    *,
    xy_source: str,
    xy_error_to_cap_m: float,
    max_cap_xy_m: float = 0.015,
) -> bool:
    if str(xy_source) == "body_center":
        return True
    return float(xy_error_to_cap_m) <= float(max_cap_xy_m) + 1e-6


def build_mustard_yaw_candidates(
    *,
    controller_yaw_rad: float,
) -> List[float]:
    bases = (
        float(DEMO_SCENE_02_MUSTARD_SCANNER_ALIGNED_COMMANDED_YAW_RAD),
        float(controller_yaw_rad),
    )
    seen: set[float] = set()
    out: List[float] = []
    for base in bases:
        for delta in (0.0, math.pi):
            yaw = _wrap_to_pi(float(base) + float(delta))
            key = round(yaw, 5)
            if key in seen:
                continue
            seen.add(key)
            out.append(yaw)
    return out


def _lerp_xy(
    a: Tuple[float, float], b: Tuple[float, float], t: float
) -> Tuple[float, float]:
    u = float(t)
    return (
        float(a[0]) + u * (float(b[0]) - float(a[0])),
        float(a[1]) + u * (float(b[1]) - float(a[1])),
    )


def build_mustard_xy_anchor_points(
    *,
    cap_center_xy: Tuple[float, float],
    body_center_xy: Tuple[float, float],
    major_axis: Optional[Tuple[float, float]],
    minor_axis: Optional[Tuple[float, float]],
) -> List[Tuple[str, Tuple[float, float]]]:
    anchors: List[Tuple[str, Tuple[float, float]]] = [
        ("cap_center", cap_center_xy),
        ("body_center", body_center_xy),
    ]
    for frac in MUSTARD_INTERPOLATION_FRACTIONS:
        anchors.append(
            (
                "interpolated",
                _lerp_xy(body_center_xy, cap_center_xy, float(frac)),
            )
        )
    base = cap_center_xy
    for axis, tag in (
        (major_axis, "axis_offset_major"),
        (minor_axis, "axis_offset_minor"),
    ):
        if axis is None:
            continue
        ax = (float(axis[0]), float(axis[1]))
        for off in MUSTARD_AXIS_OFFSETS_M:
            d = float(off)
            anchors.append((tag, (base[0] + ax[0] * d, base[1] + ax[1] * d)))
            anchors.append((tag, (base[0] - ax[0] * d, base[1] - ax[1] * d)))
    deduped: List[Tuple[str, Tuple[float, float]]] = []
    seen: set[Tuple[int, int]] = set()
    for source, xy in anchors:
        key = (round(float(xy[0]), 4), round(float(xy[1]), 4))
        if key in seen:
            continue
        seen.add(key)
        deduped.append((str(source), (float(xy[0]), float(xy[1]))))
    return deduped


def _z_pair_priority(pre_z: float, gr_z: float) -> int:
    scanner_pair = (
        abs(float(pre_z) - DEMO_SCENE_02_MUSTARD_SCANNER_ALIGNED_PREGRASP_Z) < 1e-4
        and abs(float(gr_z) - DEMO_SCENE_02_MUSTARD_SCANNER_ALIGNED_GRASP_Z) < 1e-4
    )
    if scanner_pair:
        return 0
    return int(round((float(pre_z) - float(gr_z)) * 1000.0))


def build_mustard_demo_xy_reachability_specs(
    candidate: Dict[str, Any],
    *,
    controller_grasp_xy: Tuple[float, float],
    controller_yaw_rad: float,
    effective_top_z_m: float,
    min_grasp_z_m: float,
) -> List[Dict[str, Any]]:
    cap_xy = resolve_mustard_cap_center_xy(
        candidate, controller_grasp_xy=controller_grasp_xy
    )
    body_xy = resolve_mustard_body_center_xy(candidate)
    major, minor = resolve_mustard_axis_units(candidate)
    anchors = build_mustard_xy_anchor_points(
        cap_center_xy=cap_xy,
        body_center_xy=body_xy,
        major_axis=major,
        minor_axis=minor,
    )
    yaws = build_mustard_yaw_candidates(controller_yaw_rad=float(controller_yaw_rad))
    floor_z = max(
        float(min_grasp_z_m),
        float(DEMO_SCENE_02_MUSTARD_MIN_GRASP_Z_M),
    )
    eff_top = float(effective_top_z_m)
    runtime_gr_z = round(
        float(expected_mustard_grasp_tcp_z_for_runtime_top(eff_top)), 4
    )
    grasp_z_candidates: List[float] = list(MUSTARD_DEMO_GRASP_Z_CANDIDATES)
    for extra in (runtime_gr_z,):
        if all(abs(float(extra) - float(z)) > 1e-4 for z in grasp_z_candidates):
            grasp_z_candidates.append(float(extra))
    grasp_tcp_z_nom = None
    raw_gr_z = candidate.get("grasp_tcp_z")
    if raw_gr_z is None:
        raw_gr_z = candidate.get("_runtime_grasp_tcp_z")
    try:
        if raw_gr_z is not None:
            grasp_tcp_z_nom = round(float(raw_gr_z), 4)
    except (TypeError, ValueError):
        grasp_tcp_z_nom = None
    if grasp_tcp_z_nom is not None and all(
        abs(float(grasp_tcp_z_nom) - float(z)) > 1e-4 for z in grasp_z_candidates
    ):
        grasp_z_candidates.append(float(grasp_tcp_z_nom))

    specs: List[Dict[str, Any]] = []
    seen: set[Tuple[Any, ...]] = set()

    for xy_source, grasp_xy in anchors:
        for pre_z in MUSTARD_DEMO_PREGRASP_Z_CANDIDATES:
            pre_z_f = float(pre_z)
            for gr_z in grasp_z_candidates:
                gr_z_f = float(gr_z)
                if gr_z_f + 1e-6 < floor_z:
                    continue
                if pre_z_f <= gr_z_f + 1e-4:
                    continue
                for yaw in yaws:
                    key = (
                        round(grasp_xy[0], 4),
                        round(grasp_xy[1], 4),
                        round(pre_z_f, 4),
                        round(gr_z_f, 4),
                        round(float(yaw), 5),
                    )
                    if key in seen:
                        continue
                    seen.add(key)
                    specs.append(
                        {
                            "xy_source": str(xy_source),
                            "grasp_xy": grasp_xy,
                            "cap_center_xy": cap_xy,
                            "body_center_xy": body_xy,
                            "pregrasp_tcp": (
                                float(grasp_xy[0]),
                                float(grasp_xy[1]),
                                pre_z_f,
                            ),
                            "grasp_tcp": (
                                float(grasp_xy[0]),
                                float(grasp_xy[1]),
                                gr_z_f,
                            ),
                            "depth_from_top_m": eff_top - gr_z_f,
                            "descend_delta_m": pre_z_f - gr_z_f,
                            "commanded_tcp_yaw_rad": float(yaw),
                            "source": "xy_reachability",
                            "endpoint_ik_source": "xy_reachability",
                            "scanner_aligned": (
                                str(xy_source) == "cap_center"
                                and abs(pre_z_f - DEMO_SCENE_02_MUSTARD_SCANNER_ALIGNED_PREGRASP_Z)
                                < 1e-4
                                and abs(gr_z_f - DEMO_SCENE_02_MUSTARD_SCANNER_ALIGNED_GRASP_Z)
                                < 1e-4
                                and abs(
                                    float(yaw)
                                    - DEMO_SCENE_02_MUSTARD_SCANNER_ALIGNED_COMMANDED_YAW_RAD
                                )
                                < 1e-3
                            ),
                        }
                    )

    def _sort_key(spec: Dict[str, Any]) -> Tuple[Any, ...]:
        xy_src = str(spec.get("xy_source") or "")
        xy_rank = int(MUSTARD_XY_SOURCE_PRIORITY.get(xy_src, 99))
        pre_z = float(spec["pregrasp_tcp"][2])
        gr_z = float(spec["grasp_tcp"][2])
        yaw = float(spec.get("commanded_tcp_yaw_rad") or 0.0)
        scanner_yaw = abs(
            yaw - float(DEMO_SCENE_02_MUSTARD_SCANNER_ALIGNED_COMMANDED_YAW_RAD)
        ) < 1e-3
        return (
            xy_rank,
            _z_pair_priority(pre_z, gr_z),
            0 if scanner_yaw else 1,
            -float(gr_z),
        )

    specs.sort(key=_sort_key)
    return specs


def mustard_operational_xy_search_active(scene_id: str) -> bool:
    return str(scene_id or "").strip().lower().startswith("chips_mustard")


def build_mustard_operational_xy_reachability_specs(
    candidate: Dict[str, Any],
    *,
    controller_grasp_xy: Tuple[float, float],
    controller_yaw_rad: float,
    controller_pre_plan: Tuple[float, float, float],
    controller_gr_plan: Tuple[float, float, float],
    effective_top_z_m: float,
    min_grasp_z_m: float,
    min_depth_from_top_m: float = 0.040,
) -> List[Dict[str, Any]]:
    """Pocos candidatos (cap_center + yaw/percepción) para chips_mustard_*."""
    cap_xy = resolve_mustard_cap_center_xy(
        candidate, controller_grasp_xy=controller_grasp_xy
    )
    eff_top = float(effective_top_z_m)
    floor_z = max(
        float(min_grasp_z_m),
        float(DEMO_SCENE_02_MUSTARD_MIN_GRASP_Z_M),
    )
    pre_z = float(controller_pre_plan[2])
    ctrl_gr_z = float(controller_gr_plan[2])
    runtime_gr_z = round(
        float(expected_mustard_grasp_tcp_z_for_runtime_top(eff_top)), 4
    )
    deep_gr_z = round(float(eff_top) - float(min_depth_from_top_m), 4)
    deep_extra_gr_z = round(float(eff_top) - float(min_depth_from_top_m) - 0.006, 4)
    grasp_z_candidates: List[float] = []
    for z in (deep_extra_gr_z, deep_gr_z, runtime_gr_z, ctrl_gr_z):
        if float(z) + 1e-6 < floor_z:
            continue
        if all(abs(float(z) - float(existing)) > 1e-4 for existing in grasp_z_candidates):
            grasp_z_candidates.append(float(z))

    yaw = float(controller_yaw_rad)
    specs: List[Dict[str, Any]] = []
    for gr_z_f in grasp_z_candidates:
        pre_z_f = float(pre_z)
        if pre_z_f <= gr_z_f + 0.050:
            pre_z_f = float(gr_z_f) + 0.075
        specs.append(
            {
                "xy_source": "cap_center",
                "grasp_xy": cap_xy,
                "cap_center_xy": cap_xy,
                "body_center_xy": cap_xy,
                "pregrasp_tcp": (float(cap_xy[0]), float(cap_xy[1]), pre_z_f),
                "grasp_tcp": (float(cap_xy[0]), float(cap_xy[1]), gr_z_f),
                "depth_from_top_m": eff_top - gr_z_f,
                "descend_delta_m": pre_z_f - gr_z_f,
                "commanded_tcp_yaw_rad": yaw,
                "source": "operational_xy",
                "endpoint_ik_source": "operational_xy",
                "scanner_aligned": False,
            }
        )
    specs.sort(key=lambda s: -float(s.get("depth_from_top_m", 0.0)))
    return specs


def build_mustard_scanner_locked_demo_xy_specs(
    candidate: Dict[str, Any],
    *,
    controller_grasp_xy: Tuple[float, float],
    controller_yaw_rad: float,
    controller_pre_plan: Tuple[float, float, float],
    controller_gr_plan: Tuple[float, float, float],
    effective_top_z_m: float,
    min_grasp_z_m: float,
) -> List[Dict[str, Any]]:
    """Solo cap_center + par scanner cuando el contrato bloquea pregrasp (evita grid XY)."""
    cap_xy = resolve_mustard_cap_center_xy(
        candidate, controller_grasp_xy=controller_grasp_xy
    )
    body_xy = resolve_mustard_body_center_xy(candidate)
    floor_z = max(
        float(min_grasp_z_m),
        float(DEMO_SCENE_02_MUSTARD_MIN_GRASP_Z_M),
    )
    eff_top = float(effective_top_z_m)
    specs: List[Dict[str, Any]] = []
    seen: set[Tuple[Any, ...]] = set()

    def _append_spec(spec: Dict[str, Any]) -> None:
        gr_raw = spec.get("grasp_tcp")
        pre_raw = spec.get("pregrasp_tcp")
        if not isinstance(gr_raw, (list, tuple)) or len(gr_raw) < 3:
            return
        if not isinstance(pre_raw, (list, tuple)) or len(pre_raw) < 3:
            return
        gr_z_f = float(gr_raw[2])
        pre_z_f = float(pre_raw[2])
        if gr_z_f + 1e-6 < floor_z or pre_z_f <= gr_z_f + 1e-4:
            return
        yaw = float(
            spec.get("commanded_tcp_yaw_rad")
            or DEMO_SCENE_02_MUSTARD_SCANNER_ALIGNED_COMMANDED_YAW_RAD
        )
        key = (
            round(float(cap_xy[0]), 4),
            round(float(cap_xy[1]), 4),
            round(pre_z_f, 4),
            round(gr_z_f, 4),
            round(yaw, 5),
        )
        if key in seen:
            return
        seen.add(key)
        enriched = dict(spec)
        enriched.setdefault("xy_source", "cap_center")
        enriched.setdefault("grasp_xy", cap_xy)
        enriched.setdefault("cap_center_xy", cap_xy)
        enriched.setdefault("body_center_xy", body_xy)
        enriched.setdefault("commanded_tcp_yaw_rad", yaw)
        enriched.setdefault("endpoint_ik_source", spec.get("source") or "cap_center")
        specs.append(enriched)

    scanner = build_mustard_scanner_aligned_descend_spec(
        xy=cap_xy,
        pre_plan=controller_pre_plan,
        top_z_m=eff_top,
        effective_top_z_m=eff_top,
    )
    if scanner is not None:
        _append_spec(scanner)

    ctrl_pre_z = float(controller_pre_plan[2])
    ctrl_gr_z = float(controller_gr_plan[2])
    for pre_z_f, gr_z_f in (
        (ctrl_pre_z, ctrl_gr_z),
        (
            float(DEMO_SCENE_02_MUSTARD_SCANNER_ALIGNED_PREGRASP_Z),
            float(DEMO_SCENE_02_MUSTARD_SCANNER_ALIGNED_GRASP_Z),
        ),
    ):
        for yaw in (
            float(controller_yaw_rad),
            float(DEMO_SCENE_02_MUSTARD_SCANNER_ALIGNED_COMMANDED_YAW_RAD),
        ):
            _append_spec(
                {
                    "pregrasp_tcp": (float(cap_xy[0]), float(cap_xy[1]), pre_z_f),
                    "grasp_tcp": (float(cap_xy[0]), float(cap_xy[1]), gr_z_f),
                    "depth_from_top_m": eff_top - gr_z_f,
                    "descend_delta_m": pre_z_f - gr_z_f,
                    "commanded_tcp_yaw_rad": float(yaw),
                    "source": "cap_center",
                    "scanner_aligned": (
                        abs(pre_z_f - DEMO_SCENE_02_MUSTARD_SCANNER_ALIGNED_PREGRASP_Z)
                        < 1e-4
                        and abs(gr_z_f - DEMO_SCENE_02_MUSTARD_SCANNER_ALIGNED_GRASP_Z)
                        < 1e-4
                        and abs(
                            float(yaw)
                            - DEMO_SCENE_02_MUSTARD_SCANNER_ALIGNED_COMMANDED_YAW_RAD
                        )
                        < 1e-3
                    ),
                }
            )

    def _sort_key(spec: Dict[str, Any]) -> Tuple[Any, ...]:
        pre_z = float(spec["pregrasp_tcp"][2])
        gr_z = float(spec["grasp_tcp"][2])
        yaw = float(spec.get("commanded_tcp_yaw_rad") or 0.0)
        scanner_yaw = abs(
            yaw - float(DEMO_SCENE_02_MUSTARD_SCANNER_ALIGNED_COMMANDED_YAW_RAD)
        ) < 1e-3
        return (
            0 if bool(spec.get("scanner_aligned")) else 1,
            _z_pair_priority(pre_z, gr_z),
            0 if scanner_yaw else 1,
            -float(gr_z),
        )

    specs.sort(key=_sort_key)
    return specs


def format_mustard_xy_reachability_candidate_log(fields: Dict[str, Any]) -> str:
    grasp_xy = fields.get("grasp_xy") or (0.0, 0.0)
    cap_xy = fields.get("cap_center_xy") or (0.0, 0.0)
    body_xy = fields.get("body_center_xy") or (0.0, 0.0)
    if isinstance(grasp_xy, (list, tuple)) and len(grasp_xy) >= 2:
        gx, gy = float(grasp_xy[0]), float(grasp_xy[1])
    else:
        gx, gy = 0.0, 0.0
    if isinstance(cap_xy, (list, tuple)) and len(cap_xy) >= 2:
        cx, cy = float(cap_xy[0]), float(cap_xy[1])
    else:
        cx, cy = 0.0, 0.0
    if isinstance(body_xy, (list, tuple)) and len(body_xy) >= 2:
        bx, by = float(body_xy[0]), float(body_xy[1])
    else:
        bx, by = 0.0, 0.0
    return (
        "[MUSTARD_XY_REACHABILITY_CANDIDATE]\n"
        "source=%s\n"
        "grasp_xy=(%.4f, %.4f)\n"
        "cap_center_xy=(%.4f, %.4f)\n"
        "body_center_xy=(%.4f, %.4f)\n"
        "xy_error_to_cap_m=%.4f\n"
        "pregrasp_tcp_z=%.4f\n"
        "grasp_tcp_z=%.4f\n"
        "commanded_tcp_yaw_rad=%.6f\n"
        "endpoint_ik_ok=%s\n"
        "stepwise_descend_ok=%s\n"
        "result=%s"
        % (
            str(fields.get("source") or "n/a"),
            gx,
            gy,
            cx,
            cy,
            bx,
            by,
            float(fields.get("xy_error_to_cap_m", 0.0)),
            float(fields.get("pregrasp_tcp_z", 0.0)),
            float(fields.get("grasp_tcp_z", 0.0)),
            float(fields.get("commanded_tcp_yaw_rad", 0.0)),
            str(fields.get("endpoint_ik_ok", "n/a")),
            str(fields.get("stepwise_descend_ok", "n/a")),
            str(fields.get("result") or "n/a"),
        )
    )


def format_mustard_grasp_candidate_selected_log(fields: Dict[str, Any]) -> str:
    grasp_xy = fields.get("grasp_xy") or (0.0, 0.0)
    if isinstance(grasp_xy, (list, tuple)) and len(grasp_xy) >= 2:
        gx, gy = float(grasp_xy[0]), float(grasp_xy[1])
    else:
        gx, gy = 0.0, 0.0
    return (
        "[MUSTARD_GRASP_CANDIDATE_SELECTED]\n"
        "source=%s\n"
        "grasp_xy=(%.4f, %.4f)\n"
        "grasp_tcp_z=%.4f\n"
        "commanded_tcp_yaw_rad=%.6f\n"
        "result=%s"
        % (
            str(fields.get("source") or "n/a"),
            gx,
            gy,
            float(fields.get("grasp_tcp_z", 0.0)),
            float(fields.get("commanded_tcp_yaw_rad", 0.0)),
            str(fields.get("result") or "OK"),
        )
    )
