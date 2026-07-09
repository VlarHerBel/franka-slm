"""Muestreo de centro semántico (XY) con validación de footprint en el tablero."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

from panda_vision.spawn.runtime_scene_gt_geometry import (
    get_known_box_gt_spec,
    is_known_spawn_geometry_box_label,
)

DEFAULT_SAFE_X_RANGE = (0.50, 0.66)
DEFAULT_SAFE_Y_RANGE = (-0.10, 0.10)


@dataclass(frozen=True)
class TableSpawnRegion:
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    margin_m: float = 0.03
    random_spawn_safe_region: bool = False
    safe_x_min: float = DEFAULT_SAFE_X_RANGE[0]
    safe_x_max: float = DEFAULT_SAFE_X_RANGE[1]
    safe_y_min: float = DEFAULT_SAFE_Y_RANGE[0]
    safe_y_max: float = DEFAULT_SAFE_Y_RANGE[1]


def resolve_box_dims_lwh(
    label: str,
    *,
    footprint_length_m: Optional[float] = None,
    footprint_width_m: Optional[float] = None,
) -> Tuple[float, float]:
    """Devuelve (length_m, width_m) con length >= width."""
    spec = get_known_box_gt_spec(label)
    if spec is not None:
        length_m, width_m, _ = spec.dims_lwh_m
        return float(length_m), float(width_m)
    fl = float(footprint_length_m or 0.0)
    fw = float(footprint_width_m or 0.0)
    if fl <= 0.0 or fw <= 0.0:
        return 0.158, 0.06
    return (max(fl, fw), min(fl, fw))


def semantic_footprint_corners_xy(
    center_xy: Tuple[float, float],
    yaw_rad: float,
    length_m: float,
    width_m: float,
) -> List[Tuple[float, float]]:
    """Cuatro esquinas del rectángulo L×W centrado en ``center_xy`` (eje largo = yaw)."""
    cx, cy = float(center_xy[0]), float(center_xy[1])
    half_l = 0.5 * float(length_m)
    half_w = 0.5 * float(width_m)
    c = math.cos(float(yaw_rad))
    s = math.sin(float(yaw_rad))
    e_len = (c, s)
    e_wid = (-s, c)
    corners = [
        (
            cx + half_l * e_len[0] + half_w * e_wid[0],
            cy + half_l * e_len[1] + half_w * e_wid[1],
        ),
        (
            cx + half_l * e_len[0] - half_w * e_wid[0],
            cy + half_l * e_len[1] - half_w * e_wid[1],
        ),
        (
            cx - half_l * e_len[0] - half_w * e_wid[0],
            cy - half_l * e_len[1] - half_w * e_wid[1],
        ),
        (
            cx - half_l * e_len[0] + half_w * e_wid[0],
            cy - half_l * e_len[1] + half_w * e_wid[1],
        ),
    ]
    return corners


def semantic_footprint_fits_table(
    corners_xy: List[Tuple[float, float]],
    region: TableSpawnRegion,
) -> Tuple[bool, str]:
    """Todas las esquinas dentro del tablero con margen."""
    m = float(region.margin_m)
    x_lo = float(region.x_min) + m
    x_hi = float(region.x_max) - m
    y_lo = float(region.y_min) + m
    y_hi = float(region.y_max) - m
    if x_lo >= x_hi or y_lo >= y_hi:
        return False, "table_bounds_too_small_for_margin"
    for i, (px, py) in enumerate(corners_xy):
        if px < x_lo or px > x_hi:
            return (
                False,
                "corner[%d]_x=%.4f not in [%.4f,%.4f]"
                % (i, px, x_lo, x_hi),
            )
        if py < y_lo or py > y_hi:
            return (
                False,
                "corner[%d]_y=%.4f not in [%.4f,%.4f]"
                % (i, py, y_lo, y_hi),
            )
    return True, "ok"


def _sample_center_xy(rng: random.Random, region: TableSpawnRegion) -> Tuple[float, float]:
    if region.random_spawn_safe_region:
        cx = rng.uniform(region.safe_x_min, region.safe_x_max)
        cy = rng.uniform(region.safe_y_min, region.safe_y_max)
        return float(cx), float(cy)
    cx = rng.uniform(region.x_min, region.x_max)
    cy = rng.uniform(region.y_min, region.y_max)
    return float(cx), float(cy)


def _log_sample(logger: Any, msg: str) -> None:
    if logger is None:
        return
    try:
        logger.info(msg)
    except Exception:
        pass


def sample_semantic_box_center_xy(
    rng: random.Random,
    label: str,
    yaw_rad: float,
    region: TableSpawnRegion,
    *,
    footprint_length_m: Optional[float] = None,
    footprint_width_m: Optional[float] = None,
    max_attempts: int = 500,
    logger: Any = None,
) -> Tuple[float, float]:
    """Muestra centro semántico hasta que el footprint cabe en el tablero."""
    length_m, width_m = resolve_box_dims_lwh(
        label,
        footprint_length_m=footprint_length_m,
        footprint_width_m=footprint_width_m,
    )
    for attempt in range(1, max(1, int(max_attempts)) + 1):
        cx, cy = _sample_center_xy(rng, region)
        _log_sample(
            logger,
            "[SPAWN_RANDOM_SEMANTIC_SAMPLE] label=%s attempt=%d "
            "semantic_center_xy=(%.4f,%.4f) yaw_deg=%.3f dims_lwh=(%.4f,%.4f) "
            "safe_region=%s margin_m=%.3f"
            % (
                label,
                attempt,
                cx,
                cy,
                math.degrees(yaw_rad),
                length_m,
                width_m,
                str(region.random_spawn_safe_region).lower(),
                region.margin_m,
            ),
        )
        corners = semantic_footprint_corners_xy((cx, cy), yaw_rad, length_m, width_m)
        ok, reason = semantic_footprint_fits_table(corners, region)
        if ok:
            _log_sample(
                logger,
                "[SPAWN_RANDOM_ACCEPT] label=%s semantic_center_xy=(%.4f,%.4f) "
                "yaw_deg=%.3f attempt=%d"
                % (label, cx, cy, math.degrees(yaw_rad), attempt),
            )
            return cx, cy
        _log_sample(
            logger,
            "[SPAWN_RANDOM_REJECT] label=%s semantic_center_xy=(%.4f,%.4f) "
            "yaw_deg=%.3f reason=%s attempt=%d"
            % (label, cx, cy, math.degrees(yaw_rad), reason, attempt),
        )
    raise RuntimeError(
        "No se encontró centro semántico válido para label=%s tras %d intentos"
        % (label, max_attempts)
    )


def sample_semantic_box_pose_xyyaw(
    rng: random.Random,
    label: str,
    region: TableSpawnRegion,
    *,
    footprint_length_m: Optional[float] = None,
    footprint_width_m: Optional[float] = None,
    yaw_min_rad: float = -math.pi,
    yaw_max_rad: float = math.pi,
    max_attempts: int = 500,
    logger: Any = None,
) -> Tuple[float, float, float]:
    """Muestra (cx, cy, yaw) con footprint válido en el tablero."""
    length_m, width_m = resolve_box_dims_lwh(
        label,
        footprint_length_m=footprint_length_m,
        footprint_width_m=footprint_width_m,
    )
    for attempt in range(1, max(1, int(max_attempts)) + 1):
        yaw = rng.uniform(float(yaw_min_rad), float(yaw_max_rad))
        cx, cy = _sample_center_xy(rng, region)
        _log_sample(
            logger,
            "[SPAWN_RANDOM_SEMANTIC_SAMPLE] label=%s attempt=%d "
            "semantic_center_xy=(%.4f,%.4f) yaw_deg=%.3f dims_lwh=(%.4f,%.4f) "
            "safe_region=%s margin_m=%.3f"
            % (
                label,
                attempt,
                cx,
                cy,
                math.degrees(yaw),
                length_m,
                width_m,
                str(region.random_spawn_safe_region).lower(),
                region.margin_m,
            ),
        )
        corners = semantic_footprint_corners_xy((cx, cy), yaw, length_m, width_m)
        ok, reason = semantic_footprint_fits_table(corners, region)
        if ok:
            _log_sample(
                logger,
                "[SPAWN_RANDOM_ACCEPT] label=%s semantic_center_xy=(%.4f,%.4f) "
                "yaw_deg=%.3f attempt=%d"
                % (label, cx, cy, math.degrees(yaw), attempt),
            )
            return cx, cy, yaw
        _log_sample(
            logger,
            "[SPAWN_RANDOM_REJECT] label=%s semantic_center_xy=(%.4f,%.4f) "
            "yaw_deg=%.3f reason=%s attempt=%d"
            % (label, cx, cy, math.degrees(yaw), reason, attempt),
        )
    raise RuntimeError(
        "No se encontró pose semántica válida para label=%s tras %d intentos"
        % (label, max_attempts)
    )


def placement_radius_for_box(length_m: float, width_m: float, safety_scale: float) -> float:
    """Radio conservador para separación entre objetos (layout multi-objeto)."""
    half_diag = 0.5 * math.hypot(float(length_m), float(width_m))
    return half_diag * max(1.0, float(safety_scale))


def is_known_box_label_for_semantic_sampling(label: str) -> bool:
    return is_known_spawn_geometry_box_label(label)
