"""Resolución de anchura efectiva para checks geométricos del controller."""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple


def _to_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def is_edge_grasp_policy(obj: Dict[str, Any]) -> bool:
    strategy = str(obj.get("grasp_strategy") or obj.get("primary_strategy") or "").strip()
    if bool(obj.get("edge_grasp_requested", False)):
        return True
    if strategy == "edge_grasp":
        return True
    return "edge_grasp" in strategy or bool(obj.get("prefer_edge", False))


def resolve_controller_required_width(obj: Dict[str, Any]) -> Tuple[float, str]:
    """Anchura para aceptar/rechazar grasp (orden de prioridad del payload/policy)."""
    edge = is_edge_grasp_policy(obj)

    eff = _to_float(obj.get("effective_required_grasp_width_m"))
    if eff is not None and eff > 0.0:
        return eff, "effective_required_grasp_width_m"

    req = _to_float(obj.get("required_grasp_width_m"))
    if req is not None and req > 0.0:
        return req, "required_grasp_width_m"

    if not edge:
        db = _to_float(obj.get("db_required_width_m"))
        if db is not None and db > 0.0:
            return db, "db_required_width_m"
        meas = _to_float(obj.get("measured_required_width_m"))
        if meas is not None and meas > 0.0:
            return meas, "measured_required_width_m"

    db = _to_float(obj.get("db_required_width_m"))
    if db is not None and db > 0.0:
        return db, "db_required_width_m(fallback)"
    meas = _to_float(obj.get("measured_required_width_m"))
    if meas is not None and meas > 0.0:
        return meas, "measured_required_width_m(fallback)"

    return 0.0, "unset"


def collision_footprint_xy_from_candidate(
    candidate: Dict[str, Any],
) -> Optional[Tuple[float, float]]:
    """(long_xy, short_xy) desde collision_dims.db_dims si existe."""
    cd = candidate.get("collision_dims")
    if not isinstance(cd, dict):
        return None
    dd = cd.get("db_dims")
    if not isinstance(dd, (list, tuple)) or len(dd) < 2:
        return None
    a = float(dd[0])
    b = float(dd[1])
    return max(a, b), min(a, b)


def compute_yaw_uncertainty_effective_width(
    *,
    policy_width_m: float,
    open_total_m: float,
    yaw_confidence: float,
    edge_grasp_requested: bool,
    grasp_strategy: str,
    collision_db_dims_xy: Optional[Tuple[float, float]] = None,
    min_open_margin_m: float = 0.003,
) -> Tuple[float, float, float, float, bool]:
    """Modelo de anchura con incertidumbre de yaw para el segundo GEOMETRIC_GRASP_CHECK.

    Returns:
        required_width_m, effective_width_m, yaw_uncertainty_rad, open_total_m, success
    """
    edge = edge_grasp_requested or grasp_strategy == "edge_grasp" or "edge_grasp" in (
        grasp_strategy or ""
    )
    w = float(policy_width_m)
    if w <= 0.0:
        return 0.0, 0.0, 0.0, float(open_total_m), False

    if edge:
        short_dim = w
        long_dim = w
    elif collision_db_dims_xy is not None:
        long_dim, short_dim = collision_db_dims_xy
        short_dim = w if w > 0.0 else short_dim
    else:
        short_dim = w
        long_dim = w

    yaw_unc = max(0.0, (1.0 - min(1.0, float(yaw_confidence)))) * (
        12.0 * math.pi / 180.0
    )
    eff_w = short_dim * math.cos(yaw_unc) + long_dim * math.sin(abs(yaw_unc))
    ok = eff_w <= float(open_total_m) - float(min_open_margin_m)
    return w, float(eff_w), float(yaw_unc), float(open_total_m), bool(ok)
