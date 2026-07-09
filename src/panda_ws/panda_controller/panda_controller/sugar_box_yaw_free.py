"""Candidatos yaw-free para ruta pick de sugar_box (sin ROS)."""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

YawEntry = Tuple[str, float]


def build_yaw_free_candidate_yaws(
    base_yaw: float,
    *,
    joint7_yaw: Optional[float] = None,
    current_tcp_yaw: Optional[float] = None,
) -> List[YawEntry]:
    """Variantes yaw para IK/plan (commanded TCP yaw, sin corrección física)."""
    base = _wrap_to_pi(float(base_yaw))
    entries: List[YawEntry] = []
    seen: set = set()

    def _add(name: str, yaw_cmd: float) -> None:
        y = _wrap_to_pi(float(yaw_cmd))
        key = round(y, 4)
        if key in seen:
            return
        seen.add(key)
        entries.append((str(name), float(y)))

    _add("commanded_yaw", base)
    _add("commanded_yaw_pi", _wrap_to_pi(base + math.pi))
    _add("top_down_yaw_zero", 0.0)
    _add("top_down_yaw_pi_over_2", math.pi / 2.0)
    _add("top_down_yaw_neg_pi_over_2", -math.pi / 2.0)
    _add("top_down_yaw_pi", math.pi)
    if joint7_yaw is not None:
        _add("yaw_from_current_joint7", float(joint7_yaw))
    if current_tcp_yaw is not None:
        _add("yaw_from_current_tcp", float(current_tcp_yaw))
    return entries


def _wrap_to_pi(angle: float) -> float:
    a = float(angle)
    while a > math.pi:
        a -= 2.0 * math.pi
    while a < -math.pi:
        a += 2.0 * math.pi
    return a


def sugar_pick_route_preference(*, direct_ok: bool, safe_entry_ok: bool) -> str:
    """Orden: probar directo antes que safe_entry."""
    if direct_ok:
        return "direct_pregrasp"
    if safe_entry_ok:
        return "safe_entry"
    return "abort"
