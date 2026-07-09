"""Política de lift cartesiano puro en Z para chips_can tras attach (sin ROS)."""

from __future__ import annotations

import math
from typing import List


def build_chips_can_lift_hand_z_waypoints(
    start_z: float,
    end_z: float,
    *,
    max_step_m: float,
    min_step_m: float = 0.010,
) -> List[float]:
    """Waypoints hand-Z monótonos ascendentes de start_z hasta end_z (inclusive)."""
    z0 = float(start_z)
    z1 = float(end_z)
    if z1 <= z0 + 1e-6:
        return [z1]
    max_step = max(float(min_step_m), float(max_step_m))
    total_rise = z1 - z0
    n_steps = max(1, int(math.ceil(total_rise / max_step)))
    step = total_rise / float(n_steps)
    raw: List[float] = []
    z = z0
    for _ in range(n_steps):
        z += step
        if z > z1:
            z = z1
        raw.append(round(z, 4))
    if not raw or abs(raw[-1] - z1) > 1e-4:
        if raw and raw[-1] < z1 - 1e-4:
            raw.append(round(z1, 4))
        elif not raw:
            raw = [round(z1, 4)]
        else:
            raw[-1] = round(z1, 4)
    out: List[float] = []
    for w in raw:
        if not out or abs(w - out[-1]) > 1e-4:
            out.append(w)
    return out
