"""Política de release Z en place (food_safe, modos slot) sin dependencias ROS."""

from __future__ import annotations

from typing import Dict, Optional

BOX_FOOD_SAFE_RELEASE_LABELS = frozenset(
    {"cracker_box", "sugar_box", "mustard_bottle", "chips_can"}
)

# Release TCP Z validado en sim para depósito bajo (wall_top=0.17, sin volcar).
DEMO_LOW_PLACE_RELEASE_TCP_Z_BY_LABEL: Dict[str, float] = {
    "cracker_box": 0.180,
    "chips_can": 0.195,
    "sugar_box": 0.200,
    "mustard_bottle": 0.3284,
}

OBJECT_RELEASE_HEIGHT_BY_LABEL: Dict[str, float] = {
    "cracker_box": 0.2100,
    "sugar_box": 0.1750,
    "mustard_bottle": 0.190912205,
    "bleach_cleanser": 0.250476286,
    "gelatin_box": 0.0280,
    "chips_can": 0.2403,
}

ORDERED_PLACE_SLOT_MODES = frozenset({"ordered_near_to_far", "ordered"})

# Barrido legacy de candidatos (insert_low); no usar como release canónico en demo.
LEGACY_PLACE_CANDIDATE_RELEASE_Z_LOW = 0.22


def demo_low_place_release_tcp_z(label: str) -> Optional[float]:
    """Release Z bajo para demo clear_table (evita volcar en borde de caja)."""
    lb = str(label or "").strip().lower()
    if lb in DEMO_LOW_PLACE_RELEASE_TCP_Z_BY_LABEL:
        return float(DEMO_LOW_PLACE_RELEASE_TCP_Z_BY_LABEL[lb])
    return None


def nominal_food_safe_release_tcp_z(
    deposit_box_wall_top_z_m: float,
    object_height_m: float,
    grasp_depth_from_top_m: float,
) -> float:
    """Z TCP release food_safe: top del objeto en caja menos profundidad de grasp."""
    return float(
        float(deposit_box_wall_top_z_m)
        + float(object_height_m)
        - max(0.0, float(grasp_depth_from_top_m))
    )


def resolve_box_release_object_height_m(
    label: str,
    *,
    payload_height_m: Optional[float],
    grasp_depth_from_top_m: Optional[float],
) -> Optional[float]:
    """Altura efectiva para release food_safe en cajas (deposito bajo y cuidadoso).

    La percepción a veces reporta ``object_height_m`` demasiado pequeño (p. ej. 0.013 m).
    Para depositar con cuidado en el borde de la caja (como cracker_box), usamos una
    altura acotada por la profundidad de grasp en lugar de la altura YCB completa.
    """
    lb = str(label or "").strip().lower()
    if lb not in BOX_FOOD_SAFE_RELEASE_LABELS:
        return payload_height_m
    depth = 0.030
    if grasp_depth_from_top_m is not None:
        try:
            depth = max(0.0, float(grasp_depth_from_top_m))
        except (TypeError, ValueError):
            depth = 0.030
    gentle_cap = max(depth + 0.010, 0.020)
    catalog = OBJECT_RELEASE_HEIGHT_BY_LABEL.get(lb)
    if catalog is not None:
        return float(min(float(catalog), gentle_cap))
    return float(gentle_cap)
