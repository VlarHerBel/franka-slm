"""Constantes y listas de objetos soportados por modo de ejecución (TFG Panda)."""

from __future__ import annotations

from typing import FrozenSet, Tuple

# Modo robust_only: objetos con política top-down validada primero en simulación.
ROBUST_TOPDOWN_LABELS: FrozenSet[str] = frozenset(
    {
        "cracker_box",
        "sugar_box",
        "potted_meat_can",
        "mustard_bottle",
        "bleach_cleanser",
        "chips_can",
    }
)

# experimental: añade gelatin_box y otros difíciles intentables con la pinza por defecto.
EXPERIMENTAL_LABELS: FrozenSet[str] = frozenset(
    {
        "apple",
        "banana",
        "gelatin_box",
        "pudding_box",
    }
)

# Objetos de alto riesgo / no top-down robusto con la política actual.
HIGH_RISK_UNSUPPORTED_TOPDOWN: FrozenSet[str] = frozenset(
    {
        "tuna_fish_can",
        "master_chef_can",
    }
)

# Cajas con gate estricto de pose (known_box_center + known_rectangle_fit + umbrales).
STRICT_BOX_POSE_GATE_LABELS: FrozenSet[str] = frozenset(
    {
        "cracker_box",
        "sugar_box",
        "gelatin_box",
        "pudding_box",
        "potted_meat_can",
    }
)


def normalize_label_key(label: str) -> str:
    if not label:
        return ""
    return str(label).strip().lower().replace(" ", "_").replace("-", "_")


def label_allowed_for_supported_mode(label: str, mode: str) -> Tuple[bool, str]:
    """Devuelve (permitido, razón si no permitido)."""
    key = normalize_label_key(label)
    m = (mode or "robust_only").strip().lower()
    if m == "all":
        if key in HIGH_RISK_UNSUPPORTED_TOPDOWN:
            return True, "warning_unsupported_topdown"
        return True, "ok"
    if m == "experimental":
        if key in ROBUST_TOPDOWN_LABELS or key in EXPERIMENTAL_LABELS:
            return True, "ok"
        if key in HIGH_RISK_UNSUPPORTED_TOPDOWN:
            return False, "unsupported_even_in_experimental"
        return False, "not_in_experimental_list"
    # robust_only (default): únicamente los seis objetos validados.
    if key in ROBUST_TOPDOWN_LABELS:
        return True, "ok"
    if key in EXPERIMENTAL_LABELS:
        return False, "experimental_requires_experimental_mode"
    if key in HIGH_RISK_UNSUPPORTED_TOPDOWN:
        return False, "high_risk_unsupported"
    return False, "not_in_robust_list"
