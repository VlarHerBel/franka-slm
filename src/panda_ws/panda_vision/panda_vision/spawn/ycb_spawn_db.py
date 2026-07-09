"""Base de datos de spawn YCB (solo geometría / rutas de modelo; no grasping).

Los valores numéricos de altura y offsets se alinean con ``config/ycb_obb_dataset.yaml``
(``spawn_height_m`` = mitad de altura efectiva sobre la mesa según comentarios del YAML).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

# Objetos requeridos por el TFG (orden fijo para borrado masivo determinista).
REQUIRED_YCB_LABELS: Tuple[str, ...] = (
    "cracker_box",
    "sugar_box",
    "mustard_bottle",
    "chips_can",
    "bleach_cleanser",
    "apple",
    "banana",
    "tuna_fish_can",
    "potted_meat_can",
    "gelatin_box",
    "pudding_box",
    "master_chef_can",
)


@dataclass(frozen=True)
class YcbSpawnRecord:
    label: str
    model_name: str
    height_m: float
    spawn_height_m: float
    spawn_z_offset_m: float
    footprint_width_m: float
    footprint_length_m: float
    """center: z = table_z + height_m/2 + origin_z_offset_m (geométrico).
    bottom: z = table_z + origin_z_offset_m (origen en suelo del modelo).
    runtime_yaml: z = table_z + spawn_height_m + spawn_z_offset_m + epsilon (alineado con runtime_scene_spawner).
    """
    origin_z_mode: str
    origin_z_offset_m: float
    base_roll_rad: float
    base_pitch_rad: float
    base_yaw_rad: float
    upright_by_default: bool
    shape_category: str


def _float(d: Dict[str, Any], key: str, default: float = 0.0) -> float:
    try:
        return float(d.get(key, default))
    except (TypeError, ValueError):
        return default


def load_spawn_records_from_yaml(path: Path) -> Dict[str, YcbSpawnRecord]:
    with path.expanduser().open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    out: Dict[str, YcbSpawnRecord] = {}
    for raw in data.get("classes", []):
        if not raw or not bool(raw.get("enabled", True)):
            continue
        name = str(raw.get("name", "")).strip().lower()
        if not name:
            continue
        h = _float(raw, "height_m", 0.08)
        sh = _float(raw, "spawn_height_m", h * 0.5)
        szo = _float(raw, "spawn_z_offset_m", 0.0)
        fw = _float(raw, "footprint_width_m", raw.get("width_m", 0.06))
        fl = _float(raw, "footprint_length_m", raw.get("length_m", 0.06))
        model_name = str(raw.get("model_name", name)).strip()
        shape = str(raw.get("shape_category", "")).strip().lower() or _guess_shape(name)
        out[name] = YcbSpawnRecord(
            label=name,
            model_name=model_name,
            height_m=h,
            spawn_height_m=sh,
            spawn_z_offset_m=szo,
            footprint_width_m=fw,
            footprint_length_m=fl,
            origin_z_mode="runtime_yaml",
            origin_z_offset_m=0.0,
            base_roll_rad=0.0,
            base_pitch_rad=0.0,
            base_yaw_rad=_float(raw, "visual_yaw_offset_rad", 0.0),
            upright_by_default=True,
            shape_category=shape,
        )
    return out


def _guess_shape(label: str) -> str:
    if "can" in label or "bottle" in label or label in ("apple",):
        if "box" in label:
            return "box"
        return "cylinder_like"
    if "box" in label:
        return "box"
    if label == "banana":
        return "curved"
    return "unknown"


def ensure_all_labels_present(records: Dict[str, YcbSpawnRecord]) -> List[str]:
    missing = [lb for lb in REQUIRED_YCB_LABELS if lb not in records]
    return missing
