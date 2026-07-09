"""Normalización del visual SDF runtime para seguir la geometría operativa.

La geometría operativa (RuntimeScene, top face, grasp, planning) sale **solo**
de la collision box / ``KnownBoxGtSpec``. Nunca se ajusta al visual roto del
SDF fuente.

Este módulo parchea únicamente el ``<visual>`` en la copia bajo
``/tmp/tfg_runtime_ycb_models/``. Convención: origen del link en el centro de
la base del cuboide; collision en ``(0,0,H/2)``; mesh con origen en la base →
visual runtime ``(0,0,0,0,0,0)`` (sin offsets ni RPY internos del SDF YCB).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

Pose6 = Tuple[float, float, float, float, float, float]

# Mismo conjunto que KNOWN_SPAWN_GEOMETRY_BOX_LABELS (sin importar runtime_scene_gt).
KNOWN_BOX_LABELS_NORMALIZE_VISUAL = frozenset(
    {"cracker_box", "sugar_box", "gelatin_box", "pudding_box"}
)

_COLLISION_POSE_RE = re.compile(
    r'<collision\s+name="collision"[^>]*>\s*<pose>\s*([^<]+?)\s*</pose>',
    re.IGNORECASE | re.DOTALL,
)
_COLLISION_SIZE_RE = re.compile(
    r'<collision\s+name="collision"[^>]*>.*?<box>\s*<size>\s*([^<]+?)\s*</size>',
    re.IGNORECASE | re.DOTALL,
)
_VISUAL_POSE_RE = re.compile(
    r'(<visual\s+name="visual"[^>]*>\s*<pose>)\s*([^<]+?)\s*(</pose>)',
    re.IGNORECASE | re.DOTALL,
)
_MESH_URI_RE = re.compile(
    r'<visual\s+name="visual"[^>]*>.*?<uri>\s*([^<]+?)\s*</uri>',
    re.IGNORECASE | re.DOTALL,
)


def _normalize_label(label: str) -> str:
    return str(label).strip().lower().replace(" ", "_")


def parse_pose6(text: str) -> Pose6:
    parts = [float(x) for x in str(text).strip().split()]
    if len(parts) != 6:
        raise ValueError(f"pose SDF inválida (6 valores): {text!r}")
    return (parts[0], parts[1], parts[2], parts[3], parts[4], parts[5])


def format_pose6(pose: Pose6) -> str:
    return " ".join("%.6g" % float(v) for v in pose)


def collision_pose_from_box_size(size_xyz: Tuple[float, float, float]) -> Pose6:
    """Centro del cuboide de colisión en el link (solo referencia; no es pose visual)."""
    _sx, _sy, sz = (float(size_xyz[0]), float(size_xyz[1]), float(size_xyz[2]))
    return (0.0, 0.0, 0.5 * sz, 0.0, 0.0, 0.0)


def visual_pose_for_operational_box_base_origin() -> Pose6:
    """Pose visual alineada al cuboide operativo con origen de mesh en la base del link."""
    return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)


def model_origin_to_semantic_center_offset(height_m: float) -> Tuple[float, float, float]:
    """Offset link→centro semántico cuando el origen del modelo está en la base."""
    return (0.0, 0.0, 0.5 * float(height_m))


def should_normalize_visual_for_label(label: str) -> bool:
    return _normalize_label(label) in KNOWN_BOX_LABELS_NORMALIZE_VISUAL


@dataclass(frozen=True)
class YcbVisualNormalizationEntry:
    label: str
    expected_collision_size: Tuple[float, float, float]
    collision_pose: Pose6
    original_visual_pose: Pose6
    normalized_visual_pose: Pose6
    notes: str


def _box_entry(
    label: str,
    size: Tuple[float, float, float],
    original_visual: Pose6,
    normalized_visual: Pose6,
    *,
    notes: str,
) -> YcbVisualNormalizationEntry:
    collision_pose = collision_pose_from_box_size(size)
    return YcbVisualNormalizationEntry(
        label=_normalize_label(label),
        expected_collision_size=size,
        collision_pose=collision_pose,
        original_visual_pose=original_visual,
        normalized_visual_pose=normalized_visual,
        notes=notes,
    )


# Visual runtime → cuboide operativo (base en z=0 del link). Collision sin tocar.
_VISUAL_AT_OPERATIONAL_BASE = visual_pose_for_operational_box_base_origin()

YCB_VISUAL_NORMALIZATION: Dict[str, YcbVisualNormalizationEntry] = {
    "cracker_box": _box_entry(
        "cracker_box",
        (0.060, 0.158, 0.210),
        (0.015, 0.015, 0.0, 0.0, 0.0, 0.0),
        _VISUAL_AT_OPERATIONAL_BASE,
        notes="visual follows operational box; remove SDF XY offset and internal RPY",
    ),
    "sugar_box": _box_entry(
        "sugar_box",
        (0.038, 0.089, 0.175),
        (0.0, 0.015, 0.0, 0.0, 0.05, 0.0),
        _VISUAL_AT_OPERATIONAL_BASE,
        notes="visual follows operational box; remove SDF Y offset and pitch",
    ),
    "gelatin_box": _box_entry(
        "gelatin_box",
        (0.073, 0.085, 0.028),
        (0.025, 0.005, 0.0, 0.0, 0.0, -0.2),
        _VISUAL_AT_OPERATIONAL_BASE,
        notes="visual follows operational box; remove SDF XY offset and yaw",
    ),
    "pudding_box": _box_entry(
        "pudding_box",
        (0.110, 0.089, 0.035),
        (-0.01, -0.015, 0.0, 0.0, 0.0, -0.5),
        _VISUAL_AT_OPERATIONAL_BASE,
        notes="visual follows operational box; remove SDF XY offset and yaw",
    ),
}


def get_visual_normalization_entry(label: str) -> Optional[YcbVisualNormalizationEntry]:
    return YCB_VISUAL_NORMALIZATION.get(_normalize_label(label))


def extract_sdf_collision_visual_geometry(sdf_text: str) -> Dict[str, Any]:
    """Lee collision/visual del primer link (modelos YCB estándar)."""
    size_m = _COLLISION_SIZE_RE.search(sdf_text)
    col_pose_m = _COLLISION_POSE_RE.search(sdf_text)
    vis_pose_m = _VISUAL_POSE_RE.search(sdf_text)
    mesh_m = _MESH_URI_RE.search(sdf_text)
    collision_size: Optional[Tuple[float, float, float]] = None
    if size_m:
        parts = [float(x) for x in size_m.group(1).strip().split()]
        if len(parts) == 3:
            collision_size = (parts[0], parts[1], parts[2])
    return {
        "collision_size": collision_size,
        "collision_pose": parse_pose6(col_pose_m.group(1)) if col_pose_m else None,
        "original_visual_pose": parse_pose6(vis_pose_m.group(2)) if vis_pose_m else None,
        "mesh_uri": mesh_m.group(1).strip() if mesh_m else None,
    }


PathLike = Union[str, Path]


def log_ycb_model_sdf_geometry(
    logger: Any,
    *,
    label: str,
    source_sdf: PathLike,
    geom: Optional[Dict[str, Any]] = None,
    sdf_text: Optional[str] = None,
) -> Dict[str, Any]:
    """Emite ``[YCB_MODEL_SDF_GEOMETRY]`` para el SDF fuente."""
    lb = _normalize_label(label)
    path = Path(source_sdf)
    text = sdf_text
    if text is None and path.is_file():
        text = path.read_text(encoding="utf-8", errors="replace")
    info: Dict[str, Any] = {"label": lb, "source_sdf": str(path.resolve())}
    if text:
        parsed = extract_sdf_collision_visual_geometry(text)
        info.update(parsed)
    entry = get_visual_normalization_entry(lb)
    if entry is not None:
        info["table_expected_collision_size"] = list(entry.expected_collision_size)
        info["table_normalized_visual_pose"] = list(entry.normalized_visual_pose)

    if logger is not None:
        try:
            logger.info(
                "[YCB_MODEL_SDF_GEOMETRY] label=%s source_sdf=%s collision_size=%s "
                "collision_pose=%s original_visual_pose=%s mesh_uri=%s"
                % (
                    lb,
                    info["source_sdf"],
                    info.get("collision_size"),
                    info.get("collision_pose"),
                    info.get("original_visual_pose"),
                    info.get("mesh_uri"),
                )
            )
        except Exception:
            pass
    return info


def patch_visual_pose_in_sdf(sdf_text: str, new_visual_pose: Pose6) -> str:
    if not _VISUAL_POSE_RE.search(sdf_text):
        raise ValueError("patch_visual_pose_in_sdf: bloque <visual name=\"visual\"> no encontrado")
    return _VISUAL_POSE_RE.sub(
        r"\g<1>" + format_pose6(new_visual_pose) + r"\g<3>",
        sdf_text,
        count=1,
    )


def normalize_runtime_sdf_visual_to_collision_box(
    sdf_text: str,
    label: str,
    *,
    logger: Any = None,
    runtime_sdf_path: Optional[PathLike] = None,
) -> Tuple[str, bool, Pose6, Pose6]:
    """Parchea solo el visual runtime para seguir el cuboide operativo (collision box).

    No modifica collision/inertial ni ningún campo de RuntimeScene/percepción.
    """
    lb = _normalize_label(label)
    entry = get_visual_normalization_entry(lb)
    if entry is None:
        raise ValueError(f"normalize_runtime_sdf_visual: label sin tabla: {lb}")

    parsed = extract_sdf_collision_visual_geometry(sdf_text)
    old_vis = parsed.get("original_visual_pose") or entry.original_visual_pose
    new_vis = entry.normalized_visual_pose
    out = patch_visual_pose_in_sdf(sdf_text, new_vis)

    if logger is not None:
        try:
            logger.info(
                "[YCB_RUNTIME_VISUAL_NORMALIZE] label=%s runtime_sdf=%s "
                "old_visual_pose=%s new_visual_pose=%s "
                "normalize_visual_to_collision_box=true "
                'note="visual follows operational geometry; RuntimeScene unchanged"'
                % (
                    lb,
                    str(runtime_sdf_path) if runtime_sdf_path else "n/a",
                    list(old_vis),
                    list(new_vis),
                )
            )
        except Exception:
            pass
    return out, True, old_vis, new_vis
