"""Contrato geométrico runtime: centro semántico del cuboide + adaptación Gazebo.

La geometría operativa se define solo con la collision box (``KnownBoxGtSpec``).
Los offsets/rotaciones del visual del SDF fuente no entran aquí; el visual runtime
se normaliza aparte en ``ycb_visual_normalization``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from panda_vision.grasp.object_grasp_policy import normalize_label

SOURCE_POSE_SEMANTICS_GEOMETRY_CENTER = "geometry_center"

KNOWN_SPAWN_GEOMETRY_BOX_LABELS = frozenset(
    {"cracker_box", "sugar_box", "gelatin_box", "pudding_box"}
)


@dataclass(frozen=True)
class KnownBoxGtSpec:
    """Dimensiones y offset link Gazebo (solo spawn / geometría operativa)."""

    dims_xyz_m: Tuple[float, float, float]
    height_m: float
    local_length_axis: str
    local_width_axis: str
    yaw_offset_rad: float
    model_origin_to_geometry_center_offset_xyz: Tuple[float, float, float]
    notes: str
    # Solo overlay /vision/debug_image (recorte por eje local; no afecta grasp/planning).
    overlay_top_face_inset_length_m: float = 0.0
    overlay_top_face_inset_width_m: float = 0.0

    @property
    def dims_lwh_m(self) -> Tuple[float, float, float]:
        sx, sy, sz = self.dims_xyz_m
        if self.local_length_axis == "y":
            return float(sy), float(sx), float(sz)
        if self.local_length_axis == "x":
            return float(sx), float(sy), float(sz)
        raise ValueError(f"local_length_axis inválido: {self.local_length_axis}")


KNOWN_BOX_GT_SPECS: Dict[str, KnownBoxGtSpec] = {
    "cracker_box": KnownBoxGtSpec(
        dims_xyz_m=(0.060, 0.158, 0.210),
        height_m=0.210,
        local_length_axis="y",
        local_width_axis="x",
        yaw_offset_rad=0.0,
        model_origin_to_geometry_center_offset_xyz=(0.0, 0.0, 0.105),
        notes="SDF: origen en base; centro semántico = origen + (0,0,H/2) desde collision box.",
        overlay_top_face_inset_length_m=0.0,
        overlay_top_face_inset_width_m=0.0,
    ),
    "sugar_box": KnownBoxGtSpec(
        dims_xyz_m=(0.038, 0.089, 0.175),
        height_m=0.175,
        local_length_axis="y",
        local_width_axis="x",
        yaw_offset_rad=0.0,
        model_origin_to_geometry_center_offset_xyz=(0.0, 0.0, 0.0875),
        notes="SDF collision center (0,0,h/2) en link.",
        overlay_top_face_inset_length_m=0.0,
        overlay_top_face_inset_width_m=0.0,
    ),
    "gelatin_box": KnownBoxGtSpec(
        dims_xyz_m=(0.073, 0.085, 0.028),
        height_m=0.028,
        local_length_axis="y",
        local_width_axis="x",
        yaw_offset_rad=0.0,
        model_origin_to_geometry_center_offset_xyz=(0.0, 0.0, 0.014),
        notes="SDF collision center (0,0,h/2) en link.",
        overlay_top_face_inset_length_m=0.0,
        overlay_top_face_inset_width_m=0.0,
    ),
    "pudding_box": KnownBoxGtSpec(
        dims_xyz_m=(0.110, 0.089, 0.035),
        height_m=0.035,
        local_length_axis="x",
        local_width_axis="y",
        yaw_offset_rad=0.0,
        model_origin_to_geometry_center_offset_xyz=(0.0, 0.0, 0.0175),
        notes="SDF collision center (0,0,h/2); longitud en eje X local.",
        overlay_top_face_inset_length_m=0.0,
        overlay_top_face_inset_width_m=0.0,
    ),
    "chips_can": KnownBoxGtSpec(
        dims_xyz_m=(0.075, 0.075, 0.250),
        height_m=0.250,
        local_length_axis="x",
        local_width_axis="y",
        yaw_offset_rad=0.0,
        model_origin_to_geometry_center_offset_xyz=(0.0, 0.0, 0.125),
        notes="YCB chips_can: cilindro; origen Gazebo en base; centro geométrico en z=H/2.",
        overlay_top_face_inset_length_m=0.0,
        overlay_top_face_inset_width_m=0.0,
    ),
}


def is_known_spawn_geometry_box_label(label: str) -> bool:
    return normalize_label(str(label).strip().lower()) in KNOWN_SPAWN_GEOMETRY_BOX_LABELS


def get_known_box_gt_spec(label: str) -> Optional[KnownBoxGtSpec]:
    return KNOWN_BOX_GT_SPECS.get(normalize_label(str(label).strip().lower()))


def log_synthetic_top_face_overlay_projection(
    logger: Any,
    *,
    label: str,
    synth: Dict[str, Any],
    projected_pixels: List[Tuple[int, int]],
) -> None:
    """Completa ``[SYNTHETIC_TOP_FACE_DEBUG]`` tras proyección pinhole en perception_node."""
    spec = get_known_box_gt_spec(label)
    dims_col = synth.get("dims_collision_lwh") or synth.get("dims_used_lwh") or [0, 0, 0]
    dims_used = synth.get("dims_used_lwh") or dims_col
    _log_synthetic_top_face_debug(
        logger,
        label=label,
        spec=spec,
        dims_collision_lwh=(
            float(dims_col[0]),
            float(dims_col[1]),
            float(dims_col[2]),
        ),
        overlay_length=float(dims_used[0]),
        overlay_width=float(dims_used[1]),
        top_face_corners_base=list(synth.get("top_face_corners_base") or []),
        for_overlay=True,
        projected_pixels=projected_pixels,
        projection_note="pinhole base->camera TF + CameraInfo intrinsics",
    )


def resolve_top_face_dims_lwh(
    spec: KnownBoxGtSpec,
    *,
    for_overlay: bool = False,
) -> Tuple[float, float, float, str]:
    """L,W,H para cara superior; operativa = collision; overlay recorta por eje local."""
    length_m, width_m, height_m = spec.dims_lwh_m
    source = "KnownBoxGtSpec.dims_lwh_m"
    if for_overlay:
        ins_l = float(spec.overlay_top_face_inset_length_m)
        ins_w = float(spec.overlay_top_face_inset_width_m)
        if ins_l > 0.0:
            length_m = max(float(length_m) - 2.0 * ins_l, 0.01)
        if ins_w > 0.0:
            width_m = max(float(width_m) - 2.0 * ins_w, 0.01)
        if ins_l > 0.0 or ins_w > 0.0:
            source = (
                "KnownBoxGtSpec.dims_lwh_m minus overlay insets "
                "(length/local_length_axis, width/local_width_axis)"
            )
    return float(length_m), float(width_m), float(height_m), source


def _rotate_vector_by_quaternion(
    vx: float,
    vy: float,
    vz: float,
    qx: float,
    qy: float,
    qz: float,
    qw: float,
) -> Tuple[float, float, float]:
    tx = 2.0 * (qy * vz - qz * vy)
    ty = 2.0 * (qz * vx - qx * vz)
    tz = 2.0 * (qx * vy - qy * vx)
    return (
        vx + qw * tx + (qy * tz - qz * ty),
        vy + qw * ty + (qz * tx - qx * tz),
        vz + qw * tz + (qx * ty - qy * tx),
    )


def transform_body_offset_to_world(
    origin_xyz: Tuple[float, float, float],
    quat_xyzw: Tuple[float, float, float, float],
    offset_body_xyz: Tuple[float, float, float],
) -> Tuple[float, float, float]:
    """P_world = O_world + R(q) * offset_body."""
    ox, oy, oz = (float(origin_xyz[0]), float(origin_xyz[1]), float(origin_xyz[2]))
    qx, qy, qz, qw = (
        float(quat_xyzw[0]),
        float(quat_xyzw[1]),
        float(quat_xyzw[2]),
        float(quat_xyzw[3]),
    )
    bx, by, bz = (
        float(offset_body_xyz[0]),
        float(offset_body_xyz[1]),
        float(offset_body_xyz[2]),
    )
    rx, ry, rz = _rotate_vector_by_quaternion(bx, by, bz, qx, qy, qz, qw)
    return (ox + rx, oy + ry, oz + rz)


def semantic_center_from_gazebo_model_origin(
    gazebo_origin_xyz: Tuple[float, float, float],
    quat_xyzw: Tuple[float, float, float, float],
    label: str,
) -> Tuple[float, float, float]:
    """Centro semántico del cuboide desde origen real del modelo en Gazebo."""
    spec = get_known_box_gt_spec(label)
    if spec is None:
        return (
            float(gazebo_origin_xyz[0]),
            float(gazebo_origin_xyz[1]),
            float(gazebo_origin_xyz[2]),
        )
    return transform_body_offset_to_world(
        gazebo_origin_xyz,
        quat_xyzw,
        spec.model_origin_to_geometry_center_offset_xyz,
    )


def top_face_center_from_semantic_center(
    semantic_center_xyz: Tuple[float, float, float],
    quat_xyzw: Tuple[float, float, float, float],
    label: str,
) -> Tuple[float, float, float]:
    """Centro cara superior: semántica + R(q)*[0,0,H/2] en frame cuerpo."""
    spec = get_known_box_gt_spec(label)
    if spec is None:
        return (
            float(semantic_center_xyz[0]),
            float(semantic_center_xyz[1]),
            float(semantic_center_xyz[2]),
        )
    half_h = 0.5 * float(spec.height_m)
    return transform_body_offset_to_world(
        semantic_center_xyz, quat_xyzw, (0.0, 0.0, half_h)
    )


def offset_local_to_world_delta(
    offset_local: Tuple[float, float, float], yaw_rad: float
) -> Tuple[float, float, float]:
    ox, oy, oz = (float(offset_local[0]), float(offset_local[1]), float(offset_local[2]))
    c = math.cos(float(yaw_rad))
    s = math.sin(float(yaw_rad))
    return (c * ox - s * oy, s * ox + c * oy, oz)


def semantic_center_z_world(
    table_surface_z_m: float, height_m: float, *, epsilon_m: float = 0.001
) -> float:
    """Centro geométrico Z: mesa + mitad de altura."""
    return float(table_surface_z_m) + 0.5 * float(height_m) + float(epsilon_m)


def gazebo_model_origin_from_semantic_center(
    semantic_center_xyz: Tuple[float, float, float],
    yaw_rad: float,
    label: str,
) -> Tuple[float, float, float]:
    """O_world = G_world - Rz(yaw) * model_origin_to_geometry_center_offset."""
    spec = get_known_box_gt_spec(label)
    if spec is None:
        cx, cy, cz = semantic_center_xyz
        return float(cx), float(cy), float(cz)
    dx, dy, dz = offset_local_to_world_delta(
        spec.model_origin_to_geometry_center_offset_xyz, yaw_rad
    )
    cx, cy, cz = semantic_center_xyz
    return (float(cx) - dx, float(cy) - dy, float(cz) - dz)


def log_spawn_semantic_pose(
    logger: Any,
    *,
    label: str,
    semantic_center_xyz: Tuple[float, float, float],
    yaw_rad: float,
    dims_lwh: Tuple[float, float, float],
) -> None:
    if logger is None:
        return
    try:
        logger.info(
            "[SPAWN_SEMANTIC_POSE] label=%s semantic_center=(%.4f,%.4f,%.4f) "
            "yaw=%.3fdeg dims_lwh=(%.4f,%.4f,%.4f)"
            % (
                normalize_label(label),
                semantic_center_xyz[0],
                semantic_center_xyz[1],
                semantic_center_xyz[2],
                math.degrees(yaw_rad),
                dims_lwh[0],
                dims_lwh[1],
                dims_lwh[2],
            )
        )
    except Exception:
        pass


def log_spawn_gazebo_pose(
    logger: Any,
    *,
    label: str,
    model_origin_xyz: Tuple[float, float, float],
    internal_offset: Tuple[float, float, float],
) -> None:
    if logger is None:
        return
    try:
        logger.info(
            "[SPAWN_GAZEBO_POSE] label=%s model_origin_pose=(%.4f,%.4f,%.4f) "
            "internal_model_offset=(%.4f,%.4f,%.4f) "
            'note="offset only used to place SDF model"'
            % (
                normalize_label(label),
                model_origin_xyz[0],
                model_origin_xyz[1],
                model_origin_xyz[2],
                internal_offset[0],
                internal_offset[1],
                internal_offset[2],
            )
        )
    except Exception:
        pass


def _axis_unit(axis: str) -> np.ndarray:
    if axis == "x":
        return np.array([1.0, 0.0, 0.0], dtype=np.float64)
    if axis == "y":
        return np.array([0.0, 1.0, 0.0], dtype=np.float64)
    if axis == "z":
        return np.array([0.0, 0.0, 1.0], dtype=np.float64)
    raise ValueError(f"eje inválido: {axis}")


def _rotate_z(vec: np.ndarray, yaw_rad: float) -> np.ndarray:
    c = math.cos(float(yaw_rad))
    s = math.sin(float(yaw_rad))
    x, y, z = float(vec[0]), float(vec[1]), float(vec[2])
    return np.array([c * x - s * y, s * x + c * y, z], dtype=np.float64)


def resolve_runtime_gt_spawn_axes(
    yaw_rad: float,
    *,
    local_length_axis: str = "x",
    local_width_axis: str = "y",
) -> Dict[str, Any]:
    """Ejes mundo y yaw de cierre (gap en eje corto) desde yaw de spawn."""
    gt_yaw = float(yaw_rad)
    e_len_w = _rotate_z(_axis_unit(local_length_axis), gt_yaw)
    e_wid_w = _rotate_z(_axis_unit(local_width_axis), gt_yaw)
    e_len_xy = e_len_w[:2] / max(float(np.linalg.norm(e_len_w[:2])), 1e-9)
    e_wid_xy = e_wid_w[:2] / max(float(np.linalg.norm(e_wid_w[:2])), 1e-9)
    closing_axis = e_wid_w / max(float(np.linalg.norm(e_wid_w)), 1e-9)
    closing_yaw_rad = float(math.atan2(float(e_wid_xy[1]), float(e_wid_xy[0])))
    return {
        "gt_yaw_rad": gt_yaw,
        "gt_length_axis_world": [
            float(e_len_w[0]),
            float(e_len_w[1]),
            float(e_len_w[2]),
        ],
        "gt_width_axis_world": [
            float(e_wid_w[0]),
            float(e_wid_w[1]),
            float(e_wid_w[2]),
        ],
        "gt_closing_axis_world": [
            float(closing_axis[0]),
            float(closing_axis[1]),
            float(closing_axis[2]),
        ],
        "closing_yaw_rad": closing_yaw_rad,
        "long_axis_xy": [float(e_len_xy[0]), float(e_len_xy[1])],
        "short_axis_xy": [float(e_wid_xy[0]), float(e_wid_xy[1])],
    }


def _log_synthetic_top_face_debug(
    logger: Any,
    *,
    label: str,
    spec: Optional[KnownBoxGtSpec],
    dims_collision_lwh: Tuple[float, float, float],
    overlay_length: float,
    overlay_width: float,
    top_face_corners_base: List[List[float]],
    for_overlay: bool,
    projected_pixels: Optional[List[Tuple[int, int]]] = None,
    projection_note: str = "pending",
) -> None:
    if logger is None:
        return
    ins_l = float(spec.overlay_top_face_inset_length_m) if spec else 0.0
    ins_w = float(spec.overlay_top_face_inset_width_m) if spec else 0.0
    lax = spec.local_length_axis if spec else "n/a"
    wax = spec.local_width_axis if spec else "n/a"
    try:
        logger.info(
            "[SYNTHETIC_TOP_FACE_DEBUG] label=%s for_overlay=%s "
            "dims_collision_lwh=(%.4f,%.4f,%.4f) overlay_length=%.4f "
            "overlay_width=%.4f inset_length_m=%.4f inset_width_m=%.4f "
            "local_length_axis=%s local_width_axis=%s "
            "top_face_corners_base=%s projected_pixels=%s projection=%s"
            % (
                normalize_label(label),
                str(for_overlay).lower(),
                float(dims_collision_lwh[0]),
                float(dims_collision_lwh[1]),
                float(dims_collision_lwh[2]),
                float(overlay_length),
                float(overlay_width),
                ins_l,
                ins_w,
                lax,
                wax,
                _fmt_corners_short(top_face_corners_base),
                projected_pixels if projected_pixels is not None else "pending",
                projection_note,
            )
        )
    except Exception:
        pass


def _log_synthetic_axis_debug(
    logger: Any,
    *,
    label: str,
    yaw_rad: float,
    gt_yaw_rad: float,
    spec: KnownBoxGtSpec,
    e_len: np.ndarray,
    e_wid: np.ndarray,
    closing_axis: np.ndarray,
    closing_yaw_rad: float,
    for_overlay: bool,
) -> None:
    if logger is None:
        return
    try:
        logger.info(
            "[SYNTHETIC_AXIS_DEBUG] label=%s yaw_rad=%.4f gt_yaw_rad=%.4f "
            "local_length_axis=%s local_width_axis=%s "
            "e_len=[%.4f,%.4f,%.4f] e_wid=[%.4f,%.4f,%.4f] "
            "closing_axis=[%.4f,%.4f,%.4f] closing_yaw_rad=%.4f "
            "for_overlay=%s note=operational closing on local_width_axis (short side)"
            % (
                normalize_label(label),
                float(yaw_rad),
                float(gt_yaw_rad),
                spec.local_length_axis,
                spec.local_width_axis,
                float(e_len[0]),
                float(e_len[1]),
                float(e_len[2]),
                float(e_wid[0]),
                float(e_wid[1]),
                float(e_wid[2]),
                float(closing_axis[0]),
                float(closing_axis[1]),
                float(closing_axis[2]),
                float(closing_yaw_rad),
                str(for_overlay).lower(),
            )
        )
    except Exception:
        pass


def compute_synthetic_operational_top_face_base(
    semantic_center_base: Tuple[float, float, float],
    yaw_rad: float,
    dims_lwh_m: Optional[Tuple[float, float, float]] = None,
    *,
    label: Optional[str] = None,
    apply_yaw_offset: bool = True,
    for_overlay: bool = False,
    logger: Any = None,
) -> Dict[str, Any]:
    """Cara superior sintética (base): semántica + yaw + ejes locales del modelo.

    Usa ``KnownBoxGtSpec`` (local_length_axis / local_width_axis / yaw_offset_rad).
    Longitud L = lado largo; anchura W = eje de cierre (parallel jaw).

    Con ``for_overlay=True`` aplica insets por eje local (solo debug visual);
    nunca usa ``collision_dims_inflated`` ni ``collision_margin``.
    """
    lbl = normalize_label(str(label or "").strip().lower())
    spec = get_known_box_gt_spec(lbl) if lbl else None
    source_of_dims = "dims_lwh_m argument"

    cx, cy, cz = (
        float(semantic_center_base[0]),
        float(semantic_center_base[1]),
        float(semantic_center_base[2]),
    )

    if spec is not None:
        length_m, width_m, height_m, source_of_dims = resolve_top_face_dims_lwh(
            spec, for_overlay=bool(for_overlay)
        )
        yaw_offset = float(spec.yaw_offset_rad)
        gt_yaw = float(yaw_rad) + (yaw_offset if apply_yaw_offset else 0.0)
        e_len = _rotate_z(_axis_unit(spec.local_length_axis), gt_yaw)
        e_wid = _rotate_z(_axis_unit(spec.local_width_axis), gt_yaw)
    else:
        if dims_lwh_m is None or len(dims_lwh_m) < 3:
            raise ValueError(
                "compute_synthetic_operational_top_face_base: dims_lwh_m requerido sin label conocido"
            )
        length_m, width_m, height_m = (
            float(dims_lwh_m[0]),
            float(dims_lwh_m[1]),
            float(dims_lwh_m[2]),
        )
        yaw_offset = 0.0
        gt_yaw = float(yaw_rad)
        e_len = _rotate_z(np.array([1.0, 0.0, 0.0], dtype=np.float64), gt_yaw)
        e_wid = _rotate_z(np.array([0.0, 1.0, 0.0], dtype=np.float64), gt_yaw)
        source_of_dims = "dims_lwh_m argument (no KnownBoxGtSpec)"

    half_h = 0.5 * float(height_m)
    half_l = 0.5 * float(length_m)
    half_w = 0.5 * float(width_m)
    geom = np.array([cx, cy, cz], dtype=np.float64)
    top_center = geom + np.array([0.0, 0.0, half_h], dtype=np.float64)
    closing_axis = e_wid / max(np.linalg.norm(e_wid), 1e-9)
    e_len_xy = e_len[:2] / max(np.linalg.norm(e_len[:2]), 1e-9)
    e_wid_xy = e_wid[:2] / max(np.linalg.norm(e_wid[:2]), 1e-9)
    closing_yaw_rad = float(math.atan2(float(e_wid_xy[1]), float(e_wid_xy[0])))
    top_corners = _face_corners_at_center(
        top_center,
        half_length_m=half_l,
        half_width_m=half_w,
        axis_length=e_len,
        axis_width=e_wid,
    )
    grasp_center = [cx, cy, float(top_center[2])]
    top_face_center_base = [
        float(top_center[0]),
        float(top_center[1]),
        float(top_center[2]),
    ]

    dims_collision = spec.dims_lwh_m if spec is not None else (length_m, width_m, height_m)
    if spec is not None:
        if bool(for_overlay):
            _log_synthetic_top_face_debug(
                logger,
                label=lbl,
                spec=spec,
                dims_collision_lwh=dims_collision,
                overlay_length=float(length_m),
                overlay_width=float(width_m),
                top_face_corners_base=top_corners,
                for_overlay=True,
                projection_note="awaiting pinhole project in perception_node",
            )
        _log_synthetic_axis_debug(
            logger,
            label=lbl,
            yaw_rad=float(yaw_rad),
            gt_yaw_rad=gt_yaw,
            spec=spec,
            e_len=e_len,
            e_wid=e_wid,
            closing_axis=closing_axis,
            closing_yaw_rad=closing_yaw_rad,
            for_overlay=bool(for_overlay),
        )

    return {
        "semantic_box_center_base": [cx, cy, cz],
        "top_face_center_base": top_face_center_base,
        "top_face_corners_base": top_corners,
        "grasp_center_base": grasp_center,
        "top_z_m": float(top_center[2]),
        "gt_yaw_rad": float(gt_yaw),
        "closing_yaw_rad": closing_yaw_rad,
        "dims_collision_lwh": list(dims_collision),
        "dims_source": str(source_of_dims),
        "for_overlay": bool(for_overlay),
        "overlay_inset_length_m": (
            float(spec.overlay_top_face_inset_length_m) if spec else 0.0
        ),
        "overlay_inset_width_m": (
            float(spec.overlay_top_face_inset_width_m) if spec else 0.0
        ),
        "long_axis_xy": [float(e_len_xy[0]), float(e_len_xy[1])],
        "short_axis_xy": [float(e_wid_xy[0]), float(e_wid_xy[1])],
        "dims_used_lwh": [float(length_m), float(width_m), float(height_m)],
        "local_length_axis": spec.local_length_axis if spec else "x",
        "local_width_axis": spec.local_width_axis if spec else "y",
        "top_face_source": "runtime_gt_known_box",
    }


def _face_corners_at_center(
    center: np.ndarray,
    *,
    half_length_m: float,
    half_width_m: float,
    axis_length: np.ndarray,
    axis_width: np.ndarray,
) -> List[List[float]]:
    c = np.asarray(center, dtype=np.float64).reshape(3)
    hl, hw = float(half_length_m), float(half_width_m)
    corners = [
        c + hl * axis_length + hw * axis_width,
        c + hl * axis_length - hw * axis_width,
        c - hl * axis_length - hw * axis_width,
        c - hl * axis_length + hw * axis_width,
    ]
    return [[float(p[0]), float(p[1]), float(p[2])] for p in corners]


def compute_known_box_gt_geometry(
    label: str,
    semantic_center_xyz: Tuple[float, float, float],
    yaw_rad: float,
    *,
    logger: Any = None,
) -> Optional[Dict[str, Any]]:
    """GT en world desde centro semántico del cuboide (sin offsets de percepción)."""
    spec = get_known_box_gt_spec(label)
    if spec is None:
        return None

    cx, cy, cz = (
        float(semantic_center_xyz[0]),
        float(semantic_center_xyz[1]),
        float(semantic_center_xyz[2]),
    )
    gt_yaw = float(yaw_rad) + float(spec.yaw_offset_rad)
    length_m, width_m, height_m = spec.dims_lwh_m
    half_h = 0.5 * float(height_m)
    half_l = 0.5 * float(length_m)
    half_w = 0.5 * float(width_m)

    geom_center = np.array([cx, cy, cz], dtype=np.float64)
    top_center = geom_center + np.array([0.0, 0.0, half_h], dtype=np.float64)
    bottom_center = geom_center - np.array([0.0, 0.0, half_h], dtype=np.float64)

    e_len_w = _rotate_z(_axis_unit(spec.local_length_axis), gt_yaw)
    e_wid_w = _rotate_z(_axis_unit(spec.local_width_axis), gt_yaw)
    closing_axis_world = e_wid_w / max(np.linalg.norm(e_wid_w), 1e-9)

    top_corners = _face_corners_at_center(
        top_center,
        half_length_m=half_l,
        half_width_m=half_w,
        axis_length=e_len_w,
        axis_width=e_wid_w,
    )
    bottom_corners = _face_corners_at_center(
        bottom_center,
        half_length_m=half_l,
        half_width_m=half_w,
        axis_length=e_len_w,
        axis_width=e_wid_w,
    )

    semantic = [cx, cy, cz]
    top_c = [float(top_center[0]), float(top_center[1]), float(top_center[2])]

    out: Dict[str, Any] = {
        "known_box_gt": True,
        "source_pose_semantics": SOURCE_POSE_SEMANTICS_GEOMETRY_CENTER,
        "semantic_box_center_world": list(semantic),
        "gt_geometry_center_world": list(semantic),
        "gt_top_face_center_world": top_c,
        "gt_top_face_corners_world": top_corners,
        "gt_bottom_face_corners_world": bottom_corners,
        "yaw_rad": float(gt_yaw),
        "gt_yaw_rad": float(gt_yaw),
        "gt_length_axis_world": [float(e_len_w[0]), float(e_len_w[1]), float(e_len_w[2])],
        "gt_width_axis_world": [float(e_wid_w[0]), float(e_wid_w[1]), float(e_wid_w[2])],
        "gt_closing_axis_world": [
            float(closing_axis_world[0]),
            float(closing_axis_world[1]),
            float(closing_axis_world[2]),
        ],
        "dims_used_lwh": [float(length_m), float(width_m), float(height_m)],
        "dims_used": [float(length_m), float(width_m), float(height_m)],
        "dims_xyz_m": list(spec.dims_xyz_m),
        "top_z_m": float(top_c[2]),
        "local_length_axis": spec.local_length_axis,
        "local_width_axis": spec.local_width_axis,
        "yaw_offset_rad": float(spec.yaw_offset_rad),
        "geometry_notes": spec.notes,
    }

    if logger is not None:
        try:
            logger.info(
                "[RUNTIME_SCENE_GT_GEOMETRY] label=%s "
                "gt_geometry_center=(%.4f,%.4f,%.4f) "
                "gt_top_face_center=(%.4f,%.4f,%.4f) "
                "gt_top_face_corners=%s "
                "dims_used_lwh=(%.4f,%.4f,%.4f) yaw=%.3fdeg "
                "gt_closing_axis=(%.3f,%.3f,%.3f)"
                % (
                    normalize_label(label),
                    cx,
                    cy,
                    cz,
                    top_c[0],
                    top_c[1],
                    top_c[2],
                    _fmt_corners_short(top_corners),
                    length_m,
                    width_m,
                    height_m,
                    math.degrees(gt_yaw),
                    closing_axis_world[0],
                    closing_axis_world[1],
                    closing_axis_world[2],
                )
            )
        except Exception:
            pass
    return out


def compute_tall_object_runtime_gt_geometry(
    label: str,
    semantic_center_xyz: Tuple[float, float, float],
    yaw_rad: float,
    *,
    dims_lwh_m: Tuple[float, float, float],
    local_length_axis: str = "x",
    local_width_axis: str = "y",
    logger: Any = None,
) -> Optional[Dict[str, Any]]:
    """GT operativo para objetos altos (mustard_bottle, etc.) con ejes link SDF."""
    length_m, width_m, height_m = (
        float(dims_lwh_m[0]),
        float(dims_lwh_m[1]),
        float(dims_lwh_m[2]),
    )
    if length_m <= 1e-6 or width_m <= 1e-6 or height_m <= 1e-6:
        return None

    cx, cy, cz = (
        float(semantic_center_xyz[0]),
        float(semantic_center_xyz[1]),
        float(semantic_center_xyz[2]),
    )
    gt_yaw = float(yaw_rad)
    half_h = 0.5 * float(height_m)
    half_l = 0.5 * float(length_m)
    half_w = 0.5 * float(width_m)

    geom_center = np.array([cx, cy, cz], dtype=np.float64)
    top_center = geom_center + np.array([0.0, 0.0, half_h], dtype=np.float64)
    e_len_w = _rotate_z(_axis_unit(str(local_length_axis)), gt_yaw)
    e_wid_w = _rotate_z(_axis_unit(str(local_width_axis)), gt_yaw)
    closing_axis_world = e_wid_w / max(np.linalg.norm(e_wid_w), 1e-9)
    e_len_xy = e_len_w[:2] / max(np.linalg.norm(e_len_w[:2]), 1e-9)
    e_wid_xy = e_wid_w[:2] / max(np.linalg.norm(e_wid_w[:2]), 1e-9)
    closing_yaw_rad = float(math.atan2(float(e_wid_xy[1]), float(e_wid_xy[0])))

    top_corners = _face_corners_at_center(
        top_center,
        half_length_m=half_l,
        half_width_m=half_w,
        axis_length=e_len_w,
        axis_width=e_wid_w,
    )

    top_c = [float(top_center[0]), float(top_center[1]), float(top_center[2])]
    out: Dict[str, Any] = {
        "gt_geometry_center_world": [cx, cy, cz],
        "semantic_box_center_world": [cx, cy, cz],
        "gt_top_face_center_world": top_c,
        "gt_top_face_corners_world": top_corners,
        "gt_yaw_rad": float(gt_yaw),
        "yaw_rad": float(gt_yaw),
        "gt_length_axis_world": [float(e_len_w[0]), float(e_len_w[1]), float(e_len_w[2])],
        "gt_width_axis_world": [float(e_wid_w[0]), float(e_wid_w[1]), float(e_wid_w[2])],
        "gt_closing_axis_world": [
            float(closing_axis_world[0]),
            float(closing_axis_world[1]),
            float(closing_axis_world[2]),
        ],
        "closing_yaw_rad": closing_yaw_rad,
        "long_axis_xy": [float(e_len_xy[0]), float(e_len_xy[1])],
        "short_axis_xy": [float(e_wid_xy[0]), float(e_wid_xy[1])],
        "dims_used_lwh": [float(length_m), float(width_m), float(height_m)],
        "dims_used": [float(length_m), float(width_m), float(height_m)],
        "top_z_m": float(top_c[2]),
        "local_length_axis": str(local_length_axis),
        "local_width_axis": str(local_width_axis),
    }

    if logger is not None:
        try:
            logger.info(
                "[TALL_OBJECT_RUNTIME_GT_GEOMETRY] label=%s "
                "local_length_axis=%s local_width_axis=%s "
                "dims_lwh=(%.4f,%.4f,%.4f) closing_yaw_rad=%.4f "
                "long_axis_xy=(%.4f,%.4f) short_axis_xy=(%.4f,%.4f)"
                % (
                    normalize_label(label),
                    str(local_length_axis),
                    str(local_width_axis),
                    length_m,
                    width_m,
                    height_m,
                    closing_yaw_rad,
                    float(e_len_xy[0]),
                    float(e_len_xy[1]),
                    float(e_wid_xy[0]),
                    float(e_wid_xy[1]),
                )
            )
        except Exception:
            pass
    return out


def resolve_semantic_and_gazebo_poses(
    label: str,
    center_xy: Tuple[float, float],
    yaw_rad: float,
    table_surface_z_m: float,
    height_m: float,
    *,
    center_z: Optional[float] = None,
    epsilon_m: float = 0.001,
    logger: Any = None,
) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
    """Devuelve (semantic_center_xyz, gazebo_model_origin_xyz) para cajas conocidas."""
    spec = get_known_box_gt_spec(label)
    if spec is None:
        z = float(center_z) if center_z is not None else float(center_xy[0] * 0 + table_surface_z_m)
        sem = (float(center_xy[0]), float(center_xy[1]), z)
        return sem, sem

    sem_z = (
        float(center_z)
        if center_z is not None
        else semantic_center_z_world(table_surface_z_m, height_m, epsilon_m=epsilon_m)
    )
    semantic = (float(center_xy[0]), float(center_xy[1]), sem_z)
    gazebo = gazebo_model_origin_from_semantic_center(semantic, yaw_rad, label)
    dims = spec.dims_lwh_m
    log_spawn_semantic_pose(
        logger, label=label, semantic_center_xyz=semantic, yaw_rad=yaw_rad, dims_lwh=dims
    )
    log_spawn_gazebo_pose(
        logger,
        label=label,
        model_origin_xyz=gazebo,
        internal_offset=spec.model_origin_to_geometry_center_offset_xyz,
    )
    return semantic, gazebo


def _fmt_corners_short(corners: List[List[float]]) -> str:
    if len(corners) < 4:
        return "[]"
    pts = ["(%.3f,%.3f)" % (float(c[0]), float(c[1])) for c in corners[:4]]
    return "[" + ", ".join(pts) + "]"


def _point_world_to_base(
    xyz_world: List[float], world_to_base_matrix: np.ndarray
) -> List[float]:
    pt = np.array(
        [[float(xyz_world[0]), float(xyz_world[1]), float(xyz_world[2])]],
        dtype=np.float64,
    )
    hom = np.hstack((pt, np.ones((1, 1), dtype=np.float64)))
    out = (world_to_base_matrix @ hom.T).T[0, :3]
    return [float(out[0]), float(out[1]), float(out[2])]


def _corners_world_to_base(
    corners_world: List[List[float]], world_to_base_matrix: np.ndarray
) -> List[List[float]]:
    if not corners_world:
        return []
    pts = np.asarray(corners_world, dtype=np.float64).reshape(-1, 3)
    hom = np.hstack((pts, np.ones((pts.shape[0], 1), dtype=np.float64)))
    out = (world_to_base_matrix @ hom.T).T[:, :3]
    return [[float(p[0]), float(p[1]), float(p[2])] for p in out]


def build_gt_fields_from_semantic_center(
    label: str,
    semantic_center_xyz: Tuple[float, float, float],
    yaw_rad: float,
    *,
    logger: Any = None,
) -> Dict[str, Any]:
    """Campos GT publicables; ``pose_world`` = centro semántico."""
    geom = compute_known_box_gt_geometry(
        label, semantic_center_xyz, yaw_rad, logger=logger
    )
    if geom is None:
        return {}
    cx, cy, cz = semantic_center_xyz
    gt_yaw = float(geom.get("yaw_rad", yaw_rad))
    return {
        **geom,
        "pose_world": {
            "x": float(cx),
            "y": float(cy),
            "z": float(cz),
            "roll": 0.0,
            "pitch": 0.0,
            "yaw": float(gt_yaw),
        },
        "yaw_rad": float(gt_yaw),
    }


def enrich_gt_object_entry(
    entry: Dict[str, Any],
    *,
    logger: Any = None,
) -> Dict[str, Any]:
    """Completa geometría GT si falta; ``pose_world`` se interpreta como centro semántico."""
    label = str(entry.get("label", "")).strip().lower()
    if not is_known_spawn_geometry_box_label(label):
        from panda_vision.spawn.known_object_geometry import enrich_runtime_scene_object_fields

        return enrich_runtime_scene_object_fields(entry, logger=logger)
    if entry.get("source_pose_semantics") == SOURCE_POSE_SEMANTICS_GEOMETRY_CENTER:
        if isinstance(entry.get("gt_top_face_corners_world"), list):
            from panda_vision.spawn.known_object_geometry import enrich_runtime_scene_object_fields

            return enrich_runtime_scene_object_fields(entry, logger=logger)

    pw = entry.get("pose_world")
    if not isinstance(pw, dict):
        from panda_vision.spawn.known_object_geometry import enrich_runtime_scene_object_fields

        return enrich_runtime_scene_object_fields(entry, logger=logger)
    yaw = float(pw.get("yaw", entry.get("yaw_rad", 0.0)))
    semantic = (
        float(pw["x"]),
        float(pw["y"]),
        float(pw["z"]),
    )
    fields = build_gt_fields_from_semantic_center(label, semantic, yaw, logger=logger)
    entry.update(fields)
    from panda_vision.spawn.known_object_geometry import enrich_runtime_scene_object_fields

    return enrich_runtime_scene_object_fields(entry, logger=logger)


def enrich_gt_object_entry_base(
    entry: Dict[str, Any],
    world_to_base_matrix: np.ndarray,
    *,
    logger: Any = None,
) -> Dict[str, Any]:
    """Solo proyección world→base; sin offsets adicionales."""
    enrich_gt_object_entry(entry, logger=logger)

    sem_w = entry.get("semantic_box_center_world") or entry.get("gt_geometry_center_world")
    top_c_w = entry.get("gt_top_face_center_world")
    origin_w = entry.get("model_origin_pose_world")
    if isinstance(sem_w, list) and len(sem_w) >= 3:
        entry["semantic_box_center_base"] = _point_world_to_base(sem_w, world_to_base_matrix)
        entry["gt_geometry_center_base"] = entry["semantic_box_center_base"]
        entry["spawn_position_base"] = entry["semantic_box_center_base"]
    if isinstance(top_c_w, list) and len(top_c_w) >= 3:
        entry["gt_top_face_center_base"] = _point_world_to_base(top_c_w, world_to_base_matrix)
        if entry.get("tall_object_sdf_offset_applied"):
            entry["grasp_center_base"] = list(entry["gt_top_face_center_base"])
            entry["chosen_target_center_base"] = list(entry["grasp_center_base"])
            entry["top_surface_center_base"] = list(entry["grasp_center_base"])
    old_cap_w = entry.get("mustard_old_offset_cap_center_world")
    if isinstance(old_cap_w, list) and len(old_cap_w) >= 3:
        entry["mustard_old_offset_cap_center_base"] = _point_world_to_base(
            old_cap_w, world_to_base_matrix
        )
    vert_cap_w = entry.get("mustard_vertical_axis_cap_center_world")
    if isinstance(vert_cap_w, list) and len(vert_cap_w) >= 3:
        entry["mustard_vertical_axis_cap_center_base"] = _point_world_to_base(
            vert_cap_w, world_to_base_matrix
        )
    mesh_cap_w = entry.get("mustard_mesh_local_cap_center_world")
    if isinstance(mesh_cap_w, list) and len(mesh_cap_w) >= 3:
        entry["mustard_mesh_local_cap_center_base"] = _point_world_to_base(
            mesh_cap_w, world_to_base_matrix
        )
    if isinstance(origin_w, list) and len(origin_w) >= 3:
        entry["model_origin_pose_base"] = _point_world_to_base(origin_w, world_to_base_matrix)

    top_w = entry.get("gt_top_face_corners_world")
    if isinstance(top_w, list) and len(top_w) >= 4:
        entry["gt_top_face_corners_base"] = _corners_world_to_base(top_w, world_to_base_matrix)

    bottom_w = entry.get("gt_bottom_face_corners_world")
    if isinstance(bottom_w, list) and len(bottom_w) >= 4:
        entry["gt_bottom_face_corners_base"] = _corners_world_to_base(
            bottom_w, world_to_base_matrix
        )
    return entry


def get_model_origin_to_geometry_center_offset(label: str) -> Tuple[float, float, float]:
    spec = get_known_box_gt_spec(label)
    if spec is None:
        return (0.0, 0.0, 0.0)
    return spec.model_origin_to_geometry_center_offset_xyz


if __name__ == "__main__":
    _yaw = math.radians(93.686)
    _table = 0.26
    _h = 0.21
    _sem = (0.645, -0.104, semantic_center_z_world(_table, _h))
    _g = compute_known_box_gt_geometry("cracker_box", _sem, _yaw)
    assert _g is not None
    print("semantic", _sem)
    print("geom", _g["gt_geometry_center_world"])
    print("top", _g["gt_top_face_center_world"])
    print("top_z", _g["top_z_m"])
    _gz = gazebo_model_origin_from_semantic_center(_sem, _yaw, "cracker_box")
    print("gazebo_origin", _gz)
