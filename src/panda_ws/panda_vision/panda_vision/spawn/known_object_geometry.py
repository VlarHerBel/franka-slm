"""Registro geométrico unificado para RuntimeScene (spawn, GT, percepción, planificación)."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from panda_vision.grasp.object_grasp_policy import (
    get_collision_dimensions,
    get_grasp_policy,
    normalize_label,
)
from panda_vision.grasp.object_grasp_policy import (
    resolve_tall_object_top_z_m,
    TALL_OBJECT_CAP_CENTER_SDF_OFFSET_SOURCE,
)

# Centro de agarre en tapón medido en link (mesh collision + pose SDF 0.025 -0.005 0 0 0 -1.15).
MUSTARD_TOP_CAP_CENTER_GEOMETRY_SOURCE = "runtime_gt_mustard_top_cap_center_geometry"
MUSTARD_VERTICAL_AXIS_CAP_CENTER_SOURCE = (
    "runtime_gt_mustard_vertical_axis_cap_center"
)
MUSTARD_MESH_LOCAL_CAP_CENTER_SOURCE = (
    "runtime_gt_mustard_mesh_local_cap_center"
)
MUSTARD_CAP_CENTER_MODE_SDF_OFFSET = "sdf_offset"
MUSTARD_CAP_CENTER_MODE_VERTICAL_AXIS = "vertical_axis_from_footprint"
MUSTARD_CAP_CENTER_MODE_MESH_LOCAL = "mesh_local_cap_center"
DEFAULT_MUSTARD_CAP_CENTER_MODE = MUSTARD_CAP_CENTER_MODE_MESH_LOCAL

RUNTIME_GT_TALL_CAP_CENTER_SOURCES = frozenset(
    {
        "runtime_gt_tall_object_cap_center",
        TALL_OBJECT_CAP_CENTER_SDF_OFFSET_SOURCE,
        MUSTARD_TOP_CAP_CENTER_GEOMETRY_SOURCE,
        MUSTARD_VERTICAL_AXIS_CAP_CENTER_SOURCE,
        MUSTARD_MESH_LOCAL_CAP_CENTER_SOURCE,
    }
)


def is_runtime_gt_tall_cap_center_source(source: Any) -> bool:
    return str(source or "").strip() in RUNTIME_GT_TALL_CAP_CENTER_SOURCES
from panda_vision.spawn.runtime_scene_gt_geometry import (
    KNOWN_BOX_GT_SPECS,
    compute_known_box_gt_geometry,
    compute_tall_object_runtime_gt_geometry,
    get_known_box_gt_spec,
    is_known_spawn_geometry_box_label,
    offset_local_to_world_delta,
    resolve_runtime_gt_spawn_axes,
    transform_body_offset_to_world,
)

DEFAULT_COLLISION_MARGIN_M = 0.005
ROLE_TARGET = "target"
ROLE_OBSTACLE = "obstacle"
ROLE_UNKNOWN = "unknown"


@dataclass(frozen=True)
class KnownTallObjectSdfSpec:
    """Offsets link Gazebo (mesh collision/visual) respecto al origen del modelo."""

    label: str
    model_origin_to_geometry_center_offset_xyz: Tuple[float, float, float]
    model_origin_to_top_cap_center_offset_xyz: Tuple[float, float, float]
    notes: str
    # Ejes locales del link SDF (width_x, length_y, height_z en OBJECT_DB).
    local_length_axis: str = "x"
    local_width_axis: str = "y"
    # Centro del tapón respecto al centro geométrico en frame link (medido en mesh).
    geometry_center_to_cap_center_offset_local_xyz: Tuple[float, float, float] = (
        0.0,
        0.0,
        0.0,
    )
    # Centro del tapón en frame local del modelo (origen link → punto mesh/SDF).
    cap_center_local_m: Tuple[float, float, float] = (0.0, 0.0, 0.0)


def _axis_unit_local(axis: str) -> Tuple[float, float, float]:
    ax = str(axis).strip().lower()
    if ax == "x":
        return (1.0, 0.0, 0.0)
    if ax == "y":
        return (0.0, 1.0, 0.0)
    if ax == "z":
        return (0.0, 0.0, 1.0)
    return (1.0, 0.0, 0.0)


def _decompose_offset_local_along_axes(
    offset_local: Tuple[float, float, float],
    *,
    local_length_axis: str,
    local_width_axis: str,
) -> Tuple[float, float]:
    ox, oy, oz = (float(offset_local[0]), float(offset_local[1]), float(offset_local[2]))
    lu = _axis_unit_local(local_length_axis)
    wu = _axis_unit_local(local_width_axis)
    return (
        ox * lu[0] + oy * lu[1] + oz * lu[2],
        ox * wu[0] + oy * wu[1] + oz * wu[2],
    )


def _cap_center_source_for_label(label: str) -> str:
    if normalize_label(label) == "mustard_bottle":
        return MUSTARD_TOP_CAP_CENTER_GEOMETRY_SOURCE
    return TALL_OBJECT_CAP_CENTER_SDF_OFFSET_SOURCE


def _resolve_mustard_footprint_center_xy_world(
    entry: Dict[str, Any],
    *,
    geom_center_xy: Tuple[float, float],
) -> Tuple[float, float, str]:
    """Centro XY del footprint/cuerpo (botella vertical → tapón centrado en eje)."""
    cbp = entry.get("collision_box_pose")
    if isinstance(cbp, dict):
        try:
            return (
                float(cbp["x"]),
                float(cbp["y"]),
                "collision_box_pose",
            )
        except (KeyError, TypeError, ValueError):
            pass
    ggc = entry.get("gt_geometry_center_world")
    if isinstance(ggc, (list, tuple)) and len(ggc) >= 2:
        return float(ggc[0]), float(ggc[1]), "gt_geometry_center_world"
    sbc = entry.get("semantic_box_center_world")
    if isinstance(sbc, (list, tuple)) and len(sbc) >= 2:
        return float(sbc[0]), float(sbc[1]), "semantic_box_center_world"
    return (
        float(geom_center_xy[0]),
        float(geom_center_xy[1]),
        "computed_geometry_center",
    )


def _mustard_cap_center_mode(entry: Dict[str, Any]) -> str:
    mode = str(
        entry.get("mustard_cap_center_mode", DEFAULT_MUSTARD_CAP_CENTER_MODE)
    ).strip().lower()
    if mode in (
        MUSTARD_CAP_CENTER_MODE_VERTICAL_AXIS,
        MUSTARD_CAP_CENTER_MODE_SDF_OFFSET,
        MUSTARD_CAP_CENTER_MODE_MESH_LOCAL,
        "vertical_axis_from_footprint",
        "sdf_offset",
        "mesh_local_cap_center",
    ):
        if mode in ("sdf_offset", MUSTARD_CAP_CENTER_MODE_SDF_OFFSET):
            return MUSTARD_CAP_CENTER_MODE_SDF_OFFSET
        if mode in ("mesh_local_cap_center", MUSTARD_CAP_CENTER_MODE_MESH_LOCAL):
            return MUSTARD_CAP_CENTER_MODE_MESH_LOCAL
        return MUSTARD_CAP_CENTER_MODE_VERTICAL_AXIS
    return DEFAULT_MUSTARD_CAP_CENTER_MODE


def _resolve_mustard_model_pose_for_cap(
    entry: Dict[str, Any],
    *,
    origin_xyz: Tuple[float, float, float],
    yaw_rad: float,
) -> Tuple[Optional[Tuple[float, float, float, float]], str]:
    """Quaternion y fuente de pose del modelo (readback Gazebo > pose_world > yaw)."""
    readback = str(entry.get("pose_readback_source") or "").strip()
    if readback:
        pose_src = readback
    elif entry.get("gazebo_origin_world") is not None:
        pose_src = "gazebo_origin_world"
    else:
        pose_src = "runtime_scene_pose_world"

    pw = entry.get("pose_world")
    if isinstance(pw, dict):
        try:
            qx = float(pw["qx"])
            qy = float(pw["qy"])
            qz = float(pw["qz"])
            qw = float(pw["qw"])
            if math.isfinite(qx + qy + qz + qw):
                return (qx, qy, qz, qw), pose_src
        except (KeyError, TypeError, ValueError):
            pass
    c = math.cos(0.5 * float(yaw_rad))
    s = math.sin(0.5 * float(yaw_rad))
    return (0.0, 0.0, s, c), "gt_yaw_2d"


def compute_mustard_mesh_local_cap_center_world(
    origin_xyz: Tuple[float, float, float],
    cap_local_m: Tuple[float, float, float],
    *,
    quat_xyzw: Optional[Tuple[float, float, float, float]] = None,
    yaw_rad: Optional[float] = None,
) -> Tuple[float, float, float]:
    """P_cap_world = T_world_model @ p_cap_local (origen modelo + R * offset local)."""
    if quat_xyzw is not None:
        return transform_body_offset_to_world(origin_xyz, quat_xyzw, cap_local_m)
    if yaw_rad is not None:
        dx, dy, dz = offset_local_to_world_delta(cap_local_m, float(yaw_rad))
        ox, oy, oz = origin_xyz
        return (float(ox) + dx, float(oy) + dy, float(oz) + dz)
    ox, oy, oz = origin_xyz
    lx, ly, lz = cap_local_m
    return (float(ox) + lx, float(oy) + ly, float(oz) + lz)


def _mustard_vertical_axis_cap_top(
    entry: Dict[str, Any],
    *,
    geom_center: Tuple[float, float, float],
    geom_xy: Tuple[float, float],
    top_z_m: float,
    yaw: float,
) -> Tuple[Tuple[float, float, float], str]:
    entry["gt_geometry_center_world"] = list(geom_center)
    entry["semantic_box_center_world"] = list(geom_center)
    entry["collision_box_pose"] = {
        "x": float(geom_center[0]),
        "y": float(geom_center[1]),
        "z": float(geom_center[2]),
        "yaw": float(yaw),
    }
    fp_x, fp_y, fp_src = _resolve_mustard_footprint_center_xy_world(
        entry, geom_center_xy=geom_xy
    )
    return (float(fp_x), float(fp_y), float(top_z_m)), fp_src


KNOWN_TALL_OBJECT_SDF_SPECS: Dict[str, KnownTallObjectSdfSpec] = {
    "mustard_bottle": KnownTallObjectSdfSpec(
        label="mustard_bottle",
        # collision.dae + pose 0.025 -0.005 0 0 0 -1.15 (trimesh AABB/centroid).
        model_origin_to_geometry_center_offset_xyz=(0.0236, -0.0047, 0.0818),
        # Extremo +Y (cara de tapón para agarre con botella tumbada; eje largo local Y).
        model_origin_to_top_cap_center_offset_xyz=(0.0217, 0.0311, 0.0616),
        geometry_center_to_cap_center_offset_local_xyz=(-0.0019, 0.0358, -0.0202),
        cap_center_local_m=(0.0240, -0.0049, 0.1914),
        local_length_axis="y",
        local_width_axis="x",
        notes=(
            "gazebo_ycb/mustard_bottle: mesh collision AABB ~0.058x0.095x0.191 m; "
            "tapón en extremo +local_length_axis (Y); pinza cierra sobre local_width_axis (X)."
        ),
    ),
}


def get_known_tall_object_sdf_spec(label: str) -> Optional[KnownTallObjectSdfSpec]:
    return KNOWN_TALL_OBJECT_SDF_SPECS.get(normalize_label(str(label).strip().lower()))


def _input_z_looks_like_geometry_center(
    oz: float, height_m: float, *, table_z_m: float = 0.27
) -> bool:
    """True si Z de entrada ya parece centro geométrico (mesa + H/2), no origen de modelo."""
    expected_geom_z = float(table_z_m) + 0.5 * float(height_m)
    return abs(float(oz) - expected_geom_z) < 0.05


def apply_tall_object_sdf_geometry_correction(
    entry: Dict[str, Any],
    *,
    logger: Any = None,
) -> Dict[str, Any]:
    """Corrige centro operativo XY desde origen de modelo; Z operativa sin doble offset."""
    if entry.get("tall_object_sdf_offset_applied"):
        return entry

    label = normalize_label(str(entry.get("label", "")).strip().lower())
    spec = get_known_tall_object_sdf_spec(label)
    if spec is None:
        return entry

    geo = get_known_object_geometry(label)
    if geo is None or str(geo.grasp_policy) != "tall_object_topdown":
        return entry

    pw = entry.get("pose_world")
    if not isinstance(pw, dict):
        return entry

    ox = float(pw.get("x", 0.0))
    oy = float(pw.get("y", 0.0))
    oz = float(pw.get("z", 0.0))
    yaw = float(entry.get("gt_yaw_rad", entry.get("yaw_rad", pw.get("yaw", 0.0))))
    pose_sem = str(entry.get("source_pose_semantics", "model_link_origin")).strip()

    geom_off = spec.model_origin_to_geometry_center_offset_xyz
    cap_off = spec.model_origin_to_top_cap_center_offset_xyz
    dx_g, dy_g, dz_g = offset_local_to_world_delta(geom_off, yaw)
    dx_c, dy_c, dz_c = offset_local_to_world_delta(cap_off, yaw)
    cap_to_geom_local = spec.geometry_center_to_cap_center_offset_local_xyz
    offset_long_m, offset_short_m = _decompose_offset_local_along_axes(
        cap_to_geom_local,
        local_length_axis=str(spec.local_length_axis),
        local_width_axis=str(spec.local_width_axis),
    )

    height_m = float(entry.get("height_m") or geo.height_m)
    old_top_z = _safe_optional_float(entry.get("top_z_m"))
    if old_top_z is None:
        top_face_w = entry.get("gt_top_face_center_world")
        if isinstance(top_face_w, (list, tuple)) and len(top_face_w) >= 3:
            old_top_z = _safe_optional_float(top_face_w[2])

    offset_z_ignored = (
        pose_sem == "geometry_center"
        or old_top_z is not None
        or _input_z_looks_like_geometry_center(oz, height_m)
    )

    pose_is_geometry_center = pose_sem == "geometry_center" or (
        offset_z_ignored and _input_z_looks_like_geometry_center(oz, height_m)
    )
    if pose_is_geometry_center:
        geom_center_z = float(oz)
        origin_x = float(ox) - dx_g
        origin_y = float(oy) - dy_g
        origin_z = float(oz) - dz_g
    elif offset_z_ignored:
        origin_x = float(ox)
        origin_y = float(oy)
        origin_z = float(oz)
        geom_center_z = float(oz) + float(dz_g)
    else:
        origin_x = float(ox)
        origin_y = float(oy)
        origin_z = float(oz)
        geom_center_z = float(oz) + float(dz_g)

    cap_xy = (float(origin_x) + dx_c, float(origin_y) + dy_c)
    geom_xy = (float(origin_x) + dx_g, float(origin_y) + dy_g)

    if old_top_z is not None:
        top_z_m = float(old_top_z)
    else:
        top_z_m, _ = resolve_tall_object_top_z_m(
            label,
            geom_center_z,
            height_m=height_m,
            payload_top_z_before=old_top_z,
        )

    geom_center = (float(geom_xy[0]), float(geom_xy[1]), float(geom_center_z))
    cap_top_offset = (float(cap_xy[0]), float(cap_xy[1]), float(top_z_m))
    cap_top = cap_top_offset
    grasp_source = _cap_center_source_for_label(label)
    mustard_mode = _mustard_cap_center_mode(entry) if label == "mustard_bottle" else ""

    if label == "mustard_bottle":
        origin_xyz = (float(origin_x), float(origin_y), float(origin_z))
        vert_cap_top, fp_src = _mustard_vertical_axis_cap_top(
            entry,
            geom_center=geom_center,
            geom_xy=geom_xy,
            top_z_m=float(top_z_m),
            yaw=float(yaw),
        )
        entry["mustard_cap_center_mode"] = mustard_mode
        entry["mustard_footprint_center_source"] = fp_src
        entry["mustard_old_offset_cap_center_world"] = list(cap_top_offset)
        entry["mustard_vertical_axis_cap_center_world"] = list(vert_cap_top)
        entry["mustard_cap_center_local_m"] = list(spec.cap_center_local_m)

        if mustard_mode == MUSTARD_CAP_CENTER_MODE_VERTICAL_AXIS:
            cap_top = vert_cap_top
            grasp_source = MUSTARD_VERTICAL_AXIS_CAP_CENTER_SOURCE
            delta_xy = (
                float(cap_top[0]) - float(cap_top_offset[0]),
                float(cap_top[1]) - float(cap_top_offset[1]),
            )
            entry["mustard_cap_center_delta_xy_old_to_new"] = list(delta_xy)
            if logger is not None:
                try:
                    logger.info(
                        "[MUSTARD_CAP_CENTER_VERTICAL_AXIS]\n"
                        "mode=vertical_axis_from_footprint\n"
                        "footprint_center_source=%s\n"
                        "footprint_center_base=(%.4f, %.4f)\n"
                        "old_offset_cap_center_base=(%.4f, %.4f, %.4f)\n"
                        "new_vertical_cap_center_base=(%.4f, %.4f, %.4f)\n"
                        "delta_xy_old_to_new=(%.4f, %.4f)\n"
                        "top_z_m=%.4f\n"
                        "result=OK"
                        % (
                            fp_src,
                            float(vert_cap_top[0]),
                            float(vert_cap_top[1]),
                            float(cap_top_offset[0]),
                            float(cap_top_offset[1]),
                            float(cap_top_offset[2]),
                            float(cap_top[0]),
                            float(cap_top[1]),
                            float(cap_top[2]),
                            float(delta_xy[0]),
                            float(delta_xy[1]),
                            float(top_z_m),
                        )
                    )
                except Exception:
                    pass
        elif mustard_mode == MUSTARD_CAP_CENTER_MODE_MESH_LOCAL:
            cap_local = tuple(spec.cap_center_local_m)
            quat, model_pose_source = _resolve_mustard_model_pose_for_cap(
                entry, origin_xyz=origin_xyz, yaw_rad=float(yaw)
            )
            mesh_cap = compute_mustard_mesh_local_cap_center_world(
                origin_xyz,
                cap_local,
                quat_xyzw=quat,
                yaw_rad=None if quat is not None else float(yaw),
            )
            cap_top = mesh_cap
            grasp_source = MUSTARD_MESH_LOCAL_CAP_CENTER_SOURCE
            entry["mustard_mesh_local_cap_center_world"] = list(mesh_cap)
            delta_vert_xy = math.hypot(
                float(mesh_cap[0]) - float(vert_cap_top[0]),
                float(mesh_cap[1]) - float(vert_cap_top[1]),
            )
            delta_sdf_xy = math.hypot(
                float(mesh_cap[0]) - float(cap_top_offset[0]),
                float(mesh_cap[1]) - float(cap_top_offset[1]),
            )
            entry["mustard_cap_center_delta_xy_mesh_vs_vertical"] = float(
                delta_vert_xy
            )
            entry["mustard_cap_center_delta_xy_mesh_vs_sdf"] = float(delta_sdf_xy)
            if logger is not None:
                try:
                    logger.info(
                        "[MUSTARD_CAP_CENTER_MESH_LOCAL]\n"
                        "result=OK\n"
                        "mode=mesh_local_cap_center\n"
                        "p_cap_local=[%.4f,%.4f,%.4f]\n"
                        "model_pose_source=%s\n"
                        "model_xyz=(%.4f, %.4f, %.4f)\n"
                        "model_yaw_deg=%.2f\n"
                        "cap_center_world=(%.4f, %.4f, %.4f)\n"
                        "vertical_axis_center_world=(%.4f, %.4f, %.4f)\n"
                        "sdf_offset_center_world=(%.4f, %.4f, %.4f)\n"
                        "delta_mesh_vs_vertical_xy_m=%.4f\n"
                        "delta_mesh_vs_sdf_xy_m=%.4f\n"
                        "grasp_center_source=%s"
                        % (
                            float(cap_local[0]),
                            float(cap_local[1]),
                            float(cap_local[2]),
                            model_pose_source,
                            float(origin_x),
                            float(origin_y),
                            float(origin_z),
                            math.degrees(float(yaw)),
                            float(mesh_cap[0]),
                            float(mesh_cap[1]),
                            float(mesh_cap[2]),
                            float(vert_cap_top[0]),
                            float(vert_cap_top[1]),
                            float(vert_cap_top[2]),
                            float(cap_top_offset[0]),
                            float(cap_top_offset[1]),
                            float(cap_top_offset[2]),
                            delta_vert_xy,
                            delta_sdf_xy,
                            grasp_source,
                        )
                    )
                except Exception:
                    pass
        else:
            cap_top = cap_top_offset
            grasp_source = MUSTARD_TOP_CAP_CENTER_GEOMETRY_SOURCE

    old_grasp = (
        entry.get("grasp_center_base")
        or entry.get("chosen_target_center_base")
        or [ox, oy, old_top_z if old_top_z is not None else top_z_m]
    )
    if isinstance(old_grasp, (list, tuple)) and len(old_grasp) >= 2:
        old_xy = (float(old_grasp[0]), float(old_grasp[1]))
        old_z = (
            float(old_grasp[2])
            if len(old_grasp) >= 3
            else (old_top_z if old_top_z is not None else top_z_m)
        )
    else:
        old_xy = (ox, oy)
        old_z = old_top_z if old_top_z is not None else top_z_m
    shift_xy = math.hypot(cap_top[0] - old_xy[0], cap_top[1] - old_xy[1])
    shift_z = float(top_z_m) - float(old_z)

    entry["model_origin_pose_world"] = [
        float(origin_x),
        float(origin_y),
        float(origin_z),
    ]

    entry["semantic_box_center_world"] = list(geom_center)
    entry["gt_geometry_center_world"] = list(geom_center)
    entry["gt_top_face_center_world"] = list(cap_top)
    entry["top_z_m"] = float(top_z_m)
    entry["grasp_center_base"] = list(cap_top)
    entry["grasp_center_source"] = str(grasp_source)
    entry["chosen_target_center_base"] = list(cap_top)
    entry["top_surface_center_base"] = list(cap_top)
    entry["tall_object_sdf_offset_applied"] = True
    entry["tall_object_sdf_xy_only"] = False
    entry["tall_object_body_center_world"] = list(geom_center)
    entry["tall_object_sdf_geometry_center_offset_local"] = list(geom_off)
    entry["tall_object_sdf_cap_center_offset_local"] = list(cap_off)
    entry["mustard_top_cap_center_offset_local_xyz"] = list(cap_off)
    entry["geometry_center_to_cap_center_offset_local_xyz"] = list(cap_to_geom_local)

    if logger is not None:
        try:
            if label == "mustard_bottle":
                logger.info(
                    "[MUSTARD_CAP_CENTER_GEOMETRY]\n"
                    "model_origin_world=(%.4f, %.4f, %.4f)\n"
                    "body_center_world=(%.4f, %.4f, %.4f)\n"
                    "cap_center_offset_local_xyz=(%.4f, %.4f, %.4f)\n"
                    "geometry_center_offset_local_xyz=(%.4f, %.4f, %.4f)\n"
                    "computed_cap_center_world=(%.4f, %.4f, %.4f)\n"
                    "offset_long_m=%.4f\n"
                    "offset_short_m=%.4f\n"
                    "local_length_axis=%s\n"
                    "local_width_axis=%s\n"
                    "grasp_center_source=%s\n"
                    "result=OK"
                    % (
                        float(origin_x),
                        float(origin_y),
                        float(oz),
                        float(geom_center[0]),
                        float(geom_center[1]),
                        float(geom_center[2]),
                        float(cap_off[0]),
                        float(cap_off[1]),
                        float(cap_off[2]),
                        float(geom_off[0]),
                        float(geom_off[1]),
                        float(geom_off[2]),
                        float(cap_top[0]),
                        float(cap_top[1]),
                        float(cap_top[2]),
                        float(offset_long_m),
                        float(offset_short_m),
                        str(spec.local_length_axis),
                        str(spec.local_width_axis),
                        entry["grasp_center_source"],
                    )
                )
            logger.info(
                "[MUSTARD_SDF_OFFSET_CORRECTION]\n"
                "label=%s\n"
                "input_pose_semantics=%s\n"
                "input_z=%.4f\n"
                "old_top_z_m=%s\n"
                "cap_offset_local_xyz=(%.4f,%.4f,%.4f)\n"
                "geom_offset_local_xyz=(%.4f,%.4f,%.4f)\n"
                "offset_z_ignored_for_operational_top=%s\n"
                "old_grasp_center_base=(%.4f,%.4f,%.4f)\n"
                "new_grasp_center_base=(%.4f,%.4f,%.4f)\n"
                "new_top_z_m=%.4f\n"
                "xy_shift_m=%.4f\n"
                "z_shift_m=%.4f\n"
                "grasp_center_source=%s\n"
                "result=APPLIED"
                % (
                    label,
                    pose_sem,
                    oz,
                    "n/a" if old_top_z is None else "%.4f" % float(old_top_z),
                    float(cap_off[0]),
                    float(cap_off[1]),
                    float(cap_off[2]),
                    float(geom_off[0]),
                    float(geom_off[1]),
                    float(geom_off[2]),
                    str(offset_z_ignored).lower(),
                    old_xy[0],
                    old_xy[1],
                    float(old_z),
                    cap_top[0],
                    cap_top[1],
                    cap_top[2],
                    float(top_z_m),
                    shift_xy,
                    shift_z,
                    entry["grasp_center_source"],
                )
            )
        except Exception:
            pass
    return entry


def _safe_optional_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


@dataclass(frozen=True)
class KnownObjectGeometrySpec:
    label: str
    shape: str
    dims_lwh: Tuple[float, float, float]
    local_length_axis: str
    local_width_axis: str
    height_m: float
    yaw_offset_rad: float
    collision_margin_m: float
    max_gripper_width_required_m: float
    grasp_policy: str


def _shape_from_db(label: str) -> str:
    policy = get_grasp_policy(label)
    shape = str(policy.get("shape", "unknown")).strip().lower()
    if shape in ("low_box", "low_box_wide", "low_cylinder", "cylinder_wide"):
        if "cylinder" in shape:
            return "cylinder"
        return "box"
    return shape


def _dims_lwh_for_label(label: str) -> Tuple[float, float, float]:
    lb = normalize_label(label)
    box_spec = get_known_box_gt_spec(lb)
    if box_spec is not None:
        return box_spec.dims_lwh_m
    policy = get_grasp_policy(lb)
    maj = policy.get("footprint_major_m")
    minr = policy.get("footprint_minor_m")
    h = policy.get("db_height_m") or policy.get("object_height_m")
    if maj is not None and minr is not None and h is not None:
        try:
            return (float(maj), float(minr), float(h))
        except (TypeError, ValueError):
            pass
    if "dims" in policy:
        d = policy["dims"]
        return (float(d[0]), float(d[1]), float(d[2]))
    if "diameter" in policy and "height" in policy:
        dia = float(policy["diameter"])
        h = float(policy["height"])
        return (dia, dia, h)
    return (0.05, 0.05, 0.05)


def _build_registry_entry(label: str) -> KnownObjectGeometrySpec:
    lb = normalize_label(label)
    box_spec = get_known_box_gt_spec(lb)
    policy = get_grasp_policy(lb)
    dims = _dims_lwh_for_label(lb)
    if box_spec is not None:
        return KnownObjectGeometrySpec(
            label=lb,
            shape="box",
            dims_lwh=dims,
            local_length_axis=box_spec.local_length_axis,
            local_width_axis=box_spec.local_width_axis,
            height_m=float(box_spec.height_m),
            yaw_offset_rad=float(box_spec.yaw_offset_rad),
            collision_margin_m=DEFAULT_COLLISION_MARGIN_M,
            max_gripper_width_required_m=float(policy.get("required_grasp_width_m") or 0.0),
            grasp_policy=str(policy.get("primary_strategy", "top_down_short_axis")),
        )
    shape = _shape_from_db(lb)
    tall_spec = get_known_tall_object_sdf_spec(lb)
    local_length_axis = "x"
    local_width_axis = "y"
    if tall_spec is not None:
        local_length_axis = str(tall_spec.local_length_axis)
        local_width_axis = str(tall_spec.local_width_axis)
    return KnownObjectGeometrySpec(
        label=lb,
        shape=shape,
        dims_lwh=dims,
        local_length_axis=local_length_axis,
        local_width_axis=local_width_axis,
        height_m=float(dims[2]),
        yaw_offset_rad=0.0,
        collision_margin_m=DEFAULT_COLLISION_MARGIN_M,
        max_gripper_width_required_m=float(policy.get("required_grasp_width_m") or 0.0),
        grasp_policy=str(policy.get("primary_strategy", "unknown")),
    )


KNOWN_OBJECT_GEOMETRY: Dict[str, KnownObjectGeometrySpec] = {
    normalize_label(lb): _build_registry_entry(lb)
    for lb in set(list(KNOWN_BOX_GT_SPECS.keys()) + [
        "cracker_box", "sugar_box", "gelatin_box", "pudding_box",
        "mustard_bottle", "chips_can", "bleach_cleanser", "apple", "banana",
        "tuna_fish_can", "potted_meat_can", "master_chef_can",
    ])
}


def get_known_object_geometry(label: str) -> Optional[KnownObjectGeometrySpec]:
    return KNOWN_OBJECT_GEOMETRY.get(normalize_label(str(label).strip().lower()))


def _inflate_box_dims(
    dims_lwh: Tuple[float, float, float], margin_m: float
) -> List[float]:
    m = float(margin_m)
    return [float(dims_lwh[0]) + 2.0 * m, float(dims_lwh[1]) + 2.0 * m, float(dims_lwh[2]) + 2.0 * m]


def enrich_runtime_scene_object_fields(
    entry: Dict[str, Any],
    *,
    role: Optional[str] = None,
    collision_margin_m: Optional[float] = None,
    logger: Any = None,
) -> Dict[str, Any]:
    """Añade role, colisión inflada y campos operativos a una entrada GT."""
    label = str(entry.get("label", "")).strip().lower()
    geo = get_known_object_geometry(label)
    if role is not None:
        entry["role"] = str(role).strip().lower()
    elif "role" not in entry:
        entry["role"] = ROLE_UNKNOWN

    margin = float(
        collision_margin_m
        if collision_margin_m is not None
        else (geo.collision_margin_m if geo else DEFAULT_COLLISION_MARGIN_M)
    )
    entry["collision_margin"] = margin

    if geo is not None:
        entry["shape"] = geo.shape
        entry["dims_lwh"] = list(geo.dims_lwh)
        entry["local_length_axis"] = geo.local_length_axis
        entry["local_width_axis"] = geo.local_width_axis
        entry["height_m"] = float(geo.height_m)
        entry["yaw_offset_rad"] = float(geo.yaw_offset_rad)
        entry["max_gripper_width_required"] = float(geo.max_gripper_width_required_m)
        entry["grasp_policy"] = geo.grasp_policy

    sem = entry.get("semantic_box_center_world") or entry.get("gt_geometry_center_world")
    if not (isinstance(sem, (list, tuple)) and len(sem) >= 3):
        pw = entry.get("pose_world")
        if isinstance(pw, dict):
            sem = [float(pw["x"]), float(pw["y"]), float(pw["z"])]
    yaw = float(entry.get("gt_yaw_rad", entry.get("yaw_rad", 0.0)))

    if is_known_spawn_geometry_box_label(label) and isinstance(sem, (list, tuple)) and len(sem) >= 3:
        geom = compute_known_box_gt_geometry(
            label,
            (float(sem[0]), float(sem[1]), float(sem[2])),
            float(yaw) - float(entry.get("yaw_offset_rad", 0.0) or 0.0),
            logger=logger,
        )
        if geom:
            entry.update(geom)

    if geo is not None and isinstance(sem, (list, tuple)) and len(sem) >= 3:
        dims = tuple(entry.get("dims_used_lwh") or geo.dims_lwh)
        entry["collision_shape"] = "box" if geo.shape in ("box", "low_box", "low_box_wide", "curved_long") else geo.shape
        entry["collision_dims"] = [float(dims[0]), float(dims[1]), float(dims[2])]
        entry["collision_dims_inflated"] = _inflate_box_dims(dims, margin)
        entry["required_gripper_width"] = float(
            entry.get("required_gripper_width")
            or entry.get("max_gripper_width_required")
            or geo.max_gripper_width_required_m
        )
        entry["collision_box_pose"] = {
            "x": float(sem[0]),
            "y": float(sem[1]),
            "z": float(sem[2]),
            "yaw": float(entry.get("gt_yaw_rad", yaw)),
        }
        if entry.get("gt_length_axis_world") is not None:
            entry["length_axis_world"] = list(entry["gt_length_axis_world"])
        if entry.get("gt_width_axis_world") is not None:
            entry["width_axis_world"] = list(entry["gt_width_axis_world"])
        if entry.get("gt_closing_axis_world") is not None:
            entry["closing_axis_world"] = list(entry["gt_closing_axis_world"])

    col = get_collision_dimensions(label, padding_m=margin)
    if col is not None:
        entry["collision_dims_moveit"] = col

    if (
        geo is not None
        and str(geo.grasp_policy) == "tall_object_topdown"
        and get_known_tall_object_sdf_spec(label) is not None
    ):
        entry = apply_tall_object_sdf_geometry_correction(entry, logger=logger)
        if isinstance(sem, (list, tuple)) and len(sem) >= 3:
            tall_geom = compute_tall_object_runtime_gt_geometry(
                label,
                (float(sem[0]), float(sem[1]), float(sem[2])),
                float(yaw),
                dims_lwh_m=tuple(geo.dims_lwh),
                local_length_axis=str(geo.local_length_axis),
                local_width_axis=str(geo.local_width_axis),
                logger=logger,
            )
            if tall_geom:
                sdf_spec = get_known_tall_object_sdf_spec(label)
                if entry.get("tall_object_sdf_offset_applied") and sdf_spec is not None:
                    preserve_keys = (
                        "gt_top_face_center_world",
                        "gt_top_face_center_base",
                        "grasp_center_base",
                        "grasp_center_source",
                        "chosen_target_center_base",
                        "top_surface_center_base",
                        "top_z_m",
                        "mustard_top_cap_center_offset_local_xyz",
                        "geometry_center_to_cap_center_offset_local_xyz",
                        "mustard_cap_center_mode",
                        "mustard_footprint_center_source",
                        "mustard_old_offset_cap_center_world",
                        "mustard_vertical_axis_cap_center_world",
                        "mustard_mesh_local_cap_center_world",
                        "mustard_cap_center_local_m",
                        "mustard_cap_center_delta_xy_old_to_new",
                        "mustard_cap_center_delta_xy_mesh_vs_vertical",
                        "mustard_cap_center_delta_xy_mesh_vs_sdf",
                    )
                    preserved = {
                        k: entry[k] for k in preserve_keys if k in entry
                    }
                    entry.update(tall_geom)
                    entry.update(preserved)
                else:
                    entry.update(tall_geom)

    if geo is not None and entry.get("gt_closing_axis_world") is None:
        spawn_yaw = float(entry.get("gt_yaw_rad", entry.get("yaw_rad", 0.0)))
        axes = resolve_runtime_gt_spawn_axes(
            spawn_yaw,
            local_length_axis=str(geo.local_length_axis),
            local_width_axis=str(geo.local_width_axis),
        )
        entry.update(axes)

    if logger is not None:
        try:
            logger.info(
                "[RUNTIME_SCENE_OBJECT] label=%s entity=%s role=%s shape=%s "
                "dims_lwh=%s collision_margin=%.4f required_gripper_width=%.4f "
                "grasp_policy=%s yaw=%.3fdeg"
                % (
                    label,
                    str(entry.get("entity_name", "")),
                    str(entry.get("role", ROLE_UNKNOWN)),
                    str(entry.get("shape", "")),
                    entry.get("dims_lwh"),
                    margin,
                    float(entry.get("required_gripper_width") or 0.0),
                    str(entry.get("grasp_policy", "")),
                    math.degrees(float(entry.get("gt_yaw_rad", yaw))),
                )
            )
        except Exception:
            pass
    return entry
