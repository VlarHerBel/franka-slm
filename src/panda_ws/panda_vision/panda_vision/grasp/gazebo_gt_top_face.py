"""Ground-truth top face desde pose Gazebo (solo visualización / métricas de depuración)."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import tf_transformations

from panda_vision.grasp.known_object_pose_fit import _wrap_pi
from panda_vision.grasp.model_box_top_face_fit import _build_top_corners_base
from panda_vision.grasp.object_grasp_policy import OBJECT_DB, get_collision_dimensions, normalize_label
from panda_vision.spawn.gz_spawn_runtime import filter_runtime_entity_names
from panda_vision.spawn.runtime_scene_gt_geometry import (
    enrich_gt_object_entry,
    enrich_gt_object_entry_base,
    is_known_spawn_geometry_box_label,
)


Pose6 = Tuple[float, float, float, float, float, float]


def parse_runtime_label(entity_name: str, prefix: str = "runtime_ycb") -> Optional[str]:
    """Extrae label YCB de ``runtime_ycb_<label>`` o ``runtime_ycb_<label>_seedN``."""
    p = prefix.strip()
    if not p:
        return None
    short = entity_name.split("::")[-1].strip()
    if not short.startswith(f"{p}_"):
        return None
    rest = short[len(p) + 1 :]
    if not rest:
        return None
    parts = rest.split("_")
    known = set(OBJECT_DB.keys())
    for n in range(len(parts), 0, -1):
        cand = "_".join(parts[:n]).lower()
        if cand in known:
            return cand
    return parts[0].lower() if parts else None


def _ycb_link_box_dims(label: str) -> Optional[Tuple[float, float, float]]:
    """Dimensiones en orden link SDF: (width_x, length_y, height_z)."""
    key = normalize_label(label)
    entry = OBJECT_DB.get(key)
    if entry is None:
        return None
    dims = entry.get("dims")
    if not isinstance(dims, (list, tuple)) or len(dims) < 3:
        return None
    return float(dims[0]), float(dims[1]), float(dims[2])


def _link_top_corners(
    width_m: float,
    length_m: float,
    height_m: float,
    *,
    box_center_z_m: Optional[float] = None,
) -> np.ndarray:
    """Cuatro esquinas de la cara superior en frame del link (eje Z arriba)."""
    hw = 0.5 * float(width_m)
    hl = 0.5 * float(length_m)
    hh = 0.5 * float(height_m)
    cz = float(box_center_z_m) if box_center_z_m is not None else hh
    cx, cy = 0.0, 0.0
    return np.array(
        [
            [cx + hw, cy + hl, cz + hh],
            [cx + hw, cy - hl, cz + hh],
            [cx - hw, cy - hl, cz + hh],
            [cx - hw, cy + hl, cz + hh],
        ],
        dtype=np.float64,
    )


def pose6_to_matrix(pose6: Pose6) -> np.ndarray:
    x, y, z, roll, pitch, yaw = pose6
    m = tf_transformations.euler_matrix(roll, pitch, yaw, axes="sxyz")
    m[:3, 3] = [float(x), float(y), float(z)]
    return m


def transform_points(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if pts.size == 0:
        return pts
    hom = np.hstack((pts, np.ones((pts.shape[0], 1), dtype=np.float64)))
    return (matrix @ hom.T).T[:, :3]


def build_gt_top_face_corners_base(
    pose6_world: Pose6,
    label: str,
    world_to_base_matrix: np.ndarray,
) -> Optional[List[List[float]]]:
    """Cara superior real del cuboide YCB en ``target_frame`` (p. ej. panda_link0)."""
    dims = _ycb_link_box_dims(label)
    if dims is None:
        return None
    w, ln, h = dims
    corners_link = _link_top_corners(w, ln, h, box_center_z_m=0.5 * h)
    t_world = pose6_to_matrix(pose6_world)
    corners_world = transform_points(corners_link, t_world)
    corners_base = transform_points(corners_world, world_to_base_matrix)
    return [[float(p[0]), float(p[1]), float(p[2])] for p in corners_base]


def find_runtime_entity_for_label(
    world_poses: Dict[str, Pose6],
    label: str,
    *,
    entity_prefix: str = "runtime_ycb",
) -> Optional[Tuple[str, Pose6]]:
    names = filter_runtime_entity_names(world_poses.keys(), entity_prefix, label=label)
    if not names:
        names = [
            n
            for n in world_poses
            if parse_runtime_label(n.split("::")[-1], entity_prefix) == normalize_label(label)
        ]
        names = sorted(set(names))
    if not names:
        return None
    entity = names[0]
    pose = world_poses.get(entity) or world_poses.get(entity.split("::")[-1])
    if pose is None:
        return None
    return entity, pose


def _center_and_yaw_from_corners(corners: List[List[float]]) -> Tuple[np.ndarray, float]:
    mc = np.asarray([[float(c[0]), float(c[1])] for c in corners[:4]], dtype=np.float64)
    center = np.mean(mc, axis=0)
    e0 = mc[1] - mc[0]
    yaw = math.atan2(float(e0[1]), float(e0[0]))
    return center, float(yaw)


def _corner_rmse_m(a: List[List[float]], b: List[List[float]]) -> float:
    if len(a) < 4 or len(b) < 4:
        return float("nan")
    pa = np.asarray(a[:4], dtype=np.float64)[:, :3]
    pb = np.asarray(b[:4], dtype=np.float64)[:, :3]
    d = np.linalg.norm(pa - pb, axis=1)
    return float(np.sqrt(np.mean(d * d)))


def _yaw_error_deg(yaw_a: float, yaw_b: float) -> float:
    d = abs(math.degrees(_wrap_pi(yaw_a - yaw_b)))
    if d > 90.0:
        d = 180.0 - d
    return float(d)


def _center_xy_error_m(
    corners_a: List[List[float]], corners_b: List[List[float]]
) -> float:
    ca, _ = _center_and_yaw_from_corners(corners_a)
    cb, _ = _center_and_yaw_from_corners(corners_b)
    return float(np.linalg.norm(ca[:2] - cb[:2]))


def spawner_entry_to_pose6(entry: Dict[str, Any]) -> Optional[Pose6]:
    pw = entry.get("pose_world")
    if not isinstance(pw, dict):
        return None
    try:
        return (
            float(pw["x"]),
            float(pw["y"]),
            float(pw["z"]),
            float(pw.get("roll", 0.0)),
            float(pw.get("pitch", 0.0)),
            float(pw.get("yaw", float(entry.get("yaw_rad", 0.0)))),
        )
    except (KeyError, TypeError, ValueError):
        return None


def build_gt_top_face_from_spawner_entry(
    entry: Dict[str, Any],
    world_to_base_matrix: np.ndarray,
) -> Optional[List[List[float]]]:
    """Top face GT en base: usa campos precalculados (sin offsets en percepción)."""
    pre = entry.get("gt_top_face_corners_base")
    if isinstance(pre, list) and len(pre) >= 4:
        return [[float(c[0]), float(c[1]), float(c[2])] for c in pre[:4]]

    label = str(entry.get("label", "")).strip()
    if is_known_spawn_geometry_box_label(label):
        enrich_gt_object_entry(entry)
        enrich_gt_object_entry_base(entry, world_to_base_matrix)
        pre = entry.get("gt_top_face_corners_base")
        if isinstance(pre, list) and len(pre) >= 4:
            return [[float(c[0]), float(c[1]), float(c[2])] for c in pre[:4]]

    pose6 = spawner_entry_to_pose6(entry)
    if pose6 is None:
        return None
    parsed = _ycb_link_box_dims(label)
    if parsed is None:
        return None
    w, ln, h = parsed
    corners_link = _link_top_corners(w, ln, h, box_center_z_m=0.5 * h)
    t_world = pose6_to_matrix(pose6)
    corners_world = transform_points(corners_link, t_world)
    corners_base = transform_points(corners_world, world_to_base_matrix)
    corners_out = [[float(p[0]), float(p[1]), float(p[2])] for p in corners_base]
    entry["gt_top_face_corners_base"] = corners_out
    return corners_out


def compare_model_vs_gt(
    model_corners_base: List[List[float]],
    gt_corners_base: List[List[float]],
    *,
    fx: float,
    z_est_m: float,
) -> Dict[str, Any]:
    """Métricas modelo vs ground truth (base y píxeles aproximados)."""
    mc, myaw = _center_and_yaw_from_corners(model_corners_base)
    gc, gyaw = _center_and_yaw_from_corners(gt_corners_base)
    center_err_xy = _center_xy_error_m(model_corners_base, gt_corners_base)
    yaw_err = _yaw_error_deg(myaw, gyaw)
    corner_rmse_m = _corner_rmse_m(model_corners_base, gt_corners_base)
    corner_rmse_px = float("nan")
    if math.isfinite(corner_rmse_m) and z_est_m > 0.05 and fx > 1.0:
        corner_rmse_px = float(corner_rmse_m) * float(fx) / float(z_est_m)
    return {
        "model_vs_gt_center_error_xy_m": center_err_xy,
        "model_vs_gt_center_error_m": center_err_xy,
        "model_vs_gt_yaw_error_deg": yaw_err,
        "model_vs_gt_corner_error_m": corner_rmse_m,
        "model_vs_gt_corner_rmse_m": corner_rmse_m,
        "model_vs_gt_corner_rmse_px": corner_rmse_px,
    }


def compute_top_face_gt_metrics(
    *,
    observed_corners_base: Optional[List[List[float]]],
    model_corners_base: Optional[List[List[float]]],
    gt_corners_base: Optional[List[List[float]]],
    fx: float,
    z_est_m: float,
) -> Dict[str, Any]:
    """Métricas cruzadas observed/model/gt en base (solo diagnóstico)."""
    out: Dict[str, Any] = {}
    z = max(float(z_est_m), 0.25)
    if (
        isinstance(model_corners_base, list)
        and len(model_corners_base) >= 4
        and isinstance(gt_corners_base, list)
        and len(gt_corners_base) >= 4
    ):
        out.update(compare_model_vs_gt(model_corners_base, gt_corners_base, fx=fx, z_est_m=z))
    if (
        isinstance(observed_corners_base, list)
        and len(observed_corners_base) >= 4
        and isinstance(gt_corners_base, list)
        and len(gt_corners_base) >= 4
    ):
        out["observed_vs_gt_center_error_xy_m"] = _center_xy_error_m(
            observed_corners_base, gt_corners_base
        )
        out["observed_vs_gt_corner_error_m"] = _corner_rmse_m(
            observed_corners_base, gt_corners_base
        )
    if (
        isinstance(observed_corners_base, list)
        and len(observed_corners_base) >= 4
        and isinstance(model_corners_base, list)
        and len(model_corners_base) >= 4
    ):
        out["observed_vs_model_center_error_xy_m"] = _center_xy_error_m(
            observed_corners_base, model_corners_base
        )
        out["observed_vs_model_corner_error_m"] = _corner_rmse_m(
            observed_corners_base, model_corners_base
        )
    return out


def model_top_face_mask_coherence(
    model_corners_base: List[List[float]],
    det: Any,
    project_uv_fn: Any,
    *,
    obb_center_uv: Optional[List[float]] = None,
) -> Tuple[bool, str, float]:
    """Comprueba coherencia 2D del model top face con máscara/OBB."""
    import cv2

    uvs: List[Tuple[int, int]] = []
    for c in model_corners_base[:4]:
        uv = project_uv_fn(c)
        if uv is not None:
            uvs.append(uv)
    if len(uvs) < 4:
        return False, "model_uv_projection_incomplete", float("inf")

    model_center_uv = (
        int(np.mean([u[0] for u in uvs])),
        int(np.mean([u[1] for u in uvs])),
    )
    if det.obb_polygon_uv is not None and det.obb_polygon_uv.shape[0] >= 3:
        poly = det.obb_polygon_uv.astype(np.float32)
        inside = cv2.pointPolygonTest(poly, model_center_uv, False) >= 0
        if not inside:
            if obb_center_uv is not None:
                d = math.hypot(
                    model_center_uv[0] - float(obb_center_uv[0]),
                    model_center_uv[1] - float(obb_center_uv[1]),
                )
                return False, "model_center_outside_obb", float(d)
            return False, "model_center_outside_obb", float("inf")
        return True, "ok_obb", 0.0

    if det.bbox_xyxy is not None:
        x1, y1, x2, y2 = [float(v) for v in det.bbox_xyxy]
        u, v = model_center_uv
        if not (x1 <= u <= x2 and y1 <= v <= y2):
            return False, "model_center_outside_bbox", float("inf")
        return True, "ok_bbox", 0.0

    return True, "no_2d_reference", 0.0


def evaluate_visual_pose_gate(
    metrics: Dict[str, Any],
    mask_ok: bool,
    mask_reason: str,
    *,
    enable: bool,
    max_model_vs_gt_center_xy_m: float = 0.015,
    max_observed_vs_model_center_xy_m: float = 0.025,
    require_mask_coherence: bool = True,
) -> Tuple[bool, str]:
    if not enable:
        return True, "visual_gate_disabled"
    m_gt = metrics.get("model_vs_gt_center_error_xy_m")
    if m_gt is not None and math.isfinite(float(m_gt)):
        if float(m_gt) > float(max_model_vs_gt_center_xy_m):
            return (
                False,
                "model_vs_gt_center_error_xy_m=%.4f>%.4f"
                % (float(m_gt), max_model_vs_gt_center_xy_m),
            )
    m_om = metrics.get("observed_vs_model_center_error_xy_m")
    if m_om is not None and math.isfinite(float(m_om)):
        if float(m_om) > float(max_observed_vs_model_center_xy_m):
            return (
                False,
                "observed_vs_model_center_error_xy_m=%.4f>%.4f"
                % (float(m_om), max_observed_vs_model_center_xy_m),
            )
    if require_mask_coherence and not mask_ok:
        return False, "mask_coherence_%s" % mask_reason
    return True, "ok"


def log_pose_gate_visual(
    logger: Any,
    *,
    label: str,
    accepted: bool,
    reason: str,
    metrics: Dict[str, Any],
) -> None:
    try:
        logger.info(
            "[POSE_GATE_VISUAL] label=%s accepted=%s reason=%s "
            "model_vs_gt_ctr_xy=%.4f model_vs_gt_yaw=%.2f "
            "obs_vs_model_ctr_xy=%.4f obs_vs_gt_ctr_xy=%.4f"
            % (
                label,
                str(accepted).lower(),
                reason,
                float(metrics.get("model_vs_gt_center_error_xy_m", float("nan"))),
                float(metrics.get("model_vs_gt_yaw_error_deg", float("nan"))),
                float(metrics.get("observed_vs_model_center_error_xy_m", float("nan"))),
                float(metrics.get("observed_vs_gt_center_error_xy_m", float("nan"))),
            )
        )
    except Exception:
        pass


def _project_corners_rmse_px(
    corners_a: List[List[float]],
    corners_b: List[List[float]],
    project_uv_fn: Any,
) -> float:
    uvs_a: List[Tuple[int, int]] = []
    uvs_b: List[Tuple[int, int]] = []
    for c in corners_a[:4]:
        uv = project_uv_fn(c)
        if uv is not None:
            uvs_a.append(uv)
    for c in corners_b[:4]:
        uv = project_uv_fn(c)
        if uv is not None:
            uvs_b.append(uv)
    if len(uvs_a) < 4 or len(uvs_b) < 4:
        return float("nan")
    pa = np.asarray(uvs_a, dtype=np.float64)
    pb = np.asarray(uvs_b, dtype=np.float64)
    d = np.linalg.norm(pa - pb, axis=1)
    return float(np.sqrt(np.mean(d * d)))


def check_model_dim_convention(
    gt_corners_base: List[List[float]],
    model_corners_base: List[List[float]],
    label: str,
    *,
    project_uv_fn: Any,
) -> Dict[str, Any]:
    """Prueba permutaciones L/W/yaw del fit modelo frente a GT (solo diagnóstico)."""
    dims = _ycb_link_box_dims(label)
    if dims is None:
        return {}
    d0, d1, h = dims
    L_db = max(d0, d1)
    W_db = min(d0, d1)
    mc, model_yaw = _center_and_yaw_from_corners(model_corners_base)
    top_z = float(np.mean([float(c[2]) for c in model_corners_base[:4]]))

    variants: List[Tuple[str, float, float, float]] = [
        ("normal", L_db, W_db, model_yaw),
        ("swap_lw", W_db, L_db, model_yaw),
        ("yaw_plus_90", L_db, W_db, _wrap_pi(model_yaw + math.pi / 2.0)),
        ("swap_lw_yaw_plus_90", W_db, L_db, _wrap_pi(model_yaw + math.pi / 2.0)),
        ("yaw_plus_180", L_db, W_db, _wrap_pi(model_yaw + math.pi)),
    ]

    current_rmse_px = float("nan")
    best_rmse_px = float("inf")
    best_variant = "normal"

    for name, L, W, yaw in variants:
        al = np.array([math.cos(yaw), math.sin(yaw)], dtype=np.float64)
        cand = _build_top_corners_base(mc, al, L, W, top_z)
        rmse_px = _project_corners_rmse_px(cand, gt_corners_base, project_uv_fn)
        if name == "normal" and math.isfinite(rmse_px):
            current_rmse_px = rmse_px
        if math.isfinite(rmse_px) and rmse_px < best_rmse_px:
            best_rmse_px = rmse_px
            best_variant = name

    warning = ""
    if (
        math.isfinite(current_rmse_px)
        and math.isfinite(best_rmse_px)
        and best_rmse_px + 2.0 < current_rmse_px
        and best_variant != "normal"
    ):
        warning = "dimension_axis_mismatch_possible"

    return {
        "current_dims": (L_db, W_db, h),
        "best_variant": best_variant,
        "current_rmse_px": current_rmse_px,
        "best_rmse_px": best_rmse_px if math.isfinite(best_rmse_px) else float("nan"),
        "warning": warning,
    }


def log_top_face_gt_compare(
    logger: Any,
    *,
    label: str,
    entity: str,
    metrics: Dict[str, Any],
    pose_meta: Dict[str, Any],
) -> None:
    try:
        logger.info(
            "[TOP_FACE_GT_COMPARE] label=%s entity=%s "
            "model_vs_gt_center_error_xy_m=%.4f model_vs_gt_yaw_error_deg=%.2f "
            "model_vs_gt_corner_error_m=%.4f corner_rmse_px=%.1f "
            "observed_vs_model_center_error_xy_m=%.4f observed_vs_gt_center_error_xy_m=%.4f "
            "model_fit_error=%s yaw_confidence=%.3f top_face_source=%s"
            % (
                label,
                entity,
                float(metrics.get("model_vs_gt_center_error_xy_m", float("nan"))),
                float(metrics.get("model_vs_gt_yaw_error_deg", float("nan"))),
                float(metrics.get("model_vs_gt_corner_error_m", float("nan"))),
                float(metrics.get("model_vs_gt_corner_rmse_px", float("nan"))),
                float(metrics.get("observed_vs_model_center_error_xy_m", float("nan"))),
                float(metrics.get("observed_vs_gt_center_error_xy_m", float("nan"))),
                pose_meta.get("model_fit_error", "n/a"),
                float(pose_meta.get("yaw_confidence", 0.0)),
                pose_meta.get("top_face_source", ""),
            )
        )
    except Exception:
        pass


def log_model_dim_convention_check(logger: Any, label: str, diag: Dict[str, Any]) -> None:
    if not diag:
        return
    L, W, h = diag.get("current_dims", (0.0, 0.0, 0.0))
    try:
        logger.info(
            "[MODEL_DIM_CONVENTION_CHECK] label=%s "
            "current_dims=(L=%.3f, W=%.3f, H=%.3f) best_variant=%s "
            "current_rmse_px=%.1f best_rmse_px=%.1f warning=%s"
            % (
                label,
                float(L),
                float(W),
                float(h),
                diag.get("best_variant", ""),
                float(diag.get("current_rmse_px", float("nan"))),
                float(diag.get("best_rmse_px", float("nan"))),
                diag.get("warning", ""),
            )
        )
    except Exception:
        pass


def update_world_pose_cache(
    msg: Any,
    cache: Dict[str, Pose6],
    *,
    world_frame: str,
) -> None:
    """Actualiza ``cache`` desde tf2_msgs/TFMessage (SceneBroadcaster)."""
    world_norm = world_frame.strip().lstrip("/")

    def _matches_world(parent: str) -> bool:
        if not parent:
            return False
        norm = parent.strip().lstrip("/")
        return norm == world_norm or norm.endswith(f"/{world_norm}")

    new_map: Dict[str, Pose6] = {}
    for ts in msg.transforms:
        parent = (ts.header.frame_id or "").strip()
        if not _matches_world(parent):
            continue
        child = (ts.child_frame_id or "").strip()
        if not child:
            continue
        q = ts.transform.rotation
        t = ts.transform.translation
        try:
            roll, pitch, yaw = tf_transformations.euler_from_quaternion(
                (q.x, q.y, q.z, q.w), axes="sxyz"
            )
        except ValueError:
            continue
        pose6: Pose6 = (t.x, t.y, t.z, float(roll), float(pitch), float(yaw))
        short = child.split("::")[-1]
        new_map[short] = pose6
        new_map[child] = pose6
    if new_map:
        cache.clear()
        cache.update(new_map)
