"""Ajuste de top face y pose de cajas conocidas a partir del cuboide DB + nube segmentada.

Prioriza geometría del modelo frente a contornos observados contaminados por caras laterales.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from panda_vision.grasp.known_object_pose_fit import (
    _score_single_theta,
    _wrap_pi,
    fit_known_top_rectangle_pose,
)

BOX_LIKE_SHAPES = frozenset({"box", "low_box", "low_box_wide"})

# Umbrales de aceptación del modelo (caja conocida en sim).
MODEL_MIN_SEGMENTED_POINTS = 80
MODEL_MIN_TOP_SLAB_POINTS = 25
MODEL_MIN_YAW_CONFIDENCE = 0.65
MODEL_MAX_FIT_ERROR = 0.055
MODEL_MIN_INLIER_RATIO = 0.72
MODEL_MIN_EDGE_SUPPORT = 0.10
MODEL_MAX_LENGTH_ERROR_M = 0.035
MODEL_MAX_WIDTH_ERROR_M = 0.028
MODEL_MAX_OUTSIDE_ERROR_M = 0.018
MODEL_MAX_EXTENT_LENGTH_RATIO_ERR = 0.18
MODEL_MAX_EXTENT_WIDTH_RATIO_ERR = 0.22


def is_box_like_known_shape(shape: str) -> bool:
    return str(shape or "").strip().lower() in BOX_LIKE_SHAPES


def _db_length_width_height(db_dims: Tuple[float, float, float]) -> Tuple[float, float, float]:
    d0, d1, h = float(db_dims[0]), float(db_dims[1]), float(db_dims[2])
    return max(d0, d1), min(d0, d1), float(h)


def _build_top_corners_base(
    center_xy: np.ndarray,
    axis_long: np.ndarray,
    length_m: float,
    width_m: float,
    top_z_m: float,
) -> List[List[float]]:
    cx, cy = float(center_xy[0]), float(center_xy[1])
    hl = 0.5 * float(length_m)
    hw = 0.5 * float(width_m)
    bl = axis_long * hl
    bs = np.array([-axis_long[1], axis_long[0]], dtype=np.float64) * hw
    corners_2d = [
        np.array([cx, cy]) + bl + bs,
        np.array([cx, cy]) + bl - bs,
        np.array([cx, cy]) - bl - bs,
        np.array([cx, cy]) - bl + bs,
    ]
    return [[float(p[0]), float(p[1]), float(top_z_m)] for p in corners_2d]


def _extract_top_slab_points(
    points_base: np.ndarray,
    table_z_m: float,
    height_m: float,
    *,
    slab_below_top_m: float = 0.012,
    slab_above_top_m: float = 0.006,
    min_height_fraction: float = 0.55,
) -> Tuple[np.ndarray, float]:
    """Puntos de la franja superior del cuboide (evita cara lateral en la mayoría de yaw)."""
    pts = np.asarray(points_base, dtype=np.float64).reshape(-1, 3)
    if pts.shape[0] < 5:
        return np.empty((0, 3), dtype=np.float64), float(table_z_m + height_m)

    top_z_model = float(table_z_m) + float(height_m)
    z_obs_p95 = float(np.percentile(pts[:, 2], 95.0))
    if abs(z_obs_p95 - top_z_model) <= 0.04:
        top_z = 0.65 * top_z_model + 0.35 * z_obs_p95
    else:
        top_z = top_z_model

    z_min = top_z - float(slab_below_top_m)
    z_max = top_z + float(slab_above_top_m)
    z_floor = float(table_z_m) + float(height_m) * float(min_height_fraction)
    mask = (pts[:, 2] >= max(z_min, z_floor)) & (pts[:, 2] <= z_max)
    slab = pts[mask]
    return slab, float(top_z)


def _model_acceptance(
    fit_detail: Dict[str, Any],
    *,
    n_segmented: int,
    n_slab: int,
    top_face_point_ratio: float,
) -> Tuple[bool, str]:
    if n_segmented < MODEL_MIN_SEGMENTED_POINTS:
        return False, "too_few_segmented_points"
    if n_slab < MODEL_MIN_TOP_SLAB_POINTS:
        return False, "too_few_top_slab_points"
    if float(fit_detail.get("yaw_confidence", 0.0)) < MODEL_MIN_YAW_CONFIDENCE:
        return False, "yaw_confidence_low"
    if float(fit_detail.get("fit_error", 1.0)) > MODEL_MAX_FIT_ERROR:
        return False, "fit_error_high"
    if float(fit_detail.get("inlier_ratio", 0.0)) < MODEL_MIN_INLIER_RATIO:
        return False, "inlier_ratio_low"
    if float(fit_detail.get("edge_support_score", 0.0)) < MODEL_MIN_EDGE_SUPPORT:
        return False, "edge_support_low"
    if float(fit_detail.get("length_error_m", 1.0)) > MODEL_MAX_LENGTH_ERROR_M:
        return False, "length_error_high"
    if float(fit_detail.get("width_error_m", 1.0)) > MODEL_MAX_WIDTH_ERROR_M:
        return False, "width_error_high"
    if float(fit_detail.get("outside_error_m", 1.0)) > MODEL_MAX_OUTSIDE_ERROR_M:
        return False, "outside_error_high"
    db_l = float(fit_detail.get("db_length_m", 0.0))
    db_w = float(fit_detail.get("db_width_m", 0.0))
    obs_l = float(fit_detail.get("projected_extent_length_m", 0.0))
    obs_w = float(fit_detail.get("projected_extent_width_m", 0.0))
    if db_l > 1e-6 and abs(obs_l - db_l) / db_l > MODEL_MAX_EXTENT_LENGTH_RATIO_ERR:
        return False, "length_ratio_mismatch"
    if db_w > 1e-6 and abs(obs_w - db_w) / db_w > MODEL_MAX_EXTENT_WIDTH_RATIO_ERR:
        return False, "width_ratio_mismatch"
    if top_face_point_ratio < 0.35:
        return False, "top_slab_ratio_low"
    return True, "ok"


def fit_model_cuboid_top_face(
    segmented_points_base: np.ndarray,
    label: str,
    db_dims: Tuple[float, float, float],
    table_z_m: float,
    *,
    yaw_hint_rad: Optional[float] = None,
    top_z_hint_m: Optional[float] = None,
) -> Dict[str, Any]:
    """Estima pose del cuboide conocido y devuelve top face del modelo (4 esquinas superiores)."""
    L, W, h = _db_length_width_height(db_dims)
    empty: Dict[str, Any] = {
        "success": False,
        "model_top_face_success": False,
        "top_face_source": "observed",
        "model_fit_error": float("inf"),
        "model_box_yaw_rad": 0.0,
        "model_closing_yaw_rad": 0.0,
        "model_box_center_base": [0.0, 0.0, 0.0],
        "model_top_face_corners_base": [],
        "model_major_axis_xy": [1.0, 0.0],
        "model_minor_axis_xy": [0.0, 1.0],
        "model_bottom_corners_base": [],
        "db_length_m": L,
        "db_width_m": W,
        "height_m": h,
        "message": "",
        "num_segmented_points": 0,
        "num_top_slab_points": 0,
        "top_z_model_m": float(table_z_m) + h,
    }

    pts = np.asarray(segmented_points_base, dtype=np.float64).reshape(-1, 3)
    n_seg = int(pts.shape[0])
    empty["num_segmented_points"] = n_seg
    if n_seg < MODEL_MIN_SEGMENTED_POINTS:
        empty["message"] = "too_few_segmented_points"
        return empty

    slab_pts, top_z_model = _extract_top_slab_points(
        pts, table_z_m, h, slab_below_top_m=0.014, slab_above_top_m=0.008
    )
    if top_z_hint_m is not None:
        try:
            tz = float(top_z_hint_m)
            if abs(tz - (table_z_m + h)) < 0.05:
                top_z_model = 0.5 * top_z_model + 0.5 * tz
        except (TypeError, ValueError):
            pass

    n_slab = int(slab_pts.shape[0])
    empty["num_top_slab_points"] = n_slab
    empty["top_z_model_m"] = float(top_z_model)
    if n_slab < MODEL_MIN_TOP_SLAB_POINTS:
        empty["message"] = "too_few_top_slab_points"
        return empty

    slab_ratio = float(n_slab) / float(max(n_seg, 1))
    rect = fit_known_top_rectangle_pose(
        slab_pts,
        label,
        db_dims,
        table_z_m,
        top_z_model,
        yaw_initial_rad=yaw_hint_rad,
        top_face_point_ratio=slab_ratio,
    )
    if not bool(rect.get("success", False)):
        empty["message"] = "rectangle_search_on_slab_failed"
        empty.update(
            {
                "fit_error": float(rect.get("fit_error", float("inf"))),
                "inlier_ratio": rect.get("inlier_ratio"),
                "length_error_m": rect.get("length_error_m"),
                "width_error_m": rect.get("width_error_m"),
            }
        )
        return empty

    fit_detail = dict(rect)
    fit_detail["num_segmented_points"] = n_seg
    fit_detail["num_top_slab_points"] = n_slab
    fit_detail["top_z_model_m"] = float(top_z_model)
    ok, reject = _model_acceptance(fit_detail, n_segmented=n_seg, n_slab=n_slab, top_face_point_ratio=slab_ratio)
    if not ok:
        empty["message"] = reject
        empty.update(
            {
                "fit_error": fit_detail.get("fit_error"),
                "yaw_confidence": fit_detail.get("yaw_confidence"),
                "inlier_ratio": fit_detail.get("inlier_ratio"),
            }
        )
        return empty

    yaw = float(fit_detail["yaw_rad"])
    axis_long = np.array(fit_detail["long_axis_xy"], dtype=np.float64)
    axis_short = np.array(fit_detail["short_axis_xy"], dtype=np.float64)
    center_xy = np.array(fit_detail["center_xy"], dtype=np.float64)
    top_corners = _build_top_corners_base(center_xy, axis_long, L, W, top_z_model)
    bottom_corners = [
        [float(p[0]), float(p[1]), float(table_z_m)] for p in top_corners
    ]
    closing_yaw = _wrap_pi(yaw + math.pi / 2.0)
    cx, cy = float(center_xy[0]), float(center_xy[1])
    center_xyz = [cx, cy, float(top_z_model)]

    out = {
        "success": True,
        "model_top_face_success": True,
        "top_face_source": "known_model",
        "model_fit_error": float(fit_detail.get("fit_error", 0.0)),
        "model_box_yaw_rad": float(yaw),
        "model_closing_yaw_rad": float(closing_yaw),
        "model_box_center_base": center_xyz,
        "model_top_face_corners_base": top_corners,
        "model_bottom_corners_base": bottom_corners,
        "model_major_axis_xy": [float(axis_long[0]), float(axis_long[1])],
        "model_minor_axis_xy": [float(axis_short[0]), float(axis_short[1])],
        "yaw_confidence": float(fit_detail.get("yaw_confidence", 0.0)),
        "yaw_fit_method": "model_cuboid_top_slab_search",
        "pose_fit_success": True,
        "pose_fit_error": float(fit_detail.get("fit_error", 0.0)),
        "inlier_ratio": float(fit_detail.get("inlier_ratio", 0.0)),
        "edge_support_score": float(fit_detail.get("edge_support_score", 0.0)),
        "length_error_m": float(fit_detail.get("length_error_m", 0.0)),
        "width_error_m": float(fit_detail.get("width_error_m", 0.0)),
        "outside_error_m": float(fit_detail.get("outside_error_m", 0.0)),
        "projected_extent_length_m": float(fit_detail.get("projected_extent_length_m", 0.0)),
        "projected_extent_width_m": float(fit_detail.get("projected_extent_width_m", 0.0)),
        "db_length_m": L,
        "db_width_m": W,
        "height_m": h,
        "long_dim_m": L,
        "short_dim_m": W,
        "selected_yaw_deg": float(fit_detail.get("selected_yaw_deg", math.degrees(yaw))),
        "center_method": "model_cuboid_top_slab",
        "num_segmented_points": n_seg,
        "num_top_slab_points": n_slab,
        "top_z_model_m": float(top_z_model),
        "message": "ok_model",
        # Compatibilidad con campos legacy usados por hybrid/rectangle
        "top_corners_base": top_corners,
        "bottom_corners_base": bottom_corners,
        "center_xyz": center_xyz,
        "center_xy": [cx, cy],
        "yaw_rad": float(yaw),
        "long_axis_xy": [float(axis_long[0]), float(axis_long[1])],
        "short_axis_xy": [float(axis_short[0]), float(axis_short[1])],
    }
    return out


def merge_hybrid_as_model_source(
    hybrid: Dict[str, Any],
) -> Dict[str, Any]:
    """Promueve un fit híbrido exitoso a campos model_* con top_face_source=hybrid_known_model."""
    if not bool(hybrid.get("success", False)):
        return {"success": False, "message": "hybrid_not_success"}
    yaw = float(hybrid.get("yaw_rad", 0.0))
    out = dict(hybrid)
    out["top_face_source"] = "hybrid_known_model"
    out["model_top_face_success"] = True
    out["model_fit_error"] = float(hybrid.get("fit_error", 0.0))
    out["model_box_yaw_rad"] = yaw
    out["model_closing_yaw_rad"] = _wrap_pi(yaw + math.pi / 2.0)
    out["model_box_center_base"] = list(hybrid.get("center_xyz", [0, 0, 0]))
    out["model_top_face_corners_base"] = list(hybrid.get("top_corners_base", []))
    out["model_bottom_corners_base"] = list(hybrid.get("bottom_corners_base", []))
    out["model_major_axis_xy"] = list(hybrid.get("long_axis_xy", [1.0, 0.0]))
    out["model_minor_axis_xy"] = list(hybrid.get("short_axis_xy", [0.0, 1.0]))
    return out


def observed_vs_model_corner_error_m(
    observed_top_points: np.ndarray,
    model_corners_base: List[List[float]],
) -> Tuple[float, float]:
    """Error medio XY entre centro observado y esquinas del modelo; delta yaw en grados."""
    if (
        observed_top_points is None
        or observed_top_points.size == 0
        or not model_corners_base
        or len(model_corners_base) < 3
    ):
        return float("nan"), float("nan")
    obs = np.asarray(observed_top_points, dtype=np.float64).reshape(-1, 3)
    obs_c = np.median(obs[:, :2], axis=0)
    mc = np.asarray([[float(c[0]), float(c[1])] for c in model_corners_base[:4]])
    model_c = np.mean(mc, axis=0)
    corner_err = float(np.mean(np.linalg.norm(mc - model_c, axis=1)))
    center_err = float(np.linalg.norm(obs_c - model_c))
    # Yaw desde primer arista del modelo vs PCA observado
    e0 = mc[1] - mc[0]
    yaw_model = math.atan2(float(e0[1]), float(e0[0]))
    cov = np.cov(obs[:, :2].T) if obs.shape[0] >= 3 else np.eye(2)
    ev, evec = np.linalg.eigh(cov)
    yaw_obs = math.atan2(float(evec[1, 1]), float(evec[0, 1]))
    yaw_delta_deg = abs(math.degrees(_wrap_pi(yaw_model - yaw_obs)))
    if yaw_delta_deg > 90.0:
        yaw_delta_deg = 180.0 - yaw_delta_deg
    return max(center_err, corner_err), float(yaw_delta_deg)


def log_model_top_face_summary(logger: Any, label: str, model: Dict[str, Any]) -> None:
    try:
        logger.info(
            "[MODEL_TOP_FACE] label=%s success=%s source=%s fit_err=%.5f yaw_conf=%.3f "
            "Lerr=%.4f Werr=%.4f inlier=%.3f slab_pts=%d seg_pts=%d"
            % (
                label,
                str(bool(model.get("model_top_face_success"))).lower(),
                model.get("top_face_source"),
                float(model.get("model_fit_error") or 0.0),
                float(model.get("yaw_confidence") or 0.0),
                float(model.get("length_error_m") or 0.0),
                float(model.get("width_error_m") or 0.0),
                float(model.get("inlier_ratio") or 0.0),
                int(model.get("num_top_slab_points") or 0),
                int(model.get("num_segmented_points") or 0),
            )
        )
    except Exception:
        pass
