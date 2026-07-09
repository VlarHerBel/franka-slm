"""Extracción de puntos de la cara superior del objeto (evita laterales para yaw/caja)."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

_TABLE_UP = np.array([0.0, 0.0, 1.0], dtype=np.float64)


def extract_top_face_points(
    points_base: np.ndarray,
    top_z_m: Optional[float] = None,
    table_normal: Tuple[float, float, float] = (0.0, 0.0, 1.0),
    top_band_tolerance_m: float = 0.010,
    normal_z_threshold: float = 0.85,
    plane_distance_threshold_m: float = 0.006,
    min_top_points: int = 30,
) -> Dict[str, Any]:
    """Filtra la nube para quedarse con la cara superior visible.

    Devuelve dict con success, top_points_base, side_points_base, plano estimado y métricas.
    """
    empty_top = np.empty((0, 3), dtype=np.float64)
    empty_side = np.empty((0, 3), dtype=np.float64)
    out: Dict[str, Any] = {
        "success": False,
        "top_points_base": empty_top,
        "side_points_base": empty_side,
        "top_plane_normal": None,
        "top_plane_d": None,
        "top_z_estimated": float("nan"),
        "top_point_ratio": 0.0,
        "num_top_points": 0,
        "num_side_points": 0,
        "method": "none",
        "message": "",
    }

    if points_base is None or points_base.size == 0:
        out["message"] = "empty_pointcloud"
        return out

    pts = np.asarray(points_base, dtype=np.float64).reshape(-1, 3)
    if pts.shape[0] < 5:
        out["message"] = "too_few_points"
        return out

    n_total = int(pts.shape[0])
    p98 = float(np.percentile(pts[:, 2], 98.0))
    p99 = float(np.percentile(pts[:, 2], 99.0))
    top_z_percentile = 0.5 * (p98 + p99)

    top_z_estimated = top_z_percentile
    if top_z_m is not None:
        try:
            tz = float(top_z_m)
        except (TypeError, ValueError):
            tz = None
        if tz is not None and abs(tz - top_z_percentile) <= 0.025:
            top_z_estimated = 0.6 * top_z_percentile + 0.4 * tz
        elif tz is not None and abs(tz - top_z_percentile) <= 0.06:
            top_z_estimated = 0.5 * (top_z_percentile + tz)

    out["top_z_estimated"] = float(top_z_estimated)

    tu = np.array(table_normal, dtype=np.float64)
    tu_norm = np.linalg.norm(tu)
    if tu_norm < 1e-9:
        tu = _TABLE_UP.copy()
    else:
        tu = tu / tu_norm

    band_mask = pts[:, 2] > (top_z_estimated - float(top_band_tolerance_m))
    band_pts = pts[band_mask]
    if band_pts.shape[0] < max(5, min_top_points // 2):
        out["message"] = "insufficient_band_points"
        out["method"] = "failed"
        return out

    o3d = None
    try:
        import open3d as o3d  # type: ignore
    except ImportError:
        o3d = None

    normal_mask = np.ones(band_pts.shape[0], dtype=bool)
    method_normals = "skipped_no_open3d"

    if o3d is not None and band_pts.shape[0] >= min_top_points:
        try:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(band_pts)
            pcd.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(
                    radius=max(0.02, 2.0 * top_band_tolerance_m),
                    max_nn=30,
                )
            )
            normals = np.asarray(pcd.normals, dtype=np.float64)
            # Orientar hacia +Z de la mesa
            flip = (normals @ tu) < 0.0
            normals[flip] *= -1.0
            normal_dot = normals @ tu
            normal_mask = normal_dot > float(normal_z_threshold)
            method_normals = "open3d_normals"
        except Exception as exc:
            method_normals = f"open3d_failed:{exc}"
            normal_mask = np.ones(band_pts.shape[0], dtype=bool)

    candidates = band_pts[normal_mask]
    if candidates.shape[0] < max(10, min_top_points // 3):
        candidates = band_pts
        method_normals = "z_band_only"
        normal_mask = np.ones(band_pts.shape[0], dtype=bool)

    top_pts = candidates
    top_plane_normal: Optional[List[float]] = None
    top_plane_d: Optional[float] = None
    method = method_normals

    if o3d is not None and top_pts.shape[0] >= min_top_points:
        try:
            pcd2 = o3d.geometry.PointCloud()
            pcd2.points = o3d.utility.Vector3dVector(top_pts)
            plane_model, inliers = pcd2.segment_plane(
                distance_threshold=float(plane_distance_threshold_m),
                ransac_n=3,
                num_iterations=200,
            )
            a, b, c, d = [float(x) for x in plane_model]
            nvec = np.array([a, b, c], dtype=np.float64)
            nn = np.linalg.norm(nvec)
            if nn > 1e-9:
                nvec = nvec / nn
                if float(np.dot(nvec, tu)) < 0.0:
                    nvec = -nvec
                    d = -d
                if abs(float(np.dot(nvec, tu))) > 0.90 and len(inliers) >= int(min_top_points):
                    inlier_idx = np.array(inliers, dtype=np.int64)
                    inlier_pts = np.asarray(pcd2.points, dtype=np.float64)[inlier_idx]
                    dists = np.abs(inlier_pts @ nvec + d)
                    keep = dists < float(plane_distance_threshold_m) * 1.5
                    inlier_pts = inlier_pts[keep]
                    if inlier_pts.shape[0] >= int(min_top_points):
                        top_pts = inlier_pts
                        top_plane_normal = [float(nvec[0]), float(nvec[1]), float(nvec[2])]
                        top_plane_d = float(d)
                        method = "ransac_top_plane"
        except Exception:
            pass

    if top_pts.shape[0] < int(min_top_points):
        top_pts = band_pts
        method = "fallback_z_band"
        top_plane_normal = None
        top_plane_d = None

    num_top = int(top_pts.shape[0])
    ratio = float(num_top) / float(max(1, n_total))
    if method.startswith("ransac") and top_plane_normal is not None and top_plane_d is not None:
        nv = np.array(top_plane_normal, dtype=np.float64)
        dd = float(top_plane_d)
        dist_all = np.abs(pts @ nv + dd)
        inlier_all = dist_all < float(plane_distance_threshold_m) * 2.0
        side_pts = pts[~inlier_all]
    else:
        side_mask = pts[:, 2] < (top_z_estimated - 0.5 * float(top_band_tolerance_m))
        side_pts = pts[side_mask]
        if side_pts.shape[0] == 0:
            side_pts = empty_side
    num_side = int(side_pts.shape[0])

    out.update(
        {
            "success": num_top >= int(min_top_points),
            "top_points_base": top_pts.astype(np.float64),
            "side_points_base": np.asarray(side_pts, dtype=np.float64).reshape(-1, 3),
            "top_plane_normal": top_plane_normal,
            "top_plane_d": top_plane_d,
            "top_point_ratio": ratio,
            "num_top_points": num_top,
            "num_side_points": int(side_pts.shape[0]),
            "method": method,
            "message": "ok" if num_top >= int(min_top_points) else "below_min_top_points",
        }
    )
    return out


def log_top_face_summary(logger: Any, label: str, result: Dict[str, Any]) -> None:
    msg = (
        "[TOP_FACE]\n"
        "label=%s\n"
        "success=%s\n"
        "method=%s\n"
        "num_top_points=%s\n"
        "num_side_points=%s\n"
        "top_ratio=%.3f\n"
        "top_z_estimated=%.4f\n"
        "message=%s"
        % (
            label,
            str(bool(result.get("success"))).lower(),
            result.get("method"),
            result.get("num_top_points"),
            result.get("num_side_points"),
            float(result.get("top_point_ratio") or 0.0),
            float(result.get("top_z_estimated") or float("nan")),
            result.get("message"),
        )
    )
    try:
        logger.info(msg)
    except Exception:
        pass
