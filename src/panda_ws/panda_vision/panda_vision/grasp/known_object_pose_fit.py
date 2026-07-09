"""Ajuste de pose planar de cajas conocidas usando solo la cara superior (DB YCB).

Búsqueda global de yaw en [0, pi) + refinamiento local; métricas que penalizan
desalineación L/W y baja evidencia de bordes (sin usar pose de spawn).
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def _wrap_pi(a: float) -> float:
    return float((a + math.pi) % (2.0 * math.pi) - math.pi)


def _score_single_theta(
    theta: float,
    xy: np.ndarray,
    L: float,
    W: float,
    inlier_tol_m: float,
    edge_tol_m: float,
    hinge_margin_m: float,
    pct_lo: float,
    pct_hi: float,
) -> Tuple[float, Dict[str, Any]]:
    """Evalúa un yaw theta (radianes); long_axis = [cos,sin], short = [-sin,cos]."""
    c = math.cos(theta)
    s = math.sin(theta)
    axis_long = np.array([c, s], dtype=np.float64)
    axis_short = np.array([-s, c], dtype=np.float64)

    seed_center = np.median(xy, axis=0)
    rel0 = xy - seed_center
    u0 = rel0 @ axis_long
    v0 = rel0 @ axis_short
    u_lo, u_hi = np.percentile(u0, [pct_lo, pct_hi])
    v_lo, v_hi = np.percentile(v0, [pct_lo, pct_hi])
    center_xy = (
        seed_center
        + 0.5 * (u_lo + u_hi) * axis_long
        + 0.5 * (v_lo + v_hi) * axis_short
    )
    center_shift = float(np.linalg.norm(center_xy - seed_center))

    rel = xy - center_xy
    u = rel @ axis_long
    v = rel @ axis_short

    half_l = 0.5 * L
    half_w = 0.5 * W
    m = hinge_margin_m

    observed_L = float(np.percentile(u, pct_hi) - np.percentile(u, pct_lo))
    observed_W = float(np.percentile(v, pct_hi) - np.percentile(v, pct_lo))

    length_error_m = abs(observed_L - L)
    width_error_m = abs(observed_W - W)

    hyp_match = abs(observed_L - L) + abs(observed_W - W)
    hyp_swap = abs(observed_L - W) + abs(observed_W - L)
    # Penalizar solo si los extentes encajan mejor intercambiando L/W (eje largo mal asignado).
    axis_swap_penalty = max(0.0, hyp_match - hyp_swap) * 6.0

    out_u = np.maximum(np.abs(u) - (half_l + m), 0.0)
    out_v = np.maximum(np.abs(v) - (half_w + m), 0.0)
    outside_dist = np.sqrt(out_u * out_u + out_v * out_v)
    outside_mean = float(np.mean(outside_dist))
    outside_p90 = float(np.percentile(outside_dist, 90.0))
    outside_error_m = outside_mean + 2.0 * outside_p90

    inside = (np.abs(u) <= half_l + inlier_tol_m) & (np.abs(v) <= half_w + inlier_tol_m)
    inlier_ratio = float(np.mean(inside.astype(np.float64)))

    near_long = np.abs(np.abs(u) - half_l) < edge_tol_m
    near_short = np.abs(np.abs(v) - half_w) < edge_tol_m
    edge_mask = near_long | near_short
    edge_support_score = float(np.mean(edge_mask.astype(np.float64)))

    extent_pen = 0.0
    if observed_L > L * 1.18:
        extent_pen += 0.08 * (observed_L - L * 1.18)
    if observed_W > W * 1.28:
        extent_pen += 0.08 * (observed_W - W * 1.28)

    coverage_pen = (2.0 - min(edge_support_score * 5.0, 1.0) - min(inlier_ratio * 1.2, 1.0)) * 0.008

    score = (
        outside_error_m
        + 0.55 * length_error_m
        + 0.85 * width_error_m
        + 0.05 * max(0.0, 1.0 - inlier_ratio)
        + 0.04 * max(0.0, 1.0 - edge_support_score)
        + axis_swap_penalty
        + extent_pen
        + coverage_pen
    )

    detail: Dict[str, Any] = {
        "theta_rad": float(theta),
        "center_xy": center_xy.copy(),
        "axis_long": axis_long.copy(),
        "axis_short": axis_short.copy(),
        "projected_extent_length_m": observed_L,
        "projected_extent_width_m": observed_W,
        "length_error_m": length_error_m,
        "width_error_m": width_error_m,
        "outside_error_m": outside_error_m,
        "inlier_ratio": inlier_ratio,
        "edge_support_score": edge_support_score,
        "axis_swap_penalty": float(axis_swap_penalty),
        "center_shift_from_median_m": center_shift,
        "score": float(score),
    }
    return float(score), detail


def fit_known_top_rectangle_pose(
    top_points_base: np.ndarray,
    label: str,
    db_dims: Tuple[float, float, float],
    table_z_m: float,
    top_z_m: float,
    yaw_initial_rad: Optional[float] = None,
    top_face_point_ratio: Optional[float] = None,
) -> Dict[str, Any]:
    """Ajuste rectangular con búsqueda global de yaw en [0, pi) y refinamiento fino."""
    out: Dict[str, Any] = {
        "success": False,
        "center_xy": [0.0, 0.0],
        "center_xyz": [0.0, 0.0, 0.0],
        "yaw_rad": 0.0,
        "yaw_confidence": 0.0,
        "yaw_source": "known_rectangle_fit",
        "yaw_fit_method": "global_rectangle_search",
        "top_corners_base": [],
        "bottom_corners_base": [],
        "long_axis_xy": [1.0, 0.0],
        "short_axis_xy": [0.0, 1.0],
        "long_dim_m": 0.0,
        "short_dim_m": 0.0,
        "height_m": 0.0,
        "fit_error": float("inf"),
        "message": "",
        "top_z_measured": float(top_z_m),
        "center_method": "projected_extents",
        "center_shift_from_median_m": 0.0,
        "projected_extent_length_m": 0.0,
        "projected_extent_width_m": 0.0,
        "length_error_m": 0.0,
        "width_error_m": 0.0,
        "outside_error_m": 0.0,
        "inlier_ratio": 0.0,
        "edge_support_score": 0.0,
        "yaw_margin_score": 0.0,
        "best_score": float("inf"),
        "second_best_score": float("inf"),
        "num_yaw_candidates": 0,
        "selected_yaw_deg": 0.0,
    }

    pts = np.asarray(top_points_base, dtype=np.float64).reshape(-1, 3)
    n_pts = int(pts.shape[0])
    if n_pts < 12:
        out["message"] = "too_few_top_points"
        return out

    d0, d1, h = float(db_dims[0]), float(db_dims[1]), float(db_dims[2])
    L = max(d0, d1)
    W = min(d0, d1)
    out["long_dim_m"] = L
    out["short_dim_m"] = W
    out["height_m"] = h

    xy = pts[:, :2]
    z_med = float(np.median(pts[:, 2]))

    inlier_tol_m = 0.005
    edge_tol_m = 0.006
    hinge_margin_m = 0.004
    pct_lo, pct_hi = 2.0, 98.0

    coarse_deg = np.arange(0.0, 180.0, 1.0, dtype=np.float64)
    coarse_rad = np.radians(coarse_deg)
    coarse_results: List[Tuple[float, float, Dict[str, Any]]] = []
    for th in coarse_rad:
        sc, det = _score_single_theta(
            float(th),
            xy,
            L,
            W,
            inlier_tol_m,
            edge_tol_m,
            hinge_margin_m,
            pct_lo,
            pct_hi,
        )
        coarse_results.append((float(th), sc, det))

    coarse_results.sort(key=lambda t: t[1])
    best_theta, best_score, best_detail = coarse_results[0]

    refine_half = math.radians(3.0)
    refine_step = math.radians(0.1)
    refined: List[Tuple[float, float, Dict[str, Any]]] = []
    th_lo = max(0.0, best_theta - refine_half)
    th_hi = min(math.pi, best_theta + refine_half)
    th = th_lo
    while th <= th_hi + 1e-9:
        sc, det = _score_single_theta(
            float(th),
            xy,
            L,
            W,
            inlier_tol_m,
            edge_tol_m,
            hinge_margin_m,
            pct_lo,
            pct_hi,
        )
        refined.append((float(th), sc, det))
        th += refine_step

    all_eval = coarse_results + refined
    all_eval.sort(key=lambda t: t[1])
    best_theta, best_score, best_detail = all_eval[0]

    second_score = float("inf")
    for th, sc, _det in all_eval[1:]:
        if abs(_wrap_pi(th - best_theta)) > math.radians(2.0):
            second_score = float(sc)
            break
    if not math.isfinite(second_score):
        for _th, sc, _det in all_eval[1:6]:
            second_score = min(second_score, float(sc))
    yaw_margin_score = max(0.0, second_score - best_score)

    num_candidates = len(coarse_deg) + len(refined)

    theta_out = float(best_theta)
    if yaw_initial_rad is not None:
        try:
            y0 = float(yaw_initial_rad)
            alt = _wrap_pi(theta_out + math.pi)
            if abs(_wrap_pi(alt - y0)) < abs(_wrap_pi(theta_out - y0)) - 1e-6:
                theta_out = alt
        except (TypeError, ValueError):
            pass

    axis_long = best_detail["axis_long"]
    axis_short = best_detail["axis_short"]
    center_xy = best_detail["center_xy"]
    cx, cy = float(center_xy[0]), float(center_xy[1])

    out["center_xy"] = [cx, cy]
    out["center_xyz"] = [cx, cy, z_med]
    out["yaw_rad"] = float(theta_out)
    out["long_axis_xy"] = [float(axis_long[0]), float(axis_long[1])]
    out["short_axis_xy"] = [float(axis_short[0]), float(axis_short[1])]
    out["fit_error"] = float(best_score)
    out["best_score"] = float(best_score)
    out["second_best_score"] = float(second_score)
    out["yaw_margin_score"] = float(yaw_margin_score)
    out["num_yaw_candidates"] = int(num_candidates)
    out["selected_yaw_deg"] = float(math.degrees(theta_out))

    out["center_shift_from_median_m"] = float(
        best_detail["center_shift_from_median_m"]
    )
    out["projected_extent_length_m"] = float(
        best_detail["projected_extent_length_m"]
    )
    out["projected_extent_width_m"] = float(
        best_detail["projected_extent_width_m"]
    )
    out["length_error_m"] = float(best_detail["length_error_m"])
    out["width_error_m"] = float(best_detail["width_error_m"])
    out["outside_error_m"] = float(best_detail["outside_error_m"])
    out["inlier_ratio"] = float(best_detail["inlier_ratio"])
    out["edge_support_score"] = float(best_detail["edge_support_score"])

    top_z_box = float(table_z_m) + h
    hl = L * 0.5
    hw = W * 0.5
    bl = axis_long * hl
    bs = axis_short * hw
    corners_2d = [
        np.array([cx, cy]) + bl + bs,
        np.array([cx, cy]) + bl - bs,
        np.array([cx, cy]) - bl - bs,
        np.array([cx, cy]) - bl + bs,
    ]
    out["top_corners_base"] = [
        [float(p[0]), float(p[1]), top_z_box] for p in corners_2d
    ]
    tb = float(table_z_m)
    out["bottom_corners_base"] = [
        [float(p[0]), float(p[1]), tb] for p in corners_2d
    ]

    tfr = float(top_face_point_ratio) if top_face_point_ratio is not None else 1.0
    tfr = float(np.clip(tfr, 0.0, 1.0))

    margin_norm = float(np.clip(yaw_margin_score / 0.012, 0.0, 1.0))
    score_norm = float(np.clip(1.0 - best_score / 0.12, 0.0, 1.0))
    inlier_c = float(np.clip((out["inlier_ratio"] - 0.55) / 0.45, 0.0, 1.0))
    edge_c = float(np.clip(out["edge_support_score"] / 0.35, 0.0, 1.0))
    len_c = float(np.clip(1.0 - out["length_error_m"] / 0.06, 0.0, 1.0))
    wid_c = float(np.clip(1.0 - out["width_error_m"] / 0.04, 0.0, 1.0))
    out_c = float(np.clip(1.0 - out["outside_error_m"] / 0.025, 0.0, 1.0))

    yaw_conf = (
        0.22 * margin_norm
        + 0.18 * score_norm
        + 0.22 * inlier_c
        + 0.12 * edge_c
        + 0.12 * len_c
        + 0.08 * wid_c
        + 0.06 * out_c
    )
    yaw_conf *= 0.35 + 0.65 * tfr
    out["yaw_confidence"] = float(np.clip(yaw_conf, 0.0, 1.0))

    min_pts = 200
    min_inlier = 0.80
    max_out = 0.012
    max_len_err = 0.028
    max_wid_err = 0.018
    min_margin = 0.0004
    min_edge = 0.12
    max_score = 0.048

    geom_ok = (
        n_pts >= min_pts
        and out["inlier_ratio"] >= min_inlier
        and out["outside_error_m"] <= max_out
        and out["length_error_m"] <= max_len_err
        and out["width_error_m"] <= max_wid_err
        and yaw_margin_score >= min_margin
        and out["edge_support_score"] >= min_edge
        and best_score <= max_score
        and tfr >= 0.55
    )
    len_shortfall = float(L) - float(out["projected_extent_length_m"])
    partial_top_face = (
        out["width_error_m"] <= 0.020
        and len_shortfall >= 0.020
        and len_shortfall <= 0.12
        and out["inlier_ratio"] >= 0.75
        and out["outside_error_m"] <= 0.015
    )
    out["partial_top_face_detected"] = bool(partial_top_face)
    out["db_length_m"] = float(L)
    out["db_width_m"] = float(W)
    out["observed_extent_length_m"] = float(out["projected_extent_length_m"])
    out["observed_extent_width_m"] = float(out["projected_extent_width_m"])

    out["success"] = bool(geom_ok)
    out["message"] = "ok" if out["success"] else "geometry_gates_failed"

    return out


def _normalize_axis_xy(axis: Tuple[float, float]) -> np.ndarray:
    v = np.array([float(axis[0]), float(axis[1])], dtype=np.float64)
    n = float(np.linalg.norm(v))
    if n < 1e-9:
        return np.array([1.0, 0.0], dtype=np.float64)
    return v / n


def _score_at_fixed_yaw_center(
    theta: float,
    center_xy: np.ndarray,
    xy: np.ndarray,
    L: float,
    W: float,
    inlier_tol_m: float,
    edge_tol_m: float,
    hinge_margin_m: float,
    pct_lo: float,
    pct_hi: float,
) -> Tuple[float, Dict[str, Any]]:
    """Métricas con yaw fijo y centro dado."""
    c = math.cos(theta)
    s = math.sin(theta)
    axis_long = np.array([c, s], dtype=np.float64)
    axis_short = np.array([-s, c], dtype=np.float64)
    rel = xy - center_xy
    u = rel @ axis_long
    v = rel @ axis_short
    half_l = 0.5 * L
    half_w = 0.5 * W
    m = hinge_margin_m
    observed_L = float(np.percentile(u, pct_hi) - np.percentile(u, pct_lo))
    observed_W = float(np.percentile(v, pct_hi) - np.percentile(v, pct_lo))
    length_error_m = abs(observed_L - L)
    width_error_m = abs(observed_W - W)
    out_u = np.maximum(np.abs(u) - (half_l + m), 0.0)
    out_v = np.maximum(np.abs(v) - (half_w + m), 0.0)
    outside_dist = np.sqrt(out_u * out_u + out_v * out_v)
    outside_error_m = float(np.mean(outside_dist) + 2.0 * np.percentile(outside_dist, 90.0))
    inside = (np.abs(u) <= half_l + inlier_tol_m) & (np.abs(v) <= half_w + inlier_tol_m)
    inlier_ratio = float(np.mean(inside.astype(np.float64)))
    near_long = np.abs(np.abs(u) - half_l) < edge_tol_m
    near_short = np.abs(np.abs(v) - half_w) < edge_tol_m
    edge_support_score = float(np.mean((near_long | near_short).astype(np.float64)))
    score = outside_error_m + 0.02 * max(0.0, 1.0 - inlier_ratio)
    detail = {
        "center_xy": center_xy.copy(),
        "axis_long": axis_long,
        "axis_short": axis_short,
        "projected_extent_length_m": observed_L,
        "projected_extent_width_m": observed_W,
        "length_error_m": length_error_m,
        "width_error_m": width_error_m,
        "outside_error_m": outside_error_m,
        "inlier_ratio": inlier_ratio,
        "edge_support_score": edge_support_score,
        "score": float(score),
    }
    return float(score), detail


def try_hybrid_top_face_known_dims_fit(
    top_points_base: np.ndarray,
    db_dims: Tuple[float, float, float],
    pca_yaw_rad: float,
    pca_long_axis_xy: Tuple[float, float],
    pca_short_axis_xy: Tuple[float, float],
    table_z_m: float,
    top_z_m: float,
    rectangle_fit: Optional[Dict[str, Any]] = None,
    min_top_points: int = 150,
) -> Dict[str, Any]:
    """Yaw de PCA/top_face + centro refinado con dimensiones DB (cara superior parcial)."""
    out: Dict[str, Any] = {
        "success": False,
        "hybrid_fit_success": False,
        "yaw_rad": float(pca_yaw_rad),
        "yaw_fit_method": "top_face_pca_known_dims",
        "yaw_source": "hybrid_top_face_known_dims",
        "center_xyz": [0.0, 0.0, 0.0],
        "long_axis_xy": [float(pca_long_axis_xy[0]), float(pca_long_axis_xy[1])],
        "short_axis_xy": [float(pca_short_axis_xy[0]), float(pca_short_axis_xy[1])],
        "partial_top_face_detected": False,
        "center_fit_method": "",
        "center_offset_long_m": 0.0,
        "center_offset_short_m": 0.0,
        "message": "",
    }

    pts = np.asarray(top_points_base, dtype=np.float64).reshape(-1, 3)
    if pts.shape[0] < min_top_points:
        out["message"] = "too_few_top_points_for_hybrid"
        return out

    d0, d1, h = float(db_dims[0]), float(db_dims[1]), float(db_dims[2])
    L = max(d0, d1)
    W = min(d0, d1)
    out["db_length_m"] = L
    out["db_width_m"] = W
    out["long_dim_m"] = L
    out["short_dim_m"] = W
    out["height_m"] = h

    xy = pts[:, :2]
    z_med = float(np.median(pts[:, 2]))
    theta = float(pca_yaw_rad)
    axis_long = _normalize_axis_xy(pca_long_axis_xy)
    axis_short = _normalize_axis_xy(pca_short_axis_xy)
    dot_ls = float(np.dot(axis_long, axis_short))
    if abs(dot_ls) > 0.2:
        axis_short = np.array([-axis_long[1], axis_long[0]], dtype=np.float64)

    inlier_tol_m = 0.005
    edge_tol_m = 0.006
    hinge_margin_m = 0.004
    pct_lo, pct_hi = 2.0, 98.0

    _sc0, det0 = _score_single_theta(
        theta, xy, L, W, inlier_tol_m, edge_tol_m, hinge_margin_m, pct_lo, pct_hi
    )
    obs_L = float(det0["projected_extent_length_m"])
    obs_W = float(det0["projected_extent_width_m"])
    out["observed_extent_length_m"] = obs_L
    out["observed_extent_width_m"] = obs_W
    out["length_error_m"] = float(det0["length_error_m"])
    out["width_error_m"] = float(det0["width_error_m"])
    out["outside_error_m"] = float(det0["outside_error_m"])
    out["inlier_ratio"] = float(det0["inlier_ratio"])
    out["edge_support_score"] = float(det0["edge_support_score"])
    out["projected_extent_length_m"] = obs_L
    out["projected_extent_width_m"] = obs_W

    len_shortfall = L - obs_L
    width_ok = out["width_error_m"] <= 0.020
    partial = (
        width_ok
        and len_shortfall >= 0.020
        and len_shortfall <= 0.12
        and out["inlier_ratio"] >= 0.80
        and out["outside_error_m"] <= 0.015
    )
    if rectangle_fit and bool(rectangle_fit.get("partial_top_face_detected")):
        partial = True
    out["partial_top_face_detected"] = bool(partial)

    if not partial and not width_ok:
        out["message"] = "hybrid_pattern_not_matched"
        return out
    if out["inlier_ratio"] < 0.75 or out["outside_error_m"] > 0.020:
        out["message"] = "hybrid_quality_gates_failed"
        return out

    observed_center = np.asarray(det0["center_xy"], dtype=np.float64)
    center_xy = observed_center.copy()
    center_method = "observed_top_face_extents"
    du_best = 0.0
    dv_best = 0.0

    if partial and len_shortfall > 0.015:
        du_half = float(len_shortfall) * 0.5 + 0.010
        dv_half = max(0.003, float(abs(W - obs_W)) * 0.5 + 0.004)
        best_c_score = float("inf")
        for du in np.linspace(-du_half, du_half, 9):
            for dv in np.linspace(-dv_half, dv_half, 5):
                cand = observed_center + float(du) * axis_long + float(dv) * axis_short
                sc, _det = _score_at_fixed_yaw_center(
                    theta,
                    cand,
                    xy,
                    L,
                    W,
                    inlier_tol_m,
                    edge_tol_m,
                    hinge_margin_m,
                    pct_lo,
                    pct_hi,
                )
                shift_pen = 0.15 * float(np.linalg.norm(cand - observed_center))
                total = sc + shift_pen
                if total < best_c_score:
                    best_c_score = total
                    center_xy = cand
                    du_best = float(du)
                    dv_best = float(dv)
        center_method = "partial_db_extent_search"

    cx, cy = float(center_xy[0]), float(center_xy[1])
    out["center_xy"] = [cx, cy]
    out["center_xyz"] = [cx, cy, z_med]
    out["center_fit_method"] = center_method
    out["center_offset_long_m"] = float(du_best)
    out["center_offset_short_m"] = float(dv_best)
    out["center_shift_from_median_m"] = float(
        np.linalg.norm(center_xy - np.median(xy, axis=0))
    )
    out["yaw_rad"] = theta
    out["selected_yaw_deg"] = float(math.degrees(theta))
    out["long_axis_xy"] = [float(axis_long[0]), float(axis_long[1])]
    out["short_axis_xy"] = [float(axis_short[0]), float(axis_short[1])]

    hl = L * 0.5
    hw = W * 0.5
    bl = axis_long * hl
    bs = axis_short * hw
    corners_2d = [
        np.array([cx, cy]) + bl + bs,
        np.array([cx, cy]) + bl - bs,
        np.array([cx, cy]) - bl - bs,
        np.array([cx, cy]) - bl + bs,
    ]
    top_z_box = float(table_z_m) + h
    out["top_corners_base"] = [
        [float(p[0]), float(p[1]), top_z_box] for p in corners_2d
    ]
    out["bottom_corners_base"] = [
        [float(p[0]), float(p[1]), float(table_z_m)] for p in corners_2d
    ]

    yaw_conf = float(
        np.clip(
            0.45
            + 0.35 * out["inlier_ratio"]
            + 0.15 * (1.0 - min(out["outside_error_m"] / 0.02, 1.0))
            - 0.10 * min(out["length_error_m"] / 0.08, 1.0),
            0.35,
            0.92,
        )
    )
    out["yaw_confidence"] = yaw_conf
    out["fit_error"] = float(out["outside_error_m"] + 0.25 * out["length_error_m"])
    out["pose_fit_success"] = True
    out["hybrid_fit_success"] = True
    out["success"] = True
    out["message"] = "ok_hybrid"
    return out


def log_pose_fit_summary(logger: Any, label: str, fit: Dict[str, Any]) -> None:
    msg = (
        "[POSE_FIT]\n"
        "label=%s\n"
        "success=%s\n"
        "yaw_fit_method=%s\n"
        "yaw_source=%s\n"
        "yaw_confidence=%.3f\n"
        "yaw_rad=%.4f selected_yaw_deg=%.2f\n"
        "fit_error=%.5f best_score=%.5f second_best=%.5f yaw_margin=%.5f\n"
        "Lerr=%.4f Werr=%.4f outside=%.4f inlier=%.3f edge=%.3f\n"
        "proj_L=%.4f proj_W=%.4f center_shift_m=%.5f\n"
        "center=%s long_axis=%s short_axis=%s\n"
        "top_corners=%s"
        % (
            label,
            str(bool(fit.get("success"))).lower(),
            fit.get("yaw_fit_method"),
            fit.get("yaw_source"),
            float(fit.get("yaw_confidence") or 0.0),
            float(fit.get("yaw_rad") or 0.0),
            float(fit.get("selected_yaw_deg") or 0.0),
            float(fit.get("fit_error") or 0.0),
            float(fit.get("best_score") or 0.0),
            float(fit.get("second_best_score") or 0.0),
            float(fit.get("yaw_margin_score") or 0.0),
            float(fit.get("length_error_m") or 0.0),
            float(fit.get("width_error_m") or 0.0),
            float(fit.get("outside_error_m") or 0.0),
            float(fit.get("inlier_ratio") or 0.0),
            float(fit.get("edge_support_score") or 0.0),
            float(fit.get("projected_extent_length_m") or 0.0),
            float(fit.get("projected_extent_width_m") or 0.0),
            float(fit.get("center_shift_from_median_m") or 0.0),
            fit.get("center_xyz"),
            fit.get("long_axis_xy"),
            fit.get("short_axis_xy"),
            fit.get("top_corners_base"),
        )
    )
    try:
        logger.info(msg)
    except Exception:
        pass
