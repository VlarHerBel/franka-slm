"""Top-surface estimation: normals toward camera, optical-axis filter, RANSAC plane."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import open3d as o3d
from numpy.typing import NDArray

from panda_vision.geometry.camera_projection import depth_mask_to_points_camera
from panda_vision.types import TopSurfaceResult


@dataclass(frozen=True)
class Open3DTopSurfaceConfig:
    voxel_size_m: float = 0.003
    normal_nn_max_nn: int = 30
    normal_radius_m: float = 0.02
    max_normal_angle_deg: float = 35.0
    ransac_distance_threshold_m: float = 0.004
    ransac_n: int = 3
    ransac_num_iterations: int = 1000
    min_inliers: int = 50
    z_min_depth: float = 0.05
    # If RANSAC fails, fall back to centroid of normal-filtered cloud
    fallback_centroid_on_ransac_fail: bool = True


def estimate_top_surface_plane_centroid(
    mask: NDArray[np.bool_],
    depth_m: NDArray[np.floating],
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    config: Open3DTopSurfaceConfig | None = None,
) -> TopSurfaceResult:
    """Build a point cloud from mask+depth, filter by normals, fit a plane, return centroid.

    Convention (ROS camera optical frame): +Z is the optical axis into the scene.
    After ``orient_normals_towards_camera_location`` at the origin, visible surface
    normals point toward the camera, i.e. roughly opposite to +Z. We keep points
    whose normal satisfies ``dot(n, -z_hat) >= cos(theta_max)`` so lateral faces
    (bottles, box sides) are dropped when viewing from above.
    """
    cfg = config or Open3DTopSurfaceConfig()

    pts = depth_mask_to_points_camera(
        mask, depth_m, fx, fy, cx, cy, z_min=cfg.z_min_depth
    )
    n_in = int(pts.shape[0])
    if n_in < cfg.min_inliers:
        return TopSurfaceResult(
            centroid_camera=np.zeros(3, dtype=np.float64),
            plane_normal=np.array([0.0, 0.0, 1.0], dtype=np.float64),
            inlier_ratio=0.0,
            ransac_rmse=0.0,
            num_inliers=0,
            num_points_input=n_in,
            success=False,
            message="too_few_points",
        )

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)

    if cfg.voxel_size_m > 0:
        pcd = pcd.voxel_down_sample(voxel_size=cfg.voxel_size_m)

    if len(pcd.points) < cfg.min_inliers:
        return TopSurfaceResult(
            centroid_camera=np.zeros(3, dtype=np.float64),
            plane_normal=np.array([0.0, 0.0, 1.0], dtype=np.float64),
            inlier_ratio=0.0,
            ransac_rmse=0.0,
            num_inliers=0,
            num_points_input=n_in,
            success=False,
            message="too_few_after_voxel",
        )

    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=cfg.normal_radius_m, max_nn=cfg.normal_nn_max_nn
        )
    )
    camera_location = np.zeros(3, dtype=np.float64)
    pcd.orient_normals_towards_camera_location(camera_location)

    points = np.asarray(pcd.points)
    normals = np.asarray(pcd.normals)
    z_hat = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    cos_thresh = float(np.cos(np.deg2rad(cfg.max_normal_angle_deg)))
    # Normals point toward camera -> aligned with -Z
    align = np.dot(normals, -z_hat)
    keep = align >= cos_thresh
    filtered = points[keep]
    filtered_normals = normals[keep]

    if filtered.shape[0] < cfg.min_inliers:
        return TopSurfaceResult(
            centroid_camera=np.zeros(3, dtype=np.float64),
            plane_normal=np.array([0.0, 0.0, 1.0], dtype=np.float64),
            inlier_ratio=0.0,
            ransac_rmse=0.0,
            num_inliers=0,
            num_points_input=n_in,
            success=False,
            message="normal_filter_removed_too_many",
        )

    fpcd = o3d.geometry.PointCloud()
    fpcd.points = o3d.utility.Vector3dVector(filtered)
    fpcd.normals = o3d.utility.Vector3dVector(filtered_normals)

    plane_model, inliers = fpcd.segment_plane(
        distance_threshold=cfg.ransac_distance_threshold_m,
        ransac_n=cfg.ransac_n,
        num_iterations=cfg.ransac_num_iterations,
    )
    a, b, c, d = plane_model
    plane_normal = np.array([a, b, c], dtype=np.float64)
    norm = np.linalg.norm(plane_normal)
    if norm > 1e-9:
        plane_normal = plane_normal / norm
    # Orient normal to point toward camera (consistent with -Z preference)
    if np.dot(plane_normal, -z_hat) < 0:
        plane_normal = -plane_normal

    inlier_pts = filtered[np.asarray(inliers, dtype=np.int64)]
    n_inl = int(inlier_pts.shape[0])
    inlier_ratio = n_inl / max(1, filtered.shape[0])

    if n_inl >= cfg.min_inliers:
        centroid = inlier_pts.mean(axis=0)
        dists = np.abs(a * inlier_pts[:, 0] + b * inlier_pts[:, 1] + c * inlier_pts[:, 2] + d)
        ransac_rmse = float(np.sqrt(np.mean(dists**2)))
        return TopSurfaceResult(
            centroid_camera=centroid,
            plane_normal=plane_normal,
            inlier_ratio=float(inlier_ratio),
            ransac_rmse=ransac_rmse,
            num_inliers=n_inl,
            num_points_input=n_in,
            success=True,
            message="ok",
        )

    if cfg.fallback_centroid_on_ransac_fail:
        centroid = filtered.mean(axis=0)
        return TopSurfaceResult(
            centroid_camera=centroid,
            plane_normal=plane_normal,
            inlier_ratio=float(inlier_ratio),
            ransac_rmse=0.0,
            num_inliers=n_inl,
            num_points_input=n_in,
            success=True,
            message="fallback_centroid_no_ransac_inliers",
        )

    return TopSurfaceResult(
        centroid_camera=np.zeros(3, dtype=np.float64),
        plane_normal=plane_normal,
        inlier_ratio=float(inlier_ratio),
        ransac_rmse=0.0,
        num_inliers=n_inl,
        num_points_input=n_in,
        success=False,
        message="ransac_insufficient_inliers",
    )
