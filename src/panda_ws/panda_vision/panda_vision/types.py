"""Shared typed structures for the modular perception pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray


@dataclass
class VisionDetection:
    """Single 2D vision output (mask + metadata)."""

    label: str
    score: float
    mask: NDArray[np.bool_]
    inference_ms: float
    obb_polygon_uv: Optional[NDArray[np.float64]] = None
    # Optional axis-aligned bbox [x1,y1,x2,y2] for debug
    bbox_xyxy: Optional[Tuple[int, int, int, int]] = None


@dataclass
class TopSurfaceResult:
    """3D top-plane estimate from Open3D."""

    centroid_camera: NDArray[np.float64]
    plane_normal: NDArray[np.float64]
    inlier_ratio: float
    ransac_rmse: float
    num_inliers: int
    num_points_input: int
    success: bool
    message: str = ""


@dataclass
class GraspHypothesis:
    """One SE(3) grasp candidate in the cloud frame."""

    position: Tuple[float, float, float]
    orientation_xyzw: Tuple[float, float, float, float]
    confidence: float


@dataclass
class PerceptionTelemetry:
    """Aggregated timings and backend ids for JSON export."""

    vision_backend: str
    vision_model_name: str
    vision_inference_ms_total: float
    vision_inference_ms_per_detection: List[float] = field(default_factory=list)
    text_prompt_used: str = ""
    open3d_ms_total: float = 0.0
    open3d_plane_inlier_ratio: float = 0.0
    open3d_ransac_rmse: float = 0.0
    grasp_backend: str = ""
    grasp_confidence: float = 0.0
    grasp_service_latency_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "vision_backend": self.vision_backend,
            "vision_model_name": self.vision_model_name,
            "vision_inference_ms_total": self.vision_inference_ms_total,
            "vision_inference_ms_per_detection": list(
                self.vision_inference_ms_per_detection
            ),
            "text_prompt_used": self.text_prompt_used,
            "open3d_ms_total": self.open3d_ms_total,
            "open3d_plane_inlier_ratio": self.open3d_plane_inlier_ratio,
            "open3d_ransac_rmse": self.open3d_ransac_rmse,
            "grasp_backend": self.grasp_backend,
            "grasp_confidence": self.grasp_confidence,
            "grasp_service_latency_ms": self.grasp_service_latency_ms,
        }


def pose_to_dict(
    position: Sequence[float], quat_xyzw: Sequence[float]
) -> dict[str, Any]:
    return {
        "position": [float(position[0]), float(position[1]), float(position[2])],
        "orientation": {
            "x": float(quat_xyzw[0]),
            "y": float(quat_xyzw[1]),
            "z": float(quat_xyzw[2]),
            "w": float(quat_xyzw[3]),
        },
    }
