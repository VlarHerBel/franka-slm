"""Instantiate a vision backend from ROS parameters."""

from __future__ import annotations

from typing import Callable, Optional

from panda_vision.backends.base import VisionBackend
from panda_vision.backends.grounded_sam2_backend import GroundedSAM2Backend
from panda_vision.backends.yolo26_obb_backend import YOLOv26OBBBackend


def create_vision_backend(
    backend_id: str,
    *,
    model_path: str,
    confidence: float,
    min_mask_pixels: int,
    grounded_model_name: str = "stub",
    grounded_segment_fn: Optional[
        Callable[..., object]
    ] = None,
) -> VisionBackend:
    bid = backend_id.strip().lower()
    if bid == "yolo26_obb":
        return YOLOv26OBBBackend(model_path, confidence, min_mask_pixels)
    if bid == "grounded_sam2":
        return GroundedSAM2Backend(
            model_name=grounded_model_name,
            min_mask_pixels=min_mask_pixels,
            segment_fn=grounded_segment_fn,
        )
    raise ValueError(f"Unknown vision_backend: {backend_id!r}")
