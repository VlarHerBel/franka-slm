"""Strategy B: text-prompted segmentation (optional Grounded SAM 2 stack).

This module ships a **stub** that returns no detections unless you install and
wire a real pipeline (e.g. Grounding DINO + SAM 2). Keeps CI and headless
installs lightweight.

Optional install (user environment, CUDA typical):
  pip install torch ...  # plus your Grounded-SAM2 / official repos

Subclass or replace ``_run_grounded_sam2`` when integrating.
"""

from __future__ import annotations

import time
from typing import Callable, List, Optional

import numpy as np
from numpy.typing import NDArray

from panda_vision.types import VisionDetection


class GroundedSAM2Backend:
    """Text-conditioned masks; default implementation is a no-op stub."""

    def __init__(
        self,
        model_name: str = "stub",
        min_mask_pixels: int = 200,
        segment_fn: Optional[
            Callable[[NDArray[np.uint8], str], NDArray[np.bool_]]
        ] = None,
    ) -> None:
        self._model_name = model_name
        self._min_mask_pixels = int(min_mask_pixels)
        self._segment_fn = segment_fn

    @property
    def backend_id(self) -> str:
        return "grounded_sam2"

    @property
    def model_name(self) -> str:
        return self._model_name

    def detect(
        self,
        bgr: NDArray[np.uint8],
        depth_m: NDArray[np.floating],
        *,
        text_prompt: str | None = None,
    ) -> List[VisionDetection]:
        del depth_m
        prompt = (text_prompt or "").strip()
        if not prompt:
            return []

        t0 = time.perf_counter()
        if self._segment_fn is None:
            return []

        mask = self._segment_fn(bgr, prompt)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        if mask.shape[:2] != bgr.shape[:2]:
            return []

        mbool = mask.astype(bool)
        if int(mbool.sum()) < self._min_mask_pixels:
            return []

        return [
            VisionDetection(
                label=prompt.replace(" ", "_").lower(),
                score=1.0,
                mask=mbool,
                inference_ms=float(elapsed_ms),
                obb_polygon_uv=None,
                bbox_xyxy=None,
            )
        ]
