"""Ultralytics YOLO OBB (e.g. yolo26n-obb.pt) -> binary masks + oriented polygons."""

from __future__ import annotations

import time
from typing import List

import cv2
import numpy as np
from numpy.typing import NDArray

from panda_vision.types import VisionDetection

try:
    from ultralytics import YOLO

    _ULTRA = True
except ImportError:
    YOLO = None
    _ULTRA = False


def _obb_polygons_from_result(result) -> tuple[NDArray[np.float64], NDArray, NDArray] | None:
    """Return (polygons Nx4x2 float, confidences, class_ids) or None."""
    obb = getattr(result, "obb", None)
    if obb is None:
        return None
    xy = getattr(obb, "xyxyxyxy", None)
    if xy is None:
        return None
    polys = xy.cpu().numpy()
    if polys.size == 0:
        return None
    if polys.ndim == 2 and polys.shape[1] == 8:
        polys = polys.reshape(-1, 4, 2)
    elif polys.ndim != 3 or polys.shape[1] != 4:
        return None
    conf = obb.conf.cpu().numpy()
    cls = obb.cls.cpu().numpy().astype(int)
    return polys, conf, cls


class YOLOv26OBBBackend:
    """Strategy A: oriented boxes rasterized to masks."""

    def __init__(self, model_path: str, confidence: float, min_mask_pixels: int) -> None:
        if not _ULTRA or YOLO is None:
            raise RuntimeError("ultralytics is required for YOLOv26OBBBackend")
        self._model = YOLO(model_path)
        self._confidence = float(confidence)
        self._min_mask_pixels = int(min_mask_pixels)
        self._model_path = model_path

    @property
    def backend_id(self) -> str:
        return "yolo26_obb"

    @property
    def model_name(self) -> str:
        return self._model_path

    def detect(
        self,
        bgr: NDArray[np.uint8],
        depth_m: NDArray[np.floating],
        *,
        text_prompt: str | None = None,
    ) -> List[VisionDetection]:
        del text_prompt  # class-based detection only
        t0 = time.perf_counter()
        results = self._model.predict(
            source=bgr,
            conf=self._confidence,
            verbose=False,
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        out: List[VisionDetection] = []
        if not results:
            return out

        result = results[0]
        packed = _obb_polygons_from_result(result)
        if packed is None:
            return out

        polys, confs, classes = packed
        img_h, img_w = bgr.shape[:2]
        n = polys.shape[0]
        per_ms = elapsed_ms / max(1, n)

        for i in range(n):
            pts = polys[i].astype(np.float32)
            pts_int = np.round(pts).astype(np.int32)
            mask = np.zeros((img_h, img_w), dtype=np.uint8)
            cv2.fillPoly(mask, [pts_int], 1)
            mbool = mask.astype(bool)
            if int(mbool.sum()) < self._min_mask_pixels:
                continue

            cid = int(classes[i])
            label = result.names.get(cid, f"class_{cid}")
            label = str(label).strip().lower().replace(" ", "_")

            xs, ys = pts_int[:, 0], pts_int[:, 1]
            bbox = (
                int(np.clip(xs.min(), 0, img_w - 1)),
                int(np.clip(ys.min(), 0, img_h - 1)),
                int(np.clip(xs.max(), 0, img_w - 1)),
                int(np.clip(ys.max(), 0, img_h - 1)),
            )

            out.append(
                VisionDetection(
                    label=label,
                    score=float(confs[i]),
                    mask=mbool,
                    inference_ms=float(per_ms),
                    obb_polygon_uv=polys[i].copy(),
                    bbox_xyxy=bbox,
                )
            )

        return out
