"""Vision backend protocol (Strategy pattern)."""

from __future__ import annotations

from typing import List, Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from panda_vision.types import VisionDetection


@runtime_checkable
class VisionBackend(Protocol):
    """2D detection producing binary masks (and optional OBB polygons)."""

    def detect(
        self,
        bgr: NDArray[np.uint8],
        depth_m: NDArray[np.floating],
        *,
        text_prompt: str | None = None,
    ) -> List[VisionDetection]:
        ...

    @property
    def backend_id(self) -> str:
        ...

    @property
    def model_name(self) -> str:
        ...
