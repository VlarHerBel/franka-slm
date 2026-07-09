"""Grasp / pose backend protocol."""

from __future__ import annotations

from typing import List, Protocol, runtime_checkable

from sensor_msgs.msg import PointCloud2

from panda_vision.types import GraspHypothesis


@runtime_checkable
class GraspEstimatorClient(Protocol):
    @property
    def backend_id(self) -> str:
        ...

    def query(
        self, cloud: PointCloud2, frame_id: str, timeout_sec: float = 5.0
    ) -> tuple[List[GraspHypothesis], float, float]:
        """Return (hypotheses, best_confidence, latency_ms)."""
        ...
