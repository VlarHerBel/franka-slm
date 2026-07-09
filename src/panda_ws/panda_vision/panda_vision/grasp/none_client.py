"""Disabled grasp backend (no service calls)."""

from __future__ import annotations

from typing import List

from sensor_msgs.msg import PointCloud2

from panda_vision.types import GraspHypothesis


class NoGraspClient:
    @property
    def backend_id(self) -> str:
        return "none"

    def query(
        self, cloud: PointCloud2, frame_id: str, timeout_sec: float = 5.0
    ) -> tuple[List[GraspHypothesis], float, float]:
        del cloud, frame_id, timeout_sec
        return [], 0.0, 0.0
