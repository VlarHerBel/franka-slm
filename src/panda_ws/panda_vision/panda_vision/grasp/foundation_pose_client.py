"""ROS 2 client for ``panda_vision_interfaces/GetObjectPose``."""

from __future__ import annotations

import time
from typing import List, Optional

import rclpy
from geometry_msgs.msg import Pose
from rclpy.callback_groups import CallbackGroup
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2

from panda_vision.types import GraspHypothesis

from panda_vision_interfaces.srv import GetObjectPose


class FoundationPoseClient:
    """Method A: single object pose (centre) from segmented cloud."""

    def __init__(
        self,
        node: Node,
        service_name: str,
        callback_group: Optional[CallbackGroup] = None,
    ) -> None:
        self._node = node
        kw = {}
        if callback_group is not None:
            kw["callback_group"] = callback_group
        self._client = node.create_client(GetObjectPose, service_name, **kw)
        self._service_name = service_name

    @property
    def backend_id(self) -> str:
        return "foundation_pose"

    def query(
        self, cloud: PointCloud2, frame_id: str, timeout_sec: float = 5.0
    ) -> tuple[List[GraspHypothesis], float, float]:
        t0 = time.perf_counter()
        if not self._client.wait_for_service(timeout_sec=min(2.0, timeout_sec)):
            self._node.get_logger().warn(
                f"GetObjectPose service not available: {self._service_name}"
            )
            return [], 0.0, (time.perf_counter() - t0) * 1000.0

        req = GetObjectPose.Request()
        req.object_cloud = cloud
        req.frame_id = frame_id

        future = self._client.call_async(req)
        rclpy.spin_until_future_complete(self._node, future, timeout_sec=timeout_sec)
        latency_ms = (time.perf_counter() - t0) * 1000.0

        if not future.done():
            self._node.get_logger().warn("GetObjectPose call timed out")
            return [], 0.0, latency_ms

        resp = future.result()
        if resp is None or not resp.success:
            return [], 0.0, latency_ms

        pose: Pose = resp.pose
        hyp = GraspHypothesis(
            position=(
                float(pose.position.x),
                float(pose.position.y),
                float(pose.position.z),
            ),
            orientation_xyzw=(
                float(pose.orientation.x),
                float(pose.orientation.y),
                float(pose.orientation.z),
                float(pose.orientation.w),
            ),
            confidence=float(resp.confidence),
        )
        return [hyp], float(resp.confidence), latency_ms
