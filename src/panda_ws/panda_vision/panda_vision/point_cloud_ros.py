"""ROS PointCloud2 helpers."""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
from numpy.typing import NDArray
from sensor_msgs.msg import PointCloud2, PointField
from sensor_msgs_py import point_cloud2 as pc2
from std_msgs.msg import Header


def numpy_xyz_to_pointcloud2(
    points_xyz: NDArray[np.floating],
    frame_id: str,
    stamp: Optional[Any] = None,
) -> PointCloud2:
    """Pack Nx3 float points into ``sensor_msgs/PointCloud2``."""
    pts = np.asarray(points_xyz, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError("points_xyz must be Nx3")

    header = Header()
    header.frame_id = frame_id
    if stamp is not None:
        header.stamp = stamp

    fields = [
        PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
    ]
    tuples = [tuple(row) for row in pts.tolist()]
    return pc2.create_cloud(header, fields, tuples)
