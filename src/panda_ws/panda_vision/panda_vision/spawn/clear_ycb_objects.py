#!/usr/bin/env python3
"""Borra modelos runtime YCB en Gazebo Sim (ros_gz) por prefijo con verificación y reintentos."""

from __future__ import annotations

import sys
from typing import Optional

import rclpy
from rclpy.node import Node

from panda_vision.spawn.gz_spawn_runtime import (
    clear_runtime_ycb_entities,
    gz_world_name_from_param,
)
from panda_vision.spawn.runtime_scene_gt import RuntimeSceneGtClient

LOGP = "[CLEAR_YCB]"


def main(args: Optional[list] = None) -> None:
    rclpy.init(args=args)
    node = Node("clear_ycb_objects")

    node.declare_parameter("delete_all_runtime_ycb", True)
    node.declare_parameter("spawn_name_prefix", "runtime_ycb")
    node.declare_parameter("world_name", "vision_test_ycb")
    node.declare_parameter("world_pose_ros_topic", "")
    node.declare_parameter("pose_discovery_sec", 2.0)
    node.declare_parameter("delete_backend", "gz_service_cli")
    node.declare_parameter("delete_timeout_sec", 10.0)
    node.declare_parameter("delete_retries", 3)
    node.declare_parameter("verify_after_delete", True)
    node.declare_parameter("list_only", False)
    node.declare_parameter("label", "")
    node.declare_parameter("post_delete_settle_sec", 0.35)
    node.declare_parameter("wait_until_deleted_timeout_sec", 5.0)

    delete_all = bool(node.get_parameter("delete_all_runtime_ycb").value)
    spawn_prefix = str(node.get_parameter("spawn_name_prefix").value).strip()
    world_name = str(node.get_parameter("world_name").value).strip()
    world_pose_topic_in = str(node.get_parameter("world_pose_ros_topic").value).strip()
    pose_discovery_sec = float(node.get_parameter("pose_discovery_sec").value)
    delete_backend = str(node.get_parameter("delete_backend").value).strip()
    delete_timeout = float(node.get_parameter("delete_timeout_sec").value)
    delete_retries = int(node.get_parameter("delete_retries").value)
    verify_after_delete = bool(node.get_parameter("verify_after_delete").value)
    list_only = bool(node.get_parameter("list_only").value)
    label = str(node.get_parameter("label").value).strip().lower()
    post_delete_settle = float(node.get_parameter("post_delete_settle_sec").value)
    wait_deleted_timeout = float(
        node.get_parameter("wait_until_deleted_timeout_sec").value
    )

    gz_world = gz_world_name_from_param(world_name)
    pose_topic = (
        world_pose_topic_in if world_pose_topic_in else f"/world/{gz_world}/pose/info"
    )

    if not list_only and not delete_all and not label:
        node.get_logger().fatal(
            f"{LOGP} ERROR: indique delete_all_runtime_ycb:=true o label:=<objeto>"
        )
        node.destroy_node()
        rclpy.shutdown()
        sys.exit(2)

    lb = label if label and not delete_all else None
    ok, remaining = clear_runtime_ycb_entities(
        node,
        gz_world_name=gz_world,
        pose_topic=pose_topic,
        pose_discovery_sec=pose_discovery_sec,
        spawn_name_prefix=spawn_prefix,
        delete_backend=delete_backend,
        delete_timeout_sec=delete_timeout,
        delete_retries=delete_retries,
        verify_after_delete=verify_after_delete,
        list_only=list_only,
        label=lb,
        log_prefix=LOGP,
        post_delete_settle_sec=post_delete_settle,
        wait_until_deleted_timeout_sec=wait_deleted_timeout,
    )

    if list_only:
        node.get_logger().info(f"{LOGP} list_only: {len(remaining)} entidad(es)")
        node.destroy_node()
        rclpy.shutdown()
        sys.exit(0)

    if ok or not verify_after_delete:
        RuntimeSceneGtClient(node, world_frame="world").clear()

    node.destroy_node()
    rclpy.shutdown()
    if verify_after_delete:
        sys.exit(0 if ok else 1)
    sys.exit(0)


if __name__ == "__main__":
    main()
