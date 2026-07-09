#!/usr/bin/env python3
"""Nodo persistente que publica ground-truth latched en ``/runtime_scene/gt_objects``."""

from __future__ import annotations

import rclpy
from panda_vision_interfaces.srv import SetRuntimeSceneGt
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from std_srvs.srv import Trigger

from panda_vision.spawn.runtime_scene_gt import (
    GT_CLEAR_SERVICE,
    GT_OBJECTS_TOPIC,
    GT_SET_SERVICE,
    RuntimeSceneGtStore,
    RuntimeSceneGtPublisher,
    parse_gt_payload_from_json,
)


class RuntimeSceneGtNode(Node):
    def __init__(self) -> None:
        super().__init__("runtime_scene_gt_node")
        self.declare_parameter("publish_hz", 1.0)
        self.declare_parameter("world_frame", "world")

        world_frame = str(self.get_parameter("world_frame").value)
        publish_hz = max(0.1, float(self.get_parameter("publish_hz").value))

        self._store = RuntimeSceneGtStore(world_frame=world_frame)
        self._publisher = RuntimeSceneGtPublisher(
            self, store=self._store, topic=GT_OBJECTS_TOPIC
        )

        self.create_service(
            SetRuntimeSceneGt, GT_SET_SERVICE, self._set_gt_cb
        )
        self.create_service(Trigger, GT_CLEAR_SERVICE, self._clear_gt_cb)

        period = 1.0 / publish_hz
        self.create_timer(period, self._timer_publish_cb)

        self._publisher.publish(log_tag="RUNTIME_SCENE_GT_PUBLISH")
        self.get_logger().info(
            "[RUNTIME_SCENE_GT_NODE_START] topic=%s publish_hz=%.2f services=%s,%s"
            % (GT_OBJECTS_TOPIC, publish_hz, GT_SET_SERVICE, GT_CLEAR_SERVICE)
        )

    def _set_gt_cb(self, request, response):
        ok, msg, n = self._apply_json(request.data)
        response.success = ok
        response.message = msg
        if ok:
            self._publisher.publish(log_tag="RUNTIME_SCENE_GT_PUBLISH")
            self.get_logger().info(
                "[RUNTIME_SCENE_GT_UPDATE] n=%d via_service=%s"
                % (n, GT_SET_SERVICE)
            )
        return response

    def _clear_gt_cb(self, _request, response):
        self._store.clear()
        self._publisher.publish(log_tag="RUNTIME_SCENE_GT_PUBLISH")
        self.get_logger().info("[RUNTIME_SCENE_GT_CLEAR] n=0")
        response.success = True
        response.message = "GT cleared"
        return response

    def _apply_json(self, raw: str) -> tuple[bool, str, int]:
        data, err = parse_gt_payload_from_json(raw)
        if data is None:
            return False, err or "invalid JSON", 0
        if "objects" not in data:
            return False, 'payload must contain "objects"', 0
        objects = data["objects"]
        if not isinstance(objects, list):
            return False, '"objects" must be a list', 0
        self._store.replace_all(objects)
        return True, f"updated n={len(objects)}", len(objects)

    def _timer_publish_cb(self) -> None:
        self._publisher.publish(log_tag="RUNTIME_SCENE_GT_PUBLISH")


def main(args=None) -> None:
    rclpy.init(args=args)
    node = RuntimeSceneGtNode()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
