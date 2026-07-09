"""Construcción y publicación de colisiones YCB en MoveIt planning scene."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

from panda_controller.demo_scene_reachability_scan import (
    DEFAULT_TABLE_CENTER,
    DEFAULT_TABLE_FRAME,
    DEFAULT_TABLE_SIZE,
    DEFAULT_TABLE_Z_M,
    _vision_policy_exports,
    compute_top_z_m,
    downward_yaw_quaternion,
)


def _yaw_to_quat(yaw_rad: float) -> Tuple[float, float, float, float]:
    q = downward_yaw_quaternion(float(yaw_rad))
    return (float(q[0]), float(q[1]), float(q[2]), float(q[3]))


def collision_shape_and_dims(
    label: str,
    *,
    collision_dims: Optional[Dict[str, Any]] = None,
) -> Tuple[str, List[float]]:
    col = collision_dims
    if col is None:
        _, get_collision_dimensions, _ = _vision_policy_exports()
        col = get_collision_dimensions(str(label), padding_m=0.0)
    if isinstance(col, dict):
        if col.get("shape") == "cylinder" and col.get("cylinder"):
            cyl = col["cylinder"]
            return "cylinder", [float(cyl[0]), float(cyl[1])]
        box = col.get("box") or col.get("box_fallback")
        if isinstance(box, (list, tuple)) and len(box) >= 3:
            return "box", [float(box[0]), float(box[1]), float(box[2])]
    # fallback conservador
    from panda_controller.demo_scene_layout_validator import OBJECT_FOOTPRINT_LW_M

    l_m, w_m = OBJECT_FOOTPRINT_LW_M.get(str(label), (0.10, 0.10))
    return "box", [float(l_m), float(w_m), 0.10]


def build_planning_scene_object(
    *,
    label: str,
    entity_name: str,
    spawn_x: float,
    spawn_y: float,
    yaw: float,
    table_z_m: float = DEFAULT_TABLE_Z_M,
    is_target: bool = False,
) -> Dict[str, Any]:
    export_grasp_policy_for_executor, get_collision_dimensions, _ = _vision_policy_exports()
    policy = export_grasp_policy_for_executor(str(label))
    policy["label"] = str(label)
    top_z, geometry_center_z = compute_top_z_m(str(label), float(table_z_m), policy)
    col = get_collision_dimensions(str(label), padding_m=0.0)
    shape, dims = collision_shape_and_dims(str(label), collision_dims=col)
    height = float(dims[2]) if shape == "box" else float(dims[1])
    center_z = float(table_z_m) + height / 2.0
    return {
        "label": str(label),
        "entity_name": str(entity_name),
        "is_target": bool(is_target),
        "position": [float(spawn_x), float(spawn_y), float(geometry_center_z)],
        "top_z_m": float(top_z),
        "object_yaw_rad": float(yaw),
        "collision_dims": col,
        "_collision_shape": shape,
        "_collision_dims_resolved": list(dims),
        "_collision_center_z": float(center_z),
        "_planning_collision_id": "scene_%s_%s" % (str(label), str(entity_name)),
    }


def build_scene_objects_from_layout(
    objects: Dict[str, Any],
    *,
    table_z_m: float = DEFAULT_TABLE_Z_M,
) -> List[Dict[str, Any]]:
    """objects: label -> LayoutObjectPose."""
    out: List[Dict[str, Any]] = []
    for label, pose in objects.items():
        ent = "demo_%s_0" % str(label)
        out.append(
            build_planning_scene_object(
                label=str(label),
                entity_name=ent,
                spawn_x=float(pose.spawn_x),
                spawn_y=float(pose.spawn_y),
                yaw=float(pose.yaw),
                table_z_m=float(table_z_m),
                is_target=False,
            )
        )
    return out


class MoveItCollisionScenePublisher:
    """Publica mesa + objetos YCB en /planning_scene (sin mover el robot)."""

    def __init__(self, backend: Any) -> None:
        self._backend = backend
        self._active_ids: List[str] = []

    def clear_objects(self) -> None:
        if self._backend._moveit2 is None:
            return
        PlanningScene = self._backend._PlanningScene
        CollisionObject = self._backend._CollisionObject
        for col_id in list(self._active_ids):
            col = CollisionObject()
            col.id = str(col_id)
            col.operation = CollisionObject.REMOVE
            scene = PlanningScene()
            scene.is_diff = True
            scene.world.collision_objects = [col]
            self._backend._planning_scene_pub.publish(scene)
        self._active_ids = []

    def publish_table(self) -> None:
        self._backend.apply_table_collision()

    def publish_objects(
        self,
        scene_objects: Sequence[Dict[str, Any]],
        *,
        include_labels: Optional[Sequence[str]] = None,
        exclude_labels: Optional[Sequence[str]] = None,
        target_label: str = "",
    ) -> None:
        self.clear_objects()
        self.publish_table()
        include = {str(x).strip().lower() for x in (include_labels or [])}
        exclude = {str(x).strip().lower() for x in (exclude_labels or [])}
        target_l = str(target_label or "").strip().lower()
        for obj in scene_objects:
            if not isinstance(obj, dict):
                continue
            lb = str(obj.get("label", "")).strip().lower()
            if include and lb not in include:
                continue
            if lb in exclude:
                continue
            is_target = bool(target_l and lb == target_l)
            self._publish_one(obj, is_target=is_target)

    def _publish_one(self, obj: Dict[str, Any], *, is_target: bool) -> None:
        col_id = str(obj.get("_planning_collision_id") or obj.get("label"))
        shape = str(obj.get("_collision_shape") or "box")
        dims = obj.get("_collision_dims_resolved") or [0.1, 0.1, 0.1]
        pos = obj.get("position") or [0, 0, 0]
        yaw = float(obj.get("object_yaw_rad") or 0.0)
        center_z = float(obj.get("_collision_center_z") or pos[2])

        primitive = self._backend._SolidPrimitive()
        if shape == "cylinder":
            primitive.type = self._backend._SolidPrimitive.CYLINDER
            primitive.dimensions = [float(dims[1]), float(dims[0])]
        else:
            primitive.type = self._backend._SolidPrimitive.BOX
            primitive.dimensions = [float(dims[0]), float(dims[1]), float(dims[2])]

        pose = self._backend._PoseStamped()
        pose.header.frame_id = DEFAULT_TABLE_FRAME
        pose.pose.position.x = float(pos[0])
        pose.pose.position.y = float(pos[1])
        pose.pose.position.z = float(center_z)
        qx, qy, qz, qw = _yaw_to_quat(yaw)
        pose.pose.orientation.x = qx
        pose.pose.orientation.y = qy
        pose.pose.orientation.z = qz
        pose.pose.orientation.w = qw

        col = self._backend._CollisionObject()
        col.id = col_id
        col.header.frame_id = DEFAULT_TABLE_FRAME
        col.operation = col.ADD
        col.primitives = [primitive]
        col.primitive_poses = [pose.pose]

        scene = self._backend._PlanningScene()
        scene.is_diff = True
        scene.world.collision_objects = [col]
        self._backend._planning_scene_pub.publish(scene)
        self._active_ids.append(col_id)
