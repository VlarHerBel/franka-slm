#!/usr/bin/env python3
"""Spawnea los 12 objetos YCB del dataset en mesa para foto de catálogo."""

from __future__ import annotations

import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import rclpy
import yaml
from ament_index_python.packages import get_package_share_directory
from rclpy.node import Node

from panda_vision.spawn.gz_spawn_runtime import (
    clear_runtime_ycb_entities,
    gz_world_name_from_param,
    spawn_entity,
)
from panda_vision.spawn.gazebo_spawn_pose_readback import (
    declare_spawn_pose_readback_params,
    readback_params_from_node,
    settle_and_build_gt_entry,
)
from panda_vision.spawn.runtime_scene_gt import (
    RuntimeSceneGtClient,
    make_gt_object_entry,
)
from panda_vision.spawn.runtime_scene_gt_geometry import (
    is_known_spawn_geometry_box_label,
    resolve_semantic_and_gazebo_poses,
)
from panda_vision.spawn.spawn_ycb_object import (
    _compute_z,
    quaternion_from_euler_sxyz,
)
from panda_vision.spawn.ycb_runtime_model_assets import (
    DEFAULT_RUNTIME_MODELS_ROOT,
    prepare_runtime_spawn_model,
)
from panda_vision.spawn.ycb_spawn_db import (
    REQUIRED_YCB_LABELS,
    YcbSpawnRecord,
    load_spawn_records_from_yaml,
)

LOGP = "[SPAWN_CATALOG]"


def _default_scene_path() -> Path:
    pkg_share = Path(get_package_share_directory("panda_vision"))
    return pkg_share / "config" / "ycb_catalog_photo_scene.yaml"


def _load_scene_entries(path: Path) -> Tuple[float, List[Dict[str, Any]]]:
    with path.expanduser().open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    table_z = float(raw.get("table_surface_z_m", 0.26))
    entries: List[Dict[str, Any]] = []
    for item in raw.get("objects") or []:
        if not isinstance(item, dict):
            continue
        label = str(item.get("label", "")).strip().lower()
        if not label:
            continue
        entries.append(
            {
                "label": label,
                "x": float(item.get("x", 0.0)),
                "y": float(item.get("y", 0.0)),
                "yaw": float(item.get("yaw", 0.0)),
            }
        )
    if not entries:
        raise ValueError(f"Sin objetos en escena: {path}")
    return table_z, entries


def _spawn_one(
    node: Node,
    *,
    rec: YcbSpawnRecord,
    label: str,
    entity_name: str,
    sdf_path: Path,
    x: float,
    y: float,
    yaw: float,
    table_z: float,
    spawn_eps: float,
    gz_world: str,
    spawn_backend: str,
    spawn_timeout: float,
    pose_topic: str,
    readback_params,
) -> Optional[Dict[str, object]]:
    z_val, _ = _compute_z(rec, table_z, spawn_eps, "", False, 0.0)
    roll = float(rec.base_roll_rad)
    pitch = float(rec.base_pitch_rad)
    yaw_eff = float(rec.base_yaw_rad) + float(yaw)
    qx, qy, qz, qw = quaternion_from_euler_sxyz(roll, pitch, yaw_eff)

    spawn_x, spawn_y, spawn_z = float(x), float(y), float(z_val)
    gt_x, gt_y, gt_z = spawn_x, spawn_y, spawn_z
    if is_known_spawn_geometry_box_label(label):
        semantic, gazebo = resolve_semantic_and_gazebo_poses(
            label,
            (float(x), float(y)),
            float(yaw_eff),
            table_z,
            float(rec.height_m),
            epsilon_m=spawn_eps,
            logger=node.get_logger(),
        )
        gt_x, gt_y, gt_z = semantic
        spawn_x, spawn_y, spawn_z = gazebo

    ok = spawn_entity(
        node,
        gz_world,
        entity_name,
        Path(sdf_path),
        spawn_x,
        spawn_y,
        spawn_z,
        qx,
        qy,
        qz,
        qw,
        spawn_backend,
        spawn_timeout,
    )
    if not ok:
        return None

    gt_entry = settle_and_build_gt_entry(
        node,
        entity_name=entity_name,
        label=label,
        requested_gazebo_xyz=(float(spawn_x), float(spawn_y), float(spawn_z)),
        requested_yaw_rad=float(yaw_eff),
        width_m=float(rec.footprint_width_m),
        length_m=float(rec.footprint_length_m),
        height_m=float(rec.height_m),
        world_pose_topic=pose_topic,
        params=readback_params,
        spawn_seed=None,
        logger=node.get_logger(),
    )
    if gt_entry is None:
        gt_entry = make_gt_object_entry(
            entity_name=entity_name,
            label=label,
            x=float(gt_x),
            y=float(gt_y),
            z=float(gt_z),
            roll=float(roll),
            pitch=float(pitch),
            yaw=float(yaw_eff),
            qx=float(qx),
            qy=float(qy),
            qz=float(qz),
            qw=float(qw),
            width_m=float(rec.footprint_width_m),
            length_m=float(rec.footprint_length_m),
            height_m=float(rec.height_m),
            spawn_seed=None,
            logger=node.get_logger(),
        )
    return gt_entry


def main(args: Optional[list] = None) -> None:
    rclpy.init(args=args)
    node = Node("spawn_ycb_catalog_photo")

    pkg_share = Path(get_package_share_directory("panda_vision"))
    default_config = pkg_share / "config" / "ycb_obb_dataset.yaml"
    default_scene = _default_scene_path()
    default_models = Path.home() / "tfg_robotics_ws" / "src" / "gazebo_ycb" / "models"

    node.declare_parameter("scene_path", str(default_scene))
    node.declare_parameter("config_path", str(default_config))
    node.declare_parameter("ycb_models_path", str(default_models))
    node.declare_parameter("world_name", "vision_test_ycb")
    node.declare_parameter("world_pose_ros_topic", "")
    node.declare_parameter("pose_discovery_sec", 2.0)
    node.declare_parameter("table_z", 0.0)
    node.declare_parameter("spawn_z_epsilon_m", 0.001)
    node.declare_parameter("spawn_name_prefix", "runtime_ycb")
    node.declare_parameter("delete_existing", True)
    node.declare_parameter("spawn_backend", "ros_gz_create_cli")
    node.declare_parameter("delete_backend", "gz_service_cli")
    node.declare_parameter("spawn_timeout_sec", 8.0)
    node.declare_parameter("delete_timeout_sec", 10.0)
    node.declare_parameter("delete_retries", 3)
    node.declare_parameter("verify_after_delete", True)
    node.declare_parameter("post_delete_settle_sec", 0.35)
    node.declare_parameter("wait_until_deleted_timeout_sec", 5.0)
    node.declare_parameter("spawn_settle_sec", 0.15)
    node.declare_parameter("texture_unique_runtime_models", True)
    node.declare_parameter("runtime_models_root", str(DEFAULT_RUNTIME_MODELS_ROOT))
    declare_spawn_pose_readback_params(node)

    scene_path = Path(str(node.get_parameter("scene_path").value)).expanduser()
    config_path = Path(str(node.get_parameter("config_path").value)).expanduser()
    ycb_models_path = Path(str(node.get_parameter("ycb_models_path").value)).expanduser()
    world_name = str(node.get_parameter("world_name").value).strip()
    world_pose_topic_in = str(node.get_parameter("world_pose_ros_topic").value).strip()
    pose_discovery_sec = float(node.get_parameter("pose_discovery_sec").value)
    table_z_override = float(node.get_parameter("table_z").value)
    spawn_eps = float(node.get_parameter("spawn_z_epsilon_m").value)
    spawn_prefix = str(node.get_parameter("spawn_name_prefix").value).strip()
    delete_existing = bool(node.get_parameter("delete_existing").value)
    spawn_backend = str(node.get_parameter("spawn_backend").value).strip()
    delete_backend = str(node.get_parameter("delete_backend").value).strip()
    spawn_timeout = float(node.get_parameter("spawn_timeout_sec").value)
    delete_timeout = float(node.get_parameter("delete_timeout_sec").value)
    delete_retries = int(node.get_parameter("delete_retries").value)
    verify_after_delete = bool(node.get_parameter("verify_after_delete").value)
    post_delete_settle = float(node.get_parameter("post_delete_settle_sec").value)
    wait_deleted_timeout = float(
        node.get_parameter("wait_until_deleted_timeout_sec").value
    )
    spawn_settle = float(node.get_parameter("spawn_settle_sec").value)
    texture_unique = bool(node.get_parameter("texture_unique_runtime_models").value)
    runtime_models_root = Path(
        str(node.get_parameter("runtime_models_root").value)
    ).expanduser()

    gz_world = gz_world_name_from_param(world_name)
    pose_topic = (
        world_pose_topic_in if world_pose_topic_in else f"/world/{gz_world}/pose/info"
    )

    if not scene_path.is_file():
        node.get_logger().fatal(f"{LOGP} ERROR: escena no encontrada: {scene_path}")
        node.destroy_node()
        rclpy.shutdown()
        sys.exit(2)

    try:
        table_z_yaml, entries = _load_scene_entries(scene_path)
    except Exception as exc:
        node.get_logger().fatal(f"{LOGP} ERROR cargando escena: {exc}")
        node.destroy_node()
        rclpy.shutdown()
        sys.exit(3)

    table_z = table_z_yaml if table_z_override <= 0.0 else table_z_override
    labels_in_scene = [str(e["label"]) for e in entries]
    missing = [lb for lb in REQUIRED_YCB_LABELS if lb not in labels_in_scene]
    extra = [lb for lb in labels_in_scene if lb not in REQUIRED_YCB_LABELS]
    if missing or extra or len(entries) != len(REQUIRED_YCB_LABELS):
        node.get_logger().fatal(
            f"{LOGP} ERROR: escena debe tener exactamente {list(REQUIRED_YCB_LABELS)}; "
            f"missing={missing} extra={extra} count={len(entries)}"
        )
        node.destroy_node()
        rclpy.shutdown()
        sys.exit(4)

    records = load_spawn_records_from_yaml(config_path)
    readback_params = readback_params_from_node(node)
    gt_client = RuntimeSceneGtClient(node, world_frame="world")

    node.get_logger().info(
        f"{LOGP} scene={scene_path.name} objects={len(entries)} table_z={table_z:.4f}"
    )

    if delete_existing:
        ok_clear, remaining = clear_runtime_ycb_entities(
            node,
            gz_world_name=gz_world,
            pose_topic=pose_topic,
            pose_discovery_sec=pose_discovery_sec,
            spawn_name_prefix=spawn_prefix,
            delete_backend=delete_backend,
            delete_timeout_sec=delete_timeout,
            delete_retries=delete_retries,
            verify_after_delete=verify_after_delete,
            list_only=False,
            label=None,
            log_prefix="[CLEAR_YCB]",
            post_delete_settle_sec=post_delete_settle,
            wait_until_deleted_timeout_sec=wait_deleted_timeout,
        )
        if not ok_clear:
            node.get_logger().error(
                f"{LOGP} no se limpiaron todos los objetos previos: {remaining}"
            )
            node.destroy_node()
            rclpy.shutdown()
            sys.exit(5)
        gt_client.clear()

    gt_entries: List[Dict[str, object]] = []
    failed: List[str] = []

    for idx, entry in enumerate(entries):
        label = str(entry["label"])
        rec = records.get(label)
        if rec is None:
            failed.append(label)
            continue

        source_model_dir = (ycb_models_path / rec.model_name).resolve()
        source_sdf = source_model_dir / "model.sdf"
        if not source_sdf.is_file():
            node.get_logger().error(
                f"{LOGP} model.sdf no encontrado para {label}: {source_sdf}"
            )
            failed.append(label)
            continue

        entity_name = f"{spawn_prefix}_{label}"
        if texture_unique:
            try:
                sdf_path, _, _ = prepare_runtime_spawn_model(
                    label,
                    source_model_dir,
                    runtime_models_root=runtime_models_root,
                    logger=node.get_logger(),
                )
            except Exception as exc:
                node.get_logger().error(
                    f"{LOGP} textura runtime falló para {label}: {exc}"
                )
                failed.append(label)
                continue
        else:
            sdf_path = source_sdf

        node.get_logger().info(
            f"{LOGP} [{idx + 1}/{len(entries)}] label={label} "
            f"xy=({float(entry['x']):.4f},{float(entry['y']):.4f}) "
            f"yaw={float(entry['yaw']):.4f}"
        )
        gt_entry = _spawn_one(
            node,
            rec=rec,
            label=label,
            entity_name=entity_name,
            sdf_path=Path(sdf_path),
            x=float(entry["x"]),
            y=float(entry["y"]),
            yaw=float(entry["yaw"]),
            table_z=table_z,
            spawn_eps=spawn_eps,
            gz_world=gz_world,
            spawn_backend=spawn_backend,
            spawn_timeout=spawn_timeout,
            pose_topic=pose_topic,
            readback_params=readback_params,
        )
        if gt_entry is None:
            failed.append(label)
            continue
        gt_entries.append(gt_entry)
        if spawn_settle > 0.0:
            time.sleep(spawn_settle)

    if gt_entries:
        gt_client.replace_all(gt_entries)
    else:
        gt_client.clear()

    if failed:
        node.get_logger().error(f"{LOGP} fallaron: {failed}")
        node.destroy_node()
        rclpy.shutdown()
        sys.exit(6)

    node.get_logger().info(
        f"{LOGP} OK: {len(gt_entries)} objetos spawneados. "
        "Ajusta la cámara en Gazebo y captura la imagen."
    )
    node.destroy_node()
    rclpy.shutdown()
    sys.exit(0)


if __name__ == "__main__":
    main()
