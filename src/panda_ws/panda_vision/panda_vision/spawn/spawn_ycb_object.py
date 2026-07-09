#!/usr/bin/env python3
"""Spawn determinista de un objeto YCB en Gazebo Sim (ros_gz). No publica pose a visión."""

from __future__ import annotations

import math
import random
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import rclpy
from ament_index_python.packages import get_package_share_directory
from rclpy.node import Node

from panda_vision.spawn.gz_spawn_runtime import (
    clear_runtime_ycb_entities,
    delete_entity,
    discover_runtime_entity_names,
    filter_runtime_entity_names,
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
from panda_vision.spawn.semantic_spawn_sampling import (
    TableSpawnRegion,
    sample_semantic_box_pose_xyyaw,
)
from panda_vision.spawn.ycb_runtime_model_assets import (
    DEFAULT_RUNTIME_MODELS_ROOT,
    prepare_runtime_spawn_model,
)
from panda_vision.spawn.ycb_spawn_db import (
    YcbSpawnRecord,
    ensure_all_labels_present,
    load_spawn_records_from_yaml,
)

LOGP = "[SPAWN_YCB]"

PRESETS_XY_YAW_DEG: Dict[str, Tuple[float, float, float]] = {
    "center_0deg": (0.56, 0.00, 0.0),
    "center_45deg": (0.56, 0.00, 45.0),
    "center_90deg": (0.56, 0.00, 90.0),
    "center_135deg": (0.56, 0.00, 135.0),
    "front_center_0deg": (0.50, 0.00, 0.0),
    "front_center_90deg": (0.50, 0.00, 90.0),
}


def _compute_z(
    rec: YcbSpawnRecord,
    table_z: float,
    epsilon: float,
    mode_override: str,
    use_explicit_z: bool,
    explicit_z: float,
) -> Tuple[float, str]:
    if use_explicit_z:
        return float(explicit_z), "explicit_param"
    mode = (mode_override or rec.origin_z_mode or "runtime_yaml").strip().lower()
    off = float(rec.origin_z_offset_m)
    h = float(rec.height_m)
    if mode == "runtime_yaml":
        z = (
            float(table_z)
            + float(rec.spawn_height_m)
            + float(rec.spawn_z_offset_m)
            + float(epsilon)
            + off
        )
        return z, mode
    if mode == "center":
        z = float(table_z) + 0.5 * h + off + float(epsilon)
        return z, mode
    if mode == "bottom":
        z = float(table_z) + off + float(epsilon)
        return z, mode
    if mode == "custom":
        z = float(table_z) + off + float(epsilon)
        return z, mode
    raise ValueError(f"origin_z_mode desconocido: {mode}")


def _effective_yaw_user_rad(yaw_rad: float, yaw_deg: float) -> float:
    if abs(yaw_rad) > 1e-9:
        return yaw_rad
    return math.radians(float(yaw_deg))


def quaternion_from_euler_sxyz(roll: float, pitch: float, yaw: float) -> Tuple[float, float, float, float]:
    """Cuaternión (qx,qy,qz,qw) equivalente a tf_transformations 'sxyz' (roll→pitch→yaw, ejes fijos)."""
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy
    return float(qx), float(qy), float(qz), float(qw)


def main(args: Optional[list] = None) -> None:
    rclpy.init(args=args)
    node = Node("spawn_ycb_object")

    pkg_share = Path(get_package_share_directory("panda_vision"))
    default_config = pkg_share / "config" / "ycb_obb_dataset.yaml"
    default_models = Path.home() / "tfg_robotics_ws" / "src" / "gazebo_ycb" / "models"

    node.declare_parameter("label", "")
    node.declare_parameter("preset", "custom")
    node.declare_parameter("x", 0.56)
    node.declare_parameter("y", 0.0)
    node.declare_parameter("use_z_param", False)
    node.declare_parameter("z", 0.0)
    node.declare_parameter("yaw_rad", 0.0)
    node.declare_parameter("yaw_deg", 0.0)
    node.declare_parameter("roll_rad", 0.0)
    node.declare_parameter("pitch_rad", 0.0)
    node.declare_parameter("upright", True)
    node.declare_parameter("delete_existing", True)
    node.declare_parameter("delete_existing_same_label_all_seeds", True)
    node.declare_parameter("delete_all_runtime_ycb", False)
    node.declare_parameter("random_pose", False)
    node.declare_parameter("random_seed", 0)
    node.declare_parameter("x_min", 0.45)
    node.declare_parameter("x_max", 0.70)
    node.declare_parameter("y_min", -0.20)
    node.declare_parameter("y_max", 0.20)
    node.declare_parameter("yaw_min_deg", -180.0)
    node.declare_parameter("yaw_max_deg", 180.0)
    node.declare_parameter("table_z", 0.26)
    node.declare_parameter("spawn_z_epsilon_m", 0.001)
    node.declare_parameter("spawn_name_prefix", "runtime_ycb")
    node.declare_parameter("model_name", "")
    node.declare_parameter("config_path", str(default_config))
    node.declare_parameter("ycb_models_path", str(default_models))
    node.declare_parameter("world_name", "vision_test_ycb")
    node.declare_parameter("world_pose_ros_topic", "")
    node.declare_parameter("pose_discovery_sec", 1.0)
    node.declare_parameter("spawn_backend", "ros_gz_create_cli")
    node.declare_parameter("delete_backend", "gz_service_cli")
    node.declare_parameter("spawn_timeout_sec", 8.0)
    node.declare_parameter("delete_timeout_sec", 10.0)
    node.declare_parameter("delete_retries", 3)
    node.declare_parameter("verify_after_delete", True)
    node.declare_parameter("origin_z_mode_override", "")
    node.declare_parameter("min_table_edge_margin_m", 0.03)
    node.declare_parameter("random_spawn_safe_region", False)
    node.declare_parameter("texture_unique_runtime_models", True)
    node.declare_parameter("runtime_models_root", str(DEFAULT_RUNTIME_MODELS_ROOT))
    node.declare_parameter("post_delete_settle_sec", 0.35)
    node.declare_parameter("wait_until_deleted_timeout_sec", 5.0)
    declare_spawn_pose_readback_params(node)

    label = str(node.get_parameter("label").value).strip().lower()
    preset = str(node.get_parameter("preset").value).strip().lower()
    x = float(node.get_parameter("x").value)
    y = float(node.get_parameter("y").value)
    use_z_param = bool(node.get_parameter("use_z_param").value)
    z_param = float(node.get_parameter("z").value)
    yaw_rad_p = float(node.get_parameter("yaw_rad").value)
    yaw_deg_p = float(node.get_parameter("yaw_deg").value)
    roll_off = float(node.get_parameter("roll_rad").value)
    pitch_off = float(node.get_parameter("pitch_rad").value)
    upright = bool(node.get_parameter("upright").value)
    delete_existing = bool(node.get_parameter("delete_existing").value)
    delete_same_label_all_seeds = bool(
        node.get_parameter("delete_existing_same_label_all_seeds").value
    )
    delete_all = bool(node.get_parameter("delete_all_runtime_ycb").value)
    random_pose = bool(node.get_parameter("random_pose").value)
    random_seed = int(node.get_parameter("random_seed").value)
    x_min = float(node.get_parameter("x_min").value)
    x_max = float(node.get_parameter("x_max").value)
    y_min = float(node.get_parameter("y_min").value)
    y_max = float(node.get_parameter("y_max").value)
    yaw_min_deg = float(node.get_parameter("yaw_min_deg").value)
    yaw_max_deg = float(node.get_parameter("yaw_max_deg").value)
    table_z = float(node.get_parameter("table_z").value)
    spawn_eps = float(node.get_parameter("spawn_z_epsilon_m").value)
    spawn_prefix = str(node.get_parameter("spawn_name_prefix").value).strip()
    model_name_override = str(node.get_parameter("model_name").value).strip()
    config_path = Path(str(node.get_parameter("config_path").value)).expanduser()
    ycb_models_path = Path(str(node.get_parameter("ycb_models_path").value)).expanduser()
    world_name = str(node.get_parameter("world_name").value).strip()
    world_pose_topic_in = str(node.get_parameter("world_pose_ros_topic").value).strip()
    pose_discovery_sec = float(node.get_parameter("pose_discovery_sec").value)
    spawn_backend = str(node.get_parameter("spawn_backend").value).strip()
    delete_backend = str(node.get_parameter("delete_backend").value).strip()
    spawn_timeout = float(node.get_parameter("spawn_timeout_sec").value)
    delete_timeout = float(node.get_parameter("delete_timeout_sec").value)
    delete_retries = int(node.get_parameter("delete_retries").value)
    verify_after_delete = bool(node.get_parameter("verify_after_delete").value)
    origin_z_mode_override = str(node.get_parameter("origin_z_mode_override").value).strip()
    min_table_margin = float(node.get_parameter("min_table_edge_margin_m").value)
    random_spawn_safe_region = bool(node.get_parameter("random_spawn_safe_region").value)
    texture_unique = bool(node.get_parameter("texture_unique_runtime_models").value)
    runtime_models_root = Path(
        str(node.get_parameter("runtime_models_root").value)
    ).expanduser()
    post_delete_settle = float(node.get_parameter("post_delete_settle_sec").value)
    wait_deleted_timeout = float(
        node.get_parameter("wait_until_deleted_timeout_sec").value
    )
    use_sim_time = bool(node.get_parameter("use_sim_time").value)

    if preset == "center_random_seeded":
        random_pose = True

    yaw_user = 0.0

    if abs(yaw_rad_p) > 1e-9 and abs(yaw_deg_p) > 1e-9:
        node.get_logger().warning(
            f"{LOGP} yaw_rad e yaw_deg ambos distintos de cero; se usa yaw_rad."
        )

    gz_world = gz_world_name_from_param(world_name)
    pose_topic = (
        world_pose_topic_in if world_pose_topic_in else f"/world/{gz_world}/pose/info"
    )

    node.get_logger().info(f"{LOGP} label={label}")
    node.get_logger().info(f"{LOGP} preset={preset}")
    node.get_logger().info(f"{LOGP} use_sim_time={use_sim_time}")

    if not label:
        node.get_logger().fatal(f"{LOGP} ERROR: label vacío")
        node.destroy_node()
        rclpy.shutdown()
        sys.exit(2)

    records = load_spawn_records_from_yaml(config_path)
    missing = ensure_all_labels_present(records)
    if missing:
        node.get_logger().fatal(
            f"{LOGP} ERROR: faltan clases en {config_path}: {missing}"
        )
        node.destroy_node()
        rclpy.shutdown()
        sys.exit(3)

    rec = records.get(label)
    if rec is None:
        node.get_logger().fatal(
            f"{LOGP} ERROR: model for label={label} not found en {config_path}"
        )
        node.destroy_node()
        rclpy.shutdown()
        sys.exit(4)

    source_model_dir = (ycb_models_path / rec.model_name).resolve()
    source_sdf = source_model_dir / "model.sdf"
    if not source_sdf.is_file():
        node.get_logger().fatal(
            f"{LOGP} ERROR: model for label={label} not found (sin model.sdf): {source_sdf}"
        )
        node.destroy_node()
        rclpy.shutdown()
        sys.exit(5)

    if texture_unique:
        try:
            sdf_path, _unique_model_name, runtime_dir = prepare_runtime_spawn_model(
                label,
                source_model_dir,
                runtime_models_root=runtime_models_root,
                logger=node.get_logger(),
            )
            node.get_logger().info(
                f"{LOGP} texture_unique_runtime_models=true runtime_sdf={sdf_path}"
            )
        except Exception as exc:
            node.get_logger().fatal(
                f"{LOGP} texture_unique_runtime_models falló (no se usa DAE corrupto): {exc}"
            )
            node.destroy_node()
            rclpy.shutdown()
            sys.exit(8)
    else:
        sdf_path = source_sdf

    node.get_logger().info(f"{LOGP} model_path={sdf_path}")

    if random_pose:
        rng = random.Random(random_seed)
        table_region = TableSpawnRegion(
            x_min=x_min,
            x_max=x_max,
            y_min=y_min,
            y_max=y_max,
            margin_m=min_table_margin,
            random_spawn_safe_region=random_spawn_safe_region,
        )
        if is_known_spawn_geometry_box_label(label):
            x, y, yaw_user = sample_semantic_box_pose_xyyaw(
                rng,
                label,
                table_region,
                footprint_length_m=float(rec.footprint_length_m),
                footprint_width_m=float(rec.footprint_width_m),
                yaw_min_rad=math.radians(yaw_min_deg),
                yaw_max_rad=math.radians(yaw_max_deg),
                logger=node.get_logger(),
            )
            yaw_user_deg = math.degrees(yaw_user)
        else:
            x = rng.uniform(x_min, x_max)
            y = rng.uniform(y_min, y_max)
            yaw_user_deg = rng.uniform(yaw_min_deg, yaw_max_deg)
            yaw_user = math.radians(yaw_user_deg)
        node.get_logger().info(
            f"{LOGP} random_pose=true seed={random_seed} "
            f"semantic_center_xy=({x:.4f},{y:.4f}) yaw_deg={yaw_user_deg:.3f} "
            f"safe_region={random_spawn_safe_region} margin_m={min_table_margin:.3f}"
        )
    elif preset in PRESETS_XY_YAW_DEG:
        px, py, pydeg = PRESETS_XY_YAW_DEG[preset]
        x, y = px, py
        yaw_user = math.radians(pydeg)
        node.get_logger().info(f"{LOGP} preset pos x={x:.4f} y={y:.4f} yaw_deg={pydeg:.3f}")
    elif preset == "custom":
        yaw_user = _effective_yaw_user_rad(yaw_rad_p, yaw_deg_p)
        node.get_logger().info(f"{LOGP} custom x={x:.4f} y={y:.4f}")
    else:
        node.get_logger().fatal(f"{LOGP} ERROR: preset desconocido: {preset}")
        node.destroy_node()
        rclpy.shutdown()
        sys.exit(6)

    mode_eff = (origin_z_mode_override or rec.origin_z_mode or "runtime_yaml").strip().lower()
    z_val, z_rule = _compute_z(
        rec, table_z, spawn_eps, origin_z_mode_override, use_z_param, z_param
    )

    if upright:
        roll = float(rec.base_roll_rad) + roll_off
        pitch = float(rec.base_pitch_rad) + pitch_off
        yaw = float(rec.base_yaw_rad) + float(yaw_user)
    else:
        roll, pitch, yaw = roll_off, pitch_off, yaw_user

    qx, qy, qz, qw = quaternion_from_euler_sxyz(roll, pitch, yaw)

    if model_name_override:
        entity_name = model_name_override
    elif random_pose:
        entity_name = f"{spawn_prefix}_{label}_seed{random_seed}"
    else:
        entity_name = f"{spawn_prefix}_{label}"

    node.get_logger().info(f"{LOGP} model_name={entity_name}")
    node.get_logger().info(
        f"{LOGP} pose x={x:.4f} y={y:.4f} z={z_val:.4f} "
        f"roll={roll:.4f} pitch={pitch:.4f} yaw={yaw:.4f}"
    )
    node.get_logger().info(f"{LOGP} table_z={table_z:.4f}")
    node.get_logger().info(f"{LOGP} height={rec.height_m:.4f}")
    node.get_logger().info(f"{LOGP} origin_z_mode={mode_eff} (rule={z_rule})")
    node.get_logger().info(
        f"{LOGP} spawn_height_m={rec.spawn_height_m:.4f} spawn_z_offset_m={rec.spawn_z_offset_m:.4f} "
        f"origin_z_offset_m={rec.origin_z_offset_m:.4f}"
    )
    node.get_logger().info(f"{LOGP} upright={upright} spawn_backend={spawn_backend}")

    if delete_all:
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
                f"{LOGP} delete_all_runtime_ycb falló; restantes={remaining}"
            )
            node.destroy_node()
            rclpy.shutdown()
            sys.exit(7)

    if delete_existing and not delete_all:
        discovered = discover_runtime_entity_names(
            node,
            gz_world_name=gz_world,
            pose_topic=pose_topic,
            pose_discovery_sec=pose_discovery_sec,
        )
        if delete_same_label_all_seeds:
            targets = filter_runtime_entity_names(
                discovered, spawn_prefix, label=label
            )
        else:
            targets = [entity_name]
        for name in sorted(set(targets)):
            node.get_logger().info(f"{LOGP} deleting previous entity={name}")
            delete_entity(
                node, gz_world, name, delete_backend, delete_timeout, quiet=True
            )
        if entity_name not in targets:
            node.get_logger().info(
                f"{LOGP} deleting previous entity={entity_name} (objetivo final, idempotente)"
            )
            delete_entity(
                node, gz_world, entity_name, delete_backend, delete_timeout, quiet=True
            )

    spawn_x, spawn_y, spawn_z = float(x), float(y), float(z_val)
    gt_x, gt_y, gt_z = spawn_x, spawn_y, spawn_z
    if is_known_spawn_geometry_box_label(label):
        semantic, gazebo = resolve_semantic_and_gazebo_poses(
            label,
            (float(x), float(y)),
            float(yaw),
            table_z,
            float(rec.height_m),
            epsilon_m=spawn_eps,
            logger=node.get_logger(),
        )
        gt_x, gt_y, gt_z = semantic
        spawn_x, spawn_y, spawn_z = gazebo
        node.get_logger().info(
            f"{LOGP} known_box semantic_center=({gt_x:.4f},{gt_y:.4f},{gt_z:.4f}) "
            f"gazebo_origin=({spawn_x:.4f},{spawn_y:.4f},{spawn_z:.4f})"
        )

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
    node.get_logger().info(f"{LOGP} spawn success={ok}")
    if ok:
        seed_val = int(random_seed) if random_pose else None
        readback_params = readback_params_from_node(node)
        gt_client = RuntimeSceneGtClient(node, world_frame="world")
        gt_entry = settle_and_build_gt_entry(
            node,
            entity_name=entity_name,
            label=label,
            requested_gazebo_xyz=(float(spawn_x), float(spawn_y), float(spawn_z)),
            requested_yaw_rad=float(yaw),
            width_m=float(rec.footprint_width_m),
            length_m=float(rec.footprint_length_m),
            height_m=float(rec.height_m),
            world_pose_topic=pose_topic,
            params=readback_params,
            spawn_seed=seed_val,
            logger=node.get_logger(),
        )
        if gt_entry is None:
            node.get_logger().warning(
                f"{LOGP} GT desde pose Gazebo no disponible; se publica pose comandada"
            )
            gt_entry = make_gt_object_entry(
                entity_name=entity_name,
                label=label,
                x=float(gt_x),
                y=float(gt_y),
                z=float(gt_z),
                roll=float(roll),
                pitch=float(pitch),
                yaw=float(yaw),
                qx=float(qx),
                qy=float(qy),
                qz=float(qz),
                qw=float(qw),
                width_m=float(rec.footprint_width_m),
                length_m=float(rec.footprint_length_m),
                height_m=float(rec.height_m),
                spawn_seed=seed_val,
                logger=node.get_logger(),
            )
        gt_client.replace_all([gt_entry])
    else:
        RuntimeSceneGtClient(node, world_frame="world").clear()
    node.destroy_node()
    rclpy.shutdown()
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
