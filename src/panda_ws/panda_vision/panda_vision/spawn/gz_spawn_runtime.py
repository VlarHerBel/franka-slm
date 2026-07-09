"""Utilidades mínimas para spawn/borrado YCB en Gazebo Sim (ros_gz), sin grasping."""

from __future__ import annotations

import math
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import Iterable, List, Optional, Set, Tuple

import rclpy
from rclpy.node import Node
from ros_gz_interfaces.msg import Entity as GzEntity
from ros_gz_interfaces.srv import DeleteEntity as GzDeleteEntity
from ros_gz_interfaces.srv import SpawnEntity as GzSpawnEntity
from tf2_msgs.msg import TFMessage


def gz_world_name_from_param(world_name: str) -> str:
    w = world_name.strip()
    return w if w.endswith("_world") else f"{w}_world"


def gazebo_pose_info_ros_topic(
    world_name: str, *, dynamic: bool = False
) -> str:
    """Topic ROS puenteado desde SceneBroadcaster (Gazebo Sim)."""
    gz_world = gz_world_name_from_param(world_name)
    leaf = "dynamic_pose/info" if dynamic else "pose/info"
    return f"/world/{gz_world}/{leaf}"


def _gz_binary() -> Optional[str]:
    return "ign" if shutil.which("ign") else ("gz" if shutil.which("gz") else None)


def discover_model_short_names(
    node: Node,
    pose_topic: str,
    collect_seconds: float,
) -> Set[str]:
    """Lee nombres de modelos desde el topic puenteado de SceneBroadcaster (TFMessage)."""
    names: Set[str] = set()

    def _cb(msg: TFMessage) -> None:
        for ts in msg.transforms:
            child = (ts.child_frame_id or "").strip()
            if not child:
                continue
            short = child.split("::")[-1]
            names.add(short)
            names.add(child)

    sub = node.create_subscription(TFMessage, pose_topic, _cb, 10)
    t_end = time.monotonic() + max(0.05, collect_seconds)
    try:
        while time.monotonic() < t_end and rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.05)
    finally:
        node.destroy_subscription(sub)
    return names


def filter_runtime_entity_names(
    all_names: Iterable[str],
    spawn_name_prefix: str,
    label: Optional[str] = None,
) -> List[str]:
    """Filtra nombres de entidades runtime YCB.

    Si ``label`` es None: todo nombre que empiece por ``spawn_name_prefix``
    (p. ej. ``runtime_ycb``, ``runtime_ycb_cracker_box``, ``runtime_ycb_cracker_box_seed0``).
    Si ``label`` está definido: ``{prefix}_{label}`` o ``{prefix}_{label}_*``.
    """
    p = spawn_name_prefix.strip()
    if not p:
        return []
    out: List[str] = []
    seen: Set[str] = set()
    for raw in all_names:
        name = raw.split("::")[-1].strip()
        if not name.startswith(p):
            continue
        if label is not None:
            lb = label.strip().lower()
            exact = f"{p}_{lb}"
            scoped = f"{p}_{lb}_"
            if not (name == exact or name.startswith(scoped)):
                continue
        if name not in seen:
            seen.add(name)
            out.append(name)
    return sorted(out)


def discover_model_names_gz_topic_cli(
    node: Node,
    pose_topic: str,
    duration_sec: float,
) -> Set[str]:
    """Fallback: un echo corto del topic de poses (protobuf texto) vía CLI gz/ign."""
    binary = _gz_binary()
    if binary is None:
        return set()
    duration = max(0.1, float(duration_sec))
    cmd = [binary, "topic", "-e", pose_topic, "-d", f"{duration:.3f}", "-u"]
    names: Set[str] = set()
    try:
        completed = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=duration + 8.0,
            check=False,
        )
        text = (completed.stdout or "") + "\n" + (completed.stderr or "")
        for match in re.finditer(
            r"child_frame_id:\s*\"?([^\"\s]+)\"?", text, flags=re.IGNORECASE
        ):
            short = match.group(1).split("::")[-1].strip()
            if short:
                names.add(short)
    except Exception as exc:
        node.get_logger().warning(
            "[CLEAR_YCB] gz topic echo falló topic=%s: %s" % (pose_topic, exc)
        )
    return names


def discover_model_names_gz_scene_info(
    node: Node,
    gz_world_name: str,
    timeout_sec: float,
) -> Set[str]:
    """Fallback: lista modelos del mundo vía ``/world/<world>/scene/info``."""
    binary = _gz_binary()
    if binary is None:
        return set()
    reqtype = "ignition.msgs.Empty" if binary == "ign" else "gz.msgs.Empty"
    reptype = "ignition.msgs.Scene" if binary == "ign" else "gz.msgs.Scene"
    cmd = [
        binary,
        "service",
        "-s",
        f"/world/{gz_world_name}/scene/info",
        "--reqtype",
        reqtype,
        "--reptype",
        reptype,
        "--timeout",
        str(int(max(500, timeout_sec * 1000))),
        "--req",
        "",
    ]
    names: Set[str] = set()
    try:
        completed = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=max(2.0, timeout_sec + 2.0),
            check=False,
        )
        text = (completed.stdout or "") + "\n" + (completed.stderr or "")
        for match in re.finditer(r'name:\s*"([^"]+)"', text):
            short = match.group(1).split("::")[-1].strip()
            if short:
                names.add(short)
        for match in re.finditer(r"name:\s*'([^']+)'", text):
            short = match.group(1).split("::")[-1].strip()
            if short:
                names.add(short)
    except Exception as exc:
        node.get_logger().warning(
            "[CLEAR_YCB] scene/info falló world=%s: %s" % (gz_world_name, exc)
        )
    return names


def discover_runtime_entity_names(
    node: Node,
    *,
    gz_world_name: str,
    pose_topic: str,
    pose_discovery_sec: float,
) -> Set[str]:
    """Unión de descubrimiento ROS (topic) + CLI topic echo + scene/info."""
    names: Set[str] = set()
    try:
        names |= discover_model_short_names(node, pose_topic, pose_discovery_sec)
    except Exception as exc:
        node.get_logger().warning(
            "[CLEAR_YCB] discover ROS topic=%s falló: %s" % (pose_topic, exc)
        )
    cli_topic_sec = min(max(0.25, pose_discovery_sec), 2.0)
    names |= discover_model_names_gz_topic_cli(node, pose_topic, cli_topic_sec)
    names |= discover_model_names_gz_scene_info(
        node, gz_world_name, timeout_sec=max(2.0, pose_discovery_sec)
    )
    return names


def wait_until_no_runtime_ycb_entities(
    node: Node,
    *,
    gz_world_name: str,
    pose_topic: str,
    pose_discovery_sec: float,
    spawn_name_prefix: str,
    timeout_sec: float = 5.0,
    poll_interval_sec: float = 0.25,
    label: Optional[str] = None,
    log_prefix: str = "[CLEAR_YCB]",
) -> Tuple[bool, List[str]]:
    """Espera hasta que ``pose/info`` ya no liste entidades runtime_ycb."""
    logger = node.get_logger()
    deadline = time.monotonic() + max(0.1, float(timeout_sec))
    poll = max(0.05, float(poll_interval_sec))
    last: List[str] = []
    while time.monotonic() < deadline:
        discovered = discover_runtime_entity_names(
            node,
            gz_world_name=gz_world_name,
            pose_topic=pose_topic,
            pose_discovery_sec=max(0.15, float(pose_discovery_sec)),
        )
        last = filter_runtime_entity_names(
            discovered, spawn_name_prefix, label=label
        )
        if not last:
            logger.info(
                "%s wait_until_deleted: no %s entities (%.2fs)"
                % (log_prefix, spawn_name_prefix, timeout_sec)
            )
            return True, []
        time.sleep(poll)
    logger.warning(
        "%s wait_until_deleted timeout: still present %s"
        % (log_prefix, last)
    )
    return False, last


def clear_runtime_ycb_entities(
    node: Node,
    *,
    gz_world_name: str,
    pose_topic: str,
    pose_discovery_sec: float,
    spawn_name_prefix: str,
    delete_backend: str,
    delete_timeout_sec: float,
    delete_retries: int,
    verify_after_delete: bool,
    list_only: bool,
    label: Optional[str] = None,
    log_prefix: str = "[CLEAR_YCB]",
    post_delete_settle_sec: float = 0.35,
    wait_until_deleted_timeout_sec: float = 5.0,
) -> Tuple[bool, List[str]]:
    """Borra entidades runtime YCB; devuelve (éxito, restantes)."""
    logger = node.get_logger()
    retries = max(1, int(delete_retries))
    remaining: List[str] = []

    for pass_num in range(1, retries + 1):
        discovered = discover_runtime_entity_names(
            node,
            gz_world_name=gz_world_name,
            pose_topic=pose_topic,
            pose_discovery_sec=pose_discovery_sec,
        )
        targets = filter_runtime_entity_names(
            discovered, spawn_name_prefix, label=label
        )
        logger.info("%s discovered entities=%s" % (log_prefix, targets))

        if list_only:
            return True, targets

        if not targets:
            if pass_num == 1:
                logger.info(
                    "%s success: no %s entities remain"
                    % (log_prefix, spawn_name_prefix)
                )
            return True, []

        for name in targets:
            logger.info("%s deleting entity=%s" % (log_prefix, name))
            ok = delete_entity(
                node,
                gz_world_name,
                name,
                delete_backend,
                delete_timeout_sec,
                quiet=False,
            )
            logger.info(
                "%s delete result entity=%s ok=%s"
                % (log_prefix, name, str(ok).lower())
            )

        if not verify_after_delete:
            return True, []

        settle = max(0.0, float(post_delete_settle_sec))
        if settle > 0.0:
            time.sleep(settle)
        ok_wait, remaining = wait_until_no_runtime_ycb_entities(
            node,
            gz_world_name=gz_world_name,
            pose_topic=pose_topic,
            pose_discovery_sec=pose_discovery_sec,
            spawn_name_prefix=spawn_name_prefix,
            timeout_sec=wait_until_deleted_timeout_sec,
            poll_interval_sec=0.25,
            label=label,
            log_prefix=log_prefix,
        )
        if ok_wait:
            logger.info(
                "%s success: no %s entities remain"
                % (log_prefix, spawn_name_prefix)
            )
            return True, []

        logger.warning(
            "%s remaining after pass %d: %s"
            % (log_prefix, pass_num, remaining)
        )
        if pass_num < retries:
            logger.info(
                "%s retry %d/%d" % (log_prefix, pass_num + 1, retries)
            )

    logger.error(
        "%s ERROR: entities still present after retries: %s"
        % (log_prefix, remaining)
    )
    return False, remaining


def _call_service(node: Node, client, request, timeout_sec: float):
    future = client.call_async(request)
    end = time.monotonic() + timeout_sec
    while time.monotonic() < end and rclpy.ok():
        if future.done():
            return future.result()
        time.sleep(0.05)
    raise TimeoutError("Timeout esperando servicio Gazebo.")


def _wait_for_service(node: Node, client, service_name: str, timeout_sec: float) -> None:
    node.get_logger().info(f"Esperando servicio Gazebo: {service_name}")
    if client.wait_for_service(timeout_sec=timeout_sec):
        node.get_logger().info(f"Servicio disponible: {service_name}")
        return
    raise TimeoutError(f"Servicio Gazebo no disponible: {service_name}")


def delete_entity_service(
    node: Node,
    gz_world_name: str,
    entity_name: str,
    delete_timeout_sec: float,
    quiet: bool,
) -> bool:
    client = node.create_client(GzDeleteEntity, f"/world/{gz_world_name}/remove")
    try:
        _wait_for_service(
            node, client, f"/world/{gz_world_name}/remove", delete_timeout_sec
        )
        req = GzDeleteEntity.Request()
        req.entity.name = entity_name
        req.entity.type = GzEntity.MODEL
        resp = _call_service(node, client, req, delete_timeout_sec)
        ok = bool(resp and resp.success)
        if not ok and not quiet:
            node.get_logger().warning(
                f"[SPAWN_YCB] delete service no confirmó éxito para entity={entity_name}"
            )
        return ok
    except Exception as exc:
        if not quiet:
            node.get_logger().warning(
                f"[SPAWN_YCB] delete service falló entity={entity_name}: {exc}"
            )
        return False
    finally:
        node.destroy_client(client)


def delete_entity_cli(
    node: Node,
    gz_world_name: str,
    entity_name: str,
    quiet: bool,
) -> bool:
    binary = "ign" if shutil.which("ign") else ("gz" if shutil.which("gz") else None)
    if binary is None:
        if not quiet:
            node.get_logger().error("No se encontró ni 'ign' ni 'gz' para borrado CLI.")
        return False
    reqtype = "ignition.msgs.Entity" if binary == "ign" else "gz.msgs.Entity"
    reptype = "ignition.msgs.Boolean" if binary == "ign" else "gz.msgs.Boolean"
    cmd = [
        binary,
        "service",
        "-s",
        f"/world/{gz_world_name}/remove",
        "--reqtype",
        reqtype,
        "--reptype",
        reptype,
        "--timeout",
        "5000",
        "--req",
        f'name: "{entity_name}" type: 2',
    ]
    try:
        completed = subprocess.run(
            cmd, capture_output=True, text=True, timeout=8.0, check=False
        )
        lower_stdout = (completed.stdout or "").lower()
        ok = completed.returncode == 0 and (
            "data: true" in lower_stdout or "true" in lower_stdout
        )
        if not ok and not quiet:
            node.get_logger().warning(
                f"[SPAWN_YCB] delete CLI entity={entity_name} rc={completed.returncode} "
                f"stdout={(completed.stdout or '').strip()} stderr={(completed.stderr or '').strip()}"
            )
        return ok
    except Exception as exc:
        if not quiet:
            node.get_logger().warning(
                f"[SPAWN_YCB] delete CLI excepción entity={entity_name}: {exc}"
            )
        return False


def delete_entity(
    node: Node,
    gz_world_name: str,
    entity_name: str,
    delete_backend: str,
    delete_timeout_sec: float,
    quiet: bool,
) -> bool:
    if delete_backend == "gz_service_cli":
        return delete_entity_cli(node, gz_world_name, entity_name, quiet=quiet)
    if delete_backend != "service":
        raise ValueError(
            "delete_backend inválido. Valores válidos: service, gz_service_cli"
        )
    return delete_entity_service(
        node, gz_world_name, entity_name, delete_timeout_sec, quiet=quiet
    )


def spawn_entity_service(
    node: Node,
    gz_world_name: str,
    entity_name: str,
    sdf_path: Path,
    x: float,
    y: float,
    z: float,
    qx: float,
    qy: float,
    qz: float,
    qw: float,
    spawn_timeout_sec: float,
) -> bool:
    client = node.create_client(GzSpawnEntity, f"/world/{gz_world_name}/create")
    try:
        _wait_for_service(
            node, client, f"/world/{gz_world_name}/create", spawn_timeout_sec
        )
        req = GzSpawnEntity.Request()
        req.entity_factory.name = entity_name
        req.entity_factory.allow_renaming = False
        req.entity_factory.sdf_filename = str(sdf_path)
        req.entity_factory.pose.position.x = float(x)
        req.entity_factory.pose.position.y = float(y)
        req.entity_factory.pose.position.z = float(z)
        req.entity_factory.pose.orientation.x = float(qx)
        req.entity_factory.pose.orientation.y = float(qy)
        req.entity_factory.pose.orientation.z = float(qz)
        req.entity_factory.pose.orientation.w = float(qw)
        req.entity_factory.relative_to = "world"
        resp = _call_service(node, client, req, spawn_timeout_sec)
        return bool(resp and resp.success)
    finally:
        node.destroy_client(client)


def spawn_entity_cli(
    node: Node,
    gz_world_name: str,
    entity_name: str,
    sdf_path: Path,
    x: float,
    y: float,
    z: float,
    yaw_rad: float,
) -> bool:
    cmd = [
        "ros2",
        "run",
        "ros_gz_sim",
        "create",
        "-world",
        gz_world_name,
        "-file",
        str(sdf_path),
        "-name",
        entity_name,
        "-x",
        f"{x:.6f}",
        "-y",
        f"{y:.6f}",
        "-z",
        f"{z:.6f}",
        "-Y",
        f"{yaw_rad:.6f}",
    ]
    node.get_logger().info(f"[SPAWN_YCB] spawn CLI: {' '.join(cmd)}")
    completed = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        node.get_logger().error(
            f"[SPAWN_YCB] spawn CLI falló rc={completed.returncode} "
            f"stdout={(completed.stdout or '').strip()} stderr={(completed.stderr or '').strip()}"
        )
        return False
    return True


def spawn_entity(
    node: Node,
    gz_world_name: str,
    entity_name: str,
    sdf_path: Path,
    x: float,
    y: float,
    z: float,
    qx: float,
    qy: float,
    qz: float,
    qw: float,
    spawn_backend: str,
    spawn_timeout_sec: float,
) -> bool:
    if spawn_backend == "ros_gz_create_cli":
        roll, pitch, yaw = _quat_to_rpy(qx, qy, qz, qw)
        if abs(roll) > 1e-4 or abs(pitch) > 1e-4:
            node.get_logger().warning(
                "[SPAWN_YCB] ros_gz_sim create solo aplica yaw (-Y); "
                f"roll={roll:.4f} pitch={pitch:.4f} se ignoran en CLI."
            )
        return spawn_entity_cli(node, gz_world_name, entity_name, sdf_path, x, y, z, yaw)
    if spawn_backend != "service":
        raise ValueError(
            "spawn_backend inválido. Valores válidos: service, ros_gz_create_cli"
        )
    return spawn_entity_service(
        node,
        gz_world_name,
        entity_name,
        sdf_path,
        x,
        y,
        z,
        qx,
        qy,
        qz,
        qw,
        spawn_timeout_sec,
    )


def _quat_to_rpy(qx: float, qy: float, qz: float, qw: float) -> Tuple[float, float, float]:
    sinr_cosp = 2.0 * (qw * qx + qy * qz)
    cosr_cosp = 1.0 - 2.0 * (qx * qx + qy * qy)
    roll = math.atan2(sinr_cosp, cosr_cosp)
    sinp = 2.0 * (qw * qy - qz * qx)
    pitch = math.copysign(math.pi / 2.0, sinp) if abs(sinp) >= 1.0 else math.asin(sinp)
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return float(roll), float(pitch), float(yaw)
