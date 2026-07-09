#!/usr/bin/env python3
"""Generate a synthetic YCB OBB dataset from Gazebo Sim."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from dataclasses import dataclass, replace
from pathlib import Path
import random
import shutil
import subprocess
import time
from typing import Dict, List, Literal, Optional, Set, Tuple

from ament_index_python.packages import get_package_share_directory
import cv2
from cv_bridge import CvBridge
import numpy as np

if not hasattr(np, "float"):
    setattr(np, "float", float)

import rclpy
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import QoSProfile, ReliabilityPolicy
from rclpy.time import Time
from ros_gz_interfaces.msg import Entity as GzEntity
from ros_gz_interfaces.srv import DeleteEntity as GzDeleteEntity
from ros_gz_interfaces.srv import SetEntityPose as GzSetEntityPose
from ros_gz_interfaces.srv import SpawnEntity as GzSpawnEntity
from sensor_msgs.msg import CameraInfo, Image
from tf2_msgs.msg import TFMessage
import tf2_ros
import tf_transformations
import yaml

from panda_vision.geometry.camera_projection import scaled_intrinsics_from_camera_info


def _make_autoseed() -> int:
    return (time.time_ns() ^ (os.getpid() << 16)) & 0xFFFFFFFFFFFFFFFF


def _seed_for_scene(base_seed: int, scene_idx: int) -> int:
    digest = hashlib.blake2b(
        f"{base_seed}:{scene_idx}".encode("utf-8"), digest_size=8
    ).digest()
    return int.from_bytes(digest, "little", signed=False)


def _seed_for_scene_attempt(base_seed: int, scene_idx: int, attempt_idx: int) -> int:
    digest = hashlib.blake2b(
        f"{base_seed}:{scene_idx}:{attempt_idx}".encode("utf-8"), digest_size=8
    ).digest()
    return int.from_bytes(digest, "little", signed=False)


def _stamp_to_ns(stamp) -> int:
    return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)


def _pose_signature(
    poses: List[Optional[Tuple[float, float, float, float, float, float]]]
) -> str:
    normalized: List[Optional[List[float]]] = []
    for pose6 in poses:
        if pose6 is None:
            normalized.append(None)
            continue
        normalized.append([round(float(v), 5) for v in pose6])
    return json.dumps(normalized, sort_keys=True, separators=(",", ":"))


def normalize_obb_ultralytics(pts_px: np.ndarray) -> Optional[np.ndarray]:
    pts = np.array(pts_px, dtype=np.float64).reshape(4, 2)
    if not np.all(np.isfinite(pts)):
        return None
    cx, cy = pts[:, 0].mean(), pts[:, 1].mean()
    ang = np.arctan2(pts[:, 1] - cy, pts[:, 0] - cx)
    pts = pts[np.argsort(ang)]

    idx0 = np.lexsort((pts[:, 0], pts[:, 1]))[0]
    pts = np.roll(pts, -idx0, axis=0)

    signed_area = 0.5 * np.sum(
        pts[:, 0] * np.roll(pts[:, 1], -1)
        - np.roll(pts[:, 0], -1) * pts[:, 1]
    )
    if signed_area > 0.0:
        pts = pts[::-1]
        idx0 = np.lexsort((pts[:, 0], pts[:, 1]))[0]
        pts = np.roll(pts, -idx0, axis=0)
    return pts


@dataclass(frozen=True)
class YCBClassConfig:
    name: str
    class_id: int
    model_name: str
    width_m: float
    length_m: float
    height_m: float
    spawn_height_m: float
    # Suma a la Z de contacto con la mesa (modelos YCB: origen del link suele estar en la base)
    spawn_z_offset_m: float
    # Offset del visual/collision respecto al origen del link (para proyectar OBB alineadas).
    visual_offset_x_m: float = 0.0
    visual_offset_y_m: float = 0.0
    visual_offset_z_m: float = 0.0
    visual_yaw_offset_rad: float = 0.0
    # Huella configurable (si no está, se usa width/length).
    footprint_width_m: float = 0.0
    footprint_length_m: float = 0.0
    # false para cilindros / latas: evita pitch~90° que ruedan y desplazan la pose respecto a la etiqueta
    allow_lying: bool = True


@dataclass(frozen=True)
class SpawnedObject:
    entity_name: str
    class_cfg: YCBClassConfig
    x_m: float
    y_m: float
    z_m: float
    yaw_rad: float
    roll_rad: float
    pitch_rad: float
    lying_on_side: bool


@dataclass(frozen=True)
class ProjectedLabel:
    spawned: SpawnedObject
    polygon: np.ndarray
    line: str


@dataclass(frozen=True)
class DomainRandomizationConfig:
    table_texture_dir: str
    background_texture_dir: str
    light_intensity_range: Tuple[float, float]
    light_direction_jitter_deg: float
    camera_jitter_xyz_mm: float
    camera_jitter_rpy_deg: float
    num_distractors: int


@dataclass(frozen=True)
class PoseObservation:
    pose6: Optional[Tuple[float, float, float, float, float, float]]
    source: str
    fresh: bool
    from_cache: bool
    stamp_ns: int


class YCBDatasetGenerator(Node):
    """Generate RGB images and Ultralytics OBB labels from Gazebo Sim."""

    def __init__(self) -> None:
        super().__init__(
            "generate_ycb_dataset",
            parameter_overrides=[
                Parameter("use_sim_time", Parameter.Type.BOOL, True)
            ],
        )

        pkg_share = Path(get_package_share_directory("panda_vision"))
        default_config = pkg_share / "config" / "ycb_obb_dataset.yaml"
        default_output = Path.home() / "tfg_robotics_ws" / "datasets" / "ycb_obb_v2"
        default_ycb_models = Path.home() / "tfg_robotics_ws" / "src" / "gazebo_ycb" / "models"

        self.declare_parameter("scene_count", 100)
        # Primer índice de escena en esta ejecución (p. ej. 245 para continuar sin pisar scene_00244).
        self.declare_parameter("start_scene_index", 0)
        self.declare_parameter("output_dir", str(default_output))
        self.declare_parameter("config_path", str(default_config))
        self.declare_parameter("ycb_models_path", str(default_ycb_models))
        # Todos los YCB usan texture_map.png → Ogre/Ignition la cachea por nombre y “contagia” la 1ª textura.
        # Si true, se genera una copia por modelo con PNG renombrado y model.sdf con file://
        self.declare_parameter("texture_unique_cache", True)
        self.declare_parameter("texture_cache_dir", "")
        # Mundo con mesa vacía (sin primitivas de color): vision_test_ycb.world
        self.declare_parameter("world_name", "vision_test_ycb")
        # Superficie superior del tablero en vision_test*.world: z_centro_tabla + grosor/2
        # (table_top pose z=0.24, caja grosor 0.04 -> superficie en 0.26)
        self.declare_parameter("table_surface_z_m", 0.26)
        # Margen vertical mínimo para evitar penetración numérica. Antes era 2 mm,
        # pero combinado con otros lifts dejaba los objetos visiblemente flotando.
        self.declare_parameter("spawn_z_epsilon_m", 0.0005)
        self.declare_parameter("image_topic", "/camera/image_raw")
        self.declare_parameter("camera_info_topic", "/camera/camera_info")
        self.declare_parameter("camera_optical_frame", "camera_link_optical")
        # Marco explícito donde se planifican mesa/objetos y se reciben poses del mundo.
        self.declare_parameter("world_frame", "world")
        # Compatibilidad hacia atrás: antes se asumía panda_link0 como marco de referencia.
        self.declare_parameter("reference_frame", "panda_link0")
        self.declare_parameter("min_objects", 2)
        self.declare_parameter("max_objects", 5)
        self.declare_parameter("val_split", 0.2)
        self.declare_parameter("seed", -1)
        self.declare_parameter("settle_time", 1.5)
        self.declare_parameter("spawn_x_min", 0.42)
        self.declare_parameter("spawn_x_max", 0.72)
        self.declare_parameter("spawn_y_min", -0.22)
        self.declare_parameter("spawn_y_max", 0.22)
        # Distancia mínima entre centros (suelo); la separación real exige además r_i+r_j+gap.
        self.declare_parameter("min_center_distance_m", 0.09)
        # Hueco mínimo entre siluetas aproximadas (semidiámetro XY de cada clase).
        self.declare_parameter("min_surface_gap_m", 0.055)
        # Si hay ≤ N clases en YAML y pides ≤ N objetos, no repite clase en la misma escena.
        self.declare_parameter("diverse_classes_per_scene", True)
        # Suma a la Z de contacto para todos los modelos (penetración ODE / mallas bajo el link).
        # Lift global adicional. Para dataset queremos contacto visual con la mesa,
        # así que por defecto lo dejamos en 0 y solo lo subimos si un modelo concreto
        # necesita corrección.
        self.declare_parameter("global_spawn_z_lift_m", 0.0)
        # Tras settle_time, descartar varios frames para que la sim y el sensor converjan.
        self.declare_parameter("post_settle_camera_frames", 3)
        # Tablero vision_test_ycb: centro (0.6, 0), mitades 0.36 x 0.24 m (caja 0.72 x 0.48).
        self.declare_parameter("table_center_x_m", 0.60)
        self.declare_parameter("table_center_y_m", 0.0)
        self.declare_parameter("table_half_extent_x_m", 0.36)
        self.declare_parameter("table_half_extent_y_m", 0.24)
        self.declare_parameter("table_edge_margin_m", 0.028)
        # Margen extra al colocar tumbados (huella real y rodadura cerca del borde).
        self.declare_parameter("lying_table_edge_extra_margin_m", 0.038)
        # Probabilidad de spawn con pitch ~90° (tumbado); huella = dos dimensiones menores del AABB.
        self.declare_parameter("lying_down_probability", 0.22)
        self.declare_parameter("lying_pitch_jitter_rad", 0.12)
        # Tras alinear la AABB rotada con la mesa, pequeño hueco extra solo en tumbado (no altura de caída).
        self.declare_parameter("lying_z_boost_m", 0.0)
        # Si true, roll aleatorio al tumbado (más variedad visual, OBB peor alineado con la malla).
        self.declare_parameter("lying_random_roll", False)
        # Reintentos si un sorteo de escena no cabe en la mesa (aleatorio + separación).
        self.declare_parameter("scene_layout_max_attempts", 220)
        self.declare_parameter("max_position_samples", 8000)
        self.declare_parameter("camera_timeout_sec", 10.0)
        self.declare_parameter("spawn_timeout_sec", 8.0)
        # ign service remove puede bloquearse bajo carga; el timeout de subprocess debe ser mayor que el de ign.
        self.declare_parameter("delete_timeout_sec", 60.0)
        # Zona de "parking" para el pool persistente de modelos YCB. Debe quedar
        # muy lejos de la mesa y fuera del frustum de la cámara para no contaminar
        # el dataset ni confundir al inspeccionar Gazebo.
        self.declare_parameter("parking_origin_x_m", -6.0)
        self.declare_parameter("parking_origin_y_m", -4.0)
        self.declare_parameter("parking_row_step_m", 0.45)
        self.declare_parameter("parking_col_step_m", 0.20)
        self.declare_parameter("min_fresh_frames", 2)
        self.declare_parameter("require_fresh_image", True)
        self.declare_parameter("reject_if_same_image", True)
        self.declare_parameter("max_projected_overlap_ratio", 0.02)
        self.declare_parameter("clear_output", True)
        # Borra los max_objects slots al inicio (útil si quedaron entidades de un run abortado).
        # En sim limpia puede generar mensajes "not found" en consola de Gazebo (inofensivos).
        self.declare_parameter("initial_full_purge", True)
        # Etiquetas desde pose real en sim tras settle; evita desajuste física vs pose planificada.
        self.declare_parameter("labels_use_sim_pose", True)
        self.declare_parameter("labels_use_gz_model_pose", True)
        # ros_tf: tf2_msgs/TFMessage vía ros_gz_bridge (/world/<gz_world>/pose/info). gz_cli: subprocess gz model -p.
        self.declare_parameter("labels_sim_pose_source", "ros_tf")
        # Vacío → /world/<world_name>_world/pose/info (mismo patrón que gazebo.launch.py + SDF).
        self.declare_parameter("world_pose_ros_topic", "")
        self.declare_parameter("gz_model_pose_timeout_sec", 5.0)
        self.declare_parameter("gz_pose_scene_max_retries", 6)
        # Si tras reintentos no hay poses gz, usar pose planificada (menos fiable) en lugar de omitir la escena.
        self.declare_parameter("allow_planned_pose_labels_fallback", False)
        # Objetos que caen bajo la mesa pueden seguir proyectando OBB sobre el tablero en imagen;
        # se descarta la escena si el centro (gz) sale de la banda Z/XY esperada sobre la mesa.
        self.declare_parameter("reject_scene_if_sim_pose_off_table", True)
        self.declare_parameter("label_pose_z_below_table_m", 0.06)
        self.declare_parameter("label_pose_z_above_table_m", 0.52)
        self.declare_parameter("label_pose_xy_slack_m", 0.07)
        # Solo para filtrar “¿sigue sobre el tablero?”; más estricto que label_pose_xy_slack_m.
        self.declare_parameter("label_pose_tabletop_xy_slack_m", 0.022)
        # Profundidad media de la huella en cámara vs punto de la superficie de mesa (rechaza caídas al suelo).
        self.declare_parameter("label_pose_depth_consistency_check", True)
        self.declare_parameter("label_pose_max_mean_depth_delta_from_table_m", 0.34)
        # No guardar PNG/TXT si faltan etiquetas por proyección inválida (nº líneas < nº objetos).
        self.declare_parameter("reject_scene_if_incomplete_labels", True)
        self.declare_parameter("table_texture_dir", "")
        self.declare_parameter("background_texture_dir", "")
        self.declare_parameter("light_intensity_range", [1.0, 1.0])
        self.declare_parameter("light_direction_jitter_deg", 0.0)
        self.declare_parameter("camera_jitter_xyz_mm", 0.0)
        self.declare_parameter("camera_jitter_rpy_deg", 0.0)
        self.declare_parameter("num_distractors", 0)
        # Estabilización visual: exige N frames consecutivos estables antes de guardar PNG.
        self.declare_parameter("visual_stability_frames", 3)
        self.declare_parameter("visual_stability_max_wait_sec", 6.0)
        self.declare_parameter("visual_stability_mean_diff_threshold", 0.35)
        # Validación de render real contra fondo de mesa (esperado == visible).
        self.declare_parameter("validate_rendered_visibility", True)
        self.declare_parameter("visibility_pixel_diff_threshold", 8)
        self.declare_parameter("visibility_mean_diff_threshold", 2.0)
        self.declare_parameter("visibility_min_changed_fraction", 0.025)
        self.declare_parameter("visibility_min_polygon_area_px", 120.0)
        # Solo aplica si entity_motion_mode:=set_pose (experimental).
        self.declare_parameter("set_pose_verify_xy_tol_m", 0.03)
        self.declare_parameter("set_pose_verify_z_tol_m", 0.04)
        self.declare_parameter("set_pose_verify_max_retries", 2)
        self.declare_parameter("set_pose_fallback_respawn", True)
        # "respawn": delete+spawn por escena (estable, por defecto).
        # "set_pose": servicio set_pose + verificación gz_cli (experimental, lento si gz_cli falla).
        self.declare_parameter("entity_motion_mode", "respawn")
        # Modo de calibración por clase: 1 escena, 1 objeto, artefactos debug.
        self.declare_parameter("calibration_mode", False)
        self.declare_parameter("calibration_class_name", "")
        self.declare_parameter("calibration_allow_lying", False)
        self.declare_parameter("calibration_output_dir", "")

        self._scene_count = int(self.get_parameter("scene_count").value)
        self._start_scene_index = int(self.get_parameter("start_scene_index").value)
        self._output_dir = Path(
            str(self.get_parameter("output_dir").value)
        ).expanduser()
        self._config_path = Path(
            str(self.get_parameter("config_path").value)
        ).expanduser()
        self._ycb_models_path = Path(
            str(self.get_parameter("ycb_models_path").value)
        ).expanduser()
        self._texture_unique_cache = bool(
            self.get_parameter("texture_unique_cache").value
        )
        _tex_cd = str(self.get_parameter("texture_cache_dir").value).strip()
        self._texture_cache_root = (
            Path(_tex_cd).expanduser()
            if _tex_cd
            else Path.home() / ".cache" / "panda_vision" / "ycb_texture_unique"
        )
        self._world_name = str(self.get_parameter("world_name").value)
        self._gz_world_name = self._world_name
        if not self._gz_world_name.endswith("_world"):
            self._gz_world_name = f"{self._gz_world_name}_world"
        self._table_surface_z_m = float(self.get_parameter("table_surface_z_m").value)
        self._spawn_z_epsilon_m = float(self.get_parameter("spawn_z_epsilon_m").value)
        self._image_topic = str(self.get_parameter("image_topic").value)
        self._camera_info_topic = str(self.get_parameter("camera_info_topic").value)
        self._camera_optical_frame = str(
            self.get_parameter("camera_optical_frame").value
        )
        self._legacy_reference_frame = str(
            self.get_parameter("reference_frame").value
        ).strip()
        self._world_frame = str(self.get_parameter("world_frame").value).strip()
        if not self._world_frame:
            self._world_frame = self._legacy_reference_frame
        if (
            self._legacy_reference_frame
            and self._legacy_reference_frame != self._world_frame
        ):
            self.get_logger().warning(
                "reference_frame y world_frame difieren. Se usará world_frame "
                f"para planificación/proyección ({self._world_frame})."
            )
        self._min_objects = int(self.get_parameter("min_objects").value)
        self._max_objects = int(self.get_parameter("max_objects").value)
        self._val_split = float(self.get_parameter("val_split").value)
        self._settle_time = float(self.get_parameter("settle_time").value)
        self._camera_timeout_sec = float(self.get_parameter("camera_timeout_sec").value)
        self._spawn_timeout_sec = float(self.get_parameter("spawn_timeout_sec").value)
        self._delete_timeout_sec = float(self.get_parameter("delete_timeout_sec").value)
        self._parking_origin_x_m = float(
            self.get_parameter("parking_origin_x_m").value
        )
        self._parking_origin_y_m = float(
            self.get_parameter("parking_origin_y_m").value
        )
        self._parking_row_step_m = float(
            self.get_parameter("parking_row_step_m").value
        )
        self._parking_col_step_m = float(
            self.get_parameter("parking_col_step_m").value
        )
        self._min_fresh_frames = max(
            1, int(self.get_parameter("min_fresh_frames").value)
        )
        self._require_fresh_image = bool(
            self.get_parameter("require_fresh_image").value
        )
        self._reject_if_same_image = bool(
            self.get_parameter("reject_if_same_image").value
        )
        self._max_projected_overlap_ratio = float(
            self.get_parameter("max_projected_overlap_ratio").value
        )
        self._clear_output = bool(self.get_parameter("clear_output").value)
        if self._start_scene_index > 0 and self._clear_output:
            self.get_logger().warning(
                "start_scene_index>0: desactivando clear_output para no borrar "
                "imágenes/etiquetas ya guardadas en output_dir."
            )
            self._clear_output = False
        self._initial_full_purge = bool(self.get_parameter("initial_full_purge").value)
        legacy_labels_use_sim_pose = bool(
            self.get_parameter("labels_use_gz_model_pose").value
        )
        self._labels_use_sim_pose = bool(
            self.get_parameter("labels_use_sim_pose").value
        )
        if not legacy_labels_use_sim_pose and self._labels_use_sim_pose:
            self.get_logger().warning(
                "Usando labels_use_gz_model_pose:=false por compatibilidad. "
                "El parámetro recomendado es labels_use_sim_pose."
            )
            self._labels_use_sim_pose = False
        self._labels_sim_pose_source = str(
            self.get_parameter("labels_sim_pose_source").value
        ).strip().lower()
        if self._labels_sim_pose_source == "sim_interfaces":
            self.get_logger().warning(
                "labels_sim_pose_source:=sim_interfaces no está disponible en este entorno; "
                "se usará ros_tf para obtener la pose real post-settle."
            )
            self._labels_sim_pose_source = "ros_tf"
        _wprt = str(self.get_parameter("world_pose_ros_topic").value).strip()
        self._world_pose_ros_topic = (
            _wprt
            if _wprt
            else f"/world/{self._gz_world_name}/pose/info"
        )
        self._gz_model_pose_timeout_sec = float(
            self.get_parameter("gz_model_pose_timeout_sec").value
        )
        self._gz_pose_scene_max_retries = max(
            0, int(self.get_parameter("gz_pose_scene_max_retries").value)
        )
        self._allow_planned_pose_labels_fallback = bool(
            self.get_parameter("allow_planned_pose_labels_fallback").value
        )
        self._reject_scene_if_sim_pose_off_table = bool(
            self.get_parameter("reject_scene_if_sim_pose_off_table").value
        )
        self._label_pose_z_below_table_m = float(
            self.get_parameter("label_pose_z_below_table_m").value
        )
        self._label_pose_z_above_table_m = float(
            self.get_parameter("label_pose_z_above_table_m").value
        )
        self._label_pose_xy_slack_m = float(
            self.get_parameter("label_pose_xy_slack_m").value
        )
        self._label_pose_tabletop_xy_slack_m = float(
            self.get_parameter("label_pose_tabletop_xy_slack_m").value
        )
        self._label_pose_depth_consistency_check = bool(
            self.get_parameter("label_pose_depth_consistency_check").value
        )
        self._label_pose_max_mean_depth_delta_from_table_m = float(
            self.get_parameter("label_pose_max_mean_depth_delta_from_table_m").value
        )
        self._reject_scene_if_incomplete_labels = bool(
            self.get_parameter("reject_scene_if_incomplete_labels").value
        )
        light_intensity_range = self.get_parameter("light_intensity_range").value
        light_min = float(light_intensity_range[0]) if light_intensity_range else 1.0
        light_max = (
            float(light_intensity_range[1])
            if len(light_intensity_range) > 1
            else light_min
        )
        self._domain_randomization = DomainRandomizationConfig(
            table_texture_dir=str(self.get_parameter("table_texture_dir").value).strip(),
            background_texture_dir=str(
                self.get_parameter("background_texture_dir").value
            ).strip(),
            light_intensity_range=(light_min, light_max),
            light_direction_jitter_deg=float(
                self.get_parameter("light_direction_jitter_deg").value
            ),
            camera_jitter_xyz_mm=float(
                self.get_parameter("camera_jitter_xyz_mm").value
            ),
            camera_jitter_rpy_deg=float(
                self.get_parameter("camera_jitter_rpy_deg").value
            ),
            num_distractors=max(0, int(self.get_parameter("num_distractors").value)),
        )
        self._visual_stability_frames = max(
            2, int(self.get_parameter("visual_stability_frames").value)
        )
        self._visual_stability_max_wait_sec = float(
            self.get_parameter("visual_stability_max_wait_sec").value
        )
        self._visual_stability_mean_diff_threshold = float(
            self.get_parameter("visual_stability_mean_diff_threshold").value
        )
        self._validate_rendered_visibility = bool(
            self.get_parameter("validate_rendered_visibility").value
        )
        self._visibility_pixel_diff_threshold = int(
            self.get_parameter("visibility_pixel_diff_threshold").value
        )
        self._visibility_mean_diff_threshold = float(
            self.get_parameter("visibility_mean_diff_threshold").value
        )
        self._visibility_min_changed_fraction = float(
            self.get_parameter("visibility_min_changed_fraction").value
        )
        self._visibility_min_polygon_area_px = float(
            self.get_parameter("visibility_min_polygon_area_px").value
        )
        self._set_pose_verify_xy_tol_m = float(
            self.get_parameter("set_pose_verify_xy_tol_m").value
        )
        self._set_pose_verify_z_tol_m = float(
            self.get_parameter("set_pose_verify_z_tol_m").value
        )
        self._set_pose_verify_max_retries = max(
            0, int(self.get_parameter("set_pose_verify_max_retries").value)
        )
        self._set_pose_fallback_respawn = bool(
            self.get_parameter("set_pose_fallback_respawn").value
        )
        self._entity_motion_mode = str(
            self.get_parameter("entity_motion_mode").value
        ).strip().lower()
        self._calibration_mode = bool(self.get_parameter("calibration_mode").value)
        self._calibration_class_name = str(
            self.get_parameter("calibration_class_name").value
        ).strip()
        self._calibration_allow_lying = bool(
            self.get_parameter("calibration_allow_lying").value
        )
        _cal_out = str(self.get_parameter("calibration_output_dir").value).strip()
        self._calibration_output_dir = (
            Path(_cal_out).expanduser()
            if _cal_out
            else self._output_dir / "calibration"
        )

        self._spawn_x_min = float(self.get_parameter("spawn_x_min").value)
        self._spawn_x_max = float(self.get_parameter("spawn_x_max").value)
        self._spawn_y_min = float(self.get_parameter("spawn_y_min").value)
        self._spawn_y_max = float(self.get_parameter("spawn_y_max").value)
        self._min_center_distance_m = float(
            self.get_parameter("min_center_distance_m").value
        )
        self._min_surface_gap_m = float(self.get_parameter("min_surface_gap_m").value)
        self._diverse_classes_per_scene = bool(
            self.get_parameter("diverse_classes_per_scene").value
        )
        self._global_spawn_z_lift_m = float(
            self.get_parameter("global_spawn_z_lift_m").value
        )
        self._post_settle_camera_frames = max(
            0, int(self.get_parameter("post_settle_camera_frames").value)
        )
        self._table_center_x_m = float(self.get_parameter("table_center_x_m").value)
        self._table_center_y_m = float(self.get_parameter("table_center_y_m").value)
        self._table_half_extent_x_m = float(
            self.get_parameter("table_half_extent_x_m").value
        )
        self._table_half_extent_y_m = float(
            self.get_parameter("table_half_extent_y_m").value
        )
        self._table_edge_margin_m = float(
            self.get_parameter("table_edge_margin_m").value
        )
        self._lying_table_edge_extra_margin_m = float(
            self.get_parameter("lying_table_edge_extra_margin_m").value
        )
        self._lying_down_probability = float(
            self.get_parameter("lying_down_probability").value
        )
        self._lying_pitch_jitter_rad = float(
            self.get_parameter("lying_pitch_jitter_rad").value
        )
        self._lying_z_boost_m = float(self.get_parameter("lying_z_boost_m").value)
        self._lying_random_roll = bool(
            self.get_parameter("lying_random_roll").value
        )
        self._scene_layout_max_attempts = max(
            1, int(self.get_parameter("scene_layout_max_attempts").value)
        )
        self._max_position_samples = max(
            100, int(self.get_parameter("max_position_samples").value)
        )

        self._seed_param = int(self.get_parameter("seed").value)
        self._run_meta_path = self._output_dir / "run_meta.json"
        self._base_seed = self._resolve_base_seed()
        self._bridge = CvBridge()
        self._camera_info: Optional[CameraInfo] = None
        self._last_image: Optional[Image] = None
        self._image_seq = 0
        self._last_image_stamp_ns = 0
        self._world_pose_by_model: Dict[str, Tuple[float, float, float, float, float, float]] = {}
        self._world_pose_stamp_by_model: Dict[str, int] = {}
        self._world_pose_seq = 0
        self._world_pose_msg_stamp_ns = 0
        self._world_pose_rx_stamp_ns = 0
        self._world_pose_stamp_ns = 0
        self._previous_saved_image_hash: Optional[str] = None
        self._previous_saved_pose_signature: Optional[str] = None
        self._background_reference_bgr: Optional[np.ndarray] = None

        self._validate_parameters()
        self._classes = self._load_classes(self._config_path)
        self._class_by_name = {cfg.name: cfg for cfg in self._classes}
        if self._calibration_mode and self._calibration_class_name:
            if self._calibration_class_name not in self._class_by_name:
                raise ValueError(
                    "calibration_class_name no existe en classes habilitadas: "
                    f"{self._calibration_class_name}"
                )

        self._tf_buffer = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, self)

        self.get_logger().info(
            "Dataset generator arrancado con use_sim_time=true para alinear "
            "scene timestamps, /clock e imágenes de Gazebo."
        )

        qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, depth=10)
        self.create_subscription(Image, self._image_topic, self._image_cb, qos)
        self.create_subscription(CameraInfo, self._camera_info_topic, self._cam_info_cb, qos)
        if self._labels_sim_pose_source == "ros_tf":
            self.create_subscription(
                TFMessage,
                self._world_pose_ros_topic,
                self._world_pose_tf_cb,
                qos,
            )
        self._spawn_entity_client = self.create_client(
            GzSpawnEntity, f"/world/{self._gz_world_name}/create"
        )
        self._delete_entity_client = self.create_client(
            GzDeleteEntity, f"/world/{self._gz_world_name}/remove"
        )
        self._set_pose_client = self.create_client(
            GzSetEntityPose, f"/world/{self._gz_world_name}/set_pose"
        )
        self._pool_entities_by_class = {
            class_cfg.name: [
                f"ycb_dataset_{class_cfg.model_name}_{replica_idx:02d}"
                for replica_idx in range(self._max_objects)
            ]
            for class_cfg in self._classes
        }
        self._all_pool_entity_names = [
            entity_name
            for entity_names in self._pool_entities_by_class.values()
            for entity_name in entity_names
        ]
        self._active_entity_names_previous: List[str] = []
        self._scene_activation_verify_notes: List[str] = []
        self._run_stats = {
            "scene_rejections": 0,
            "labels_incomplete_rejections": 0,
            "sim_pose_rejections": 0,
            "missing_pose_rejections": 0,
            "stale_image_rejections": 0,
            "duplicate_image_pose_mismatch_rejections": 0,
            "projected_overlap_rejections": 0,
            "render_visibility_rejections": 0,
            "set_pose_recovery_count": 0,
            "activation_verify_uncertain": 0,
            "saved_train": 0,
            "saved_val": 0,
        }

    def _validate_parameters(self) -> None:
        if self._scene_count <= 0:
            raise ValueError("scene_count must be > 0")
        if self._start_scene_index < 0:
            raise ValueError("start_scene_index must be >= 0")
        if self._min_objects <= 0:
            raise ValueError("min_objects must be > 0")
        if self._min_objects > self._max_objects:
            raise ValueError("min_objects must be <= max_objects")
        if not 0.0 <= self._val_split <= 1.0:
            raise ValueError("val_split must be in [0, 1]")
        if self._spawn_x_min >= self._spawn_x_max:
            raise ValueError("spawn_x_min must be < spawn_x_max")
        if not 0.0 <= self._max_projected_overlap_ratio < 1.0:
            raise ValueError("max_projected_overlap_ratio must be in [0, 1)")
        if self._spawn_y_min >= self._spawn_y_max:
            raise ValueError("spawn_y_min must be < spawn_y_max")
        if self._table_half_extent_x_m <= 0.0 or self._table_half_extent_y_m <= 0.0:
            raise ValueError("table half extents must be > 0")
        if not 0.0 <= self._lying_down_probability <= 1.0:
            raise ValueError("lying_down_probability must be in [0, 1]")
        if not self._config_path.is_file():
            raise FileNotFoundError(f"Config not found: {self._config_path}")
        if not self._ycb_models_path.is_dir():
            raise FileNotFoundError(
                f"YCB models directory not found: {self._ycb_models_path}"
            )
        if self._label_pose_z_below_table_m < 0.0 or self._label_pose_z_above_table_m < 0.0:
            raise ValueError("label_pose_z_below_table_m and label_pose_z_above_table_m must be >= 0")
        if self._label_pose_xy_slack_m < 0.0:
            raise ValueError("label_pose_xy_slack_m must be >= 0")
        if self._label_pose_tabletop_xy_slack_m < 0.0:
            raise ValueError("label_pose_tabletop_xy_slack_m must be >= 0")
        if (
            self._label_pose_depth_consistency_check
            and self._label_pose_max_mean_depth_delta_from_table_m <= 0.0
        ):
            raise ValueError(
                "label_pose_max_mean_depth_delta_from_table_m must be > 0 "
                "when label_pose_depth_consistency_check is true"
            )
        if self._lying_table_edge_extra_margin_m < 0.0:
            raise ValueError("lying_table_edge_extra_margin_m must be >= 0")
        if self._labels_sim_pose_source not in ("ros_tf", "gz_cli"):
            raise ValueError(
                "labels_sim_pose_source must be 'ros_tf' or 'gz_cli'"
            )
        if self._domain_randomization.light_intensity_range[0] <= 0.0:
            raise ValueError("light_intensity_range min must be > 0")
        if (
            self._domain_randomization.light_intensity_range[1]
            < self._domain_randomization.light_intensity_range[0]
        ):
            raise ValueError("light_intensity_range max must be >= min")
        if self._domain_randomization.num_distractors < 0:
            raise ValueError("num_distractors must be >= 0")
        if self._parking_row_step_m <= 0.0 or self._parking_col_step_m <= 0.0:
            raise ValueError("parking_row_step_m and parking_col_step_m must be > 0")
        if self._min_fresh_frames <= 0:
            raise ValueError("min_fresh_frames must be >= 1")
        if not self._world_frame:
            raise ValueError("world_frame must not be empty")
        if self._visual_stability_frames < 2:
            raise ValueError("visual_stability_frames must be >= 2")
        if self._visual_stability_max_wait_sec <= 0.0:
            raise ValueError("visual_stability_max_wait_sec must be > 0")
        if self._visual_stability_mean_diff_threshold < 0.0:
            raise ValueError("visual_stability_mean_diff_threshold must be >= 0")
        if self._visibility_pixel_diff_threshold < 0:
            raise ValueError("visibility_pixel_diff_threshold must be >= 0")
        if self._visibility_mean_diff_threshold < 0.0:
            raise ValueError("visibility_mean_diff_threshold must be >= 0")
        if not 0.0 <= self._visibility_min_changed_fraction <= 1.0:
            raise ValueError("visibility_min_changed_fraction must be in [0, 1]")
        if self._visibility_min_polygon_area_px < 0.0:
            raise ValueError("visibility_min_polygon_area_px must be >= 0")
        if self._set_pose_verify_xy_tol_m < 0.0 or self._set_pose_verify_z_tol_m < 0.0:
            raise ValueError("set_pose verify tolerances must be >= 0")
        if self._entity_motion_mode not in ("set_pose", "respawn"):
            raise ValueError("entity_motion_mode must be 'set_pose' or 'respawn'")

    def _read_existing_run_meta(self) -> Dict[str, object]:
        if not self._run_meta_path.is_file():
            return {}
        try:
            return json.loads(self._run_meta_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}

    def _resolve_base_seed(self) -> int:
        existing = self._read_existing_run_meta()
        if (
            self._seed_param < 0
            and self._start_scene_index > 0
            and not self._clear_output
            and isinstance(existing.get("base_seed"), int)
        ):
            base_seed = int(existing["base_seed"])
            self.get_logger().info(
                f"Reutilizando base_seed={base_seed} desde {self._run_meta_path} "
                "para continuar el dataset sin repetir prefijos."
            )
            return base_seed
        return _make_autoseed() if self._seed_param < 0 else self._seed_param

    def _write_run_meta(self) -> None:
        self._output_dir.mkdir(parents=True, exist_ok=True)
        existing = self._read_existing_run_meta()
        payload = {
            **existing,
            "base_seed": int(self._base_seed),
            "seed_param": int(self._seed_param),
            "start_scene_index": int(self._start_scene_index),
            "scene_count": int(self._scene_count),
            "world_name": self._world_name,
            "world_frame": self._world_frame,
            "labels_use_sim_pose": bool(self._labels_use_sim_pose),
            "labels_sim_pose_source": self._labels_sim_pose_source,
            "visual_stability": {
                "frames": self._visual_stability_frames,
                "max_wait_sec": self._visual_stability_max_wait_sec,
                "mean_diff_threshold": self._visual_stability_mean_diff_threshold,
            },
            "entity_motion_mode": self._entity_motion_mode,
            "calibration_mode": bool(self._calibration_mode),
            "calibration_class_name": self._calibration_class_name,
            "domain_randomization": {
                "table_texture_dir": self._domain_randomization.table_texture_dir,
                "background_texture_dir": self._domain_randomization.background_texture_dir,
                "light_intensity_range": list(
                    self._domain_randomization.light_intensity_range
                ),
                "light_direction_jitter_deg": self._domain_randomization.light_direction_jitter_deg,
                "camera_jitter_xyz_mm": self._domain_randomization.camera_jitter_xyz_mm,
                "camera_jitter_rpy_deg": self._domain_randomization.camera_jitter_rpy_deg,
                "num_distractors": self._domain_randomization.num_distractors,
            },
            "stats": self._run_stats,
        }
        self._run_meta_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    def _load_classes(self, config_path: Path) -> List[YCBClassConfig]:
        with config_path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}

        classes = []
        enabled_raw_classes = [
            raw for raw in data.get("classes", [])
            if bool(raw.get("enabled", True))
        ]
        for remapped_class_id, raw in enumerate(enabled_raw_classes):
            classes.append(
                YCBClassConfig(
                    name=str(raw["name"]),
                    class_id=remapped_class_id,
                    model_name=str(raw.get("model_name", raw["name"])),
                    width_m=float(raw["width_m"]),
                    length_m=float(raw["length_m"]),
                    height_m=float(raw.get("height_m", raw["spawn_height_m"] * 2.0)),
                    spawn_height_m=float(raw.get("spawn_height_m", 0.0)),
                    spawn_z_offset_m=float(raw.get("spawn_z_offset_m", 0.0)),
                    visual_offset_x_m=float(raw.get("visual_offset_x_m", 0.0)),
                    visual_offset_y_m=float(raw.get("visual_offset_y_m", 0.0)),
                    visual_offset_z_m=float(raw.get("visual_offset_z_m", 0.0)),
                    visual_yaw_offset_rad=float(raw.get("visual_yaw_offset_rad", 0.0)),
                    footprint_width_m=float(raw.get("footprint_width_m", 0.0)),
                    footprint_length_m=float(raw.get("footprint_length_m", 0.0)),
                    allow_lying=bool(raw.get("allow_lying", True)),
                )
            )

        if not classes:
            raise ValueError(f"No classes defined in {config_path}")
        return sorted(classes, key=lambda item: item.class_id)

    def _image_cb(self, msg: Image) -> None:
        self._last_image = msg
        self._image_seq += 1
        self._last_image_stamp_ns = _stamp_to_ns(msg.header.stamp)

    def _cam_info_cb(self, msg: CameraInfo) -> None:
        self._camera_info = msg

    @staticmethod
    def _normalize_frame_id(frame_id: str) -> str:
        return frame_id.strip().lstrip("/")

    def _frame_matches_world(self, frame_id: str) -> bool:
        if not frame_id:
            return False
        norm = self._normalize_frame_id(frame_id)
        world_norm = self._normalize_frame_id(self._world_frame)
        return norm == world_norm or norm.endswith(f"/{world_norm}")

    def _world_pose_tf_cb(self, msg: TFMessage) -> None:
        """Actualiza mapa nombre_modelo → (x,y,z,roll,pitch,yaw) desde SceneBroadcaster."""
        new_map: Dict[str, Tuple[float, float, float, float, float, float]] = {}
        new_stamp_map: Dict[str, int] = {}
        rx_stamp_ns = self.get_clock().now().nanoseconds
        latest_msg_stamp_ns = 0
        dataset_entities_seen: Set[str] = set()
        for ts in msg.transforms:
            parent = (ts.header.frame_id or "").strip()
            if parent and not self._frame_matches_world(parent):
                continue
            child = (ts.child_frame_id or "").strip()
            if not child:
                continue
            q = ts.transform.rotation
            t = ts.transform.translation
            try:
                roll, pitch, yaw = tf_transformations.euler_from_quaternion(
                    (q.x, q.y, q.z, q.w), axes="sxyz"
                )
            except ValueError:
                continue
            msg_stamp_ns = _stamp_to_ns(ts.header.stamp)
            latest_msg_stamp_ns = max(latest_msg_stamp_ns, msg_stamp_ns)
            pose6 = (t.x, t.y, t.z, float(roll), float(pitch), float(yaw))
            short = child.split("::")[-1]
            effective_stamp_ns = msg_stamp_ns if msg_stamp_ns > 0 else rx_stamp_ns
            new_map[short] = pose6
            new_map[child] = pose6
            new_stamp_map[short] = effective_stamp_ns
            new_stamp_map[child] = effective_stamp_ns
            if short.startswith("ycb_dataset_"):
                dataset_entities_seen.add(short)
        self._world_pose_by_model = new_map
        self._world_pose_stamp_by_model = new_stamp_map
        self._world_pose_seq += 1
        latest_stamp_ns = latest_msg_stamp_ns
        if latest_stamp_ns == 0:
            latest_stamp_ns = rx_stamp_ns
        self._world_pose_msg_stamp_ns = latest_msg_stamp_ns
        self._world_pose_rx_stamp_ns = rx_stamp_ns
        self._world_pose_stamp_ns = latest_stamp_ns
        self.get_logger().debug(
            "world_pose_tf_cb: "
            f"msg_stamp_ns={self._world_pose_msg_stamp_ns} "
            f"rx_stamp_ns={self._world_pose_rx_stamp_ns} "
            f"world_pose_stamp_ns={self._world_pose_stamp_ns} "
            f"dataset_entities={len(dataset_entities_seen)}"
        )

    def _spin_until(self, predicate, timeout_sec: float) -> bool:
        deadline = time.time() + timeout_sec
        while time.time() < deadline:
            rclpy.spin_once(self, timeout_sec=0.1)
            if predicate():
                return True
        return predicate()

    def _spin_for_seconds(self, seconds: float) -> None:
        deadline = time.time() + max(0.0, seconds)
        while time.time() < deadline:
            rclpy.spin_once(self, timeout_sec=0.05)

    def _wait_for_camera_ready(self) -> None:
        if self._spin_until(
            lambda: self._camera_info is not None and self._last_image is not None,
            self._camera_timeout_sec,
        ):
            return
        raise TimeoutError(
            "Timed out waiting for /camera/image_raw and /camera/camera_info"
        )

    def _wait_for_fresh_image(self, previous_seq: int) -> Image:
        ready = self._spin_until(
            lambda: self._last_image is not None and self._image_seq > previous_seq,
            self._camera_timeout_sec,
        )
        if not ready or self._last_image is None:
            last_stamp = self._last_image_stamp_ns if self._last_image is not None else -1
            raise TimeoutError(
                "Timed out waiting for a fresh RGB frame after spawning / moving objects. "
                f"last_image_seq={self._image_seq}, previous_seq={previous_seq}, "
                f"last_image_stamp_ns={last_stamp}. "
                "Comprueba que Gazebo no este en pausa y que /camera/image_raw siga publicando."
            )
        return self._last_image

    def _image_to_bgr(self, image_msg: Image) -> np.ndarray:
        return self._bridge.imgmsg_to_cv2(image_msg, desired_encoding="bgr8")

    @staticmethod
    def _mean_abs_diff_bgr(frame_a: np.ndarray, frame_b: np.ndarray) -> float:
        if frame_a.shape != frame_b.shape:
            return float("inf")
        diff = cv2.absdiff(frame_a, frame_b)
        return float(np.mean(diff))

    def _wait_for_visually_stable_scene_image(
        self, min_stamp_ns: int, previous_seq: int
    ) -> Tuple[Image, np.ndarray, int, int]:
        """Espera frames frescos y devuelve el primero con estabilidad visual consecutiva."""
        deadline = time.time() + self._visual_stability_max_wait_sec
        current_seq = previous_seq
        discarded_frames = 0
        accepted_frames = 0
        stable_streak = 0
        min_accepted_frames = max(self._min_fresh_frames, self._visual_stability_frames)
        prev_bgr: Optional[np.ndarray] = None
        prev_hash: Optional[str] = None
        last_image: Optional[Image] = None
        last_bgr: Optional[np.ndarray] = None

        while time.time() < deadline:
            image_msg = self._wait_for_fresh_image(current_seq)
            current_seq = self._image_seq
            image_stamp_ns = _stamp_to_ns(image_msg.header.stamp)
            if self._require_fresh_image and image_stamp_ns <= min_stamp_ns:
                discarded_frames += 1
                continue
            bgr = self._image_to_bgr(image_msg)
            frame_hash = self._image_md5(bgr)
            if prev_bgr is None:
                stable_streak = 1
            else:
                mean_diff = self._mean_abs_diff_bgr(bgr, prev_bgr)
                if (
                    frame_hash == prev_hash
                    or mean_diff <= self._visual_stability_mean_diff_threshold
                ):
                    stable_streak += 1
                else:
                    stable_streak = 1
            prev_hash = frame_hash
            prev_bgr = bgr
            accepted_frames += 1
            last_image = image_msg
            last_bgr = bgr
            if (
                stable_streak >= self._visual_stability_frames
                and accepted_frames >= min_accepted_frames
            ):
                return image_msg, bgr, discarded_frames, accepted_frames

        if last_image is None or last_bgr is None:
            raise TimeoutError(
                "No se encontró ningún frame fresco para estabilización visual. "
                f"min_stamp_ns={min_stamp_ns}, last_image_seq={self._image_seq}, "
                f"last_image_stamp_ns={self._last_image_stamp_ns}"
            )
        raise TimeoutError(
            "Timeout esperando estabilidad visual. "
            f"stable_streak={stable_streak}, required={self._visual_stability_frames}, "
            f"accepted_frames={accepted_frames}, discarded_frames={discarded_frames}, "
            f"last_image_stamp_ns={_stamp_to_ns(last_image.header.stamp)}"
        )

    def _wait_for_world_pose_update(self, previous_seq: int) -> bool:
        """Al menos un TFMessage nuevo tras la imagen (misma ventana temporal aprox.)."""
        timeout = min(2.0, max(0.35, self._camera_timeout_sec * 0.25))
        return self._spin_until(lambda: self._world_pose_seq > previous_seq, timeout)

    def _wait_for_world_pose_after(self, min_stamp_ns: int) -> None:
        if not (self._labels_use_sim_pose and self._labels_sim_pose_source == "ros_tf"):
            return
        timeout = min(2.5, max(0.5, self._camera_timeout_sec * 0.35))
        self._spin_until(lambda: self._world_pose_stamp_ns > min_stamp_ns, timeout)

    def _wait_for_world_pose_snapshot_after(
        self,
        scene: List[SpawnedObject],
        min_stamp_ns: int,
        min_world_pose_seq: int = 0,
    ) -> Tuple[Dict[str, Tuple[float, float, float, float, float, float]], int]:
        """Espera un snapshot TF coherente para esta escena y devuelve una copia inmutable."""
        if self._labels_sim_pose_source != "ros_tf":
            return dict(self._world_pose_by_model), self._world_pose_stamp_ns

        expected_names = [spawned.entity_name for spawned in scene]
        timeout = min(3.0, max(0.8, self._camera_timeout_sec * 0.4))
        deadline = time.time() + timeout
        snapshot_map: Dict[str, Tuple[float, float, float, float, float, float]] = {}
        snapshot_stamp_ns = 0

        while time.time() < deadline:
            rclpy.spin_once(self, timeout_sec=0.1)
            if self._world_pose_seq <= min_world_pose_seq:
                continue
            snapshot_stamp_ns = self._world_pose_stamp_ns
            if snapshot_stamp_ns <= min_stamp_ns:
                continue
            snapshot_map = dict(self._world_pose_by_model)
            if all(self._lookup_pose_in_snapshot(snapshot_map, name) is not None for name in expected_names):
                return snapshot_map, snapshot_stamp_ns

        return dict(self._world_pose_by_model), self._world_pose_stamp_ns

    def _wait_for_world_pose_bridge_ready(self) -> None:
        if self._labels_sim_pose_source != "ros_tf":
            return
        deadline = time.time() + self._camera_timeout_sec
        while time.time() < deadline:
            rclpy.spin_once(self, timeout_sec=0.1)
            if self._world_pose_seq > 0 and self._world_pose_by_model:
                return
        raise TimeoutError(
            f"No se recibió ningún mensaje en {self._world_pose_ros_topic} "
            "(tf2_msgs/TFMessage). Arranca ros_gz_bridge con el puente "
            "/world/<gz_world>/pose/info o usa labels_sim_pose_source:=gz_cli."
        )

    @staticmethod
    def _lookup_pose_with_stamp_in_maps(
        pose_map: Dict[str, Tuple[float, float, float, float, float, float]],
        stamp_map: Dict[str, int],
        entity_name: str,
    ) -> Tuple[Optional[Tuple[float, float, float, float, float, float]], int]:
        if entity_name in pose_map:
            return pose_map[entity_name], int(stamp_map.get(entity_name, 0))
        link_key = f"{entity_name}::link"
        if link_key in pose_map:
            return pose_map[link_key], int(stamp_map.get(link_key, 0))
        for key, pose6 in pose_map.items():
            if key.endswith(f"::{entity_name}") or key.startswith(f"{entity_name}::"):
                return pose6, int(stamp_map.get(key, 0))
        return None, 0

    def _model_pose_rpy_with_stamp_from_ros(
        self, entity_name: str
    ) -> Tuple[Optional[Tuple[float, float, float, float, float, float]], int]:
        return self._lookup_pose_with_stamp_in_maps(
            self._world_pose_by_model,
            self._world_pose_stamp_by_model,
            entity_name,
        )

    def _model_pose_rpy_from_ros(
        self, entity_name: str
    ) -> Optional[Tuple[float, float, float, float, float, float]]:
        pose6, _stamp_ns = self._model_pose_rpy_with_stamp_from_ros(entity_name)
        return pose6

    @staticmethod
    def _lookup_pose_in_snapshot(
        snapshot_map: Dict[str, Tuple[float, float, float, float, float, float]],
        entity_name: str,
    ) -> Optional[Tuple[float, float, float, float, float, float]]:
        if entity_name in snapshot_map:
            return snapshot_map[entity_name]
        link_key = f"{entity_name}::link"
        if link_key in snapshot_map:
            return snapshot_map[link_key]
        for key, pose6 in snapshot_map.items():
            if key.endswith(f"::{entity_name}"):
                return pose6
            if key.startswith(f"{entity_name}::"):
                return pose6
        return None

    def _sim_poses_for_scene(
        self, scene: List[SpawnedObject]
    ) -> List[Optional[Tuple[float, float, float, float, float, float]]]:
        if self._labels_sim_pose_source == "ros_tf":
            prev_wp = self._world_pose_seq
            self._wait_for_world_pose_update(prev_wp)
            return [self._model_pose_rpy_from_ros(s.entity_name) for s in scene]
        return [self._gz_model_pose_rpy(s.entity_name) for s in scene]

    def _sim_poses_for_scene_after(
        self, scene: List[SpawnedObject], min_stamp_ns: int
    ) -> List[Optional[Tuple[float, float, float, float, float, float]]]:
        if self._labels_sim_pose_source == "ros_tf":
            self._wait_for_world_pose_after(min_stamp_ns)
            return [self._model_pose_rpy_from_ros(s.entity_name) for s in scene]
        return [self._gz_model_pose_rpy(s.entity_name) for s in scene]

    def _sim_poses_from_snapshot(
        self,
        scene: List[SpawnedObject],
        snapshot_map: Dict[str, Tuple[float, float, float, float, float, float]],
    ) -> List[Optional[Tuple[float, float, float, float, float, float]]]:
        return [
            self._lookup_pose_in_snapshot(snapshot_map, spawned.entity_name)
            for spawned in scene
        ]

    @staticmethod
    def _image_md5(bgr: np.ndarray) -> str:
        return hashlib.md5(np.ascontiguousarray(bgr).tobytes()).hexdigest()

    def _wait_for_service(self, client, service_name: str, timeout_sec: float) -> None:
        if client.wait_for_service(timeout_sec=timeout_sec):
            return
        raise TimeoutError(f"Servicio ROS2 no disponible: {service_name}")

    def _call_service(self, client, request, service_name: str, timeout_sec: float):
        future = client.call_async(request)
        deadline = time.time() + timeout_sec
        while time.time() < deadline:
            rclpy.spin_once(self, timeout_sec=0.1)
            if future.done():
                response = future.result()
                if response is None:
                    raise RuntimeError(f"Respuesta vacía en servicio {service_name}")
                return response
        raise TimeoutError(f"Timeout esperando respuesta de {service_name}")

    def _wait_for_world_services_ready(self) -> None:
        self._wait_for_service(
            self._spawn_entity_client,
            f"/world/{self._gz_world_name}/create",
            self._spawn_timeout_sec,
        )
        self._wait_for_service(
            self._delete_entity_client,
            f"/world/{self._gz_world_name}/remove",
            self._delete_timeout_sec,
        )
        self._wait_for_service(
            self._set_pose_client,
            f"/world/{self._gz_world_name}/set_pose",
            self._spawn_timeout_sec,
        )

    def _apply_domain_randomization_hooks(self, scene_idx: int) -> None:
        cfg = self._domain_randomization
        if (
            not cfg.table_texture_dir
            and not cfg.background_texture_dir
            and cfg.light_intensity_range == (1.0, 1.0)
            and cfg.light_direction_jitter_deg == 0.0
            and cfg.camera_jitter_xyz_mm == 0.0
            and cfg.camera_jitter_rpy_deg == 0.0
            and cfg.num_distractors == 0
        ):
            return
        self.get_logger().debug(
            f"scene_{scene_idx:05d}: hooks de domain randomization preparados "
            "(requieren mundo/plugins para aplicar cambios visuales en runtime)."
        )

    def _command_env(self) -> Dict[str, str]:
        env = os.environ.copy()
        resource_paths = [
            path for path in env.get("GZ_SIM_RESOURCE_PATH", "").split(os.pathsep) if path
        ]
        ycb_path = str(self._ycb_models_path)
        if ycb_path not in resource_paths:
            resource_paths.append(ycb_path)
            env["GZ_SIM_RESOURCE_PATH"] = os.pathsep.join(resource_paths)
        return env

    def _delete_entity(self, entity_name: str) -> None:
        request = GzDeleteEntity.Request()
        request.entity.name = entity_name
        request.entity.type = GzEntity.MODEL
        response = self._call_service(
            self._delete_entity_client,
            request,
            f"/world/{self._gz_world_name}/remove",
            self._delete_timeout_sec,
        )
        if not response.success:
            self.get_logger().warning(
                f"No se pudo borrar '{entity_name}' del mundo {self._gz_world_name}."
            )

    def _purge_all_dataset_slots(self) -> None:
        """Elimina todas las entidades del pool si quedaron restos de otro run."""
        for entity_name in self._all_pool_entity_names:
            self._delete_entity(entity_name)
            time.sleep(0.01)
        time.sleep(0.10)

    def _clear_previous_scene_slots(self) -> None:
        """Vacía la escena anterior usando la estrategia de movimiento activa."""
        if not self._active_entity_names_previous:
            return
        for entity_name in self._active_entity_names_previous:
            self.verify_parking_entity_best_effort(
                self._park_pose_for_entity(entity_name)
            )
            time.sleep(0.002)
        self._active_entity_names_previous = []
        time.sleep(0.02)

    def _find_texture_map_png(self, model_root: Path) -> Optional[Path]:
        for rel in (
            Path("meshes") / "texture_map.png",
            Path("materials") / "textures" / "texture_map.png",
        ):
            candidate = model_root / rel
            if candidate.is_file():
                return candidate.resolve()
        return None

    def _model_source_fingerprint(self, model_name: str) -> str:
        root = self._ycb_models_path / model_name
        lines: List[str] = []
        sdf_path = root / "model.sdf"
        if sdf_path.is_file():
            st = sdf_path.stat()
            lines.append(f"model.sdf:{st.st_mtime_ns}:{st.st_size}")
        meshes_dir = root / "meshes"
        if meshes_dir.is_dir():
            for path in sorted(meshes_dir.glob("*.dae")):
                st = path.resolve().stat()
                lines.append(f"meshes/{path.name}:{st.st_mtime_ns}:{st.st_size}")
        return "\n".join(lines)

    def _build_texture_unique_model_sdf(self, model_name: str) -> Path:
        """Evita que Ogre reutilice la misma textura para todos los YCB (todas usan texture_map.png)."""
        root = self._ycb_models_path / model_name
        src_sdf = root / "model.sdf"
        if not src_sdf.is_file():
            raise FileNotFoundError(f"model.sdf no encontrado: {src_sdf}")
        tex_src = self._find_texture_map_png(root)
        if tex_src is None:
            raise FileNotFoundError("sin texture_map.png en meshes/ ni materials/")
        fingerprint = self._model_source_fingerprint(model_name)
        cache_model_dir = self._texture_cache_root / model_name
        stamp_path = cache_model_dir / "source_fingerprint.txt"
        out_sdf = cache_model_dir / "model.sdf"
        meshes_out = cache_model_dir / "meshes"
        if (
            out_sdf.is_file()
            and stamp_path.is_file()
            and stamp_path.read_text(encoding="utf-8") == fingerprint
        ):
            return out_sdf
        if cache_model_dir.exists():
            shutil.rmtree(cache_model_dir)
        meshes_out.mkdir(parents=True)
        unique_png = f"{model_name}_panda_ycb_albedo.png"
        shutil.copy2(tex_src, meshes_out / unique_png)
        meshes_src = root / "meshes"
        if not meshes_src.is_dir():
            raise FileNotFoundError(f"meshes/ no encontrado: {meshes_src}")
        for dae_path in sorted(meshes_src.glob("*.dae")):
            text = dae_path.read_text(encoding="utf-8", errors="replace")
            if "texture_map.png" in text:
                text = text.replace(
                    "<init_from>texture_map.png</init_from>",
                    f"<init_from>{unique_png}</init_from>",
                )
            (meshes_out / dae_path.name).write_text(text, encoding="utf-8")
        sdf_text = src_sdf.read_text(encoding="utf-8")
        pattern = re.compile(
            rf'model://{re.escape(model_name)}/meshes/([^\s<>"\']+)'
        )

        def replacer(match: re.Match[str]) -> str:
            fname = match.group(1)
            target = (meshes_out / fname).resolve()
            if not target.is_file():
                raise FileNotFoundError(f"malla no copiada a caché: {fname}")
            return target.as_uri()

        new_sdf = pattern.sub(replacer, sdf_text)
        out_sdf.write_text(new_sdf, encoding="utf-8")
        stamp_path.write_text(fingerprint, encoding="utf-8")
        self.get_logger().info(
            f"Caché textura única para '{model_name}' → {cache_model_dir}"
        )
        return out_sdf

    def _resolve_spawn_model_sdf(self, model_name: str) -> Path:
        if not self._texture_unique_cache:
            return self._ycb_models_path / model_name / "model.sdf"
        try:
            return self._build_texture_unique_model_sdf(model_name)
        except FileNotFoundError as exc:
            self.get_logger().warning(
                f"texture_unique_cache falló para {model_name} ({exc}); "
                "usando model.sdf del paquete (texturas pueden mezclarse)."
            )
            return self._ycb_models_path / model_name / "model.sdf"

    def _entity_quaternion(
        self, roll_rad: float, pitch_rad: float, yaw_rad: float
    ) -> Tuple[float, float, float, float]:
        qx, qy, qz, qw = tf_transformations.quaternion_from_euler(
            roll_rad, pitch_rad, yaw_rad, axes="sxyz"
        )
        return float(qx), float(qy), float(qz), float(qw)

    def _spawn_entity(self, spawned: SpawnedObject) -> None:
        model_sdf = self._resolve_spawn_model_sdf(spawned.class_cfg.model_name)
        if not model_sdf.is_file():
            raise FileNotFoundError(f"Model SDF not found: {model_sdf}")
        request = GzSpawnEntity.Request()
        request.entity_factory.name = spawned.entity_name
        request.entity_factory.allow_renaming = False
        request.entity_factory.sdf_filename = str(model_sdf)
        request.entity_factory.pose.position.x = float(spawned.x_m)
        request.entity_factory.pose.position.y = float(spawned.y_m)
        request.entity_factory.pose.position.z = float(spawned.z_m)
        qx, qy, qz, qw = self._entity_quaternion(
            spawned.roll_rad, spawned.pitch_rad, spawned.yaw_rad
        )
        request.entity_factory.pose.orientation.x = qx
        request.entity_factory.pose.orientation.y = qy
        request.entity_factory.pose.orientation.z = qz
        request.entity_factory.pose.orientation.w = qw
        request.entity_factory.relative_to = "world"
        response = self._call_service(
            self._spawn_entity_client,
            request,
            f"/world/{self._gz_world_name}/create",
            self._spawn_timeout_sec,
        )
        if not response.success:
            raise RuntimeError(
                f"No se pudo spawnear '{spawned.entity_name}' en {self._gz_world_name}."
            )

    def _set_entity_pose(self, spawned: SpawnedObject) -> None:
        request = GzSetEntityPose.Request()
        request.entity.name = spawned.entity_name
        request.entity.type = GzEntity.MODEL
        request.pose.position.x = float(spawned.x_m)
        request.pose.position.y = float(spawned.y_m)
        request.pose.position.z = float(spawned.z_m)
        qx, qy, qz, qw = self._entity_quaternion(
            spawned.roll_rad, spawned.pitch_rad, spawned.yaw_rad
        )
        request.pose.orientation.x = qx
        request.pose.orientation.y = qy
        request.pose.orientation.z = qz
        request.pose.orientation.w = qw
        response = self._call_service(
            self._set_pose_client,
            request,
            f"/world/{self._gz_world_name}/set_pose",
            self._spawn_timeout_sec,
        )
        if not response.success:
            raise RuntimeError(
                f"No se pudo recolocar '{spawned.entity_name}' en {self._gz_world_name}."
            )

    def _park_pose_for_entity(self, entity_name: str) -> SpawnedObject:
        try:
            pool_idx = self._all_pool_entity_names.index(entity_name)
        except ValueError as exc:
            raise KeyError(f"Entidad fuera del pool: {entity_name}") from exc
        class_cfg = next(
            cfg
            for cfg in self._classes
            if entity_name in self._pool_entities_by_class[cfg.name]
        )
        col = pool_idx % 6
        row = pool_idx // 6
        # Aparcamos el pool muy lejos de la mesa y de la cámara. Así mantenemos
        # la ventaja del pool persistente sin dejar “filas” de objetos visibles
        # en la zona de trabajo.
        x_m = self._parking_origin_x_m - self._parking_row_step_m * row
        y_m = self._parking_origin_y_m - self._parking_col_step_m * col
        z_m = class_cfg.height_m * 0.5 + self._spawn_z_epsilon_m
        return SpawnedObject(
            entity_name=entity_name,
            class_cfg=class_cfg,
            x_m=x_m,
            y_m=y_m,
            z_m=z_m,
            yaw_rad=0.0,
            roll_rad=0.0,
            pitch_rad=0.0,
            lying_on_side=False,
        )

    def _spawn_entity_pool(self) -> None:
        total_entities = len(self._all_pool_entity_names)
        created_count = 0
        self.get_logger().info(
            f"Creando pool persistente de {total_entities} entidades YCB. "
            "Esto puede tardar bastante la primera vez mientras Gazebo carga mallas y texturas."
        )
        for class_cfg in self._classes:
            for entity_name in self._pool_entities_by_class[class_cfg.name]:
                parked = self._park_pose_for_entity(entity_name)
                self._spawn_entity(parked)
                created_count += 1
                if created_count % 10 == 0 or created_count == total_entities:
                    self.get_logger().info(
                        f"Pool YCB: {created_count}/{total_entities} entidades listas."
                    )
                time.sleep(0.01)
        self.get_logger().info(
            "Pool persistente YCB inicializado. A partir de aquí ya empiezan las escenas del dataset."
        )

    def _assign_scene_entity_names(
        self, scene: List[SpawnedObject]
    ) -> List[SpawnedObject]:
        class_instance_count: Dict[str, int] = {}
        assigned: List[SpawnedObject] = []
        for spawned in scene:
            used = class_instance_count.get(spawned.class_cfg.name, 0)
            if used >= len(self._pool_entities_by_class[spawned.class_cfg.name]):
                raise RuntimeError(
                    f"No hay suficientes réplicas en el pool para {spawned.class_cfg.name}"
                )
            entity_name = self._pool_entities_by_class[spawned.class_cfg.name][used]
            class_instance_count[spawned.class_cfg.name] = used + 1
            assigned.append(replace(spawned, entity_name=entity_name))
        return assigned

    def _set_pose_matches_target(
        self,
        target: SpawnedObject,
        pose6: Optional[Tuple[float, float, float, float, float, float]],
    ) -> bool:
        if pose6 is None:
            return False
        return (
            abs(float(pose6[0]) - target.x_m) <= self._set_pose_verify_xy_tol_m
            and abs(float(pose6[1]) - target.y_m) <= self._set_pose_verify_xy_tol_m
            and abs(float(pose6[2]) - target.z_m) <= self._set_pose_verify_z_tol_m
        )

    def _collect_ros_keys_for_entity(self, entity_name: str) -> Set[str]:
        keys: Set[str] = {entity_name, f"{entity_name}::link"}
        all_keys = set(self._world_pose_by_model.keys()) | set(self._world_pose_stamp_by_model.keys())
        for key in all_keys:
            if key.endswith(f"::{entity_name}") or key.startswith(f"{entity_name}::"):
                keys.add(key)
        return keys

    def _invalidate_ros_pose_cache_for_entity(self, entity_name: str) -> None:
        for key in self._collect_ros_keys_for_entity(entity_name):
            self._world_pose_by_model.pop(key, None)
            self._world_pose_stamp_by_model.pop(key, None)

    def _verify_entity_pose_after_set(self, target: SpawnedObject) -> PoseObservation:
        """Verificación inmediata post set_pose/respawn: solo `gz model -p` (fiable).

        No usa ROS TF para esta comprobación: el bridge puede seguir exponiendo la pose
        de parking unos instantes y provocar falsos fallos.
        """
        cli_timeout = min(1.0, max(0.35, self._gz_model_pose_timeout_sec * 0.25))
        for attempt in range(3):
            pose_cli = self._gz_model_pose_rpy(
                target.entity_name, timeout_sec=cli_timeout
            )
            if pose_cli is not None:
                return PoseObservation(
                    pose6=pose_cli,
                    source="gz_cli",
                    fresh=True,
                    from_cache=False,
                    stamp_ns=self.get_clock().now().nanoseconds,
                )
            if attempt < 2:
                self._spin_for_seconds(0.05)
        return PoseObservation(
            pose6=None,
            source="none",
            fresh=False,
            from_cache=False,
            stamp_ns=0,
        )

    def _classify_immediate_pose_verification(
        self,
        target: SpawnedObject,
        observation: PoseObservation,
    ) -> Literal["matched", "mismatched", "uncertain"]:
        """Clasifica la verificación inmediata; `ros_tf_aux` nunca cuenta como fuente fiable."""
        if observation.source == "ros_tf_aux":
            return "uncertain"
        if observation.source == "gz_cli" and observation.pose6 is not None:
            if self._set_pose_matches_target(target, observation.pose6):
                return "matched"
            return "mismatched"
        if observation.pose6 is None or observation.source in ("none", ""):
            return "uncertain"
        return "uncertain"

    def _log_pose_activation_verify(
        self,
        *,
        entity_role: str,
        target: SpawnedObject,
        observation: PoseObservation,
        verdict: str,
        phase: str,
        fallback_respawn_used: bool,
    ) -> None:
        self.get_logger().info(
            f"pose_activation_verify entity={target.entity_name} role={entity_role} "
            f"phase={phase} "
            f"target_xyz={[round(target.x_m, 4), round(target.y_m, 4), round(target.z_m, 4)]} "
            f"verify_source={observation.source} verify_result={verdict} "
            f"observed_pose={self._pose6_to_str(observation.pose6)} "
            f"fallback_respawn={fallback_respawn_used}"
        )

    @staticmethod
    def _pose6_to_str(
        pose6: Optional[Tuple[float, float, float, float, float, float]]
    ) -> str:
        if pose6 is None:
            return "None"
        return "[" + ", ".join(f"{float(v):.4f}" for v in pose6) + "]"

    def _move_entity_via_respawn(
        self,
        target: SpawnedObject,
        *,
        strict: bool,
        entity_role: str,
    ) -> bool:
        self.get_logger().info(
            f"respawn move role={entity_role} entity={target.entity_name} "
            f"target={[round(target.x_m, 4), round(target.y_m, 4), round(target.z_m, 4)]}"
        )
        try:
            self._delete_entity(target.entity_name)
            if entity_role != "parking":
                time.sleep(0.01)
                self._spawn_entity(target)
            return True
        except BaseException as exc:
            message = (
                f"respawn move falló role={entity_role} entity={target.entity_name}: {exc}"
            )
            if strict:
                raise RuntimeError(message) from exc
            self.get_logger().warning(f"{message}. Continuando en best-effort.")
            return False

    def _set_entity_pose_with_recovery(
        self,
        target: SpawnedObject,
        *,
        strict: bool,
        entity_role: str,
    ) -> bool:
        last_observation = PoseObservation(
            pose6=None,
            source="none",
            fresh=False,
            from_cache=False,
            stamp_ns=0,
        )
        last_verdict: Literal["matched", "mismatched", "uncertain"] = "uncertain"
        last_error: Optional[BaseException] = None
        fallback_respawn_used = False
        set_pose_succeeded_once = False

        def _note_uncertain_active(reason: str) -> None:
            if entity_role != "active":
                return
            self._run_stats["activation_verify_uncertain"] += 1
            note = (
                f"entity={target.entity_name} role={entity_role} "
                f"verification_uncertain_after_respawn={fallback_respawn_used} "
                f"reason={reason}"
            )
            self._scene_activation_verify_notes.append(note)

        for retry in range(self._set_pose_verify_max_retries + 1):
            self._invalidate_ros_pose_cache_for_entity(target.entity_name)
            try:
                self._set_entity_pose(target)
            except BaseException as exc:
                last_error = exc
                if retry < self._set_pose_verify_max_retries:
                    continue
                break
            set_pose_succeeded_once = True
            self._spin_for_seconds(0.03)
            last_observation = self._verify_entity_pose_after_set(target)
            last_verdict = self._classify_immediate_pose_verification(
                target, last_observation
            )
            self._log_pose_activation_verify(
                entity_role=entity_role,
                target=target,
                observation=last_observation,
                verdict=last_verdict,
                phase=f"set_pose_attempt_{retry + 1}_of_{self._set_pose_verify_max_retries + 1}",
                fallback_respawn_used=False,
            )
            if last_verdict == "matched":
                return True
            if retry < self._set_pose_verify_max_retries:
                continue

        if not set_pose_succeeded_once:
            message = (
                f"set_pose falló en todos los intentos role={entity_role} "
                f"entity={target.entity_name} last_error={last_error}"
            )
            if strict:
                raise RuntimeError(message)
            self.get_logger().warning(f"{message}. Continuando en best-effort.")
            return False

        if not self._set_pose_fallback_respawn:
            if last_verdict == "mismatched" and strict:
                raise RuntimeError(
                    f"set_pose verificación estricta (gz_cli) role={entity_role} "
                    f"entity={target.entity_name}: pose no coincide con objetivo "
                    f"pose={self._pose6_to_str(last_observation.pose6)} last_error={last_error}"
                )
            if last_verdict == "mismatched" and not strict:
                self.get_logger().warning(
                    f"set_pose role={entity_role} entity={target.entity_name}: "
                    f"verify_result=mismatched source=gz_cli "
                    f"pose={self._pose6_to_str(last_observation.pose6)} last_error={last_error}. "
                    "Continuando en best-effort."
                )
                return False
            message = (
                f"set_pose verify_result=uncertain role={entity_role} entity={target.entity_name}: "
                f"sin lectura fiable de gz_cli tras reintentos; last_error={last_error}"
            )
            if strict and entity_role == "active":
                self.get_logger().warning(
                    f"{message}. No se aborta la escena; la validación visual posterior decidirá."
                )
                _note_uncertain_active("no_respawn_fallback")
                return True
            if strict:
                raise RuntimeError(message)
            self.get_logger().warning(f"{message}. Continuando en best-effort.")
            return False

        if last_verdict == "mismatched":
            self.get_logger().warning(
                f"{target.entity_name}: set_pose verify_result=mismatched (gz_cli) "
                f"(role={entity_role}); usando fallback delete+spawn."
            )
        else:
            self.get_logger().warning(
                f"{target.entity_name}: set_pose verify_result=uncertain (role={entity_role}); "
                "usando fallback selectivo delete+spawn."
            )
        fallback_respawn_used = True
        try:
            self._delete_entity(target.entity_name)
            time.sleep(0.01)
            self._spawn_entity(target)
        except BaseException as exc:
            if strict:
                raise RuntimeError(
                    f"Fallback delete+spawn falló para {target.entity_name}: {exc}"
                ) from exc
            self.get_logger().warning(
                f"Fallback delete+spawn falló role={entity_role} entity={target.entity_name}: {exc}. "
                "Continuando en best-effort."
            )
            return False

        self._invalidate_ros_pose_cache_for_entity(target.entity_name)
        self._spin_for_seconds(0.05)
        last_observation = self._verify_entity_pose_after_set(target)
        last_verdict = self._classify_immediate_pose_verification(
            target, last_observation
        )
        self._log_pose_activation_verify(
            entity_role=entity_role,
            target=target,
            observation=last_observation,
            verdict=last_verdict,
            phase="after_delete_spawn_fallback",
            fallback_respawn_used=True,
        )
        if last_verdict == "matched":
            self._run_stats["set_pose_recovery_count"] += 1
            return True
        if last_verdict == "mismatched" and strict:
            raise RuntimeError(
                f"Fallback delete+spawn: verify_result=mismatched (gz_cli) "
                f"role={entity_role} entity={target.entity_name} "
                f"pose={self._pose6_to_str(last_observation.pose6)}"
            )
        if last_verdict == "mismatched" and not strict:
            self.get_logger().warning(
                f"Fallback delete+spawn role={entity_role} entity={target.entity_name}: "
                f"verify_result=mismatched source=gz_cli "
                f"pose={self._pose6_to_str(last_observation.pose6)}. "
                "Continuando en best-effort."
            )
            return False

        self.get_logger().warning(
            f"Fallback delete+spawn: verify_result=uncertain role={entity_role} "
            f"entity={target.entity_name} (gz_cli no devolvió pose a tiempo). "
            "No se aborta; la validación visual posterior decidirá."
        )
        if entity_role == "active":
            _note_uncertain_active("after_delete_spawn_fallback")
        if strict and entity_role != "active":
            return False
        return True

    def _move_entity(
        self,
        target: SpawnedObject,
        *,
        strict: bool,
        entity_role: str,
    ) -> bool:
        """Backend de movimiento: respawn (delete+spawn) o set_pose experimental."""
        if self._entity_motion_mode == "respawn":
            return self._move_entity_via_respawn(
                target,
                strict=strict,
                entity_role=entity_role,
            )
        return self._set_entity_pose_with_recovery(
            target,
            strict=strict,
            entity_role=entity_role,
        )

    def verify_parking_entity_best_effort(self, target: SpawnedObject) -> bool:
        return self._move_entity(
            target,
            strict=False,
            entity_role="parking",
        )

    def verify_active_entity_strict(self, target: SpawnedObject) -> bool:
        return self._move_entity(
            target,
            strict=True,
            entity_role="active",
        )

    @staticmethod
    def _sorted_entity_list(items: Set[str]) -> List[str]:
        return sorted(items)

    def _activate_scene_entities(self, scene: List[SpawnedObject]) -> None:
        """Activa una escena distinguiendo reutilización, parking y activación real."""
        self._scene_activation_verify_notes.clear()
        new_active: Set[str] = {spawned.entity_name for spawned in scene}
        old_active: Set[str] = set(self._active_entity_names_previous)
        reused_entities = old_active & new_active
        entities_to_park = old_active - new_active
        entities_to_activate = new_active

        self.get_logger().info(
            f"Movimiento escena mode={self._entity_motion_mode} "
            f"old_active={self._sorted_entity_list(old_active)} "
            f"new_active={self._sorted_entity_list(new_active)} "
            f"entities_to_park={self._sorted_entity_list(entities_to_park)} "
            f"entities_to_activate={self._sorted_entity_list(entities_to_activate)} "
            f"reused={self._sorted_entity_list(reused_entities)}"
        )

        for entity_name in self._sorted_entity_list(entities_to_park):
            self.verify_parking_entity_best_effort(
                self._park_pose_for_entity(entity_name)
            )
            time.sleep(0.002)

        for spawned in scene:
            self.verify_active_entity_strict(spawned)
            time.sleep(0.002)

        self._active_entity_names_previous = [spawned.entity_name for spawned in scene]
        self.get_logger().info("Entidades activas recolocadas.")

    @staticmethod
    def _effective_footprint_dims(
        class_cfg: YCBClassConfig,
    ) -> Tuple[float, float]:
        width = (
            class_cfg.footprint_width_m
            if class_cfg.footprint_width_m > 0.0
            else class_cfg.width_m
        )
        length = (
            class_cfg.footprint_length_m
            if class_cfg.footprint_length_m > 0.0
            else class_cfg.length_m
        )
        return width, length

    @staticmethod
    def _footprint_half_axes(
        class_cfg: YCBClassConfig, lying_on_side: bool
    ) -> Tuple[float, float]:
        w, ell = YCBDatasetGenerator._effective_footprint_dims(class_cfg)
        h = class_cfg.height_m
        if not lying_on_side:
            return w / 2.0, ell / 2.0
        s0, s1, _s2 = sorted([w, ell, h])
        return s0 / 2.0, s1 / 2.0

    def _xy_footprint_radius_for_pose(
        self, class_cfg: YCBClassConfig, lying_on_side: bool
    ) -> float:
        hw, hl = self._footprint_half_axes(class_cfg, lying_on_side)
        return math.hypot(hw, hl)

    @staticmethod
    def _collision_corners_bottom_centered(
        width_m: float, length_m: float, height_m: float
    ) -> np.ndarray:
        """8 esquinas de la AABB con z=0 en la base (alineado con model.sdf YCB típicos)."""
        hx = width_m / 2.0
        hy = length_m / 2.0
        h = height_m
        corners: List[List[float]] = []
        for sx in (-1.0, 1.0):
            for sy in (-1.0, 1.0):
                for sz in (0.0, 1.0):
                    corners.append([sx * hx, sy * hy, sz * h])
        return np.array(corners, dtype=np.float64)

    @staticmethod
    def _min_world_z_rotated_bbox_bottom_centered(
        width_m: float,
        length_m: float,
        height_m: float,
        roll_rad: float,
        pitch_rad: float,
        yaw_rad: float,
    ) -> float:
        corners = YCBDatasetGenerator._collision_corners_bottom_centered(
            width_m, length_m, height_m
        )
        rot_mat = tf_transformations.euler_matrix(
            roll_rad, pitch_rad, yaw_rad, axes="sxyz"
        )[:3, :3]
        z_coords = (rot_mat @ corners.T).T[:, 2]
        return float(np.min(z_coords))

    def _spawn_origin_z_for_table_contact(
        self,
        class_cfg: YCBClassConfig,
        roll_rad: float,
        pitch_rad: float,
        yaw_rad: float,
        lying_on_side: bool,
    ) -> float:
        """Z del origen del modelo para que el punto más bajo de la AABB rotada quede sobre la mesa."""
        rz_min = self._min_world_z_rotated_bbox_bottom_centered(
            class_cfg.width_m,
            class_cfg.length_m,
            class_cfg.height_m,
            roll_rad,
            pitch_rad,
            yaw_rad,
        )
        z_table = (
            self._table_surface_z_m
            + self._spawn_z_epsilon_m
            + self._global_spawn_z_lift_m
            + class_cfg.spawn_z_offset_m
        )
        if lying_on_side:
            z_table += self._lying_z_boost_m
        return z_table - rz_min

    def _table_spawn_bounds_xy(
        self, r_m: float, lying_on_side: bool
    ) -> Tuple[float, float, float, float]:
        """Interseca la caja de spawn del usuario con el tablero menos margen y radio de huella."""
        cx = self._table_center_x_m
        cy = self._table_center_y_m
        hx = self._table_half_extent_x_m
        hy = self._table_half_extent_y_m
        margin = self._table_edge_margin_m + (
            self._lying_table_edge_extra_margin_m if lying_on_side else 0.0
        )
        tx0 = cx - hx + margin + r_m
        tx1 = cx + hx - margin - r_m
        ty0 = cy - hy + margin + r_m
        ty1 = cy + hy - margin - r_m
        xmin = max(self._spawn_x_min, tx0)
        xmax = min(self._spawn_x_max, tx1)
        ymin = max(self._spawn_y_min, ty0)
        ymax = min(self._spawn_y_max, ty1)
        return xmin, xmax, ymin, ymax

    def _required_center_distance_m(
        self, r_new: float, r_old: float
    ) -> float:
        return max(
            r_new + r_old + self._min_surface_gap_m,
            self._min_center_distance_m,
        )

    @staticmethod
    def _polygon_area(poly: np.ndarray) -> float:
        contour = np.asarray(poly, dtype=np.float32).reshape(-1, 1, 2)
        return abs(float(cv2.contourArea(contour)))

    @staticmethod
    def _polygon_intersection_area(poly_a: np.ndarray, poly_b: np.ndarray) -> float:
        contour_a = np.asarray(poly_a, dtype=np.float32).reshape(-1, 1, 2)
        contour_b = np.asarray(poly_b, dtype=np.float32).reshape(-1, 1, 2)
        try:
            area, _ = cv2.intersectConvexConvex(contour_a, contour_b)
        except cv2.error:
            return 0.0
        return max(0.0, float(area))

    @staticmethod
    def _point_segment_distance(
        point: np.ndarray, seg_a: np.ndarray, seg_b: np.ndarray
    ) -> float:
        segment = seg_b - seg_a
        denom = float(np.dot(segment, segment))
        if denom <= 1e-12:
            return float(np.linalg.norm(point - seg_a))
        t = float(np.dot(point - seg_a, segment) / denom)
        t = max(0.0, min(1.0, t))
        closest = seg_a + t * segment
        return float(np.linalg.norm(point - closest))

    @classmethod
    def _segment_distance(
        cls,
        a0: np.ndarray,
        a1: np.ndarray,
        b0: np.ndarray,
        b1: np.ndarray,
    ) -> float:
        return min(
            cls._point_segment_distance(a0, b0, b1),
            cls._point_segment_distance(a1, b0, b1),
            cls._point_segment_distance(b0, a0, a1),
            cls._point_segment_distance(b1, a0, a1),
        )

    @classmethod
    def _polygon_min_distance(cls, poly_a: np.ndarray, poly_b: np.ndarray) -> float:
        if cls._polygon_intersection_area(poly_a, poly_b) > 1e-9:
            return 0.0
        min_distance = float("inf")
        for idx_a in range(len(poly_a)):
            a0 = poly_a[idx_a]
            a1 = poly_a[(idx_a + 1) % len(poly_a)]
            for idx_b in range(len(poly_b)):
                b0 = poly_b[idx_b]
                b1 = poly_b[(idx_b + 1) % len(poly_b)]
                min_distance = min(
                    min_distance, cls._segment_distance(a0, a1, b0, b1)
                )
        return min_distance

    def _footprint_polygon_xy(
        self,
        class_cfg: YCBClassConfig,
        x_m: float,
        y_m: float,
        yaw_rad: float,
        lying_on_side: bool,
    ) -> np.ndarray:
        half_width, half_length = self._footprint_half_axes(class_cfg, lying_on_side)
        local = np.array(
            [
                [-half_width, -half_length],
                [half_width, -half_length],
                [half_width, half_length],
                [-half_width, half_length],
            ],
            dtype=np.float64,
        )
        yaw_total = yaw_rad + class_cfg.visual_yaw_offset_rad
        cz = math.cos(yaw_total)
        sz = math.sin(yaw_total)
        rotation = np.array(
            [
                [cz, -sz],
                [sz, cz],
            ],
            dtype=np.float64,
        )
        polygon = (rotation @ local.T).T
        center_offset = rotation @ np.array(
            [class_cfg.visual_offset_x_m, class_cfg.visual_offset_y_m],
            dtype=np.float64,
        )
        polygon[:, 0] += x_m + float(center_offset[0])
        polygon[:, 1] += y_m + float(center_offset[1])
        return polygon

    def _sample_object_position(
        self,
        placed: List[np.ndarray],
        class_cfg: YCBClassConfig,
        lying_on_side: bool,
        yaw_rad: float,
        rng: random.Random,
    ) -> Tuple[float, float]:
        r_new = self._xy_footprint_radius_for_pose(class_cfg, lying_on_side)
        xmin, xmax, ymin, ymax = self._table_spawn_bounds_xy(r_new, lying_on_side)
        if xmin >= xmax or ymin >= ymax:
            raise RuntimeError(
                f"No cabe la huella (r≈{r_new:.3f} m) dentro del tablero con márgenes. "
                "Reduce objetos grandes tumbados, amplía table_half_extent_* o spawn_*."
            )
        for _ in range(self._max_position_samples):
            x_m = rng.uniform(xmin, xmax)
            y_m = rng.uniform(ymin, ymax)
            candidate_polygon = self._footprint_polygon_xy(
                class_cfg, x_m, y_m, yaw_rad, lying_on_side
            )
            if all(
                self._polygon_min_distance(candidate_polygon, placed_polygon)
                >= self._min_surface_gap_m
                for placed_polygon in placed
            ):
                return x_m, y_m
        raise RuntimeError(
            "Could not sample a non-overlapping pose. Increase the spawn area, "
            "raise min_surface_gap_m slightly, or reduce min_objects/max_objects."
        )

    def _sample_scene_class_order(
        self, object_count: int, rng: random.Random
    ) -> List[YCBClassConfig]:
        if not self._diverse_classes_per_scene:
            return [rng.choice(self._classes) for _ in range(object_count)]
        n_cls = len(self._classes)
        if object_count <= n_cls:
            return rng.sample(self._classes, object_count)
        base = list(self._classes)
        rng.shuffle(base)
        extra = object_count - n_cls
        return base + rng.choices(self._classes, k=extra)

    def _sample_scene_layout(self, rng: random.Random) -> List[SpawnedObject]:
        """Un intento de escena: coloca primero las huellas más grandes (mejor empaquetado)."""
        object_count = rng.randint(self._min_objects, self._max_objects)
        class_order = self._sample_scene_class_order(object_count, rng)
        pending: List[Tuple[YCBClassConfig, bool, float]] = []
        for class_cfg in class_order:
            lying_on_side = class_cfg.allow_lying and (
                rng.random() < self._lying_down_probability
            )
            r = self._xy_footprint_radius_for_pose(class_cfg, lying_on_side)
            pending.append((class_cfg, lying_on_side, r))
        pending.sort(key=lambda item: item[2], reverse=True)

        placed: List[np.ndarray] = []
        scene: List[SpawnedObject] = []
        for slot_idx, (class_cfg, lying_on_side, _) in enumerate(pending):
            yaw_rad = rng.uniform(-math.pi, math.pi)
            x_m, y_m = self._sample_object_position(
                placed, class_cfg, lying_on_side, yaw_rad, rng
            )
            if lying_on_side:
                pitch_rad = (
                    math.pi / 2.0
                    + rng.uniform(
                        -self._lying_pitch_jitter_rad, self._lying_pitch_jitter_rad
                    )
                )
                roll_rad = (
                    rng.uniform(-math.pi, math.pi)
                    if self._lying_random_roll
                    else 0.0
                )
            else:
                pitch_rad = 0.0
                roll_rad = 0.0
            z_contact = self._spawn_origin_z_for_table_contact(
                class_cfg, roll_rad, pitch_rad, yaw_rad, lying_on_side
            )
            scene.append(
                SpawnedObject(
                    entity_name=f"scene_slot_{slot_idx}",
                    class_cfg=class_cfg,
                    x_m=x_m,
                    y_m=y_m,
                    z_m=z_contact,
                    yaw_rad=yaw_rad,
                    roll_rad=roll_rad,
                    pitch_rad=pitch_rad,
                    lying_on_side=lying_on_side,
                )
            )
            placed.append(
                self._footprint_polygon_xy(
                    class_cfg, x_m, y_m, yaw_rad, lying_on_side
                )
            )
        return scene

    def _sample_calibration_scene(self, rng: random.Random) -> List[SpawnedObject]:
        if self._calibration_class_name:
            class_cfg = self._class_by_name[self._calibration_class_name]
        else:
            class_cfg = self._classes[0]
        lying_on_side = (
            self._calibration_allow_lying
            and class_cfg.allow_lying
            and self._lying_down_probability > 0.0
        )
        yaw_rad = rng.uniform(-math.pi, math.pi)
        x_m, y_m = self._sample_object_position(
            [],
            class_cfg,
            lying_on_side,
            yaw_rad,
            rng,
        )
        if lying_on_side:
            pitch_rad = math.pi / 2.0
            roll_rad = 0.0
        else:
            pitch_rad = 0.0
            roll_rad = 0.0
        z_contact = self._spawn_origin_z_for_table_contact(
            class_cfg, roll_rad, pitch_rad, yaw_rad, lying_on_side
        )
        return [
            SpawnedObject(
                entity_name="scene_slot_0",
                class_cfg=class_cfg,
                x_m=x_m,
                y_m=y_m,
                z_m=z_contact,
                yaw_rad=yaw_rad,
                roll_rad=roll_rad,
                pitch_rad=pitch_rad,
                lying_on_side=lying_on_side,
            )
        ]

    def _sample_scene(self, rng: random.Random) -> List[SpawnedObject]:
        if self._calibration_mode:
            return self._sample_calibration_scene(rng)
        last_exc: Optional[BaseException] = None
        for _ in range(self._scene_layout_max_attempts):
            try:
                return self._sample_scene_layout(rng)
            except RuntimeError as exc:
                last_exc = exc
        raise RuntimeError(
            "No se pudo colocar ninguna escena tras "
            f"{self._scene_layout_max_attempts} intentos. "
            "Prueba a bajar max_objects, subir scene_layout_max_attempts, "
            "reducir min_surface_gap_m o lying_down_probability. "
            f"Último error: {last_exc}"
        ) from last_exc

    def _lookup_world_to_camera(self, camera_frame: str):
        try:
            return self._tf_buffer.lookup_transform(
                camera_frame,
                self._world_frame,
                Time(),
                timeout=Duration(seconds=0.5),
            )
        except (
            tf2_ros.LookupException,
            tf2_ros.ConnectivityException,
            tf2_ros.ExtrapolationException,
        ) as exc:
            raise RuntimeError(
                f"Could not resolve TF {self._world_frame}->{camera_frame}: {exc}"
            ) from exc

    def _gz_model_pose_rpy(
        self, entity_name: str, timeout_sec: Optional[float] = None
    ) -> Optional[Tuple[float, float, float, float, float, float]]:
        """Pose del modelo en mundo: x y z roll pitch yaw (rad), vía `gz model -p`."""
        command = [
            "gz",
            "model",
            "-m",
            entity_name,
            "-w",
            self._gz_world_name,
            "-p",
        ]
        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=max(0.2, timeout_sec if timeout_sec is not None else self._gz_model_pose_timeout_sec),
                env=self._command_env(),
                check=False,
            )
        except subprocess.TimeoutExpired:
            return None
        if result.returncode != 0:
            return None
        text_out = (result.stdout or "").strip()
        if not text_out:
            return None
        line = text_out.splitlines()[-1].strip()
        parts = line.split()
        if len(parts) < 6:
            return None
        try:
            return (
                float(parts[0]),
                float(parts[1]),
                float(parts[2]),
                float(parts[3]),
                float(parts[4]),
                float(parts[5]),
            )
        except ValueError:
            return None

    def _sim_pose_matches_expected(
        self,
        spawned: SpawnedObject,
        pose6: Tuple[float, float, float, float, float, float],
    ) -> bool:
        """True si la pose real sigue razonablemente cerca de la pose objetivo.

        Validar contra la mesa usando el origen del modelo provoca falsos rechazos con
        algunos YCB, porque su frame no siempre coincide con el centro geométrico que
        asumimos al razonar sobre la huella. Comparar contra la pose planificada del
        propio modelo es más robusto: si tras `set_pose` + `settle_time` el objeto sigue
        cerca de su objetivo, lo aceptamos; si cae al suelo o se desplaza mucho, lo
        rechazamos.
        """
        x_m, y_m, z_m = pose6[0], pose6[1], pose6[2]
        z_lo = self._table_surface_z_m - self._label_pose_z_below_table_m
        z_hi = spawned.z_m + self._label_pose_z_above_table_m
        if not (z_lo <= z_m <= z_hi):
            return False
        if abs(x_m - spawned.x_m) > self._label_pose_xy_slack_m:
            return False
        if abs(y_m - spawned.y_m) > self._label_pose_xy_slack_m:
            return False
        return True

    def _project_world_corners_to_polygon(
        self,
        corners_world: np.ndarray,
        camera_info: CameraInfo,
        image_height: int,
        image_width: int,
    ) -> Optional[np.ndarray]:
        fx, fy, cx, cy = scaled_intrinsics_from_camera_info(
            camera_info, image_height, image_width
        )
        camera_frame = camera_info.header.frame_id or self._camera_optical_frame
        transform = self._lookup_world_to_camera(camera_frame)
        corners_camera = self._transform_points(corners_world, transform)
        if np.any(corners_camera[:, 2] <= 1e-6):
            return None
        if self._label_pose_depth_consistency_check:
            table_world = np.array(
                [
                    [
                        self._table_center_x_m,
                        self._table_center_y_m,
                        self._table_surface_z_m,
                    ]
                ],
                dtype=np.float64,
            )
            table_cam = self._transform_points(table_world, transform)[0]
            z_table = float(table_cam[2])
            mean_corner_z = float(np.mean(corners_camera[:, 2]))
            if (
                abs(mean_corner_z - z_table)
                > self._label_pose_max_mean_depth_delta_from_table_m
            ):
                return None
        u = (corners_camera[:, 0] * fx / corners_camera[:, 2]) + cx
        v = (corners_camera[:, 1] * fy / corners_camera[:, 2]) + cy
        polygon = np.column_stack((u, v))
        if (
            np.any(polygon[:, 0] < 0.0)
            or np.any(polygon[:, 0] > image_width - 1.0)
            or np.any(polygon[:, 1] < 0.0)
            or np.any(polygon[:, 1] > image_height - 1.0)
        ):
            return None
        polygon = normalize_obb_ultralytics(polygon)
        if polygon is None:
            return None
        x_span = float(np.max(polygon[:, 0]) - np.min(polygon[:, 0]))
        y_span = float(np.max(polygon[:, 1]) - np.min(polygon[:, 1]))
        if x_span < 2.0 or y_span < 2.0:
            return None
        return polygon

    @staticmethod
    def _transform_points(points_ref: np.ndarray, transform) -> np.ndarray:
        quat = [
            transform.transform.rotation.x,
            transform.transform.rotation.y,
            transform.transform.rotation.z,
            transform.transform.rotation.w,
        ]
        matrix = tf_transformations.quaternion_matrix(quat)
        matrix[:3, 3] = [
            transform.transform.translation.x,
            transform.transform.translation.y,
            transform.transform.translation.z,
        ]
        homogeneous = np.hstack((points_ref, np.ones((points_ref.shape[0], 1))))
        return (matrix @ homogeneous.T).T[:, :3]

    def _obb_corners_in_link_frame(self, spawned: SpawnedObject) -> np.ndarray:
        """Corners OBB en frame del link, aplicando offsets internos por clase."""
        class_cfg = spawned.class_cfg
        half_width, half_length = self._footprint_half_axes(
            class_cfg, spawned.lying_on_side
        )
        local = np.array(
            [
                [-half_width, -half_length, 0.0],
                [half_width, -half_length, 0.0],
                [half_width, half_length, 0.0],
                [-half_width, half_length, 0.0],
            ],
            dtype=np.float64,
        )
        cy = math.cos(class_cfg.visual_yaw_offset_rad)
        sy = math.sin(class_cfg.visual_yaw_offset_rad)
        yaw_offset_rot = np.array(
            [
                [cy, -sy, 0.0],
                [sy, cy, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        corners = (yaw_offset_rot @ local.T).T
        corners[:, 0] += class_cfg.visual_offset_x_m
        corners[:, 1] += class_cfg.visual_offset_y_m
        corners[:, 2] += class_cfg.visual_offset_z_m
        return corners

    def _project_obb_polygon_pose(
        self,
        spawned: SpawnedObject,
        pose6: Tuple[float, float, float, float, float, float],
        camera_info: CameraInfo,
        image_height: int,
        image_width: int,
    ) -> Optional[np.ndarray]:
        x_m, y_m, z_m, roll, pitch, yaw = pose6
        local_corners = self._obb_corners_in_link_frame(spawned)
        rot_mat = tf_transformations.euler_matrix(roll, pitch, yaw, axes="sxyz")[
            :3, :3
        ]
        corners_world = (rot_mat @ local_corners.T).T + np.array(
            [x_m, y_m, z_m], dtype=np.float64
        )
        return self._project_world_corners_to_polygon(
            corners_world, camera_info, image_height, image_width
        )

    def _project_obb_polygon(
        self,
        spawned: SpawnedObject,
        camera_info: CameraInfo,
        image_height: int,
        image_width: int,
    ) -> Optional[np.ndarray]:
        return self._project_obb_polygon_pose(
            spawned,
            (
                spawned.x_m,
                spawned.y_m,
                spawned.z_m,
                spawned.roll_rad,
                spawned.pitch_rad,
                spawned.yaw_rad,
            ),
            camera_info,
            image_height,
            image_width,
        )

    @staticmethod
    def _polygon_to_label_line(
        class_id: int, polygon: np.ndarray, image_height: int, image_width: int
    ) -> str:
        normalized = []
        for u_px, v_px in polygon:
            normalized.extend(
                [
                    float(u_px / image_width),
                    float(v_px / image_height),
                ]
            )
        values = " ".join(f"{value:.6f}" for value in normalized)
        return f"{class_id} {values}"

    def _scene_to_projected_labels(
        self,
        scene: List[SpawnedObject],
        camera_info: CameraInfo,
        image_height: int,
        image_width: int,
        sim_poses: Optional[
            List[Optional[Tuple[float, float, float, float, float, float]]]
        ] = None,
    ) -> List[ProjectedLabel]:
        projected: List[ProjectedLabel] = []
        for slot_idx, spawned in enumerate(scene):
            pose6: Optional[Tuple[float, float, float, float, float, float]] = None
            if sim_poses is not None and slot_idx < len(sim_poses):
                pose6 = sim_poses[slot_idx]
            if pose6 is not None:
                polygon = self._project_obb_polygon_pose(
                    spawned, pose6, camera_info, image_height, image_width
                )
            else:
                polygon = self._project_obb_polygon(
                    spawned, camera_info, image_height, image_width
                )
            if polygon is None:
                self.get_logger().warn(
                    f"Skipping label for {spawned.entity_name}: projected OBB invalid."
                )
                continue
            projected.append(
                ProjectedLabel(
                    spawned=spawned,
                    polygon=polygon,
                    line=self._polygon_to_label_line(
                        spawned.class_cfg.class_id, polygon, image_height, image_width
                    ),
                )
            )
        return projected

    def _scene_to_label_lines(
        self,
        scene: List[SpawnedObject],
        camera_info: CameraInfo,
        image_height: int,
        image_width: int,
        sim_poses: Optional[
            List[Optional[Tuple[float, float, float, float, float, float]]]
        ] = None,
    ) -> List[str]:
        return [
            item.line
            for item in self._scene_to_projected_labels(
                scene,
                camera_info,
                image_height,
                image_width,
                sim_poses=sim_poses,
            )
        ]

    def _find_projected_label_overlaps(
        self, projected_labels: List[ProjectedLabel]
    ) -> List[str]:
        overlap_messages: List[str] = []
        for idx_a in range(len(projected_labels)):
            label_a = projected_labels[idx_a]
            area_a = self._polygon_area(label_a.polygon)
            if area_a <= 1e-6:
                continue
            for idx_b in range(idx_a + 1, len(projected_labels)):
                label_b = projected_labels[idx_b]
                area_b = self._polygon_area(label_b.polygon)
                if area_b <= 1e-6:
                    continue
                inter_area = self._polygon_intersection_area(
                    label_a.polygon, label_b.polygon
                )
                if inter_area <= 1e-6:
                    continue
                overlap_ratio = inter_area / min(area_a, area_b)
                if overlap_ratio > self._max_projected_overlap_ratio:
                    overlap_messages.append(
                        f"{label_a.spawned.entity_name}<->{label_b.spawned.entity_name} "
                        f"(ratio={overlap_ratio:.3f})"
                    )
        return overlap_messages

    def _capture_background_reference_image(self) -> None:
        baseline_seq = self._image_seq
        threshold_ns = max(self.get_clock().now().nanoseconds, self._last_image_stamp_ns)
        try:
            _msg, bgr, discarded, accepted = self._wait_for_visually_stable_scene_image(
                threshold_ns, baseline_seq
            )
            self._background_reference_bgr = bgr
            self.get_logger().info(
                "Referencia de fondo capturada para validación de render: "
                f"discarded_frames={discarded}, accepted_frames={accepted}."
            )
        except TimeoutError as exc:
            self._background_reference_bgr = None
            self.get_logger().warning(
                f"No se pudo capturar fondo estable ({exc}). "
                "Se desactiva validate_rendered_visibility en este run."
            )
            self._validate_rendered_visibility = False

    def _validate_rendered_visibility_for_scene(
        self,
        scene: List[SpawnedObject],
        projected_labels: List[ProjectedLabel],
        image_bgr: np.ndarray,
    ) -> Tuple[bool, List[str]]:
        if not self._validate_rendered_visibility:
            return True, []
        if self._background_reference_bgr is None:
            return True, []
        if image_bgr.shape != self._background_reference_bgr.shape:
            return True, []

        gray_diff = cv2.cvtColor(
            cv2.absdiff(image_bgr, self._background_reference_bgr), cv2.COLOR_BGR2GRAY
        )
        projected_by_name = {
            item.spawned.entity_name: item for item in projected_labels
        }
        visible_count = 0
        problems: List[str] = []
        for spawned in scene:
            item = projected_by_name.get(spawned.entity_name)
            if item is None:
                problems.append(f"{spawned.entity_name}:sin_poligono")
                continue
            mask = np.zeros(gray_diff.shape, dtype=np.uint8)
            pts = np.round(item.polygon).astype(np.int32).reshape(-1, 1, 2)
            cv2.fillConvexPoly(mask, pts, 255)
            area_px = int(np.count_nonzero(mask))
            if area_px < self._visibility_min_polygon_area_px:
                problems.append(f"{spawned.entity_name}:area_pequena={area_px}")
                continue
            mean_diff = float(cv2.mean(gray_diff, mask=mask)[0])
            changed_px = np.logical_and(
                gray_diff >= self._visibility_pixel_diff_threshold,
                mask > 0,
            )
            changed_fraction = float(np.count_nonzero(changed_px)) / float(area_px)
            if (
                mean_diff >= self._visibility_mean_diff_threshold
                and changed_fraction >= self._visibility_min_changed_fraction
            ):
                visible_count += 1
            else:
                problems.append(
                    f"{spawned.entity_name}:mean={mean_diff:.2f},frac={changed_fraction:.3f}"
                )
        if visible_count != len(scene):
            problems.insert(0, f"visible={visible_count}/{len(scene)}")
            return False, problems
        return True, []

    def _world_obb_corners_for_pose(
        self, spawned: SpawnedObject, pose6: Tuple[float, float, float, float, float, float]
    ) -> np.ndarray:
        x_m, y_m, z_m, roll, pitch, yaw = pose6
        local = self._obb_corners_in_link_frame(spawned)
        rot_mat = tf_transformations.euler_matrix(roll, pitch, yaw, axes="sxyz")[:3, :3]
        return (rot_mat @ local.T).T + np.array([x_m, y_m, z_m], dtype=np.float64)

    def _save_calibration_artifacts(
        self,
        scene_name: str,
        bgr: np.ndarray,
        scene: List[SpawnedObject],
        projected_labels: List[ProjectedLabel],
        sim_poses: Optional[List[Optional[Tuple[float, float, float, float, float, float]]]],
        split: str,
    ) -> None:
        if not self._calibration_mode:
            return
        out_dir = self._calibration_output_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        image_path = out_dir / f"{scene_name}_{split}.png"
        overlay_path = out_dir / f"{scene_name}_{split}_overlay.png"
        json_path = out_dir / f"{scene_name}_{split}.json"

        cv2.imwrite(str(image_path), bgr)
        overlay = bgr.copy()
        projected_by_name = {p.spawned.entity_name: p for p in projected_labels}
        objects_payload: List[Dict[str, object]] = []
        for idx, spawned in enumerate(scene):
            used_pose = (
                sim_poses[idx]
                if sim_poses is not None and idx < len(sim_poses) and sim_poses[idx] is not None
                else (
                    spawned.x_m,
                    spawned.y_m,
                    spawned.z_m,
                    spawned.roll_rad,
                    spawned.pitch_rad,
                    spawned.yaw_rad,
                )
            )
            world_corners = self._world_obb_corners_for_pose(spawned, used_pose)
            label = projected_by_name.get(spawned.entity_name)
            polygon_2d = label.polygon.tolist() if label is not None else []
            if label is not None:
                pts = np.round(label.polygon).astype(np.int32).reshape(-1, 1, 2)
                cv2.polylines(overlay, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
                cx = int(round(float(np.mean(label.polygon[:, 0]))))
                cy = int(round(float(np.mean(label.polygon[:, 1]))))
                cv2.putText(
                    overlay,
                    spawned.class_cfg.name,
                    (cx - 40, cy - 6),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (0, 255, 0),
                    1,
                    cv2.LINE_AA,
                )
            objects_payload.append(
                {
                    "entity_name": spawned.entity_name,
                    "class_name": spawned.class_cfg.name,
                    "planned_pose_world_xyzrpy": [
                        spawned.x_m,
                        spawned.y_m,
                        spawned.z_m,
                        spawned.roll_rad,
                        spawned.pitch_rad,
                        spawned.yaw_rad,
                    ],
                    "pose_used_world_xyzrpy": list(used_pose),
                    "corners_link_3d_m": self._obb_corners_in_link_frame(spawned).tolist(),
                    "corners_world_3d_m": world_corners.tolist(),
                    "corners_image_2d_px": polygon_2d,
                }
            )
        cv2.imwrite(str(overlay_path), overlay)
        payload = {
            "scene_name": scene_name,
            "split": split,
            "world_frame": self._world_frame,
            "objects": objects_payload,
        }
        json_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

    def _reset_output_dir(self) -> None:
        if self._clear_output and self._output_dir.exists():
            for rel_path in [
                "images/train",
                "images/val",
                "labels/train",
                "labels/val",
            ]:
                target = self._output_dir / rel_path
                if target.exists():
                    shutil.rmtree(target)
            data_yaml = self._output_dir / "data.yaml"
            if data_yaml.exists():
                data_yaml.unlink()

        for rel_path in [
            "images/train",
            "images/val",
            "labels/train",
            "labels/val",
        ]:
            (self._output_dir / rel_path).mkdir(parents=True, exist_ok=True)

    def _write_data_yaml(self) -> None:
        names = {class_cfg.class_id: class_cfg.name for class_cfg in self._classes}
        payload = {
            "path": str(self._output_dir.resolve()),
            "train": "images/train",
            "val": "images/val",
            "names": names,
        }
        with (self._output_dir / "data.yaml").open("w", encoding="utf-8") as handle:
            yaml.safe_dump(payload, handle, sort_keys=False)

    def generate(self) -> None:
        self.get_logger().info(
            "Waiting for camera topics before generating the synthetic dataset."
        )
        self._wait_for_camera_ready()
        self._reset_output_dir()
        self._write_data_yaml()
        self._write_run_meta()
        effective_scene_count = 1 if self._calibration_mode else self._scene_count
        self.get_logger().info(
            f"Dataset base_seed={self._base_seed} (seed param={self._seed_param})."
        )

        last_idx = self._start_scene_index + effective_scene_count - 1
        self.get_logger().info(
            f"Generando {effective_scene_count} escenas: scene_{self._start_scene_index:05d} "
            f"hasta scene_{last_idx:05d} (incl.). world_frame={self._world_frame}"
        )

        self._wait_for_world_services_ready()
        self.get_logger().info("Servicios de mundo disponibles.")
        if self._initial_full_purge:
            self._purge_all_dataset_slots()
        if self._entity_motion_mode == "set_pose":
            self.get_logger().info(
                "entity_motion_mode=set_pose: preparando pool persistente de entidades YCB..."
            )
            self._spawn_entity_pool()
            self._clear_previous_scene_slots()
            self.get_logger().info(
                "Pool preparado y zona de trabajo despejada. Empezando generación de escenas."
            )
        else:
            self.get_logger().info(
                "entity_motion_mode=respawn (por defecto): sin pool persistente; "
                "cada activación es delete+spawn (sin set_pose ni verificación inmediata gz_cli)."
            )

        if self._labels_sim_pose_source == "ros_tf":
            self.get_logger().info(
                "Poses de mundo desde "
                f"{self._world_pose_ros_topic} (tf2_msgs/TFMessage), "
                f"world_frame={self._world_frame}."
            )
            self._wait_for_world_pose_bridge_ready()
        elif self._labels_use_sim_pose:
            self.get_logger().info(
                "Etiquetas OBB desde pose real en Gazebo (gz model -p). Se asume que esa "
                f"pose está en world_frame={self._world_frame}."
            )

        if self._validate_rendered_visibility:
            self._spin_for_seconds(0.2)
            self._capture_background_reference_image()

        train_count = 0
        val_count = 0
        max_pose_attempts = self._gz_pose_scene_max_retries + 1
        for scene_idx in range(
            self._start_scene_index, self._start_scene_index + effective_scene_count
        ):
            if self._calibration_mode:
                split = "train"
            else:
                split_rng = random.Random(_seed_for_scene(self._base_seed, scene_idx))
                split = "val" if split_rng.random() < self._val_split else "train"
            scene_name = f"scene_{scene_idx:05d}"
            scene_saved = False

            for attempt in range(max_pose_attempts):
                self._apply_domain_randomization_hooks(scene_idx)
                self.get_logger().info(
                    f"{scene_name}: preparando escena (intento {attempt + 1}/{max_pose_attempts})."
                )
                layout_rng = random.Random(
                    _seed_for_scene_attempt(self._base_seed, scene_idx, attempt)
                )
                scene = self._assign_scene_entity_names(self._sample_scene(layout_rng))
                self.get_logger().info(
                    f"{scene_name}: layout muestreado con {len(scene)} objetos."
                )
                self._activate_scene_entities(scene)

                # Drena mensajes de transición (spawn/delete, poses antiguas, frames de cámara
                # todavía en cola) antes de fijar la nueva línea base temporal de esta escena.
                self._spin_for_seconds(0.12)
                baseline_image_seq = self._image_seq
                baseline_world_pose_seq = self._world_pose_seq
                baseline_image_stamp_ns = self._last_image_stamp_ns
                baseline_world_pose_stamp_ns = self._world_pose_stamp_ns
                self.get_logger().info(
                    f"{scene_name}: baselines tras flush "
                    f"image_seq={baseline_image_seq} "
                    f"world_pose_seq={baseline_world_pose_seq} "
                    f"image_stamp_ns={baseline_image_stamp_ns} "
                    f"world_pose_stamp_ns={baseline_world_pose_stamp_ns}"
                )

                self.get_logger().info(
                    f"{scene_name}: esperando settle_time={self._settle_time:.2f}s."
                )
                self._spin_for_seconds(self._settle_time)
                t_scene_ready_ns = max(
                    self.get_clock().now().nanoseconds,
                    self._last_image_stamp_ns,
                    self._world_pose_stamp_ns,
                )
                pose_snapshot_map: Optional[
                    Dict[str, Tuple[float, float, float, float, float, float]]
                ] = None
                pose_snapshot_stamp_ns: Optional[int] = None
                image_threshold_ns = t_scene_ready_ns
                if self._labels_use_sim_pose and self._labels_sim_pose_source == "ros_tf":
                    self.get_logger().info(
                        f"{scene_name}: esperando snapshot TF coherente de la escena..."
                    )
                    pose_threshold_ns = max(
                        t_scene_ready_ns,
                        baseline_world_pose_stamp_ns,
                    )
                    (
                        pose_snapshot_map,
                        pose_snapshot_stamp_ns,
                    ) = self._wait_for_world_pose_snapshot_after(
                        scene,
                        pose_threshold_ns,
                        min_world_pose_seq=baseline_world_pose_seq,
                    )
                    image_threshold_ns = max(
                        image_threshold_ns,
                        baseline_image_stamp_ns,
                        int(pose_snapshot_stamp_ns),
                    )
                else:
                    image_threshold_ns = max(image_threshold_ns, baseline_image_stamp_ns)
                self.get_logger().info(
                    f"{scene_name}: escena estable en t={t_scene_ready_ns}. "
                    "Esperando estabilización visual..."
                )
                try:
                    (
                        image_msg,
                        bgr,
                        discarded_frames,
                        accepted_frames,
                    ) = self._wait_for_visually_stable_scene_image(
                        image_threshold_ns, baseline_image_seq
                    )
                except TimeoutError as exc:
                    self.get_logger().warning(
                        f"{scene_name}: no se alcanzó estabilidad visual ({exc}); reintentando."
                    )
                    self._run_stats["scene_rejections"] += 1
                    self._run_stats["stale_image_rejections"] += 1
                    time.sleep(0.12)
                    continue
                if self._camera_info is None:
                    raise RuntimeError(
                        "CameraInfo disappeared during dataset generation"
                    )

                image_height, image_width = bgr.shape[:2]
                image_stamp_ns = _stamp_to_ns(image_msg.header.stamp)

                sim_poses_list: Optional[
                    List[Optional[Tuple[float, float, float, float, float, float]]]
                ] = None
                use_sim_poses = False
                if self._labels_use_sim_pose:
                    self.get_logger().info(
                        f"{scene_name}: obteniendo poses reales desde {self._labels_sim_pose_source}."
                    )
                    if (
                        self._labels_sim_pose_source == "ros_tf"
                        and pose_snapshot_map is not None
                    ):
                        sim_poses_list = self._sim_poses_from_snapshot(
                            scene, pose_snapshot_map
                        )
                    else:
                        sim_poses_list = self._sim_poses_for_scene_after(
                            scene, image_stamp_ns
                        )
                    use_sim_poses = all(p is not None for p in sim_poses_list)
                    if not use_sim_poses:
                        if attempt + 1 < max_pose_attempts:
                            src = (
                                "poses ROS (tf)"
                                if self._labels_sim_pose_source == "ros_tf"
                                else "gz model -p"
                            )
                            self.get_logger().warning(
                                f"{scene_name}: {src} incompleto "
                                f"(intento {attempt + 1}/{max_pose_attempts}), "
                                "reintentando escena."
                            )
                            self._run_stats["scene_rejections"] += 1
                            self._run_stats["missing_pose_rejections"] += 1
                            time.sleep(0.12)
                            continue
                        if not self._allow_planned_pose_labels_fallback:
                            self.get_logger().error(
                                f"{scene_name}: sin poses de simulación tras reintentos; "
                                "escena omitida (allow_planned_pose_labels_fallback:=false)."
                            )
                            self._run_stats["scene_rejections"] += 1
                            self._run_stats["missing_pose_rejections"] += 1
                            self._clear_previous_scene_slots()
                            break
                        self.get_logger().error(
                            f"{scene_name}: sin poses de simulación; usando etiquetas por pose "
                            "PLANIFICADA (pueden desalinearse de la imagen)."
                        )
                else:
                    sim_poses_list = None

                self.get_logger().info(
                    f"{scene_name}: baseline_image_seq={baseline_image_seq} "
                    f"baseline_world_pose_seq={baseline_world_pose_seq} "
                    f"baseline_image_stamp_ns={baseline_image_stamp_ns} "
                    f"baseline_world_pose_stamp_ns={baseline_world_pose_stamp_ns} "
                    f"scene_ts={t_scene_ready_ns} "
                    f"pose_ts={pose_snapshot_stamp_ns if pose_snapshot_stamp_ns is not None else -1} "
                    f"image_ts={image_stamp_ns} "
                    f"discarded_frames={discarded_frames} "
                    f"accepted_frames={accepted_frames} "
                    f"stable_frames_required={self._visual_stability_frames}"
                )

                if self._require_fresh_image and image_stamp_ns <= image_threshold_ns:
                    self.get_logger().warning(
                        f"{scene_name}: imagen no fresca para la escena "
                        f"(image_ts={image_stamp_ns} <= threshold_ts={image_threshold_ns}); reintentando."
                    )
                    self._run_stats["scene_rejections"] += 1
                    self._run_stats["stale_image_rejections"] += 1
                    time.sleep(0.12)
                    continue

                if image_stamp_ns <= baseline_image_stamp_ns:
                    self.get_logger().warning(
                        f"{scene_name}: imagen no supera baseline de escena "
                        f"(image_ts={image_stamp_ns} <= baseline_image_stamp_ns={baseline_image_stamp_ns}); "
                        "reintentando."
                    )
                    self._run_stats["scene_rejections"] += 1
                    self._run_stats["stale_image_rejections"] += 1
                    time.sleep(0.12)
                    continue

                if (
                    pose_snapshot_stamp_ns is not None
                    and image_stamp_ns <= pose_snapshot_stamp_ns
                ):
                    self.get_logger().warning(
                        f"{scene_name}: imagen no es posterior al snapshot TF "
                        f"(image_ts={image_stamp_ns} <= pose_snapshot_stamp_ns={pose_snapshot_stamp_ns}); "
                        "reintentando."
                    )
                    self._run_stats["scene_rejections"] += 1
                    self._run_stats["stale_image_rejections"] += 1
                    time.sleep(0.12)
                    continue

                if (
                    use_sim_poses
                    and self._reject_scene_if_sim_pose_off_table
                    and sim_poses_list is not None
                ):
                    bad_entries = [
                        (
                            s.entity_name,
                            [round(float(v), 4) for v in p[:3]],
                            [round(float(s.x_m), 4), round(float(s.y_m), 4), round(float(s.z_m), 4)],
                        )
                        for s, p in zip(scene, sim_poses_list)
                        if not self._sim_pose_matches_expected(s, p)
                    ]
                    bad_names = [entry[0] for entry in bad_entries]
                    if bad_names:
                        bad_debug = ", ".join(
                            f"{name}: sim={sim_xyz} expected={expected_xyz}"
                            for name, sim_xyz, expected_xyz in bad_entries
                        )
                        if attempt + 1 < max_pose_attempts:
                            self.get_logger().warning(
                                f"{scene_name}: centro del modelo fuera de banda mesa "
                                f"(Z/XY) {bad_names}; intento {attempt + 1}/{max_pose_attempts}. "
                                f"Detalles: {bad_debug}"
                            )
                            self._run_stats["scene_rejections"] += 1
                            self._run_stats["sim_pose_rejections"] += 1
                            time.sleep(0.12)
                            continue
                        self.get_logger().error(
                            f"{scene_name}: pose(s) fuera de mesa tras reintentos "
                            f"{bad_names}; escena omitida. Detalles: {bad_debug}"
                        )
                        self._run_stats["scene_rejections"] += 1
                        self._run_stats["sim_pose_rejections"] += 1
                        self._clear_previous_scene_slots()
                        break

                pose_arg = sim_poses_list if use_sim_poses else None
                projected_labels = self._scene_to_projected_labels(
                    scene,
                    self._camera_info,
                    image_height,
                    image_width,
                    sim_poses=pose_arg,
                )
                label_lines = [item.line for item in projected_labels]

                if (
                    self._reject_scene_if_incomplete_labels
                    and len(scene) > 0
                    and len(label_lines) != len(scene)
                ):
                    if attempt + 1 < max_pose_attempts:
                        self.get_logger().warning(
                            f"{scene_name}: etiquetas incompletas "
                            f"{len(label_lines)}/{len(scene)}; reintentando."
                        )
                        self._run_stats["scene_rejections"] += 1
                        self._run_stats["labels_incomplete_rejections"] += 1
                        time.sleep(0.12)
                        continue
                    self.get_logger().error(
                        f"{scene_name}: etiquetas incompletas tras reintentos; escena omitida."
                    )
                    self._run_stats["scene_rejections"] += 1
                    self._run_stats["labels_incomplete_rejections"] += 1
                    self._clear_previous_scene_slots()
                    break

                overlap_messages = self._find_projected_label_overlaps(projected_labels)
                if overlap_messages:
                    if attempt + 1 < max_pose_attempts:
                        self.get_logger().warning(
                            f"{scene_name}: OBB proyectadas con solape excesivo "
                            f"{overlap_messages}; reintentando."
                        )
                        self._run_stats["scene_rejections"] += 1
                        self._run_stats["projected_overlap_rejections"] += 1
                        time.sleep(0.12)
                        continue
                    self.get_logger().error(
                        f"{scene_name}: solape excesivo entre OBB proyectadas tras "
                        f"reintentos; escena omitida. Detalles: {overlap_messages}"
                    )
                    self._run_stats["scene_rejections"] += 1
                    self._run_stats["projected_overlap_rejections"] += 1
                    self._clear_previous_scene_slots()
                    break

                visibility_ok, visibility_problems = (
                    self._validate_rendered_visibility_for_scene(
                        scene, projected_labels, bgr
                    )
                )
                if not visibility_ok:
                    act_notes = (
                        f" activation_verify_notes={self._scene_activation_verify_notes}"
                        if self._scene_activation_verify_notes
                        else ""
                    )
                    if attempt + 1 < max_pose_attempts:
                        self.get_logger().warning(
                            f"{scene_name}: validación de render falló "
                            f"{visibility_problems}; reintentando.{act_notes}"
                        )
                        self._run_stats["scene_rejections"] += 1
                        self._run_stats["render_visibility_rejections"] += 1
                        time.sleep(0.12)
                        continue
                    self.get_logger().error(
                        f"{scene_name}: entidades no visibles/renderizadas tras reintentos: "
                        f"{visibility_problems}. Escena omitida.{act_notes}"
                    )
                    self._run_stats["scene_rejections"] += 1
                    self._run_stats["render_visibility_rejections"] += 1
                    self._clear_previous_scene_slots()
                    break

                image_hash = self._image_md5(bgr)
                pose_signature = (
                    _pose_signature(sim_poses_list)
                    if sim_poses_list is not None
                    else json.dumps(
                        [
                            {
                                "entity_name": spawned.entity_name,
                                "x_m": round(float(spawned.x_m), 5),
                                "y_m": round(float(spawned.y_m), 5),
                                "z_m": round(float(spawned.z_m), 5),
                                "yaw_rad": round(float(spawned.yaw_rad), 5),
                                "roll_rad": round(float(spawned.roll_rad), 5),
                                "pitch_rad": round(float(spawned.pitch_rad), 5),
                            }
                            for spawned in scene
                        ],
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                )
                if (
                    self._reject_if_same_image
                    and self._previous_saved_image_hash is not None
                    and image_hash == self._previous_saved_image_hash
                    and pose_signature != self._previous_saved_pose_signature
                ):
                    self.get_logger().warning(
                        f"{scene_name}: imagen duplicada con poses distintas; escena descartada "
                        "para evitar desincronización imagen-etiquetas."
                    )
                    self._run_stats["scene_rejections"] += 1
                    self._run_stats["duplicate_image_pose_mismatch_rejections"] += 1
                    time.sleep(0.12)
                    continue

                image_path = self._output_dir / "images" / split / f"{scene_name}.png"
                label_path = self._output_dir / "labels" / split / f"{scene_name}.txt"
                cv2.imwrite(str(image_path), bgr)
                label_path.write_text(
                    "\n".join(label_lines) + ("\n" if label_lines else ""),
                    encoding="utf-8",
                )
                self._save_calibration_artifacts(
                    scene_name=scene_name,
                    bgr=bgr,
                    scene=scene,
                    projected_labels=projected_labels,
                    sim_poses=sim_poses_list if use_sim_poses else None,
                    split=split,
                )

                if (
                    not self._reject_scene_if_incomplete_labels
                    and len(scene) > 0
                    and len(label_lines) < len(scene)
                ):
                    self.get_logger().warn(
                        f"{scene_name}: {len(label_lines)}/{len(scene)} etiquetas válidas "
                        "(proyección fuera de imagen o z<=0 en cámara)."
                    )
                if (
                    not self._reject_scene_if_incomplete_labels
                    and len(scene) > 0
                    and len(label_lines) == 0
                ):
                    self.get_logger().warn(
                        f"{scene_name}: imagen sin etiquetas OBB pese a {len(scene)} spawns: "
                        "revisa Z de modelos (spawn_z_offset_m), cámara o settle_time."
                    )

                if split == "train":
                    train_count += 1
                    self._run_stats["saved_train"] = train_count
                else:
                    val_count += 1
                    self._run_stats["saved_val"] = val_count

                if use_sim_poses:
                    label_src = (
                        "ros_tf_pose"
                        if self._labels_sim_pose_source == "ros_tf"
                        else "gz_cli_pose"
                    )
                else:
                    label_src = "planned_pose"
                self.get_logger().info(
                    f"Generated {scene_name} ({split}) with "
                    f"{len(scene)} objects, {len(label_lines)} labels ({label_src})."
                )
                self._previous_saved_image_hash = image_hash
                self._previous_saved_pose_signature = pose_signature
                scene_saved = True
                self._write_run_meta()
                break

            if not scene_saved:
                continue

            if self._calibration_mode:
                break

        self._clear_previous_scene_slots()
        self._write_run_meta()
        self.get_logger().info(
            f"Dataset generation complete. train={train_count}, val={val_count}, "
            f"output={self._output_dir}"
        )


def main(args: Optional[List[str]] = None) -> None:
    rclpy.init(args=args)
    node = YCBDatasetGenerator()
    try:
        node.generate()
    finally:
        node.destroy_node()
        rclpy.shutdown()
