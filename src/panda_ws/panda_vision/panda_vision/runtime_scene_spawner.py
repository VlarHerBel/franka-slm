#!/usr/bin/env python3
"""Runtime random YCB scene spawner for Gazebo."""

from __future__ import annotations

import math
import os
import random
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import rclpy
import yaml
from ament_index_python.packages import get_package_share_directory
from rclpy.clock import Clock, ClockType
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from ros_gz_interfaces.msg import Entity as GzEntity
from ros_gz_interfaces.srv import (
    DeleteEntity as GzDeleteEntity,
    SpawnEntity as GzSpawnEntity,
)
from std_srvs.srv import Trigger

from panda_vision.spawn.gazebo_spawn_pose_readback import (
    declare_spawn_pose_readback_params,
    readback_params_from_node,
    settle_and_build_gt_entry,
)
from panda_vision.spawn.runtime_scene_gt import (
    RuntimeSceneGtClient,
    make_gt_object_entry,
)
from panda_vision.spawn.known_object_geometry import ROLE_OBSTACLE, ROLE_TARGET
from panda_vision.spawn.runtime_scene_gt_geometry import (
    is_known_spawn_geometry_box_label,
    resolve_semantic_and_gazebo_poses,
    semantic_center_z_world,
)
from panda_vision.spawn.ycb_runtime_model_assets import (
    DEFAULT_RUNTIME_MODELS_ROOT,
    prepare_runtime_spawn_model,
)
from panda_vision.spawn.semantic_spawn_sampling import (
    TableSpawnRegion,
    is_known_box_label_for_semantic_sampling,
    placement_radius_for_box,
    resolve_box_dims_lwh,
    sample_semantic_box_pose_xyyaw,
)
from panda_vision.spawn.demo_scene_yaml_spawn import (
    demo_scene_yaml_path,
    load_demo_scene_deposit_spawn_entries,
    load_demo_scene_spawn_entries,
)
from panda_vision.spawn.gz_spawn_runtime import clear_runtime_ycb_entities
from panda_vision.spawn.chips_mustard_random_layout import (
    is_chips_mustard_random_scene,
    sample_chips_mustard_random_spawn_entries,
)
from panda_vision.spawn.pair_scene_random_layout import (
    is_random_spawn_scene,
    pair_labels_from_scene_yaml,
    sample_pair_random_spawn_entries,
    scene_random_seed_from_yaml,
)
from panda_vision.spawn.demo_scene_presets import (
    DEMO_SCENE_LAYOUT_FOOTPRINT_SCALE,
    DEMO_SCENE_PRESETS,
    DemoSceneObjectPose,
    DemoScenePreset,
    demo_scene_to_legacy_spec,
    get_demo_scene_preset,
    is_demo_scene_preset,
    log_demo_scene_preset_validation,
    validate_demo_scene_preset,
)

DEFAULT_EXCLUDED_RANDOM_SPAWN_OBJECTS = frozenset({"tomato_soup_can"})

# Posiciones semánticas (x, y, yaw) en el marco de la mesa de trabajo.
_SCENE_PRESET_SPECS: Dict[str, List[Dict[str, object]]] = {
    "two_boxes_easy": [
        {"label": "cracker_box", "x": 0.50, "y": -0.06, "yaw": 0.0},
        {"label": "sugar_box", "x": 0.62, "y": 0.08, "yaw": 0.0},
    ],
    "four_objects_easy": [
        {"label": "cracker_box", "x": 0.48, "y": -0.10, "yaw": 0.0},
        {"label": "sugar_box", "x": 0.62, "y": -0.10, "yaw": 0.0},
        {"label": "pudding_box", "x": 0.48, "y": 0.10, "yaw": 0.0},
        {"label": "gelatin_box", "x": 0.62, "y": 0.10, "yaw": 0.0},
    ],
    "four_objects_demo": [
        {"label": "cracker_box", "x": 0.46, "y": -0.12, "yaw": 0.0},
        {"label": "sugar_box", "x": 0.64, "y": -0.12, "yaw": 0.0},
        {"label": "pudding_box", "x": 0.46, "y": 0.12, "yaw": 0.0},
        {"label": "gelatin_box", "x": 0.64, "y": 0.12, "yaw": 0.0},
    ],
    "catalog_photo_12": [
        {"label": "cracker_box", "x": 0.4000, "y": 0.1300, "yaw": 0.0},
        {"label": "chips_can", "x": 0.5600, "y": 0.1300, "yaw": 0.0},
        {"label": "mustard_bottle", "x": 0.7200, "y": 0.1300, "yaw": 0.0},
        {"label": "bleach_cleanser", "x": 0.8800, "y": 0.1300, "yaw": 0.0},
        {"label": "sugar_box", "x": 0.4000, "y": -0.0100, "yaw": 0.0},
        {"label": "potted_meat_can", "x": 0.5600, "y": -0.0100, "yaw": 0.0},
        {"label": "master_chef_can", "x": 0.7277, "y": -0.0124, "yaw": 0.0},
        {"label": "pudding_box", "x": 0.8800, "y": -0.0100, "yaw": 0.0},
        {"label": "banana", "x": 0.4000, "y": -0.1500, "yaw": 1.5708},
        {"label": "apple", "x": 0.5600, "y": -0.1500, "yaw": 0.0},
        {"label": "tuna_fish_can", "x": 0.7200, "y": -0.1500, "yaw": 0.0},
        {"label": "gelatin_box", "x": 0.8800, "y": -0.1500, "yaw": 0.0},
    ],
}

for _demo_id, _demo_preset in DEMO_SCENE_PRESETS.items():
    _SCENE_PRESET_SPECS[_demo_id] = demo_scene_to_legacy_spec(_demo_preset)


@dataclass(frozen=True)
class YCBClass:
    name: str
    model_name: str
    spawn_height_m: float
    spawn_z_offset_m: float
    footprint_width_m: float
    footprint_length_m: float
    height_m: float


@dataclass(frozen=True)
class SpawnPlan:
    entity_name: str
    cls: YCBClass
    x: float
    y: float
    z: float
    yaw: float
    radius: float


class RuntimeSceneSpawner(Node):
    def __init__(self) -> None:
        super().__init__("runtime_scene_spawner")
        self._cb_group = ReentrantCallbackGroup()

        pkg_share = Path(get_package_share_directory("panda_vision"))
        default_config = pkg_share / "config" / "ycb_obb_dataset.yaml"
        default_models = Path.home() / "tfg_robotics_ws" / "src" / "gazebo_ycb" / "models"

        self.declare_parameter("config_path", str(default_config))
        self.declare_parameter("ycb_models_path", str(default_models))
        self.declare_parameter("world_name", "vision_test_ycb")
        self.declare_parameter("table_surface_z_m", 0.26)
        self.declare_parameter("spawn_z_epsilon_m", 0.001)
        self.declare_parameter("spawn_x_min", 0.42)
        self.declare_parameter("spawn_x_max", 0.72)
        self.declare_parameter("spawn_y_min", -0.22)
        self.declare_parameter("spawn_y_max", 0.22)
        self.declare_parameter("min_objects", 1)
        self.declare_parameter("max_objects", 2)
        self.declare_parameter("min_surface_gap_m", 0.03)
        self.declare_parameter("scene_layout_max_attempts", 1000)
        self.declare_parameter("scene_combination_attempts", 100)
        self.declare_parameter("footprint_safety_scale", 1.1)
        self.declare_parameter("allow_fallback_to_one_object", True)
        self.declare_parameter("spawn_timeout_sec", 8.0)
        self.declare_parameter("delete_timeout_sec", 10.0)
        self.declare_parameter("spawn_backend", "ros_gz_create_cli")
        self.declare_parameter("delete_backend", "gz_service_cli")
        self.declare_parameter("allow_spawn_without_clear", False)
        self.declare_parameter("clear_all_runtime_ycb_on_spawn", True)
        self.declare_parameter("pose_discovery_sec", 2.0)
        self.declare_parameter("texture_unique_cache", True)
        self.declare_parameter("texture_cache_dir", "")
        self.declare_parameter(
            "runtime_models_root", str(DEFAULT_RUNTIME_MODELS_ROOT)
        )
        self.declare_parameter(
            "excluded_spawn_objects",
            list(DEFAULT_EXCLUDED_RANDOM_SPAWN_OBJECTS),
        )
        self.declare_parameter("min_table_edge_margin_m", 0.03)
        self.declare_parameter("random_spawn_safe_region", False)
        self.declare_parameter("scene_target_label", "")
        self.declare_parameter("scene_preset", "")
        self.declare_parameter("scene_random_seed", 0)
        self.declare_parameter("demo_scene_min_clearance_m", 0.03)
        self.declare_parameter("spawn_scene_on_startup", False)
        self.declare_parameter("spawn_scene_on_startup_delay_sec", 5.0)
        self.declare_parameter("allowed_spawn_labels", "")
        self.declare_parameter("allow_duplicate_labels", False)
        self.declare_parameter("world_pose_ros_topic", "")
        declare_spawn_pose_readback_params(self)

        self._config_path = Path(str(self.get_parameter("config_path").value)).expanduser()
        self._ycb_models_path = Path(str(self.get_parameter("ycb_models_path").value)).expanduser()
        self._world_name = str(self.get_parameter("world_name").value)
        self._gz_world_name = (
            self._world_name if self._world_name.endswith("_world") else f"{self._world_name}_world"
        )
        self._table_surface_z = float(self.get_parameter("table_surface_z_m").value)
        self._spawn_z_epsilon = float(self.get_parameter("spawn_z_epsilon_m").value)
        self._spawn_x_min = float(self.get_parameter("spawn_x_min").value)
        self._spawn_x_max = float(self.get_parameter("spawn_x_max").value)
        self._spawn_y_min = float(self.get_parameter("spawn_y_min").value)
        self._spawn_y_max = float(self.get_parameter("spawn_y_max").value)
        self._min_objects = int(self.get_parameter("min_objects").value)
        self._max_objects = int(self.get_parameter("max_objects").value)
        self._min_surface_gap = float(self.get_parameter("min_surface_gap_m").value)
        self._layout_attempts = int(self.get_parameter("scene_layout_max_attempts").value)
        self._scene_combination_attempts = int(
            self.get_parameter("scene_combination_attempts").value
        )
        self._footprint_safety_scale = float(
            self.get_parameter("footprint_safety_scale").value
        )
        self._allow_fallback_to_one_object = bool(
            self.get_parameter("allow_fallback_to_one_object").value
        )
        self._spawn_timeout = float(self.get_parameter("spawn_timeout_sec").value)
        self._delete_timeout = float(self.get_parameter("delete_timeout_sec").value)
        self._spawn_backend = str(self.get_parameter("spawn_backend").value)
        self._delete_backend = str(self.get_parameter("delete_backend").value)
        self._allow_spawn_without_clear = bool(
            self.get_parameter("allow_spawn_without_clear").value
        )
        self._texture_unique_cache = bool(
            self.get_parameter("texture_unique_cache").value
        )
        self._runtime_models_root = self._resolve_runtime_models_root_param()
        self._runtime_models_root.mkdir(parents=True, exist_ok=True)

        self._min_table_edge_margin = float(
            self.get_parameter("min_table_edge_margin_m").value
        )
        self._random_spawn_safe_region = bool(
            self.get_parameter("random_spawn_safe_region").value
        )
        self._scene_target_label = str(
            self.get_parameter("scene_target_label").value
        ).strip().lower()
        self._excluded_spawn_names = self._parse_excluded_spawn_objects()
        self._allowed_spawn_labels = self._parse_allowed_spawn_labels()
        self._allow_duplicate_labels = bool(
            self.get_parameter("allow_duplicate_labels").value
        )
        self._scene_preset = str(self.get_parameter("scene_preset").value).strip()
        self._scene_random_seed = int(self.get_parameter("scene_random_seed").value)
        self._demo_scene_min_clearance = float(
            self.get_parameter("demo_scene_min_clearance_m").value
        )
        self._world_pose_ros_topic = self._resolve_world_pose_topic()
        self._classes = self._load_classes(self._config_path)
        self._spawned_entities: List[str] = []
        self._startup_spawn_done = False
        self._startup_spawn_timer = None
        self._spawn_lock = threading.Lock()
        self._pose_bridge_hint_logged = False
        self._gt_client = RuntimeSceneGtClient(self, world_frame="world")

        self._spawn_client = self.create_client(
            GzSpawnEntity,
            f"/world/{self._gz_world_name}/create",
            callback_group=self._cb_group,
        )
        self._delete_client = self.create_client(
            GzDeleteEntity,
            f"/world/{self._gz_world_name}/remove",
            callback_group=self._cb_group,
        )

        self.create_service(
            Trigger,
            "/runtime_scene/spawn_random_scene",
            self._spawn_random_scene_cb,
            callback_group=self._cb_group,
        )
        self.create_service(
            Trigger,
            "/runtime_scene/clear_scene",
            self._clear_scene_cb,
            callback_group=self._cb_group,
        )
        self.get_logger().info(
            f"runtime_scene_spawner listo. world={self._gz_world_name}, clases={len(self._classes)}"
        )
        self.get_logger().info(f"Usando spawn_backend={self._spawn_backend}")
        self.get_logger().info(f"Usando delete_backend={self._delete_backend}")
        self.get_logger().info(
            f"texture_unique_cache {'activo' if self._texture_unique_cache else 'inactivo'}"
        )
        self.get_logger().info(
            f"runtime_models_root usado: {self._runtime_models_root}"
        )
        self.get_logger().info(
            "excluded_spawn_objects (spawn aleatorio): %s"
            % sorted(self._excluded_spawn_names)
        )
        preset_raw = str(self.get_parameter("scene_preset").value).strip()
        self.get_logger().info(
            "[STARTUP_SCENE_SPAWN] scene_preset='%s' spawn_on_startup=%s"
            % (
                preset_raw or "(vacío → aleatorio)",
                self._should_spawn_scene_on_startup(),
            )
        )
        self._maybe_schedule_startup_scene_spawn()

    def _should_spawn_scene_on_startup(self) -> bool:
        explicit = bool(self.get_parameter("spawn_scene_on_startup").value)
        preset_active = bool((self._scene_preset or "").strip())
        return explicit or preset_active

    def _maybe_schedule_startup_scene_spawn(self) -> None:
        if not self._should_spawn_scene_on_startup():
            return
        delay = max(0.5, float(self.get_parameter("spawn_scene_on_startup_delay_sec").value))
        preset = (self._scene_preset or "").strip() or "random"
        self.get_logger().info(
            "[STARTUP_SCENE_SPAWN] scheduled preset=%s delay_sec=%.1f "
            "(reloj de pared; use_sim_time no retrasa este timer)"
            % (preset, delay)
        )
        # Reloj de sistema: con use_sim_time=True un create_timer normal no dispara
        # hasta que /clock avance (Gazebo puede arrancar después que este nodo).
        self._startup_spawn_timer = self.create_timer(
            delay,
            self._startup_spawn_cb,
            callback_group=self._cb_group,
            clock=Clock(clock_type=ClockType.SYSTEM_TIME),
        )

    def _startup_spawn_cb(self) -> None:
        if self._startup_spawn_done:
            return
        self._startup_spawn_done = True
        if self._startup_spawn_timer is not None:
            self._startup_spawn_timer.cancel()
            self._startup_spawn_timer = None
        self._refresh_runtime_parameters()
        preset = (self._scene_preset or "").strip() or "random"
        self.get_logger().info(
            "[STARTUP_SCENE_SPAWN] begin preset=%s" % preset
        )
        try:
            ok, msg = self.spawn_random_scene()
            level = self.get_logger().info if ok else self.get_logger().error
            level(
                "[STARTUP_SCENE_SPAWN] result=%s preset=%s msg=%s"
                % ("ok" if ok else "fail", preset, msg)
            )
        except Exception as exc:
            self.get_logger().error(
                "[STARTUP_SCENE_SPAWN] result=fail preset=%s error=%s"
                % (preset, exc)
            )

    def _resolve_world_pose_topic(self) -> str:
        raw = str(self.get_parameter("world_pose_ros_topic").value).strip()
        if raw:
            return raw
        return f"/world/{self._gz_world_name}/pose/info"

    def _parse_excluded_spawn_objects(self) -> Set[str]:
        raw = self.get_parameter("excluded_spawn_objects").value
        if isinstance(raw, (list, tuple)):
            return {str(x).strip() for x in raw if str(x).strip()}
        if isinstance(raw, str) and raw.strip():
            return {p.strip() for p in raw.split(",") if p.strip()}
        return set(DEFAULT_EXCLUDED_RANDOM_SPAWN_OBJECTS)

    def _parse_allowed_spawn_labels(self) -> Optional[Set[str]]:
        raw = self.get_parameter("allowed_spawn_labels").value
        if isinstance(raw, (list, tuple)):
            labels = {str(x).strip().lower() for x in raw if str(x).strip()}
        elif isinstance(raw, str) and raw.strip():
            labels = {p.strip().lower() for p in raw.split(",") if p.strip()}
        else:
            return None
        return labels if labels else None

    def _class_by_name(self, name: str) -> Optional[YCBClass]:
        key = str(name).strip().lower()
        for cls in self._classes:
            if cls.name.lower() == key:
                return cls
        return None

    def _refresh_runtime_parameters(self) -> None:
        """Refresh runtime-tunable parameters before each spawn request."""
        self._table_surface_z = float(self.get_parameter("table_surface_z_m").value)
        self._spawn_z_epsilon = float(self.get_parameter("spawn_z_epsilon_m").value)
        self._spawn_x_min = float(self.get_parameter("spawn_x_min").value)
        self._spawn_x_max = float(self.get_parameter("spawn_x_max").value)
        self._spawn_y_min = float(self.get_parameter("spawn_y_min").value)
        self._spawn_y_max = float(self.get_parameter("spawn_y_max").value)
        self._min_objects = int(self.get_parameter("min_objects").value)
        self._max_objects = int(self.get_parameter("max_objects").value)
        self._min_surface_gap = float(self.get_parameter("min_surface_gap_m").value)
        self._layout_attempts = int(self.get_parameter("scene_layout_max_attempts").value)
        self._scene_combination_attempts = int(
            self.get_parameter("scene_combination_attempts").value
        )
        self._footprint_safety_scale = float(
            self.get_parameter("footprint_safety_scale").value
        )
        self._min_table_edge_margin = float(
            self.get_parameter("min_table_edge_margin_m").value
        )
        self._random_spawn_safe_region = bool(
            self.get_parameter("random_spawn_safe_region").value
        )
        self._scene_target_label = str(
            self.get_parameter("scene_target_label").value
        ).strip().lower()
        self._allow_fallback_to_one_object = bool(
            self.get_parameter("allow_spawn_without_clear").value
        )
        self._spawn_backend = str(self.get_parameter("spawn_backend").value)
        self._delete_backend = str(self.get_parameter("delete_backend").value)
        self._allow_spawn_without_clear = bool(
            self.get_parameter("allow_spawn_without_clear").value
        )
        self._texture_unique_cache = bool(
            self.get_parameter("texture_unique_cache").value
        )
        self._runtime_models_root = self._resolve_runtime_models_root_param()
        self._runtime_models_root.mkdir(parents=True, exist_ok=True)
        self._excluded_spawn_names = self._parse_excluded_spawn_objects()
        self._allowed_spawn_labels = self._parse_allowed_spawn_labels()
        self._allow_duplicate_labels = bool(
            self.get_parameter("allow_duplicate_labels").value
        )
        self._scene_preset = str(self.get_parameter("scene_preset").value).strip()
        self._scene_random_seed = int(self.get_parameter("scene_random_seed").value)
        self._demo_scene_min_clearance = float(
            self.get_parameter("demo_scene_min_clearance_m").value
        )
        self._world_pose_ros_topic = self._resolve_world_pose_topic()

    def _resolve_runtime_models_root_param(self) -> Path:
        texture_cache_dir = str(self.get_parameter("texture_cache_dir").value).strip()
        runtime_models_root = str(
            self.get_parameter("runtime_models_root").value
        ).strip()
        if texture_cache_dir:
            return Path(texture_cache_dir).expanduser().resolve()
        if runtime_models_root:
            return Path(runtime_models_root).expanduser().resolve()
        return DEFAULT_RUNTIME_MODELS_ROOT.resolve()

    def _class_radius(self, cls: YCBClass) -> float:
        radius = 0.5 * max(cls.footprint_width_m, cls.footprint_length_m)
        return radius * max(1.0, self._footprint_safety_scale)

    def _filter_feasible_classes(self) -> List[YCBClass]:
        x_span = self._spawn_x_max - self._spawn_x_min
        y_span = self._spawn_y_max - self._spawn_y_min
        feasible: List[YCBClass] = []
        for cls in self._classes:
            radius = self._class_radius(cls)
            if (2.0 * radius) > x_span or (2.0 * radius) > y_span:
                self.get_logger().warning(
                    f"Clase descartada por área de spawn: {cls.name} "
                    f"(radius={radius:.3f}, x_span={x_span:.3f}, y_span={y_span:.3f})"
                )
                continue
            feasible.append(cls)
        return feasible

    def _load_classes(self, path: Path) -> List[YCBClass]:
        with path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
        out: List[YCBClass] = []
        for raw in data.get("classes", []):
            if not bool(raw.get("enabled", True)):
                continue
            width = float(raw.get("footprint_width_m", raw.get("width_m", 0.06)))
            length = float(raw.get("footprint_length_m", raw.get("length_m", 0.06)))
            spawn_h = float(raw.get("spawn_height_m", 0.04))
            out.append(
                YCBClass(
                    name=str(raw["name"]),
                    model_name=str(raw.get("model_name", raw["name"])),
                    spawn_height_m=spawn_h,
                    spawn_z_offset_m=float(raw.get("spawn_z_offset_m", 0.0)),
                    footprint_width_m=max(width, 0.02),
                    footprint_length_m=max(length, 0.02),
                    height_m=float(raw.get("height_m", spawn_h * 2.0)),
                )
            )
        if not out:
            raise RuntimeError(f"Sin clases en {path}")
        return out

    def _resolve_spawn_model_sdf(self, label: str, model_name: str) -> Path:
        original = self._ycb_models_path / model_name / "model.sdf"
        if not self._texture_unique_cache:
            return original
        source_dir = self._ycb_models_path / model_name
        try:
            sdf_path, _, _ = prepare_runtime_spawn_model(
                label,
                source_dir,
                runtime_models_root=self._runtime_models_root,
                logger=self.get_logger(),
            )
            return sdf_path
        except Exception as exc:
            self.get_logger().warning(
                f"texture_unique_cache falló para {label} ({model_name}): {exc}; "
                "usando original"
            )
            return original

    def _wait_for_service(self, client, service_name: str, timeout_sec: float) -> None:
        self.get_logger().info(f"Esperando servicio Gazebo: {service_name}")
        if client.wait_for_service(timeout_sec=timeout_sec):
            self.get_logger().info(f"Servicio disponible: {service_name}")
            return
        raise TimeoutError(f"Servicio Gazebo no disponible: {service_name}")

    def _call_service(self, client, request, timeout_sec: float):
        future = client.call_async(request)
        end = time.monotonic() + timeout_sec
        while time.monotonic() < end and rclpy.ok():
            if future.done():
                return future.result()
            time.sleep(0.05)
        raise TimeoutError("Timeout esperando servicio Gazebo.")

    @staticmethod
    def _entity_quaternion(yaw_rad: float) -> Tuple[float, float, float, float]:
        # Rotación solo en Z (sin tf_transformations: evita transforms3d + np.float en NumPy>=1.24).
        half = yaw_rad * 0.5
        return 0.0, 0.0, float(math.sin(half)), float(math.cos(half))

    def _table_spawn_region(self) -> TableSpawnRegion:
        return TableSpawnRegion(
            x_min=self._spawn_x_min,
            x_max=self._spawn_x_max,
            y_min=self._spawn_y_min,
            y_max=self._spawn_y_max,
            margin_m=self._min_table_edge_margin,
            random_spawn_safe_region=self._random_spawn_safe_region,
        )

    def _sample_pose(self, footprint_radius: float) -> Tuple[float, float]:
        x = random.uniform(self._spawn_x_min + footprint_radius, self._spawn_x_max - footprint_radius)
        y = random.uniform(self._spawn_y_min + footprint_radius, self._spawn_y_max - footprint_radius)
        return x, y

    def _spawn_positions(
        self, selected_classes: List[YCBClass]
    ) -> List[Tuple[float, float, float]]:
        """Centros semánticos (x, y, yaw) válidos en el tablero."""
        placed: List[Tuple[float, float, float, float]] = []
        out: List[Tuple[float, float, float]] = []
        region = self._table_spawn_region()

        for cls in selected_classes:
            done = False
            if is_known_box_label_for_semantic_sampling(cls.name):
                length_m, width_m = resolve_box_dims_lwh(
                    cls.name,
                    footprint_length_m=cls.footprint_length_m,
                    footprint_width_m=cls.footprint_width_m,
                )
                radius = placement_radius_for_box(
                    length_m, width_m, self._footprint_safety_scale
                )
                for _ in range(self._layout_attempts):
                    try:
                        cx, cy, yaw = sample_semantic_box_pose_xyyaw(
                            random,
                            cls.name,
                            region,
                            footprint_length_m=cls.footprint_length_m,
                            footprint_width_m=cls.footprint_width_m,
                            logger=self.get_logger(),
                        )
                    except RuntimeError:
                        continue
                    if all(
                        math.hypot(cx - px, cy - py) >= (radius + pr + self._min_surface_gap)
                        for px, py, _, pr in placed
                    ):
                        placed.append((cx, cy, yaw, radius))
                        out.append((cx, cy, yaw))
                        done = True
                        break
            else:
                radius = self._class_radius(cls)
                for _ in range(self._layout_attempts):
                    x, y = self._sample_pose(radius)
                    yaw = random.uniform(-math.pi, math.pi)
                    if all(
                        math.hypot(x - px, y - py) >= (radius + pr + self._min_surface_gap)
                        for px, py, _, pr in placed
                    ):
                        placed.append((x, y, yaw, radius))
                        out.append((x, y, yaw))
                        done = True
                        break
            if not done:
                self.get_logger().warning(
                    "Layout fallido para combinación: "
                    f"classes={[c.name for c in selected_classes]}, "
                    f"radii={[round(self._class_radius(c), 4) for c in selected_classes]}, "
                    f"spawn_area=({self._spawn_x_min:.3f},{self._spawn_x_max:.3f})x"
                    f"({self._spawn_y_min:.3f},{self._spawn_y_max:.3f}), "
                    f"attempts_per_object={self._layout_attempts}"
                )
                raise RuntimeError("No se pudo generar layout sin colisiones de spawn.")
        return out

    def _build_spawn_plan(
        self,
        selected_classes: List[YCBClass],
        positions: List[Tuple[float, float, float]],
    ) -> List[SpawnPlan]:
        plan: List[SpawnPlan] = []
        timestamp = int(time.time() * 1000) % 1000000
        for idx, (cls, (x, y, yaw)) in enumerate(zip(selected_classes, positions)):
            entity_name = f"runtime_ycb_{cls.name}_{idx}_{timestamp}"
            z = self._table_surface_z + cls.spawn_height_m + cls.spawn_z_offset_m + self._spawn_z_epsilon
            if is_known_spawn_geometry_box_label(cls.name):
                length_m, width_m = resolve_box_dims_lwh(
                    cls.name,
                    footprint_length_m=cls.footprint_length_m,
                    footprint_width_m=cls.footprint_width_m,
                )
                radius = placement_radius_for_box(
                    length_m, width_m, self._footprint_safety_scale
                )
            else:
                radius = self._class_radius(cls)
            plan.append(
                SpawnPlan(
                    entity_name=entity_name,
                    cls=cls,
                    x=x,
                    y=y,
                    z=z,
                    yaw=yaw,
                    radius=radius,
                )
            )
        return plan

    def _filter_spawnable_classes(self, feasible_classes: List[YCBClass]) -> List[YCBClass]:
        excluded = self._excluded_spawn_names
        spawnable = [
            c
            for c in feasible_classes
            if c.name not in excluded and c.model_name not in excluded
        ]
        if self._allowed_spawn_labels is not None:
            spawnable = [
                c for c in spawnable if c.name.lower() in self._allowed_spawn_labels
            ]
            self.get_logger().info(
                "[SPAWNER_ALLOWED_LABELS] labels=%s"
                % sorted(self._allowed_spawn_labels)
            )
        return spawnable

    def _sample_spawn_classes(self, spawnable: List[YCBClass], count: int) -> List[YCBClass]:
        count = min(count, len(spawnable))
        if self._allow_duplicate_labels:
            return random.choices(spawnable, k=count)
        if count > len(spawnable):
            raise RuntimeError(
                "allow_duplicate_labels=false pero se pidieron %d objetos con solo %d labels."
                % (count, len(spawnable))
            )
        return random.sample(spawnable, k=count)

    def _generate_preset_scene_plan(self, preset_name: str) -> Tuple[List[SpawnPlan], bool]:
        if (
            preset_name not in _SCENE_PRESET_SPECS
            and not is_random_spawn_scene(preset_name)
            and demo_scene_yaml_path(preset_name) is None
        ):
            raise ValueError(f"scene_preset desconocido: '{preset_name}'")
        spec = _SCENE_PRESET_SPECS.get(preset_name) or []
        self.get_logger().info("[SCENE_PRESET] name=%s" % preset_name)
        yaml_spec = load_demo_scene_spawn_entries(preset_name)
        if is_random_spawn_scene(preset_name):
            labels = pair_labels_from_scene_yaml(preset_name)
            if len(labels) != 2:
                raise RuntimeError(
                    "spawn_mode=random requiere pick_order con 2 objetos en %s"
                    % preset_name
                )
            seed = int(self._scene_random_seed)
            if seed <= 0:
                seed = int(scene_random_seed_from_yaml(preset_name))
            if seed <= 0:
                seed = int(time.time() * 1000) % 2_147_483_647
            rng = random.Random(seed)
            self.get_logger().info(
                "[PAIR_RANDOM_SPAWN] scene_id=%s labels=%s random_seed=%d"
                % (preset_name, list(labels), seed)
            )
            if is_chips_mustard_random_scene(preset_name):
                spec = sample_chips_mustard_random_spawn_entries(
                    rng,
                    tuple(labels),
                    region=self._table_spawn_region(),
                    min_clearance_m=self._demo_scene_min_clearance,
                    random_seed=seed,
                    max_attempts=self._layout_attempts,
                    logger=self.get_logger(),
                )
            else:
                spec = sample_pair_random_spawn_entries(
                    rng,
                    labels,
                    region=self._table_spawn_region(),
                    min_clearance_m=self._demo_scene_min_clearance,
                    random_seed=seed,
                    max_attempts=self._layout_attempts,
                    logger=self.get_logger(),
                    log_tag="%s_RANDOM_LAYOUT" % preset_name.upper(),
                )
            demo_poses = tuple(
                DemoSceneObjectPose(
                    str(entry["label"]),
                    float(entry["x"]),
                    float(entry["y"]),
                    float(entry["yaw"]),
                    order_index=int(entry.get("order_index", 0)),
                )
                for entry in spec
            )
            ok, reason = validate_demo_scene_preset(
                DemoScenePreset(scene_id=preset_name, objects=demo_poses),
                min_clearance_m=self._demo_scene_min_clearance,
                footprint_safety_scale=DEMO_SCENE_LAYOUT_FOOTPRINT_SCALE,
            )
            if not ok:
                raise RuntimeError(
                    "Layout aleatorio '%s' no pasó validación: %s"
                    % (preset_name, reason)
                )
        elif yaml_spec is not None:
            self.get_logger().info(
                "[DEMO_SCENE_POLICY_LOAD]\n"
                "scene_id=%s\n"
                "spawn_entries=%d\n"
                "result=OK"
                % (preset_name, len(yaml_spec))
            )
            spec = yaml_spec
        elif is_demo_scene_preset(preset_name):
            demo_preset = get_demo_scene_preset(preset_name)
            ok, reason = log_demo_scene_preset_validation(
                self.get_logger(),
                demo_preset,
                min_clearance_m=self._demo_scene_min_clearance,
                footprint_safety_scale=DEMO_SCENE_LAYOUT_FOOTPRINT_SCALE,
            )
            if not ok:
                raise RuntimeError(
                    "Preset demo '%s' no pasó validación (clearance=%.3f scale=%.2f): %s"
                    % (
                        preset_name,
                        self._demo_scene_min_clearance,
                        DEMO_SCENE_LAYOUT_FOOTPRINT_SCALE,
                        reason,
                    )
                )
            spec = demo_scene_to_legacy_spec(demo_preset)
        selected: List[YCBClass] = []
        positions: List[Tuple[float, float, float]] = []
        for entry in spec:
            label = str(entry["label"]).strip().lower()
            cls = self._class_by_name(label)
            if cls is None:
                raise RuntimeError(f"Preset requiere clase no cargada: {label}")
            if (
                self._allowed_spawn_labels is not None
                and label not in self._allowed_spawn_labels
            ):
                raise RuntimeError(
                    f"Preset label '{label}' no está en allowed_spawn_labels."
                )
            x = float(entry["x"])
            y = float(entry["y"])
            yaw = float(entry.get("yaw", 0.0))
            selected.append(cls)
            positions.append((x, y, yaw))
            seed_raw = entry.get("seed")
            seed_s = (
                " seed=%s" % int(seed_raw) if seed_raw is not None else ""
            )
            self.get_logger().info(
                "[SCENE_OBJECT] label=%s semantic_center=(%.3f, %.3f) yaw=%.3f%s"
                % (label, x, y, yaw, seed_s)
            )
        self.get_logger().info(
            "[SPAWNER_SELECTED_LABELS] labels=%s"
            % [c.name for c in selected]
        )
        plan = self._build_spawn_plan(selected, positions)
        deposit_spec = load_demo_scene_deposit_spawn_entries(preset_name)
        if deposit_spec:
            self.get_logger().info(
                "[DEPOSIT_PRELOAD_SPAWN]\n"
                "scene_id=%s\n"
                "deposit_objects=%d\n"
                "result=OK"
                % (preset_name, len(deposit_spec))
            )
            dep_selected: List[YCBClass] = []
            dep_positions: List[Tuple[float, float, float]] = []
            dep_z: List[float] = []
            for entry in deposit_spec:
                label = str(entry["label"]).strip().lower()
                cls = self._class_by_name(label)
                if cls is None:
                    raise RuntimeError(
                        "Deposit preload requiere clase no cargada: %s" % label
                    )
                dep_selected.append(cls)
                dep_positions.append(
                    (float(entry["x"]), float(entry["y"]), float(entry.get("yaw", 0.0)))
                )
                dep_z.append(float(entry.get("z", 0.22)))
                self.get_logger().info(
                    "[DEPOSIT_PRELOAD_OBJECT] label=%s slot=%s xy=(%.3f, %.3f)"
                    % (
                        label,
                        str(entry.get("deposit_slot_index", "n/a")),
                        float(entry["x"]),
                        float(entry["y"]),
                    )
                )
            dep_plan = self._build_spawn_plan(dep_selected, dep_positions)
            for i, sp in enumerate(dep_plan):
                z_val = dep_z[i] if i < len(dep_z) else sp.z
                plan.append(
                    SpawnPlan(
                        entity_name=sp.entity_name,
                        cls=sp.cls,
                        x=sp.x,
                        y=sp.y,
                        z=float(z_val),
                        yaw=sp.yaw,
                        radius=sp.radius,
                    )
                )
        return plan, False

    def _generate_scene_plan(self) -> Tuple[List[SpawnPlan], bool]:
        preset = (self._scene_preset or "").strip()
        if preset:
            return self._generate_preset_scene_plan(preset)

        feasible_classes = self._filter_feasible_classes()
        if not feasible_classes:
            raise RuntimeError("No hay clases viables para el área de spawn.")

        before = [f"{c.name} (model={c.model_name})" for c in feasible_classes]
        self.get_logger().info(
            "Random spawn candidates before filtering: %s" % before
        )
        self.get_logger().info(
            "Excluded spawn objects: %s" % sorted(self._excluded_spawn_names)
        )
        spawnable = self._filter_spawnable_classes(feasible_classes)
        after = [f"{c.name} (model={c.model_name})" for c in spawnable]
        self.get_logger().info(
            "Random spawn candidates after filtering: %s" % after
        )
        if not spawnable:
            raise RuntimeError(
                "No spawnable YCB objects available after applying filters."
            )

        max_count = max(self._min_objects, self._max_objects)
        max_count = min(max_count, len(spawnable))
        if not self._allow_duplicate_labels:
            max_count = min(max_count, len(spawnable))
        min_count = max(1, min(self._min_objects, max_count))
        counts = list(range(max_count, min_count - 1, -1))
        if self._allow_fallback_to_one_object and 1 not in counts:
            counts.append(1)

        for count in counts:
            attempts = max(1, self._scene_combination_attempts)
            for _ in range(attempts):
                try:
                    selected = self._sample_spawn_classes(spawnable, count)
                except RuntimeError:
                    continue
                self.get_logger().info(
                    "[SPAWNER_SELECTED_LABELS] labels=%s"
                    % [c.name for c in selected]
                )
                try:
                    positions = self._spawn_positions(selected)
                except RuntimeError:
                    continue
                plan = self._build_spawn_plan(selected, positions)
                self.get_logger().info(
                    "Selected object(s): %s"
                    % [f"{item.cls.name} (model={item.cls.model_name})" for item in plan]
                )
                self.get_logger().info(f"Layout generado con {len(plan)} objetos.")
                for item in plan:
                    self.get_logger().info(
                        f"plan: class={item.cls.name}, x={item.x:.3f}, y={item.y:.3f}, "
                        f"z={item.z:.3f}, yaw={item.yaw:.3f}, radius={item.radius:.3f}"
                    )
                used_fallback = len(plan) < max(self._min_objects, self._max_objects)
                return plan, used_fallback
            self.get_logger().warning(
                f"No se encontró layout para {count} objetos tras {attempts} combinaciones."
            )
        raise RuntimeError("No se pudo generar layout sin colisiones de spawn.")

    def _gazebo_spawn_xyz(
        self, cls: YCBClass, center_xy: Tuple[float, float], yaw: float
    ) -> Tuple[float, float, float, Tuple[float, float, float]]:
        """(gazebo_x, gazebo_y, gazebo_z, semantic_center_xyz)."""
        if is_known_spawn_geometry_box_label(cls.name):
            semantic, gazebo = resolve_semantic_and_gazebo_poses(
                cls.name,
                center_xy,
                yaw,
                self._table_surface_z,
                cls.height_m,
                epsilon_m=self._spawn_z_epsilon,
                logger=self.get_logger(),
            )
            return float(gazebo[0]), float(gazebo[1]), float(gazebo[2]), semantic
        z = (
            self._table_surface_z
            + cls.spawn_height_m
            + cls.spawn_z_offset_m
            + self._spawn_z_epsilon
        )
        sem_z = semantic_center_z_world(
            self._table_surface_z, cls.height_m, epsilon_m=self._spawn_z_epsilon
        )
        return float(center_xy[0]), float(center_xy[1]), float(z), (
            float(center_xy[0]),
            float(center_xy[1]),
            float(sem_z),
        )

    def _spawn_entity_service(self, entity_name: str, cls: YCBClass, x: float, y: float, yaw: float) -> None:
        model_sdf = self._resolve_spawn_model_sdf(cls.name, cls.model_name)
        if not model_sdf.is_file():
            raise FileNotFoundError(f"model.sdf no encontrado: {model_sdf}")
        gx, gy, gz, _ = self._gazebo_spawn_xyz(cls, (x, y), yaw)
        req = GzSpawnEntity.Request()
        req.entity_factory.name = entity_name
        req.entity_factory.allow_renaming = False
        req.entity_factory.sdf_filename = str(model_sdf)
        req.entity_factory.pose.position.x = float(gx)
        req.entity_factory.pose.position.y = float(gy)
        req.entity_factory.pose.position.z = float(gz)
        qx, qy, qz, qw = self._entity_quaternion(yaw)
        req.entity_factory.pose.orientation.x = qx
        req.entity_factory.pose.orientation.y = qy
        req.entity_factory.pose.orientation.z = qz
        req.entity_factory.pose.orientation.w = qw
        req.entity_factory.relative_to = "world"
        resp = self._call_service(self._spawn_client, req, self._spawn_timeout)
        if not resp or not resp.success:
            raise RuntimeError(f"Falló spawn de {entity_name}")

    def _spawn_subprocess_env(self) -> dict:
        """Herencia de entorno con rutas Gazebo para modelos runtime/YCB."""
        env = os.environ.copy()
        extra = [
            str(self._ycb_models_path),
            str(self._runtime_models_root),
        ]
        for key in ("GZ_SIM_RESOURCE_PATH", "IGN_GAZEBO_RESOURCE_PATH"):
            cur = env.get(key, "")
            parts = [p for p in extra + ([cur] if cur else []) if p]
            env[key] = os.pathsep.join(dict.fromkeys(parts))
        return env

    def _spawn_entity_cli(self, entity_name: str, cls: YCBClass, x: float, y: float, yaw: float) -> None:
        model_sdf = self._resolve_spawn_model_sdf(cls.name, cls.model_name)
        if not model_sdf.is_file():
            raise FileNotFoundError(f"model.sdf no encontrado: {model_sdf}")
        gx, gy, gz, _ = self._gazebo_spawn_xyz(cls, (x, y), yaw)
        cmd = [
            "ros2",
            "run",
            "ros_gz_sim",
            "create",
            "-world",
            self._gz_world_name,
            "-file",
            str(model_sdf),
            "-name",
            entity_name,
            "-x",
            f"{gx:.6f}",
            "-y",
            f"{gy:.6f}",
            "-z",
            f"{gz:.6f}",
            "-Y",
            f"{yaw:.6f}",
        ]
        self.get_logger().info(f"Spawneando con ros_gz_sim create: {entity_name}")
        completed = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False,
            env=self._spawn_subprocess_env(),
        )
        if completed.returncode != 0:
            stdout = (completed.stdout or "").strip()
            stderr = (completed.stderr or "").strip()
            if stdout:
                self.get_logger().warning(f"[{entity_name}] stdout: {stdout}")
            if stderr:
                self.get_logger().warning(f"[{entity_name}] stderr: {stderr}")
            raise RuntimeError(f"Falló spawn CLI de {entity_name} (exit_code={completed.returncode})")

    def _spawn_entity(self, entity_name: str, cls: YCBClass, x: float, y: float, yaw: float) -> None:
        if self._spawn_backend == "ros_gz_create_cli":
            self._spawn_entity_cli(entity_name, cls, x, y, yaw)
            return
        self._spawn_entity_service(entity_name, cls, x, y, yaw)

    def _gt_entry_after_spawn(self, item: SpawnPlan, *, role: str) -> Dict[str, object]:
        gx, gy, gz, _ = self._gazebo_spawn_xyz(item.cls, (item.x, item.y), item.yaw)
        readback_params = readback_params_from_node(self)
        entry = settle_and_build_gt_entry(
            self,
            entity_name=item.entity_name,
            label=item.cls.name,
            requested_gazebo_xyz=(float(gx), float(gy), float(gz)),
            requested_yaw_rad=float(item.yaw),
            width_m=float(item.cls.footprint_width_m),
            length_m=float(item.cls.footprint_length_m),
            height_m=float(item.cls.height_m),
            world_pose_topic=self._world_pose_ros_topic,
            params=readback_params,
            role=role,
            logger=self.get_logger(),
        )
        if entry is not None:
            return entry
        self.get_logger().warning(
            "GT desde pose Gazebo no disponible para %s; se usa pose comandada"
            % item.entity_name
        )
        cy = math.cos(float(item.yaw) * 0.5)
        sy = math.sin(float(item.yaw) * 0.5)
        if is_known_spawn_geometry_box_label(item.cls.name):
            sem_z = semantic_center_z_world(
                self._table_surface_z,
                item.cls.height_m,
                epsilon_m=self._spawn_z_epsilon,
            )
            gt_x, gt_y, gt_z = float(item.x), float(item.y), float(sem_z)
        else:
            gt_x, gt_y, gt_z = float(item.x), float(item.y), float(item.z)
        return make_gt_object_entry(
            entity_name=item.entity_name,
            label=item.cls.name,
            logger=self.get_logger(),
            x=float(gt_x),
            y=float(gt_y),
            z=float(gt_z),
            roll=0.0,
            pitch=0.0,
            yaw=float(item.yaw),
            qx=0.0,
            qy=0.0,
            qz=float(sy),
            qw=float(cy),
            width_m=float(item.cls.footprint_width_m),
            length_m=float(item.cls.footprint_length_m),
            height_m=float(item.cls.height_m),
            role=role,
        )

    def _delete_entity(self, entity_name: str) -> None:
        if self._delete_backend == "gz_service_cli":
            self._delete_entity_cli(entity_name)
            return
        if self._delete_backend != "service":
            raise ValueError(
                "delete_backend inválido. Valores válidos: service, gz_service_cli"
            )
        req = GzDeleteEntity.Request()
        req.entity.name = entity_name
        req.entity.type = GzEntity.MODEL
        resp = self._call_service(self._delete_client, req, self._delete_timeout)
        if resp and not resp.success:
            raise RuntimeError(f"No se pudo borrar {entity_name} con backend service.")

    def _delete_entity_cli(self, entity_name: str) -> None:
        binary = "ign" if shutil.which("ign") else ("gz" if shutil.which("gz") else None)
        if binary is None:
            raise RuntimeError("No se encontró ni 'ign' ni 'gz' para borrado CLI.")
        reqtype = "ignition.msgs.Entity" if binary == "ign" else "gz.msgs.Entity"
        reptype = "ignition.msgs.Boolean" if binary == "ign" else "gz.msgs.Boolean"
        cmd = [
            binary,
            "service",
            "-s",
            f"/world/{self._gz_world_name}/remove",
            "--reqtype",
            reqtype,
            "--reptype",
            reptype,
            "--timeout",
            "5000",
            "--req",
            f'name: "{entity_name}" type: 2',
        ]
        self.get_logger().info(f"Delete CLI cmd: {' '.join(cmd)}")
        completed = subprocess.run(
            cmd, capture_output=True, text=True, timeout=8.0, check=False
        )
        self.get_logger().info(f"Delete CLI returncode={completed.returncode}")
        self.get_logger().info(f"Delete CLI stdout: {(completed.stdout or '').strip()}")
        self.get_logger().info(f"Delete CLI stderr: {(completed.stderr or '').strip()}")
        if completed.returncode != 0:
            raise RuntimeError(
                f"Falló borrado CLI de {entity_name} (exit_code={completed.returncode})"
            )
        lower_stdout = (completed.stdout or "").lower()
        if "data: true" not in lower_stdout and "true" not in lower_stdout:
            raise RuntimeError(f"Borrado CLI no confirmado para {entity_name}")

    def clear_scene(self) -> Tuple[bool, str]:
        """Elimina todas las entidades runtime_ycb del mundo (no solo las rastreadas)."""
        clear_all = bool(self.get_parameter("clear_all_runtime_ycb_on_spawn").value)
        if clear_all:
            pose_topic = self._resolve_world_pose_topic()
            pose_discovery = float(self.get_parameter("pose_discovery_sec").value)
            ok, remaining = clear_runtime_ycb_entities(
                self,
                gz_world_name=self._gz_world_name,
                pose_topic=pose_topic,
                pose_discovery_sec=pose_discovery,
                spawn_name_prefix="runtime_ycb",
                delete_backend=self._delete_backend,
                delete_timeout_sec=self._delete_timeout,
                delete_retries=2,
                verify_after_delete=True,
                list_only=False,
                log_prefix="[SPAWNER_CLEAR]",
            )
            self._spawned_entities = []
            self._gt_client.clear()
            if not ok and remaining:
                msg = "Quedan entidades runtime sin borrar: %s" % ", ".join(remaining)
                if not self._allow_spawn_without_clear:
                    return False, msg
                self.get_logger().warning("[SPAWNER_CLEAR] %s (allow_spawn_without_clear=true)" % msg)
            return True, "Escena runtime limpiada."

        if not self._spawned_entities:
            return True, "No había entidades runtime para borrar."
        failed = []
        if self._delete_backend == "service":
            self._wait_for_service(
                self._delete_client,
                f"/world/{self._gz_world_name}/remove",
                self._delete_timeout,
            )
        for entity_name in list(self._spawned_entities):
            if not entity_name.startswith("runtime_ycb_"):
                self.get_logger().warning(
                    f"Se omite entidad no runtime: {entity_name}"
                )
                continue
            try:
                self._delete_entity(entity_name)
            except Exception as exc:
                self.get_logger().warning(f"No se pudo borrar {entity_name}: {exc}")
                failed.append(entity_name)
        if failed:
            self._spawned_entities = failed
            return False, f"No se pudieron borrar: {', '.join(failed)}"
        self._spawned_entities = []
        self._gt_client.clear()
        return True, "Escena runtime limpiada."

    def _warn_pose_bridge_if_needed(self) -> None:
        if self._pose_bridge_hint_logged:
            return
        params = readback_params_from_node(self)
        if not params.update_runtime_scene_from_actual_gazebo_pose:
            return
        topic = self._world_pose_ros_topic
        try:
            pub_count = int(self.count_publishers(topic))
        except Exception:
            pub_count = 0
        if pub_count > 0:
            return
        self.get_logger().warning(
            "[GAZEBO_POSE_BRIDGE] topic=%s sin publicadores ROS. "
            "El spawn en Gazebo puede funcionar igual; GT usará pose comandada. "
            "Usa el launch completo (bridge_world_pose_info:=true) o "
            "-p update_runtime_scene_from_actual_gazebo_pose:=false si solo pruebas ros2 run."
            % topic
        )
        self._pose_bridge_hint_logged = True

    def spawn_random_scene(self) -> Tuple[bool, str]:
        if not self._spawn_lock.acquire(blocking=False):
            return False, "Spawn ya en curso."
        try:
            return self._spawn_random_scene_impl()
        finally:
            self._spawn_lock.release()

    def _spawn_random_scene_impl(self) -> Tuple[bool, str]:
        self._refresh_runtime_parameters()
        if self._spawn_backend == "service":
            self._wait_for_service(
                self._spawn_client,
                f"/world/{self._gz_world_name}/create",
                self._spawn_timeout,
            )
            self._wait_for_service(
                self._delete_client,
                f"/world/{self._gz_world_name}/remove",
                self._delete_timeout,
            )
        elif self._spawn_backend != "ros_gz_create_cli":
            raise ValueError(
                "spawn_backend inválido. Valores válidos: service, ros_gz_create_cli"
            )

        plan, used_fallback = self._generate_scene_plan()
        self._warn_pose_bridge_if_needed()

        ok, _ = self.clear_scene()
        if not ok and not self._allow_spawn_without_clear:
            return False, "No se pudo limpiar escena previa."

        created = []
        gt_entries = []
        target_assigned = False
        for item in plan:
            self._spawn_entity(item.entity_name, item.cls, item.x, item.y, item.yaw)
            created.append(item.entity_name)
            lbl_l = str(item.cls.name).strip().lower()
            if self._scene_target_label and lbl_l == self._scene_target_label:
                role = ROLE_TARGET
                target_assigned = True
            elif not self._scene_target_label and not target_assigned:
                role = ROLE_TARGET
                target_assigned = True
            else:
                role = ROLE_OBSTACLE
            gt_entries.append(self._gt_entry_after_spawn(item, role=role))

        self._spawned_entities = created
        self._gt_client.replace_all(gt_entries)
        if used_fallback:
            return True, f"Escena creada con {len(created)} objeto(s) tras fallback."
        return True, f"Escena creada con {len(created)} objetos."

    def _spawn_random_scene_cb(self, _req, res):
        try:
            ok, msg = self.spawn_random_scene()
            res.success = ok
            res.message = msg
        except Exception as exc:
            res.success = False
            res.message = str(exc)
        return res

    def _clear_scene_cb(self, _req, res):
        try:
            ok, msg = self.clear_scene()
            res.success = ok
            res.message = msg
        except Exception as exc:
            res.success = False
            res.message = str(exc)
        return res


def main(args=None) -> None:
    rclpy.init(args=args)
    node = RuntimeSceneSpawner()
    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        executor.remove_node(node)
        node.destroy_node()
        try:
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    main()
