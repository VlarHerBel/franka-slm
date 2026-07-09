#!/usr/bin/env python3
"""Scanner top-down de alcanzabilidad para objetos demo (sin tocar demo_scene_02).

Barrido en malla (x, y, yaw) sobre la mesa; valida IK/plan/FK/descenso cartesiano
usando las políticas reales de agarre por objeto.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import yaml

from panda_controller.attached_transport_phases import joint_limit_margin_min
from panda_controller.demo_cracker_box_cartesian_prevalidate import (
    vertical_descend_volume_clear_of_obstacles,
)
from panda_controller.sugar_box_yaw_free import build_yaw_free_candidate_yaws
from panda_controller.tcp_hand_pose_convert import hand_pose_from_desired_tcp
from panda_controller.tfg_motion_waypoints import (
    PANDA_ARM_JOINT_NAMES,
    get_waypoint_joint_positions,
    joint_values_7d_from_any,
    load_waypoints_file,
    resolve_waypoints_yaml_path,
)

DEMO_SCAN_LABELS: Tuple[str, ...] = (
    "cracker_box",
    "chips_can",
    "sugar_box",
    "mustard_bottle",
)

CSV_FIELDS: Tuple[str, ...] = (
    "label",
    "x",
    "y",
    "yaw",
    "top_z",
    "grasp_tcp_x",
    "grasp_tcp_y",
    "grasp_tcp_z",
    "pregrasp_tcp_x",
    "pregrasp_tcp_y",
    "pregrasp_tcp_z",
    "target_hand_x",
    "target_hand_y",
    "target_hand_z",
    "quat_x",
    "quat_y",
    "quat_z",
    "quat_w",
    "target_link",
    "moveit_target_link",
    "use_grasp_tcp",
    "seed_state_name",
    "pregrasp_ok",
    "plan_to_pregrasp_ok",
    "start_tcp_error_m",
    "endpoint_ik_ok",
    "cartesian_fraction",
    "collision_ok",
    "joint_limits_ok",
    "pregrasp_ik_error_code",
    "endpoint_ik_error_code",
    "result",
    "reason",
    "variant_budget",
    "attempts_used",
    "early_stop_used",
    "total_possible_variants",
    "evaluated_variants",
    "cell_fully_reachable",
    "input_mode",
    "spawn_x",
    "spawn_y",
    "operational_grasp_x",
    "operational_grasp_y",
    "sdf_offset_rotated_x",
    "sdf_offset_rotated_y",
    "binary_color",
)

REACHABILITY_INPUT_MODES: Tuple[str, ...] = (
    "spawn_origin",
    "operational_grasp_xy",
)
BINARY_HEATMAP_REACHABLE_COLOR = "#FFD700"
BINARY_HEATMAP_UNREACHABLE_COLOR = "#000000"

SINGLE_CELL_BEST_CSV_FIELDS: Tuple[str, ...] = (
    "label",
    "x",
    "y",
    "yaw",
    "top_z",
    "total_variants",
    "pregrasp_success",
    "plan_success",
    "endpoint_success",
    "ok_count",
    "cell_fully_reachable",
    "best_variant",
    "best_seed",
    "best_result",
    "best_reason",
    "best_pregrasp_tcp_z",
    "best_grasp_tcp_z",
    "best_commanded_tcp_yaw_rad",
    "best_cartesian_fraction",
    "variant_budget",
    "attempts_used",
    "early_stop_used",
    "total_possible_variants",
    "evaluated_variants",
    "input_mode",
    "spawn_x",
    "spawn_y",
    "operational_grasp_x",
    "operational_grasp_y",
    "sdf_offset_rotated_x",
    "sdf_offset_rotated_y",
    "pregrasp_ik_ok",
    "plan_to_pregrasp_ok",
    "endpoint_ik_ok",
    "pregrasp_ik_error_code",
    "endpoint_ik_error_code",
    "cartesian_fraction",
    "binary_color",
    "result",
    "reason",
)

DEBUG_CALIBRATION_CELLS: Dict[str, Tuple[float, float, float]] = {
    "cracker_box": (0.455, 0.115, 2.9155),
    "chips_can": (0.520, -0.095, 1.3953),
    "sugar_box": (0.630, -0.175, -3.0159),
    "mustard_bottle": (0.660, 0.060, 1.6392),
}

GOLDEN_DEBUG_REFERENCES: Dict[str, Dict[str, Any]] = {
    "cracker_box": {
        "top_z": 0.4700,
        "pregrasp_tcp": (0.455, 0.115, 0.5620),
        "grasp_tcp": (0.455, 0.115, 0.4370),
        "commanded_tcp_yaw_rad": 1.344703673205025,
        "depth_from_top_m": 0.0330,
    },
    "chips_can": {
        "top_z": 0.5100,
        "pregrasp_tcp": (0.520, -0.095, 0.5450),
        "grasp_tcp": (0.520, -0.095, 0.4750),
        "commanded_tcp_yaw_rad": math.pi,
        "depth_from_top_m": 0.0350,
    },
}

DEFAULT_TABLE_Z_M = 0.27
DEFAULT_TABLE_CENTER = (0.60, 0.0, 0.24)
DEFAULT_TABLE_SIZE = (0.72, 0.48, 0.04)
DEFAULT_TABLE_FRAME = "panda_link0"
DEFAULT_CARTESIAN_FRACTION_THRESHOLD = 0.95
DEFAULT_JOINT_LIMIT_MARGIN_MIN_RAD = -0.02
PAIRED_PREGRASP_FK_ERROR_THRESHOLD_M = 0.005
SCANNER_GRASP_HAND_TARGET_Z_TOLERANCE_M = 0.001

VARIANT_SEARCH_DEBUG_LABELS = frozenset({"sugar_box", "mustard_bottle"})

SUGAR_BOX_DEBUG_PREGRASP_ABOVE_TOP_M: Tuple[float, ...] = (
    0.055,
    0.070,
    0.085,
    0.100,
    0.120,
)

MUSTARD_BOTTLE_DEBUG_PREGRASP_ABOVE_TOP_M: Tuple[float, ...] = (
    0.030,
    0.050,
    0.070,
    0.085,
    0.100,
)

DEBUG_IK_SEED_LABELS: Tuple[str, ...] = (
    "pick_workspace_ready",
    "home",
    "current_joint_state",
    "joint7_near_zero",
)

DEBUG_IK_WAIT_TIMEOUT_SEC = 5.0
DEBUG_IK_RESULT_DEADLINE_SEC = 8.0

VariantBudgetMode = str  # fast | balanced | exhaustive

VARIANT_BUDGET_CHOICES: Tuple[str, ...] = ("fast", "balanced", "exhaustive")
VARIANT_BUDGET_MAX_VARIANTS: Dict[str, int] = {
    "fast": 12,
    "balanced": 40,
}


@dataclass(frozen=True)
class VariantSpec:
    yaw_key: str
    clearance_m: float
    depth_m: float
    seed: str
    grasp_mode: str = "depth_from_top"


SUGAR_BOX_FAST_VARIANT_SPECS: Tuple[VariantSpec, ...] = (
    VariantSpec("closing_yaw", 0.055, 0.025, "pick_workspace_ready"),
    VariantSpec("closing_yaw", 0.070, 0.028, "pick_workspace_ready"),
    VariantSpec("closing_yaw_pi", 0.055, 0.025, "home"),
    VariantSpec("closing_yaw_pi", 0.070, 0.028, "home"),
    VariantSpec("closing_yaw", 0.085, 0.025, "pick_workspace_ready"),
    VariantSpec("spawn_yaw", 0.055, 0.025, "pick_workspace_ready"),
    VariantSpec("closing_yaw", 0.055, 0.030, "home"),
    VariantSpec("closing_yaw", 0.100, 0.025, "pick_workspace_ready"),
    VariantSpec("yaw_zero", 0.055, 0.025, "pick_workspace_ready"),
    VariantSpec("closing_yaw", 0.055, 0.025, "joint7_near_zero"),
    VariantSpec("closing_yaw", 0.070, 0.025, "current_joint_state"),
    VariantSpec("closing_yaw_pi", 0.070, 0.028, "pick_workspace_ready"),
)

MUSTARD_BOTTLE_FAST_VARIANT_SPECS: Tuple[VariantSpec, ...] = (
    VariantSpec("closing_yaw", 0.030, 0.030, "pick_workspace_ready", "palm_bridge"),
    VariantSpec("closing_yaw", 0.050, 0.035, "pick_workspace_ready", "palm_bridge"),
    VariantSpec("closing_yaw_pi", 0.030, 0.030, "home", "palm_bridge"),
    VariantSpec("closing_yaw_pi", 0.050, 0.035, "home", "palm_bridge"),
    VariantSpec("closing_yaw", 0.030, 0.030, "pick_workspace_ready", "depth_from_top"),
    VariantSpec("closing_yaw", 0.070, 0.030, "pick_workspace_ready", "palm_bridge"),
    VariantSpec("spawn_yaw", 0.030, 0.030, "pick_workspace_ready", "palm_bridge"),
    VariantSpec("closing_yaw", 0.050, 0.030, "home", "palm_bridge"),
    VariantSpec("closing_yaw_pi", 0.030, 0.035, "pick_workspace_ready", "palm_bridge"),
    VariantSpec("closing_yaw", 0.030, 0.035, "joint7_near_zero", "palm_bridge"),
    VariantSpec("closing_yaw", 0.085, 0.030, "pick_workspace_ready", "palm_bridge"),
    VariantSpec("closing_yaw_pi", 0.050, 0.035, "pick_workspace_ready", "palm_bridge"),
)

SUGAR_BOX_BALANCED_EXTRA_VARIANT_SPECS: Tuple[VariantSpec, ...] = (
    VariantSpec("spawn_yaw_pi", 0.055, 0.025, "pick_workspace_ready"),
    VariantSpec("closing_yaw", 0.055, 0.028, "home"),
    VariantSpec("closing_yaw_pi", 0.085, 0.028, "pick_workspace_ready"),
    VariantSpec("yaw_pi_over_2", 0.055, 0.025, "pick_workspace_ready"),
    VariantSpec("yaw_neg_pi_over_2", 0.055, 0.025, "pick_workspace_ready"),
    VariantSpec("closing_yaw", 0.120, 0.025, "pick_workspace_ready"),
    VariantSpec("closing_yaw", 0.070, 0.030, "joint7_near_zero"),
    VariantSpec("spawn_yaw", 0.070, 0.028, "home"),
    VariantSpec("closing_yaw_pi", 0.100, 0.028, "current_joint_state"),
    VariantSpec("yaw_from_workspace_tcp", 0.055, 0.025, "pick_workspace_ready"),
    VariantSpec("closing_yaw", 0.055, 0.035, "pick_workspace_ready"),
    VariantSpec("closing_yaw_pi", 0.070, 0.030, "joint7_near_zero"),
    VariantSpec("spawn_yaw", 0.085, 0.028, "home"),
    VariantSpec("yaw_pi", 0.055, 0.025, "pick_workspace_ready"),
    VariantSpec("closing_yaw", 0.070, 0.035, "home"),
    VariantSpec("closing_yaw_pi", 0.055, 0.030, "current_joint_state"),
    VariantSpec("spawn_yaw_pi", 0.070, 0.028, "pick_workspace_ready"),
    VariantSpec("closing_yaw", 0.085, 0.030, "home"),
    VariantSpec("yaw_zero", 0.070, 0.028, "home"),
    VariantSpec("closing_yaw", 0.055, 0.028, "current_joint_state"),
    VariantSpec("spawn_yaw", 0.100, 0.025, "pick_workspace_ready"),
    VariantSpec("closing_yaw_pi", 0.120, 0.025, "home"),
    VariantSpec("yaw_from_workspace_tcp", 0.070, 0.028, "home"),
    VariantSpec("closing_yaw", 0.100, 0.030, "pick_workspace_ready"),
    VariantSpec("spawn_yaw", 0.055, 0.030, "joint7_near_zero"),
    VariantSpec("closing_yaw_pi", 0.085, 0.025, "home"),
    VariantSpec("yaw_pi_over_2", 0.070, 0.028, "pick_workspace_ready"),
    VariantSpec("yaw_neg_pi_over_2", 0.070, 0.028, "home"),
)

MUSTARD_BOTTLE_BALANCED_EXTRA_VARIANT_SPECS: Tuple[VariantSpec, ...] = (
    VariantSpec("spawn_yaw_pi", 0.030, 0.030, "pick_workspace_ready", "palm_bridge"),
    VariantSpec("closing_yaw", 0.030, 0.035, "home", "palm_bridge"),
    VariantSpec("closing_yaw_pi", 0.070, 0.035, "pick_workspace_ready", "palm_bridge"),
    VariantSpec("mustard_gap_axis_yaw_offset_plus_90", 0.030, 0.030, "pick_workspace_ready", "palm_bridge"),
    VariantSpec("mustard_gap_axis_yaw_offset_minus_90", 0.030, 0.030, "pick_workspace_ready", "palm_bridge"),
    VariantSpec("closing_yaw", 0.100, 0.030, "pick_workspace_ready", "palm_bridge"),
    VariantSpec("spawn_yaw", 0.050, 0.035, "home", "palm_bridge"),
    VariantSpec("closing_yaw_pi", 0.085, 0.030, "home", "palm_bridge"),
    VariantSpec("yaw_from_workspace_tcp", 0.030, 0.030, "pick_workspace_ready", "palm_bridge"),
    VariantSpec("closing_yaw", 0.050, 0.030, "current_joint_state", "palm_bridge"),
    VariantSpec("closing_yaw_pi", 0.030, 0.030, "joint7_near_zero", "palm_bridge"),
    VariantSpec("spawn_yaw", 0.070, 0.035, "pick_workspace_ready", "palm_bridge"),
    VariantSpec("closing_yaw", 0.085, 0.035, "home", "palm_bridge"),
    VariantSpec("yaw_zero", 0.030, 0.030, "pick_workspace_ready", "palm_bridge"),
    VariantSpec("closing_yaw_pi", 0.050, 0.030, "current_joint_state", "palm_bridge"),
    VariantSpec("spawn_yaw_pi", 0.050, 0.035, "pick_workspace_ready", "palm_bridge"),
    VariantSpec("closing_yaw", 0.030, 0.030, "current_joint_state", "palm_bridge"),
    VariantSpec("mustard_gap_axis_yaw_offset_plus_90", 0.050, 0.035, "home", "palm_bridge"),
    VariantSpec("mustard_gap_axis_yaw_offset_minus_90", 0.050, 0.035, "home", "palm_bridge"),
    VariantSpec("yaw_pi_over_2", 0.030, 0.030, "pick_workspace_ready", "palm_bridge"),
    VariantSpec("yaw_neg_pi_over_2", 0.030, 0.030, "pick_workspace_ready", "palm_bridge"),
    VariantSpec("closing_yaw", 0.070, 0.035, "joint7_near_zero", "palm_bridge"),
    VariantSpec("spawn_yaw", 0.085, 0.030, "home", "palm_bridge"),
    VariantSpec("closing_yaw_pi", 0.100, 0.030, "pick_workspace_ready", "palm_bridge"),
    VariantSpec("yaw_from_workspace_tcp", 0.050, 0.035, "home", "palm_bridge"),
    VariantSpec("closing_yaw", 0.050, 0.035, "joint7_near_zero", "palm_bridge"),
    VariantSpec("spawn_yaw_pi", 0.030, 0.035, "home", "palm_bridge"),
    VariantSpec("closing_yaw_pi", 0.070, 0.030, "pick_workspace_ready", "palm_bridge"),
)

_VISION_POLICY_MODULE = "panda_vision.grasp.object_grasp_policy"
_OBJECT_GRASP_POLICY_MOD: Any = None


def _log_policy_import(
    *,
    method: str,
    panda_vision_file: str,
    result: str,
    logger: Any = None,
) -> None:
    msg = (
        "[REACHABILITY_SCAN_POLICY_IMPORT]\n"
        "method=%s\n"
        "panda_vision_file=%s\n"
        "result=%s"
        % (method, panda_vision_file, result)
    )
    if logger is not None:
        logger.info(msg)
    else:
        print(msg, file=sys.stderr)


def _object_grasp_policy_candidates() -> List[Any]:
    from pathlib import Path

    paths: List[Any] = []
    try:
        from ament_index_python.packages import get_package_prefix

        prefix = Path(get_package_prefix("panda_vision"))
        paths.extend(
            prefix.glob(
                "lib/python*/site-packages/panda_vision/grasp/object_grasp_policy.py"
            )
        )
        paths.append(
            prefix / "lib" / "panda_vision" / "grasp" / "object_grasp_policy.py"
        )
    except Exception:
        pass

    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = (
            parent
            / "panda_vision"
            / "panda_vision"
            / "grasp"
            / "object_grasp_policy.py"
        )
        if candidate not in paths:
            paths.append(candidate)
    return paths


def _load_object_grasp_policy_module(*, logger: Any = None) -> Any:
    global _OBJECT_GRASP_POLICY_MOD
    if _OBJECT_GRASP_POLICY_MOD is not None:
        return _OBJECT_GRASP_POLICY_MOD

    import importlib
    import importlib.util

    last_err: Optional[BaseException] = None

    try:
        mod = importlib.import_module(_VISION_POLICY_MODULE)
        mod_file = str(getattr(mod, "__file__", "") or "n/a")
        _OBJECT_GRASP_POLICY_MOD = mod
        _log_policy_import(
            method="python_import",
            panda_vision_file=mod_file,
            result="OK",
            logger=logger,
        )
        return mod
    except Exception as exc:
        last_err = exc

    for mod_path in _object_grasp_policy_candidates():
        if not mod_path.is_file():
            continue
        try:
            spec = importlib.util.spec_from_file_location(_VISION_POLICY_MODULE, mod_path)
            if spec is None or spec.loader is None:
                continue
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            _OBJECT_GRASP_POLICY_MOD = mod
            method = "ament_fallback" if "install" in str(mod_path) else "source_fallback"
            _log_policy_import(
                method=method,
                panda_vision_file=str(mod_path),
                result="OK",
                logger=logger,
            )
            return mod
        except Exception as exc:
            last_err = exc

    _log_policy_import(
        method="failed",
        panda_vision_file="n/a",
        result="FAIL",
        logger=logger,
    )
    raise ImportError(
        "Could not import %s (last error: %s)" % (_VISION_POLICY_MODULE, last_err)
    ) from last_err


def _vision_policy_exports(*, logger: Any = None):
    mod = _load_object_grasp_policy_module(logger=logger)
    return mod.export_grasp_policy_for_executor, mod.get_collision_dimensions, mod.resolve_tall_object_top_z_m


def _grasp_generator():
    from panda_controller.grasp_candidate_generator import (
        GraspCandidate,
        generate_grasp_candidates,
    )

    return GraspCandidate, generate_grasp_candidates


def _palm_bridge():
    from panda_controller.palm_bridge_policy import (
        compute_palm_bridge_grasp_tcp_z,
        resolve_effective_top_z_for_palm_bridge,
    )

    return resolve_effective_top_z_for_palm_bridge, compute_palm_bridge_grasp_tcp_z


def _quaternion_multiply(a, b):
    from panda_controller.math_tf_utils import quaternion_multiply

    return quaternion_multiply(a, b)


def wrap_to_pi(angle: float) -> float:
    a = float(angle)
    while a > math.pi:
        a -= 2.0 * math.pi
    while a < -math.pi:
        a += 2.0 * math.pi
    return a


def downward_yaw_quaternion(yaw_rad: float) -> Tuple[float, float, float, float]:
    base_down = [0.0, 1.0, 0.0, 0.0]
    yaw_q = [0.0, 0.0, math.sin(yaw_rad / 2.0), math.cos(yaw_rad / 2.0)]
    q = _quaternion_multiply(base_down, yaw_q)
    return _as_float_quat(q)


def object_axes_from_spawn_yaw(spawn_yaw: float) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    cy = math.cos(float(spawn_yaw))
    sy = math.sin(float(spawn_yaw))
    major = (cy, sy)
    minor = (-sy, cy)
    return major, minor


def resolve_closing_yaw_rad(
    spawn_yaw: float,
    policy: Dict[str, Any],
) -> float:
    yaw_policy = str(policy.get("yaw_policy", "") or "").strip().lower()
    if yaw_policy == "yaw_free":
        return wrap_to_pi(float(spawn_yaw))
    preferred = str(policy.get("preferred_closing_axis", "short_axis")).lower()
    if preferred in ("short_axis", "perpendicular_to_long_axis"):
        return wrap_to_pi(float(spawn_yaw) + math.pi / 2.0)
    return wrap_to_pi(float(spawn_yaw))


def _object_height_m(policy: Dict[str, Any]) -> Optional[float]:
    _, get_collision_dimensions, _ = _vision_policy_exports()
    for key in ("object_height_m", "db_height_m", "effective_height_m"):
        raw = policy.get(key)
        if raw is not None:
            try:
                h = float(raw)
                if h > 1e-6:
                    return h
            except (TypeError, ValueError):
                continue
    dims = policy.get("dims_lwh") or policy.get("dims")
    if isinstance(dims, (list, tuple)) and len(dims) >= 3:
        try:
            return float(dims[2])
        except (TypeError, ValueError):
            pass
    col = get_collision_dimensions(str(policy.get("label", "")), padding_m=0.0)
    if col and isinstance(col.get("db_dims"), (list, tuple)) and len(col["db_dims"]) >= 3:
        return float(col["db_dims"][2])
    return None


def compute_top_z_m(
    label: str,
    table_z_m: float,
    policy: Dict[str, Any],
) -> Tuple[float, float]:
    _, _, resolve_tall_object_top_z_m = _vision_policy_exports()
    height_m = _object_height_m(policy)
    if height_m is None:
        raise ValueError("missing height for label=%s" % label)
    geometry_center_z = float(table_z_m) + float(height_m) / 2.0
    top_z, _dbg = resolve_tall_object_top_z_m(
        label,
        geometry_center_z,
        height_m=float(height_m),
    )
    return float(top_z), float(geometry_center_z)


def build_detection_for_cell(
    label: str,
    x: float,
    y: float,
    spawn_yaw: float,
    *,
    table_z_m: float,
    policy: Dict[str, Any],
) -> Dict[str, Any]:
    major, minor = object_axes_from_spawn_yaw(spawn_yaw)
    top_z, geometry_center_z = compute_top_z_m(label, table_z_m, policy)
    closing_yaw = resolve_closing_yaw_rad(spawn_yaw, policy)
    height_m = _object_height_m(policy)
    return {
        "label": str(label),
        "position": [float(x), float(y), float(geometry_center_z)],
        "chosen_target_center_base": [float(x), float(y), float(top_z)],
        "top_z_m": float(top_z),
        "closing_yaw_rad": float(closing_yaw),
        "grasp_yaw_rad": float(spawn_yaw),
        "major_axis_xy": [float(major[0]), float(major[1])],
        "minor_axis_xy": [float(minor[0]), float(minor[1])],
        "height_m": float(height_m) if height_m is not None else None,
        "object_height_m": float(height_m) if height_m is not None else None,
    }


@dataclass(frozen=True)
class ReachabilityCellCoordinates:
    label: str
    input_mode: str
    grid_x: float
    grid_y: float
    yaw: float
    spawn_x: float
    spawn_y: float
    operational_grasp_x: float
    operational_grasp_y: float
    sdf_offset_local_xy: Tuple[float, float]
    sdf_offset_rotated_xy: Tuple[float, float]


def default_input_mode_for_label(label: str) -> str:
    """mustard_bottle: heatmap/scan sobre cap center operativo; resto: spawn origin."""
    if str(label or "").strip().lower() == "mustard_bottle":
        return "operational_grasp_xy"
    return "spawn_origin"


def _normalize_reachability_input_mode(mode: str) -> str:
    key = str(mode or "spawn_origin").strip().lower()
    if key not in REACHABILITY_INPUT_MODES:
        raise ValueError(
            "input_mode must be one of %s; got %r"
            % (list(REACHABILITY_INPUT_MODES), mode)
        )
    return key


# mesh_local cap center (gazebo_ycb/mustard_bottle); evita import pesado en tests offline.
MUSTARD_CAP_CENTER_OFFSET_LOCAL_XY: Tuple[float, float] = (0.0240, -0.0049)


def mustard_cap_center_offset_local_xy() -> Tuple[float, float]:
    try:
        from panda_vision.spawn.known_object_geometry import get_known_tall_object_sdf_spec

        spec = get_known_tall_object_sdf_spec("mustard_bottle")
        if spec is not None:
            local = tuple(spec.cap_center_local_m)
            return (float(local[0]), float(local[1]))
    except Exception:
        pass
    return MUSTARD_CAP_CENTER_OFFSET_LOCAL_XY


def rotate_mustard_sdf_offset_xy(
    local_offset_xy: Tuple[float, float],
    yaw_rad: float,
) -> Tuple[float, float]:
    lx, ly = float(local_offset_xy[0]), float(local_offset_xy[1])
    c = math.cos(float(yaw_rad))
    s = math.sin(float(yaw_rad))
    return (lx * c - ly * s, lx * s + ly * c)


def resolve_reachability_cell_coordinates(
    label: str,
    x: float,
    y: float,
    yaw: float,
    *,
    input_mode: str = "spawn_origin",
) -> ReachabilityCellCoordinates:
    """Convierte coordenadas de celda entre spawn origin y cap center operativo."""
    lb = str(label or "").strip().lower()
    mode = _normalize_reachability_input_mode(input_mode)
    gx, gy = float(x), float(y)
    yaw_f = float(yaw)
    if lb != "mustard_bottle":
        return ReachabilityCellCoordinates(
            label=lb,
            input_mode="spawn_origin",
            grid_x=gx,
            grid_y=gy,
            yaw=yaw_f,
            spawn_x=gx,
            spawn_y=gy,
            operational_grasp_x=gx,
            operational_grasp_y=gy,
            sdf_offset_local_xy=(0.0, 0.0),
            sdf_offset_rotated_xy=(0.0, 0.0),
        )
    local_xy = mustard_cap_center_offset_local_xy()
    rot_x, rot_y = rotate_mustard_sdf_offset_xy(local_xy, yaw_f)
    if mode == "operational_grasp_xy":
        spawn_x = gx - rot_x
        spawn_y = gy - rot_y
        op_x, op_y = gx, gy
    else:
        spawn_x, spawn_y = gx, gy
        op_x = gx + rot_x
        op_y = gy + rot_y
    return ReachabilityCellCoordinates(
        label=lb,
        input_mode=mode,
        grid_x=gx,
        grid_y=gy,
        yaw=yaw_f,
        spawn_x=float(spawn_x),
        spawn_y=float(spawn_y),
        operational_grasp_x=float(op_x),
        operational_grasp_y=float(op_y),
        sdf_offset_local_xy=local_xy,
        sdf_offset_rotated_xy=(float(rot_x), float(rot_y)),
    )


def compute_compensated_spawn_xy_from_operational(
    operational_x: float,
    operational_y: float,
    yaw: float,
    *,
    label: str = "mustard_bottle",
) -> Tuple[float, float]:
    coords = resolve_reachability_cell_coordinates(
        label,
        float(operational_x),
        float(operational_y),
        float(yaw),
        input_mode="operational_grasp_xy",
    )
    return (float(coords.spawn_x), float(coords.spawn_y))


def enrich_mustard_detection_for_reachability(
    detection: Dict[str, Any],
    coords: ReachabilityCellCoordinates,
    *,
    table_z_m: float,
) -> Dict[str, Any]:
    """Aplica corrección SDF mesh_local para alinear TCP/cap center con el controller."""
    from panda_vision.spawn.known_object_geometry import (
        MUSTARD_CAP_CENTER_MODE_MESH_LOCAL,
        apply_tall_object_sdf_geometry_correction,
    )

    height_m = detection.get("height_m") or detection.get("object_height_m")
    entry: Dict[str, Any] = {
        "label": "mustard_bottle",
        "source_pose_semantics": "model_link_origin",
        "pose_world": {
            "x": float(coords.spawn_x),
            "y": float(coords.spawn_y),
            "z": float(table_z_m),
            "yaw": float(coords.yaw),
        },
        "yaw_rad": float(coords.yaw),
        "grasp_policy": "tall_object_topdown",
        "height_m": float(height_m) if height_m is not None else None,
        "top_z_m": float(detection["top_z_m"]),
        "mustard_cap_center_mode": MUSTARD_CAP_CENTER_MODE_MESH_LOCAL,
    }
    corrected = apply_tall_object_sdf_geometry_correction(entry)
    cap = corrected.get("grasp_center_base") or corrected.get(
        "gt_top_face_center_world"
    )
    geom = corrected.get("semantic_box_center_world") or corrected.get(
        "gt_geometry_center_world"
    )
    if isinstance(cap, (list, tuple)) and len(cap) >= 3:
        detection["chosen_target_center_base"] = [
            float(cap[0]),
            float(cap[1]),
            float(cap[2]),
        ]
        detection["grasp_center_base"] = list(detection["chosen_target_center_base"])
        detection["top_surface_center_base"] = list(
            detection["chosen_target_center_base"]
        )
    if isinstance(geom, (list, tuple)) and len(geom) >= 3:
        detection["position"] = [float(geom[0]), float(geom[1]), float(geom[2])]
    detection["tall_object_sdf_offset_applied"] = True
    detection["reachability_spawn_xy"] = [float(coords.spawn_x), float(coords.spawn_y)]
    detection["reachability_operational_grasp_xy"] = [
        float(coords.operational_grasp_x),
        float(coords.operational_grasp_y),
    ]
    return detection


def build_detection_for_reachability_cell(
    label: str,
    x: float,
    y: float,
    spawn_yaw: float,
    *,
    table_z_m: float,
    policy: Dict[str, Any],
    input_mode: str = "spawn_origin",
) -> Tuple[Dict[str, Any], ReachabilityCellCoordinates]:
    coords = resolve_reachability_cell_coordinates(
        label,
        x,
        y,
        spawn_yaw,
        input_mode=input_mode,
    )
    detection = build_detection_for_cell(
        label,
        float(coords.spawn_x),
        float(coords.spawn_y),
        spawn_yaw,
        table_z_m=table_z_m,
        policy=policy,
    )
    if coords.label == "mustard_bottle":
        detection = enrich_mustard_detection_for_reachability(
            detection,
            coords,
            table_z_m=float(table_z_m),
        )
    return detection, coords


def apply_palm_bridge_to_candidate(
    candidate: Any,
    *,
    detection: Dict[str, Any],
    policy: Dict[str, Any],
    table_z_m: float,
) -> Any:
    if not bool(policy.get("use_palm_bridge_z_constraint")):
        return candidate
    resolve_effective_top_z_for_palm_bridge, compute_palm_bridge_grasp_tcp_z = _palm_bridge()
    raw_top = float(detection["top_z_m"])
    payload = dict(detection)
    payload.update(
        {
            "top_z_m_payload": raw_top,
            "recommended_grasp_depth_from_top_m": float(candidate.depth_from_top_m),
        }
    )
    effective_top, _src, _meta = resolve_effective_top_z_for_palm_bridge(
        payload,
        raw_top,
        table_z_m=float(table_z_m),
    )
    clearance = float(policy.get("palm_bridge_clearance_above_top_m", 0.003))
    below_hand = float(policy.get("palm_bridge_below_panda_hand_m", 0.063))
    hand_to_tcp_z = float(policy.get("panda_hand_to_grasp_tcp_z_m", 0.100))
    grasp_z, _hand_z, _bridge = compute_palm_bridge_grasp_tcp_z(
        effective_top,
        clearance_m=clearance,
        palm_bridge_below_panda_hand_m=below_hand,
        panda_hand_to_grasp_tcp_z_m=hand_to_tcp_z,
    )
    approach = float(policy.get("approach_distance_min_m", 0.12))
    pre_clr = float(policy.get("pregrasp_clearance_above_top_m", 0.08))
    cx, cy = candidate.tcp_grasp_xyz[0], candidate.tcp_grasp_xyz[1]
    pre_z = max(float(grasp_z) + approach, float(effective_top) + pre_clr)
    min_pre = policy.get("min_pregrasp_tcp_z_m")
    if min_pre is not None:
        pre_z = max(pre_z, float(min_pre))
    candidate.tcp_grasp_xyz = (float(cx), float(cy), float(grasp_z))
    candidate.tcp_pregrasp_xyz = (float(cx), float(cy), float(pre_z))
    return candidate


def iter_grasp_variants_for_cell(
    label: str,
    x: float,
    y: float,
    spawn_yaw: float,
    *,
    table_z_m: float,
    gripper_jaw_axis_offset_rad: float = 0.0,
) -> List[Any]:
    export_grasp_policy_for_executor, _, _ = _vision_policy_exports()
    GraspCandidate, generate_grasp_candidates = _grasp_generator()
    policy = export_grasp_policy_for_executor(label)
    policy["label"] = str(label)
    detection = build_detection_for_cell(
        label,
        x,
        y,
        spawn_yaw,
        table_z_m=table_z_m,
        policy=policy,
    )
    base_candidates = generate_grasp_candidates(
        detection,
        policy,
        gripper_jaw_axis_offset_rad=float(gripper_jaw_axis_offset_rad),
    )
    out: List[Any] = []
    yaw_policy = str(policy.get("yaw_policy", "") or "").lower()
    if yaw_policy == "yaw_free":
        yaw_entries = build_yaw_free_candidate_yaws(float(spawn_yaw))
        seen: set = set()
        for cand in base_candidates:
            for _yname, tcp_yaw in yaw_entries:
                key = (
                    round(cand.tcp_pregrasp_xyz[2], 4),
                    round(cand.tcp_grasp_xyz[2], 4),
                    round(tcp_yaw, 4),
                )
                if key in seen:
                    continue
                seen.add(key)
                clone = GraspCandidate(
                    idx=cand.idx,
                    label=cand.label,
                    strategy=cand.strategy,
                    center_xyz=cand.center_xyz,
                    tcp_grasp_xyz=cand.tcp_grasp_xyz,
                    tcp_pregrasp_xyz=cand.tcp_pregrasp_xyz,
                    tcp_safe_pregrasp_xyz=cand.tcp_safe_pregrasp_xyz,
                    final_tcp_yaw_rad=float(tcp_yaw),
                    desired_closing_yaw_rad=cand.desired_closing_yaw_rad,
                    open_joint_m=cand.open_joint_m,
                    close_joint_m=cand.close_joint_m,
                    depth_from_top_m=cand.depth_from_top_m,
                    center_offset_m=cand.center_offset_m,
                    center_offset_axis=cand.center_offset_axis,
                    yaw_offset_rad=cand.yaw_offset_rad,
                    min_contact_margin_m=cand.min_contact_margin_m,
                    score=cand.score,
                    notes=cand.notes + ";yaw_free=%.3f" % float(tcp_yaw),
                )
                out.append(
                    apply_palm_bridge_to_candidate(
                        clone,
                        detection=detection,
                        policy=policy,
                        table_z_m=table_z_m,
                    )
                )
    else:
        for cand in base_candidates:
            out.append(
                apply_palm_bridge_to_candidate(
                    cand,
                    detection=detection,
                    policy=policy,
                    table_z_m=table_z_m,
                )
            )
    out.sort(key=lambda c: c.score, reverse=True)
    return out


def yaw_from_quaternion_xyzw(quat: Sequence[float]) -> float:
    qx, qy, qz, qw = (float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3]))
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    return wrap_to_pi(math.atan2(siny_cosp, cosy_cosp))


def build_expanded_debug_yaw_entries(
    *,
    spawn_yaw: float,
    closing_yaw: float,
    label: str,
    workspace_tcp_yaw: Optional[float] = None,
) -> List[Tuple[str, float]]:
    """Yaw ampliado para debug sugar_box / mustard_bottle (sin tocar cracker/chips)."""
    entries: List[Tuple[str, float]] = []
    seen: set = set()

    def _add(name: str, yaw_rad: float) -> None:
        y = wrap_to_pi(float(yaw_rad))
        key = round(y, 4)
        if key in seen:
            return
        seen.add(key)
        entries.append((str(name), y))

    _add("closing_yaw", closing_yaw)
    _add("closing_yaw_pi", closing_yaw + math.pi)
    _add("spawn_yaw", spawn_yaw)
    _add("spawn_yaw_pi", spawn_yaw + math.pi)
    for fixed, tag in (
        (0.0, "yaw_zero"),
        (math.pi / 2.0, "yaw_pi_over_2"),
        (-math.pi / 2.0, "yaw_neg_pi_over_2"),
        (math.pi, "yaw_pi"),
    ):
        _add(tag, fixed)

    for yaw_name, yaw_val in build_yaw_free_candidate_yaws(closing_yaw):
        _add("yaw_free_%s" % yaw_name, yaw_val)

    if str(label) == "mustard_bottle":
        _add("mustard_pipeline_top_down_yaw", closing_yaw)
        _add("mustard_pipeline_top_down_yaw_pi", closing_yaw + math.pi)
        _add("mustard_pipeline_spawn_top_down_yaw", spawn_yaw)
        _add("mustard_pipeline_spawn_top_down_yaw_pi", spawn_yaw + math.pi)
        for delta, tag in (
            (math.pi / 2.0, "mustard_gap_axis_yaw_offset_plus_90"),
            (-math.pi / 2.0, "mustard_gap_axis_yaw_offset_minus_90"),
        ):
            _add(tag, closing_yaw + delta)

    if workspace_tcp_yaw is not None:
        _add("yaw_from_workspace_tcp", float(workspace_tcp_yaw))

    return entries


def _debug_pregrasp_clearances_for_label(label: str) -> Tuple[float, ...]:
    if str(label) == "sugar_box":
        return SUGAR_BOX_DEBUG_PREGRASP_ABOVE_TOP_M
    if str(label) == "mustard_bottle":
        return MUSTARD_BOTTLE_DEBUG_PREGRASP_ABOVE_TOP_M
    return ()


def _depth_candidates_from_policy(policy: Dict[str, Any]) -> List[float]:
    depths = policy.get("depth_candidates_from_top_m") or [
        policy.get("recommended_grasp_depth_from_top_m", 0.05)
    ]
    out: List[float] = []
    for d in depths:
        try:
            out.append(float(d))
        except (TypeError, ValueError):
            continue
    if not out:
        out = [0.05]
    return out


def _make_debug_search_candidate(
    *,
    label: str,
    x: float,
    y: float,
    top_z: float,
    grasp_z: float,
    pre_z: float,
    tcp_yaw: float,
    depth: float,
    notes: str,
    strategy: str = "debug_variant_search",
) -> Any:
    GraspCandidate, _ = _grasp_generator()
    cx, cy = float(x), float(y)
    return GraspCandidate(
        idx=-1,
        label=str(label),
        strategy=strategy,
        center_xyz=(cx, cy, float(top_z)),
        tcp_grasp_xyz=(cx, cy, float(grasp_z)),
        tcp_pregrasp_xyz=(cx, cy, float(pre_z)),
        tcp_safe_pregrasp_xyz=(cx, cy, float(pre_z) + 0.05),
        final_tcp_yaw_rad=float(tcp_yaw),
        desired_closing_yaw_rad=float(tcp_yaw),
        open_joint_m=0.04,
        close_joint_m=0.027,
        depth_from_top_m=float(depth),
        center_offset_m=0.0,
        center_offset_axis="major_axis",
        yaw_offset_rad=0.0,
        min_contact_margin_m=0.003,
        score=0.0,
        notes=str(notes),
    )


def iter_expanded_debug_variant_jobs(
    label: str,
    x: float,
    y: float,
    spawn_yaw: float,
    *,
    table_z_m: float,
    policy: Dict[str, Any],
    detection: Dict[str, Any],
    ik_seeds: Dict[str, List[float]],
    workspace_tcp_yaw: Optional[float] = None,
) -> List[Tuple[Any, str, List[float]]]:
    """Jobs (candidate, seed_name, seed_joints) para búsqueda debug sugar/mustard."""
    if str(label) not in VARIANT_SEARCH_DEBUG_LABELS:
        return []

    top_z = float(detection["top_z_m"])
    closing_yaw = resolve_closing_yaw_rad(spawn_yaw, policy)
    yaw_entries = build_expanded_debug_yaw_entries(
        spawn_yaw=float(spawn_yaw),
        closing_yaw=float(closing_yaw),
        label=str(label),
        workspace_tcp_yaw=workspace_tcp_yaw,
    )
    pregrasp_clearances = _debug_pregrasp_clearances_for_label(label)
    depths = _depth_candidates_from_policy(policy)

    jobs: List[Tuple[Any, str, List[float]]] = []
    seen_geom: set = set()

    for yaw_name, tcp_yaw in yaw_entries:
        for depth in depths:
            grasp_z_nominal = float(top_z) - float(depth)
            grasp_z_palm = grasp_z_nominal
            if bool(policy.get("use_palm_bridge_z_constraint")):
                palm_cand = _make_debug_search_candidate(
                    label=label,
                    x=x,
                    y=y,
                    top_z=top_z,
                    grasp_z=grasp_z_nominal,
                    pre_z=grasp_z_nominal + 0.12,
                    tcp_yaw=tcp_yaw,
                    depth=depth,
                    notes="palm_bridge_probe",
                )
                palm_cand = apply_palm_bridge_to_candidate(
                    palm_cand,
                    detection=detection,
                    policy=policy,
                    table_z_m=float(table_z_m),
                )
                grasp_z_palm = float(palm_cand.tcp_grasp_xyz[2])

            for clearance in pregrasp_clearances:
                pre_z = float(top_z) + float(clearance)
                for grasp_mode, grasp_z in (
                    ("depth_from_top", grasp_z_nominal),
                    ("palm_bridge", grasp_z_palm),
                ):
                    if str(label) != "mustard_bottle" and grasp_mode == "palm_bridge":
                        continue
                    if pre_z <= float(grasp_z) + 1e-4:
                        continue
                    geom_key = (
                        round(grasp_z, 4),
                        round(pre_z, 4),
                        round(tcp_yaw, 4),
                        str(grasp_mode),
                    )
                    if geom_key in seen_geom:
                        continue
                    seen_geom.add(geom_key)
                    notes = (
                        "debug_search yaw=%s(%.4f) pregrasp_z=%.4f depth=%.4f "
                        "clearance_above_top=%.4f grasp_mode=%s"
                        % (
                            yaw_name,
                            float(tcp_yaw),
                            float(pre_z),
                            float(depth),
                            float(clearance),
                            grasp_mode,
                        )
                    )
                    cand = _make_debug_search_candidate(
                        label=label,
                        x=x,
                        y=y,
                        top_z=top_z,
                        grasp_z=float(grasp_z),
                        pre_z=float(pre_z),
                        tcp_yaw=float(tcp_yaw),
                        depth=float(depth),
                        notes=notes,
                    )
                    for seed_name in DEBUG_IK_SEED_LABELS:
                        seed_js = ik_seeds.get(seed_name)
                        if seed_js is None:
                            continue
                        jobs.append((cand, str(seed_name), list(seed_js)))

    return jobs


def resolve_variant_spec_yaw(
    yaw_key: str,
    *,
    closing_yaw: float,
    spawn_yaw: float,
    workspace_tcp_yaw: Optional[float] = None,
) -> Optional[float]:
    key = str(yaw_key).strip()
    if key == "closing_yaw":
        return wrap_to_pi(float(closing_yaw))
    if key == "closing_yaw_pi":
        return wrap_to_pi(float(closing_yaw) + math.pi)
    if key == "spawn_yaw":
        return wrap_to_pi(float(spawn_yaw))
    if key == "spawn_yaw_pi":
        return wrap_to_pi(float(spawn_yaw) + math.pi)
    if key == "yaw_zero":
        return 0.0
    if key == "yaw_pi_over_2":
        return math.pi / 2.0
    if key == "yaw_neg_pi_over_2":
        return -math.pi / 2.0
    if key == "yaw_pi":
        return math.pi
    if key == "yaw_from_workspace_tcp":
        if workspace_tcp_yaw is None:
            return None
        return wrap_to_pi(float(workspace_tcp_yaw))
    if key == "mustard_gap_axis_yaw_offset_plus_90":
        return wrap_to_pi(float(closing_yaw) + math.pi / 2.0)
    if key == "mustard_gap_axis_yaw_offset_minus_90":
        return wrap_to_pi(float(closing_yaw) - math.pi / 2.0)
    return None


def _budget_specs_for_label(label: str, budget: str) -> Tuple[VariantSpec, ...]:
    mode = str(budget).strip().lower()
    if str(label) == "sugar_box":
        if mode == "fast":
            return SUGAR_BOX_FAST_VARIANT_SPECS
        if mode == "balanced":
            return SUGAR_BOX_FAST_VARIANT_SPECS + SUGAR_BOX_BALANCED_EXTRA_VARIANT_SPECS
    if str(label) == "mustard_bottle":
        if mode == "fast":
            return MUSTARD_BOTTLE_FAST_VARIANT_SPECS
        if mode == "balanced":
            return (
                MUSTARD_BOTTLE_FAST_VARIANT_SPECS
                + MUSTARD_BOTTLE_BALANCED_EXTRA_VARIANT_SPECS
            )
    return ()


def _compute_grasp_z_for_spec(
    *,
    label: str,
    x: float,
    y: float,
    top_z: float,
    depth: float,
    tcp_yaw: float,
    grasp_mode: str,
    policy: Dict[str, Any],
    detection: Dict[str, Any],
    table_z_m: float,
) -> Optional[float]:
    grasp_z_nominal = float(top_z) - float(depth)
    if str(grasp_mode) != "palm_bridge":
        return grasp_z_nominal
    if str(label) != "mustard_bottle":
        return grasp_z_nominal
    palm_cand = _make_debug_search_candidate(
        label=label,
        x=x,
        y=y,
        top_z=top_z,
        grasp_z=grasp_z_nominal,
        pre_z=grasp_z_nominal + 0.12,
        tcp_yaw=tcp_yaw,
        depth=depth,
        notes="palm_bridge_probe",
        strategy="budget_variant_search",
    )
    palm_cand = apply_palm_bridge_to_candidate(
        palm_cand,
        detection=detection,
        policy=policy,
        table_z_m=float(table_z_m),
    )
    return float(palm_cand.tcp_grasp_xyz[2])


def build_jobs_from_variant_specs(
    label: str,
    x: float,
    y: float,
    spawn_yaw: float,
    *,
    table_z_m: float,
    policy: Dict[str, Any],
    detection: Dict[str, Any],
    ik_seeds: Dict[str, List[float]],
    specs: Sequence[VariantSpec],
    workspace_tcp_yaw: Optional[float] = None,
    max_variants: Optional[int] = None,
) -> Tuple[List[Tuple[Any, str, List[float]]], int]:
    top_z = float(detection["top_z_m"])
    closing_yaw = resolve_closing_yaw_rad(spawn_yaw, policy)
    jobs: List[Tuple[Any, str, List[float]]] = []
    seen_geom: set = set()

    for spec in specs:
        if max_variants is not None and len(jobs) >= int(max_variants):
            break
        tcp_yaw = resolve_variant_spec_yaw(
            spec.yaw_key,
            closing_yaw=float(closing_yaw),
            spawn_yaw=float(spawn_yaw),
            workspace_tcp_yaw=workspace_tcp_yaw,
        )
        if tcp_yaw is None:
            continue
        seed_js = ik_seeds.get(str(spec.seed))
        if seed_js is None:
            continue
        grasp_z = _compute_grasp_z_for_spec(
            label=label,
            x=x,
            y=y,
            top_z=top_z,
            depth=float(spec.depth_m),
            tcp_yaw=float(tcp_yaw),
            grasp_mode=str(spec.grasp_mode),
            policy=policy,
            detection=detection,
            table_z_m=float(table_z_m),
        )
        if grasp_z is None:
            continue
        pre_z = float(top_z) + float(spec.clearance_m)
        if pre_z <= float(grasp_z) + 1e-4:
            continue
        geom_key = (
            round(float(grasp_z), 4),
            round(float(pre_z), 4),
            round(float(tcp_yaw), 4),
            str(spec.grasp_mode),
            str(spec.seed),
        )
        if geom_key in seen_geom:
            continue
        seen_geom.add(geom_key)
        notes = (
            "budget_search yaw=%s(%.4f) pregrasp_z=%.4f depth=%.4f "
            "clearance_above_top=%.4f grasp_mode=%s"
            % (
                spec.yaw_key,
                float(tcp_yaw),
                float(pre_z),
                float(spec.depth_m),
                float(spec.clearance_m),
                str(spec.grasp_mode),
            )
        )
        cand = _make_debug_search_candidate(
            label=label,
            x=x,
            y=y,
            top_z=top_z,
            grasp_z=float(grasp_z),
            pre_z=float(pre_z),
            tcp_yaw=float(tcp_yaw),
            depth=float(spec.depth_m),
            notes=notes,
            strategy="budget_variant_search",
        )
        jobs.append((cand, str(spec.seed), list(seed_js)))

    return jobs, len(specs)


def iter_budgeted_variant_jobs(
    label: str,
    x: float,
    y: float,
    spawn_yaw: float,
    *,
    table_z_m: float,
    policy: Dict[str, Any],
    detection: Dict[str, Any],
    ik_seeds: Dict[str, List[float]],
    workspace_tcp_yaw: Optional[float] = None,
    budget: str = "fast",
) -> Tuple[List[Tuple[Any, str, List[float]]], Dict[str, Any]]:
    mode = str(budget).strip().lower()
    if mode not in VARIANT_BUDGET_CHOICES:
        raise ValueError(
            "variant budget must be one of %s; got %r"
            % (VARIANT_BUDGET_CHOICES, budget)
        )
    if str(label) not in VARIANT_SEARCH_DEBUG_LABELS:
        return [], {
            "variant_budget": mode,
            "max_variants": 0,
            "total_possible_variants": 0,
            "early_stop": False,
        }

    if mode == "exhaustive":
        jobs = iter_expanded_debug_variant_jobs(
            label,
            x,
            y,
            spawn_yaw,
            table_z_m=float(table_z_m),
            policy=policy,
            detection=detection,
            ik_seeds=ik_seeds,
            workspace_tcp_yaw=workspace_tcp_yaw,
        )
        return jobs, {
            "variant_budget": mode,
            "max_variants": len(jobs),
            "total_possible_variants": len(jobs),
            "early_stop": False,
        }

    specs = _budget_specs_for_label(label, mode)
    max_variants = int(VARIANT_BUDGET_MAX_VARIANTS.get(mode, 12))
    jobs, _spec_count = build_jobs_from_variant_specs(
        label,
        x,
        y,
        spawn_yaw,
        table_z_m=float(table_z_m),
        policy=policy,
        detection=detection,
        ik_seeds=ik_seeds,
        specs=specs,
        workspace_tcp_yaw=workspace_tcp_yaw,
        max_variants=max_variants,
    )
    return jobs, {
        "variant_budget": mode,
        "max_variants": max_variants,
        "total_possible_variants": len(jobs),
        "early_stop": True,
    }


def format_variant_budget_log(
    *,
    label: str,
    mode: str,
    max_variants: int,
    early_stop: bool,
) -> str:
    return (
        "[REACHABILITY_VARIANT_BUDGET]\n"
        "label=%s\n"
        "mode=%s\n"
        "max_variants=%d\n"
        "early_stop=%s"
        % (str(label), str(mode), int(max_variants), str(bool(early_stop)).lower())
    )


def format_variant_early_stop_log(
    *,
    label: str,
    variant_notes: str,
    attempts_used: int,
    result: str,
) -> str:
    return (
        "[REACHABILITY_VARIANT_SEARCH_EARLY_STOP]\n"
        "label=%s\n"
        "variant=%s\n"
        "attempts_used=%d\n"
        "result=%s"
        % (str(label), str(variant_notes), int(attempts_used), str(result))
    )


def frange(start: float, stop: float, step: float) -> List[float]:
    if step <= 0.0:
        raise ValueError("step must be positive")
    vals: List[float] = []
    v = float(start)
    stop_f = float(stop)
    while v <= stop_f + 1e-9:
        vals.append(round(v, 6))
        v += float(step)
    return vals


def parse_float_list(raw: str) -> List[float]:
    out: List[float] = []
    for part in str(raw or "").split(","):
        part = part.strip()
        if not part:
            continue
        out.append(float(part))
    return out


def _as_float_list(values: Sequence[Any]) -> List[float]:
    return [float(v) for v in values]


def _as_float_xyz(xyz: Sequence[Any]) -> Tuple[float, float, float]:
    return (float(xyz[0]), float(xyz[1]), float(xyz[2]))


def _as_float_quat(quat: Sequence[Any]) -> Tuple[float, float, float, float]:
    return (float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3]))


def _assign_pose(pose: Any, xyz: Sequence[Any], quat: Sequence[Any]) -> None:
    pose.position.x = float(xyz[0])
    pose.position.y = float(xyz[1])
    pose.position.z = float(xyz[2])
    pose.orientation.x = float(quat[0])
    pose.orientation.y = float(quat[1])
    pose.orientation.z = float(quat[2])
    pose.orientation.w = float(quat[3])


def _type_names(values: Sequence[Any]) -> List[str]:
    return [type(v).__name__ for v in values]


def normalize_start_joint_positions(start_js: Any) -> Optional[List[float]]:
    vals = joint_values_7d_from_any(start_js, context="reachability_ik_seed")
    if vals is None:
        return None
    return _as_float_list(vals)


def make_joint_state_message(start_js: Any) -> Optional[Any]:
    positions = normalize_start_joint_positions(start_js)
    if positions is None:
        return None
    from sensor_msgs.msg import JointState

    js = JointState()
    js.name = [str(n) for n in PANDA_ARM_JOINT_NAMES]
    js.position = list(positions)
    return js


def _fmt_xyz(pos: Optional[Tuple[float, float, float]]) -> str:
    if pos is None:
        return "n/a"
    return "(%.4f, %.4f, %.4f)" % (float(pos[0]), float(pos[1]), float(pos[2]))


def _fmt_quat(quat: Optional[Tuple[float, float, float, float]]) -> str:
    if quat is None:
        return "n/a"
    return "(%.4f, %.4f, %.4f, %.4f)" % (
        float(quat[0]),
        float(quat[1]),
        float(quat[2]),
        float(quat[3]),
    )


def format_seed_joints(joint_state: Any) -> str:
    vals = joint_values_7d_from_any(joint_state, context="reachability_seed_joints")
    if vals is None:
        return "n/a"
    return str([round(float(v), 4) for v in vals])


def build_golden_debug_variants(label: str) -> List[Any]:
    """Candidato con poses/yaw del golden demo_scene_02 (solo calibración)."""
    ref = GOLDEN_DEBUG_REFERENCES.get(str(label))
    if ref is None:
        return []
    GraspCandidate, _generate = _grasp_generator()
    pre = tuple(ref["pregrasp_tcp"])
    gr = tuple(ref["grasp_tcp"])
    top_z = float(ref.get("top_z", pre[2]))
    yaw = float(ref["commanded_tcp_yaw_rad"])
    return [
        GraspCandidate(
            idx=-1,
            label=str(label),
            strategy="golden_reference",
            center_xyz=(float(pre[0]), float(pre[1]), top_z),
            tcp_grasp_xyz=(float(gr[0]), float(gr[1]), float(gr[2])),
            tcp_pregrasp_xyz=(float(pre[0]), float(pre[1]), float(pre[2])),
            tcp_safe_pregrasp_xyz=(float(pre[0]), float(pre[1]), float(pre[2]) + 0.05),
            final_tcp_yaw_rad=yaw,
            desired_closing_yaw_rad=yaw,
            open_joint_m=0.04,
            close_joint_m=0.027,
            depth_from_top_m=float(ref.get("depth_from_top_m", top_z - float(gr[2]))),
            center_offset_m=0.0,
            center_offset_axis="major_axis",
            yaw_offset_rad=0.0,
            min_contact_margin_m=0.003,
            score=999.0,
            notes="golden_reference",
        )
    ]


def iter_debug_calibration_grid(
    labels: Sequence[str],
) -> List[Tuple[str, float, float, float]]:
    out: List[Tuple[str, float, float, float]] = []
    for label in labels:
        cell = DEBUG_CALIBRATION_CELLS.get(str(label))
        if cell is None:
            continue
        out.append((str(label), float(cell[0]), float(cell[1]), float(cell[2])))
    return out


def resolve_single_cell_from_args(
    *,
    label: str,
    x: Optional[float],
    y: Optional[float],
    yaw: Optional[float],
) -> Optional[Tuple[str, float, float, float]]:
    """Una celda arbitraria sugar_box/mustard_bottle para búsqueda expandida puntual."""
    has_any = bool(str(label or "").strip()) or any(
        v is not None for v in (x, y, yaw)
    )
    if not has_any:
        return None
    missing = [
        name
        for name, val in (
            ("label", label),
            ("x", x),
            ("y", y),
            ("yaw", yaw),
        )
        if val is None or (name == "label" and not str(val).strip())
    ]
    if missing:
        raise ValueError(
            "single-cell mode requires label, x, y, yaw; missing: %s"
            % ", ".join(missing)
        )
    resolved_label = str(label).strip()
    if resolved_label not in VARIANT_SEARCH_DEBUG_LABELS:
        raise ValueError(
            "single-cell mode only supports %s; got %r"
            % (sorted(VARIANT_SEARCH_DEBUG_LABELS), resolved_label)
        )
    return (
        resolved_label,
        float(x),
        float(y),
        float(yaw),
    )


def _ik_error_is_success(code: Any) -> bool:
    return str(code or "").strip().upper() == "SUCCESS"


def is_binary_reachable_cell(
    row: ScanCellResult,
    *,
    cartesian_fraction_threshold: float = DEFAULT_CARTESIAN_FRACTION_THRESHOLD,
) -> bool:
    """Criterio binario amarillo: pregrasp+plan+endpoint IK+cartesian completo."""
    return bool(
        row.pregrasp_ok
        and row.plan_to_pregrasp_ok
        and row.endpoint_ik_ok
        and _ik_error_is_success(row.pregrasp_ik_error_code)
        and _ik_error_is_success(row.endpoint_ik_error_code)
        and row.cartesian_fraction is not None
        and float(row.cartesian_fraction) + 1e-6 >= float(cartesian_fraction_threshold)
        and row.collision_ok
        and row.joint_limits_ok
        and str(row.result).upper() == "OK"
        and str(row.reason).lower() == "reachable"
    )


def binary_color_for_cell(
    row: ScanCellResult,
    *,
    cartesian_fraction_threshold: float = DEFAULT_CARTESIAN_FRACTION_THRESHOLD,
) -> str:
    return (
        "yellow"
        if is_binary_reachable_cell(
            row, cartesian_fraction_threshold=cartesian_fraction_threshold
        )
        else "black"
    )


def is_fully_reachable_variant(
    row: ScanCellResult,
    *,
    cartesian_fraction_threshold: float = DEFAULT_CARTESIAN_FRACTION_THRESHOLD,
) -> bool:
    """Alias del criterio binario estricto (compat tests/logs existentes)."""
    return is_binary_reachable_cell(
        row, cartesian_fraction_threshold=cartesian_fraction_threshold
    )


def reachability_heatmap_cell_value(row: Dict[str, Any]) -> float:
    """1.0 = amarillo (pick completo), 0.0 = negro. Sin valores intermedios."""
    if str(row.get("binary_color", "")).lower() == "yellow":
        return 1.0
    if str(row.get("cell_fully_reachable", "")).lower() == "true":
        return 1.0
    if str(row.get("binary_color", "")).lower() == "black":
        return 0.0
    if str(row.get("result", "")).upper() != "OK":
        return 0.0
    for key in ("endpoint_ik_ok", "pregrasp_ok", "plan_to_pregrasp_ok"):
        if key in row and str(row.get(key, "")).lower() != "true":
            return 0.0
    frac = row.get("cartesian_fraction")
    if frac not in (None, ""):
        try:
            if float(frac) + 1e-6 < DEFAULT_CARTESIAN_FRACTION_THRESHOLD:
                return 0.0
        except (TypeError, ValueError):
            return 0.0
    if str(row.get("endpoint_ik_error_code", "SUCCESS")).upper() not in (
        "",
        "SUCCESS",
    ):
        return 0.0
    return 1.0


def format_reachability_operational_center_log(
    coords: ReachabilityCellCoordinates,
) -> str:
    return (
        "[REACHABILITY_OPERATIONAL_CENTER]\n"
        "label=%s\n"
        "input_mode=%s\n"
        "spawn_xy=(%.4f, %.4f)\n"
        "operational_grasp_xy=(%.4f, %.4f)\n"
        "sdf_offset_local_xy=(%.4f, %.4f)\n"
        "sdf_offset_rotated_xy=(%.4f, %.4f)\n"
        "yaw=%.4f\n"
        "result=OK"
        % (
            coords.label,
            coords.input_mode,
            float(coords.spawn_x),
            float(coords.spawn_y),
            float(coords.operational_grasp_x),
            float(coords.operational_grasp_y),
            float(coords.sdf_offset_local_xy[0]),
            float(coords.sdf_offset_local_xy[1]),
            float(coords.sdf_offset_rotated_xy[0]),
            float(coords.sdf_offset_rotated_xy[1]),
            float(coords.yaw),
        )
    )


def format_reachability_binary_cell_decision_log(
    row: ScanCellResult,
    *,
    cartesian_fraction_threshold: float = DEFAULT_CARTESIAN_FRACTION_THRESHOLD,
) -> str:
    color = binary_color_for_cell(
        row, cartesian_fraction_threshold=cartesian_fraction_threshold
    )
    frac = row.cartesian_fraction
    return (
        "[REACHABILITY_BINARY_CELL_DECISION]\n"
        "label=%s\n"
        "x=%.4f\n"
        "y=%.4f\n"
        "yaw=%.4f\n"
        "pregrasp_ik_ok=%s\n"
        "plan_to_pregrasp_ok=%s\n"
        "endpoint_ik_ok=%s\n"
        "endpoint_ik_error_code=%s\n"
        "cartesian_fraction=%s\n"
        "result=%s\n"
        "reason=%s\n"
        "binary_color=%s"
        % (
            row.label,
            float(row.x),
            float(row.y),
            float(row.yaw),
            str(bool(row.pregrasp_ok)).lower(),
            str(bool(row.plan_to_pregrasp_ok)).lower(),
            str(bool(row.endpoint_ik_ok)).lower(),
            str(row.endpoint_ik_error_code or "n/a"),
            "n/a" if frac is None else "%.5f" % float(frac),
            str(row.result),
            str(row.reason),
            color,
        )
    )


def apply_reachability_coordinate_metadata(
    row: ScanCellResult,
    coords: ReachabilityCellCoordinates,
    *,
    cartesian_fraction_threshold: float = DEFAULT_CARTESIAN_FRACTION_THRESHOLD,
) -> ScanCellResult:
    row.input_mode = str(coords.input_mode)
    row.spawn_x = float(coords.spawn_x)
    row.spawn_y = float(coords.spawn_y)
    row.operational_grasp_x = float(coords.operational_grasp_x)
    row.operational_grasp_y = float(coords.operational_grasp_y)
    row.sdf_offset_rotated_x = float(coords.sdf_offset_rotated_xy[0])
    row.sdf_offset_rotated_y = float(coords.sdf_offset_rotated_xy[1])
    row.sdf_offset_local_x = float(coords.sdf_offset_local_xy[0])
    row.sdf_offset_local_y = float(coords.sdf_offset_local_xy[1])
    row.binary_color = binary_color_for_cell(
        row, cartesian_fraction_threshold=cartesian_fraction_threshold
    )
    row.cell_fully_reachable = row.binary_color == "yellow"
    return row


def single_cell_best_csv_row(
    *,
    label: str,
    x: float,
    y: float,
    yaw: float,
    top_z: Optional[float],
    summary: Dict[str, Any],
) -> Dict[str, Any]:
    best = summary.get("best")
    cell_ok = any(
        is_binary_reachable_cell(r)
        for r in (summary.get("all_rows") or [])
    )
    def _f(v: Optional[float]) -> str:
        return "" if v is None else "%.4f" % float(v)

    row: Dict[str, Any] = {
        "label": str(label),
        "x": "%.4f" % float(x),
        "y": "%.4f" % float(y),
        "yaw": "%.4f" % float(yaw),
        "top_z": _f(top_z),
        "total_variants": str(int(summary.get("total_variants", 0))),
        "pregrasp_success": str(int(summary.get("pregrasp_success", 0))),
        "plan_success": str(int(summary.get("plan_success", 0))),
        "endpoint_success": str(int(summary.get("endpoint_success", 0))),
        "ok_count": str(int(summary.get("ok_count", 0))),
        "cell_fully_reachable": str(bool(cell_ok)).lower(),
        "variant_budget": str(summary.get("variant_budget", "")),
        "attempts_used": str(int(summary.get("attempts_used", 0))),
        "early_stop_used": str(bool(summary.get("early_stop_used", False))).lower(),
        "total_possible_variants": str(int(summary.get("total_possible_variants", 0))),
        "evaluated_variants": str(int(summary.get("evaluated_variants", 0))),
        "best_variant": "n/a",
        "best_seed": "n/a",
        "best_result": "n/a",
        "best_reason": "n/a",
        "best_pregrasp_tcp_z": "",
        "best_grasp_tcp_z": "",
        "best_commanded_tcp_yaw_rad": "",
        "best_cartesian_fraction": "",
        "input_mode": "spawn_origin",
        "spawn_x": "",
        "spawn_y": "",
        "operational_grasp_x": "",
        "operational_grasp_y": "",
        "sdf_offset_rotated_x": "",
        "sdf_offset_rotated_y": "",
        "pregrasp_ik_ok": "",
        "plan_to_pregrasp_ok": "",
        "endpoint_ik_ok": "",
        "pregrasp_ik_error_code": "",
        "endpoint_ik_error_code": "",
        "cartesian_fraction": "",
        "binary_color": "black",
        "result": "FAIL",
        "reason": "no_variant",
    }
    if not isinstance(best, ScanCellResult):
        return row
    pre = best.pregrasp_tcp or (None, None, None)
    gr = best.grasp_tcp or (None, None, None)
    row.update(
        {
            "best_variant": str(best.variant_notes or "n/a"),
            "best_seed": str(best.seed_state_name or "n/a"),
            "best_result": str(best.result),
            "best_reason": str(best.reason),
            "best_pregrasp_tcp_z": _f(pre[2]),
            "best_grasp_tcp_z": _f(gr[2]),
            "best_commanded_tcp_yaw_rad": (
                ""
                if best.commanded_tcp_yaw_rad is None
                else "%.6f" % float(best.commanded_tcp_yaw_rad)
            ),
            "best_cartesian_fraction": (
                ""
                if best.cartesian_fraction is None
                else "%.5f" % float(best.cartesian_fraction)
            ),
            "input_mode": str(best.input_mode or "spawn_origin"),
            "spawn_x": _f(best.spawn_x),
            "spawn_y": _f(best.spawn_y),
            "operational_grasp_x": _f(best.operational_grasp_x),
            "operational_grasp_y": _f(best.operational_grasp_y),
            "sdf_offset_rotated_x": _f(best.sdf_offset_rotated_x),
            "sdf_offset_rotated_y": _f(best.sdf_offset_rotated_y),
            "pregrasp_ik_ok": str(bool(best.pregrasp_ok)).lower(),
            "plan_to_pregrasp_ok": str(bool(best.plan_to_pregrasp_ok)).lower(),
            "endpoint_ik_ok": str(bool(best.endpoint_ik_ok)).lower(),
            "pregrasp_ik_error_code": str(best.pregrasp_ik_error_code or ""),
            "endpoint_ik_error_code": str(best.endpoint_ik_error_code or ""),
            "cartesian_fraction": (
                ""
                if best.cartesian_fraction is None
                else "%.5f" % float(best.cartesian_fraction)
            ),
            "binary_color": (
                "yellow"
                if cell_ok
                else binary_color_for_cell(best)
            ),
            "result": "OK" if cell_ok else str(best.result),
            "reason": "reachable" if cell_ok else str(best.reason),
        }
    )
    return row


def write_single_cell_best_csv(path: str, row: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(SINGLE_CELL_BEST_CSV_FIELDS))
        writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in SINGLE_CELL_BEST_CSV_FIELDS})


def validate_scanner_grasp_hand_target_z(
    grasp_tcp: Tuple[float, float, float],
    grasp_hand_target: Tuple[float, float, float],
    hand_to_tcp_translation: Tuple[float, float, float],
    *,
    tolerance_m: float = SCANNER_GRASP_HAND_TARGET_Z_TOLERANCE_M,
) -> Dict[str, Any]:
    """Guard: grasp_hand_target_z debe ser grasp_tcp_z + hand_to_tcp_z."""
    expected_z = float(grasp_tcp[2]) + float(hand_to_tcp_translation[2])
    actual_z = float(grasp_hand_target[2])
    delta_z = abs(actual_z - expected_z)
    ok = delta_z <= float(tolerance_m)
    return {
        "ok": bool(ok),
        "reason": "ok" if ok else "scanner_grasp_hand_target_mismatch",
        "expected_grasp_hand_z": expected_z,
        "grasp_hand_target_z": actual_z,
        "delta_z": delta_z,
    }


def format_reachability_cell_debug_log(result: "ScanCellResult") -> str:
    ht = result.hand_to_tcp_translation or (0.0, 0.0, 0.10)
    pre_hand = result.pregrasp_hand_target or result.target_hand
    gr_hand = result.grasp_hand_target
    return (
        "[REACHABILITY_CELL_DEBUG]\n"
        "label=%s\n"
        "x=%.4f\n"
        "y=%.4f\n"
        "yaw=%.4f\n"
        "variant_notes=%s\n"
        "top_z=%s\n"
        "pregrasp_tcp=%s\n"
        "grasp_tcp=%s\n"
        "pregrasp_hand_target=%s\n"
        "grasp_hand_target=%s\n"
        "hand_to_tcp_translation=%s\n"
        "commanded_tcp_yaw_rad=%s\n"
        "target_link=%s\n"
        "moveit_target_link=%s\n"
        "use_grasp_tcp=%s\n"
        "seed_state_name=%s\n"
        "seed_joints=%s\n"
        "pregrasp_ik_ok=%s\n"
        "pregrasp_ik_error_code=%s\n"
        "plan_to_pregrasp_ok=%s\n"
        "endpoint_ik_ok=%s\n"
        "endpoint_ik_error_code=%s\n"
        "cartesian_fraction=%s\n"
        "result=%s\n"
        "reason=%s"
        % (
            result.label,
            float(result.x),
            float(result.y),
            float(result.yaw),
            str(result.variant_notes or "n/a"),
            "n/a" if result.top_z is None else "%.4f" % float(result.top_z),
            _fmt_xyz(result.pregrasp_tcp),
            _fmt_xyz(result.grasp_tcp),
            _fmt_xyz(pre_hand),
            _fmt_xyz(gr_hand),
            _fmt_xyz(ht),
            "n/a"
            if result.commanded_tcp_yaw_rad is None
            else "%.6f" % float(result.commanded_tcp_yaw_rad),
            str(result.target_link or "n/a"),
            str(result.moveit_target_link or "n/a"),
            str(bool(result.use_grasp_tcp)).lower(),
            str(result.seed_state_name or "n/a"),
            str(result.seed_joints or "n/a"),
            str(bool(result.pregrasp_ok)).lower(),
            str(result.pregrasp_ik_error_code or "n/a"),
            str(bool(result.plan_to_pregrasp_ok)).lower(),
            str(bool(result.endpoint_ik_ok)).lower(),
            str(result.endpoint_ik_error_code or "n/a"),
            "n/a"
            if result.cartesian_fraction is None
            else "%.5f" % float(result.cartesian_fraction),
            str(result.result),
            str(result.reason),
        )
    )


@dataclass
class ScanCellResult:
    label: str
    x: float
    y: float
    yaw: float
    top_z: Optional[float] = None
    grasp_tcp: Optional[Tuple[float, float, float]] = None
    pregrasp_tcp: Optional[Tuple[float, float, float]] = None
    pregrasp_hand_target: Optional[Tuple[float, float, float]] = None
    grasp_hand_target: Optional[Tuple[float, float, float]] = None
    hand_to_tcp_translation: Optional[Tuple[float, float, float]] = None
    target_hand: Optional[Tuple[float, float, float]] = None
    quat: Optional[Tuple[float, float, float, float]] = None
    commanded_tcp_yaw_rad: Optional[float] = None
    target_link: str = "panda_grasp_tcp"
    moveit_target_link: str = "panda_hand"
    use_grasp_tcp: bool = True
    seed_state_name: str = "pick_workspace_ready"
    seed_joints: str = "n/a"
    variant_notes: str = ""
    pregrasp_ok: bool = False
    plan_to_pregrasp_ok: bool = False
    start_tcp_error_m: Optional[float] = None
    endpoint_ik_ok: bool = False
    cartesian_fraction: Optional[float] = None
    collision_ok: bool = False
    joint_limits_ok: bool = False
    pregrasp_ik_error_code: str = ""
    endpoint_ik_error_code: str = ""
    result: str = "FAIL"
    reason: str = "no_variant"
    variant_budget: str = ""
    attempts_used: int = 0
    early_stop_used: bool = False
    total_possible_variants: int = 0
    evaluated_variants: int = 0
    cell_fully_reachable: bool = False
    input_mode: str = "spawn_origin"
    spawn_x: Optional[float] = None
    spawn_y: Optional[float] = None
    operational_grasp_x: Optional[float] = None
    operational_grasp_y: Optional[float] = None
    sdf_offset_rotated_x: Optional[float] = None
    sdf_offset_rotated_y: Optional[float] = None
    sdf_offset_local_x: Optional[float] = None
    sdf_offset_local_y: Optional[float] = None
    binary_color: str = "black"

    def to_csv_row(self) -> Dict[str, Any]:
        def _f(v: Optional[float]) -> str:
            return "" if v is None else "%.4f" % float(v)

        gr = self.grasp_tcp or (None, None, None)
        pre = self.pregrasp_tcp or (None, None, None)
        hand = (
            self.pregrasp_hand_target
            or self.target_hand
            or (None, None, None)
        )
        quat = self.quat or (None, None, None, None)
        return {
            "label": self.label,
            "x": "%.4f" % float(self.x),
            "y": "%.4f" % float(self.y),
            "yaw": "%.4f" % float(self.yaw),
            "top_z": _f(self.top_z),
            "grasp_tcp_x": _f(gr[0]),
            "grasp_tcp_y": _f(gr[1]),
            "grasp_tcp_z": _f(gr[2]),
            "pregrasp_tcp_x": _f(pre[0]),
            "pregrasp_tcp_y": _f(pre[1]),
            "pregrasp_tcp_z": _f(pre[2]),
            "target_hand_x": _f(hand[0]),
            "target_hand_y": _f(hand[1]),
            "target_hand_z": _f(hand[2]),
            "quat_x": _f(quat[0]),
            "quat_y": _f(quat[1]),
            "quat_z": _f(quat[2]),
            "quat_w": _f(quat[3]),
            "target_link": str(self.target_link),
            "moveit_target_link": str(self.moveit_target_link),
            "use_grasp_tcp": str(bool(self.use_grasp_tcp)).lower(),
            "seed_state_name": str(self.seed_state_name),
            "pregrasp_ok": str(bool(self.pregrasp_ok)).lower(),
            "plan_to_pregrasp_ok": str(bool(self.plan_to_pregrasp_ok)).lower(),
            "start_tcp_error_m": ""
            if self.start_tcp_error_m is None
            else "%.5f" % float(self.start_tcp_error_m),
            "endpoint_ik_ok": str(bool(self.endpoint_ik_ok)).lower(),
            "cartesian_fraction": ""
            if self.cartesian_fraction is None
            else "%.5f" % float(self.cartesian_fraction),
            "collision_ok": str(bool(self.collision_ok)).lower(),
            "joint_limits_ok": str(bool(self.joint_limits_ok)).lower(),
            "pregrasp_ik_error_code": str(self.pregrasp_ik_error_code or ""),
            "endpoint_ik_error_code": str(self.endpoint_ik_error_code or ""),
            "result": str(self.result),
            "reason": str(self.reason),
            "variant_budget": str(self.variant_budget or ""),
            "attempts_used": str(int(self.attempts_used)),
            "early_stop_used": str(bool(self.early_stop_used)).lower(),
            "total_possible_variants": str(int(self.total_possible_variants)),
            "evaluated_variants": str(int(self.evaluated_variants)),
            "cell_fully_reachable": str(bool(self.cell_fully_reachable)).lower(),
            "input_mode": str(self.input_mode or "spawn_origin"),
            "spawn_x": _f(self.spawn_x),
            "spawn_y": _f(self.spawn_y),
            "operational_grasp_x": _f(self.operational_grasp_x),
            "operational_grasp_y": _f(self.operational_grasp_y),
            "sdf_offset_rotated_x": _f(self.sdf_offset_rotated_x),
            "sdf_offset_rotated_y": _f(self.sdf_offset_rotated_y),
            "binary_color": str(self.binary_color or "black"),
        }


def joint_limits_ok_for_state(joint_state: Any, *, min_margin_rad: float) -> bool:
    vals = joint_values_7d_from_any(joint_state, context="reachability_joint_limits")
    if vals is None:
        return False
    return float(joint_limit_margin_min(vals)) > float(min_margin_rad)


def aggregate_cell_results(variant_rows: Sequence[ScanCellResult]) -> ScanCellResult:
    if not variant_rows:
        return ScanCellResult(label="", x=0.0, y=0.0, yaw=0.0, reason="no_variants")
    base = variant_rows[0]
    for row in variant_rows:
        if is_binary_reachable_cell(row):
            return row
    for row in variant_rows:
        if row.result == "OK":
            return row
    best = max(
        variant_rows,
        key=lambda r: (
            1 if r.pregrasp_ok else 0,
            1 if r.plan_to_pregrasp_ok else 0,
            -(r.start_tcp_error_m if r.start_tcp_error_m is not None else 999.0),
            1 if r.endpoint_ik_ok else 0,
            float(r.cartesian_fraction or 0.0),
            1 if r.collision_ok else 0,
            1 if r.joint_limits_ok else 0,
        ),
    )
    best.label = base.label
    best.x = base.x
    best.y = base.y
    best.yaw = base.yaw
    return best


def aggregate_budgeted_cell_results(
    variant_rows: Sequence[ScanCellResult],
    *,
    label: str,
    x: float,
    y: float,
    yaw: float,
    budget_meta: Dict[str, Any],
    evaluated_variants: int,
    early_stop_used: bool,
) -> ScanCellResult:
    best = aggregate_cell_results(list(variant_rows))
    best.label = str(label)
    best.x = float(x)
    best.y = float(y)
    best.yaw = float(yaw)
    best.variant_budget = str(budget_meta.get("variant_budget", ""))
    best.attempts_used = int(evaluated_variants)
    best.early_stop_used = bool(early_stop_used)
    best.total_possible_variants = int(
        budget_meta.get("total_possible_variants", evaluated_variants)
    )
    best.evaluated_variants = int(evaluated_variants)
    best.cell_fully_reachable = any(is_binary_reachable_cell(r) for r in variant_rows)
    best.binary_color = "yellow" if best.cell_fully_reachable else "black"
    return best


def summarize_variant_search_results(
    rows: Sequence["ScanCellResult"],
) -> Dict[str, Any]:
    pregrasp_success = sum(1 for r in rows if r.pregrasp_ok)
    plan_success = sum(1 for r in rows if r.plan_to_pregrasp_ok)
    endpoint_success = sum(1 for r in rows if r.endpoint_ik_ok)
    ok_rows = [r for r in rows if is_binary_reachable_cell(r)]
    best: Optional[ScanCellResult] = None
    if ok_rows:
        best = ok_rows[0]
    elif rows:
        best = aggregate_cell_results(list(rows))
    return {
        "total_variants": len(rows),
        "pregrasp_success": int(pregrasp_success),
        "plan_success": int(plan_success),
        "endpoint_success": int(endpoint_success),
        "cartesian_success": sum(
            1
            for r in rows
            if r.cartesian_fraction is not None
            and float(r.cartesian_fraction) + 1e-6
            >= DEFAULT_CARTESIAN_FRACTION_THRESHOLD
        ),
        "ok_count": len(ok_rows),
        "fully_reachable_count": len(ok_rows),
        "best": best,
        "all_rows": list(rows),
    }


def format_variant_search_summary_log(
    *,
    label: str,
    summary: Dict[str, Any],
) -> str:
    best = summary.get("best")
    best_notes = "n/a"
    best_result = "n/a"
    best_reason = "n/a"
    best_seed = "n/a"
    if isinstance(best, ScanCellResult):
        best_notes = str(best.variant_notes or "n/a")
        best_result = str(best.result)
        best_reason = str(best.reason)
        best_seed = str(best.seed_state_name or "n/a")
    return (
        "[REACHABILITY_VARIANT_SEARCH_SUMMARY]\n"
        "label=%s\n"
        "total_variants=%d\n"
        "pregrasp_success=%d\n"
        "plan_success=%d\n"
        "endpoint_success=%d\n"
        "cartesian_success=%d\n"
        "ok_count=%d\n"
        "best_variant=%s\n"
        "best_seed=%s\n"
        "best_result=%s\n"
        "best_reason=%s"
        % (
            str(label),
            int(summary.get("total_variants", 0)),
            int(summary.get("pregrasp_success", 0)),
            int(summary.get("plan_success", 0)),
            int(summary.get("endpoint_success", 0)),
            int(summary.get("cartesian_success", 0)),
            int(summary.get("ok_count", 0)),
            best_notes,
            best_seed,
            best_result,
            best_reason,
        )
    )


def write_csv_rows(path: str, rows: Sequence[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(CSV_FIELDS))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in CSV_FIELDS})


def load_csv_rows(path: str) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def reachable_cells_from_rows(
    rows: Sequence[Dict[str, str]],
    *,
    label: Optional[str] = None,
) -> List[Tuple[str, float, float, float]]:
    out: List[Tuple[str, float, float, float]] = []
    for row in rows:
        if reachability_heatmap_cell_value(row) < 0.5:
            continue
        lb = str(row.get("label", ""))
        if label is not None and lb != label:
            continue
        out.append(
            (
                lb,
                float(row["x"]),
                float(row["y"]),
                float(row["yaw"]),
            )
        )
    return out


def compute_common_reachable_xy(
    rows: Sequence[Dict[str, str]],
    labels: Sequence[str],
    *,
    yaw_quantize: float = 0.05,
) -> List[Tuple[float, float]]:
    by_label: Dict[str, set] = {lb: set() for lb in labels}
    for row in rows:
        if reachability_heatmap_cell_value(row) < 0.5:
            continue
        lb = str(row.get("label", ""))
        if lb not in by_label:
            continue
        key = (round(float(row["x"]), 3), round(float(row["y"]), 3))
        by_label[lb].add(key)
    if not labels:
        return []
    common = set.intersection(*[by_label[lb] for lb in labels if lb in by_label])
    return sorted(common)


def summarize_scan_csv(
    path: str,
    *,
    labels: Sequence[str] = DEMO_SCAN_LABELS,
) -> Dict[str, Any]:
    rows = load_csv_rows(path)
    summary: Dict[str, Any] = {"csv": path, "labels": {}}
    for lb in labels:
        lb_rows = [r for r in rows if str(r.get("label")) == lb]
        ok_rows = [r for r in lb_rows if reachability_heatmap_cell_value(r) >= 0.5]
        xs = [float(r["x"]) for r in ok_rows]
        ys = [float(r["y"]) for r in ok_rows]
        summary["labels"][lb] = {
            "total": len(lb_rows),
            "ok": len(ok_rows),
            "x_min": min(xs) if xs else None,
            "x_max": max(xs) if xs else None,
            "y_min": min(ys) if ys else None,
            "y_max": max(ys) if ys else None,
        }
    common_xy = compute_common_reachable_xy(rows, labels)
    summary["common_xy_count"] = len(common_xy)
    summary["common_xy_sample"] = common_xy[:20]
    return summary


def propose_demo_scene_yaml(
    rows: Sequence[Dict[str, str]],
    *,
    scene_id: str = "demo_scene_03_reachable",
    labels: Sequence[str] = DEMO_SCAN_LABELS,
    table_z_m: float = DEFAULT_TABLE_Z_M,
) -> Dict[str, Any]:
    """Propone poses dentro de la intersección alcanzable (una por label)."""
    common = compute_common_reachable_xy(rows, labels)
    if not common:
        raise ValueError("no common reachable xy cells in scan results")
    objects: Dict[str, Any] = {}
    for idx, lb in enumerate(labels):
        lb_ok = [
            r
            for r in rows
            if str(r.get("label")) == lb and str(r.get("result", "")).upper() == "OK"
        ]
        if not lb_ok:
            raise ValueError("label %s has no OK cells" % lb)
        best = None
        best_dist = float("inf")
        for cx, cy in common:
            for row in lb_ok:
                x = float(row["x"])
                y = float(row["y"])
                d = math.hypot(x - cx, y - cy)
                if d < best_dist:
                    best_dist = d
                    best = row
        if best is None:
            raise ValueError("failed to assign pose for %s" % lb)
        objects[lb] = {
            "role": "target_first" if idx == 0 else "obstacle_then_target",
            "pose": {
                "x": round(float(best["x"]), 4),
                "y": round(float(best["y"]), 4),
                "yaw": round(float(best["yaw"]), 4),
            },
            "preferred_slot": int(idx),
        }
        if lb == "chips_can":
            objects[lb]["seed"] = 1004
    return {
        "scene_id": str(scene_id),
        "description": (
            "Escena propuesta desde demo_scene_reachability_scan "
            "(intersección alcanzable top-down para 4 objetos demo)"
        ),
        "pick_order": list(labels),
        "objects": objects,
        "transport_policy": {
            "forbidden_waypoints_when_obstacles_remaining": ["carry_front_high"],
            "local_exit_candidates": [
                "rear_retreat_x_negative",
                "rear_retreat_x_negative_slight_raise",
                "vertical_raise_then_rear_retreat",
                "rear_retreat_x_negative_far",
                "rear_retreat_x_negative_raise_far",
            ],
            "reconfiguration_waypoints": ["carry_mid_high"],
            "transport_route": [
                "carry_mid_high",
                "turn_back_extended_aligned",
                "box_front_high",
                "box_high",
            ],
            "backend": "direct_action",
            "validate_attached_swept_volume": True,
            "defer_named_hub_to_deterministic_transport": True,
            "use_lateral_transport_corridors": False,
        },
        "safety": {
            "obstacle_disturbance_xy_threshold_m": 0.010,
            "obstacle_disturbance_z_threshold_m": 0.020,
            "attached_transport_safety_margin_tolerance_m": 0.006,
            "local_exit_required_clearance_m": 0.050,
            "local_exit_min_table_clearance_m": 0.200,
            "reconfiguration_min_table_clearance_m": 0.200,
            "reconfiguration_min_xy_clearance_m": 0.080,
            "reconfiguration_required_clearance_m": 0.080,
            "global_route_required_clearance_m": 0.100,
        },
        "transport_phases": {
            "local_escape": [
                "carry_safe_height",
                "rear_retreat_x_negative",
                "rear_retreat_x_negative_slight_raise",
                "vertical_raise_then_rear_retreat",
                "rear_retreat_x_negative_far",
                "rear_retreat_x_negative_raise_far",
            ],
            "reconfiguration": {
                "requires_zone_ok": True,
                "backend": "direct_action",
                "waypoints": ["carry_mid_high"],
            },
            "global_transport": {
                "backend": "direct_action",
                "waypoints": [
                    "carry_mid_high",
                    "turn_back_extended_aligned",
                    "box_front_high",
                    "box_high",
                ],
            },
        },
        "place_policy": {
            "slot_mode": "ordered_near_to_far",
            "use_food_safe_dynamic_release_z": True,
        },
        "_reachability_scan_meta": {
            "table_z_m": float(table_z_m),
            "common_xy_cells": len(common),
        },
    }


def try_write_heatmap(
    rows: Sequence[Dict[str, str]],
    *,
    label: str,
    output_path: str,
) -> bool:
    try:
        import matplotlib.colors as mcolors
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        return False
    lb_rows = [r for r in rows if str(r.get("label")) == label]
    if not lb_rows:
        return False
    xs = sorted({round(float(r["x"]), 3) for r in lb_rows})
    ys = sorted({round(float(r["y"]), 3) for r in lb_rows})
    if not xs or not ys:
        return False
    grid = np.zeros((len(ys), len(xs)), dtype=float)
    x_index = {v: i for i, v in enumerate(xs)}
    y_index = {v: i for i, v in enumerate(ys)}
    for row in lb_rows:
        xi = x_index[round(float(row["x"]), 3)]
        yi = y_index[round(float(row["y"]), 3)]
        grid[yi, xi] = max(grid[yi, xi], reachability_heatmap_cell_value(row))
    cmap = mcolors.ListedColormap(
        [BINARY_HEATMAP_UNREACHABLE_COLOR, BINARY_HEATMAP_REACHABLE_COLOR]
    )
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.imshow(
        grid,
        origin="lower",
        aspect="auto",
        extent=[min(xs), max(xs), min(ys), max(ys)],
        cmap=cmap,
        vmin=0.0,
        vmax=1.0,
        interpolation="nearest",
    )
    ax.set_title("Reachability %s (binary)" % label)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    from matplotlib.patches import Patch

    ax.legend(
        handles=[
            Patch(facecolor=BINARY_HEATMAP_REACHABLE_COLOR, label="reachable"),
            Patch(facecolor=BINARY_HEATMAP_UNREACHABLE_COLOR, label="unreachable"),
        ],
        loc="upper right",
    )
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return True


@dataclass
class ReachabilityScanConfig:
    labels: Tuple[str, ...] = DEMO_SCAN_LABELS
    x_min: float = 0.40
    x_max: float = 0.75
    x_step: float = 0.03
    y_min: float = -0.22
    y_max: float = 0.22
    y_step: float = 0.03
    yaw_values: Tuple[float, ...] = (0.0, math.pi / 2.0, math.pi, -math.pi / 2.0)
    table_z_m: float = DEFAULT_TABLE_Z_M
    table_center: Tuple[float, float, float] = DEFAULT_TABLE_CENTER
    table_size: Tuple[float, float, float] = DEFAULT_TABLE_SIZE
    table_frame: str = DEFAULT_TABLE_FRAME
    waypoints_yaml: str = ""
    workspace_waypoint: str = "pick_workspace_ready"
    cartesian_fraction_threshold: float = DEFAULT_CARTESIAN_FRACTION_THRESHOLD
    joint_limit_margin_min_rad: float = DEFAULT_JOINT_LIMIT_MARGIN_MIN_RAD
    gripper_jaw_axis_offset_rad: float = 0.0
    use_grasp_tcp: bool = True
    debug_cell: bool = False
    single_cell: Optional[Tuple[str, float, float, float]] = None
    variant_budget: str = "fast"
    input_mode: str = ""
    output_csv: str = "/tmp/demo_scene_reachability_scan.csv"
    heatmap_dir: str = ""
    group_name: str = "arm"
    grasp_tcp_frame: str = "panda_grasp_tcp"
    moveit_target_link: str = "panda_hand"
    planning_time: float = 3.0


class ReachabilityMoveItBackend:
    """Adaptador MoveIt/ROS para validación de celdas."""

    def __init__(self, node: Any, config: ReachabilityScanConfig) -> None:
        self._node = node
        self._config = config
        self._moveit2 = None
        self._cartesian_client = None
        self._planning_scene_pub = None
        self._hand_to_tcp_translation = (0.0, 0.0, 0.10)
        self._hand_to_tcp_quat = (0.0, 0.0, 0.0, 1.0)
        self._table_applied = False

    def setup(self) -> None:
        import rclpy
        from geometry_msgs.msg import PoseStamped
        from moveit_msgs.msg import CollisionObject, PlanningScene
        from moveit_msgs.srv import GetCartesianPath
        from pymoveit2 import MoveIt2
        from pymoveit2.robots import panda as panda_robot
        from rclpy.qos import QoSProfile
        try:
            from rclpy.qos import DurabilityPolicy, HistoryPolicy, ReliabilityPolicy
        except ImportError:
            from rclpy.qos import QoSDurabilityPolicy as DurabilityPolicy
            from rclpy.qos import QoSHistoryPolicy as HistoryPolicy
            from rclpy.qos import QoSReliabilityPolicy as ReliabilityPolicy
        from shape_msgs.msg import SolidPrimitive

        self._PoseStamped = PoseStamped
        self._CollisionObject = CollisionObject
        self._PlanningScene = PlanningScene
        self._SolidPrimitive = SolidPrimitive
        self._GetCartesianPath = GetCartesianPath
        self._panda = panda_robot

        self._moveit2 = MoveIt2(
            node=self._node,
            joint_names=panda_robot.joint_names(),
            base_link_name=panda_robot.base_link_name(),
            end_effector_name=panda_robot.end_effector_name(),
            group_name=self._config.group_name,
        )
        self._moveit2.planning_time = float(self._config.planning_time)
        self._moveit2.num_planning_attempts = 3
        self._planning_frame = panda_robot.base_link_name()
        self._planning_scene_pub = self._node.create_publisher(
            PlanningScene, "/planning_scene", 10
        )
        self._cartesian_client = self._node.create_client(
            GetCartesianPath,
            "compute_cartesian_path",
            qos_profile=QoSProfile(
                durability=DurabilityPolicy.VOLATILE,
                reliability=ReliabilityPolicy.RELIABLE,
                history=HistoryPolicy.KEEP_LAST,
                depth=1,
            ),
        )
        self._load_hand_to_tcp_from_tf()

    def _load_hand_to_tcp_from_tf(self) -> None:
        try:
            from tf2_ros import Buffer, TransformListener

            if not hasattr(self, "_tf_buffer"):
                self._tf_buffer = Buffer()
                self._tf_listener = TransformListener(self._tf_buffer, self._node)
            import rclpy

            deadline = time.monotonic() + 5.0
            while time.monotonic() < deadline:
                rclpy.spin_once(self._node, timeout_sec=0.1)
                try:
                    tf = self._tf_buffer.lookup_transform(
                        self._config.moveit_target_link,
                        self._config.grasp_tcp_frame,
                        rclpy.time.Time(),
                    )
                    t = tf.transform.translation
                    q = tf.transform.rotation
                    self._hand_to_tcp_translation = (float(t.x), float(t.y), float(t.z))
                    self._hand_to_tcp_quat = (
                        float(q.x),
                        float(q.y),
                        float(q.z),
                        float(q.w),
                    )
                    return
                except Exception:
                    continue
        except Exception:
            pass

    def apply_table_collision(self) -> None:
        if self._table_applied:
            return
        margin = 0.0
        size = (
            self._config.table_size[0] + margin,
            self._config.table_size[1] + margin,
            self._config.table_size[2] + margin,
        )
        primitive = self._SolidPrimitive()
        primitive.type = self._SolidPrimitive.BOX
        primitive.dimensions = [float(size[0]), float(size[1]), float(size[2])]
        pose = self._PoseStamped()
        pose.header.frame_id = self._config.table_frame
        pose.pose.position.x = float(self._config.table_center[0])
        pose.pose.position.y = float(self._config.table_center[1])
        pose.pose.position.z = float(self._config.table_center[2])
        pose.pose.orientation.w = 1.0
        col = self._CollisionObject()
        col.id = "reachability_scan_table"
        col.header.frame_id = self._config.table_frame
        col.operation = self._CollisionObject.ADD
        col.primitives = [primitive]
        col.primitive_poses = [pose.pose]
        scene = self._PlanningScene()
        scene.is_diff = True
        scene.world.collision_objects = [col]
        self._planning_scene_pub.publish(scene)
        self._table_applied = True

    def workspace_joint_state(self) -> List[float]:
        wp_path = resolve_waypoints_yaml_path(self._config.waypoints_yaml)
        data = load_waypoints_file(wp_path)
        joints = get_waypoint_joint_positions(data, self._config.workspace_waypoint)
        if joints is None:
            raise RuntimeError(
                "waypoint %s not configured in %s"
                % (self._config.workspace_waypoint, wp_path)
            )
        return _as_float_list(joints)

    def home_joint_state(self) -> Optional[List[float]]:
        wp_path = resolve_waypoints_yaml_path(self._config.waypoints_yaml)
        data = load_waypoints_file(wp_path)
        joints = get_waypoint_joint_positions(data, "home")
        if joints is None:
            return None
        return _as_float_list(joints)

    def current_joint_state(self) -> Optional[List[float]]:
        if self._moveit2 is None:
            return None
        js = getattr(self._moveit2, "joint_state", None)
        if js is None:
            return None
        return normalize_start_joint_positions(js)

    @staticmethod
    def joint7_near_zero_positions(base_js: Sequence[float]) -> List[float]:
        vals = _as_float_list(base_js)
        if len(vals) < 7:
            return list(vals)
        vals[6] = 0.0
        return vals

    def resolve_debug_ik_seeds(
        self, workspace_js: Sequence[float]
    ) -> Dict[str, List[float]]:
        seeds: Dict[str, List[float]] = {
            "pick_workspace_ready": _as_float_list(workspace_js),
        }
        home = self.home_joint_state()
        if home is not None:
            seeds["home"] = home
        current = self.current_joint_state()
        if current is not None:
            seeds["current_joint_state"] = current
        seeds["joint7_near_zero"] = self.joint7_near_zero_positions(workspace_js)
        return seeds

    def fk_link_pose(
        self, joint_state: Any, link: str
    ) -> Optional[Tuple[Tuple[float, float, float], Tuple[float, float, float, float]]]:
        if self._moveit2 is None:
            return None
        try:
            future = self._moveit2.compute_fk_async(
                joint_state, fk_link_names=[str(link)]
            )
        except Exception:
            return None
        import rclpy

        deadline = time.monotonic() + 3.0
        while not future.done() and time.monotonic() < deadline:
            rclpy.spin_once(self._node, timeout_sec=0.05)
        if not future.done():
            return None
        try:
            poses = self._moveit2.get_compute_fk_result(
                future, fk_link_names=[str(link)]
            )
        except Exception:
            return None
        if not poses:
            return None
        p = poses[0].pose.position
        o = poses[0].pose.orientation
        pos = (float(p.x), float(p.y), float(p.z))
        quat = (float(o.x), float(o.y), float(o.z), float(o.w))
        return pos, quat

    def fk_tcp_yaw_from_joint_state(self, joint_state: Any) -> Optional[float]:
        pose = self.fk_link_pose(joint_state, self._config.grasp_tcp_frame)
        if pose is None:
            return None
        _pos, quat = pose
        return yaw_from_quaternion_xyzw(quat)

    def _log_ik_request_types(
        self,
        *,
        seed_js: Any,
        position: Sequence[Any],
        quat: Sequence[Any],
    ) -> None:
        seed_positions = normalize_start_joint_positions(seed_js)
        self._node.get_logger().error(
            "[REACHABILITY_IK_REQUEST_TYPES]\n"
            "seed_joint_types=%s\n"
            "target_position_types=%s\n"
            "quat_types=%s"
            % (
                _type_names(seed_positions or []),
                _type_names(position),
                _type_names(quat),
            )
        )

    def tcp_to_hand_pose(
        self,
        tcp_pos: Tuple[float, float, float],
        tcp_quat: Tuple[float, float, float, float],
    ) -> Tuple[Tuple[float, float, float], Tuple[float, float, float, float]]:
        hand_pos, hand_quat = hand_pose_from_desired_tcp(
            _as_float_xyz(tcp_pos),
            _as_float_quat(tcp_quat),
            _as_float_xyz(self._hand_to_tcp_translation),
            _as_float_quat(self._hand_to_tcp_quat),
        )
        return _as_float_xyz(hand_pos), _as_float_quat(hand_quat)

    def compute_ik_with_error(
        self,
        position: Tuple[float, float, float],
        quat: Tuple[float, float, float, float],
        ik_link_name: str,
        start_js: Any,
        *,
        wait_timeout_sec: float = 2.0,
        result_deadline_sec: float = 3.0,
    ) -> Tuple[Any, str]:
        if self._moveit2 is None:
            return None, "moveit_unavailable"

        pos = _as_float_xyz(position)
        q = _as_float_quat(quat)
        seed_positions = normalize_start_joint_positions(start_js)
        if seed_positions is None:
            self._log_ik_request_types(seed_js=start_js, position=pos, quat=q)
            return None, "invalid_seed_joint_state"

        try:
            future = self._moveit2.compute_ik_async(
                position=pos,
                quat_xyzw=q,
                ik_link_name=str(ik_link_name),
                start_joint_state=seed_positions,
                wait_for_server_timeout_sec=float(wait_timeout_sec),
            )
        except Exception as exc:
            self._log_ik_request_types(seed_js=start_js, position=pos, quat=q)
            return None, "exception:%s" % exc
        if future is None:
            return None, "ik_future_none"
        import rclpy
        from moveit_msgs.msg import MoveItErrorCodes
        from pymoveit2.utils import enum_to_str

        deadline = time.monotonic() + float(result_deadline_sec)
        while not future.done() and time.monotonic() < deadline:
            rclpy.spin_once(self._node, timeout_sec=0.05)
        if not future.done():
            return None, "ik_timeout"
        try:
            res = future.result()
        except Exception as exc:
            self._log_ik_request_types(seed_js=start_js, position=pos, quat=q)
            return None, "exception:%s" % exc
        code = int(res.error_code.val)
        code_name = enum_to_str(MoveItErrorCodes, code)
        if code == MoveItErrorCodes.SUCCESS:
            return res.solution.joint_state, code_name
        return None, code_name

    def compute_ik_tcp(
        self,
        tcp_pos: Tuple[float, float, float],
        tcp_quat: Tuple[float, float, float, float],
        start_js: Any,
    ) -> Any:
        js, _code = self.compute_ik_with_error(
            tcp_pos,
            tcp_quat,
            self._config.grasp_tcp_frame,
            start_js,
        )
        return js

    def compute_ik_hand(
        self,
        hand_pos: Tuple[float, float, float],
        hand_quat: Tuple[float, float, float, float],
        start_js: Any,
    ) -> Any:
        js, _code = self.compute_ik_with_error(
            hand_pos,
            hand_quat,
            self._config.moveit_target_link,
            start_js,
        )
        return js

    def fk_link_position(self, joint_state: Any, link: str) -> Optional[Tuple[float, float, float]]:
        if self._moveit2 is None:
            return None
        try:
            future = self._moveit2.compute_fk_async(
                joint_state, fk_link_names=[str(link)]
            )
        except Exception:
            return None
        import rclpy

        deadline = time.monotonic() + 3.0
        while not future.done() and time.monotonic() < deadline:
            rclpy.spin_once(self._node, timeout_sec=0.05)
        if not future.done():
            return None
        try:
            poses = self._moveit2.get_compute_fk_result(
                future, fk_link_names=[str(link)]
            )
        except Exception:
            return None
        if not poses:
            return None
        p = poses[0].pose.position
        return (float(p.x), float(p.y), float(p.z))

    def plan_to_joint_state(self, start_js: Any, goal_js: Any) -> bool:
        if self._moveit2 is None:
            return False
        goal_vals = joint_values_7d_from_any(goal_js, context="reachability_plan_goal")
        start_vals = normalize_start_joint_positions(start_js)
        if goal_vals is None or start_vals is None:
            return False
        try:
            traj = self._moveit2.plan(
                joint_positions=_as_float_list(goal_vals),
                joint_names=list(PANDA_ARM_JOINT_NAMES),
                start_joint_state=start_vals,
            )
        except Exception:
            return False
        return traj is not None and bool(getattr(traj, "points", None))

    def cartesian_descend_fraction(
        self,
        start_js: Any,
        hand_goal: Tuple[float, float, float],
        hand_quat: Tuple[float, float, float, float],
    ) -> Tuple[Optional[float], bool, bool]:
        from panda_controller.get_cartesian_path_audit import (
            build_get_cartesian_path_request,
            evaluate_get_cartesian_path_start_state_audit,
        )

        import rclpy

        empty = (None, False, False)
        if self._moveit2 is None or self._cartesian_client is None:
            return empty
        if not self._cartesian_client.wait_for_service(timeout_sec=3.0):
            return empty
        req = build_get_cartesian_path_request(
            planning_frame=str(self._planning_frame),
            group_name=str(self._config.group_name),
            link_name=str(self._config.moveit_target_link),
            start_js=start_js,
            hand_goal=hand_goal,
            hand_quat=hand_quat,
            max_step=0.0025,
            jump_threshold=0.0,
            avoid_collisions=True,
        )
        if req is None:
            return empty
        future = self._cartesian_client.call_async(req)
        deadline = time.monotonic() + max(5.0, float(self._config.planning_time) + 2.0)
        while not future.done() and time.monotonic() < deadline:
            rclpy.spin_once(self._node, timeout_sec=0.05)
        if not future.done():
            return empty
        response = future.result()
        audit = evaluate_get_cartesian_path_start_state_audit(
            requested_start_js=start_js,
            response=response,
        )
        fraction = audit.get("fraction")
        honored = audit.get("start_state_honored")
        collision_ok = bool(
            response is not None
            and getattr(response, "error_code", None) is not None
            and int(response.error_code.val) == 1
        )
        cart_ok = bool(
            honored
            and fraction is not None
            and float(fraction) + 1e-6 >= float(self._config.cartesian_fraction_threshold)
        )
        return (
            float(fraction) if fraction is not None else None,
            cart_ok,
            collision_ok,
        )


def validate_variant_with_moveit(
    backend: ReachabilityMoveItBackend,
    variant: Any,
    *,
    workspace_js: Any,
    table_z_m: float,
    scene_obstacles: Sequence[Dict[str, Any]],
    joint_limit_margin_min_rad: float,
    cell_meta: Optional[Dict[str, Any]] = None,
    ik_seed_js: Optional[Any] = None,
    seed_state_name: Optional[str] = None,
    ik_wait_timeout_sec: float = 2.0,
    ik_result_deadline_sec: float = 3.0,
) -> ScanCellResult:
    from panda_controller.simple_direct_pick_route import (
        evaluate_simple_direct_pregrasp_start_fk,
    )

    meta = dict(cell_meta or {})
    pre_tcp = variant.tcp_pregrasp_xyz
    gr_tcp = variant.tcp_grasp_xyz
    commanded_yaw = float(variant.final_tcp_yaw_rad)
    quat = downward_yaw_quaternion(commanded_yaw)
    cfg = backend._config
    seed_js = ik_seed_js if ik_seed_js is not None else workspace_js
    seed_name = str(seed_state_name or cfg.workspace_waypoint)
    result = ScanCellResult(
        label=str(meta.get("label", variant.label)),
        x=float(meta.get("x", pre_tcp[0])),
        y=float(meta.get("y", pre_tcp[1])),
        yaw=float(meta.get("yaw", 0.0)),
        top_z=meta.get("top_z"),
        grasp_tcp=(float(gr_tcp[0]), float(gr_tcp[1]), float(gr_tcp[2])),
        pregrasp_tcp=(float(pre_tcp[0]), float(pre_tcp[1]), float(pre_tcp[2])),
        commanded_tcp_yaw_rad=commanded_yaw,
        quat=quat,
        target_link=str(cfg.grasp_tcp_frame),
        moveit_target_link=str(cfg.moveit_target_link),
        use_grasp_tcp=bool(cfg.use_grasp_tcp),
        seed_state_name=seed_name,
        seed_joints=format_seed_joints(seed_js),
        variant_notes=str(getattr(variant, "notes", "") or ""),
    )

    ht = _as_float_xyz(backend._hand_to_tcp_translation)
    pre_hand, pre_hand_q = backend.tcp_to_hand_pose(pre_tcp, quat)
    gr_hand, gr_hand_q = backend.tcp_to_hand_pose(gr_tcp, quat)
    result.hand_to_tcp_translation = ht
    result.pregrasp_hand_target = (
        float(pre_hand[0]),
        float(pre_hand[1]),
        float(pre_hand[2]),
    )
    result.grasp_hand_target = (
        float(gr_hand[0]),
        float(gr_hand[1]),
        float(gr_hand[2]),
    )
    result.target_hand = result.pregrasp_hand_target

    hand_guard = validate_scanner_grasp_hand_target_z(gr_tcp, gr_hand, ht)
    if not hand_guard.get("ok"):
        result.reason = str(hand_guard.get("reason") or "scanner_grasp_hand_target_mismatch")
        return result

    pregrasp_js, pre_err = backend.compute_ik_with_error(
        pre_hand,
        pre_hand_q,
        cfg.moveit_target_link,
        seed_js,
        wait_timeout_sec=float(ik_wait_timeout_sec),
        result_deadline_sec=float(ik_result_deadline_sec),
    )
    result.pregrasp_ik_error_code = str(pre_err)
    result.pregrasp_ok = pregrasp_js is not None
    if not result.pregrasp_ok:
        result.reason = "pregrasp_ik_fail"
        return result

    result.plan_to_pregrasp_ok = backend.plan_to_joint_state(seed_js, pregrasp_js)
    if not result.plan_to_pregrasp_ok:
        result.reason = "plan_to_pregrasp_fail"
        return result

    fk_tcp = backend.fk_link_position(pregrasp_js, cfg.grasp_tcp_frame)
    fk_hand = backend.fk_link_position(pregrasp_js, cfg.moveit_target_link)
    fk_eval = evaluate_simple_direct_pregrasp_start_fk(
        pregrasp_tcp_desired=pre_tcp,
        pre_hand_plan=pre_hand,
        fk_tcp=fk_tcp,
        fk_hand=fk_hand,
        error_threshold_m=float(PAIRED_PREGRASP_FK_ERROR_THRESHOLD_M),
    )
    result.start_tcp_error_m = fk_eval.get("start_tcp_error_m")
    if not fk_eval.get("ok"):
        result.reason = str(fk_eval.get("reason") or "start_fk_fail")
        return result

    endpoint_js, endpoint_err = backend.compute_ik_with_error(
        gr_hand,
        gr_hand_q,
        cfg.moveit_target_link,
        pregrasp_js,
        wait_timeout_sec=float(ik_wait_timeout_sec),
        result_deadline_sec=float(ik_result_deadline_sec),
    )
    result.endpoint_ik_error_code = str(endpoint_err)
    result.endpoint_ik_ok = endpoint_js is not None
    result.joint_limits_ok = (
        joint_limits_ok_for_state(endpoint_js, min_margin_rad=joint_limit_margin_min_rad)
        if endpoint_js is not None
        else False
    )
    if not result.endpoint_ik_ok:
        result.reason = "endpoint_ik_fail"
        return result
    if not result.joint_limits_ok:
        result.reason = "joint_limit"
        return result

    geom_ok, geom_reason = vertical_descend_volume_clear_of_obstacles(
        pre_tcp, gr_tcp, scene_obstacles
    )
    if not geom_ok:
        result.collision_ok = False
        result.reason = geom_reason
        return result

    fraction, cart_ok, moveit_collision_ok = backend.cartesian_descend_fraction(
        pregrasp_js, gr_hand, gr_hand_q
    )
    result.cartesian_fraction = fraction
    result.collision_ok = bool(geom_ok and moveit_collision_ok)
    if not cart_ok:
        result.reason = "cartesian_fraction_low"
        return result
    if not result.collision_ok:
        result.reason = "collision"
        return result

    min_grasp_z = float(table_z_m) + 0.012
    if float(gr_tcp[2]) + 1e-6 < min_grasp_z:
        result.reason = "grasp_below_table_clearance"
        return result

    result.result = "OK"
    result.reason = "reachable"
    return result


class DemoSceneReachabilityScannerNode:
    """Nodo ROS2 que ejecuta el barrido completo."""

    def __init__(self, config: ReachabilityScanConfig) -> None:
        import rclpy
        from rclpy.node import Node

        self._config = config
        self._node = Node("demo_scene_reachability_scan")
        self._backend = ReachabilityMoveItBackend(self._node, config)
        self._logger = self._node.get_logger()

    def run(self) -> int:
        import rclpy

        _vision_policy_exports(logger=self._logger)
        self._backend.setup()
        self._backend.apply_table_collision()
        if self._config.debug_cell or self._config.single_cell is not None:
            self._logger.info(
                "[REACHABILITY_SCAN_DEBUG]\n"
                "mode=%s\n"
                "hand_to_tcp_translation=%s\n"
                "hand_to_tcp_quat=%s"
                % (
                    "single_cell"
                    if self._config.single_cell is not None
                    else "debug_cell",
                    str(self._backend._hand_to_tcp_translation),
                    str(self._backend._hand_to_tcp_quat),
                )
            )
        time.sleep(0.5)
        try:
            workspace_js = self._backend.workspace_joint_state()
        except RuntimeError as exc:
            self._logger.error(str(exc))
            return 1

        scene_obstacles: List[Dict[str, Any]] = []
        rows: List[Dict[str, Any]] = []

        if self._config.single_cell is not None:
            grid_cells = [self._config.single_cell]
        elif self._config.debug_cell:
            grid_cells = iter_debug_calibration_grid(self._config.labels)
        else:
            grid_cells = [
                (str(label), float(x), float(y), float(yaw))
                for label in self._config.labels
                for x in frange(self._config.x_min, self._config.x_max, self._config.x_step)
                for y in frange(
                    self._config.y_min, self._config.y_max, self._config.y_step
                )
                for yaw in self._config.yaw_values
            ]

        total_cells = len(grid_cells)
        done = 0
        scan_mode = (
            "single_cell"
            if self._config.single_cell is not None
            else ("debug_cell" if self._config.debug_cell else "grid")
        )
        self._logger.info(
            "Reachability scan: mode=%s labels=%s cells=%d output=%s"
            % (
                scan_mode,
                list(self._config.labels),
                total_cells,
                self._config.output_csv,
            )
        )
        if self._config.single_cell is not None:
            sc_label, sc_x, sc_y, sc_yaw = self._config.single_cell
            self._logger.info(
                "[REACHABILITY_SINGLE_CELL]\n"
                "label=%s\n"
                "x=%.4f\n"
                "y=%.4f\n"
                "yaw=%.4f"
                % (sc_label, sc_x, sc_y, sc_yaw)
            )

        export_grasp_policy_for_executor, _, _ = _vision_policy_exports(
            logger=self._logger
        )

        for label, x, y, yaw in grid_cells:
            done += 1
            policy = export_grasp_policy_for_executor(label)
            policy["label"] = str(label)
            cell_input_mode = (
                str(self._config.input_mode).strip()
                if str(self._config.input_mode or "").strip()
                else default_input_mode_for_label(str(label))
            )
            detection, cell_coords = build_detection_for_reachability_cell(
                label,
                x,
                y,
                yaw,
                table_z_m=float(self._config.table_z_m),
                policy=policy,
                input_mode=cell_input_mode,
            )
            self._logger.info(format_reachability_operational_center_log(cell_coords))
            cell_meta = {
                "label": label,
                "x": x,
                "y": y,
                "yaw": yaw,
                "top_z": float(detection["top_z_m"]),
                "cell_coords": cell_coords,
            }
            variant_results: List[ScanCellResult] = []
            cell_summary: Optional[Dict[str, Any]] = None
            use_budgeted_variant_search = str(label) in VARIANT_SEARCH_DEBUG_LABELS
            variant_budget = str(self._config.variant_budget or "fast").strip().lower()
            ik_timeouts = (
                (DEBUG_IK_WAIT_TIMEOUT_SEC, DEBUG_IK_RESULT_DEADLINE_SEC)
                if variant_budget == "exhaustive"
                else (2.0, 3.0)
            )

            if use_budgeted_variant_search:
                ik_seeds = self._backend.resolve_debug_ik_seeds(workspace_js)
                workspace_tcp_yaw = self._backend.fk_tcp_yaw_from_joint_state(
                    ik_seeds.get("pick_workspace_ready", workspace_js)
                )
                jobs, budget_meta = iter_budgeted_variant_jobs(
                    label,
                    x,
                    y,
                    yaw,
                    table_z_m=float(self._config.table_z_m),
                    policy=policy,
                    detection=detection,
                    ik_seeds=ik_seeds,
                    workspace_tcp_yaw=workspace_tcp_yaw,
                    budget=variant_budget,
                )
                early_stop_enabled = bool(budget_meta.get("early_stop"))
                verbose_variant_logs = bool(
                    self._config.single_cell is not None
                    or self._config.debug_cell
                    or variant_budget == "exhaustive"
                )
                self._logger.info(
                    format_variant_budget_log(
                        label=str(label),
                        mode=str(budget_meta.get("variant_budget", variant_budget)),
                        max_variants=int(
                            budget_meta.get(
                                "max_variants",
                                budget_meta.get("total_possible_variants", len(jobs)),
                            )
                        ),
                        early_stop=early_stop_enabled,
                    )
                )
                self._logger.info(
                    "[REACHABILITY_VARIANT_SEARCH_START]\n"
                    "label=%s\n"
                    "jobs=%d\n"
                    "ik_seeds=%s"
                    % (label, len(jobs), sorted(ik_seeds.keys()))
                )
                early_stop_used = False
                for variant, seed_name, seed_js in jobs:
                    notes = str(getattr(variant, "notes", "") or "")
                    variant.notes = "%s;seed=%s" % (notes, seed_name)
                    row = validate_variant_with_moveit(
                        self._backend,
                        variant,
                        workspace_js=workspace_js,
                        table_z_m=float(self._config.table_z_m),
                        scene_obstacles=scene_obstacles,
                        joint_limit_margin_min_rad=float(
                            self._config.joint_limit_margin_min_rad
                        ),
                        cell_meta=cell_meta,
                        ik_seed_js=seed_js,
                        seed_state_name=seed_name,
                        ik_wait_timeout_sec=float(ik_timeouts[0]),
                        ik_result_deadline_sec=float(ik_timeouts[1]),
                    )
                    apply_reachability_coordinate_metadata(
                        row,
                        cell_coords,
                        cartesian_fraction_threshold=float(
                            self._config.cartesian_fraction_threshold
                        ),
                    )
                    variant_results.append(row)
                    if verbose_variant_logs:
                        self._logger.info(format_reachability_cell_debug_log(row))
                    if early_stop_enabled and is_binary_reachable_cell(row):
                        early_stop_used = True
                        self._logger.info(
                            format_variant_early_stop_log(
                                label=str(label),
                                variant_notes=str(row.variant_notes or "n/a"),
                                attempts_used=len(variant_results),
                                result=str(row.result),
                            )
                        )
                        break
                cell_summary = summarize_variant_search_results(variant_results)
                cell_summary.update(
                    {
                        "variant_budget": str(
                            budget_meta.get("variant_budget", variant_budget)
                        ),
                        "attempts_used": len(variant_results),
                        "early_stop_used": early_stop_used,
                        "total_possible_variants": int(
                            budget_meta.get("total_possible_variants", len(jobs))
                        ),
                        "evaluated_variants": len(variant_results),
                    }
                )
                self._logger.info(
                    format_variant_search_summary_log(label=label, summary=cell_summary)
                )
            else:
                variants: List[Any] = []
                if self._config.debug_cell:
                    variants.extend(build_golden_debug_variants(label))
                variants.extend(
                    iter_grasp_variants_for_cell(
                        label,
                        x,
                        y,
                        yaw,
                        table_z_m=float(self._config.table_z_m),
                        gripper_jaw_axis_offset_rad=float(
                            self._config.gripper_jaw_axis_offset_rad
                        ),
                    )
                )
                for variant in variants:
                    row = validate_variant_with_moveit(
                        self._backend,
                        variant,
                        workspace_js=workspace_js,
                        table_z_m=float(self._config.table_z_m),
                        scene_obstacles=scene_obstacles,
                        joint_limit_margin_min_rad=float(
                            self._config.joint_limit_margin_min_rad
                        ),
                        cell_meta=cell_meta,
                        ik_wait_timeout_sec=float(ik_timeouts[0]),
                        ik_result_deadline_sec=float(ik_timeouts[1]),
                    )
                    apply_reachability_coordinate_metadata(
                        row,
                        cell_coords,
                        cartesian_fraction_threshold=float(
                            self._config.cartesian_fraction_threshold
                        ),
                    )
                    variant_results.append(row)
                    if self._config.debug_cell:
                        self._logger.info(format_reachability_cell_debug_log(row))
                    if is_binary_reachable_cell(row) and not self._config.debug_cell:
                        break
            if self._config.single_cell is not None:
                if cell_summary is None:
                    cell_summary = summarize_variant_search_results(variant_results)
                rows.append(
                    single_cell_best_csv_row(
                        label=str(label),
                        x=float(x),
                        y=float(y),
                        yaw=float(yaw),
                        top_z=cell_meta.get("top_z"),
                        summary=cell_summary,
                    )
                )
                progress_row = cell_summary.get("best")
            elif use_budgeted_variant_search:
                agg = aggregate_budgeted_cell_results(
                    variant_results,
                    label=str(label),
                    x=float(x),
                    y=float(y),
                    yaw=float(yaw),
                    budget_meta={
                        "variant_budget": (
                            cell_summary.get("variant_budget", variant_budget)
                            if cell_summary
                            else variant_budget
                        ),
                        "total_possible_variants": (
                            cell_summary.get("total_possible_variants", len(variant_results))
                            if cell_summary
                            else len(variant_results)
                        ),
                    },
                    evaluated_variants=int(
                        cell_summary.get("evaluated_variants", len(variant_results))
                        if cell_summary
                        else len(variant_results)
                    ),
                    early_stop_used=bool(
                        cell_summary.get("early_stop_used", False)
                        if cell_summary
                        else False
                    ),
                )
                progress_row = agg
            else:
                progress_row = aggregate_cell_results(variant_results)
            if isinstance(progress_row, ScanCellResult):
                apply_reachability_coordinate_metadata(
                    progress_row,
                    cell_coords,
                    cartesian_fraction_threshold=float(
                        self._config.cartesian_fraction_threshold
                    ),
                )
                self._logger.info(
                    format_reachability_binary_cell_decision_log(
                        progress_row,
                        cartesian_fraction_threshold=float(
                            self._config.cartesian_fraction_threshold
                        ),
                    )
                )
                if self._config.single_cell is None:
                    rows.append(progress_row.to_csv_row())
            if (
                self._config.debug_cell
                or self._config.single_cell is not None
                or done % 10 == 0
                or (
                    isinstance(progress_row, ScanCellResult)
                    and progress_row.binary_color == "yellow"
                )
            ):
                if isinstance(progress_row, ScanCellResult):
                    agg_result = progress_row.result
                    agg_reason = progress_row.reason
                    agg_color = progress_row.binary_color
                else:
                    agg_result = "n/a"
                    agg_reason = "n/a"
                    agg_color = "n/a"
                self._logger.info(
                    "[%d/%d] %s (%.3f, %.3f) yaw=%.2f -> %s (%s) color=%s"
                    % (
                        done,
                        total_cells,
                        label,
                        x,
                        y,
                        yaw,
                        agg_result,
                        agg_reason,
                        agg_color,
                    )
                )
            rclpy.spin_once(self._node, timeout_sec=0.0)

        if self._config.single_cell is not None:
            write_single_cell_best_csv(self._config.output_csv, rows[0] if rows else {})
        else:
            write_csv_rows(self._config.output_csv, rows)
        self._logger.info("CSV written: %s (%d rows)" % (self._config.output_csv, len(rows)))

        if self._config.heatmap_dir:
            for label in self._config.labels:
                path = os.path.join(
                    self._config.heatmap_dir,
                    "reachability_%s.png" % label,
                )
                if try_write_heatmap(rows, label=label, output_path=path):
                    self._logger.info("Heatmap: %s" % path)
        return 0


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Top-down reachability scanner for demo scene objects (Panda + MoveIt)."
    )
    parser.add_argument(
        "--labels",
        default=",".join(DEMO_SCAN_LABELS),
        help="Comma-separated object labels",
    )
    parser.add_argument("--x-min", type=float, default=0.40)
    parser.add_argument("--x-max", type=float, default=0.75)
    parser.add_argument("--x-step", type=float, default=0.03)
    parser.add_argument("--y-min", type=float, default=-0.22)
    parser.add_argument("--y-max", type=float, default=0.22)
    parser.add_argument("--y-step", type=float, default=0.03)
    parser.add_argument(
        "--yaws",
        default="0,1.5708,3.1416,-1.5708",
        help="Comma-separated spawn yaw values [rad]",
    )
    parser.add_argument("--table-z", type=float, default=DEFAULT_TABLE_Z_M)
    parser.add_argument("--output-csv", default="/tmp/demo_scene_reachability_scan.csv")
    parser.add_argument("--heatmap-dir", default="", help="Optional directory for PNG heatmaps")
    parser.add_argument("--waypoints-yaml", default="")
    parser.add_argument(
        "--debug-cell",
        action="store_true",
        help=(
            "Calibrate exact demo_scene_02 cells (cracker_box/chips_can) with "
            "golden reference variants and full [REACHABILITY_CELL_DEBUG] logs. "
            "For sugar_box/mustard_bottle, use with --cell-label/--cell-x/--cell-y/"
            "--cell-yaw to probe one arbitrary pose with expanded variant search."
        ),
    )
    parser.add_argument(
        "--single-cell-label",
        "--cell-label",
        dest="single_cell_label",
        default="",
        help="Probe one sugar_box/mustard_bottle pose (expanded variant search)",
    )
    parser.add_argument(
        "--single-cell-x",
        "--cell-x",
        dest="single_cell_x",
        type=float,
        default=None,
        help="Probe cell x [m]",
    )
    parser.add_argument(
        "--single-cell-y",
        "--cell-y",
        dest="single_cell_y",
        type=float,
        default=None,
        help="Probe cell y [m]",
    )
    parser.add_argument(
        "--single-cell-yaw",
        "--cell-yaw",
        dest="single_cell_yaw",
        type=float,
        default=None,
        help="Probe cell spawn yaw [rad]",
    )
    parser.add_argument(
        "--variant-budget",
        choices=list(VARIANT_BUDGET_CHOICES),
        default="fast",
        help=(
            "Variant search budget for sugar_box/mustard_bottle: "
            "fast (~12), balanced (~40), exhaustive (full combinatorics)"
        ),
    )
    parser.add_argument(
        "--input-mode",
        choices=list(REACHABILITY_INPUT_MODES),
        default="",
        help=(
            "Coordinate frame for cell x,y: spawn_origin or operational_grasp_xy "
            "(mustard default=operational_grasp_xy; otros=spawn_origin). "
            "Vacío = automático por label."
        ),
    )
    parser.add_argument(
        "--summarize-csv",
        default="",
        help="Offline: summarize an existing scan CSV and exit",
    )
    parser.add_argument(
        "--propose-scene-yaml",
        default="",
        help="Offline: write proposed demo_scene_03_reachable.yaml from CSV",
    )
    parser.add_argument(
        "--propose-scene-output",
        default="/tmp/demo_scene_03_reachable.yaml",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.summarize_csv:
        summary = summarize_scan_csv(args.summarize_csv)
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0

    if args.propose_scene_yaml:
        rows = load_csv_rows(args.propose_scene_yaml)
        scene = propose_demo_scene_yaml(rows, table_z_m=float(args.table_z))
        os.makedirs(os.path.dirname(args.propose_scene_output) or ".", exist_ok=True)
        with open(args.propose_scene_output, "w", encoding="utf-8") as handle:
            yaml.safe_dump(
                scene,
                handle,
                default_flow_style=False,
                sort_keys=False,
                allow_unicode=True,
            )
        print("Proposed scene written to %s" % args.propose_scene_output)
        return 0

    try:
        import rclpy
    except ImportError:
        print(
            "ROS/rclpy required for live scan. Use --summarize-csv for offline analysis.",
            file=sys.stderr,
        )
        return 1

    try:
        single_cell = resolve_single_cell_from_args(
            label=str(args.single_cell_label or ""),
            x=args.single_cell_x,
            y=args.single_cell_y,
            yaw=args.single_cell_yaw,
        )
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    labels = tuple(s.strip() for s in args.labels.split(",") if s.strip())
    if single_cell is not None:
        labels = (single_cell[0],)

    config = ReachabilityScanConfig(
        labels=labels,
        x_min=float(args.x_min),
        x_max=float(args.x_max),
        x_step=float(args.x_step),
        y_min=float(args.y_min),
        y_max=float(args.y_max),
        y_step=float(args.y_step),
        yaw_values=tuple(parse_float_list(args.yaws)),
        table_z_m=float(args.table_z),
        output_csv=str(args.output_csv),
        heatmap_dir=str(args.heatmap_dir or ""),
        waypoints_yaml=str(args.waypoints_yaml or ""),
        debug_cell=bool(args.debug_cell),
        single_cell=single_cell,
        variant_budget=str(args.variant_budget),
        input_mode=str(args.input_mode or ""),
    )

    rclpy.init()
    scanner = DemoSceneReachabilityScannerNode(config)
    try:
        return int(scanner.run())
    finally:
        scanner._node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
