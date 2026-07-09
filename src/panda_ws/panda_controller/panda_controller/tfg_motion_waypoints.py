"""Carga y validación de waypoints articulares para movimiento determinista TFG."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import yaml

PANDA_ARM_JOINT_NAMES: Tuple[str, ...] = tuple(f"panda_joint{i}" for i in range(1, 8))

WAYPOINT_TEMPLATE_KEYS: Tuple[str, ...] = (
    "home",
    "transport_ready_unwind",
    "transport_home_high",
    "elbow_unwind_high",
    "carry_front_high",
    "carry_mid_high",
    "carry_extend_high",
    "turn_back_extended_aligned",
    "box_front_high",
    "carry_high",
    "box_high",
    "post_place_safe",
    "home_after_place",
)

# Orden de transporte determinista activo (carry_extend_high queda como fallback en YAML).
DETERMINISTIC_CARRY_WAYPOINT_SEQUENCE: Tuple[str, ...] = (
    "carry_front_high",
    "carry_mid_high",
    "turn_back_extended_aligned",
    "box_front_high",
)

# Secuencia completa de transporte + place (tiempos validados con follow_joint_trajectory).
DETERMINISTIC_TRANSPORT_FULL_SEQUENCE: Tuple[str, ...] = (
    "carry_front_high",
    "carry_mid_high",
    "turn_back_extended_aligned",
    "box_front_high",
    "box_high",
)

DETERMINISTIC_TRANSPORT_TIME_BY_WAYPOINT: Dict[str, float] = {
    "carry_front_high": 5.0,
    "carry_mid_high": 10.0,
    "turn_back_extended_aligned": 20.0,
    "box_front_high": 27.0,
    "box_high": 34.0,
}

# Transiciones con reparto interno en espacio articular (pasos intermedios + velocidad baja).
DETERMINISTIC_CARRY_BLENDED_TRANSITIONS: Dict[Tuple[str, str], int] = {
    ("carry_mid_high", "turn_back_extended_aligned"): 4,
}


def max_joint_delta_rad(
    prev: List[float], nxt: List[float]
) -> float:
    """Máximo |Δq| entre dos configuraciones de 7 joints."""
    n = min(len(prev), len(nxt))
    if n == 0:
        return 0.0
    return max(abs(float(nxt[i]) - float(prev[i])) for i in range(n))


def compute_transport_segment_time_s(
    max_joint_delta: float,
    *,
    nominal_joint_speed_rad_s: float,
    min_segment_time_s: float,
    max_segment_time_s: float,
    segment_padding_s: float,
    round_time_s: float = 0.0,
) -> float:
    speed = max(0.05, float(nominal_joint_speed_rad_s))
    raw = float(segment_padding_s) + float(max_joint_delta) / speed
    t = max(float(min_segment_time_s), min(float(max_segment_time_s), raw))
    rt = float(round_time_s)
    if rt > 1e-6:
        t = round(t / rt) * rt
    return float(t)


def transport_times_distance_based(
    waypoint_names: List[str],
    waypoints_data: Dict[str, Any],
    start_joints: Optional[List[float]],
    *,
    time_scale: float = 1.0,
    nominal_joint_speed_rad_s: float = 0.45,
    min_segment_time_s: float = 1.0,
    max_segment_time_s: float = 9.0,
    segment_padding_s: float = 0.35,
    first_segment_min_time_s: float = 3.0,
    first_segment_fallback_s: float = 4.0,
    round_time_s: float = 0.1,
) -> Tuple[List[float], List[Dict[str, Any]]]:
    """Tiempos acumulados time_from_start y metadatos por segmento."""
    scale = max(0.1, float(time_scale))
    cumulative: List[float] = []
    segment_logs: List[Dict[str, Any]] = []
    t_accum = 0.0
    prev: Optional[List[float]] = (
        [float(v) for v in start_joints] if start_joints is not None else None
    )
    prev_label = "current_joint_state"

    for i, name in enumerate(waypoint_names):
        joints = get_waypoint_joint_positions(waypoints_data, name)
        if joints is None:
            continue
        if prev is None:
            seg_time = float(first_segment_fallback_s) * scale
            max_delta = float("nan")
        else:
            max_delta = max_joint_delta_rad(prev, joints)
            min_seg = (
                float(first_segment_min_time_s)
                if i == 0
                else float(min_segment_time_s)
            )
            seg_time = (
                compute_transport_segment_time_s(
                    max_delta,
                    nominal_joint_speed_rad_s=nominal_joint_speed_rad_s,
                    min_segment_time_s=min_seg,
                    max_segment_time_s=max_segment_time_s,
                    segment_padding_s=segment_padding_s,
                    round_time_s=round_time_s,
                )
                * scale
            )
        t_accum += float(seg_time)
        cumulative.append(float(t_accum))
        segment_logs.append(
            {
                "segment": "%s->%s" % (prev_label, name),
                "max_joint_delta_rad": max_delta,
                "segment_time_s": float(seg_time),
                "cumulative_time_s": float(t_accum),
            }
        )
        prev = joints
        prev_label = name

    return cumulative, segment_logs


def transport_time_from_start_for_waypoints(
    waypoint_names: List[str],
    time_scale: float = 1.0,
) -> List[float]:
    """Segundos acumulados time_from_start por waypoint (orden de la lista)."""
    scale = max(0.1, float(time_scale))
    if tuple(waypoint_names) == DETERMINISTIC_TRANSPORT_FULL_SEQUENCE:
        return [
            float(DETERMINISTIC_TRANSPORT_TIME_BY_WAYPOINT[n]) * scale
            for n in waypoint_names
        ]
    times: List[float] = []
    t = 5.0 * scale
    for i, _name in enumerate(waypoint_names):
        if i == 0:
            times.append(t)
        else:
            t += 7.0 * scale
            times.append(t)
    return times


def blended_carry_transition_steps(previous: str, current: str) -> int:
    """Pasos intermedios (sin contar el waypoint destino) para suavizar un tramo."""
    return int(
        DETERMINISTIC_CARRY_BLENDED_TRANSITIONS.get((previous, current), 0)
    )


def interpolate_arm_joint_positions(
    start: List[float],
    end: List[float],
    alpha: float,
) -> List[float]:
    t = max(0.0, min(1.0, float(alpha)))
    return [float(start[i]) + t * (float(end[i]) - float(start[i])) for i in range(len(start))]


def interpolate_arm_joint_positions_staged(
    start: List[float],
    end: List[float],
    alpha: float,
    *,
    delayed_joint_indices: Tuple[int, ...] = (6,),
    delay_fraction: float = 0.55,
) -> List[float]:
    """Interpola en bloque; joint7 (índice 6) arranca después para evitar giro brusco del TCP."""
    t = max(0.0, min(1.0, float(alpha)))
    delay = max(0.0, min(0.95, float(delay_fraction)))
    out: List[float] = []
    for i in range(len(start)):
        if i in delayed_joint_indices:
            if t <= delay:
                local_t = 0.0
            else:
                local_t = (t - delay) / (1.0 - delay)
            out.append(float(start[i]) + local_t * (float(end[i]) - float(start[i])))
        else:
            out.append(float(start[i]) + t * (float(end[i]) - float(start[i])))
    return out


def deterministic_carry_waypoints_required(data: Dict[str, Any]) -> List[str]:
    """Waypoints de transporte requeridos; legado carry_high si no hay carry_front_high."""
    if waypoint_is_configured(data, "carry_front_high"):
        return list(DETERMINISTIC_CARRY_WAYPOINT_SEQUENCE)
    if waypoint_is_configured(data, "carry_high"):
        return ["carry_high"]
    return list(DETERMINISTIC_CARRY_WAYPOINT_SEQUENCE)


def default_waypoints_yaml_path() -> str:
    try:
        from ament_index_python.packages import get_package_share_directory

        share = get_package_share_directory("panda_controller")
        return os.path.join(share, "config", "tfg_motion_waypoints.yaml")
    except Exception:
        return ""


def resolve_waypoints_yaml_path(user_path: str) -> str:
    path = str(user_path or "").strip()
    if path:
        return path
    default = default_waypoints_yaml_path()
    if default and os.path.isfile(default):
        return default
    # Desarrollo sin install: ruta relativa al paquete fuente
    here = Path(__file__).resolve().parent.parent / "config" / "tfg_motion_waypoints.yaml"
    if here.is_file():
        return str(here)
    return default or str(here)


def load_waypoints_file(path: str) -> Dict[str, Any]:
    if not path or not os.path.isfile(path):
        return {}
    with open(path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    return data if isinstance(data, dict) else {}


def save_waypoints_file(path: str, data: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(
            data,
            handle,
            default_flow_style=False,
            sort_keys=False,
            allow_unicode=True,
        )


def _joints_dict_from_waypoint_entry(entry: Any) -> Optional[Dict[str, float]]:
    if not isinstance(entry, dict):
        return None
    joints = entry.get("joints")
    if not isinstance(joints, dict):
        return None
    out: Dict[str, float] = {}
    for name in PANDA_ARM_JOINT_NAMES:
        if name not in joints:
            return None
        val = joints[name]
        if val is None:
            return None
        try:
            out[name] = float(val)
        except (TypeError, ValueError):
            return None
    return out


def waypoint_is_configured(data: Dict[str, Any], name: str) -> bool:
    return _joints_dict_from_waypoint_entry(data.get(name)) is not None


def get_waypoint_joint_positions(
    data: Dict[str, Any], name: str
) -> Optional[List[float]]:
    joint_map = _joints_dict_from_waypoint_entry(data.get(name))
    if joint_map is None:
        return None
    return [float(joint_map[j]) for j in PANDA_ARM_JOINT_NAMES]


def joint_values_7d_from_any(
    value: Any,
    *,
    context: str = "",
    log_error: Optional[Callable[[str], None]] = None,
) -> Optional[List[float]]:
    """Normaliza posiciones articulares a 7 floats en orden panda_joint1..7."""
    if value is None:
        return None

    if hasattr(value, "joint_state") and not isinstance(value, (list, tuple, dict)):
        value = value.joint_state

    if hasattr(value, "name") and hasattr(value, "position"):
        try:
            names = list(value.name)
            positions = list(value.position)
        except (TypeError, ValueError):
            if log_error is not None:
                log_error(
                    "[JOINT_STATE_NORMALIZE_FAIL] context=%s type=%s reason=joint_state_attrs_invalid"
                    % (context, type(value))
                )
            return None
        joint_map = {str(n): float(p) for n, p in zip(names, positions)}
        missing = [n for n in PANDA_ARM_JOINT_NAMES if n not in joint_map]
        if missing:
            if log_error is not None:
                log_error(
                    "[JOINT_STATE_NORMALIZE_FAIL] context=%s missing=%s names=%s"
                    % (context, missing, names)
                )
            return None
        return [joint_map[n] for n in PANDA_ARM_JOINT_NAMES]

    if isinstance(value, dict):
        missing = [n for n in PANDA_ARM_JOINT_NAMES if n not in value]
        if missing:
            if log_error is not None:
                log_error(
                    "[JOINT_STATE_NORMALIZE_FAIL] context=%s missing=%s dict_keys=%s"
                    % (context, missing, list(value.keys()))
                )
            return None
        try:
            return [float(value[n]) for n in PANDA_ARM_JOINT_NAMES]
        except (TypeError, ValueError):
            if log_error is not None:
                log_error(
                    "[JOINT_STATE_NORMALIZE_FAIL] context=%s reason=dict_values_not_float"
                    % context
                )
            return None

    try:
        values = [float(x) for x in value]
    except TypeError:
        if log_error is not None:
            log_error(
                "[JOINT_STATE_NORMALIZE_FAIL] context=%s type=%s reason=not_iterable"
                % (context, type(value))
            )
        return None

    if len(values) != 7:
        if log_error is not None:
            log_error(
                "[JOINT_STATE_NORMALIZE_FAIL] context=%s len=%s expected=7"
                % (context, len(values))
            )
        return None

    return values


def missing_waypoint_names(
    data: Dict[str, Any], required: List[str]
) -> List[str]:
    missing: List[str] = []
    for name in required:
        if not waypoint_is_configured(data, name):
            missing.append(name)
    return missing


def ensure_template_structure(data: Dict[str, Any]) -> Dict[str, Any]:
    """Garantiza entradas para todos los waypoints conocidos (sin inventar valores)."""
    out = dict(data) if isinstance(data, dict) else {}
    for key in WAYPOINT_TEMPLATE_KEYS:
        if key not in out or not isinstance(out.get(key), dict):
            out[key] = {
                "joints": {j: None for j in PANDA_ARM_JOINT_NAMES},
                "description": (
                    "Record with: ros2 run panda_controller record_joint_waypoint "
                    "--ros-args -p waypoint_name:=%s"
                )
                % key,
            }
        else:
            entry = out[key]
            if "joints" not in entry or not isinstance(entry.get("joints"), dict):
                entry["joints"] = {j: None for j in PANDA_ARM_JOINT_NAMES}
            else:
                for j in PANDA_ARM_JOINT_NAMES:
                    entry["joints"].setdefault(j, None)
    return out


def update_waypoint_from_joint_state(
    data: Dict[str, Any],
    name: str,
    joint_positions: Dict[str, float],
) -> Dict[str, Any]:
    out = ensure_template_structure(data)
    joints: Dict[str, float] = {}
    for j in PANDA_ARM_JOINT_NAMES:
        if j not in joint_positions:
            raise ValueError("Missing joint in /joint_states: %s" % j)
        joints[j] = float(joint_positions[j])
    out[name] = {
        "joints": joints,
        "description": "Recorded from /joint_states",
    }
    return out
