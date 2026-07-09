"""Preflight checks antes de ejecutar movimientos en Gazebo/ROS 2."""

from __future__ import annotations

import json
import os
import re
import subprocess
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

_CONTROLLERS_MAX_ATTEMPTS = 5
_CONTROLLERS_RETRY_SLEEP_S = 1.0
_REQUIRED_CONTROLLERS = ("arm_controller", "gripper_controller")

ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-9;]*m")


@dataclass
class PreflightResult:
    ok: bool
    blocking_errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    details: Dict[str, Any] = field(default_factory=dict)


def strip_ansi(text: str) -> str:
    return ANSI_ESCAPE_RE.sub("", text or "")


def parse_active_controllers(stdout: str) -> Set[str]:
    """Parsea `ros2 control list_controllers` (estado en la última columna)."""
    clean_stdout = strip_ansi(stdout)
    active: Set[str] = set()
    for line in clean_stdout.splitlines():
        cols = line.split()
        if len(cols) < 2:
            continue
        name = cols[0].strip()
        state = cols[-1].strip()
        if state == "active":
            active.add(name)
    return active


def _no_color_env() -> Dict[str, str]:
    env = os.environ.copy()
    env["RCUTILS_COLORIZED_OUTPUT"] = "0"
    env["PY_COLORS"] = "0"
    env["NO_COLOR"] = "1"
    env["TERM"] = "dumb"
    return env


def _run(
    cmd: List[str],
    timeout_s: float = 5.0,
    env: Optional[Dict[str, str]] = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=float(timeout_s),
        check=False,
        env=env,
    )


def _contains_line(output: str, needle: str) -> bool:
    clean = strip_ansi(output)
    for line in clean.splitlines():
        if line.strip() == needle:
            return True
    return False


def _check_controllers_with_retries(
    *,
    timeout_s: float,
    details: Dict[str, Any],
    blocking_errors: List[str],
) -> None:
    last_stdout_raw = ""
    last_stdout_clean = ""
    last_stderr = ""
    last_rc: int | None = None
    last_active: Set[str] = set()
    ok_attempt: int | None = None
    no_color_env = _no_color_env()

    for attempt in range(1, _CONTROLLERS_MAX_ATTEMPTS + 1):
        print(
            "[ROS_PREFLIGHT] checking controllers attempt=%d/%d"
            % (attempt, _CONTROLLERS_MAX_ATTEMPTS),
            flush=True,
        )
        try:
            cp = _run(
                ["ros2", "control", "list_controllers"],
                timeout_s=timeout_s,
                env=no_color_env,
            )
            last_stdout_raw = cp.stdout or ""
            last_stdout_clean = strip_ansi(last_stdout_raw)
            last_stderr = cp.stderr or ""
            last_rc = cp.returncode
            print(
                "[ROS_PREFLIGHT] controllers_stdout_raw=%s"
                % last_stdout_raw.strip(),
                flush=True,
            )
            print(
                "[ROS_PREFLIGHT] controllers_stdout_clean=%s"
                % last_stdout_clean.strip(),
                flush=True,
            )
            if last_stderr.strip():
                print(
                    "[ROS_PREFLIGHT] controllers_stderr=%s"
                    % strip_ansi(last_stderr).strip(),
                    flush=True,
                )

            if cp.returncode != 0:
                if attempt < _CONTROLLERS_MAX_ATTEMPTS:
                    time.sleep(_CONTROLLERS_RETRY_SLEEP_S)
                continue

            last_active = parse_active_controllers(last_stdout_clean)
            print(
                "[ROS_PREFLIGHT] active_controllers=%s"
                % sorted(last_active),
                flush=True,
            )

            arm_ok = _REQUIRED_CONTROLLERS[0] in last_active
            grip_ok = _REQUIRED_CONTROLLERS[1] in last_active
            if arm_ok and grip_ok:
                ok_attempt = attempt
                print(
                    "[ROS_PREFLIGHT] controllers_ok attempt=%d" % attempt,
                    flush=True,
                )
                break

        except (subprocess.TimeoutExpired, OSError) as exc:
            last_stderr = str(exc)
            last_rc = None
            if attempt < _CONTROLLERS_MAX_ATTEMPTS:
                time.sleep(_CONTROLLERS_RETRY_SLEEP_S)
            continue

        if attempt < _CONTROLLERS_MAX_ATTEMPTS:
            time.sleep(_CONTROLLERS_RETRY_SLEEP_S)

    details["controllers_stdout_raw"] = last_stdout_raw
    details["controllers_stdout_clean"] = last_stdout_clean
    details["controllers_stdout"] = last_stdout_raw
    details["controllers_stderr"] = last_stderr
    details["controllers_returncode"] = last_rc
    details["active_controllers"] = sorted(last_active)
    details["controllers_attempts"] = _CONTROLLERS_MAX_ATTEMPTS
    details["controllers_ok_attempt"] = ok_attempt
    details["arm_controller_active"] = _REQUIRED_CONTROLLERS[0] in last_active
    details["gripper_controller_active"] = _REQUIRED_CONTROLLERS[1] in last_active

    if ok_attempt is not None:
        return

    stdout_snippet = last_stdout_clean.strip() or last_stdout_raw.strip() or "(vacío)"
    if _REQUIRED_CONTROLLERS[0] not in last_active:
        blocking_errors.append(
            "arm_controller no está activo. stdout recibido: %s" % stdout_snippet
        )
    if _REQUIRED_CONTROLLERS[1] not in last_active:
        blocking_errors.append(
            "gripper_controller no está activo. stdout recibido: %s" % stdout_snippet
        )


def run_gazebo_preflight_checks(*, timeout_s: float = 10.0) -> PreflightResult:
    """Checks básicos: ROS, /clock, controladores, MoveIt/actions, percepción."""
    t0 = time.perf_counter()
    res = PreflightResult(ok=False)
    details: Dict[str, Any] = {}

    # 1) ROS 2 disponible
    try:
        cp = _run(["ros2", "node", "list"], timeout_s=timeout_s, env=_no_color_env())
        details["ros2_node_list_rc"] = cp.returncode
        details["ros2_node_list_stdout"] = strip_ansi(cp.stdout or "")
        if cp.returncode != 0:
            res.blocking_errors.append("ROS 2 no disponible: `ros2 node list` falló.")
            res.details = details
            return res
        nodes = [
            ln.strip()
            for ln in strip_ansi(cp.stdout or "").splitlines()
            if ln.strip()
        ]
        details["nodes"] = nodes
    except (subprocess.TimeoutExpired, OSError) as exc:
        res.blocking_errors.append("ROS 2 no disponible: %s" % exc)
        res.details = details
        return res

    # 2) /clock en topics
    try:
        cp = _run(["ros2", "topic", "list"], timeout_s=timeout_s, env=_no_color_env())
        details["ros2_topic_list_rc"] = cp.returncode
        if cp.returncode == 0:
            topic_out = strip_ansi(cp.stdout or "")
            has_clock = _contains_line(topic_out, "/clock")
            details["has_clock"] = has_clock
            if not has_clock:
                res.blocking_errors.append(
                    "No se detecta /clock (Gazebo/Sim time no activo)."
                )
        else:
            res.warnings.append("No se pudo listar topics (`ros2 topic list`).")
    except (subprocess.TimeoutExpired, OSError) as exc:
        res.warnings.append("No se pudo listar topics: %s" % exc)

    # 3) Controladores (reintentos)
    _check_controllers_with_retries(
        timeout_s=timeout_s,
        details=details,
        blocking_errors=res.blocking_errors,
    )

    # 4) MoveIt / acciones típicas
    try:
        cp = _run(["ros2", "action", "list"], timeout_s=timeout_s, env=_no_color_env())
        details["ros2_action_list_rc"] = cp.returncode
        if cp.returncode == 0:
            actions = [
                ln.strip()
                for ln in strip_ansi(cp.stdout or "").splitlines()
                if ln.strip()
            ]
            details["actions"] = actions
            needed = [
                "/execute_trajectory",
                "/arm_controller/follow_joint_trajectory",
                "/gripper_controller/follow_joint_trajectory",
            ]
            missing = [a for a in needed if a not in actions]
            if missing:
                res.warnings.append(
                    "Acciones MoveIt/control no detectadas: %s" % ", ".join(missing)
                )
        else:
            res.warnings.append("No se pudo listar acciones (`ros2 action list`).")
    except (subprocess.TimeoutExpired, OSError) as exc:
        res.warnings.append("No se pudo listar acciones: %s" % exc)

    # 5) Percepción / interfaces
    try:
        cp = _run(["ros2", "topic", "list"], timeout_s=timeout_s, env=_no_color_env())
        if cp.returncode == 0:
            topics = [
                ln.strip()
                for ln in strip_ansi(cp.stdout or "").splitlines()
                if ln.strip()
            ]
            details["topics"] = topics
            if "/vision_to_executor" not in topics:
                res.warnings.append(
                    "No se detecta /vision_to_executor "
                    "(percepción/bridge puede no estar activo)."
                )
        else:
            res.warnings.append("No se pudo listar topics para comprobar percepción.")
    except (subprocess.TimeoutExpired, OSError) as exc:
        res.warnings.append("No se pudo comprobar percepción: %s" % exc)

    # 6) RuntimeScene/GT opcional
    if "topics" in details and isinstance(details["topics"], list):
        if "/runtime_scene/gt_objects" not in details["topics"]:
            res.warnings.append(
                "No se detecta /runtime_scene/gt_objects (ok si no es necesario)."
            )

    res.details = details
    res.ok = len(res.blocking_errors) == 0
    res.details["duration_s"] = round(time.perf_counter() - t0, 4)
    if res.ok:
        print("[ROS_PREFLIGHT] ok", flush=True)
    else:
        for err in res.blocking_errors:
            print("[ROS_PREFLIGHT] blocking_error=%s" % err, flush=True)
    return res


def format_preflight_result(result: PreflightResult) -> str:
    payload: Dict[str, Any] = {
        "ok": result.ok,
        "blocking_errors": list(result.blocking_errors),
        "warnings": list(result.warnings),
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)
