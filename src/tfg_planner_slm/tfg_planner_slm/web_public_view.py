"""Traducción de resultado interno del SLM → respuesta pública para la UI.

Capa de presentación: recibe `BackendCommandResult` (y opcionalmente el
resultado de ejecución ROS) y lo convierte en mensajes humanos. No llama al
modelo ni reimplementa guardrails. Nunca expone JSON interno al frontend.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from .ros_pick_place_cmd import clear_table_max_steps
from .ros_command_executor import ExecutionResult
from .slm_backend_session import BackendCommandResult
from .web_robot_bridge import GAZEBO_MISSING_MESSAGE

# Nombres humanos para los 4 objetos soportados (mundo cerrado actual).
HUMAN_OBJECT_NAMES: Dict[str, str] = {
    "cracker_box": "caja de galletas",
    "sugar_box": "caja de azúcar",
    "chips_can": "lata de patatas",
    "mustard_bottle": "bote de mostaza",
}

# Pasos públicos para una orden en preview (sin progreso real del robot).
_PREVIEW_STEP_LABELS: List[str] = [
    "Orden recibida",
    "Interpretando petición",
    "Validando orden de forma segura",
    "Preparando acción del robot",
    "Comando preparado",
]

_EXECUTION_STEP_LABELS: List[str] = [
    "Orden recibida",
    "Interpretando petición",
    "Validando orden de forma segura",
    "Comprobando escena",
    "Ejecutando movimiento en Gazebo",
    "Tarea completada",
]


def human_object_name(target_label: Optional[str]) -> str:
    if not target_label:
        return "el objeto"
    return HUMAN_OBJECT_NAMES.get(target_label, target_label)


def human_slot_number(slot_index: Optional[int]) -> Optional[int]:
    """slot_index interno (0..3) → número de cajón humano (1..4)."""
    if slot_index is None:
        return None
    return int(slot_index) + 1


def sanitize_public_text(text: str) -> str:
    """Sustituye etiquetas internas por nombres humanos en texto saliente."""
    if not text:
        return ""
    out = text
    for label, human in HUMAN_OBJECT_NAMES.items():
        out = re.sub(re.escape(label), human, out, flags=re.IGNORECASE)
    return out


def _steps(done: List[str], active: Optional[str] = None, error: Optional[str] = None) -> List[Dict[str, str]]:
    steps: List[Dict[str, str]] = [{"label": label, "state": "done"} for label in done]
    if active is not None:
        steps.append({"label": active, "state": "active"})
    if error is not None:
        steps.append({"label": error, "state": "error"})
    return steps


def _preview_steps_all_done() -> List[Dict[str, str]]:
    return [{"label": label, "state": "done"} for label in _PREVIEW_STEP_LABELS]


def _reject_message(reason: str) -> str:
    if reason == "object_not_supported":
        return "Ese objeto no forma parte de la escena actual."
    if reason == "unsafe_request":
        return "No puedo realizar esa orden de forma segura."
    if reason == "out_of_domain":
        return "Solo puedo ayudarte con tareas de recoger y colocar objetos."
    return "No puedo realizar esa orden de forma segura."


def build_public_response(
    result: BackendCommandResult,
    *,
    simulation_unavailable: bool = False,
    execution_result: Optional[ExecutionResult] = None,
    scene_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Convierte el resultado interno en respuesta pública (sin JSON interno)."""
    scene_total_objects = clear_table_max_steps(scene_id)
    final = result.parsed.final_intent
    action = result.action

    if final is None:
        return {
            "status": "error",
            "ready": True,
            "public_message": "No he podido interpretar la orden correctamente.",
            "current_step": "Error de interpretación",
            "steps": _steps(["Orden recibida"], error="No se pudo interpretar"),
            "can_execute": False,
            "requires_clarification": False,
            "clarification_question": None,
        }

    intent = str(final.get("intent") or "")

    if intent == "pick_place":
        target = final.get("target_label")
        slot_index = (final.get("destination") or {}).get("slot_index")
        cajon = human_slot_number(slot_index)
        understood = "He entendido que quieres dejar la %s en el cajón %s." % (
            human_object_name(target),
            cajon,
        )

        if simulation_unavailable:
            return {
                "status": "simulation_unavailable",
                "ready": True,
                "public_message": GAZEBO_MISSING_MESSAGE,
                "current_step": "Simulación no disponible",
                "steps": _steps(
                    [
                        "Orden recibida",
                        "Interpretando petición",
                        "Validando orden de forma segura",
                        "Comprobando escena",
                    ],
                    error="Simulación no disponible",
                ),
                "can_execute": False,
                "requires_clarification": False,
                "clarification_question": None,
            }

        if execution_result is not None:
            if execution_result.success:
                return {
                    "status": "execution_success",
                    "ready": True,
                    "public_message": "%s Movimiento completado correctamente." % understood,
                    "current_step": "Tarea completada",
                    "steps": [
                        {"label": label, "state": "done"}
                        for label in _EXECUTION_STEP_LABELS
                    ],
                    "can_execute": False,
                    "requires_clarification": False,
                    "clarification_question": None,
                }
            return {
                "status": "execution_failed",
                "ready": True,
                "public_message": (
                    "%s El robot no ha podido completar el movimiento de forma segura."
                    % understood
                ),
                "current_step": "Movimiento fallido",
                "steps": _steps(
                    [
                        "Orden recibida",
                        "Interpretando petición",
                        "Validando orden de forma segura",
                        "Comprobando escena",
                        "Ejecutando movimiento en Gazebo",
                    ],
                    error="Movimiento fallido",
                ),
                "can_execute": False,
                "requires_clarification": False,
                "clarification_question": None,
            }

        return {
            "status": "preview_ready",
            "ready": True,
            "public_message": understood,
            "current_step": "Comando preparado",
            "steps": _preview_steps_all_done(),
            "can_execute": bool(action.execution_supported),
            "requires_clarification": False,
            "clarification_question": None,
        }

    if intent == "clear_table":
        understood = (
            "He entendido que quieres recoger todos los objetos de la mesa "
            "y colocarlos en los cajones."
        )

        if simulation_unavailable:
            return {
                "status": "simulation_unavailable",
                "ready": True,
                "public_message": GAZEBO_MISSING_MESSAGE,
                "current_step": "Simulación no disponible",
                "steps": _steps(
                    [
                        "Orden recibida",
                        "Interpretando petición",
                        "Validando orden de forma segura",
                        "Comprobando escena",
                    ],
                    error="Simulación no disponible",
                ),
                "can_execute": False,
                "requires_clarification": False,
                "clarification_question": None,
            }

        if execution_result is not None:
            prog = getattr(execution_result, "progress", None) or {}
            prog_steps = prog.get("steps") if isinstance(prog, dict) else None
            prog_timings = prog.get("timings") if isinstance(prog, dict) else None
            elapsed = prog.get("elapsed_s") if isinstance(prog, dict) else None
            extra_timing = ""
            if elapsed is not None:
                extra_timing = " Tiempo total: %.1fs." % float(elapsed)
            if execution_result.success:
                steps_out = prog_steps if prog_steps else [
                    {"label": label, "state": "done"} for label in _EXECUTION_STEP_LABELS
                ]
                moved = int(getattr(execution_result, "steps_completed", 0) or 0)
                if moved <= 0:
                    moved = scene_total_objects
                return {
                    "status": "execution_success",
                    "ready": True,
                    "public_message": (
                        "%s Mesa recogida: %d objetos movidos correctamente.%s"
                        % (
                            understood,
                            moved,
                            extra_timing,
                        )
                    ),
                    "current_step": "Tarea completada",
                    "steps": steps_out,
                    "timings": prog_timings or [],
                    "elapsed_s": elapsed,
                    "can_execute": False,
                    "requires_clarification": False,
                    "clarification_question": None,
                }
            steps_done = int(getattr(execution_result, "steps_completed", 0) or 0)
            partial = (
                " Se completaron %d de %d objetos antes del fallo."
                % (steps_done, scene_total_objects)
                if steps_done > 0
                else ""
            )
            steps_out = prog_steps if prog_steps else _steps(
                [
                    "Orden recibida",
                    "Interpretando petición",
                    "Validando orden de forma segura",
                    "Comprobando escena",
                    "Ejecutando movimiento en Gazebo",
                ],
                error="Movimiento fallido",
            )
            return {
                "status": "execution_failed",
                "ready": True,
                "public_message": (
                    "%s El robot no ha podido completar la recogida de la mesa.%s%s"
                    % (understood, partial, extra_timing)
                ),
                "current_step": "Movimiento fallido",
                "steps": steps_out,
                "timings": prog_timings or [],
                "elapsed_s": elapsed,
                "can_execute": False,
                "requires_clarification": False,
                "clarification_question": None,
            }

        return {
            "status": "preview_ready",
            "ready": True,
            "public_message": understood,
            "current_step": "Comando preparado",
            "steps": _preview_steps_all_done(),
            "can_execute": bool(action.execution_supported),
            "requires_clarification": False,
            "clarification_question": None,
        }

    if intent == "go_home":
        return {
            "status": "preview_ready",
            "ready": True,
            "public_message": "He preparado la orden para llevar el robot a su posición de inicio.",
            "current_step": "Comando preparado",
            "steps": _preview_steps_all_done(),
            "can_execute": False,
            "requires_clarification": False,
            "clarification_question": None,
        }

    if intent in ("open_gripper", "close_gripper"):
        accion = "abrir" if intent == "open_gripper" else "cerrar"
        return {
            "status": "preview_ready",
            "ready": True,
            "public_message": "He preparado la orden para %s la pinza." % accion,
            "current_step": "Comando preparado",
            "steps": _preview_steps_all_done(),
            "can_execute": False,
            "requires_clarification": False,
            "clarification_question": None,
        }

    if intent == "status":
        return {
            "status": "preview_ready",
            "ready": True,
            "public_message": "He preparado la consulta de estado del robot.",
            "current_step": "Comando preparado",
            "steps": _preview_steps_all_done(),
            "can_execute": False,
            "requires_clarification": False,
            "clarification_question": None,
        }

    if intent == "ask_clarification":
        question = sanitize_public_text(
            str((final.get("safety") or {}).get("clarification_question") or "")
        ).strip()
        if not question:
            question = "¿Podrías indicarme qué objeto y en qué cajón?"
        return {
            "status": "needs_clarification",
            "ready": True,
            "public_message": "Necesito una aclaración para continuar.",
            "current_step": "Aclaración necesaria",
            "steps": _steps(
                [
                    "Orden recibida",
                    "Interpretando petición",
                    "Validando orden de forma segura",
                ],
                active="Aclaración necesaria",
            ),
            "can_execute": False,
            "requires_clarification": True,
            "clarification_question": question,
        }

    if intent == "reject":
        reason = str((final.get("safety") or {}).get("reject_reason") or "").strip()
        return {
            "status": "rejected",
            "ready": True,
            "public_message": _reject_message(reason),
            "current_step": "Orden rechazada",
            "steps": _steps(
                [
                    "Orden recibida",
                    "Interpretando petición",
                    "Validando orden de forma segura",
                ],
                error="Orden rechazada",
            ),
            "can_execute": False,
            "requires_clarification": False,
            "clarification_question": None,
        }

    return {
        "status": "error",
        "ready": True,
        "public_message": "No he podido interpretar la orden correctamente.",
        "current_step": "Error de interpretación",
        "steps": _steps(["Orden recibida"], error="No se pudo interpretar"),
        "can_execute": False,
        "requires_clarification": False,
        "clarification_question": None,
    }
