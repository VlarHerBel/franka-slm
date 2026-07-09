"""Dispatcher seguro: intent JSON v1.1 → acción interna (preview, sin ROS por defecto)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .ros_pick_place_cmd import (
    DEFAULT_PICK_PLACE_SCENE_ID,
    build_clear_table_ros2_command_string,
    build_pick_place_ros2_command_string,
)

SUPPORTED_OBJECTS = ("cracker_box", "sugar_box", "chips_can", "mustard_bottle")
ALLOWED_SLOTS = (0, 1, 2, 3)


@dataclass
class InternalAction:
    """Acción interna segura derivada del intent final."""

    intent: str
    preview: str
    execution_supported: bool = False
    dry_run: bool = True
    require_confirmation: bool = True
    ros_command_preview: Optional[str] = None
    target_label: Optional[str] = None
    slot_index: Optional[int] = None
    slot_order: Optional[List[int]] = None
    clarification_question: str = ""
    reject_reason: str = ""
    details: Dict[str, Any] = field(default_factory=dict)


def _pick_place_ros_preview(target_label: str, slot_index: int) -> str:
    return build_pick_place_ros2_command_string(
        dry_run=True,
        target_label=target_label,
        slot_index=slot_index,
        slot_user_specified=True,
    )


def dispatch_command(final_intent: Optional[Dict[str, Any]]) -> InternalAction:
    """Convierte JSON final validado en acción interna (no ejecuta ROS)."""
    if not final_intent:
        return InternalAction(
            intent="error",
            preview="No hay intent final validado.",
            execution_supported=False,
        )

    intent = str(final_intent.get("intent") or "")
    execution = final_intent.get("execution") or {}
    safety = final_intent.get("safety") or {}
    dry_run = bool(execution.get("dry_run", True))
    require_confirmation = bool(execution.get("require_confirmation", True))

    if intent == "pick_place":
        target = final_intent.get("target_label")
        dest = final_intent.get("destination") or {}
        slot = dest.get("slot_index")

        if target not in SUPPORTED_OBJECTS:
            return InternalAction(
                intent=intent,
                preview="pick_place rechazado: target_label no soportado (%s)" % target,
                execution_supported=False,
                dry_run=dry_run,
            )
        if dest.get("type") != "slot" or slot not in ALLOWED_SLOTS:
            return InternalAction(
                intent=intent,
                preview="pick_place rechazado: destination.type debe ser slot con slot_index 0..3",
                execution_supported=False,
                dry_run=dry_run,
            )

        ros_preview = _pick_place_ros_preview(str(target), int(slot))
        return InternalAction(
            intent=intent,
            preview=(
                "pick_place preview (dry_run): mover %s al slot %d"
                % (target, slot)
            ),
            execution_supported=True,
            dry_run=True,
            require_confirmation=require_confirmation,
            ros_command_preview=ros_preview,
            target_label=str(target),
            slot_index=int(slot),
        )

    if intent == "clear_table":
        dest = final_intent.get("destination") or {}
        slot_order = dest.get("slot_order") or [0, 1, 2, 3]
        slot_order_list = list(slot_order) if isinstance(slot_order, list) else [0, 1, 2, 3]
        ros_preview = build_clear_table_ros2_command_string(
            dry_run=True,
            reset_completed_state=True,
            scene_id=DEFAULT_PICK_PLACE_SCENE_ID,
        )
        return InternalAction(
            intent=intent,
            preview=(
                "clear_table preview: recoger objetos de la escena %s "
                "y colocarlos en slots %s"
                % (DEFAULT_PICK_PLACE_SCENE_ID, slot_order_list)
            ),
            execution_supported=True,
            dry_run=True,
            require_confirmation=require_confirmation,
            ros_command_preview=ros_preview,
            slot_order=slot_order_list,
        )

    if intent == "go_home":
        return InternalAction(
            intent=intent,
            preview="go_home preview: volver a posición inicial (dry_run, sin ejecución ROS aún)",
            execution_supported=False,
            dry_run=True,
            require_confirmation=require_confirmation,
            ros_command_preview=(
                "# futuro: ros2 service / acción go_home con dry_run:=true"
            ),
        )

    if intent in ("open_gripper", "close_gripper"):
        return InternalAction(
            intent=intent,
            preview="%s preview (dry_run, sin ejecución ROS aún)" % intent,
            execution_supported=False,
            dry_run=True,
            require_confirmation=require_confirmation,
        )

    if intent == "status":
        return InternalAction(
            intent=intent,
            preview="status preview: consultar estado del robot (sin ejecución ROS aún)",
            execution_supported=False,
            dry_run=True,
        )

    if intent == "ask_clarification":
        question = str(safety.get("clarification_question") or "").strip()
        return InternalAction(
            intent=intent,
            preview="ask_clarification: %s" % (question or "(sin pregunta)"),
            execution_supported=False,
            dry_run=True,
            clarification_question=question,
        )

    if intent == "reject":
        reason = str(safety.get("reject_reason") or "").strip()
        return InternalAction(
            intent=intent,
            preview="reject: %s" % (reason or "(sin motivo)"),
            execution_supported=False,
            dry_run=True,
            reject_reason=reason,
        )

    return InternalAction(
        intent=intent or "unknown",
        preview="Intent desconocido: %s" % intent,
        execution_supported=False,
    )
