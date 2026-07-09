import json
import unicodedata
from pathlib import Path
from typing import Any, Dict, Optional, Set

import requests
import rclpy
import yaml
from ament_index_python.packages import get_package_share_directory
from rclpy.node import Node
from std_msgs.msg import String


def _normalize(text: str) -> str:
    text = unicodedata.normalize("NFD", text).encode("ascii", "ignore").decode("ascii")
    return " ".join(text.lower().strip().split())


class LLMTaskPlanner(Node):
    def __init__(self):
        super().__init__("llm_task_planner")

        default_params = (
            Path(get_package_share_directory("tfg_planner_slm"))
            / "config"
            / "pick_place_params.yaml"
        )
        self.declare_parameter("pick_params_path", str(default_params))
        self.declare_parameter("model", "qwen2.5:1.5b")
        self.declare_parameter("api_url", "http://localhost:11434/api/generate")

        self.model = str(self.get_parameter("model").value)
        self.api_url = str(self.get_parameter("api_url").value)

        cfg = self._load_yaml(Path(str(self.get_parameter("pick_params_path").value)))
        self.general_cfg = cfg.get("general", {}) if isinstance(cfg, dict) else {}
        language_cfg = cfg.get("language", {}) if isinstance(cfg, dict) else {}
        self.require_confirmation = bool(language_cfg.get("require_confirmation", True))
        self.synonyms = self._build_synonym_index(language_cfg.get("synonyms", {}))

        detected_topic = str(self.general_cfg.get("detected_objects_topic", "/detected_objects"))
        command_topic = str(self.general_cfg.get("human_command_topic", "/human_command"))
        secondary_topic = str(self.general_cfg.get("pick_request_text_topic", "/pick_request_text"))
        plan_topic = str(self.general_cfg.get("robot_plan_topic", "/robot_plan"))

        self.current_scene_objects: Dict[str, Dict[str, Any]] = {}
        self.pending_plan = None
        self.awaiting_confirmation = False

        self.create_subscription(String, detected_topic, self.vision_callback, 10)
        self.create_subscription(String, command_topic, self.listener_callback, 10)
        if secondary_topic != command_topic:
            self.create_subscription(String, secondary_topic, self.listener_callback, 10)

        self.plan_publisher = self.create_publisher(String, plan_topic, 10)
        self.chat_publisher = self.create_publisher(String, "/chat_response", 10)

        self.get_logger().info(
            f"Nodo LLM listo. command_topic={command_topic}, plan_topic={plan_topic}"
        )
        self.publish_chat_response(
            "Listo para pick and place. Dime qué objeto quieres coger."
        )

    @staticmethod
    def _load_yaml(path: Path) -> Dict[str, Any]:
        if not path.is_file():
            return {}
        try:
            with path.open("r", encoding="utf-8") as handle:
                loaded = yaml.safe_load(handle) or {}
                if isinstance(loaded, dict):
                    return loaded
        except Exception:
            pass
        return {}

    def _build_synonym_index(self, mapping: Dict[str, Any]) -> Dict[str, Set[str]]:
        out: Dict[str, Set[str]] = {}
        for canonical, synonyms in mapping.items():
            key = str(canonical).strip()
            if not key:
                continue
            normalized = {_normalize(key)}
            if isinstance(synonyms, list):
                for synonym in synonyms:
                    normalized.add(_normalize(str(synonym)))
            out[key] = {entry for entry in normalized if entry}
        return out

    def _resolve_visible_target(self, command: str) -> Optional[str]:
        normalized_command = _normalize(command)
        visible_labels = set()
        for obj in self.current_scene_objects.values():
            visible_labels.add(str(obj.get("label") or "").strip())

        for canonical, entries in self.synonyms.items():
            if not any(alias in normalized_command for alias in entries):
                continue
            if canonical in visible_labels:
                return canonical
        for label in visible_labels:
            if label and _normalize(label) in normalized_command:
                return label
        return None

    def vision_callback(self, msg: String) -> None:
        try:
            data = json.loads(msg.data)
            if isinstance(data, dict):
                self.current_scene_objects = data
            else:
                self.get_logger().warn("Formato de /detected_objects inválido.")
        except json.JSONDecodeError as exc:
            self.get_logger().warn(f"Error parseando /detected_objects: {exc}")

    def listener_callback(self, msg: String) -> None:
        command = msg.data.strip()
        if not command:
            return
        self.get_logger().info(f"Comando recibido: '{command}'")

        if self.awaiting_confirmation:
            self.handle_confirmation(command)
            return

        if not self.current_scene_objects:
            self.publish_chat_response(
                "No hay objetos visibles ahora mismo. Genera escena y vuelve a intentarlo."
            )
            return

        direct_target = self._resolve_visible_target(command)
        if direct_target:
            dialog = {
                "intent": "PICK",
                "target": direct_target,
                "ready_to_execute": True,
                "chat_response": "",
                "missing_fields": [],
            }
            self.process_dialog_result(dialog)
            return

        dialog = self.query_ollama(command, self.current_scene_objects)
        if dialog is None:
            self.publish_chat_response(
                "No he podido interpretar la orden. "
                f"Objetos visibles: {self.visible_objects_as_text()}."
            )
            return
        self.process_dialog_result(dialog)

    def handle_confirmation(self, command: str) -> None:
        normalized = _normalize(command)
        if normalized in {"si", "ok", "vale", "confirmo", "adelante", "ejecuta"}:
            plan_msg = String()
            plan_msg.data = json.dumps(self.pending_plan)
            self.plan_publisher.publish(plan_msg)
            self.publish_chat_response("Plan confirmado y enviado al executor.")
            self.awaiting_confirmation = False
            self.pending_plan = None
            return
        if normalized in {"no", "cancelar", "cancela", "para"}:
            self.awaiting_confirmation = False
            self.pending_plan = None
            self.publish_chat_response("Acción cancelada.")
            return
        self.publish_chat_response("Responde 'sí' para ejecutar o 'no' para cancelar.")

    def process_dialog_result(self, dialog: Dict[str, Any]) -> None:
        target = dialog.get("target")
        target = str(target).strip() if target is not None else ""
        missing_fields = dialog.get("missing_fields") or []
        if not isinstance(missing_fields, list):
            missing_fields = [str(missing_fields)]
        ready_to_execute = bool(dialog.get("ready_to_execute"))
        chat_response = str(dialog.get("chat_response") or "").strip()

        visible_labels = {
            str(obj.get("label") or "").strip()
            for obj in self.current_scene_objects.values()
            if isinstance(obj, dict)
        }

        if target and target not in visible_labels:
            ready_to_execute = False
            if "target" not in missing_fields:
                missing_fields.append("target")
            chat_response = (
                f"Has pedido '{target}', pero no está visible. "
                f"Objetos visibles: {self.visible_objects_as_text()}."
            )

        if ready_to_execute and target:
            self.pending_plan = {
                "steps": [{"action": "PICK", "target": target}],
                "target_class": target,
            }
            if self.require_confirmation:
                self.awaiting_confirmation = True
                self.publish_chat_response(
                    f"He resuelto la orden a '{target}'. Responde 'sí' para ejecutar o 'no' para cancelar.",
                    ready_to_execute=True,
                    awaiting_confirmation=True,
                    plan_preview=self.pending_plan,
                )
            else:
                msg = String()
                msg.data = json.dumps(self.pending_plan)
                self.plan_publisher.publish(msg)
                self.publish_chat_response(f"Ejecutando pick para '{target}'.")
            return

        if not chat_response:
            chat_response = (
                "No puedo resolver un target visible con esa orden. "
                f"Objetos visibles: {self.visible_objects_as_text()}."
            )
        self.publish_chat_response(
            chat_response,
            ready_to_execute=False,
            awaiting_confirmation=False,
            missing_fields=missing_fields,
        )

    def query_ollama(self, prompt: str, available_objects: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        objects_visible = sorted(
            {
                str(obj.get("label") or "").strip()
                for obj in available_objects.values()
                if isinstance(obj, dict)
            }
        )
        objects_str = ", ".join(objects_visible)
        synonyms_hint = ", ".join(sorted(self.synonyms.keys()))
        system_prompt = f"""
ERES UN ASISTENTE DE ROBOTICA PARA PICK AND PLACE.
OBJETOS VISIBLES: [{objects_str}]
CLASES CANONICAS: [{synonyms_hint}]
Responde SOLO JSON:
{{
  "intent": "PICK" o "UNKNOWN",
  "target": "clase_canonica" o null,
  "missing_fields": ["target"] o [],
  "ready_to_execute": true o false,
  "chat_response": "mensaje breve en espanol"
}}
"""
        payload = {
            "model": self.model,
            "prompt": prompt,
            "system": system_prompt,
            "stream": False,
            "format": "json",
        }
        try:
            response = requests.post(self.api_url, json=payload, timeout=15)
            response.raise_for_status()
            raw = response.json().get("response", "")
            clean = str(raw).replace("```json", "").replace("```", "").strip()
            parsed = json.loads(clean)
            return parsed if isinstance(parsed, dict) else None
        except Exception as exc:
            self.get_logger().warn(f"Fallo consultando SLM: {exc}")
            return None

    def visible_objects_as_text(self) -> str:
        if not self.current_scene_objects:
            return "ninguno"
        labels = sorted(
            {
                str(obj.get("label") or "").strip()
                for obj in self.current_scene_objects.values()
                if isinstance(obj, dict)
            }
        )
        return ", ".join([label for label in labels if label]) or "ninguno"

    def publish_chat_response(
        self,
        message: str,
        ready_to_execute: bool = False,
        awaiting_confirmation: bool = False,
        missing_fields: Optional[list] = None,
        plan_preview: Optional[Dict[str, Any]] = None,
    ) -> None:
        payload = {
            "role": "assistant",
            "message": message,
            "ready_to_execute": ready_to_execute,
            "awaiting_confirmation": awaiting_confirmation,
            "visible_objects": sorted(self.current_scene_objects.keys()),
        }
        if missing_fields:
            payload["missing_fields"] = missing_fields
        if plan_preview is not None:
            payload["plan_preview"] = plan_preview
        msg = String()
        msg.data = json.dumps(payload)
        self.chat_publisher.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = LLMTaskPlanner()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()