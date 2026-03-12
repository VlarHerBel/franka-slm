import json

import requests
import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class LLMTaskPlanner(Node):
    def __init__(self):
        super().__init__("llm_task_planner")

        self.declare_parameter("model", "qwen2.5:1.5b")
        self.declare_parameter("api_url", "http://localhost:11434/api/generate")

        self.model = self.get_parameter("model").value
        self.api_url = self.get_parameter("api_url").value

        self.current_scene_objects = {}
        self.pending_plan = None
        self.awaiting_confirmation = False

        self.vision_sub = self.create_subscription(
            String, "/detected_objects", self.vision_callback, 10
        )
        self.subscription = self.create_subscription(
            String, "/human_command", self.listener_callback, 10
        )

        self.plan_publisher = self.create_publisher(String, "/robot_plan", 10)
        self.chat_publisher = self.create_publisher(String, "/chat_response", 10)

        self.get_logger().info("Nodo LLM conversacional listo.")
        self.publish_chat_response(
            "Hola. Puedo ayudarte con el pick and place. "
            "Dime qué objeto quieres coger y te pediré lo que falte."
        )

    def vision_callback(self, msg):
        try:
            data = json.loads(msg.data)
            if isinstance(data, dict):
                self.current_scene_objects = data
            else:
                self.get_logger().warn(
                    "Se ha ignorado /detected_objects porque no tenía formato de diccionario."
                )
        except json.JSONDecodeError as exc:
            self.get_logger().warn(f"Error parseando /detected_objects: {exc}")

    def listener_callback(self, msg):
        command = msg.data.strip()
        if not command:
            return

        self.get_logger().info(f"Nuevo mensaje de chat recibido: '{command}'")

        if self.awaiting_confirmation:
            self.handle_confirmation(command)
            return

        if not self.current_scene_objects:
            self.publish_chat_response(
                "Ahora mismo no veo objetos disponibles. "
                "Asegura la visión y vuelve a intentarlo."
            )
            return

        dialog = self.query_ollama(command, self.current_scene_objects)
        if dialog is None:
            self.publish_chat_response(
                "No he podido interpretar la orden porque el SLM no respondió correctamente."
            )
            return

        self.process_dialog_result(dialog)

    def handle_confirmation(self, command):
        normalized = command.strip().lower()
        if normalized in {"si", "sí", "ok", "vale", "confirmo", "adelante", "ejecuta"}:
            plan_msg = String()
            plan_msg.data = json.dumps(self.pending_plan)
            self.plan_publisher.publish(plan_msg)
            self.publish_chat_response(
                "Perfecto. Plan confirmado y enviado al executor."
            )
            self.awaiting_confirmation = False
            self.pending_plan = None
            return

        if normalized in {"no", "cancelar", "cancela", "para"}:
            self.awaiting_confirmation = False
            self.pending_plan = None
            self.publish_chat_response(
                "He cancelado la acción pendiente. Dime una nueva orden cuando quieras."
            )
            return

        self.publish_chat_response(
            "Solo necesito una confirmación simple. "
            "Responde 'sí' para ejecutar o 'no' para cancelar."
        )

    def process_dialog_result(self, dialog):
        target = dialog.get("target")
        if target is not None:
            target = str(target)

        missing_fields = dialog.get("missing_fields") or []
        if not isinstance(missing_fields, list):
            missing_fields = [str(missing_fields)]
        chat_response = dialog.get("chat_response") or ""
        ready_to_execute = bool(dialog.get("ready_to_execute"))

        if target and target not in self.current_scene_objects:
            ready_to_execute = False
            if "target" not in missing_fields:
                missing_fields.append("target")
            visible_objects = self.visible_objects_as_text()
            chat_response = (
                f"No veo '{target}' en la escena actual. "
                f"Objetos visibles: {visible_objects}."
            )

        if ready_to_execute and target:
            self.pending_plan = {
                "steps": [
                    {
                        "action": (dialog.get("intent") or "PICK").upper(),
                        "target": target,
                    }
                ]
            }
            self.awaiting_confirmation = True
            self.publish_chat_response(
                f"He entendido que quieres coger '{target}'. "
                "Responde 'sí' para ejecutar o 'no' para cancelar.",
                ready_to_execute=True,
                awaiting_confirmation=True,
                plan_preview=self.pending_plan,
            )
            return

        if not chat_response:
            if missing_fields:
                chat_response = (
                    "Todavía me falta información para actuar: "
                    + ", ".join(missing_fields)
                    + "."
                )
            else:
                chat_response = (
                    "No he podido construir un plan válido. "
                    "Prueba a reformular tu petición."
                )

        self.publish_chat_response(
            chat_response,
            ready_to_execute=False,
            awaiting_confirmation=False,
            missing_fields=missing_fields,
        )

    def query_ollama(self, prompt, available_objects):
        objects_str = ", ".join(sorted(available_objects.keys()))

        system_prompt = f"""
        ERES UN ASISTENTE DE ROBÓTICA PARA PICK AND PLACE.

        OBJETOS VISIBLES ACTUALES:
        [{objects_str}]

        REGLAS:
        1. El usuario habla en español.
        2. Los IDs de objetos están en inglés y debes mapear expresiones como
           "cubo rojo" a IDs visibles como "red_cube".
        3. Solo puedes proponer un target que esté exactamente en la lista de objetos visibles.
        4. Si falta información o el usuario es ambiguo, debes preguntar de forma natural.
        5. Si ya tienes suficiente para ejecutar un pick, marca ready_to_execute=true.
        6. No asumas destinos ni acciones complejas. Para este sistema la acción válida es PICK.
        7. Responde siempre JSON puro.

        FORMATO JSON OBLIGATORIO:
        {{
          "intent": "PICK" o "UNKNOWN",
          "target": "ID_VISIBLE" o null,
          "missing_fields": ["target"] o [],
          "ready_to_execute": true o false,
          "chat_response": "mensaje breve en español para el usuario"
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
            self.get_logger().info(
                f"Consultando Ollama con objetos visibles: {objects_str}"
            )
            response = requests.post(self.api_url, json=payload, timeout=15)
            response.raise_for_status()

            raw_response = response.json()["response"]
            clean_json = raw_response.replace("```json", "").replace("```", "").strip()
            self.get_logger().info(f"Respuesta cruda de Ollama: {clean_json}")
            return json.loads(clean_json)

        except requests.exceptions.Timeout:
            self.get_logger().error("Ollama ha agotado el tiempo de espera.")
            return None
        except (KeyError, json.JSONDecodeError) as exc:
            self.get_logger().error(f"La respuesta del SLM no era JSON válido: {exc}")
            return None
        except Exception as exc:
            self.get_logger().error(f"Error conectando con Ollama: {exc}")
            return None

    def visible_objects_as_text(self):
        if not self.current_scene_objects:
            return "ninguno"
        return ", ".join(sorted(self.current_scene_objects.keys()))

    def publish_chat_response(
        self,
        message,
        ready_to_execute=False,
        awaiting_confirmation=False,
        missing_fields=None,
        plan_preview=None,
    ):
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


if __name__ == '__main__':
    main()