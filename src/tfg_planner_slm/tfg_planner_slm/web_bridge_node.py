import json
import queue
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class WebBridgeNode(Node):
    def __init__(self):
        super().__init__("web_bridge_node")

        self.declare_parameter("host", "127.0.0.1")
        self.declare_parameter("port", 8000)

        self.host = self.get_parameter("host").value
        self.port = int(self.get_parameter("port").value)
        self.web_dir = Path(__file__).resolve().parent / "web"

        self.command_queue = queue.Queue()
        self.state_lock = threading.Lock()
        self.messages = [
            {
                "role": "assistant",
                "message": (
                    "Chat web inicializado. Cuando el SLM esté activo podrás darle órdenes."
                ),
            }
        ]
        self.visible_objects = []
        self.executor_status = {
            "status": "idle",
            "message": "Esperando eventos del sistema.",
        }

        self.command_publisher = self.create_publisher(String, "/human_command", 10)
        self.chat_subscription = self.create_subscription(
            String, "/chat_response", self.chat_response_callback, 10
        )
        self.executor_subscription = self.create_subscription(
            String, "/executor_status", self.executor_status_callback, 10
        )
        self.vision_subscription = self.create_subscription(
            String, "/detected_objects", self.detected_objects_callback, 10
        )

        self.create_timer(0.1, self.flush_command_queue)

        self.http_server = None
        self.http_thread = None
        self.start_http_server()

        self.get_logger().info(
            f"Chat web disponible en http://{self.host}:{self.port}"
        )

    def start_http_server(self):
        node = self

        class RequestHandler(BaseHTTPRequestHandler):
            def do_GET(self):
                if self.path == "/api/state":
                    self.respond_json(node.get_state_snapshot())
                    return

                if self.path == "/" or self.path == "/index.html":
                    self.serve_static("index.html")
                    return

                if self.path == "/app.js":
                    self.serve_static("app.js")
                    return

                if self.path == "/styles.css":
                    self.serve_static("styles.css")
                    return

                self.send_error(HTTPStatus.NOT_FOUND, "Ruta no encontrada")

            def do_POST(self):
                if self.path != "/api/chat":
                    self.send_error(HTTPStatus.NOT_FOUND, "Ruta no encontrada")
                    return

                content_length = int(self.headers.get("Content-Length", "0"))
                raw_body = self.rfile.read(content_length).decode("utf-8")

                try:
                    payload = json.loads(raw_body) if raw_body else {}
                except json.JSONDecodeError:
                    self.respond_json(
                        {"ok": False, "error": "El cuerpo no es JSON válido."},
                        status=HTTPStatus.BAD_REQUEST,
                    )
                    return

                message = str(payload.get("message", "")).strip()
                if not message:
                    self.respond_json(
                        {"ok": False, "error": "El mensaje está vacío."},
                        status=HTTPStatus.BAD_REQUEST,
                    )
                    return

                node.enqueue_user_message(message)
                self.respond_json({"ok": True})

            def log_message(self, format, *args):
                return

            def serve_static(self, filename):
                file_path = node.web_dir / filename
                if not file_path.exists():
                    self.send_error(HTTPStatus.NOT_FOUND, "Archivo no encontrado")
                    return

                content_type = {
                    ".html": "text/html; charset=utf-8",
                    ".js": "application/javascript; charset=utf-8",
                    ".css": "text/css; charset=utf-8",
                }.get(file_path.suffix, "text/plain; charset=utf-8")

                data = file_path.read_bytes()
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def respond_json(self, payload, status=HTTPStatus.OK):
                body = json.dumps(payload).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        self.http_server = ThreadingHTTPServer((self.host, self.port), RequestHandler)
        self.http_thread = threading.Thread(
            target=self.http_server.serve_forever,
            daemon=True,
        )
        self.http_thread.start()

    def enqueue_user_message(self, message):
        with self.state_lock:
            self.append_message({"role": "user", "message": message})
        self.command_queue.put(message)

    def flush_command_queue(self):
        while not self.command_queue.empty():
            message = self.command_queue.get_nowait()
            ros_message = String()
            ros_message.data = message
            self.command_publisher.publish(ros_message)

    def chat_response_callback(self, msg):
        payload = self.safe_json_loads(msg.data, fallback={"message": msg.data})
        if "role" not in payload:
            payload["role"] = "assistant"

        with self.state_lock:
            self.append_message(payload)

    def executor_status_callback(self, msg):
        payload = self.safe_json_loads(
            msg.data,
            fallback={"status": "unknown", "message": msg.data},
        )
        with self.state_lock:
            self.executor_status = payload

    def detected_objects_callback(self, msg):
        payload = self.safe_json_loads(msg.data, fallback={})
        if isinstance(payload, dict):
            with self.state_lock:
                self.visible_objects = sorted(payload.keys())

    def get_state_snapshot(self):
        with self.state_lock:
            return {
                "messages": list(self.messages),
                "visible_objects": list(self.visible_objects),
                "executor_status": dict(self.executor_status),
            }

    def append_message(self, payload):
        self.messages.append(payload)
        self.messages = self.messages[-100:]

    @staticmethod
    def safe_json_loads(raw_data, fallback):
        try:
            return json.loads(raw_data)
        except json.JSONDecodeError:
            return fallback

    def destroy_node(self):
        if self.http_server is not None:
            self.http_server.shutdown()
            self.http_server.server_close()
        return super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = WebBridgeNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
