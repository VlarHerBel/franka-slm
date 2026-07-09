"""Backend HTTP mínimo para la UI web del SLM (sin dependencias externas).

Usa la librería estándar (`http.server`) para no añadir dependencias nuevas
(FastAPI/uvicorn no son necesarios en esta fase). Reutiliza
`SlmBackendSession` y NO duplica parse_user_command, guardrails, SlotOccupancy
ni command_dispatcher.

Endpoints públicos:
  GET  /api/health          -> estado warm-up (Iniciando asistente... / Asistente listo.)
  POST /api/robot/command   -> interpreta una orden y devuelve mensajes humanos

Además sirve la UI estática de `web_ui/` (mismo origen, sin build).

Arranque:
  python3 -m tfg_planner_slm.web_api --host 0.0.0.0 --port 8010
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from .execution_jobs import create_job, finish_job, get_job, job_to_public_dict, update_job_progress
from .ros_pick_place_cmd import DEFAULT_PICK_PLACE_SCENE_ID
from .ollama_client import DEFAULT_MODEL, DEFAULT_OLLAMA_BASE_URL
from .slm_backend_session import BackendNotReadyError, SlmBackendSession
from .web_public_view import build_public_response
from .web_robot_bridge import attempt_robot_action_in_gazebo

_STATIC_CONTENT_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".js": "application/javascript; charset=utf-8",
    ".mjs": "application/javascript; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".json": "application/json; charset=utf-8",
    ".svg": "image/svg+xml",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".ico": "image/x-icon",
}


def _resolve_web_ui_dir() -> Optional[Path]:
    """Localiza web_ui: prioriza árbol fuente (dev) sobre share instalado."""
    here = Path(__file__).resolve()
    candidates = [
        here.parent.parent / "web_ui",  # src/tfg_planner_slm/web_ui
        here.parent / "web_ui",
    ]
    try:
        from ament_index_python.packages import get_package_share_directory

        candidates.append(
            Path(get_package_share_directory("tfg_planner_slm")) / "web_ui"
        )
    except Exception:
        pass
    for cand in candidates:
        if cand.is_dir():
            return cand.resolve()
    return None


def _safe_static_file(web_ui_dir: Path, rel: str) -> Optional[Path]:
    """Resuelve un fichero estático sin escapar de web_ui (compatible con symlink-install)."""
    rel_path = Path(str(rel).lstrip("/"))
    if rel_path.is_absolute() or ".." in rel_path.parts:
        return None
    raw = web_ui_dir / rel_path
    try:
        raw.relative_to(web_ui_dir)
    except ValueError:
        return None
    resolved = raw.resolve()
    if resolved.is_file():
        return resolved
    return None


class SlmWebApi:
    """Contenedor de la sesión SLM + carpeta estática para el handler HTTP."""

    def __init__(
        self,
        session: SlmBackendSession,
        web_ui_dir: Optional[Path],
        *,
        log_json: bool = False,
        auto_execute: bool = True,
        ros_timeout_sec: float = 300.0,
        clear_table_ros_timeout_sec: float = 300.0,
        scene_id: str = DEFAULT_PICK_PLACE_SCENE_ID,
    ) -> None:
        self.session = session
        self.web_ui_dir = web_ui_dir
        self.log_json = log_json
        self.auto_execute = auto_execute
        self.ros_timeout_sec = float(ros_timeout_sec)
        self.clear_table_ros_timeout_sec = float(clear_table_ros_timeout_sec)
        self.scene_id = str(scene_id or DEFAULT_PICK_PLACE_SCENE_ID).strip()
        self._exec_lock = threading.Lock()


def _make_handler(api: SlmWebApi):
    class Handler(BaseHTTPRequestHandler):
        server_version = "tfg_slm_web_api/1.0"

        def log_message(self, *args: Any) -> None:  # silencio por defecto
            return

        # --- utilidades de respuesta ---
        def _set_cors(self) -> None:
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")

        def _send_json(self, payload: Dict[str, Any], status: int = HTTPStatus.OK) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self._set_cors()
            self.end_headers()
            self.wfile.write(body)

        def _read_json_body(self) -> Tuple[Optional[Dict[str, Any]], str]:
            length = int(self.headers.get("Content-Length", "0") or 0)
            if length <= 0:
                return {}, ""
            raw = self.rfile.read(length).decode("utf-8", errors="replace")
            if not raw.strip():
                return {}, ""
            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                return None, "invalid_json"
            if not isinstance(data, dict):
                return None, "invalid_json"
            return data, ""

        # --- HTTP verbs ---
        def do_OPTIONS(self) -> None:
            self.send_response(HTTPStatus.NO_CONTENT)
            self._set_cors()
            self.end_headers()

        def do_GET(self) -> None:
            path = self.path.split("?", 1)[0]
            if path == "/api/health":
                self._send_json(api.session.get_status())
                return
            if path.startswith("/api/robot/job/"):
                job_id = path.rsplit("/", 1)[-1]
                job = get_job(job_id)
                if job is None:
                    self._send_json(
                        {"status": "error", "ready": True, "public_message": "Job no encontrado."},
                        status=HTTPStatus.NOT_FOUND,
                    )
                    return
                self._send_json(job_to_public_dict(job))
                return
            self._serve_static(path)

        def do_POST(self) -> None:
            path = self.path.split("?", 1)[0]
            if path == "/api/robot/command":
                self._handle_command()
                return
            self._send_json({"error": "not_found"}, status=HTTPStatus.NOT_FOUND)

        # --- handlers ---
        def _handle_command(self) -> None:
            data, err = self._read_json_body()
            if err or data is None:
                self._send_json(
                    {
                        "status": "error",
                        "ready": True,
                        "public_message": "No he recibido una orden válida.",
                    },
                    status=HTTPStatus.BAD_REQUEST,
                )
                return

            text = str(data.get("text", "")).strip()
            if not text:
                self._send_json(
                    {
                        "status": "error",
                        "ready": True,
                        "public_message": "Escribe una orden para el robot.",
                    },
                    status=HTTPStatus.BAD_REQUEST,
                )
                return

            status = api.session.get_status()
            if not status.get("ready"):
                self._send_json(
                    {
                        "status": status.get("status", "warming_up"),
                        "ready": False,
                        "message": status.get("message", "Iniciando asistente..."),
                    },
                    status=HTTPStatus.SERVICE_UNAVAILABLE,
                )
                return

            try:
                result = api.session.parse(text)
            except BackendNotReadyError as exc:
                self._send_json(
                    {"status": "warming_up", "ready": False, "message": str(exc)},
                    status=HTTPStatus.SERVICE_UNAVAILABLE,
                )
                return
            except Exception:
                self._send_json(
                    {
                        "status": "error",
                        "ready": True,
                        "public_message": "No he podido procesar la orden en este momento.",
                    },
                    status=HTTPStatus.INTERNAL_SERVER_ERROR,
                )
                return

            if api.log_json:
                intent = result.parsed.final_intent
                print(
                    "[WEB_API] orden=%r slm_latency_s=%.2f intent_final=%s"
                    % (
                        text,
                        float(result.parsed.ollama_latency_s),
                        json.dumps(intent, ensure_ascii=False) if intent else "null",
                    ),
                    flush=True,
                )

            simulation_unavailable = False
            execution_result = None
            action = result.action
            if (
                api.auto_execute
                and action.execution_supported
                and action.intent in ("pick_place", "clear_table")
            ):
                preview = build_public_response(result, scene_id=api.scene_id)
                job = create_job(
                    command_text=text,
                    initial_progress={
                        "status": "running",
                        "current_step": "Comprobando escena",
                        "steps": preview.get("steps", []),
                    },
                )
                preview["status"] = "running"
                preview["job_id"] = job.job_id
                preview["public_message"] = (
                    preview.get("public_message", "")
                    + " Ejecutando en Gazebo..."
                ).strip()

                def _run_in_background() -> None:
                    def _on_progress(progress: dict) -> None:
                        update_job_progress(job.job_id, progress)

                    with api._exec_lock:
                        exec_result, sim_unavail = attempt_robot_action_in_gazebo(
                            api.session,
                            action,
                            ros_timeout_sec=api.ros_timeout_sec,
                            clear_table_ros_timeout_sec=api.clear_table_ros_timeout_sec,
                            progress_callback=_on_progress,
                            scene_id=api.scene_id,
                        )
                    final = build_public_response(
                        result,
                        simulation_unavailable=sim_unavail,
                        execution_result=exec_result,
                        scene_id=api.scene_id,
                    )
                    finish_job(
                        job.job_id,
                        final_response=final,
                        status=str(final.get("status", "error")),
                    )

                threading.Thread(target=_run_in_background, daemon=True).start()
                self._send_json(preview, status=HTTPStatus.ACCEPTED)
                return

            self._send_json(
                build_public_response(
                    result,
                    simulation_unavailable=simulation_unavailable,
                    execution_result=execution_result,
                    scene_id=api.scene_id,
                )
            )

        def _serve_static(self, path: str) -> None:
            if api.web_ui_dir is None:
                self._send_json(
                    {"error": "web_ui_not_found"}, status=HTTPStatus.NOT_FOUND
                )
                return

            rel = "index.html" if path in ("/", "") else path.lstrip("/")
            target = _safe_static_file(api.web_ui_dir, rel)
            if target is None:
                target = _safe_static_file(api.web_ui_dir, "index.html")
            if target is None:
                self._send_json({"error": "not_found"}, status=HTTPStatus.NOT_FOUND)
                return

            content_type = _STATIC_CONTENT_TYPES.get(
                target.suffix.lower(), "application/octet-stream"
            )
            data = target.read_bytes()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self._set_cors()
            self.end_headers()
            self.wfile.write(data)

    return Handler


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Backend HTTP del SLM para la UI web")
    parser.add_argument("--host", default="0.0.0.0", help="Host de escucha")
    parser.add_argument("--port", type=int, default=8010, help="Puerto de escucha")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Modelo Ollama")
    parser.add_argument(
        "--ollama-url", default=DEFAULT_OLLAMA_BASE_URL, help="URL base de Ollama"
    )
    parser.add_argument("--timeout", type=float, default=90.0, help="Timeout HTTP (s)")
    parser.add_argument(
        "--warmup-count", type=int, default=2, help="Inferencias de warm-up (0 desactiva)"
    )
    parser.add_argument(
        "--no-warmup-on-start",
        action="store_true",
        help="Omitir inferencias Ollama al arrancar; el asistente queda listo al instante",
    )
    parser.add_argument(
        "--log-json",
        action="store_true",
        help="Imprimir JSON final (tras guardrails) en terminal; la UI sigue sin mostrarlo",
    )
    parser.add_argument(
        "--no-auto-execute",
        action="store_true",
        help="Solo interpretar la orden; no lanzar pick_place en Gazebo",
    )
    parser.add_argument(
        "--ros-timeout",
        type=float,
        default=300.0,
        help="Timeout (s) por paso ROS (pick_place o cada objeto de clear_table)",
    )
    parser.add_argument(
        "--open-browser",
        action="store_true",
        help="Abrir la UI en el navegador al arrancar (usa localhost)",
    )
    parser.add_argument(
        "--clear-table-step-timeout",
        type=float,
        default=0.0,
        help="Timeout (s) por objeto en clear_table (0 = usar --ros-timeout)",
    )
    parser.add_argument(
        "--scene-id",
        default=DEFAULT_PICK_PLACE_SCENE_ID,
        help="scene_id para perception_to_pregrasp_test (p. ej. two_boxes_01)",
    )
    return parser


def _argv_for_parse(argv: Optional[list]) -> list:
    """Ignora flags de `ros2 launch` (--ros-args, -r __node:=...)."""
    if argv is None:
        argv = sys.argv[1:]
    out: list = []
    for a in argv:
        if a == "--ros-args":
            break
        out.append(a)
    return out


def main(argv: Optional[list] = None) -> int:
    args = build_arg_parser().parse_args(_argv_for_parse(argv))

    session = SlmBackendSession(
        model=args.model,
        ollama_url=args.ollama_url,
        timeout_s=args.timeout,
        warmup_count=args.warmup_count,
    )
    session.configure_scene(str(args.scene_id))
    web_ui_dir = _resolve_web_ui_dir()
    clear_table_timeout = (
        float(args.clear_table_step_timeout)
        if float(args.clear_table_step_timeout) > 0.0
        else float(args.ros_timeout)
    )
    api = SlmWebApi(
        session=session,
        web_ui_dir=web_ui_dir,
        log_json=args.log_json,
        auto_execute=not args.no_auto_execute,
        ros_timeout_sec=args.ros_timeout,
        clear_table_ros_timeout_sec=clear_table_timeout,
        scene_id=str(args.scene_id),
    )

    if args.no_warmup_on_start:
        session.warmup_count = 0
    session.start_warmup(background=True)

    handler = _make_handler(api)
    httpd = ThreadingHTTPServer((args.host, args.port), handler)

    ui_info = str(web_ui_dir) if web_ui_dir else "(web_ui no encontrada)"
    print("[WEB_API] escuchando en http://%s:%d" % (args.host, args.port), flush=True)
    print("[WEB_API] UI estática: %s" % ui_info, flush=True)
    print(
        "[WEB_API] warm-up=%s model=%s log_json=%s auto_execute=%s scene_id=%s"
        % (
            "off" if args.no_warmup_on_start else "on",
            args.model,
            str(args.log_json).lower(),
            str(api.auto_execute).lower(),
            api.scene_id,
        ),
        flush=True,
    )
    if args.open_browser:
        browse_host = "127.0.0.1" if args.host in ("0.0.0.0", "::", "") else args.host
        browse_url = "http://%s:%d" % (browse_host, int(args.port))
        print("[WEB_API] abriendo navegador: %s" % browse_url, flush=True)
        threading.Timer(0.8, lambda: webbrowser.open(browse_url)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[WEB_API] detenido", flush=True)
    finally:
        httpd.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
