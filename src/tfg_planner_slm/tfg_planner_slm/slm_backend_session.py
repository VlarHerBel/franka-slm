"""Sesión backend para UI: warm-up controlado + conversación SLM v1.1.

Este módulo no sirve HTML ni decide la interfaz. Expone un estado sencillo para
que el frontend muestre "Iniciando asistente..." hasta que Ollama esté caliente.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any, Dict, Optional

from .command_dispatcher import InternalAction, dispatch_command
from .intent_parser import ParsedCommandResult, parse_user_command
from .ollama_client import DEFAULT_MODEL, DEFAULT_OLLAMA_BASE_URL, warmup_model
from .slot_state import SlotOccupancy

STATUS_IDLE = "idle"
STATUS_WARMING_UP = "warming_up"
STATUS_READY = "ready"
STATUS_ERROR = "error"

WARMUP_UI_MESSAGE = "Iniciando asistente..."


class BackendNotReadyError(RuntimeError):
    """Se lanza si la UI intenta enviar órdenes antes de que el SLM esté listo."""


@dataclass
class BackendCommandResult:
    parsed: ParsedCommandResult
    action: InternalAction


class SlmBackendSession:
    """Estado persistente del backend SLM para una conversación de usuario."""

    def __init__(
        self,
        *,
        model: str = DEFAULT_MODEL,
        ollama_url: str = DEFAULT_OLLAMA_BASE_URL,
        timeout_s: float = 90.0,
        warmup_count: int = 2,
    ) -> None:
        self.model = model
        self.ollama_url = ollama_url
        self.timeout_s = timeout_s
        self.warmup_count = max(0, int(warmup_count))
        self.slot_occupancy = SlotOccupancy()
        self.scene_id = ""

    def configure_scene(self, scene_id: str) -> None:
        """Precarga ocupación de cajones según initial_deposits del YAML de escena."""
        from .deposit_scene_loader import seed_slot_occupancy_from_scene

        self.scene_id = str(scene_id or "").strip()
        self.slot_occupancy.reset()
        seeded = seed_slot_occupancy_from_scene(self.slot_occupancy, self.scene_id)
        if seeded > 0:
            print(
                "[DEPOSIT_STATE] scene=%s preloaded_slots=%d status=%s"
                % (
                    self.scene_id,
                    seeded,
                    self.slot_occupancy.format_status(),
                ),
                flush=True,
            )

        self._lock = threading.Lock()
        self._status = STATUS_IDLE
        self._status_message = WARMUP_UI_MESSAGE
        self._warmup_error = ""
        self._warmup_thread: Optional[threading.Thread] = None

    def get_status(self) -> Dict[str, Any]:
        """Snapshot serializable para un endpoint tipo GET /api/health."""
        with self._lock:
            return {
                "status": self._status,
                "ready": self._status == STATUS_READY,
                "message": self._status_message,
                "error": self._warmup_error,
            }

    def start_warmup(self, *, background: bool = True) -> None:
        """Arranca el warm-up una sola vez; con warmup_count=0 pasa a ready sin Ollama."""
        with self._lock:
            if self._status in (STATUS_WARMING_UP, STATUS_READY):
                return
            if self.warmup_count <= 0:
                self._status = STATUS_READY
                self._status_message = "Asistente listo."
                self._warmup_error = ""
                return
            self._status = STATUS_WARMING_UP
            self._status_message = WARMUP_UI_MESSAGE
            self._warmup_error = ""

        if background:
            self._warmup_thread = threading.Thread(
                target=self._run_warmup,
                name="slm-warmup",
                daemon=True,
            )
            self._warmup_thread.start()
        else:
            self._run_warmup()

    def _run_warmup(self) -> None:
        try:
            results = warmup_model(
                self.model,
                self.warmup_count,
                timeout_s=self.timeout_s,
                ollama_url=self.ollama_url,
            )
            failed = [
                r.request_error or r.parse_error or "invalid_warmup_json"
                for r in results
                if r.request_error or r.parsed_json is None
            ]
            if failed:
                self._set_error("; ".join(failed))
                return
            with self._lock:
                self._status = STATUS_READY
                self._status_message = "Asistente listo."
        except Exception as exc:  # pragma: no cover - protección de backend
            self._set_error(str(exc))

    def _set_error(self, message: str) -> None:
        with self._lock:
            self._status = STATUS_ERROR
            self._status_message = "No se pudo iniciar el asistente."
            self._warmup_error = message

    def parse(self, text: str) -> BackendCommandResult:
        """Procesa una orden solo cuando el warm-up ya ha terminado correctamente."""
        status = self.get_status()
        if not status["ready"]:
            raise BackendNotReadyError(status["message"])

        parsed = parse_user_command(
            text,
            model=self.model,
            timeout_s=self.timeout_s,
            ollama_url=self.ollama_url,
            slot_occupancy=self.slot_occupancy,
        )
        action = dispatch_command(parsed.final_intent)
        return BackendCommandResult(parsed=parsed, action=action)

