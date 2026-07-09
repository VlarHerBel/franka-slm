"""Seguimiento de progreso y tiempos durante ejecución ROS (parseo de logs)."""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from .ros_pick_place_cmd import CLEAR_TABLE_PICK_ORDER, resolve_clear_table_pick_order

ProgressCallback = Callable[[Dict[str, Any]], None]

_HUMAN_OBJECT_NAMES: Dict[str, str] = {
    "cracker_box": "caja de galletas",
    "sugar_box": "caja de azúcar",
    "chips_can": "lata de patatas",
    "mustard_bottle": "bote de mostaza",
}


def human_object_name(target_label: Optional[str]) -> str:
    if not target_label:
        return "el objeto"
    return _HUMAN_OBJECT_NAMES.get(str(target_label).strip().lower(), target_label)


PHASE_LABELS: Dict[str, str] = {
    "object_start": "Recogiendo {obj} ({n}/{total})",
    "safe_above": "Posición segura sobre objeto",
    "joint7": "Alineación joint 7 (pinza)",
    "grasp": "Descenso y agarre",
    "transport": "Transporte al cajón",
    "place": "Colocando en cajón",
    "home": "Volviendo a home",
    "object_done": "{obj} completado",
    "table_done": "Mesa recogida",
    "failed": "Movimiento fallido",
}

OBJECT_PHASE_ORDER: Tuple[str, ...] = (
    "object_start",
    "safe_above",
    "joint7",
    "grasp",
    "transport",
    "place",
    "home",
    "object_done",
)

_OBJECT_LABEL_FROM_LOG = re.compile(
    r"(?:selected_label|next_label)=([a-z_]+)", re.I
)
_TIMING_LOG = re.compile(r"\[PERCEPTION_PROFILE\].*total=([\d.]+)", re.I)

_PROGRESS_LOG_TAGS = (
    "[CLEAR_TABLE_MANUAL_STEP]",
    "[CLEAR_TABLE_OBJECTS]",
    "[DEMO_REMAINING_OBJECTS]",
    "[PICK_PRELUDE]",
    "[OBJECT_SAFE_ABOVE",
    "[GRIPPER_AXIS",
    "[CHIPS_CAN_FAST_JOINT7",
    "[ATTACH]",
    "[POST_PICK_TRANSPORT",
    "[PLACE_CANDIDATE]",
    "[PLACE]",
    "[CLEAR_TABLE_TARGET_DONE]",
    "[DEMO_TARGET_DONE]",
    "[CLEAR_TABLE_COMPLETE]",
    "[CYCLE]",
    "[PAIRED_GRID",
    "[POST_PRELUDE",
    "[GRIPPER_AXIS_ALIGN]",
    "[PLAN_BEFORE_MOTION_FAIL]",
    "[PERCEPTION_PROFILE]",
    "Traceback",
    "AttributeError",
    "Fallo etapa",
)


@dataclass
class PhaseTiming:
    phase: str
    label: str
    started_at: float
    ended_at: Optional[float] = None

    @property
    def duration_s(self) -> Optional[float]:
        if self.ended_at is None:
            return None
        return max(0.0, self.ended_at - self.started_at)


@dataclass
class ObjectStepTiming:
    step_index: int
    label: str
    duration_s: float
    ok: bool


@dataclass
class ExecutionProgressTracker:
    """Estado mutable compartido entre executor y web (job polling)."""

    intent: str = "clear_table"
    total_objects: int = 4
    progress_index_offset: int = 0
    pick_order: Tuple[str, ...] = field(default_factory=lambda: CLEAR_TABLE_PICK_ORDER)
    object_index: int = 0
    object_label: str = ""
    current_phase: str = "preflight"
    current_label: str = "Comprobando escena"
    started_at: float = field(default_factory=time.perf_counter)
    phases: List[PhaseTiming] = field(default_factory=list)
    step_rows: List[Dict[str, Any]] = field(default_factory=list)
    object_timings: List[ObjectStepTiming] = field(default_factory=list)
    done: bool = False
    success: bool = False
    error_message: str = ""
    _on_update: Optional[ProgressCallback] = None
    _last_phase_key: Optional[str] = None
    _active_row_key: Optional[str] = None
    _motion_started: bool = False
    _motion_aborted: bool = False

    def set_callback(self, cb: Optional[ProgressCallback]) -> None:
        self._on_update = cb

    def _display_index(self, step_idx: int) -> int:
        return int(step_idx) + int(self.progress_index_offset)

    def _emit(self) -> None:
        if self._on_update is None:
            return
        self._on_update(self.to_public_dict())

    def _phase_rank(self, phase_key: str) -> int:
        try:
            return OBJECT_PHASE_ORDER.index(phase_key)
        except ValueError:
            return len(OBJECT_PHASE_ORDER) + 1

    def _row_key(self, object_index: int, phase_key: str) -> str:
        return "%d:%s" % (int(object_index), str(phase_key))

    def _insert_index_for_row(self, object_index: int, phase_key: str) -> int:
        target_rank = self._phase_rank(phase_key)
        for idx, row in enumerate(self.step_rows):
            row_obj = int(row.get("object_index") or 0)
            row_phase = str(row.get("phase_key") or "")
            if row_obj > object_index:
                return idx
            if row_obj == object_index:
                if self._phase_rank(row_phase) > target_rank:
                    return idx
        return len(self.step_rows)

    def _mark_active_row_done(self) -> None:
        if not self._active_row_key:
            return
        for row in self.step_rows:
            if row.get("row_key") == self._active_row_key and row.get("state") == "active":
                row["state"] = "done"
                break
        self._active_row_key = None

    def _upsert_object_row(
        self,
        *,
        object_index: int,
        phase_key: str,
        label: str,
        state: str,
    ) -> None:
        row_key = self._row_key(object_index, phase_key)
        existing = next((r for r in self.step_rows if r.get("row_key") == row_key), None)
        if existing is not None:
            existing["label"] = label
            existing["state"] = state
            self._active_row_key = row_key if state == "active" else None
            return
        row = {
            "label": label,
            "state": state,
            "object_index": int(object_index),
            "phase_key": phase_key,
            "row_key": row_key,
        }
        insert_at = self._insert_index_for_row(object_index, phase_key)
        self.step_rows.insert(insert_at, row)
        self._active_row_key = row_key if state == "active" else None

    def _close_active_timing(self) -> None:
        now = time.perf_counter()
        if self.phases and self.phases[-1].ended_at is None:
            self.phases[-1].ended_at = now

    def _go_phase(self, phase_key: str, label: str) -> None:
        if not self.object_index:
            return
        if self._last_phase_key == phase_key:
            return
        self._close_active_timing()
        now = time.perf_counter()
        self.current_phase = phase_key
        self.current_label = label
        self._last_phase_key = phase_key
        self.phases.append(PhaseTiming(phase=phase_key, label=label, started_at=now))
        self._mark_active_row_done()
        self._upsert_object_row(
            object_index=self.object_index,
            phase_key=phase_key,
            label=label,
            state="active",
        )
        self._emit()

    def mark_interpretation_done(self) -> None:
        self.step_rows = [
            {"label": "Orden recibida", "state": "done", "object_index": 0, "phase_key": "global"},
            {"label": "Interpretando petición", "state": "done", "object_index": 0, "phase_key": "global"},
            {"label": "Validando orden de forma segura", "state": "done", "object_index": 0, "phase_key": "global"},
            {"label": "Comprobando escena", "state": "done", "object_index": 0, "phase_key": "global"},
        ]
        self.current_label = "Comprobando escena"
        self._emit()

    def on_step_start(self, step_idx: int, label: str) -> None:
        display_idx = self._display_index(step_idx)
        self.object_index = display_idx
        self.object_label = str(label).strip().lower()
        self._last_phase_key = None
        self._motion_started = False
        self._motion_aborted = False
        obj_human = human_object_name(self.object_label)
        start_label = PHASE_LABELS["object_start"].format(
            obj=obj_human,
            n=display_idx,
            total=int(self.total_objects),
        )
        self._upsert_object_row(
            object_index=display_idx,
            phase_key="object_start",
            label=start_label,
            state="active",
        )
        self.current_label = start_label
        self._active_row_key = self._row_key(display_idx, "object_start")
        self._emit()

    def _apply_object_label_from_log(self, label: str) -> None:
        norm = str(label).strip().lower()
        order = tuple(self.pick_order) if self.pick_order else CLEAR_TABLE_PICK_ORDER
        if norm not in order:
            return
        if norm == self.object_label and self.object_index > 0:
            return
        self.object_label = norm
        self.object_index = order.index(norm) + 1

    def on_log_line(self, line: str, *, stream: str = "stdout") -> None:
        raw = (line or "").strip()
        if not raw:
            return
        lower = raw.lower()

        if any(tag in raw for tag in _PROGRESS_LOG_TAGS):
            elapsed = time.perf_counter() - self.started_at
            print(
                "[ROS_EXECUTOR][%.1fs][%s] %s" % (elapsed, stream, raw),
                flush=True,
            )

        if "Traceback" in raw or "AttributeError" in raw or "Process exited with failure" in raw:
            self._mark_failed("Error interno del controlador ROS")

        mobj = _OBJECT_LABEL_FROM_LOG.search(raw)
        if mobj:
            self._apply_object_label_from_log(mobj.group(1))

        if "[OBJECT_SAFE_ABOVE" in raw or "[PICK_PRELUDE]" in raw:
            if not self._motion_aborted:
                self._go_phase("safe_above", PHASE_LABELS["safe_above"])

        if "[GRIPPER_AXIS" in raw or "[CHIPS_CAN_FAST_JOINT7" in raw:
            if not self._motion_aborted:
                self._motion_started = True
                self._go_phase("joint7", PHASE_LABELS["joint7"])

        if "[ATTACH]" in raw:
            self._motion_started = True
            self._go_phase("grasp", PHASE_LABELS["grasp"])
        elif "Fallo etapa grasp" in raw:
            self._go_phase("grasp", PHASE_LABELS["grasp"])

        if "[POST_PICK_TRANSPORT" in raw:
            self._motion_started = True
            self._go_phase("transport", PHASE_LABELS["transport"])

        if "[PLACE_CANDIDATE]" in raw and "result=OK" in raw:
            self._motion_started = True
            self._go_phase("place", PHASE_LABELS["place"])
        elif "[DETACH]" in raw or (
            "[PLACE]" in raw and "deterministic sequence completed" in lower
        ):
            self._motion_started = True
            self._go_phase("place", PHASE_LABELS["place"])

        if (
            not self._motion_aborted
            and self._motion_started
            and "[CYCLE]" in raw
            and (
                "returning home" in lower
                or "home reached" in lower
                or "home already reached" in lower
            )
        ):
            self._go_phase("home", PHASE_LABELS["home"])

        if "[PLAN_BEFORE_MOTION_FAIL]" in raw:
            self._motion_aborted = True
            self._mark_failed("Planificación fallida (sin movimiento)")

        if "[CLEAR_TABLE_TARGET_DONE]" in raw or "[DEMO_TARGET_DONE]" in raw:
            obj_human = human_object_name(self.object_label)
            done_label = PHASE_LABELS["object_done"].format(obj=obj_human)
            self._close_active_timing()
            self._mark_active_row_done()
            self._upsert_object_row(
                object_index=self.object_index,
                phase_key="object_done",
                label=done_label,
                state="done",
            )
            self._last_phase_key = None
            self._emit()

        if "[CLEAR_TABLE_COMPLETE]" in raw and "result=OK" in raw:
            self.done = True
            self.success = True
            self._close_active_timing()
            self._mark_active_row_done()
            self.step_rows.append(
                {"label": PHASE_LABELS["table_done"], "state": "done", "object_index": 0, "phase_key": "global"}
            )
            self._emit()

        if "[CLEAR_TABLE_MANUAL_STEP]" in raw and "result=FAIL" in raw:
            self._mark_failed("Paso clear_table fallido")

        if "[GRIPPER_AXIS_ALIGN]" in raw and "pick aborted" in lower:
            self._mark_failed("Alineación joint 7 no alcanzada")

        if "[POST_PRELUDE_PREGRASP_REPLAN]" in raw and "result=FAIL" in raw:
            self._mark_failed("Pregrasp no alcanzable")

        tmatch = _TIMING_LOG.search(raw)
        if tmatch:
            print(
                "[ROS_EXECUTOR][timing] perception_cycle_total_ms=%s"
                % tmatch.group(1),
                flush=True,
            )

    def _mark_failed(self, message: str) -> None:
        self.done = True
        self.success = False
        self.error_message = message
        self._close_active_timing()
        self._mark_active_row_done()
        if not any(r.get("state") == "error" for r in self.step_rows):
            self.step_rows.append(
                {"label": PHASE_LABELS["failed"], "state": "error", "object_index": self.object_index, "phase_key": "failed"}
            )
        self._emit()

    def on_step_finished(self, step_idx: int, label: str, duration_s: float, ok: bool) -> None:
        display_idx = self._display_index(step_idx)
        obj_human = human_object_name(label)
        print(
            "[ROS_EXECUTOR] clear_table step_done=%d label=%s duration_s=%.2f ok=%s"
            % (display_idx, label, duration_s, str(ok).lower()),
            flush=True,
        )
        self.object_timings.append(
            ObjectStepTiming(
                step_index=display_idx,
                label=label,
                duration_s=float(duration_s),
                ok=bool(ok),
            )
        )
        done_label = PHASE_LABELS["object_done"].format(obj=obj_human)
        self._close_active_timing()
        self._mark_active_row_done()
        if ok:
            self._upsert_object_row(
                object_index=display_idx,
                phase_key="object_done",
                label=done_label,
                state="done",
            )
        else:
            self.done = True
            self.success = False
            self.error_message = "Fallo en %s (%.1fs)" % (obj_human, duration_s)
            if not any(r.get("state") == "error" for r in self.step_rows):
                self.step_rows.append(
                    {
                        "label": PHASE_LABELS["failed"],
                        "state": "error",
                        "object_index": step_idx,
                        "phase_key": "failed",
                    }
                )
        self._last_phase_key = None
        self._emit()

    def to_public_dict(self) -> Dict[str, Any]:
        timings: List[Dict[str, Any]] = []
        for ot in self.object_timings:
            timings.append(
                {
                    "phase": "object_step",
                    "label": "%s (%d/%d)"
                    % (
                        human_object_name(ot.label),
                        ot.step_index,
                        int(self.total_objects),
                    ),
                    "duration_s": round(ot.duration_s, 2),
                    "ok": ot.ok,
                }
            )
        total_s = time.perf_counter() - self.started_at
        status = "running"
        if self.done:
            status = "success" if self.success else "failed"
        public_steps = [
            {"label": str(row.get("label") or ""), "state": str(row.get("state") or "")}
            for row in self.step_rows
        ]
        return {
            "status": status,
            "current_step": self.current_label,
            "steps": public_steps,
            "timings": timings,
            "elapsed_s": round(total_s, 2),
            "object_index": self.object_index,
            "object_label": self.object_label,
            "error_message": self.error_message,
        }
