"""Estado de jobs de ejecución ROS (polling desde la UI)."""

from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

_lock = threading.Lock()
_jobs: Dict[str, "ExecutionJob"] = {}


@dataclass
class ExecutionJob:
    job_id: str
    status: str = "running"
    progress: Dict[str, Any] = field(default_factory=dict)
    final_response: Optional[Dict[str, Any]] = None
    command_text: str = ""


def create_job(*, command_text: str, initial_progress: Optional[dict] = None) -> ExecutionJob:
    job_id = uuid.uuid4().hex[:12]
    job = ExecutionJob(
        job_id=job_id,
        command_text=command_text,
        progress=initial_progress or {"status": "running", "steps": []},
    )
    with _lock:
        _jobs[job_id] = job
    return job


def update_job_progress(job_id: str, progress: dict) -> None:
    with _lock:
        job = _jobs.get(job_id)
        if job is None:
            return
        job.progress = dict(progress)


def finish_job(job_id: str, *, final_response: dict, status: str) -> None:
    with _lock:
        job = _jobs.get(job_id)
        if job is None:
            return
        job.status = status
        job.final_response = final_response


def get_job(job_id: str) -> Optional[ExecutionJob]:
    with _lock:
        return _jobs.get(job_id)


def job_to_public_dict(job: ExecutionJob) -> Dict[str, Any]:
    if job.final_response is not None:
        return dict(job.final_response)
    out = {
        "status": "running",
        "ready": True,
        "job_id": job.job_id,
        "public_message": "Ejecutando movimiento en Gazebo...",
        "current_step": job.progress.get("current_step", "Ejecutando..."),
        "steps": job.progress.get("steps", []),
        "timings": job.progress.get("timings", []),
        "elapsed_s": job.progress.get("elapsed_s", 0.0),
        "can_execute": False,
        "requires_clarification": False,
        "clarification_question": None,
    }
    return out
