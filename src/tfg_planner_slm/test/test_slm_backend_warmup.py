"""Warm-up del backend SLM."""

from tfg_planner_slm.slm_backend_session import (
    STATUS_READY,
    SlmBackendSession,
)


def test_no_warmup_marks_session_ready() -> None:
    session = SlmBackendSession(warmup_count=0)
    session.configure_scene("deposit_02_cracker_chips")
    session.start_warmup(background=False)
    status = session.get_status()
    assert status["status"] == STATUS_READY
    assert status["ready"] is True
