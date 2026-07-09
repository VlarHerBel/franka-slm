"""Tests: bookkeeping post-place sin crashear si candidate es None."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Set
from unittest.mock import MagicMock

from panda_controller.demo_object_order_policy import (
    load_demo_completed_state,
    save_demo_completed_state,
)
from panda_controller.demo_pick_bookkeeping import resolve_candidate_for_cleanup


class _CleanupProbe:
    """Stub mínimo de cleanup post-place."""

    def __init__(self) -> None:
        self.logs: List[str] = []
        self._current_target_collision_spec: Optional[Dict[str, Any]] = None
        self._current_target_collision_id: Optional[str] = None
        self._last_target_collision_id: Optional[str] = None
        self._planning_scene_object_ids: Set[str] = set()
        self._current_target_attached = True
        self._current_attached_object_id = "obj"
        self._force_detach_calls: List[str] = []
        self._remove_calls: List[str] = []

    def get_logger(self) -> MagicMock:
        logger = MagicMock()

        def _info(msg: str, *args: Any) -> None:
            self.logs.append(str(msg) % args if args else str(msg))

        logger.info = _info
        logger.warning = _info
        return logger

    def _resolve_target_entity_name(self, _det: Any) -> str:
        return ""

    def _force_detach_planning_scene_object(self, oid: str) -> None:
        self._force_detach_calls.append(oid)

    def _remove_collision_object(self, oid: str) -> None:
        self._remove_calls.append(oid)

    def _cleanup_demo_target_collision_after_place(
        self, candidate: Optional[Dict[str, Any]] = None
    ) -> bool:
        if candidate is None or not isinstance(candidate, dict):
            self.get_logger().info(
                "[DEMO_TARGET_COLLISION_CLEANUP]\n"
                "result=SKIP\n"
                "reason=no_candidate"
            )
            return True
        label = str(candidate.get("label", "")).strip()
        self.get_logger().info(
            "[TARGET_COLLISION_CLEANUP]\n"
            "stage=after_place\n"
            "label=%s\n"
            "result=REMOVED"
            % (label or "n/a")
        )
        self._current_target_attached = False
        return True


def test_mark_demo_pick_completed_with_place_cand_not_candidate() -> None:
    place_cand = {
        "label": "chips_can",
        "x": -0.37,
        "y": -0.10,
        "place_slot_index": 2,
        "place_slot_label": "slot_3",
        "release_tcp_z": 0.41,
    }
    resolved = resolve_candidate_for_cleanup(
        place_cand=place_cand,
        candidate=None,
    )
    assert resolved is not None
    assert resolved["label"] == "chips_can"
    assert resolved["y"] == -0.10


def test_cleanup_demo_target_collision_none_candidate_does_not_crash() -> None:
    probe = _CleanupProbe()
    assert probe._cleanup_demo_target_collision_after_place(None) is True
    assert any("DEMO_TARGET_COLLISION_CLEANUP" in line for line in probe.logs)
    assert any("reason=no_candidate" in line for line in probe.logs)


def test_completed_state_saved_even_if_cleanup_candidate_missing(
    tmp_path,
) -> None:
    """Simula orden mark: guardar estado antes de cleanup con candidato ausente."""
    path = str(tmp_path / "demo_state.json")
    place_cand = {
        "release_label": "cracker_box",
        "x": -0.37,
        "y": 0.10,
        "place_slot_index": 0,
        "place_slot_label": "slot_1",
        "release_tcp_z": 0.42,
    }
    payload_label = "cracker_box"
    deposits = [
        {
            "label": payload_label,
            "entity": "runtime_ycb_cracker_1",
            "slot_name": "slot_1",
            "slot_index": 0,
            "x": -0.37,
            "y": 0.10,
            "release_tcp_z": 0.42,
        }
    ]
    save_demo_completed_state(
        path,
        completed_entities={"runtime_ycb_cracker_1"},
        completed_labels={payload_label},
        completed_deposits=deposits,
    )
    probe = _CleanupProbe()
    cleanup_candidate = resolve_candidate_for_cleanup(
        place_cand=place_cand,
        candidate=None,
    )
    assert probe._cleanup_demo_target_collision_after_place(cleanup_candidate) is True
    _, labels, loaded = load_demo_completed_state(path)
    assert payload_label in labels
    assert len(loaded) == 1
    assert loaded[0]["slot_name"] == "slot_1"
