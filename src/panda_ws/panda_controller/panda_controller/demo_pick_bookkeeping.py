"""Helpers de bookkeeping post-place (sin dependencias ROS)."""

from __future__ import annotations

from typing import Any, Dict, Optional


def resolve_candidate_for_cleanup(
    *,
    place_cand: Optional[Dict[str, Any]] = None,
    candidate: Optional[Dict[str, Any]] = None,
    snapshot_candidate: Optional[Dict[str, Any]] = None,
    current_candidate: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    """Candidato para cleanup post-place; prioriza place_cand."""
    for src in (place_cand, candidate, snapshot_candidate, current_candidate):
        if isinstance(src, dict) and src:
            return dict(src)
    return None
