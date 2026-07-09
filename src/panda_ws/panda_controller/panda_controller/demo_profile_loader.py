"""Perfiles demo con defaults ROS y referencia a candidato golden."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from panda_controller.demo_golden_pick_candidate import resolve_golden_candidate_path
from panda_controller.demo_golden_execution_candidate import (
    default_full_execution_golden_path,
)
from panda_vision.spawn.demo_scene_presets import demo_scene_policy_scene_id_for_preset


def default_demo_profiles_dir() -> str:
    fallback = str(Path(__file__).resolve().parent.parent / "config" / "demo_profiles")
    try:
        from ament_index_python.packages import get_package_share_directory

        share = os.path.join(
            get_package_share_directory("panda_controller"), "config", "demo_profiles"
        )
        if os.path.isdir(share):
            return share
    except Exception:
        pass
    return fallback


def demo_profile_yaml_path(profile_id: str, *, profiles_dir: Optional[str] = None) -> str:
    pid = str(profile_id or "").strip().lower()
    base = str(profiles_dir or default_demo_profiles_dir())
    return os.path.join(base, f"{pid}.yaml")


def load_demo_profile(profile_id: str, *, profiles_dir: Optional[str] = None) -> Optional[Dict[str, Any]]:
    path = demo_profile_yaml_path(profile_id, profiles_dir=profiles_dir)
    if not os.path.isfile(path):
        return None
    with open(path, "r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    return normalize_demo_profile(raw, source_file=path)


def normalize_demo_profile(raw: Dict[str, Any], *, source_file: str = "") -> Optional[Dict[str, Any]]:
    if not isinstance(raw, dict):
        return None
    profile_id = str(raw.get("profile_id", "")).strip()
    scene_id = str(raw.get("scene_id", "")).strip().lower()
    target_label = str(raw.get("target_label", "")).strip().lower()
    if not profile_id or not scene_id or not target_label:
        return None
    params = raw.get("parameters") or {}
    if not isinstance(params, dict):
        params = {}
    return {
        "profile_id": profile_id,
        "scene_id": scene_id,
        "target_label": target_label,
        "layout_version": str(raw.get("layout_version", "")).strip(),
        "golden_candidate_path": str(raw.get("golden_candidate_path", "")).strip(),
        "parameters": dict(params),
        "source_file": str(source_file),
    }


_DEMO_GOLDEN_PROFILE_BY_LABEL: Dict[str, str] = {
    "cracker_box": "demo_scene_02_cracker_box",
    "chips_can": "demo_scene_02_chips_can",
    "sugar_box": "demo_scene_02_sugar_box",
    "mustard_bottle": "demo_scene_02_mustard_bottle",
}

# Parámetros del perfil golden que no deben aplicarse en escenas operativas distintas.
_OPERATIONAL_PROFILE_STRIP_PARAMS: frozenset = frozenset(
    {
        "demo_authoritative_scene",
        "clear_table_manual_step",
        "scene_id",
        "place_slot_index",
        "demo_persist_completed_objects",
        "demo_completed_state_file",
        "use_golden_execution_candidate",
        "require_golden_execution_candidate",
        "execution_mode",
        "mustard_pregrasp_ik_joint_goal",
    }
)


def _adapt_operational_demo_profile(
    profile: Dict[str, Any], *, scene_id: str
) -> Dict[str, Any]:
    """Reutiliza defaults de grasp de demo_scene_02 en escenas chips_mustard_*."""
    sid = str(scene_id or "").strip().lower()
    params = {
        k: v
        for k, v in dict(profile.get("parameters") or {}).items()
        if k not in _OPERATIONAL_PROFILE_STRIP_PARAMS
    }
    label = str(profile.get("target_label") or "").strip().lower()
    return {
        **profile,
        "scene_id": sid,
        "profile_id": f"{sid}_{label}_operational",
        "golden_candidate_path": "",
        "parameters": params,
    }


def resolve_demo_profile(
    scene_id: str,
    target_label: str,
    *,
    profiles_dir: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    sid = str(scene_id or "").strip().lower()
    label = str(target_label or "").strip().lower()
    profile_id = _DEMO_GOLDEN_PROFILE_BY_LABEL.get(label)
    if profile_id is None:
        return None
    profile = load_demo_profile(profile_id, profiles_dir=profiles_dir)
    if profile is None:
        return None
    expected_sid = str(profile.get("scene_id") or "").strip().lower()
    if not sid or sid == expected_sid:
        return profile
    parent_sid = demo_scene_policy_scene_id_for_preset(sid)
    if parent_sid == expected_sid:
        return _adapt_operational_demo_profile(profile, scene_id=sid)
    if label == "mustard_bottle" and sid.startswith("chips_mustard"):
        return _adapt_operational_demo_profile(profile, scene_id=sid)
    return None


def resolve_golden_path_from_profile(profile: Dict[str, Any]) -> str:
    rel = str(profile.get("golden_candidate_path") or "").strip()
    if rel:
        return resolve_golden_candidate_path(rel)
    scene_id = demo_scene_policy_scene_id_for_preset(
        str(profile.get("scene_id") or "")
    )
    return resolve_golden_candidate_path(
        "demo_candidate_cache/%s_%s_golden.yaml"
        % (scene_id, profile.get("target_label"))
    )


def resolve_golden_execution_path_from_profile(profile: Dict[str, Any]) -> str:
    rel = str(profile.get("golden_execution_candidate_path") or "").strip()
    if rel:
        return resolve_golden_candidate_path(rel)
    scene_id = demo_scene_policy_scene_id_for_preset(
        str(profile.get("scene_id") or "")
    )
    return default_full_execution_golden_path(
        scene_id,
        str(profile.get("target_label") or ""),
        slot_index=0,
    )


def log_demo_profile_load(logger: Any, profile: Optional[Dict[str, Any]]) -> None:
    if profile is None:
        logger.info(
            "[DEMO_PROFILE_LOAD]\n"
            "scene_id=\n"
            "target_label=\n"
            "profile=\n"
            "result=FAIL"
        )
        return
    logger.info(
        "[DEMO_PROFILE_LOAD]\n"
        "scene_id=%s\n"
        "target_label=%s\n"
        "profile=%s\n"
        "result=OK"
        % (
            profile.get("scene_id", ""),
            profile.get("target_label", ""),
            profile.get("profile_id", ""),
        )
    )


def _values_equal(a: Any, b: Any) -> bool:
    if isinstance(a, float) or isinstance(b, float):
        try:
            return abs(float(a) - float(b)) < 1e-9
        except (TypeError, ValueError):
            pass
    return a == b


def log_demo_profile_parameter_apply(logger: Any, node: Any, profile: Dict[str, Any]) -> None:
    params = profile.get("parameters") or {}
    for name, profile_value in sorted(params.items()):
        overridden = False
        actual_value = profile_value
        if node.has_parameter(name):
            actual_value = node.get_parameter(name).value
            overridden = not _values_equal(actual_value, profile_value)
        logger.info(
            "[DEMO_PROFILE_PARAM_APPLY]\n"
            "param=%s\n"
            "value=%s\n"
            "source=profile\n"
            "overridden_by_cli=%s"
            % (
                name,
                actual_value,
                str(bool(overridden)).lower(),
            )
        )
