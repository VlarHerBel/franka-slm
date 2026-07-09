"""Entrada safe multiobjeto para sugar_box (sin ROS)."""

from __future__ import annotations

from typing import AbstractSet, List, Optional, Sequence, Tuple

DEMO_SCENE_02_SUGAR_BOX_REMAINING_COMPLETED_LABELS: frozenset = frozenset(
    {"cracker_box", "chips_can"}
)
DEMO_SCENE_02_3OBJ_PRIOR_LABELS: frozenset = frozenset({"cracker_box", "chips_can"})
SUGAR_GOLDEN_DEBUG_SCENE_IDS = frozenset(
    {
        "demo_scene_02_remaining_sugar_mustard",
        "deposit_02_cracker_chips",
    }
)


def sugar_box_scene_allows_golden(scene_id: str) -> bool:
    return str(scene_id or "").strip().lower() in SUGAR_GOLDEN_DEBUG_SCENE_IDS


def sugar_box_hold_gripper_frozen_after_grasp(
    *,
    label: str,
    scene_id: str,
) -> bool:
    """Tras grasp OK: mantener dedos congelados durante lift/transporte (demo_scene_02)."""
    if str(label or "").strip().lower() != "sugar_box":
        return False
    from panda_controller.demo_golden_pick_candidate import demo_golden_policy_scene_id

    return demo_golden_policy_scene_id(str(scene_id or "").strip()) == "demo_scene_02"


DEFAULT_SAFE_ENTRY_CLEARANCE_STEPS_M = (
    0.100,
    0.090,
    0.080,
    0.070,
    0.060,
)


def sugar_box_nominal_safe_tcp_z(
    top_z_m: float,
    clearance_above_top_m: float,
) -> float:
    return float(top_z_m) + float(clearance_above_top_m)


def sugar_box_safe_entry_tcp_z_candidates(
    top_z_m: float,
    *,
    min_clearance_above_top_m: float = 0.055,
    clearance_steps_m: Optional[Sequence[float]] = None,
) -> List[float]:
    """Candidatos TCP Z ordenados de mayor a menor clearance (sin mezclar pregrasp+extra)."""
    steps = clearance_steps_m or DEFAULT_SAFE_ENTRY_CLEARANCE_STEPS_M
    min_c = float(min_clearance_above_top_m)
    out: List[float] = []
    seen: set = set()
    for step in steps:
        clearance = max(float(step), min_c)
        z = float(top_z_m) + clearance
        key = round(z, 4)
        if key in seen:
            continue
        seen.add(key)
        out.append(z)
    return out


def resolve_sugar_box_operative_center_xy(
    candidate: dict,
    *,
    fallback_xy: Sequence[float],
) -> Tuple[float, float, str]:
    for key in (
        "grasp_center_base",
        "chosen_target_center_base",
        "position",
        "known_box_center_base",
    ):
        raw = candidate.get(key)
        if isinstance(raw, (list, tuple)) and len(raw) >= 2:
            return float(raw[0]), float(raw[1]), str(key)
    return float(fallback_xy[0]), float(fallback_xy[1]), "pregrasp_tcp_fallback"


def build_sugar_box_object_safe_above_tcp(
    candidate: dict,
    *,
    safe_above_tcp_z: float,
    fallback_xy: Sequence[float],
) -> Tuple[Tuple[float, float, float], str]:
    cx, cy, source = resolve_sugar_box_operative_center_xy(
        candidate, fallback_xy=fallback_xy
    )
    return (cx, cy, float(safe_above_tcp_z)), source


def resolve_sugar_box_multiobject_object_safe_above(
    candidate: dict,
    *,
    safe_above_tcp_z: float,
    pregrasp_tcp: Sequence[float],
    grasp_tcp: Sequence[float],
) -> Tuple[Optional[Tuple[float, float, float]], str, str]:
    """Resuelve object_safe_above_tcp para ruta multiobjeto sugar_box."""
    pre: Optional[Tuple[float, float, float]] = None
    gr: Optional[Tuple[float, float, float]] = None
    if isinstance(pregrasp_tcp, (list, tuple)) and len(pregrasp_tcp) >= 3:
        pre = (float(pregrasp_tcp[0]), float(pregrasp_tcp[1]), float(pregrasp_tcp[2]))
    if isinstance(grasp_tcp, (list, tuple)) and len(grasp_tcp) >= 3:
        gr = (float(grasp_tcp[0]), float(grasp_tcp[1]), float(grasp_tcp[2]))
    if pre is None and gr is None:
        return None, "", format_sugar_box_object_safe_above_missing_log()
    fallback = pre or gr
    assert fallback is not None
    tcp, source = build_sugar_box_object_safe_above_tcp(
        candidate,
        safe_above_tcp_z=float(safe_above_tcp_z),
        fallback_xy=(float(fallback[0]), float(fallback[1])),
    )
    log = format_sugar_box_object_safe_above_resolved_log(
        {
            "center_source": source,
            "object_safe_above_tcp": tcp,
            "pregrasp_tcp": pre or tcp,
            "grasp_tcp": gr or tcp,
        }
    )
    return tcp, source, log


def apply_sugar_box_multiobject_safe_route_fields(
    candidate: dict,
    seq: dict,
    *,
    safe_above_tcp_z: float,
    clearance_above_top_m: float,
) -> Tuple[bool, str]:
    """Construye y propaga object_safe_above_tcp en candidate y seq."""
    tcp, _source, log = resolve_sugar_box_multiobject_object_safe_above(
        candidate,
        safe_above_tcp_z=float(safe_above_tcp_z),
        pregrasp_tcp=seq.get("pregrasp_tcp") or (),
        grasp_tcp=seq.get("grasp_tcp") or (),
    )
    if tcp is None:
        return False, log
    seq["object_safe_above_tcp"] = tcp
    seq["safe_pregrasp_tcp"] = tcp
    candidate["object_safe_above_tcp"] = [float(tcp[0]), float(tcp[1]), float(tcp[2])]
    candidate["sugar_box_safe_above_tcp_z"] = float(safe_above_tcp_z)
    candidate["sugar_box_multiobject_safe_route"] = True
    candidate["sugar_box_multiobject_safe_route_enabled"] = True
    candidate["selected_entry_target"] = "object_safe_above_tcp"
    candidate["object_safe_above_clearance_m"] = float(clearance_above_top_m)
    return True, log


def format_sugar_box_object_safe_above_resolved_log(fields: dict) -> str:
    obj = fields.get("object_safe_above_tcp") or (0.0, 0.0, 0.0)
    pre = fields.get("pregrasp_tcp") or (0.0, 0.0, 0.0)
    gr = fields.get("grasp_tcp") or (0.0, 0.0, 0.0)
    return (
        "[SUGAR_BOX_OBJECT_SAFE_ABOVE_RESOLVED]\n"
        "center_source=%s\n"
        "object_safe_above_tcp=(%.3f, %.3f, %.3f)\n"
        "pregrasp_tcp=(%.3f, %.3f, %.3f)\n"
        "grasp_tcp=(%.3f, %.3f, %.3f)\n"
        "result=OK"
        % (
            str(fields.get("center_source", "n/a")),
            float(obj[0]),
            float(obj[1]),
            float(obj[2]),
            float(pre[0]),
            float(pre[1]),
            float(pre[2]),
            float(gr[0]),
            float(gr[1]),
            float(gr[2]),
        )
    )


def format_sugar_box_object_safe_above_missing_log() -> str:
    return "[SUGAR_BOX_OBJECT_SAFE_ABOVE_MISSING]\nresult=FAIL"


def sugar_box_multiobject_final_descend_target_removal_required(
    candidate: dict,
) -> bool:
    """True cuando el descenso post-pregrasp debe validar sin colisión del target."""
    return (
        str(candidate.get("label", "")).strip().lower() == "sugar_box"
        and bool(candidate.get("sugar_box_multiobject_safe_route"))
    )


def sugar_box_multiobject_use_object_safe_above_stage(candidate: dict) -> bool:
    """Ruta pick_workspace_ready -> object_safe_above -> pregrasp (no safe_pregrasp genérico)."""
    if bool(candidate.get("_sugar_box_demo_scene_02_remaining_equivalent")):
        return False
    return sugar_box_multiobject_final_descend_target_removal_required(candidate)


def sugar_box_demo_scene_02_remaining_pick_equivalent(
    *,
    label: str,
    scene_id: str,
    completed_labels: AbstractSet[str],
    active_obstacle_labels: Optional[AbstractSet[str]] = None,
    present_table_labels: Optional[AbstractSet[str]] = None,
) -> bool:
    """Tras cracker+chips (o solo sugar en mesa en *_3obj), sugar_box usa simple_direct."""
    if str(label or "").strip().lower() != "sugar_box":
        return False
    from panda_controller.demo_golden_pick_candidate import runtime_uses_demo_scene_02_golden

    sid = str(scene_id or "").strip().lower()
    if sid in SUGAR_GOLDEN_DEBUG_SCENE_IDS:
        return True
    if sid == "demo_scene_02_clear_table":
        pass
    elif not runtime_uses_demo_scene_02_golden(sid):
        return False
    done = {str(x).strip().lower() for x in completed_labels if str(x).strip()}
    present = {
        str(x).strip().lower() for x in (present_table_labels or ()) if str(x).strip()
    }
    if DEMO_SCENE_02_SUGAR_BOX_REMAINING_COMPLETED_LABELS.issubset(done):
        return True
    if (
        present
        and present <= {"sugar_box", "mustard_bottle"}
        and "sugar_box" in present
        and sid in ("demo_scene_02", "demo_scene_02_clear_table")
    ):
        return True
    if sid.endswith("_3obj") and present == {"sugar_box"}:
        return True
    return False


def sugar_box_demo_golden_fast_execute_allowed(candidate: dict) -> bool:
    """Sugar sin golden solo en clear_table 4obj con cracker/chips aún en mesa."""
    if str(candidate.get("label", "")).strip().lower() != "sugar_box":
        return True
    if bool(candidate.get("_sugar_box_demo_scene_02_remaining_equivalent")):
        return True
    scene_id = str(
        candidate.get("scene_id") or candidate.get("_runtime_scene_id") or ""
    ).strip().lower()
    if sugar_box_scene_allows_golden(scene_id):
        return True
    return False


def sanitize_sugar_box_direct_plan_targets(
    candidate: dict, plan_targets: dict
) -> None:
    """Alinea safe/object_safe_above con pregrasp_tcp en ruta direct_pregrasp."""
    pre = plan_targets.get("pregrasp_tcp")
    if isinstance(pre, (list, tuple)) and len(pre) >= 3:
        direct = (float(pre[0]), float(pre[1]), float(pre[2]))
        plan_targets["safe_pregrasp_tcp"] = direct
    plan_targets.pop("object_safe_above_tcp", None)
    candidate.pop("sugar_box_multiobject_safe_route", None)
    candidate.pop("sugar_box_multiobject_safe_route_enabled", None)
    candidate.pop("object_safe_above_tcp", None)
    candidate["_sugar_box_selected_route"] = "direct_pregrasp"
    candidate["selected_entry_target"] = "pregrasp_tcp"


def apply_sugar_box_demo_scene_02_remaining_equivalent_fields(candidate: dict) -> bool:
    """Marca ruta simple_direct cuando queda sugar tras cracker+chips (mostaza no cambia ruta)."""
    if not bool(candidate.get("_sugar_box_demo_scene_02_remaining_equivalent")):
        return False
    candidate.pop("sugar_box_multiobject_safe_route", None)
    candidate.pop("sugar_box_multiobject_safe_route_enabled", None)
    candidate.pop("object_safe_above_tcp", None)
    candidate["_simple_direct_pick_route"] = True
    candidate["_sugar_box_selected_route"] = "direct_pregrasp"
    candidate["selected_entry_target"] = "pregrasp_tcp"
    return True


def format_sugar_box_demo_scene_02_remaining_equivalent_log(
    *,
    scene_id: str = "demo_scene_02",
    completed_labels: Optional[AbstractSet[str]] = None,
    present_table_labels: Optional[AbstractSet[str]] = None,
) -> str:
    done = sorted(completed_labels or ())
    present = sorted(present_table_labels or ())
    reason = "demo_scene_02_remaining_sugar_mustard"
    if present == ["sugar_box"]:
        reason = "only_sugar_on_table_3obj"
    return (
        "[SUGAR_BOX_DEMO_SCENE_02_REMAINING_EQUIVALENT]\n"
        "scene_id=%s\n"
        "completed_labels=%s\n"
        "present_table_labels=%s\n"
        "route=simple_direct_pregrasp\n"
        "policy_source=%s\n"
        "result=OK"
        % (
            str(scene_id or "demo_scene_02"),
            ",".join(done) if done else "n/a",
            ",".join(present) if present else "n/a",
            reason,
        )
    )


def sugar_box_object_safe_above_tcp_resolved(candidate: dict) -> bool:
    raw = candidate.get("object_safe_above_tcp")
    return isinstance(raw, (list, tuple)) and len(raw) >= 3


def sugar_box_multiobject_full_pick_prevalidate_required(candidate: dict) -> bool:
    """True: prevalidar ruta completa con object_safe_above (no simple_direct)."""
    return sugar_box_multiobject_use_object_safe_above_stage(
        candidate
    ) and sugar_box_object_safe_above_tcp_resolved(candidate)
