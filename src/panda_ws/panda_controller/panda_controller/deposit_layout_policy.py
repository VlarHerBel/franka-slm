"""Slots físicos + política por defecto por etiqueta (demo depósito TFG)."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

DEMO_FINAL_DEPOSIT_LABELS = frozenset(
    {
        "cracker_box",
        "chips_can",
        "sugar_box",
        "mustard_bottle",
    }
)

# Bandeja compacta (centro -0.46, 0); interior ~0.37 x 0.37 m.
DEPOSIT_BOX_CENTER_X_M = -0.46
DEPOSIT_BOX_CENTER_Y_M = 0.00
DEPOSIT_BOX_INTERIOR_X_M = 0.37
DEPOSIT_BOX_INTERIOR_Y_M = 0.37
DEPOSIT_BOX_FLOOR_TOP_Z_M = 0.01
DEPOSIT_BOX_WALL_TOP_Z_M = 0.17
DEPOSIT_BOX_X_MIN_M = -0.645
DEPOSIT_BOX_X_MAX_M = -0.275
DEPOSIT_BOX_Y_MIN_M = -0.195
DEPOSIT_BOX_Y_MAX_M = 0.195

# Layout A: huecos físicos (índice 0-based = slot_N - 1).
DEFAULT_PLACE_SLOTS_ORDERED: Tuple[Dict[str, Any], ...] = (
    {"name": "slot_1", "slot_number": 1, "x": -0.37, "y": 0.08},
    {"name": "slot_2", "slot_number": 2, "x": -0.54, "y": 0.08},
    {"name": "slot_3", "slot_number": 3, "x": -0.37, "y": -0.10},
    {"name": "slot_4", "slot_number": 4, "x": -0.54, "y": -0.10},
)

# Layout B: política por defecto si el usuario NO pide slot explícito.
DEFAULT_LABEL_SLOT_INDEX: Dict[str, int] = {
    "cracker_box": 0,
    "sugar_box": 1,
    "chips_can": 2,
    "mustard_bottle": 3,
}

DEPOSIT_FOOTPRINT_XY_RADIUS_BY_LABEL: Dict[str, float] = {
    "cracker_box": 0.0810,
    "chips_can": 0.0375,
    "sugar_box": 0.0650,
    "mustard_bottle": 0.0320,
}

DEFAULT_DEPOSIT_SAFETY_MARGIN_XY_M = 0.03


def _label_lower(label: Any) -> str:
    return str(label or "").strip().lower()


def default_label_slot_index(label: str) -> Optional[int]:
    return DEFAULT_LABEL_SLOT_INDEX.get(_label_lower(label))


def default_slot_name_for_label(
    label: str, slots: Sequence[Mapping[str, Any]] = DEFAULT_PLACE_SLOTS_ORDERED
) -> str:
    idx = default_label_slot_index(label)
    if idx is None or idx < 0 or idx >= len(slots):
        return "n/a"
    return str(slots[idx].get("name", "n/a"))


def resolve_slot_index_from_name(
    slot_name: Any,
    slots: Sequence[Mapping[str, Any]] = DEFAULT_PLACE_SLOTS_ORDERED,
) -> Optional[int]:
    raw = str(slot_name or "").strip().lower()
    if not raw:
        return None
    if raw.isdigit():
        num = int(raw)
        if 1 <= num <= len(slots):
            return num - 1
        if 0 <= num < len(slots):
            return num
    if raw.startswith("slot_"):
        suffix = raw[5:]
        if suffix.isdigit():
            num = int(suffix)
            if 1 <= num <= len(slots):
                return num - 1
    for i, slot in enumerate(slots):
        if str(slot.get("name", "")).strip().lower() == raw:
            return i
    return None


def footprint_radius_xy(
    label: str,
    candidate: Optional[Mapping[str, Any]] = None,
    *,
    radius_by_label: Optional[Mapping[str, float]] = None,
) -> float:
    table = dict(radius_by_label or DEPOSIT_FOOTPRINT_XY_RADIUS_BY_LABEL)
    lb = _label_lower(label)
    if candidate is not None:
        fp_major = candidate.get("footprint_major_m")
        fp_minor = candidate.get("footprint_minor_m")
        try:
            maj = float(fp_major) if fp_major is not None else 0.0
            mn = float(fp_minor) if fp_minor is not None else 0.0
        except (TypeError, ValueError):
            maj = mn = 0.0
        if maj > 0.0 or mn > 0.0:
            return max(maj, mn) * 0.5
        cd = candidate.get("collision_dims")
        if isinstance(cd, dict):
            dd = cd.get("db_dims")
            if isinstance(dd, (list, tuple)) and len(dd) >= 2:
                try:
                    a = abs(float(dd[0]))
                    b = abs(float(dd[1]))
                    return max(a, b) * 0.5
                except (TypeError, ValueError):
                    pass
    if lb in table:
        return float(table[lb])
    return 0.050


def required_separation_xy(
    radius_a: float,
    radius_b: float,
    *,
    safety_margin_m: float = DEFAULT_DEPOSIT_SAFETY_MARGIN_XY_M,
) -> float:
    return float(radius_a) + float(radius_b) + max(0.0, float(safety_margin_m))


def center_distance_xy(
    x_a: float,
    y_a: float,
    x_b: float,
    y_b: float,
) -> float:
    return math.hypot(float(x_a) - float(x_b), float(y_a) - float(y_b))


def deposit_slot_collision_check(
    *,
    new_label: str,
    new_x: float,
    new_y: float,
    existing_label: str,
    existing_x: float,
    existing_y: float,
    new_radius: float,
    existing_radius: float,
    safety_margin_m: float = DEFAULT_DEPOSIT_SAFETY_MARGIN_XY_M,
) -> Tuple[str, float, float]:
    dist = center_distance_xy(new_x, new_y, existing_x, existing_y)
    req = required_separation_xy(
        new_radius, existing_radius, safety_margin_m=safety_margin_m
    )
    if dist + 1e-6 < req:
        return "TOO_CLOSE", dist, req
    return "OK", dist, req


def _occupied_xy_list(
    occupied: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for ent in occupied:
        lb = _label_lower(ent.get("label"))
        try:
            x = float(ent.get("x"))
            y = float(ent.get("y"))
        except (TypeError, ValueError):
            continue
        if not lb:
            continue
        try:
            slot_index = int(ent.get("slot_index", ent.get("slot", -1)))
        except (TypeError, ValueError):
            slot_index = -1
        out.append(
            {
                "label": lb,
                "x": x,
                "y": y,
                "slot_index": slot_index,
                "slot_name": str(ent.get("slot_name") or ent.get("slot") or ""),
                "entity": str(ent.get("entity") or ent.get("entity_name") or ""),
            }
        )
    return out


def _slot_occupied_by_other_label(
    occupied: Sequence[Mapping[str, Any]],
    slot_index: int,
    label: str,
) -> Optional[str]:
    lb = _label_lower(label)
    for o in _occupied_xy_list(occupied):
        if int(o["slot_index"]) == int(slot_index) and o["label"] != lb:
            return o["label"]
    return None


def _evaluate_slot(
    *,
    label: str,
    slot_index: int,
    slots: Sequence[Mapping[str, Any]],
    occupied: Sequence[Mapping[str, Any]],
    candidate: Optional[Mapping[str, Any]],
    safety_margin_m: float,
    plan_checks: List[Dict[str, Any]],
) -> Tuple[bool, Optional[Dict[str, Any]], float, str]:
    n = len(slots)
    if slot_index < 0 or slot_index >= n:
        return False, None, -1.0, "invalid_slot_index"
    lb = _label_lower(label)
    new_r = footprint_radius_xy(lb, candidate)
    occ = _occupied_xy_list(occupied)
    blocker = _slot_occupied_by_other_label(occ, slot_index, lb)
    if blocker:
        return False, None, -1.0, "slot_occupied_by_%s" % blocker
    slot = dict(slots[slot_index])
    sx = float(slot["x"])
    sy = float(slot["y"])
    min_dist = float("inf")
    for o in occ:
        if o["label"] == lb:
            continue
        ex_r = footprint_radius_xy(o["label"])
        result, dist, req = deposit_slot_collision_check(
            new_label=lb,
            new_x=sx,
            new_y=sy,
            existing_label=o["label"],
            existing_x=o["x"],
            existing_y=o["y"],
            new_radius=new_r,
            existing_radius=ex_r,
            safety_margin_m=safety_margin_m,
        )
        plan_checks.append(
            {
                "new_label": lb,
                "existing_label": o["label"],
                "center_distance_xy": dist,
                "required_distance_xy": req,
                "result": result,
                "candidate_slot_index": slot_index,
            }
        )
        if result == "TOO_CLOSE":
            return False, None, dist, "footprint_too_close"
        min_dist = min(min_dist, dist)
    if not occ:
        min_dist = float("nan")
    return True, slot, min_dist, ""


def plan_deposit_layout_slot(
    *,
    label: str,
    slots: Sequence[Mapping[str, Any]],
    occupied: Sequence[Mapping[str, Any]],
    candidate: Optional[Mapping[str, Any]] = None,
    safety_margin_m: float = DEFAULT_DEPOSIT_SAFETY_MARGIN_XY_M,
    user_requested_slot: bool = False,
    requested_slot_index: Optional[int] = None,
    requested_slot_name: Optional[str] = None,
    slot_counter: int = 0,
    label_slot_map: Optional[Mapping[str, int]] = None,
) -> Tuple[Optional[int], Optional[Dict[str, Any]], Dict[str, Any]]:
    lb = _label_lower(label)
    n = len(slots)
    default_idx = default_label_slot_index(lb)
    default_name = default_slot_name_for_label(lb, slots)
    occ_summary = [
        {
            "label": o["label"],
            "entity": o.get("entity", ""),
            "slot_name": o.get("slot_name", ""),
            "slot": o.get("slot_index", -1),
            "x": o["x"],
            "y": o["y"],
        }
        for o in _occupied_xy_list(occupied)
    ]
    plan: Dict[str, Any] = {
        "label": lb or "n/a",
        "requested_slot": "none",
        "default_slot": default_name,
        "selection_mode": "n/a",
        "safety_margin_m": float(safety_margin_m),
        "completed_deposits": occ_summary,
        "collision_checks": [],
    }

    req_idx: Optional[int] = None
    candidates_idx: List[int]
    if user_requested_slot:
        plan["selection_mode"] = "user_requested_slot"
        if requested_slot_index is not None:
            try:
                req_idx = int(requested_slot_index)
            except (TypeError, ValueError):
                req_idx = None
        if req_idx is None and requested_slot_name:
            req_idx = resolve_slot_index_from_name(requested_slot_name, slots)
        if req_idx is None:
            plan.update({"result": "REJECTED", "reason": "invalid_user_slot_request"})
            return None, None, plan
        plan["requested_slot"] = str(slots[req_idx].get("name", "slot_%d" % (req_idx + 1)))
        candidates_idx = [req_idx]
    else:
        scene_slot: Optional[int] = None
        if label_slot_map and lb in label_slot_map:
            try:
                scene_slot = int(label_slot_map[lb]) % max(1, n)
            except (TypeError, ValueError):
                scene_slot = None
        if scene_slot is not None:
            plan["selection_mode"] = "scene_preferred_slot"
            candidates_idx = [scene_slot]
        elif default_idx is not None:
            plan["selection_mode"] = "label_default_slot"
            candidates_idx = [int(default_idx)]
        else:
            plan["selection_mode"] = "layout_aware_fallback"
            order = list(range(n))
            if not _occupied_xy_list(occupied):
                idx0 = int(slot_counter) % max(1, n)
                order.remove(idx0)
                order.insert(0, idx0)
            candidates_idx = order

    checks: List[Dict[str, Any]] = []
    last_reason = "no_slot_passes_checks"
    for idx in candidates_idx:
        ok, slot, _min_d, reason = _evaluate_slot(
            label=lb,
            slot_index=idx,
            slots=slots,
            occupied=occupied,
            candidate=candidate,
            safety_margin_m=safety_margin_m,
            plan_checks=checks,
        )
        if ok and slot is not None:
            plan["collision_checks"] = checks
            plan.update(
                {
                    "selected_slot": str(slot.get("name", "")),
                    "selected_slot_index": int(idx),
                    "selected_x": float(slot["x"]),
                    "selected_y": float(slot["y"]),
                    "result": "OK",
                }
            )
            return int(idx), slot, plan
        if reason:
            last_reason = reason
    plan["collision_checks"] = checks
    plan.update(
        {
            "selected_slot": "n/a",
            "result": "REJECTED",
            "reason": last_reason,
        }
    )
    return None, None, plan


def format_deposit_layout_plan_log(plan: Mapping[str, Any]) -> str:
    lines = [
        "[DEPOSIT_LAYOUT_PLAN]",
        "label=%s" % plan.get("label", "n/a"),
        "requested_slot=%s" % plan.get("requested_slot", "none"),
        "selection_mode=%s" % plan.get("selection_mode", "n/a"),
        "default_slot=%s" % plan.get("default_slot", "n/a"),
        "selected_slot=%s" % plan.get("selected_slot", "n/a"),
        "selected_x=%s"
        % (
            "n/a"
            if plan.get("selected_x") is None
            else "%.4f" % float(plan["selected_x"])
        ),
        "selected_y=%s"
        % (
            "n/a"
            if plan.get("selected_y") is None
            else "%.4f" % float(plan["selected_y"])
        ),
        "completed_deposits=%s" % plan.get("completed_deposits", []),
        "result=%s" % plan.get("result", "n/a"),
    ]
    if plan.get("reason"):
        lines.append("reason=%s" % plan["reason"])
    return "\n".join(lines)


def format_deposit_collision_check_logs(
    checks: Sequence[Mapping[str, Any]],
) -> List[str]:
    out: List[str] = []
    for chk in checks:
        out.append(
            "[DEPOSIT_SLOT_COLLISION_CHECK]\n"
            "new_label=%s\n"
            "existing_label=%s\n"
            "center_distance_xy=%.4f\n"
            "required_distance_xy=%.4f\n"
            "result=%s"
            % (
                chk.get("new_label", "n/a"),
                chk.get("existing_label", "n/a"),
                float(chk.get("center_distance_xy", 0.0)),
                float(chk.get("required_distance_xy", 0.0)),
                chk.get("result", "n/a"),
            )
        )
    return out
