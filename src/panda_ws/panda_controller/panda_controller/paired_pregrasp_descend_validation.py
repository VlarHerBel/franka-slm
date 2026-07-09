"""Validación emparejada pregrasp + descenso cartesiano desde el mismo joint_state (demo_scene_02)."""

from panda_controller.demo_cracker_collision_off_final_descend import (
    DEMO_COLLISION_OFF_FINAL_DESCEND_SOURCE,
    DEMO_COLLISION_OFF_FINAL_DESCEND_STAGED_SOURCE,
)

from typing import Any, Dict, List, Optional, Sequence, Tuple

import math

from panda_controller.tfg_motion_waypoints import PANDA_ARM_JOINT_NAMES

from panda_controller.demo_cracker_box_cartesian_prevalidate import DEMO_SCENE_02_IDS

DEFAULT_PREGRASP_JS_DISTANCE_THRESHOLD = 0.08
DEFAULT_PREGRASP_TCP_ERROR_THRESHOLD_M = 0.015
PAIRED_PREGRASP_FK_ERROR_THRESHOLD_M = 0.005

PAIRED_REJECT_REASON_NO_CARTESIAN = "no_pregrasp_candidate_with_valid_cartesian_descend"
PAIRED_REJECT_REASON_NO_LIFT = "no_pregrasp_candidate_with_valid_lift"
PAIRED_REJECT_REASON_LIFT_AFTER_VALID_DESCEND = "lift_fail_after_valid_descend"
PAIRED_REJECT_REASON_NO_TRANSPORT = "no_pregrasp_candidate_with_valid_post_lift_exit"
PAIRED_REJECT_REASON_NO_TRANSPORT_EXIT = (
    "no_pregrasp_candidate_with_valid_transport_exit"
)
PAIRED_REJECT_REASON_TRANSPORT_AFTER_VALID_LIFT = "transport_fail_after_valid_lift"
PAIRED_REJECT_REASON_FRAME_CONTRACT = "paired_pregrasp_frame_contract_failed"


def _xyz_error_m(
    a: Optional[Tuple[float, float, float]],
    b: Tuple[float, float, float],
) -> Optional[float]:
    if a is None:
        return None
    return float(
        (
            (float(a[0]) - float(b[0])) ** 2
            + (float(a[1]) - float(b[1])) ** 2
            + (float(a[2]) - float(b[2])) ** 2
        )
        ** 0.5
    )


def _fmt_xyz(pos: Optional[Tuple[float, float, float]]) -> str:
    if pos is None:
        return "n/a"
    return "(%.4f, %.4f, %.4f)" % (float(pos[0]), float(pos[1]), float(pos[2]))


def evaluate_paired_pregrasp_fk_contract(
    *,
    expected_tcp: Tuple[float, float, float],
    expected_hand: Tuple[float, float, float],
    actual_tcp: Optional[Tuple[float, float, float]],
    actual_hand: Optional[Tuple[float, float, float]],
    error_threshold_m: float = PAIRED_PREGRASP_FK_ERROR_THRESHOLD_M,
) -> Dict[str, Any]:
    """Verifica FK(panda_grasp_tcp)≈pregrasp_tcp y FK(panda_hand)≈pregrasp_hand."""
    tcp_err = _xyz_error_m(actual_tcp, expected_tcp)
    hand_err = _xyz_error_m(actual_hand, expected_hand)
    hand_vs_tcp_err = _xyz_error_m(actual_hand, expected_tcp)
    frame_mismatch = bool(
        hand_vs_tcp_err is not None
        and hand_err is not None
        and float(hand_vs_tcp_err) + 1e-9 < float(error_threshold_m)
        and float(hand_err) >= 0.05
    )
    ok = bool(
        tcp_err is not None
        and hand_err is not None
        and float(tcp_err) + 1e-9 < float(error_threshold_m)
        and float(hand_err) + 1e-9 < float(error_threshold_m)
    )
    return {
        "expected_tcp": expected_tcp,
        "expected_hand": expected_hand,
        "actual_tcp": actual_tcp,
        "actual_hand": actual_hand,
        "tcp_error_m": tcp_err,
        "hand_error_m": hand_err,
        "hand_vs_tcp_error_m": hand_vs_tcp_err,
        "frame_mismatch": frame_mismatch,
        "ok": ok,
    }


def format_paired_pregrasp_fk_verify_log(
    *,
    label: str,
    candidate_idx: int,
    fk_result: Dict[str, Any],
    result: str,
) -> str:
    return (
        "[PAIRED_PREGRASP_FK_VERIFY]\n"
        "label=%s\n"
        "candidate_idx=%s\n"
        "actual_tcp=%s\n"
        "expected_tcp=%s\n"
        "tcp_error_m=%s\n"
        "actual_hand=%s\n"
        "expected_hand=%s\n"
        "hand_error_m=%s\n"
        "result=%s"
        % (
            label,
            candidate_idx,
            _fmt_xyz(fk_result.get("actual_tcp")),
            _fmt_xyz(fk_result.get("expected_tcp")),
            "n/a"
            if fk_result.get("tcp_error_m") is None
            else "%.4f" % float(fk_result["tcp_error_m"]),
            _fmt_xyz(fk_result.get("actual_hand")),
            _fmt_xyz(fk_result.get("expected_hand")),
            "n/a"
            if fk_result.get("hand_error_m") is None
            else "%.4f" % float(fk_result["hand_error_m"]),
            result,
        )
    )


def format_paired_pregrasp_frame_mismatch_log(
    *,
    label: str,
    candidate_idx: int,
    fk_result: Dict[str, Any],
) -> str:
    return (
        "[PAIRED_PREGRASP_FRAME_MISMATCH]\n"
        "label=%s\n"
        "candidate_idx=%s\n"
        "hand_error_m=%s\n"
        "tcp_error_m=%s\n"
        "hand_vs_tcp_error_m=%s\n"
        "expected_tcp=%s\n"
        "expected_hand=%s\n"
        "actual_tcp=%s\n"
        "actual_hand=%s\n"
        "result=FAIL_FRAME_CONTRACT"
        % (
            label,
            candidate_idx,
            "n/a"
            if fk_result.get("hand_error_m") is None
            else "%.4f" % float(fk_result["hand_error_m"]),
            "n/a"
            if fk_result.get("tcp_error_m") is None
            else "%.4f" % float(fk_result["tcp_error_m"]),
            "n/a"
            if fk_result.get("hand_vs_tcp_error_m") is None
            else "%.4f" % float(fk_result["hand_vs_tcp_error_m"]),
            _fmt_xyz(fk_result.get("expected_tcp")),
            _fmt_xyz(fk_result.get("expected_hand")),
            _fmt_xyz(fk_result.get("actual_tcp")),
            _fmt_xyz(fk_result.get("actual_hand")),
        )
    )


def paired_pregrasp_descend_validation_required(
    *,
    candidate: Dict[str, Any],
    demo_authoritative_scene: bool,
    scene_id: str,
) -> bool:
    """demo_authoritative_scene + demo multiobjeto: exige validación emparejada."""
    if not bool(demo_authoritative_scene):
        return False
    from panda_vision.spawn.demo_scene_presets import demo_scene_policy_scene_id_for_preset

    sid = str(scene_id or candidate.get("scene_id") or "").strip().lower()
    parent = demo_scene_policy_scene_id_for_preset(sid)
    return parent in DEMO_SCENE_02_IDS


def paired_validation_forbids_geometric_fallback(
    *,
    candidate: Dict[str, Any],
    demo_authoritative_scene: bool,
    scene_id: str,
) -> bool:
    return paired_pregrasp_descend_validation_required(
        candidate=candidate,
        demo_authoritative_scene=demo_authoritative_scene,
        scene_id=scene_id,
    )


PAIRED_ACCEPTED_PREVALIDATION_SOURCES = frozenset(
    {
        "moveit",
        DEMO_COLLISION_OFF_FINAL_DESCEND_SOURCE,
        DEMO_COLLISION_OFF_FINAL_DESCEND_STAGED_SOURCE,
    }
)

SIMPLE_DIRECT_PAIRED_GEOMETRIC_FALLBACK_LABELS = frozenset(
    {"sugar_box", "mustard_bottle"}
)


def simple_direct_geometric_fallback_paired_allowed(
    *,
    label: str,
    prevalidation_source: str,
    simple_direct_route: bool = False,
) -> bool:
    if not bool(simple_direct_route):
        return False
    if str(prevalidation_source or "").strip() != "geometric_fallback":
        return False
    return str(label or "").strip().lower() in SIMPLE_DIRECT_PAIRED_GEOMETRIC_FALLBACK_LABELS


def paired_prevalidation_source_acceptable(
    *,
    label: str,
    prevalidation_source: str,
    simple_direct_route: bool = False,
) -> bool:
    src = str(prevalidation_source or "").strip()
    if src in PAIRED_ACCEPTED_PREVALIDATION_SOURCES:
        return True
    return simple_direct_geometric_fallback_paired_allowed(
        label=label,
        prevalidation_source=src,
        simple_direct_route=simple_direct_route,
    )


def cartesian_descend_prevalidation_acceptable(
    *,
    cartesian_ok: bool,
    cartesian_fraction: float,
    fraction_threshold: float,
    prevalidation_source: str,
    paired_validation_required: bool,
    label: str = "",
    simple_direct_route: bool = False,
) -> bool:
    if not cartesian_ok:
        return False
    if not paired_validation_required:
        return True
    src = str(prevalidation_source or "").strip()
    if simple_direct_geometric_fallback_paired_allowed(
        label=label,
        prevalidation_source=src,
        simple_direct_route=simple_direct_route,
    ):
        return True
    if src == "geometric_fallback":
        return False
    if src in (
        "paired_safe_geometric",
        DEMO_COLLISION_OFF_FINAL_DESCEND_SOURCE,
        DEMO_COLLISION_OFF_FINAL_DESCEND_STAGED_SOURCE,
    ):
        return bool(cartesian_ok)
    return float(cartesian_fraction) + 1e-6 >= float(fraction_threshold)


def build_paired_candidate_result(
    *,
    label: str,
    candidate_idx: int,
    yaw_variant: float,
    ik_pregrasp_ok: bool,
    plan_to_pregrasp_ok: bool,
    candidate_pregrasp_js: Any,
    fk_contract_ok: bool = True,
    cartesian_descend_fraction: float,
    cartesian_descend_ok: bool,
    lift_ok: bool,
    post_lift_exit_ok: bool,
    direct_action_to_hub_ok: bool,
    local_escape_ok: Optional[bool] = None,
    global_route_ok: Optional[bool] = None,
    prevalidation_source: str = "moveit",
    simple_direct_route: bool = False,
) -> Dict[str, Any]:
    local_ok = bool(
        local_escape_ok if local_escape_ok is not None else post_lift_exit_ok
    )
    global_ok = bool(
        global_route_ok if global_route_ok is not None else direct_action_to_hub_ok
    )
    source_ok = paired_prevalidation_source_acceptable(
        label=label,
        prevalidation_source=str(prevalidation_source or ""),
        simple_direct_route=bool(simple_direct_route),
    )
    accept = bool(
        ik_pregrasp_ok
        and plan_to_pregrasp_ok
        and candidate_pregrasp_js is not None
        and fk_contract_ok
        and cartesian_descend_ok
        and lift_ok
        and local_ok
        and source_ok
    )
    return {
        "label": str(label),
        "candidate_idx": int(candidate_idx),
        "yaw_variant": float(yaw_variant),
        "ik_pregrasp_ok": bool(ik_pregrasp_ok),
        "plan_to_pregrasp_ok": bool(plan_to_pregrasp_ok),
        "fk_contract_ok": bool(fk_contract_ok),
        "candidate_pregrasp_js": candidate_pregrasp_js,
        "cartesian_descend_from_same_js_fraction": float(cartesian_descend_fraction),
        "cartesian_descend_ok": bool(cartesian_descend_ok),
        "lift_ok": bool(lift_ok),
        "local_escape_ok": local_ok,
        "global_route_ok": global_ok,
        "post_lift_exit_ok": local_ok,
        "direct_action_to_hub_ok": global_ok,
        "prevalidation_source": str(prevalidation_source or "moveit"),
        "result": "ACCEPT" if accept else "REJECT",
    }


def format_paired_pregrasp_descend_candidate_log(result: Dict[str, Any]) -> str:
    js = result.get("candidate_pregrasp_js")
    js_repr = "n/a" if js is None else str(list(getattr(js, "position", js)))
    return (
        "[PAIRED_PREGRASP_DESCEND_CANDIDATE]\n"
        "label=%s\n"
        "candidate_idx=%s\n"
        "yaw_variant=%s\n"
        "ik_pregrasp_ok=%s\n"
        "plan_to_pregrasp_ok=%s\n"
        "fk_contract_ok=%s\n"
        "candidate_pregrasp_js=%s\n"
        "cartesian_descend_from_same_js_fraction=%s\n"
        "cartesian_descend_ok=%s\n"
        "lift_ok=%s\n"
        "local_escape_ok=%s\n"
        "global_route_ok=%s\n"
        "post_lift_exit_ok=%s\n"
        "direct_action_to_hub_ok=%s\n"
        "prevalidation_source=%s\n"
        "result=%s"
        % (
            result.get("label", "n/a"),
            result.get("candidate_idx", "n/a"),
            result.get("yaw_variant", "n/a"),
            str(bool(result.get("ik_pregrasp_ok"))).lower(),
            str(bool(result.get("plan_to_pregrasp_ok"))).lower(),
            str(bool(result.get("fk_contract_ok", True))).lower(),
            js_repr,
            result.get("cartesian_descend_from_same_js_fraction", "n/a"),
            str(bool(result.get("cartesian_descend_ok"))).lower(),
            str(bool(result.get("lift_ok"))).lower(),
            str(bool(result.get("local_escape_ok", result.get("post_lift_exit_ok")))).lower(),
            str(bool(result.get("global_route_ok", result.get("direct_action_to_hub_ok")))).lower(),
            str(bool(result.get("post_lift_exit_ok"))).lower(),
            str(bool(result.get("direct_action_to_hub_ok"))).lower(),
            result.get("prevalidation_source", "moveit"),
            str(result.get("result", "REJECT")),
        )
    )


def format_paired_candidate_acceptance_decision_log(
    *,
    candidate_idx: int,
    descend_ok: bool,
    lift_ok: bool,
    local_escape_ok: bool,
    global_route_ok: bool,
    accept: bool,
    reject_reason: str = "",
) -> str:
    return (
        "[PAIRED_CANDIDATE_ACCEPTANCE_DECISION]\n"
        "candidate_idx=%d\n"
        "descend_ok=%s\n"
        "lift_ok=%s\n"
        "local_escape_ok=%s\n"
        "global_route_ok=%s\n"
        "accept=%s\n"
        "reject_reason=%s"
        % (
            int(candidate_idx),
            str(bool(descend_ok)).lower(),
            str(bool(lift_ok)).lower(),
            str(bool(local_escape_ok)).lower(),
            str(bool(global_route_ok)).lower(),
            str(bool(accept)).lower(),
            str(reject_reason or "n/a"),
        )
    )


def format_paired_pregrasp_descend_selected_log(
    *,
    label: str,
    candidate_idx: int,
    cartesian_descend_fraction: float,
    post_lift_exit_ok: bool,
    hub_segment_validated: bool,
) -> str:
    return (
        "[PAIRED_PREGRASP_DESCEND_SELECTED]\n"
        "label=%s\n"
        "candidate_idx=%s\n"
        "cartesian_descend_fraction=%s\n"
        "post_lift_exit_ok=%s\n"
        "hub_segment_validated=%s\n"
        "result=OK"
        % (
            label,
            candidate_idx,
            cartesian_descend_fraction,
            str(bool(post_lift_exit_ok)).lower(),
            str(bool(hub_segment_validated)).lower(),
        )
    )


def format_pregrasp_execution_state_verify_log(
    *,
    label: str,
    current_js_distance_to_validated_js: Optional[float],
    current_tcp_error_m: Optional[float],
    js_threshold: float,
    tcp_threshold_m: float,
    result: str,
    reason: str = "",
) -> str:
    return (
        "[PREGRASP_EXECUTION_STATE_VERIFY]\n"
        "label=%s\n"
        "current_js_distance_to_validated_js=%s\n"
        "current_tcp_error_m=%s\n"
        "js_threshold=%.4f\n"
        "tcp_threshold_m=%.4f\n"
        "result=%s\n"
        "reason=%s"
        % (
            label,
            "n/a"
            if current_js_distance_to_validated_js is None
            else "%.4f" % float(current_js_distance_to_validated_js),
            "n/a"
            if current_tcp_error_m is None
            else "%.4f" % float(current_tcp_error_m),
            float(js_threshold),
            float(tcp_threshold_m),
            str(result),
            str(reason or "n/a"),
        )
    )


def compute_wrapped_joint_diffs(
    current_js: Any,
    reference_js: Any,
) -> Tuple[List[float], float, str]:
    """Diferencias articulares envueltas current vs referencia."""
    if current_js is None or reference_js is None:
        return [], float("inf"), "n/a"
    try:
        cur_names = list(current_js.name)
        cur_pos = [float(v) for v in current_js.position]
        ref_names = list(reference_js.name)
        ref_pos = [float(v) for v in reference_js.position]
    except (TypeError, ValueError, AttributeError):
        return [], float("inf"), "n/a"
    cur_map = {str(n): float(p) for n, p in zip(cur_names, cur_pos)}
    ref_map = {str(n): float(p) for n, p in zip(ref_names, ref_pos)}
    diffs: List[float] = []
    largest = 0.0
    largest_name = "n/a"
    for name in PANDA_ARM_JOINT_NAMES:
        if name not in cur_map or name not in ref_map:
            diffs.append(float("nan"))
            continue
        delta = math.atan2(
            math.sin(float(cur_map[name]) - float(ref_map[name])),
            math.cos(float(cur_map[name]) - float(ref_map[name])),
        )
        diffs.append(float(delta))
        if abs(delta) > abs(largest):
            largest = float(delta)
            largest_name = str(name)
    return diffs, largest, largest_name


def format_pregrasp_execution_state_verify_debug_log(
    *,
    current_js: Any,
    validated_js_before_axis: Any,
    validated_js_after_axis: Any,
    joint_diffs_wrapped: Sequence[float],
    largest_joint_diff: float,
    largest_joint_name: str,
    axis_correction_applied: bool,
    axis_correction_joint: str = "panda_joint7",
) -> str:
    def _fmt_js(js: Any) -> str:
        if js is None:
            return "n/a"
        try:
            pos = [float(v) for v in js.position]
            return "[%s]" % ", ".join("%.4f" % v for v in pos[:7])
        except (TypeError, ValueError, AttributeError):
            return "n/a"

    diffs_repr = (
        "n/a"
        if not joint_diffs_wrapped
        else "[%s]"
        % ", ".join(
            "n/a" if v != v else "%.4f" % float(v) for v in joint_diffs_wrapped
        )
    )
    return (
        "[PREGRASP_EXECUTION_STATE_VERIFY_DEBUG]\n"
        "current_js=%s\n"
        "validated_js_before_axis=%s\n"
        "validated_js_after_axis=%s\n"
        "joint_diffs_wrapped=%s\n"
        "largest_joint_diff=%.4f\n"
        "largest_joint_name=%s\n"
        "axis_correction_applied=%s\n"
        "axis_correction_joint=%s"
        % (
            _fmt_js(current_js),
            _fmt_js(validated_js_before_axis),
            _fmt_js(validated_js_after_axis),
            diffs_repr,
            float(largest_joint_diff),
            str(largest_joint_name),
            str(bool(axis_correction_applied)).lower(),
            str(axis_correction_joint or "panda_joint7"),
        )
    )


def evaluate_pregrasp_execution_state_verify(
    *,
    js_distance: Optional[float],
    js_threshold: float,
    tcp_error_m: Optional[float],
    tcp_threshold_m: float,
    gap_ok: bool,
    centering_ok: bool,
    joint_limits_ok: bool,
) -> Tuple[bool, str]:
    """Política: coherencia articular estricta o verificación física post-corrección."""
    js_dist = float(js_distance) if js_distance is not None else float("inf")
    tcp_err = float(tcp_error_m) if tcp_error_m is not None else float("inf")
    strict_ok = js_dist + 1e-9 < float(js_threshold) and tcp_err + 1e-9 < float(
        tcp_threshold_m
    )
    if strict_ok:
        return True, "js_and_tcp_ok"
    physical_ok = bool(
        tcp_err + 1e-9 < float(tcp_threshold_m)
        and gap_ok
        and centering_ok
        and joint_limits_ok
    )
    if physical_ok:
        return True, "tcp_centering_gap_ok_joint7_correction_expected"
    return False, "execution_state_mismatch"


def evaluate_validated_joint7_runtime_match(
    *,
    expected_joint7: Optional[float],
    actual_joint7: Optional[float],
    joint7_tol_rad: float,
    expected_gap_error_deg: Optional[float],
    actual_gap_error_deg: Optional[float],
    gap_extra_deg: float,
    target_gap_deg: float,
    hard_max_gap_deg: float,
    axis_correction_applied: bool,
    gap_ok: bool,
    centering_ok: bool,
    joint_limits_ok: bool,
    tcp_ok: bool,
) -> Tuple[bool, str, str]:
    """Permite continuar si el gap físico real es OK tras corrección in-place de joint7."""
    if actual_joint7 is None:
        return False, "ABORT_BEFORE_DESCEND", "missing_actual_joint7"
    actual_err = (
        float(actual_gap_error_deg)
        if actual_gap_error_deg is not None
        else float("inf")
    )
    physical_ok = bool(
        gap_ok
        and centering_ok
        and joint_limits_ok
        and tcp_ok
        and actual_err <= float(hard_max_gap_deg) + 1e-6
    )
    if axis_correction_applied and physical_ok:
        return True, "ALLOW_PHYSICAL_GAP_OK", "joint7_corrected_in_place"
    if expected_joint7 is None:
        return True, "OK", "no_expected_joint7"
    j7_ok = abs(float(actual_joint7) - float(expected_joint7)) <= (
        float(joint7_tol_rad) + 1e-6
    )
    err_tol = float(target_gap_deg) + float(gap_extra_deg)
    err_ok = expected_gap_error_deg is not None and actual_err <= (
        float(expected_gap_error_deg) + err_tol + 1e-6
    )
    if j7_ok and err_ok:
        return True, "OK", "joint7_and_gap_match"
    if physical_ok:
        reason = (
            "joint7_corrected_in_place"
            if axis_correction_applied
            else "physical_gap_ok_despite_joint7_delta"
        )
        return True, "ALLOW_PHYSICAL_GAP_OK", reason
    return False, "ABORT_BEFORE_DESCEND", "joint7_or_gap_mismatch"


def format_validated_joint7_runtime_mismatch_log(
    *,
    expected_joint7: Optional[float],
    actual_joint7: Optional[float],
    expected_gap_error_deg: Optional[float],
    actual_gap_error_deg: Optional[float],
    action: str,
    reason: str = "",
) -> str:
    return (
        "[VALIDATED_JOINT7_RUNTIME_MISMATCH]\n"
        "expected_joint7=%s\n"
        "actual_joint7=%s\n"
        "expected_gap_error_deg=%s\n"
        "actual_gap_error_deg=%s\n"
        "action=%s\n"
        "reason=%s"
        % (
            "n/a" if expected_joint7 is None else "%.4f" % float(expected_joint7),
            "n/a" if actual_joint7 is None else "%.4f" % float(actual_joint7),
            "n/a"
            if expected_gap_error_deg is None
            else "%.2f" % float(expected_gap_error_deg),
            "n/a"
            if actual_gap_error_deg is None
            else "%.2f" % float(actual_gap_error_deg),
            str(action),
            str(reason or "n/a"),
        )
    )


def format_pre_descend_gripper_reverify_log(
    *,
    gap_ok: bool,
    centering_ok: bool,
    tcp_ok: bool,
    joint7_mismatch_ignored: bool,
    reason: str,
    result: str,
) -> str:
    return (
        "[PRE_DESCEND_GRIPPER_REVERIFY]\n"
        "gap_ok=%s\n"
        "centering_ok=%s\n"
        "tcp_ok=%s\n"
        "joint7_mismatch_ignored=%s\n"
        "reason=%s\n"
        "result=%s"
        % (
            str(bool(gap_ok)).lower(),
            str(bool(centering_ok)).lower(),
            str(bool(tcp_ok)).lower(),
            str(bool(joint7_mismatch_ignored)).lower(),
            str(reason or "n/a"),
            str(result),
        )
    )


def select_paired_pregrasp_candidate(
    paired_results: Sequence[Dict[str, Any]],
    transport_scores: Sequence[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """Selecciona candidato ACCEPT; prioriza transporte virtual, sin fallback joint_dist."""
    accepted = [r for r in paired_results if r.get("result") == "ACCEPT"]
    if not accepted:
        return None
    score_by_idx = {
        int(s.get("candidate_idx", -1)): s
        for s in transport_scores
        if s.get("result") == "ACCEPT"
    }
    ranked: List[Dict[str, Any]] = []
    for r in accepted:
        idx = int(r.get("candidate_idx", -1))
        ts = score_by_idx.get(idx)
        if ts is None:
            continue
        ranked.append({**r, "_transport_score": ts})
    if not ranked:
        return None
    ranked.sort(
        key=lambda item: (
            float(item["_transport_score"].get("joint_distance_to_hub", 1e9)),
            float(item["_transport_score"].get("wrist_twist_score", 1e9)),
            float(item["_transport_score"].get("elbow_score", 1e9)),
        )
    )
    return ranked[0]


def paired_preflight_fail_reason(
    paired_results: Sequence[Dict[str, Any]],
) -> str:
    if not paired_results:
        return PAIRED_REJECT_REASON_NO_CARTESIAN
    any_descend = any(bool(r.get("cartesian_descend_ok")) for r in paired_results)
    if not any_descend:
        return PAIRED_REJECT_REASON_NO_CARTESIAN
    any_valid_descend_lift_fail = any(
        bool(r.get("cartesian_descend_ok")) and not bool(r.get("lift_ok"))
        for r in paired_results
    )
    if any_valid_descend_lift_fail:
        return PAIRED_REJECT_REASON_NO_LIFT
    any_valid_lift_transport_fail = any(
        bool(r.get("cartesian_descend_ok"))
        and bool(r.get("lift_ok"))
        and not bool(r.get("local_escape_ok", r.get("post_lift_exit_ok")))
        and str(r.get("result", "")).upper() != "ACCEPT"
        for r in paired_results
    )
    if any_valid_lift_transport_fail:
        return PAIRED_REJECT_REASON_TRANSPORT_AFTER_VALID_LIFT
    return PAIRED_REJECT_REASON_NO_TRANSPORT_EXIT
