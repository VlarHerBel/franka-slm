"""Búsqueda depth/descend mustard_bottle demo_scene_02 (sin ROS)."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

from panda_controller.mustard_simple_descend_prevalidate import (
    build_mustard_descend_candidate_specs,
)
from panda_controller.mustard_top_z_policy import (
    MUSTARD_OPERATIONAL_GRASP_DEPTH_FROM_TOP_M,
    MUSTARD_SCANNER_CONTRACT_GRASP_TCP_Z_M,
    MUSTARD_SCANNER_CONTRACT_PREGRASP_TCP_Z_M,
    MUSTARD_SCANNER_CONTRACT_TOP_Z_M,
    MUSTARD_SCANNER_Z_MATCH_TOLERANCE_M,
    expected_mustard_grasp_tcp_z_for_runtime_top,
    mustard_tall_object_topdown_active,
)

# Golden reachability scanner (demo_scene_02 mustard pose 0.660, 0.060, 1.6392).
DEMO_SCENE_02_MUSTARD_SCANNER_ALIGNED_PREGRASP_Z = 0.4909
DEMO_SCENE_02_MUSTARD_SCANNER_ALIGNED_GRASP_Z = (
    MUSTARD_SCANNER_CONTRACT_TOP_Z_M - MUSTARD_OPERATIONAL_GRASP_DEPTH_FROM_TOP_M
)
DEMO_SCENE_02_MUSTARD_SCANNER_ALIGNED_DEPTH_FROM_EFFECTIVE_TOP_M = (
    MUSTARD_OPERATIONAL_GRASP_DEPTH_FROM_TOP_M
)
DEMO_SCENE_02_MUSTARD_SCANNER_ALIGNED_COMMANDED_YAW_RAD = -3.073189
DEMO_SCENE_02_MUSTARD_SCANNER_ALIGNED_SEED = "pick_workspace_ready"
DEMO_SCENE_02_MUSTARD_MIN_GRASP_Z_M = 0.388

MUSTARD_GRASP_ELEVATION_FALLBACK_Z_M: Tuple[float, ...] = (
    0.422,
    0.427,
    0.432,
)


def mustard_bottle_extended_pick_scene_active(scene_id: str) -> bool:
    """demo_scene_02, chips_mustard_* o deposit_* con política demo_scene_02."""
    sid = str(scene_id or "").strip().lower()
    if sid == "demo_scene_02" or sid.startswith("chips_mustard"):
        return True
    from panda_vision.spawn.demo_scene_presets import demo_scene_policy_scene_id_for_preset

    return demo_scene_policy_scene_id_for_preset(sid) == "demo_scene_02"


def mustard_demo_scene_depth_search_active(label: str, scene_id: str) -> bool:
    return (
        str(label or "").strip().lower() == "mustard_bottle"
        and mustard_bottle_extended_pick_scene_active(scene_id)
    )


MUSTARD_DEMO_SCENE_02_ADAPTIVE_LIFT_M: Tuple[float, ...] = (
    0.060,
    0.080,
    0.100,
    0.120,
    0.150,
)


def mustard_adaptive_lift_prevalidate_active(label: str, scene_id: str) -> bool:
    """Lift cartesiano adaptivo en prevalidación (mustard_bottle + demo_scene_02)."""
    return mustard_demo_scene_depth_search_active(label, scene_id)


def _wrap_to_pi(yaw: float) -> float:
    return float(math.atan2(math.sin(float(yaw)), math.cos(float(yaw))))


def build_mustard_scanner_aligned_descend_spec(
    *,
    xy: Tuple[float, float],
    pre_plan: Tuple[float, float, float],
    top_z_m: float,
    effective_top_z_m: Optional[float] = None,
) -> Optional[Dict[str, Any]]:
    """Candidato prioritario: pregrasp scanner + grasp a profundidad operacional en tapón."""
    eff_top = (
        float(effective_top_z_m)
        if effective_top_z_m is not None
        else float(top_z_m)
    )
    depth_m = float(MUSTARD_OPERATIONAL_GRASP_DEPTH_FROM_TOP_M)
    gr_z = float(eff_top) - depth_m
    scanner_pre_z = float(MUSTARD_SCANNER_CONTRACT_PREGRASP_TCP_Z_M)
    if scanner_pre_z <= gr_z + 1e-6:
        return None
    depth = float(eff_top) - gr_z
    return {
        "pregrasp_tcp": (float(xy[0]), float(xy[1]), scanner_pre_z),
        "grasp_tcp": (float(xy[0]), float(xy[1]), gr_z),
        "depth_from_top_m": depth,
        "descend_delta_m": scanner_pre_z - gr_z,
        "source": "scanner_aligned",
        "scanner_aligned": True,
        "scanner_pregrasp_tcp_z": scanner_pre_z,
        "scanner_grasp_tcp_z": gr_z,
        "scanner_reference_top_z": eff_top,
        "scanner_seed": DEMO_SCENE_02_MUSTARD_SCANNER_ALIGNED_SEED,
        "commanded_tcp_yaw_rad": float(
            DEMO_SCENE_02_MUSTARD_SCANNER_ALIGNED_COMMANDED_YAW_RAD
        ),
        "endpoint_ik_source": "scanner_aligned",
    }


def build_mustard_grasp_elevation_fallback_specs(
    *,
    xy: Tuple[float, float],
    pre_z: float,
    base_grasp_z: float,
    top_z_m: float,
    effective_top_z_m: Optional[float] = None,
) -> List[Dict[str, Any]]:
    """Eleva grasp_tcp_z desde el nominal fallido (+5/+10/+15 mm), sin bajar de 0.417."""
    specs: List[Dict[str, Any]] = []
    pre = float(pre_z)
    eff_top = (
        float(effective_top_z_m)
        if effective_top_z_m is not None
        else float(top_z_m)
    )
    for gz in MUSTARD_GRASP_ELEVATION_FALLBACK_Z_M:
        gz_f = float(gz)
        if gz_f + 1e-6 < float(DEMO_SCENE_02_MUSTARD_MIN_GRASP_Z_M):
            continue
        if gz_f + 1e-6 < float(base_grasp_z):
            continue
        if gz_f >= pre - 1e-4:
            continue
        specs.append(
            {
                "pregrasp_tcp": (float(xy[0]), float(xy[1]), pre),
                "grasp_tcp": (float(xy[0]), float(xy[1]), gz_f),
                "depth_from_top_m": float(eff_top) - gz_f,
                "descend_delta_m": pre - gz_f,
                "source": "endpoint_ik_elevation_fallback",
                "elevation_grasp_tcp_z": gz_f,
                "endpoint_ik_source": "endpoint_ik_elevation_fallback",
            }
        )
    return specs


def resolve_mustard_descend_candidate_pose(
    spec: Dict[str, Any],
    *,
    nominal_pre_plan: Tuple[float, float, float],
    nominal_gr_plan: Tuple[float, float, float],
    variant_commanded_yaw_rad: float,
) -> Tuple[
    Tuple[float, float, float],
    Tuple[float, float, float],
    float,
    str,
]:
    """Resuelve pregrasp/grasp/yaw efectivos para evaluar endpoint IK."""
    xy = (float(nominal_pre_plan[0]), float(nominal_pre_plan[1]))
    endpoint_source = str(spec.get("endpoint_ik_source") or spec.get("source") or "depth_search")

    pre_raw = spec.get("pregrasp_tcp")
    gr_raw = spec.get("grasp_tcp")
    if isinstance(pre_raw, (list, tuple)) and len(pre_raw) >= 3:
        pre_plan = (float(pre_raw[0]), float(pre_raw[1]), float(pre_raw[2]))
    else:
        pre_plan = (xy[0], xy[1], float(nominal_pre_plan[2]))
    if isinstance(gr_raw, (list, tuple)) and len(gr_raw) >= 3:
        gr_plan = (float(gr_raw[0]), float(gr_raw[1]), float(gr_raw[2]))
    else:
        gr_plan = nominal_gr_plan

    yaw_raw = spec.get("commanded_tcp_yaw_rad")
    if yaw_raw is not None:
        commanded_yaw = _wrap_to_pi(float(yaw_raw))
    else:
        commanded_yaw = _wrap_to_pi(float(variant_commanded_yaw_rad))

    if bool(spec.get("scanner_aligned")):
        endpoint_source = "scanner_aligned"
    elif str(spec.get("source")) == "nominal":
        endpoint_source = "controller_nominal"

    return pre_plan, gr_plan, commanded_yaw, endpoint_source


def mustard_scanner_aligned_contract_active(
    candidate: Dict[str, Any],
    scene_id: str,
) -> bool:
    return mustard_tall_object_topdown_active(candidate) and mustard_demo_scene_depth_search_active(
        str(candidate.get("label", "")),
        str(scene_id or ""),
    )


def apply_mustard_scanner_aligned_pregrasp_to_seq(
    candidate: Dict[str, Any],
    seq: Dict[str, Any],
    *,
    top_z: float,
    scene_id: str,
) -> Optional[Dict[str, Any]]:
    """Fuerza pregrasp/grasp Z del contrato scanner (demo_scene_02 mustard)."""
    if not mustard_scanner_aligned_contract_active(candidate, scene_id):
        return None
    pre = seq.get("pregrasp_tcp")
    gr = seq.get("grasp_tcp")
    if not isinstance(pre, (list, tuple)) or len(pre) < 3:
        return None
    if not isinstance(gr, (list, tuple)) or len(gr) < 3:
        return None
    old_pre_z = float(pre[2])
    new_pre_z = float(MUSTARD_SCANNER_CONTRACT_PREGRASP_TCP_Z_M)
    new_grasp_z = float(expected_mustard_grasp_tcp_z_for_runtime_top(float(top_z)))
    new_gr = (float(gr[0]), float(gr[1]), new_grasp_z)
    seq["grasp_tcp"] = new_gr
    new_pre = (float(pre[0]), float(pre[1]), new_pre_z)
    seq["pregrasp_tcp"] = new_pre
    seq["final_descend_m"] = max(0.0, new_pre_z - new_grasp_z)
    seq["requested_descend_m"] = float(seq["final_descend_m"])
    seq["descend_limit_reason"] = "mustard_scanner_aligned_contract"
    safe = seq.get("safe_pregrasp_tcp")
    if isinstance(safe, (list, tuple)) and len(safe) >= 3:
        safe_z = max(float(safe[2]), new_pre_z)
        seq["safe_pregrasp_tcp"] = (float(safe[0]), float(safe[1]), float(safe_z))
    candidate["mustard_scanner_aligned_pregrasp_locked"] = True
    candidate["_mustard_scanner_aligned_pregrasp_tcp_z"] = new_pre_z
    candidate["_mustard_scanner_aligned_locked_pregrasp_tcp"] = [
        float(new_pre[0]),
        float(new_pre[1]),
        float(new_pre[2]),
    ]
    candidate["_mustard_scanner_aligned_locked_grasp_tcp"] = [
        float(new_gr[0]),
        float(new_gr[1]),
        float(new_grasp_z),
    ]
    candidate["_mustard_scanner_aligned_locked_top_z_m"] = float(top_z)
    return {
        "old_pregrasp_tcp_z": old_pre_z,
        "new_pregrasp_tcp_z": new_pre_z,
        "grasp_tcp_z": new_grasp_z,
        "top_z": float(top_z),
        "source": "scanner_aligned_contract",
        "result": "OK",
    }


def format_mustard_scanner_aligned_pregrasp_applied_log(fields: Dict[str, Any]) -> str:
    return (
        "[MUSTARD_SCANNER_ALIGNED_PREGRASP_APPLIED]\n"
        "old_pregrasp_tcp_z=%.4f\n"
        "new_pregrasp_tcp_z=%.4f\n"
        "grasp_tcp_z=%.4f\n"
        "top_z=%.4f\n"
        "source=%s\n"
        "result=%s"
        % (
            float(fields.get("old_pregrasp_tcp_z", 0.0)),
            float(fields.get("new_pregrasp_tcp_z", 0.0)),
            float(fields.get("grasp_tcp_z", 0.0)),
            float(fields.get("top_z", 0.0)),
            str(fields.get("source") or "scanner_aligned_contract"),
            str(fields.get("result") or "OK"),
        )
    )


def mustard_scanner_aligned_pregrasp_locked(candidate: Dict[str, Any]) -> bool:
    return bool(candidate.get("mustard_scanner_aligned_pregrasp_locked"))


def enforce_mustard_scanner_aligned_pregrasp_plan_targets(
    candidate: Dict[str, Any],
    plan_targets: Dict[str, Any],
) -> Tuple[bool, str, Optional[float], Optional[float]]:
    """Restaura pregrasp/grasp bloqueados por contrato scanner-aligned."""
    if not mustard_scanner_aligned_pregrasp_locked(candidate):
        return True, "not_locked", None, None

    locked_pre = candidate.get("_mustard_scanner_aligned_locked_pregrasp_tcp")
    locked_gr = candidate.get("_mustard_scanner_aligned_locked_grasp_tcp")
    if not isinstance(locked_pre, (list, tuple)) or len(locked_pre) < 3:
        return False, "missing_locked_pregrasp_tcp", None, None
    if not isinstance(locked_gr, (list, tuple)) or len(locked_gr) < 3:
        return False, "missing_locked_grasp_tcp", None, None

    expected_pre_z = float(MUSTARD_SCANNER_CONTRACT_PREGRASP_TCP_Z_M)
    locked_pre_tuple = (
        float(locked_pre[0]),
        float(locked_pre[1]),
        expected_pre_z,
    )
    locked_gr_tuple = (
        float(locked_gr[0]),
        float(locked_gr[1]),
        float(
            expected_mustard_grasp_tcp_z_for_runtime_top(
                float(
                    candidate.get("_mustard_scanner_aligned_locked_top_z_m")
                    or candidate.get("top_z_m")
                    or MUSTARD_SCANNER_CONTRACT_TOP_Z_M
                )
            )
        ),
    )
    plan_targets["pregrasp_tcp"] = locked_pre_tuple
    plan_targets["grasp_tcp"] = locked_gr_tuple
    candidate["pregrasp_tcp"] = list(locked_pre_tuple)
    candidate["grasp_tcp"] = list(locked_gr_tuple)
    candidate["pregrasp_tcp_z"] = expected_pre_z
    candidate["grasp_tcp_z"] = float(locked_gr_tuple[2])

    seq = candidate.get("_descend_tcp_sequence")
    if isinstance(seq, dict):
        seq["pregrasp_tcp"] = locked_pre_tuple
        seq["grasp_tcp"] = locked_gr_tuple
        seq["final_descend_m"] = max(
            0.0, expected_pre_z - float(locked_gr_tuple[2])
        )

    final_descend = expected_pre_z - float(locked_gr_tuple[2])
    candidate["vertical_descend_tcp_m"] = final_descend
    candidate["effective_approach_distance_m"] = final_descend

    selected_z: Optional[float] = None
    pre_tcp = plan_targets.get("pregrasp_tcp")
    if isinstance(pre_tcp, (list, tuple)) and len(pre_tcp) >= 3:
        selected_z = float(pre_tcp[2])
        if selected_z + 1e-6 < expected_pre_z:
            return (
                False,
                "pregrasp_lowered_after_scanner_contract",
                selected_z,
                expected_pre_z,
            )
    return True, "locked", selected_z, expected_pre_z


def verify_mustard_scanner_aligned_post_reachability(
    *,
    pregrasp_tcp_z: float,
    grasp_tcp_z: float,
    top_z: Optional[float] = None,
    tolerance_m: float = MUSTARD_SCANNER_Z_MATCH_TOLERANCE_M,
) -> Tuple[bool, Dict[str, Any]]:
    if top_z is None:
        tol = float(tolerance_m)
        pre_ok = (
            abs(float(pregrasp_tcp_z) - MUSTARD_SCANNER_CONTRACT_PREGRASP_TCP_Z_M)
            <= tol
        )
        gr_ok = (
            abs(float(grasp_tcp_z) - MUSTARD_SCANNER_CONTRACT_GRASP_TCP_Z_M) <= tol
        )
        ok = bool(pre_ok and gr_ok)
        fields = {
            "pregrasp_tcp_z": float(pregrasp_tcp_z),
            "expected_pregrasp_tcp_z": float(MUSTARD_SCANNER_CONTRACT_PREGRASP_TCP_Z_M),
            "grasp_tcp_z": float(grasp_tcp_z),
            "expected_grasp_tcp_z": float(MUSTARD_SCANNER_CONTRACT_GRASP_TCP_Z_M),
            "top_z": None,
            "expected_top_z": float(MUSTARD_SCANNER_CONTRACT_TOP_Z_M),
            "result": "OK" if ok else "FAIL",
        }
        return ok, fields
    from panda_controller.mustard_top_z_policy import verify_mustard_scanner_aligned_z_contract

    ok, core = verify_mustard_scanner_aligned_z_contract(
        controller_top_z=float(top_z),
        controller_pregrasp_tcp_z=float(pregrasp_tcp_z),
        controller_grasp_tcp_z=float(grasp_tcp_z),
        tolerance_m=float(tolerance_m),
    )
    expected_gr = expected_mustard_grasp_tcp_z_for_runtime_top(float(top_z))
    fields = {
        "pregrasp_tcp_z": float(pregrasp_tcp_z),
        "expected_pregrasp_tcp_z": float(MUSTARD_SCANNER_CONTRACT_PREGRASP_TCP_Z_M),
        "grasp_tcp_z": float(grasp_tcp_z),
        "expected_grasp_tcp_z": float(expected_gr),
        "top_z": float(top_z),
        "expected_top_z": float(MUSTARD_SCANNER_CONTRACT_TOP_Z_M),
        "verify_mode": str(core.get("verify_mode") or "n/a"),
        "result": "OK" if ok else "FAIL",
    }
    return ok, fields


def format_mustard_scanner_aligned_post_reachability_verify_log(
    fields: Dict[str, Any],
) -> str:
    top_z = fields.get("top_z")
    return (
        "[MUSTARD_SCANNER_ALIGNED_POST_REACHABILITY_VERIFY]\n"
        "pregrasp_tcp_z=%.4f\n"
        "expected_pregrasp_tcp_z=%.4f\n"
        "grasp_tcp_z=%.4f\n"
        "expected_grasp_tcp_z=%.4f\n"
        "top_z=%s\n"
        "expected_top_z=%.4f\n"
        "result=%s"
        % (
            float(fields["pregrasp_tcp_z"]),
            float(fields["expected_pregrasp_tcp_z"]),
            float(fields["grasp_tcp_z"]),
            float(fields["expected_grasp_tcp_z"]),
            "n/a" if top_z is None else "%.4f" % float(top_z),
            float(fields["expected_top_z"]),
            str(fields.get("result", "FAIL")),
        )
    )


def format_mustard_scanner_aligned_candidate_log(fields: Dict[str, Any]) -> str:
    return (
        "[MUSTARD_SCANNER_ALIGNED_CANDIDATE]\n"
        "scanner_pregrasp_tcp_z=%.4f\n"
        "scanner_grasp_tcp_z=%.4f\n"
        "controller_pregrasp_tcp_z=%.4f\n"
        "controller_grasp_tcp_z=%.4f\n"
        "selected=%s\n"
        "result=%s"
        % (
            float(DEMO_SCENE_02_MUSTARD_SCANNER_ALIGNED_PREGRASP_Z),
            float(DEMO_SCENE_02_MUSTARD_SCANNER_ALIGNED_GRASP_Z),
            float(fields.get("controller_pregrasp_tcp_z", 0.0)),
            float(fields.get("controller_grasp_tcp_z", 0.0)),
            str(bool(fields.get("selected", False))).lower(),
            str(fields.get("result") or "n/a"),
        )
    )


def format_mustard_endpoint_ik_candidate_selected_log(fields: Dict[str, Any]) -> str:
    return (
        "[MUSTARD_ENDPOINT_IK_CANDIDATE_SELECTED]\n"
        "pregrasp_tcp_z=%.4f\n"
        "grasp_tcp_z=%.4f\n"
        "commanded_tcp_yaw_rad=%.6f\n"
        "source=%s"
        % (
            float(fields.get("pregrasp_tcp_z", 0.0)),
            float(fields.get("grasp_tcp_z", 0.0)),
            float(fields.get("commanded_tcp_yaw_rad", 0.0)),
            str(fields.get("source") or "n/a"),
        )
    )


def resolve_mustard_descend_fail_reason(
    *,
    last_reason: str,
    endpoint_ik_attempted: bool,
    endpoint_ik_failed: bool,
) -> str:
    if endpoint_ik_failed or (
        endpoint_ik_attempted and str(last_reason) == "depth_too_shallow"
    ):
        return "endpoint_ik_fail"
    return str(last_reason or "no_valid_descend_candidate")


def build_mustard_demo_descend_candidate_specs(
    *,
    pre_plan: Tuple[float, float, float],
    top_z_m: float,
    xy: Tuple[float, float],
    min_grasp_z_m: Optional[float],
    gr_plan_nominal: Tuple[float, float, float],
    effective_top_z_m: Optional[float] = None,
) -> List[Dict[str, Any]]:
    """Orden: scanner-aligned, elevación IK, resto filtrado (grasp_z >= 0.417)."""
    specs: List[Dict[str, Any]] = []
    seen_z: set[float] = set()
    floor_z = max(
        float(min_grasp_z_m) if min_grasp_z_m is not None else 0.0,
        float(DEMO_SCENE_02_MUSTARD_MIN_GRASP_Z_M),
    )
    pre_z = float(pre_plan[2])

    def _append(spec: Dict[str, Any]) -> None:
        gr_raw = spec.get("grasp_tcp")
        if not isinstance(gr_raw, (list, tuple)) or len(gr_raw) < 3:
            return
        gz = round(float(gr_raw[2]), 5)
        key = round(gz, 3)
        if key in seen_z:
            return
        if gz + 1e-6 < floor_z:
            return
        pre_for_limit = float(spec.get("pregrasp_tcp", pre_plan)[2]) if isinstance(
            spec.get("pregrasp_tcp"), (list, tuple)
        ) else pre_z
        if gz >= float(pre_for_limit) - 1e-4:
            return
        seen_z.add(key)
        specs.append(spec)

    scanner = build_mustard_scanner_aligned_descend_spec(
        xy=xy,
        pre_plan=pre_plan,
        top_z_m=float(top_z_m),
        effective_top_z_m=effective_top_z_m,
    )
    if scanner is not None:
        _append(scanner)

    for elev_spec in build_mustard_grasp_elevation_fallback_specs(
        xy=xy,
        pre_z=pre_z,
        base_grasp_z=float(gr_plan_nominal[2]),
        top_z_m=float(top_z_m),
        effective_top_z_m=effective_top_z_m,
    ):
        _append(elev_spec)

    for spec in build_mustard_descend_candidate_specs(
        pre_plan=pre_plan,
        top_z_m=float(top_z_m),
        xy=xy,
        min_grasp_z_m=floor_z,
    ):
        spec = dict(spec)
        spec.setdefault(
            "pregrasp_tcp",
            (float(xy[0]), float(xy[1]), pre_z),
        )
        spec.setdefault("endpoint_ik_source", "depth_search")
        _append(spec)

    return specs
