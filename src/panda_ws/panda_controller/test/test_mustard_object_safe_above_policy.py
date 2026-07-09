"""Tests política object_safe_above baja para mustard_bottle."""

from panda_controller.mustard_object_safe_above_policy import (
    build_mustard_object_safe_above_z_candidates,
    format_mustard_object_safe_above_height_policy_log,
    mustard_object_safe_above_policy_active,
    resolve_mustard_object_safe_above_tcp_z,
)


def test_policy_active_only_mustard_tall_topdown() -> None:
    assert mustard_object_safe_above_policy_active("mustard_bottle", "tall_object_topdown")
    assert not mustard_object_safe_above_policy_active("mustard_bottle", "short_axis")
    assert not mustard_object_safe_above_policy_active("sugar_box", "tall_object_topdown")


def test_resolve_lower_than_generic_150_clearance() -> None:
    top_z = 0.437
    pregrasp_z = 0.492
    resolved = resolve_mustard_object_safe_above_tcp_z(
        top_z_m=top_z,
        selected_pregrasp_tcp_z=pregrasp_z,
        old_safe_above_tcp_z=0.587,
    )
    assert abs(resolved["new_safe_above_tcp_z"] - 0.522) < 1e-6
    assert resolved["new_safe_above_tcp_z"] < 0.587
    assert abs(resolved["clearance_above_pregrasp_m"] - 0.030) < 1e-6


def test_z_candidates_primary_then_increments_deduped() -> None:
    cands = build_mustard_object_safe_above_z_candidates(
        selected_pregrasp_tcp_z=0.492,
        top_z_m=0.437,
    )
    assert cands[0] == 0.522
    assert 0.512 in cands
    assert 0.542 in cands
    assert len(cands) == len(set(round(z, 6) for z in cands))


def test_height_policy_log_format() -> None:
    log = format_mustard_object_safe_above_height_policy_log(
        {
            "top_z": "0.437",
            "selected_pregrasp_tcp_z": "0.492",
            "old_safe_above_tcp_z": "0.587",
            "new_safe_above_tcp_z": "0.522",
            "old_hand_z": "0.687",
            "new_hand_z": "0.622",
            "ik_ok": "true",
            "plan_ok": "true",
            "result": "OK",
            "reason": "pick_workspace_ready_to_object_safe_above",
        }
    )
    assert "[MUSTARD_OBJECT_SAFE_ABOVE_HEIGHT_POLICY]" in log
    assert "new_safe_above_tcp_z=0.522" in log
    assert "result=OK" in log
