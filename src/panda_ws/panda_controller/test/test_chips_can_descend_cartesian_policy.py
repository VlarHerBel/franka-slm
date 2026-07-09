"""Regresión: política cartesiana Z pura y gates XY/pre-descend chips_can."""

from panda_controller.chips_can_descend_cartesian import (
    build_chips_can_descend_z_waypoints,
    chips_can_descend_step_in_contact_zone,
    chips_can_descend_tcp_above_table_floor,
    chips_can_descend_use_cartesian_effective,
    chips_can_final_descend_avoid_collisions_effective,
    chips_can_pre_descend_pose_gate_ok,
    chips_can_skip_gripper_centering_verify,
    chips_can_tcp_in_final_descend_relax_zone,
    chips_can_xy_drift_m,
    chips_can_xy_drift_ok,
    subdivide_chips_can_descend_z_range,
)


def test_skip_gripper_centering_after_gt_verify() -> None:
    cand = {
        "label": "chips_can",
        "_chips_can_gt_centering_verified_at_pregrasp": True,
    }
    assert chips_can_skip_gripper_centering_verify(cand) is True
    assert chips_can_skip_gripper_centering_verify({"label": "cracker_box"}) is False


def test_descend_cartesian_forced_in_demo_fast() -> None:
    assert (
        chips_can_descend_use_cartesian_effective(
            param_value=False,
            demo_fast_mode=True,
            demo_motion_profile_active=False,
        )
        is True
    )


def test_descend_cartesian_forced_with_demo_profile() -> None:
    assert (
        chips_can_descend_use_cartesian_effective(
            param_value=False,
            demo_fast_mode=False,
            demo_motion_profile_active=True,
        )
        is True
    )


def test_descend_cartesian_respects_param_when_not_demo() -> None:
    assert (
        chips_can_descend_use_cartesian_effective(
            param_value=True,
            demo_fast_mode=False,
            demo_motion_profile_active=False,
        )
        is True
    )
    assert (
        chips_can_descend_use_cartesian_effective(
            param_value=False,
            demo_fast_mode=False,
            demo_motion_profile_active=False,
        )
        is False
    )


def test_xy_drift_gate() -> None:
    drift = chips_can_xy_drift_m((0.5713, -0.0571), (0.5856, -0.0533))
    assert drift > 0.01
    assert chips_can_xy_drift_ok(0.002, 0.003) is True
    assert chips_can_xy_drift_ok(0.004, 0.003) is False


def test_micro_descend_waypoints_from_610_to_475() -> None:
    wps = build_chips_can_descend_z_waypoints(0.610, 0.475, max_step_m=0.025)
    assert len(wps) >= 5
    assert wps[0] < 0.610
    assert abs(wps[-1] - 0.475) < 1e-3
    for a, b in zip(wps, wps[1:]):
        assert a > b
        assert a - b <= 0.025 + 1e-3


def test_subdivide_failed_segment() -> None:
    wps = subdivide_chips_can_descend_z_range(
        0.610, 0.525, max_step_m=0.025, min_step_m=0.010
    )
    assert len(wps) >= 3
    assert abs(wps[-1] - 0.525) < 1e-3


def test_pre_descend_pose_gate() -> None:
    top_z = 0.55
    ok, z_ok, c_ok = chips_can_pre_descend_pose_gate_ok(
        actual_tcp_z=top_z + 0.10,
        top_z_m=top_z,
        min_clearance_above_top_m=0.10,
        centering_error_xy_m=0.005,
        max_centering_error_xy_m=0.006,
    )
    assert ok is True
    assert z_ok is True
    assert c_ok is True

    ok2, z_ok2, _ = chips_can_pre_descend_pose_gate_ok(
        actual_tcp_z=top_z + 0.05,
        top_z_m=top_z,
        min_clearance_above_top_m=0.10,
        centering_error_xy_m=0.005,
        max_centering_error_xy_m=0.006,
    )
    assert ok2 is False
    assert z_ok2 is False


def test_final_descend_relax_zone() -> None:
    top_z = 0.50
    assert chips_can_tcp_in_final_descend_relax_zone(0.5488, top_z, 0.05) is True
    assert chips_can_tcp_in_final_descend_relax_zone(0.561, top_z, 0.05) is False


def test_descend_step_in_contact_zone() -> None:
    top_z = 0.50
    margin = 0.05
    assert chips_can_descend_step_in_contact_zone(0.5488, 0.5200, top_z, margin)
    assert not chips_can_descend_step_in_contact_zone(0.610, 0.5875, top_z, margin)


def test_final_descend_avoid_collisions_policy() -> None:
    top_z = 0.50
    margin = 0.05
    assert (
        chips_can_final_descend_avoid_collisions_effective(
            policy_enabled=True,
            avoid_collisions_in_contact_zone=False,
            current_tcp_z=0.5488,
            target_tcp_z=0.5200,
            top_z_m=top_z,
            contact_zone_above_top_m=margin,
            contact_zone_latched=False,
        )
        is False
    )
    assert (
        chips_can_final_descend_avoid_collisions_effective(
            policy_enabled=True,
            avoid_collisions_in_contact_zone=False,
            current_tcp_z=0.610,
            target_tcp_z=0.5875,
            top_z_m=top_z,
            contact_zone_above_top_m=margin,
            contact_zone_latched=False,
        )
        is None
    )
    assert (
        chips_can_final_descend_avoid_collisions_effective(
            policy_enabled=True,
            avoid_collisions_in_contact_zone=False,
            current_tcp_z=0.610,
            target_tcp_z=0.5875,
            top_z_m=top_z,
            contact_zone_above_top_m=margin,
            contact_zone_latched=True,
        )
        is False
    )


def test_descend_tcp_above_table_floor() -> None:
    assert chips_can_descend_tcp_above_table_floor(0.30, 0.27, 0.015) is True
    assert chips_can_descend_tcp_above_table_floor(0.28, 0.27, 0.015) is False
