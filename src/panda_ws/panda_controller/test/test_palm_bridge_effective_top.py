"""Tests de effective_top_z y fórmula palm bridge para mustard_bottle."""

from panda_controller.palm_bridge_policy import (
    compute_palm_bridge_grasp_tcp_z,
    resolve_effective_top_z_for_palm_bridge,
)


def test_effective_top_uses_max_of_payload_expected_and_mesh() -> None:
    candidate = {
        "top_z_m": 0.4366,
        "height_m": 0.1909,
        "mustard_mesh_local_cap_center_base": [0.65, -0.18, 0.4514],
    }
    eff, src, meta = resolve_effective_top_z_for_palm_bridge(
        candidate, 0.4366, table_z_m=0.27
    )
    assert abs(eff - 0.4609) < 1e-4
    assert src == "expected_top_z_table_plus_height"
    assert abs(float(meta["mesh_local_cap_center_z"]) - 0.4514) < 1e-4


def test_effective_top_prefers_expected_when_higher_than_mesh() -> None:
    candidate = {
        "top_z_m": 0.4366,
        "height_m": 0.1909,
        "mustard_mesh_local_cap_center_base": [0.0, 0.0, 0.44],
    }
    eff, src, _ = resolve_effective_top_z_for_palm_bridge(
        candidate, 0.4366, table_z_m=0.27
    )
    assert abs(eff - 0.4609) < 1e-4
    assert src == "expected_top_z_table_plus_height"


def test_palm_bridge_grasp_tcp_from_effective_top() -> None:
    eff = 0.4609
    grasp_z, hand_z, bridge_z = compute_palm_bridge_grasp_tcp_z(
        eff,
        clearance_m=0.003,
        palm_bridge_below_panda_hand_m=0.063,
        panda_hand_to_grasp_tcp_z_m=0.100,
    )
    assert abs(bridge_z - 0.4639) < 1e-6
    assert abs(hand_z - 0.5269) < 1e-6
    assert abs(grasp_z - 0.4269) < 1e-6
    assert grasp_z > 0.4016
