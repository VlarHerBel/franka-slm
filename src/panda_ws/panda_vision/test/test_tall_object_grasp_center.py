"""Tests de centro operativo top-down (tapón/cuello) para objetos altos."""

import math
from typing import Any, Dict

from panda_vision.grasp import object_grasp_policy as ogp
from panda_vision.grasp.object_grasp_policy import (
    TALL_OBJECT_CAP_CENTER_SOURCE,
    apply_tall_object_topdown_grasp_center_offset,
    get_grasp_policy,
    resolve_tall_object_top_z_m,
    tall_object_topdown_cap_offset_configured,
)


def test_mustard_bottle_cap_offset_configured() -> None:
    policy = get_grasp_policy("mustard_bottle")
    assert tall_object_topdown_cap_offset_configured(policy)
    assert policy.get("topdown_grasp_center_offset_long_m") == 0.0
    assert policy.get("topdown_grasp_center_offset_short_m") == 0.0


def test_cracker_box_no_cap_offset_path() -> None:
    policy = get_grasp_policy("cracker_box")
    assert not tall_object_topdown_cap_offset_configured(policy)


def test_resolve_top_z_from_geometry_center_not_payload_geom_z() -> None:
    top_z, dbg = resolve_tall_object_top_z_m(
        "mustard_bottle",
        0.356,
        height_m=0.1909,
        payload_top_z_before=0.356,
    )
    assert abs(top_z - 0.45145) < 1e-4
    assert dbg["source"] == "known_geometry_height"


def test_mustard_policy_min_z_fields() -> None:
    policy = get_grasp_policy("mustard_bottle")
    assert float(policy["min_top_z_m"]) == 0.42
    assert float(policy["min_pregrasp_tcp_z_m"]) == 0.49
    assert policy.get("use_palm_bridge_z_constraint") is True
    assert float(policy["palm_bridge_clearance_above_top_m"]) == 0.003
    assert float(policy["palm_bridge_below_panda_hand_m"]) == 0.063
    assert float(policy["panda_hand_to_grasp_tcp_z_m"]) == 0.100


def test_mustard_palm_bridge_grasp_tcp_formula() -> None:
    top_z = 0.45145
    clearance = 0.003
    below = 0.063
    hand_to_tcp = 0.100
    desired_bridge = top_z + clearance
    hand_z = desired_bridge + below
    grasp_tcp_z = hand_z - hand_to_tcp
    assert abs(grasp_tcp_z - (top_z + clearance + below - hand_to_tcp)) < 1e-6
    assert abs(grasp_tcp_z - (top_z - 0.035)) < 0.002


def test_mustard_zero_offset_keeps_body_center() -> None:
    grasp, src, dbg = apply_tall_object_topdown_grasp_center_offset(
        "mustard_bottle",
        (0.6492, -0.1765),
        1.2,
        0.45145,
    )
    assert src == TALL_OBJECT_CAP_CENTER_SOURCE
    assert dbg["applied"] is True
    assert abs(grasp[0] - 0.6492) < 1e-6
    assert abs(grasp[1] - (-0.1765)) < 1e-6
    assert abs(grasp[2] - 0.45145) < 1e-6


def test_mustard_offset_long_short_yaw() -> None:
    orig = ogp.get_grasp_policy

    def _policy_with_offset(label: str, **kwargs: Any) -> Dict[str, Any]:
        p = dict(orig(label, **kwargs))
        p["topdown_grasp_center_offset_long_m"] = 0.01
        p["topdown_grasp_center_offset_short_m"] = 0.02
        return p

    ogp.get_grasp_policy = _policy_with_offset  # type: ignore[method-assign]
    try:
        yaw = math.pi / 4.0
        grasp, src, dbg = apply_tall_object_topdown_grasp_center_offset(
            "mustard_bottle",
            (1.0, 2.0),
            yaw,
            0.5,
        )
        assert src == TALL_OBJECT_CAP_CENTER_SOURCE
        ol, os = 0.01, 0.02
        c, s = math.cos(yaw), math.sin(yaw)
        dx = ol * c - os * s
        dy = ol * s + os * c
        assert abs(grasp[0] - (1.0 + dx)) < 1e-6
        assert abs(grasp[1] - (2.0 + dy)) < 1e-6
        assert abs(dbg["offset_base_xy"][0] - dx) < 1e-6
        assert abs(dbg["offset_base_xy"][1] - dy) < 1e-6
    finally:
        ogp.get_grasp_policy = orig  # type: ignore[method-assign]
