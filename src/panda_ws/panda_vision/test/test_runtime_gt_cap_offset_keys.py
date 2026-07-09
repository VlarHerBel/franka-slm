"""Claves opcionales del RuntimeScene al resolver centro de tapón (sin KeyError)."""

from panda_vision.nodes.perception_node import _resolve_tall_cap_offset_local_xy


def test_offset_local_xy_from_mustard_entry_field() -> None:
    entry = {
        "mustard_top_cap_center_offset_local_xyz": [0.0217, 0.0311, 0.0616],
    }
    xy, src = _resolve_tall_cap_offset_local_xy(entry, {})
    assert xy == (0.0217, 0.0311)
    assert src == "mustard_top_cap_center_offset_local_xyz"


def test_offset_local_xy_prefers_tall_dbg_when_present() -> None:
    entry = {"mustard_top_cap_center_offset_local_xyz": [1.0, 2.0, 3.0]}
    xy, src = _resolve_tall_cap_offset_local_xy(
        entry, {"offset_local_xy": (0.5, -0.5)}
    )
    assert xy == (0.5, -0.5)
    assert src == "offset_local_xy"


def test_offset_local_xy_fallback_zero() -> None:
    xy, src = _resolve_tall_cap_offset_local_xy({}, {})
    assert xy == (0.0, 0.0)
    assert src == "fallback_zero"
