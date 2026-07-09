"""Tests para filtrado de collision IDs stale en ejecución secuencial."""


def _is_stale_planning_scene_collision_id(col_id: str) -> bool:
    cid = str(col_id or "").strip()
    if not cid or cid == "vision_test_table":
        return False
    if cid.startswith("target_runtime_ycb_"):
        return True
    if cid.startswith("target_"):
        return True
    if cid.startswith("obstacle_"):
        return True
    return False


def test_stale_target_runtime_ycb() -> None:
    assert _is_stale_planning_scene_collision_id(
        "target_runtime_ycb_sugar_box_seed1002"
    )


def test_stale_target_label() -> None:
    assert _is_stale_planning_scene_collision_id("target_sugar_box")


def test_stale_obstacle() -> None:
    assert _is_stale_planning_scene_collision_id(
        "obstacle_runtime_ycb_cracker_box_seed1"
    )


def test_table_not_stale() -> None:
    assert not _is_stale_planning_scene_collision_id("vision_test_table")


def test_empty_not_stale() -> None:
    assert not _is_stale_planning_scene_collision_id("")
