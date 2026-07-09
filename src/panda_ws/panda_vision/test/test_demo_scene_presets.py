"""Regresión: presets demo_scene_01/02/03 (4 objetos, chips_can seeds, separación)."""

from __future__ import annotations

import math

from panda_vision.spawn.demo_scene_presets import (
    assert_builtin_presets_valid,
    CHIPS_CAN_ALLOWED_SEEDS,
    CHIPS_CAN_BANNED_SEED,
    DEMO_SCENE_02_PICK_ORDER,
    DEMO_SCENE_02_REMAINING_SUGAR_MUSTARD_PICK_ORDER,
    DEMO_SCENE_02_REMAINING_SUGAR_MUSTARD_SCENE_ID,
    DEMO_SCENE_3OBJ_OMITTED_LABEL,
    DEMO_SCENE_3OBJ_PARENT_SCENE_IDS,
    DEMO_SCENE_3OBJ_PICK_ORDER,
    DEMO_SCENE_LAYOUT_FOOTPRINT_SCALE,
    DEMO_SCENE_OBJECT_LABELS,
    DEMO_SCENE_PRESETS,
    chips_can_seed_pose_uniform,
    demo_scene_policy_scene_id_for_preset,
    get_demo_scene_preset,
    is_chips_can_banned_seed,
    is_chips_can_demo_xy_allowed,
    is_consolidated_demo_scene_objects,
    is_demo_scene_3obj_scene_id,
    log_demo_scene_vision_labels,
    object_footprint_fits_working_table,
    object_in_demo_scene_02_reach_zone,
    runtime_labels_from_scene_objects,
    validate_demo_scene_02_clear_table_layout,
    validate_demo_scene_02_remaining_sugar_mustard_layout,
    validate_demo_scene_layout,
    validate_demo_scene_preset,
    is_two_boxes_scene_preset,
)

CONSOLIDATED_FOUR_OBJECT_PRESET_IDS = (
    "demo_scene_01",
    "demo_scene_02",
    "demo_scene_03",
)


def test_builtin_presets_pass_import_time_assert() -> None:
    assert_builtin_presets_valid()


def test_all_demo_presets_validate_at_default_clearance() -> None:
    for scene_id in (
        "demo_scene_01",
        "demo_scene_02",
        "demo_scene_03",
        "two_boxes_01",
        "two_boxes_02",
        "two_boxes_03",
        "chips_mustard_01",
        "chips_mustard_02",
    ):
        ok, reason = validate_demo_scene_preset(
            get_demo_scene_preset(scene_id),
            footprint_safety_scale=DEMO_SCENE_LAYOUT_FOOTPRINT_SCALE,
        )
        assert ok, f"{scene_id}: {reason}"


def test_demo_presets_fail_if_spawner_random_footprint_scale_applied() -> None:
    """Regresión: footprint_safety_scale=1.1 del spawner rechazaba presets válidos."""
    for scene_id in ("demo_scene_01", "demo_scene_03"):
        ok, _reason = validate_demo_scene_preset(
            get_demo_scene_preset(scene_id),
            min_clearance_m=0.03,
            footprint_safety_scale=1.1,
        )
        assert not ok, f"{scene_id} no debe validarse con escala aleatoria 1.1"


def test_demo_scene_02_has_extra_margin_at_spawner_footprint_scale() -> None:
    """demo_scene_02 v2: válido con escala layout 1.0 (no escala aleatoria 1.1)."""
    preset = get_demo_scene_preset("demo_scene_02")
    ok, reason = validate_demo_scene_02_clear_table_layout(
        preset,
        min_clearance_m=0.03,
        footprint_safety_scale=DEMO_SCENE_LAYOUT_FOOTPRINT_SCALE,
    )
    assert ok, reason
    ok_loose, _reason = validate_demo_scene_preset(
        preset,
        min_clearance_m=0.03,
        footprint_safety_scale=1.1,
    )
    assert not ok_loose


def test_demo_scene_01_fails_with_clearance_005() -> None:
    ok, _reason = validate_demo_scene_preset(
        get_demo_scene_preset("demo_scene_01"),
        min_clearance_m=0.05,
        footprint_safety_scale=DEMO_SCENE_LAYOUT_FOOTPRINT_SCALE,
    )
    assert not ok


def test_demo_presets_pass_launch_clearance_with_layout_scale() -> None:
    for scene_id in ("demo_scene_01", "demo_scene_02", "demo_scene_03"):
        ok, reason = validate_demo_scene_preset(
            get_demo_scene_preset(scene_id),
            min_clearance_m=0.03,
            footprint_safety_scale=DEMO_SCENE_LAYOUT_FOOTPRINT_SCALE,
        )
        assert ok, f"{scene_id}: {reason}"


def test_demo_presets_have_four_consolidated_labels() -> None:
    for scene_id in CONSOLIDATED_FOUR_OBJECT_PRESET_IDS:
        preset = get_demo_scene_preset(scene_id)
        labels = {o.label for o in preset.objects}
        assert labels == set(DEMO_SCENE_OBJECT_LABELS)


def test_demo_scene_02_remaining_sugar_mustard_layout() -> None:
    preset = get_demo_scene_preset(DEMO_SCENE_02_REMAINING_SUGAR_MUSTARD_SCENE_ID)
    assert preset.scene_id == DEMO_SCENE_02_REMAINING_SUGAR_MUSTARD_SCENE_ID
    labels = {o.label for o in preset.objects}
    assert labels == {"sugar_box", "mustard_bottle"}
    sugar = next(o for o in preset.objects if o.label == "sugar_box")
    mustard = next(o for o in preset.objects if o.label == "mustard_bottle")
    assert math.isclose(sugar.x, 0.6300, abs_tol=1e-4)
    assert math.isclose(sugar.y, -0.1750, abs_tol=1e-4)
    assert math.isclose(sugar.yaw, -3.0159, abs_tol=1e-3)
    assert sugar.order_index == 0
    assert math.isclose(mustard.x, 0.6570, abs_tol=1e-4)
    assert math.isclose(mustard.y, 0.0360, abs_tol=1e-4)
    assert math.isclose(mustard.yaw, 1.6392, abs_tol=1e-3)
    assert mustard.order_index == 1
    ok, reason = validate_demo_scene_02_remaining_sugar_mustard_layout(preset)
    assert ok, reason
    ok2, reason2 = validate_demo_scene_02_clear_table_layout(preset)
    assert ok2, reason2
    assert list(DEMO_SCENE_02_REMAINING_SUGAR_MUSTARD_PICK_ORDER) == [
        "sugar_box",
        "mustard_bottle",
    ]
    assert demo_scene_policy_scene_id_for_preset(
        DEMO_SCENE_02_REMAINING_SUGAR_MUSTARD_SCENE_ID
    ) == "demo_scene_02"
    assert demo_scene_policy_scene_id_for_preset("deposit_02_cracker_chips") == (
        "demo_scene_02"
    )
    assert demo_scene_policy_scene_id_for_preset("deposit_03_mustard_only") == (
        "demo_scene_02"
    )
    assert demo_scene_policy_scene_id_for_preset("chips_mustard_02") == (
        "chips_mustard_02"
    )
    assert demo_scene_policy_scene_id_for_preset("deposit_full_1table") == (
        "deposit_full_1table"
    )


def test_three_object_demo_presets_derived_from_parent_layouts() -> None:
    for parent_id in DEMO_SCENE_3OBJ_PARENT_SCENE_IDS:
        child_id = f"{parent_id}_3obj"
        assert is_demo_scene_3obj_scene_id(child_id)
        parent = get_demo_scene_preset(parent_id)
        child = get_demo_scene_preset(child_id)
        assert len(child.objects) == 3
        labels = {o.label for o in child.objects}
        assert labels == set(DEMO_SCENE_3OBJ_PICK_ORDER)
        assert DEMO_SCENE_3OBJ_OMITTED_LABEL not in labels
        for obj in child.objects:
            parent_obj = next(o for o in parent.objects if o.label == obj.label)
            assert math.isclose(obj.x, parent_obj.x, abs_tol=1e-4)
            assert math.isclose(obj.y, parent_obj.y, abs_tol=1e-4)
            assert math.isclose(obj.yaw, parent_obj.yaw, abs_tol=1e-4)
        ok, reason = validate_demo_scene_preset(
            child,
            footprint_safety_scale=DEMO_SCENE_LAYOUT_FOOTPRINT_SCALE,
        )
        assert ok, f"{child_id}: {reason}"
        assert demo_scene_policy_scene_id_for_preset(child_id) == parent_id


def test_nogolden_3obj_preset_resolves_parent_policy_scene() -> None:
    assert is_demo_scene_3obj_scene_id("demo_scene_02_3obj_nogolden")
    assert demo_scene_policy_scene_id_for_preset("demo_scene_02_3obj_nogolden") == (
        "demo_scene_02"
    )


def test_demo_scene_02_3obj_clear_table_layout() -> None:
    preset = get_demo_scene_preset("demo_scene_02_3obj")
    ok, reason = validate_demo_scene_02_clear_table_layout(preset)
    assert ok, reason


def test_chips_can_uses_allowed_seeds_per_scene() -> None:
    seeds = []
    for scene_id in CONSOLIDATED_FOUR_OBJECT_PRESET_IDS:
        preset = get_demo_scene_preset(scene_id)
        chips = [o for o in preset.objects if o.label == "chips_can"][0]
        assert chips.seed in CHIPS_CAN_ALLOWED_SEEDS
        seeds.append(int(chips.seed))
    assert sorted(seeds) == sorted(CHIPS_CAN_ALLOWED_SEEDS)


def test_chips_can_seed1001_pose_is_banned_region() -> None:
    x, y, _yaw = chips_can_seed_pose_uniform(CHIPS_CAN_BANNED_SEED)
    allowed, _reason = is_chips_can_demo_xy_allowed(x, y)
    assert not allowed
    assert abs(x - 0.6492) < 0.002
    assert abs(y + 0.1765) < 0.002


def test_chips_can_allowed_seeds_near_uniform_sampler() -> None:
    """XY puede ajustarse ligeramente para layout; yaw y seed se conservan."""
    for seed in CHIPS_CAN_ALLOWED_SEEDS:
        preset = next(
            p
            for p in DEMO_SCENE_PRESETS.values()
            if any(o.seed == seed for o in p.objects if o.label == "chips_can")
        )
        chips = [o for o in preset.objects if o.label == "chips_can"][0]
        sx, sy, syaw = chips_can_seed_pose_uniform(seed)
        if preset.scene_id == "demo_scene_02":
            # Layout v2 fijo: posición manual dentro de zona reach (no muestreador).
            assert chips.seed == seed
            assert math.isclose(chips.yaw, syaw, abs_tol=1e-3)
            allowed, _ = is_chips_can_demo_xy_allowed(chips.x, chips.y)
            assert allowed
            continue
        assert math.isclose(chips.x, sx, abs_tol=0.03)
        assert math.isclose(chips.y, sy, abs_tol=0.03)
        assert math.isclose(chips.yaw, syaw, abs_tol=1e-3)
        allowed, _ = is_chips_can_demo_xy_allowed(chips.x, chips.y)
        assert allowed


def test_demo_objects_have_varied_yaw() -> None:
    for preset in DEMO_SCENE_PRESETS.values():
        if preset.scene_id == "demo_scene_02":
            continue
        if is_two_boxes_scene_preset(preset.scene_id):
            continue
        if preset.scene_id == "chips_mustard_02":
            continue
        yaws = [o.yaw for o in preset.objects]
        spread = min(
            abs((yaws[i] - yaws[j] + math.pi) % (2 * math.pi) - math.pi)
            for i in range(len(yaws))
            for j in range(i + 1, len(yaws))
        )
        assert spread > 0.8


def test_all_objects_fully_on_table() -> None:
    for preset in DEMO_SCENE_PRESETS.values():
        for obj in preset.objects:
            ok, reason = object_footprint_fits_working_table(obj)
            assert ok, f"{preset.scene_id}/{obj.label}: {reason}"


def test_banned_seed_helper() -> None:
    assert is_chips_can_banned_seed(1001)
    assert not is_chips_can_banned_seed(1002)


def test_demo_scene_02_v3_clear_table_layout() -> None:
    """Layout v3: zona reach, orden pick, separación mínima y chips alejado de cracker."""
    preset = get_demo_scene_preset("demo_scene_02")
    ok, reason = validate_demo_scene_02_clear_table_layout(preset)
    assert ok, reason
    cracker = next(o for o in preset.objects if o.label == "cracker_box")
    chips = next(o for o in preset.objects if o.label == "chips_can")
    sugar = next(o for o in preset.objects if o.label == "sugar_box")
    mustard = next(o for o in preset.objects if o.label == "mustard_bottle")
    assert math.isclose(cracker.x, 0.4550, abs_tol=1e-4)
    assert math.isclose(cracker.y, 0.1150, abs_tol=1e-4)
    assert math.isclose(cracker.yaw, 2.9155, abs_tol=1e-3)
    assert cracker.order_index == 0
    assert chips.seed == 1003
    assert math.isclose(chips.x, 0.5200, abs_tol=1e-4)
    assert math.isclose(chips.y, -0.0950, abs_tol=1e-4)
    assert chips.order_index == 1
    assert math.isclose(sugar.x, 0.6300, abs_tol=1e-4)
    assert math.isclose(sugar.y, -0.1750, abs_tol=1e-4)
    assert math.isclose(sugar.yaw, -3.0159, abs_tol=1e-3)
    assert sugar.order_index == 2
    assert math.isclose(mustard.x, 0.6600, abs_tol=1e-4)
    assert math.isclose(mustard.y, 0.0600, abs_tol=1e-4)
    assert math.isclose(mustard.yaw, 1.6392, abs_tol=1e-3)
    assert mustard.order_index == 3
    assert sugar.x < 0.705
    assert mustard.x < 0.705
    assert mustard.y - sugar.y > 0.17
    for obj in preset.objects:
        reach_ok, reach_reason = object_in_demo_scene_02_reach_zone(obj)
        assert reach_ok, f"{obj.label}: {reach_reason}"
    _, pairs = validate_demo_scene_layout(
        preset.objects, footprint_safety_scale=DEMO_SCENE_LAYOUT_FOOTPRINT_SCALE
    )
    min_extra_m = min(p.distance_xy - p.required_min_distance for p in pairs)
    assert min_extra_m >= 0.0005, f"min layout margin {min_extra_m*1000:.1f} mm"
    assert list(DEMO_SCENE_02_PICK_ORDER) == [
        "cracker_box",
        "chips_can",
        "sugar_box",
        "mustard_bottle",
    ]


def test_demo_scene_02_v3_cracker_rear_retreat_clearance() -> None:
    """Tras lift, rear_retreat_x_negative debe dejar >=40 mm a chips_can."""
    import math

    from panda_controller.generic_known_scene_carry_planner import (
        attached_obstacle_clearance_3d,
    )

    preset = get_demo_scene_preset("demo_scene_02")
    chips = next(o for o in preset.objects if o.label == "chips_can")
    post_lift_hand = (0.456, 0.115, 0.587)
    retreat_hand = (max(0.30, post_lift_hand[0] - 0.056), post_lift_hand[1], post_lift_hand[2])
    attached_geom = {
        "carried_object_below_hand_m": 0.192,
        "carried_object_radius_xy_m": 0.1046,
        "attached_collision_padding_m": 0.020,
        "dims_lwh": [0.158, 0.060, 0.213],
    }
    obs = {
        "label": "chips_can",
        "entity_name": "runtime_ycb_chips_can_1",
        "position": (chips.x, chips.y, 0.47),
        "collision_dims": {"shape": "cylinder", "cylinder": [0.033, 0.19]},
        "top_z_m": 0.520,
    }
    for hand in (post_lift_hand, retreat_hand):
        chk = attached_obstacle_clearance_3d(
            hand,
            attached_geom,
            obs,
            table_top_z=0.270,
            required_xy_clearance_m=0.04,
        )
        assert float(chk["xy_clearance"]) >= 0.04, (
            "hand=%s xy_clearance=%.4f" % (hand, float(chk["xy_clearance"]))
        )


def test_demo_scene_01_sugar_pose_on_table() -> None:
    sugar = next(
        o for o in get_demo_scene_preset("demo_scene_01").objects if o.label == "sugar_box"
    )
    ok, reason = object_footprint_fits_working_table(sugar)
    assert ok, reason
    assert sugar.x == 0.43
    assert sugar.y == -0.145
    assert abs(sugar.yaw - 0.85) < 1e-6


def test_demo_scene_vision_labels_log_ok_and_fail() -> None:
    scene_objects = [{"label": lb, "role": "obstacle"} for lb in DEMO_SCENE_OBJECT_LABELS]
    assert is_consolidated_demo_scene_objects(scene_objects)
    runtime = runtime_labels_from_scene_objects(scene_objects)
    assert runtime == list(DEMO_SCENE_OBJECT_LABELS)
    ok, missing = log_demo_scene_vision_labels(
        None,
        scene_preset="demo_scene_01",
        runtime_labels=runtime,
        vision_labels=["cracker_box", "sugar_box", "mustard_bottle", "chips_can"],
    )
    assert ok and missing == []
    ok2, missing2 = log_demo_scene_vision_labels(
        None,
        scene_preset="demo_scene_01",
        runtime_labels=runtime,
        vision_labels=["cracker_box", "mustard_bottle", "chips_can"],
    )
    assert not ok2 and missing2 == ["sugar_box"]
