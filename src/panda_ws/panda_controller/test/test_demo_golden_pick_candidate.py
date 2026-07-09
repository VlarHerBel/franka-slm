"""Tests candidato pick golden demo_scene_02 + cracker_box."""

from __future__ import annotations

import math
import os

import yaml

from panda_controller.demo_golden_pick_candidate import (
    GOLDEN_CENTER_XY_TOL_M,
    GOLDEN_FAST_EXECUTE_LABELS,
    GOLDEN_YAW_TOL_DEG,
    PLAN_PREFLIGHT_PICK_ONLY_STATUS,
    apply_golden_plan_targets,
    build_golden_fast_execute_detection_entry,
    build_golden_fast_execute_grasp_valid_entry,
    chips_can_golden_legacy_probe_targets,
    enrich_chips_can_legacy_golden_fields,
    format_chips_can_golden_required_missing_log,
    golden_entry_to_grid_spec,
    golden_fast_execute_label_supported,
    golden_scene_id_compatible,
    load_golden_pick_candidate,
    normalize_golden_pick_candidate,
    resolve_runtime_scene_yaw_rad,
    validate_golden_candidate_compatibility,
    validate_golden_candidate_identity,
)
from panda_controller.demo_profile_loader import (
    load_demo_profile,
    resolve_demo_profile,
    resolve_golden_path_from_profile,
)


def _sample_golden_raw() -> dict:
    return {
        "scene_id": "demo_scene_02",
        "layout_version": "v3_clear_table_transport",
        "target_label": "cracker_box",
        "status": "validated_pick_place",
        "object_pose": {
            "semantic_center_xy": [0.455, 0.115],
            "top_z": 0.4700,
            "yaw_rad": 2.9155,
        },
        "candidate": {
            "candidate_idx": 0,
            "yaw_deg": 77.05,
            "commanded_tcp_yaw_rad": 1.344703673205025,
            "pregrasp_tcp": [0.455, 0.115, 0.5620],
            "grasp_tcp": [0.455, 0.115, 0.4370],
            "lift_tcp": [0.455, 0.115, 0.5870],
            "depth_from_top_m": 0.0330,
            "ik_seed": "pick_workspace_ready",
            "prevalidation_source": "demo_collision_off_final_descend",
            "cartesian_descend_fraction": 1.0,
        },
        "transport": {
            "selected_transport_entry": "vertical_raise_then_rear_retreat",
            "route": [
                "carry_mid_high",
                "turn_back_extended_aligned",
                "box_front_high",
                "box_high",
            ],
        },
        "place": {"slot_index": 0},
    }


def test_normalize_golden_pick_candidate() -> None:
    golden = normalize_golden_pick_candidate(_sample_golden_raw())
    assert golden is not None
    assert golden["scene_id"] == "demo_scene_02"
    assert golden["target_label"] == "cracker_box"
    assert golden["status"] == "validated_pick_place"


def test_golden_compatibility_ok_at_reference_pose() -> None:
    golden = normalize_golden_pick_candidate(_sample_golden_raw())
    assert golden is not None
    ok, reason, details = validate_golden_candidate_compatibility(
        golden,
        scene_id="demo_scene_02",
        layout_version="v3_clear_table_transport",
        target_label="cracker_box",
        runtime_xy=(0.455, 0.115),
        runtime_top_z=0.4700,
        runtime_scene_yaw_rad=2.9155,
        runtime_scene_yaw_source="object_yaw_rad:runtime_gt_spawn_yaw",
        runtime_commanded_tcp_yaw_rad=1.344703673205025,
    )
    assert ok is True
    assert reason == "OK"
    assert details["yaw_compare_mode"] == "scene_yaw_vs_scene_yaw"
    assert details["center_error_m"] <= GOLDEN_CENTER_XY_TOL_M
    assert details["yaw_error_deg"] <= GOLDEN_YAW_TOL_DEG


def test_golden_compatibility_scene_yaw_from_runtime_scene_object() -> None:
    golden = normalize_golden_pick_candidate(_sample_golden_raw())
    assert golden is not None
    candidate = {
        "label": "cracker_box",
        "_runtime_scene_objects": [
            {
                "label": "cracker_box",
                "yaw_source": "runtime_gt_spawn_yaw",
                "yaw_rad": 2.9155,
            }
        ],
        "_base_commanded_tcp_yaw_rad": 1.3447,
        "object_yaw_rad": 1.2370,
        "grasp_yaw_rad": 1.2370,
    }
    scene_yaw, source = resolve_runtime_scene_yaw_rad(candidate, target_label="cracker_box")
    assert scene_yaw is not None
    assert math.isclose(scene_yaw, 2.9155, abs_tol=1e-4)
    assert source.startswith("runtime_scene_object:")
    ok, reason, details = validate_golden_candidate_compatibility(
        golden,
        scene_id="demo_scene_02",
        layout_version="v3_clear_table_transport",
        target_label="cracker_box",
        runtime_xy=(0.455, 0.115),
        runtime_top_z=0.4700,
        runtime_scene_yaw_rad=scene_yaw,
        runtime_scene_yaw_source=source,
        runtime_commanded_tcp_yaw_rad=float(
            candidate["_base_commanded_tcp_yaw_rad"]
        ),
    )
    assert ok is True
    assert reason == "OK"
    assert details["yaw_compare_mode"] == "scene_yaw_vs_scene_yaw"


def test_golden_compatibility_scene_yaw_from_explicit_candidate_field() -> None:
    golden = normalize_golden_pick_candidate(_sample_golden_raw())
    assert golden is not None
    candidate = {
        "label": "cracker_box",
        "object_yaw_rad": 2.9155,
        "yaw_source": "runtime_gt_spawn_yaw",
        "grasp_yaw_rad": 1.2370,
    }
    scene_yaw, source = resolve_runtime_scene_yaw_rad(candidate)
    assert scene_yaw is not None
    assert source == "object_yaw_rad:runtime_gt_spawn_yaw"
    ok, reason, details = validate_golden_candidate_compatibility(
        golden,
        scene_id="demo_scene_02",
        layout_version="v3_clear_table_transport",
        target_label="cracker_box",
        runtime_xy=(0.455, 0.115),
        runtime_top_z=0.4700,
        runtime_scene_yaw_rad=scene_yaw,
        runtime_scene_yaw_source=source,
        runtime_commanded_tcp_yaw_rad=1.344703673205025,
    )
    assert ok is True
    assert details["yaw_compare_mode"] == "scene_yaw_vs_scene_yaw"


def test_resolve_runtime_scene_yaw_ignores_grasp_yaw_without_explicit_source() -> None:
    candidate = {
        "label": "cracker_box",
        "object_yaw_rad": 1.2370,
        "grasp_yaw_rad": 1.2370,
        "yaw": 1.2370,
        "yaw_source": "known_rectangle_fit",
    }
    scene_yaw, source = resolve_runtime_scene_yaw_rad(candidate)
    assert scene_yaw is None
    assert source == "none"


def test_golden_compatibility_commanded_fallback_when_no_scene_yaw() -> None:
    golden = normalize_golden_pick_candidate(_sample_golden_raw())
    assert golden is not None
    candidate = {
        "label": "cracker_box",
        "object_yaw_rad": 1.2370,
        "grasp_yaw_rad": 1.2370,
        "yaw_source": "known_rectangle_fit",
        "_base_commanded_tcp_yaw_rad": 1.344703673205025,
    }
    scene_yaw, source = resolve_runtime_scene_yaw_rad(candidate)
    assert scene_yaw is None
    assert source == "none"
    ok, reason, details = validate_golden_candidate_compatibility(
        golden,
        scene_id="demo_scene_02",
        layout_version="v3_clear_table_transport",
        target_label="cracker_box",
        runtime_xy=(0.455, 0.115),
        runtime_top_z=0.4700,
        runtime_scene_yaw_rad=scene_yaw,
        runtime_scene_yaw_source=source,
        runtime_commanded_tcp_yaw_rad=float(
            candidate["_base_commanded_tcp_yaw_rad"]
        ),
    )
    assert ok is True
    assert reason == "OK"
    assert details["yaw_compare_mode"] == "commanded_tcp_yaw_vs_commanded_tcp_yaw"
    assert details["yaw_error_deg"] <= GOLDEN_YAW_TOL_DEG


def test_golden_compatibility_fail_if_non_explicit_scene_yaw_used() -> None:
    golden = normalize_golden_pick_candidate(_sample_golden_raw())
    assert golden is not None
    ok, reason, details = validate_golden_candidate_compatibility(
        golden,
        scene_id="demo_scene_02",
        layout_version="v3_clear_table_transport",
        target_label="cracker_box",
        runtime_xy=(0.455, 0.115),
        runtime_top_z=0.4700,
        runtime_scene_yaw_rad=1.344703673205025,
        runtime_scene_yaw_source="object_yaw_rad:known_rectangle_fit",
        runtime_commanded_tcp_yaw_rad=1.344703673205025,
    )
    assert ok is False
    assert reason == "yaw_out_of_tolerance"
    assert details["yaw_compare_mode"] == "scene_yaw_vs_scene_yaw"
    assert details["yaw_error_deg"] > 80.0


def test_golden_compatibility_fail_center_xy() -> None:
    golden = normalize_golden_pick_candidate(_sample_golden_raw())
    assert golden is not None
    ok, reason, _ = validate_golden_candidate_compatibility(
        golden,
        scene_id="demo_scene_02",
        layout_version="v3_clear_table_transport",
        target_label="cracker_box",
        runtime_xy=(0.500, 0.115),
        runtime_top_z=0.4700,
        runtime_scene_yaw_rad=2.9155,
        runtime_scene_yaw_source="object_yaw_rad:runtime_gt_spawn_yaw",
        runtime_commanded_tcp_yaw_rad=1.344703673205025,
    )
    assert ok is False
    assert reason == "center_xy_out_of_tolerance"


def test_golden_entry_to_grid_spec() -> None:
    golden = normalize_golden_pick_candidate(_sample_golden_raw())
    assert golden is not None
    spec = golden_entry_to_grid_spec(golden, gripper_physical_yaw_correction_rad=0.0)
    assert spec["source"] == "demo_golden_candidate"
    assert spec["grid_idx"] == 0
    assert math.isclose(spec["pre_plan"][2], 0.5620, abs_tol=1e-4)
    assert math.isclose(spec["gr_plan"][2], 0.4370, abs_tol=1e-4)


def test_load_golden_yaml_from_repo() -> None:
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(
        here,
        "config",
        "demo_candidate_cache",
        "demo_scene_02_cracker_box_golden.yaml",
    )
    golden = load_golden_pick_candidate(path)
    assert golden is not None
    assert golden["layout_version"] == "v3_clear_table_transport"


def test_resolve_demo_profile_cracker_box() -> None:
    profile = resolve_demo_profile("demo_scene_02", "cracker_box")
    assert profile is not None
    assert profile["profile_id"] == "demo_scene_02_cracker_box"
    assert profile["parameters"]["execution_mode"] == "pick_place"
    assert profile["parameters"]["paired_grid_max_runtime_sec"] == 120.0


def test_resolve_demo_profile_target_only() -> None:
    assert resolve_demo_profile("", "cracker_box") is None


def test_chips_can_golden_required_missing_log() -> None:
    log = format_chips_can_golden_required_missing_log(
        {
            "scene_id": "demo_scene_02",
            "target_label": "chips_can",
            "reason": "golden_candidate_file_not_loaded",
        }
    )
    assert "[CHIPS_CAN_GOLDEN_REQUIRED_MISSING]" in log
    assert "target_label=chips_can" in log
    assert "result=FAIL" in log


def test_resolve_golden_path_from_profile() -> None:
    profile = load_demo_profile("demo_scene_02_cracker_box")
    assert profile is not None
    path = resolve_golden_path_from_profile(profile)
    assert path.endswith("demo_scene_02_cracker_box_golden.yaml")


def test_resolve_demo_profile_chips_can() -> None:
    profile = resolve_demo_profile("demo_scene_02", "chips_can")
    assert profile is not None
    assert profile["profile_id"] == "demo_scene_02_chips_can"
    assert profile["parameters"]["chips_can_use_legacy_successful_pick_policy"] is True
    assert profile["parameters"]["place_slot_index"] == 1


def test_resolve_demo_profile_chips_can_3obj_scene() -> None:
    profile = resolve_demo_profile("demo_scene_02_3obj", "chips_can")
    assert profile is not None
    assert profile["scene_id"] == "demo_scene_02_3obj"
    assert profile["parameters"]["chips_can_use_legacy_successful_pick_policy"] is True
    assert "demo_authoritative_scene" not in profile["parameters"]
    path = resolve_golden_path_from_profile(profile)
    assert path.endswith("demo_scene_02_chips_can_golden.yaml")


def test_resolve_demo_profile_cracker_and_sugar_3obj_scene() -> None:
    for label, golden_name in (
        ("cracker_box", "demo_scene_02_cracker_box_golden.yaml"),
        ("sugar_box", "demo_scene_02_sugar_box_golden.yaml"),
    ):
        profile = resolve_demo_profile("demo_scene_02_3obj", label)
        assert profile is not None, label
        assert profile["scene_id"] == "demo_scene_02_3obj"
        path = resolve_golden_path_from_profile(profile)
        assert path.endswith(golden_name), label


def test_golden_scene_id_compatible_3obj_runtime() -> None:
    assert golden_scene_id_compatible("demo_scene_02", "demo_scene_02_3obj")
    assert golden_scene_id_compatible("plan_certified_layout_01", "demo_scene_02_3obj")


def test_default_golden_candidate_path_3obj_uses_parent_layout() -> None:
    from panda_controller.demo_golden_pick_candidate import default_golden_candidate_path

    path = default_golden_candidate_path("demo_scene_02_3obj", "cracker_box")
    assert path.endswith("demo_scene_02_cracker_box_golden.yaml")


def test_load_chips_can_golden_yaml_from_repo() -> None:
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(
        here,
        "config",
        "demo_candidate_cache",
        "demo_scene_02_chips_can_golden.yaml",
    )
    golden = load_golden_pick_candidate(path)
    assert golden is not None
    assert golden["target_label"] == "chips_can"
    assert golden["status"] == "validated_pick_place"
    legacy = golden.get("legacy") or {}
    assert legacy.get("contract") == "OK_CHIPS_LEGACY_PENDING_ACTUAL_TF_DESCEND"
    assert float(legacy.get("pregrasp_height_above_top_m")) == 0.035


def test_chips_can_golden_legacy_probe_targets() -> None:
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(
        here,
        "config",
        "demo_candidate_cache",
        "demo_scene_02_chips_can_golden.yaml",
    )
    golden = load_golden_pick_candidate(path)
    assert golden is not None
    targets = chips_can_golden_legacy_probe_targets(
        golden,
        grasp_xy=(0.520, -0.095),
        top_z_m=0.510,
    )
    assert targets is not None
    assert math.isclose(float(targets["pre_plan"][2]), 0.545, abs_tol=1e-4)
    assert math.isclose(float(targets["gr_plan"][2]), 0.475, abs_tol=1e-4)
    assert float(targets["object_high_to_low_fraction"]) >= 0.95


def test_chips_can_golden_compatibility_at_reference_pose() -> None:
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(
        here,
        "config",
        "demo_candidate_cache",
        "demo_scene_02_chips_can_golden.yaml",
    )
    golden = load_golden_pick_candidate(path)
    assert golden is not None
    ok, reason, details = validate_golden_candidate_compatibility(
        golden,
        scene_id="demo_scene_02",
        layout_version="v3_clear_table_transport",
        target_label="chips_can",
        runtime_xy=(0.520, -0.095),
        runtime_top_z=0.510,
        runtime_scene_yaw_rad=1.3953,
        runtime_scene_yaw_source="object_yaw_rad:runtime_gt_spawn_yaw",
        runtime_commanded_tcp_yaw_rad=3.141592653589793,
    )
    assert ok is True
    assert reason == "OK"
    assert details["center_error_m"] <= GOLDEN_CENTER_XY_TOL_M


def test_plan_preflight_golden_accepts_demo_scene_alias() -> None:
    golden = {
        "scene_id": "plan_certified_layout_01",
        "layout_version": "v3_clear_table_transport",
        "target_label": "cracker_box",
        "status": PLAN_PREFLIGHT_PICK_ONLY_STATUS,
    }
    assert golden_scene_id_compatible("plan_certified_layout_01", "demo_scene_02")
    ok, reason = validate_golden_candidate_identity(
        golden,
        scene_id="demo_scene_02",
        layout_version="v3_clear_table_transport",
        target_label="cracker_box",
    )
    assert ok is True
    assert reason == "OK"


def test_plan_preflight_chips_golden_enriched_for_legacy_runtime() -> None:
    raw = {
        "scene_id": "plan_certified_layout_01",
        "layout_version": "v3_clear_table_transport",
        "target_label": "chips_can",
        "status": PLAN_PREFLIGHT_PICK_ONLY_STATUS,
        "object_pose": {
            "semantic_center_xy": [0.52, -0.095],
            "top_z": 0.52,
            "yaw_rad": 1.3953,
        },
        "candidate": {
            "commanded_tcp_yaw_rad": 1.3953,
            "pregrasp_tcp": [0.52, -0.095, 0.64],
            "grasp_tcp": [0.52, -0.095, 0.485],
            "cartesian_descend_fraction": 1.0,
        },
    }
    golden = enrich_chips_can_legacy_golden_fields(
        raw,
        grasp_xy=(0.52, -0.095),
    )
    assert golden["object_pose"]["top_z"] == 0.5100
    assert golden["legacy"]["depth_from_top_m"] == 0.035
    targets = chips_can_golden_legacy_probe_targets(
        golden,
        grasp_xy=(0.52, -0.095),
        top_z_m=0.510,
    )
    assert targets is not None
    ok, reason, _details = validate_golden_candidate_compatibility(
        golden,
        scene_id="demo_scene_02",
        layout_version="v3_clear_table_transport",
        target_label="chips_can",
        runtime_xy=(0.52, -0.095),
        runtime_top_z=0.510,
        runtime_scene_yaw_rad=1.3953,
        runtime_scene_yaw_source="runtime_gt_spawn_yaw",
        runtime_commanded_tcp_yaw_rad=1.3953,
    )
    assert ok is True
    assert reason == "OK"


def test_sugar_and_mustard_demo_profiles_resolve() -> None:
    for label in ("sugar_box", "mustard_bottle"):
        profile = resolve_demo_profile("demo_scene_02", label)
        assert profile is not None
        assert profile["target_label"] == label


def test_mustard_operational_profile_chips_mustard() -> None:
    profile = resolve_demo_profile("chips_mustard_01", "mustard_bottle")
    assert profile is not None
    assert profile["scene_id"] == "chips_mustard_01"
    params = profile.get("parameters") or {}
    assert params.get("mustard_close_joint_m") == 0.018
    assert params.get("mustard_extra_micro_descend_after_cartesian_m") == 0.022
    assert "mustard_pregrasp_ik_joint_goal" not in params
    assert "demo_authoritative_scene" not in params
    assert "scene_id" not in params


def test_golden_fast_execute_labels() -> None:
    assert golden_fast_execute_label_supported("cracker_box")
    assert golden_fast_execute_label_supported("mustard_bottle")
    assert not golden_fast_execute_label_supported("tomato_soup_can")
    assert len(GOLDEN_FAST_EXECUTE_LABELS) == 4


def test_apply_golden_plan_targets() -> None:
    golden = normalize_golden_pick_candidate(_sample_golden_raw())
    assert golden is not None
    plan_targets = {"pregrasp_tcp": (0.0, 0.0, 0.0), "grasp_tcp": (0.0, 0.0, 0.0)}
    apply_golden_plan_targets(plan_targets, golden)
    assert math.isclose(plan_targets["pregrasp_tcp"][2], 0.5620, abs_tol=1e-4)
    assert math.isclose(plan_targets["grasp_tcp"][2], 0.4370, abs_tol=1e-4)
    assert math.isclose(plan_targets["lift_tcp"][2], 0.5870, abs_tol=1e-4)


def test_build_golden_fast_execute_grasp_valid_entry() -> None:
    golden = normalize_golden_pick_candidate(_sample_golden_raw())
    assert golden is not None
    entry = build_golden_fast_execute_grasp_valid_entry(
        golden,
        gripper_physical_yaw_correction_rad=0.0,
    )
    assert entry["variant_name"] == "demo_golden_fast_execute"
    assert entry["_lift_ok"] is True
    assert entry["_demo_golden_fast_execute"] is True
    assert math.isclose(entry["_cart_frac"], 1.0, abs_tol=1e-6)


def test_build_golden_fast_execute_detection_entry() -> None:
    golden = normalize_golden_pick_candidate(_sample_golden_raw())
    assert golden is not None
    det = build_golden_fast_execute_detection_entry(
        golden,
        target_label="cracker_box",
        entity_name="runtime_ycb_cracker_1",
    )
    assert det is not None
    assert det["label"] == "cracker_box"
    assert det["_golden_fast_execute_detection_fallback"] is True
    assert math.isclose(det["top_z_m"], 0.47, abs_tol=1e-3)
    assert det["entity_name"] == "runtime_ycb_cracker_1"


def test_apply_golden_runtime_overrides_preserves_locked_user_place_slot() -> None:
    from panda_controller.demo_golden_pick_candidate import apply_golden_runtime_overrides

    golden = normalize_golden_pick_candidate(_sample_golden_raw())
    assert golden is not None
    golden["place"] = {"slot_index": 1, "slot_name": "slot_2"}
    candidate = {
        "label": "chips_can",
        "place_slot_user_specified": True,
        "place_slot_index": 3,
        "place_slot_name": "slot_4",
        "_place_slot_request_locked": True,
    }
    apply_golden_runtime_overrides(candidate, golden)
    assert candidate["_golden_place_slot_index"] == 1
    assert candidate["place_slot_index"] == 3
    assert candidate["place_slot_name"] == "slot_4"


def test_apply_golden_runtime_overrides_sets_place_slot_when_not_locked() -> None:
    from panda_controller.demo_golden_pick_candidate import apply_golden_runtime_overrides

    golden = normalize_golden_pick_candidate(_sample_golden_raw())
    assert golden is not None
    golden["place"] = {"slot_index": 1, "slot_name": "slot_2"}
    candidate = {"label": "chips_can"}
    apply_golden_runtime_overrides(candidate, golden)
    assert candidate["place_slot_index"] == 1
    assert candidate["place_slot_name"] == "slot_2"
    assert candidate["place_slot_user_specified"] is True


def test_runtime_uses_demo_golden_for_scene_01_and_03() -> None:
    from panda_controller.demo_golden_pick_candidate import runtime_uses_demo_scene_02_golden

    assert runtime_uses_demo_scene_02_golden("demo_scene_01")
    assert runtime_uses_demo_scene_02_golden("demo_scene_02")
    assert runtime_uses_demo_scene_02_golden("demo_scene_03")
    assert runtime_uses_demo_scene_02_golden("demo_scene_02_3obj")
    assert runtime_uses_demo_scene_02_golden("demo_scene_02_3obj_nogolden")
    assert runtime_uses_demo_scene_02_golden("deposit_02_cracker_chips")
    assert runtime_uses_demo_scene_02_golden("deposit_03_mustard_only")
    assert not runtime_uses_demo_scene_02_golden("deposit_full_1table")


def test_default_golden_candidate_path_deposit_02_maps_to_scene_02() -> None:
    from panda_controller.demo_golden_pick_candidate import default_golden_candidate_path

    path = default_golden_candidate_path(
        "deposit_02_cracker_chips", "mustard_bottle"
    )
    assert path.endswith("demo_scene_02_mustard_bottle_golden.yaml")


def test_default_golden_candidate_path_scene_01_falls_back_to_scene_02() -> None:
    from panda_controller.demo_golden_pick_candidate import default_golden_candidate_path

    path = default_golden_candidate_path("demo_scene_01", "cracker_box")
    assert path.endswith("demo_scene_02_cracker_box_golden.yaml")


def test_prepare_demo_golden_pose_adaptive_for_scene_01_cracker() -> None:
    from panda_controller.demo_golden_pick_candidate import (
        prepare_demo_golden_for_runtime,
    )

    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(
        here,
        "config",
        "demo_candidate_cache",
        "demo_scene_02_cracker_box_golden.yaml",
    )
    golden = load_golden_pick_candidate(path)
    assert golden is not None
    adapted, ok, reason, details = prepare_demo_golden_for_runtime(
        golden,
        scene_id="demo_scene_01",
        layout_version="",
        target_label="cracker_box",
        runtime_xy=(0.470, 0.132),
        runtime_top_z=0.470,
        runtime_scene_yaw_rad=-0.6275,
        runtime_scene_yaw_source="runtime_gt_spawn_yaw",
        runtime_commanded_tcp_yaw_rad=-2.198,
    )
    assert ok is True
    assert reason == "OK_pose_adaptive"
    assert adapted is not None
    assert adapted.get("_pose_adaptive_from_demo_scene_02") is True
    cand = adapted.get("candidate") or {}
    pre = cand.get("pregrasp_tcp")
    assert pre is not None
    assert math.isclose(float(pre[0]), 0.470, abs_tol=1e-3)
    assert math.isclose(float(pre[1]), 0.132, abs_tol=1e-3)
    assert details.get("pose_adaptive") is True


def test_prepare_demo_golden_pose_adaptive_mustard_top_z_deposit() -> None:
    from panda_controller.demo_golden_pick_candidate import (
        prepare_demo_golden_for_runtime,
    )

    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(
        here,
        "config",
        "demo_candidate_cache",
        "demo_scene_02_mustard_bottle_golden.yaml",
    )
    golden = load_golden_pick_candidate(path)
    assert golden is not None
    adapted, ok, reason, details = prepare_demo_golden_for_runtime(
        golden,
        scene_id="deposit_03_mustard_only",
        layout_version="",
        target_label="mustard_bottle",
        runtime_xy=(0.664, 0.108),
        runtime_top_z=0.4609,
        runtime_scene_yaw_rad=-3.0732,
        runtime_scene_yaw_source="runtime_gt_spawn_yaw",
        runtime_commanded_tcp_yaw_rad=-3.0732,
    )
    assert ok is True
    assert reason == "OK_pose_adaptive"
    cand = (adapted or {}).get("candidate") or {}
    gr = cand.get("grasp_tcp")
    assert gr is not None
    assert math.isclose(float(gr[2]), 0.4149, abs_tol=1e-3)
    assert details.get("pose_adaptive_reason") == "top_z_out_of_tolerance"

