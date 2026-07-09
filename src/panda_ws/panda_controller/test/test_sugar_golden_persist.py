"""Persist sugar golden from successful pick."""

from panda_controller.demo_golden_pick_candidate import (
    build_chips_can_pick_golden_from_success,
    build_mustard_pick_golden_from_success,
    build_sugar_pick_golden_from_success,
    chips_can_pick_golden_persist_eligible,
    golden_fast_execute_available,
    mustard_pick_golden_persist_eligible,
    persist_chips_can_golden_from_success,
    persist_mustard_demo_scene_02_golden_from_success,
    persist_sugar_demo_scene_02_golden_from_success,
    save_golden_pick_candidate,
    sugar_pick_golden_persist_eligible,
)


def test_sugar_golden_persist_eligible_deposit_and_demo_scene_02() -> None:
    assert sugar_pick_golden_persist_eligible("deposit_02_cracker_chips")
    assert sugar_pick_golden_persist_eligible("demo_scene_02")
    assert not sugar_pick_golden_persist_eligible("deposit_full_1table")


def test_build_sugar_golden_from_success_uses_executed_tcps() -> None:
    candidate = {
        "label": "sugar_box",
        "grasp_center_base": [0.630, -0.175, 0.435],
        "top_z_m": 0.435,
        "object_yaw_rad": -3.0159,
        "_base_commanded_tcp_yaw_rad": 1.6965,
        "_pregrasp_tcp_planning": [0.630, -0.175, 0.472],
        "_grasp_tcp_planning": [0.630, -0.175, 0.407],
        "_lift_tcp_planning": [0.630, -0.175, 0.557],
        "_final_release_tcp_z": 0.200,
        "place_slot_index": 2,
    }
    golden = build_sugar_pick_golden_from_success(
        candidate,
        place_payload={"place_slot_index": 2, "x": -0.37, "y": -0.1, "release_tcp_z": 0.2},
    )
    assert golden["candidate"]["pregrasp_tcp"] == [0.630, -0.175, 0.472]
    assert golden["candidate"]["grasp_tcp"] == [0.630, -0.175, 0.407]
    assert golden["place"]["release_tcp_z"] == 0.2


def test_persist_sugar_golden_writes_file(tmp_path) -> None:
    config_dir = str(tmp_path)
    candidate = {
        "label": "sugar_box",
        "grasp_center_base": [0.630, -0.175, 0.435],
        "top_z_m": 0.435,
        "_pregrasp_tcp_planning": [0.630, -0.175, 0.472],
        "_grasp_tcp_planning": [0.630, -0.175, 0.407],
        "_lift_tcp_planning": [0.630, -0.175, 0.557],
    }
    ok, path = persist_sugar_demo_scene_02_golden_from_success(
        candidate,
        scene_id="deposit_02_cracker_chips",
        place_payload={"place_slot_index": 2, "release_tcp_z": 0.2},
        config_dir=config_dir,
    )
    assert ok
    assert path.endswith("demo_scene_02_sugar_box_golden.yaml")


def test_mustard_golden_persist_eligible() -> None:
    assert mustard_pick_golden_persist_eligible("deposit_02_cracker_chips")
    assert mustard_pick_golden_persist_eligible("demo_scene_02")
    assert mustard_pick_golden_persist_eligible("chips_mustard_02")


def test_chips_can_golden_persist_eligible() -> None:
    assert chips_can_pick_golden_persist_eligible("chips_mustard_02")
    assert not chips_can_pick_golden_persist_eligible("demo_scene_02")


def test_golden_fast_execute_requires_scene_local_yaml(tmp_path) -> None:
    assert not golden_fast_execute_available(
        "chips_mustard_02", "chips_can", config_dir=str(tmp_path)
    )
    golden = build_chips_can_pick_golden_from_success(
        {
            "label": "chips_can",
            "grasp_center_base": [0.500, -0.080, 0.510],
            "_pregrasp_tcp_planning": [0.500, -0.080, 0.545],
            "_grasp_tcp_planning": [0.500, -0.080, 0.475],
        },
        scene_id="chips_mustard_02",
    )
    path = str(tmp_path / "demo_candidate_cache/chips_mustard_02_chips_can_golden.yaml")
    assert save_golden_pick_candidate(golden, path)
    assert golden_fast_execute_available(
        "chips_mustard_02", "chips_can", config_dir=str(tmp_path)
    )


def test_persist_chips_can_golden_writes_scene_file(tmp_path) -> None:
    ok, path = persist_chips_can_golden_from_success(
        {
            "label": "chips_can",
            "grasp_center_base": [0.500, -0.080, 0.510],
            "_pregrasp_tcp_planning": [0.500, -0.080, 0.545],
            "_grasp_tcp_planning": [0.500, -0.080, 0.475],
        },
        scene_id="chips_mustard_02",
        place_payload={"place_slot_index": 1, "release_tcp_z": 0.195},
        config_dir=str(tmp_path),
    )
    assert ok
    assert path.endswith("chips_mustard_02_chips_can_golden.yaml")


def test_build_mustard_golden_from_success() -> None:
    candidate = {
        "label": "mustard_bottle",
        "grasp_center_base": [0.664, 0.108, 0.437],
        "top_z_m": 0.437,
        "_base_commanded_tcp_yaw_rad": -3.0732,
        "_pregrasp_tcp_planning": [0.664, 0.108, 0.491],
        "_grasp_tcp_planning": [0.664, 0.108, 0.427],
        "_lift_tcp_planning": [0.664, 0.108, 0.632],
        "_final_release_tcp_z": 0.3284,
        "place_slot_index": 3,
        "grasp_strategy": "tall_object_topdown",
    }
    golden = build_mustard_pick_golden_from_success(
        candidate,
        place_payload={
            "place_slot_index": 3,
            "x": -0.54,
            "y": -0.1,
            "release_tcp_z": 0.3284,
        },
    )
    assert golden["place"]["release_tcp_z"] == 0.3284
    assert golden["place"]["release_source"] == "golden_place"
    assert golden["candidate"]["grasp_strategy"] == "tall_object_topdown"


def test_persist_mustard_golden_writes_file(tmp_path) -> None:
    ok, path = persist_mustard_demo_scene_02_golden_from_success(
        {
            "label": "mustard_bottle",
            "grasp_center_base": [0.664, 0.108, 0.437],
            "_pregrasp_tcp_planning": [0.664, 0.108, 0.491],
            "_grasp_tcp_planning": [0.664, 0.108, 0.427],
        },
        scene_id="deposit_02_cracker_chips",
        place_payload={"place_slot_index": 3, "release_tcp_z": 0.3284},
        config_dir=str(tmp_path),
    )
    assert ok
    assert path.endswith("demo_scene_02_mustard_bottle_golden.yaml")
