"""Tests política ejecución conservadora demo_scene_02 cracker_box."""

from __future__ import annotations

import unittest

from panda_controller.demo_cracker_box_execution_policy import (
    DEFAULT_XY_CORRECTION_MAX_STEP_M,
    MAX_SANE_GRIPPER_CENTERING_ERROR_XY_M,
    apply_demo_locked_pregrasp_z_floor_to_reachability,
    demo_pregrasp_xy_correction_cartesian_only,
    enforce_demo_locked_pregrasp_plan_targets,
    gripper_centering_correction_coherent,
    gripper_centering_error_sane,
    lock_demo_cracker_pregrasp_on_candidate,
)


class TestDemoCrackerPregraspLock(unittest.TestCase):
    def _locked_candidate(self) -> dict:
        c: dict = {"label": "cracker_box"}
        lock_demo_cracker_pregrasp_on_candidate(
            c,
            pregrasp_tcp=(0.455, 0.115, 0.555),
            object_safe_above_tcp=(0.455, 0.115, 0.620),
        )
        return c

    def test_cannot_lower_pregrasp_from_0555_to_0512(self):
        candidate = self._locked_candidate()
        plan_targets = {
            "pregrasp_tcp": (0.455, 0.115, 0.512),
            "grasp_tcp": (0.455, 0.115, 0.437),
        }
        ok, reason, selected_z, demo_z = enforce_demo_locked_pregrasp_plan_targets(
            candidate, plan_targets
        )
        self.assertFalse(ok)
        self.assertEqual(reason, "pregrasp_lowered_after_demo_policy")
        self.assertAlmostEqual(selected_z, 0.512)
        self.assertAlmostEqual(demo_z, 0.555)

    def test_reachability_z_floor_blocks_lowering_search(self):
        candidate = self._locked_candidate()
        min_z, desired_z, start_z, z_vals, floor = (
            apply_demo_locked_pregrasp_z_floor_to_reachability(
                candidate,
                min_pre_z=0.437 + 0.04,
                desired_pre_z=0.512,
                start_z=0.512,
                z_values=[0.512, 0.522, 0.532],
            )
        )
        self.assertAlmostEqual(floor, 0.555)
        self.assertGreaterEqual(min_z, 0.555)
        self.assertGreaterEqual(desired_z, 0.555)
        self.assertGreaterEqual(start_z, 0.555)
        self.assertTrue(all(z >= 0.555 - 1e-6 for z in z_vals))
        self.assertNotIn(0.512, z_vals)

    def test_post_prelude_locked_pregrasp_restored(self):
        candidate = self._locked_candidate()
        plan_targets = {
            "pregrasp_tcp": (0.455, 0.115, 0.555),
            "grasp_tcp": (0.455, 0.115, 0.437),
        }
        ok, reason, selected_z, demo_z = enforce_demo_locked_pregrasp_plan_targets(
            candidate, plan_targets
        )
        self.assertTrue(ok)
        self.assertEqual(reason, "locked")
        self.assertAlmostEqual(selected_z, 0.555)
        self.assertAlmostEqual(demo_z, 0.555)
        self.assertAlmostEqual(plan_targets["pregrasp_tcp"][2], 0.555)
        self.assertAlmostEqual(candidate["object_safe_above_tcp"][2], 0.620)


class TestGripperCenteringSanity(unittest.TestCase):
    def test_large_error_xy_aborts_sanity(self):
        self.assertFalse(gripper_centering_error_sane(0.4267))
        self.assertTrue(
            gripper_centering_error_sane(MAX_SANE_GRIPPER_CENTERING_ERROR_XY_M)
        )

    def test_incoherent_small_correction_rejected(self):
        ok, reason = gripper_centering_correction_coherent(
            before_error_xy_m=0.4267,
            after_error_xy_m=0.0054,
            step_m=0.0060,
            max_step_m=DEFAULT_XY_CORRECTION_MAX_STEP_M,
        )
        self.assertFalse(ok)
        self.assertEqual(reason, "incoherent_correction_without_fresh_tf")

    def test_incoherent_ok_with_fresh_tf(self):
        ok, reason = gripper_centering_correction_coherent(
            before_error_xy_m=0.4267,
            after_error_xy_m=0.0054,
            step_m=0.0060,
            tf_fresh_validated=True,
        )
        self.assertTrue(ok)
        self.assertEqual(reason, "ok_fresh_tf")


class TestPregraspXyCorrectionPolicy(unittest.TestCase):
    def test_demo_authoritative_cartesian_only(self):
        self.assertTrue(
            demo_pregrasp_xy_correction_cartesian_only(
                demo_authoritative_scene=True,
                scene_id="demo_scene_02",
                label="cracker_box",
            )
        )
        self.assertFalse(
            demo_pregrasp_xy_correction_cartesian_only(
                demo_authoritative_scene=True,
                scene_id="demo_scene_02",
                label="chips_can",
            )
        )


if __name__ == "__main__":
    unittest.main()
