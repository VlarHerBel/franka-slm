#!/usr/bin/env python3
"""Evalúa yaw activo vs candidato de fit para presets deterministas (solo medición)."""

from __future__ import annotations

import csv
import json
import math
import os
import subprocess
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

PRESETS: List[Tuple[str, float]] = [
    ("center_0deg", 90.0),
    ("center_45deg", 135.0),
    ("center_90deg", 0.0),
    ("center_135deg", 45.0),
]

CSV_FIELDNAMES: List[str] = [
    "preset",
    "error",
    "samples_used",
    "expected_yaw_deg",
    "active_yaw_deg_mod180",
    "active_yaw_error_deg",
    "active_yaw_source",
    "fit_candidate_yaw_deg_mod180",
    "fit_candidate_yaw_error_deg",
    "yaw_fit_method",
    "pose_fit_success",
    "grasp_center_source",
    "yaw_confidence",
    "pose_fit_error",
    "projected_extent_length_m",
    "projected_extent_width_m",
    "length_error_m",
    "width_error_m",
    "inlier_ratio",
    "outside_error_m",
    "top_face_point_ratio",
    "partial_top_face_detected",
    "hybrid_fit_success",
]


def angular_error_deg(yaw_deg: float, expected_deg: float) -> float:
    """Error angular mod 180° entre orientaciones de recta (0..90]."""
    return abs(((float(yaw_deg) - float(expected_deg) + 90.0) % 180.0) - 90.0)


def yaw_mod180_from_rad(rad: Any) -> float:
    return float(rad) * 180.0 / math.pi % 180.0


def fit_candidate_yaw_mod180(obj: Dict[str, Any]) -> float:
    sel = obj.get("selected_yaw_deg")
    if sel is not None:
        try:
            return float(sel) % 180.0
        except (TypeError, ValueError):
            pass
    fit_deg = obj.get("fit_candidate_yaw_deg")
    if fit_deg is not None:
        try:
            return float(fit_deg) % 180.0
        except (TypeError, ValueError):
            pass
    kb = obj.get("known_box_yaw_rad")
    if kb is not None:
        try:
            return yaw_mod180_from_rad(kb)
        except (TypeError, ValueError):
            pass
    return float("nan")


def active_yaw_mod180(obj: Dict[str, Any]) -> float:
    """Yaw que perception_node usa operativamente según yaw_source."""
    ys = str(obj.get("yaw_source", "")).strip().lower()
    if ys in (
        "known_rectangle_fit",
        "hybrid_top_face_known_dims",
        "top_face_pca_known_dims",
    ):
        kb = obj.get("known_box_yaw_rad")
        if kb is not None:
            try:
                return yaw_mod180_from_rad(kb)
            except (TypeError, ValueError):
                pass
    if ys == "pca_raw":
        oy = obj.get("object_yaw_rad")
        if oy is not None:
            try:
                return yaw_mod180_from_rad(oy)
            except (TypeError, ValueError):
                pass
    ay = obj.get("active_yaw_rad")
    if ay is not None:
        try:
            return yaw_mod180_from_rad(ay)
        except (TypeError, ValueError):
            pass
    gy = obj.get("grasp_yaw_rad")
    if gy is not None:
        try:
            return yaw_mod180_from_rad(gy)
        except (TypeError, ValueError):
            pass
    oy = obj.get("object_yaw_rad")
    if oy is not None:
        try:
            return yaw_mod180_from_rad(oy)
        except (TypeError, ValueError):
            pass
    return float("nan")


def circular_mean_deg_mod180(degs: List[float]) -> float:
    valid = [d for d in degs if math.isfinite(d)]
    if not valid:
        return float("nan")
    if len(valid) == 1:
        return float(valid[0]) % 180.0
    rad2 = np.deg2rad(np.asarray(valid, dtype=np.float64) * 2.0)
    c = float(np.mean(np.cos(rad2)))
    s = float(np.mean(np.sin(rad2)))
    return (float(np.degrees(np.arctan2(s, c))) * 0.5) % 180.0


def median_float(vals: List[float]) -> float:
    valid = [v for v in vals if math.isfinite(v)]
    if not valid:
        return float("nan")
    return float(np.median(np.asarray(valid, dtype=np.float64)))


def aggregate_samples(objs: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not objs:
        return {}
    if len(objs) == 1:
        return dict(objs[0])

    act_degs = [active_yaw_mod180(o) for o in objs]
    fit_degs = [fit_candidate_yaw_mod180(o) for o in objs]
    out = dict(objs[-1])
    out["_agg_active_yaw_deg"] = circular_mean_deg_mod180(act_degs)
    out["_agg_fit_yaw_deg"] = circular_mean_deg_mod180(fit_degs)

    for key in (
        "yaw_confidence",
        "pose_fit_error",
        "top_face_point_ratio",
        "inlier_ratio",
        "outside_error_m",
        "projected_extent_length_m",
        "projected_extent_width_m",
        "length_error_m",
        "width_error_m",
    ):
        nums: List[float] = []
        for o in objs:
            v = o.get(key)
            if v is None:
                v = o.get("top_face_ratio") if key == "top_face_point_ratio" else None
            if v is not None:
                try:
                    nums.append(float(v))
                except (TypeError, ValueError):
                    pass
        if nums:
            out[key] = median_float(nums)
    return out


def _row_template() -> Dict[str, str]:
    return {k: "" for k in CSV_FIELDNAMES}


def _fill_row(
    preset: str,
    exp_deg: float,
    agg: Dict[str, Any],
    samples_used: int,
    error_field: str,
) -> Dict[str, str]:
    row = _row_template()
    act_deg = float(agg.get("_agg_active_yaw_deg", active_yaw_mod180(agg)))
    fit_deg = float(agg.get("_agg_fit_yaw_deg", fit_candidate_yaw_mod180(agg)))
    row["preset"] = preset
    row["error"] = error_field
    row["samples_used"] = str(samples_used)
    row["expected_yaw_deg"] = f"{exp_deg:.1f}"
    row["active_yaw_deg_mod180"] = f"{act_deg:.2f}" if math.isfinite(act_deg) else ""
    row["active_yaw_error_deg"] = (
        f"{angular_error_deg(act_deg, exp_deg):.2f}" if math.isfinite(act_deg) else ""
    )
    row["active_yaw_source"] = str(agg.get("yaw_source", ""))
    row["fit_candidate_yaw_deg_mod180"] = f"{fit_deg:.2f}" if math.isfinite(fit_deg) else ""
    row["fit_candidate_yaw_error_deg"] = (
        f"{angular_error_deg(fit_deg, exp_deg):.2f}" if math.isfinite(fit_deg) else ""
    )
    for k in (
        "yaw_fit_method",
        "grasp_center_source",
        "partial_top_face_detected",
        "hybrid_fit_success",
    ):
        v = agg.get(k)
        row[k] = "" if v is None else str(v)
    row["pose_fit_success"] = str(bool(agg.get("pose_fit_success", False))).lower()
    for k in (
        "yaw_confidence",
        "pose_fit_error",
        "projected_extent_length_m",
        "projected_extent_width_m",
        "length_error_m",
        "width_error_m",
        "inlier_ratio",
        "outside_error_m",
    ):
        v = agg.get(k)
        row[k] = "" if v is None else str(float(v))
    tfr = agg.get("top_face_point_ratio", agg.get("top_face_ratio"))
    row["top_face_point_ratio"] = "" if tfr is None else str(float(tfr))
    return row


class MeasureYawTable(Node):
    def __init__(self) -> None:
        super().__init__("measure_ycb_yaw_table")
        self.declare_parameter("vision_topic", "/vision_to_executor")
        self.declare_parameter("ws_install_setup", "")
        self.declare_parameter("label", "cracker_box")
        self.declare_parameter("settle_s", 2.5)
        self.declare_parameter("samples_per_preset", 3)
        self.declare_parameter("timeout_s", 15.0)
        self.declare_parameter("output_csv", "/tmp/cracker_yaw_table.csv")
        self.declare_parameter("require_stamp_after_spawn", False)

        self._topic = str(self.get_parameter("vision_topic").value).strip()
        self._install_setup = str(self.get_parameter("ws_install_setup").value).strip()
        self._label = str(self.get_parameter("label").value).strip().lower()
        self._settle_s = float(self.get_parameter("settle_s").value)
        self._samples_per_preset = max(1, int(self.get_parameter("samples_per_preset").value))
        self._msg_wait = float(self.get_parameter("timeout_s").value)
        self._output_csv = str(self.get_parameter("output_csv").value).strip()
        self._require_stamp = bool(self.get_parameter("require_stamp_after_spawn").value)

        self._last_payload: Optional[Dict[str, Any]] = None
        self._msg_id: int = 0
        self._sub = self.create_subscription(String, self._topic, self._cb, 10)

    def _cb(self, msg: String) -> None:
        try:
            self._last_payload = json.loads(msg.data)
            self._msg_id += 1
        except json.JSONDecodeError:
            self.get_logger().warning("JSON inválido en vision topic")

    def _extract_object(self, pl: Any) -> Optional[Dict[str, Any]]:
        if not isinstance(pl, dict):
            return None
        for obj in pl.get("objects", []) or []:
            if str(obj.get("label", "")).strip().lower() == self._label:
                return obj
        return None

    def drain_subscription(self) -> None:
        self._last_payload = None
        t_end = time.monotonic() + 0.25
        while time.monotonic() < t_end and rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.05)

    def _stamp_ok(self, pl: Dict[str, Any], stamp_floor_sec: Optional[float]) -> bool:
        if not self._require_stamp:
            return True
        if stamp_floor_sec is None:
            return True
        try:
            return float(pl.get("stamp_sec", -1.0)) >= float(stamp_floor_sec) - 1.0
        except (TypeError, ValueError):
            return False

    def collect_samples_after_spawn(
        self,
        preset: str,
        msg_id_before_spawn: int,
        stamp_floor_sec: Optional[float],
    ) -> List[Dict[str, Any]]:
        n = self._samples_per_preset
        samples: List[Dict[str, Any]] = []
        ref_id = int(msg_id_before_spawn)
        deadline = time.monotonic() + max(0.0, self._settle_s) + max(0.0, self._msg_wait)

        while len(samples) < n and time.monotonic() < deadline and rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.05)
            if self._msg_id <= ref_id:
                continue
            pl = self._last_payload
            if not isinstance(pl, dict):
                ref_id = self._msg_id
                continue
            if not self._stamp_ok(pl, stamp_floor_sec):
                ref_id = self._msg_id
                continue
            obj = self._extract_object(pl)
            if obj is None:
                ref_id = self._msg_id
                continue
            mid = int(self._msg_id)
            act = active_yaw_mod180(obj)
            fit = fit_candidate_yaw_mod180(obj)
            self.get_logger().info(
                f"[MEASURE_YAW] accepted detection preset={preset} "
                f"sample={len(samples) + 1}/{n} msg_id={mid} "
                f"stamp_sec={pl.get('stamp_sec', 'n/a')} "
                f"yaw_source={obj.get('yaw_source')} "
                f"active_yaw={act:.1f} fit_yaw={fit:.1f}"
            )
            samples.append(dict(obj))
            ref_id = mid
        return samples


def _bash_env(install_setup: str) -> Dict[str, str]:
    env = os.environ.copy()
    setup = (install_setup or "").strip() or os.path.expanduser(
        "~/tfg_robotics_ws/install/setup.bash"
    )
    if os.path.isfile(setup):
        out = subprocess.run(
            ["bash", "-lc", f"set -e; source {setup} >/dev/null 2>&1; env -0"],
            capture_output=True,
            text=True,
            check=False,
        )
        if out.returncode == 0 and out.stdout:
            for line in out.stdout.split("\0"):
                if "=" in line:
                    k, v = line.split("=", 1)
                    env[k] = v
    return env


def run_ros2(args: List[str], logger: Any, install_setup: str) -> None:
    env = _bash_env(install_setup)
    cmd = ["ros2"] + args
    logger.info(f"[MEASURE_YAW] ejecutando: {' '.join(cmd)}")
    p = subprocess.run(cmd, capture_output=True, text=True, env=env, check=False)
    if p.returncode != 0:
        logger.warning(
            f"[MEASURE_YAW] rc={p.returncode} stderr={(p.stderr or '').strip()[:400]}"
        )


def run_clear_scene(logger: Any, install_setup: str) -> None:
    run_ros2(
        [
            "run",
            "panda_vision",
            "clear_ycb_objects",
            "--ros-args",
            "-p",
            "delete_all_runtime_ycb:=true",
        ],
        logger,
        install_setup,
    )


def main() -> None:
    rclpy.init()
    node = MeasureYawTable()
    rows: List[Dict[str, str]] = []

    for _ in range(5):
        rclpy.spin_once(node, timeout_sec=0.02)

    node.get_logger().info(
        f"[MEASURE_YAW] require_stamp_after_spawn={node._require_stamp}"
    )

    hdr = (
        f"{'Preset':<14} {'active':>7} {'exp':>6} {'a_err':>6} "
        f"{'yaw_src':<22} {'fit_yaw':>7} {'f_err':>6} "
        f"{'center_src':<18} {'ok':>5} {'n':>3}"
    )
    print(hdr)

    for preset, exp_deg in PRESETS:
        node.drain_subscription()
        run_clear_scene(node.get_logger(), node._install_setup)
        time.sleep(0.6)

        msg_id_before = int(node._msg_id)
        node.get_logger().info(f"[MEASURE_YAW] msg_id_before_spawn={msg_id_before}")

        run_ros2(
            [
                "run",
                "panda_vision",
                "spawn_ycb_object",
                "--ros-args",
                "-p",
                f"label:={node._label}",
                "-p",
                f"preset:={preset}",
                "-p",
                "delete_existing:=true",
            ],
            node.get_logger(),
            node._install_setup,
        )

        stamp_floor: Optional[float] = None
        if node._require_stamp:
            stamp_floor = node.get_clock().now().nanoseconds * 1e-9

        samples = node.collect_samples_after_spawn(
            preset, msg_id_before, stamp_floor
        )

        if len(samples) == 0:
            node.get_logger().error(
                f"[MEASURE_YAW] no valid detection after spawn for preset={preset}"
            )
            row = _row_template()
            row["preset"] = preset
            row["error"] = "no_detection"
            row["samples_used"] = "0"
            row["expected_yaw_deg"] = f"{exp_deg:.1f}"
            rows.append(row)
            print(f"{preset:<14} {'—':>7}")
            continue

        if len(samples) < node._samples_per_preset:
            node.get_logger().warning(
                f"[MEASURE_YAW] preset={preset}: {len(samples)}/"
                f"{node._samples_per_preset} muestras"
            )

        agg = aggregate_samples(samples)
        err_field = "partial_samples" if len(samples) < node._samples_per_preset else ""
        row = _fill_row(preset, exp_deg, agg, len(samples), err_field)
        rows.append(row)

        act_deg = float(agg.get("_agg_active_yaw_deg", active_yaw_mod180(agg)))
        fit_deg = float(agg.get("_agg_fit_yaw_deg", fit_candidate_yaw_mod180(agg)))
        a_err = angular_error_deg(act_deg, exp_deg) if math.isfinite(act_deg) else float("nan")
        f_err = angular_error_deg(fit_deg, exp_deg) if math.isfinite(fit_deg) else float("nan")
        print(
            f"{preset:<14} {act_deg:7.1f} {exp_deg:6.1f} {a_err:6.1f} "
            f"{str(agg.get('yaw_source', ''))[:22]:<22} "
            f"{fit_deg:7.1f} {f_err:6.1f} "
            f"{str(agg.get('grasp_center_source', ''))[:18]:<18} "
            f"{str(bool(agg.get('pose_fit_success'))).lower():>5} {len(samples):3d}"
        )

    out_csv = node._output_csv or "/tmp/cracker_yaw_table.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in CSV_FIELDNAMES})

    node.get_logger().info(f"[MEASURE_YAW] CSV guardado en {out_csv}")
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
