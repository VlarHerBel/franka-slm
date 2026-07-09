#!/usr/bin/env python3
"""Imprime contrato geométrico YCB (colisión, visual, RuntimeScene, grasp)."""

from __future__ import annotations

import argparse
import importlib.util
import sys
import types
from pathlib import Path
from typing import Any, List, Optional

from panda_vision.spawn.ycb_visual_normalization import (
    YCB_VISUAL_NORMALIZATION,
    _normalize_label,
    extract_sdf_collision_visual_geometry,
    get_visual_normalization_entry,
    model_origin_to_semantic_center_offset,
)


def _bootstrap_spawn_geometry_modules() -> Any:
    """Carga GT + grasp policy sin ``grasp/__init__.py`` (evita rclpy)."""
    pkg_root = Path(__file__).resolve().parents[1]
    policy_path = pkg_root / "grasp" / "object_grasp_policy.py"
    spec = importlib.util.spec_from_file_location(
        "panda_vision.grasp.object_grasp_policy", policy_path
    )
    ogp = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    grasp_pkg = types.ModuleType("panda_vision.grasp")
    sys.modules.setdefault("panda_vision", types.ModuleType("panda_vision"))
    sys.modules["panda_vision.grasp"] = grasp_pkg
    sys.modules["panda_vision.grasp.object_grasp_policy"] = ogp
    spec.loader.exec_module(ogp)
    grasp_pkg.object_grasp_policy = ogp

    rsg_path = pkg_root / "spawn" / "runtime_scene_gt_geometry.py"
    rsg_spec = importlib.util.spec_from_file_location(
        "panda_vision.spawn.runtime_scene_gt_geometry", rsg_path
    )
    rsg = importlib.util.module_from_spec(rsg_spec)
    assert rsg_spec.loader is not None
    sys.modules["panda_vision.spawn.runtime_scene_gt_geometry"] = rsg
    rsg_spec.loader.exec_module(rsg)
    return rsg


def _default_models_root() -> Path:
    return Path.home() / "tfg_robotics_ws" / "src" / "gazebo_ycb" / "models"


def _grasp_mode(label: str, get_grasp_policy) -> str:
    policy = get_grasp_policy(label)
    primary = str(policy.get("primary_strategy", "")).strip().lower()
    if "top_down" in primary:
        return "top_down_graspable"
    if primary in ("edge_grasp", "oblique_short_axis", "oblique_edge_grasp"):
        return "special_grasp"
    return "obstacle_or_unknown"


def _print_box_report(
    label: str,
    models_root: Path,
    *,
    get_known_box_gt_spec,
    get_grasp_policy,
    resolve_top_face_dims_lwh=None,
) -> None:
    lb = _normalize_label(label)
    spec = get_known_box_gt_spec(lb)
    vis_entry = get_visual_normalization_entry(lb)
    sdf_path = models_root / lb / "model.sdf"
    parsed = {}
    if sdf_path.is_file():
        parsed = extract_sdf_collision_visual_geometry(
            sdf_path.read_text(encoding="utf-8", errors="replace")
        )

    print("=" * 72)
    print(f"label={lb}")
    print(f"source_sdf={sdf_path.resolve()}")
    print(f"sdf_exists={sdf_path.is_file()}")

    if parsed.get("collision_size"):
        print(f"collision_size (SDF)={parsed['collision_size']}")
    if parsed.get("collision_pose"):
        print(f"collision_pose (SDF)={list(parsed['collision_pose'])}")
    if parsed.get("original_visual_pose"):
        print(f"visual_pose_original (SDF)={list(parsed['original_visual_pose'])}")

    if vis_entry is not None:
        print(f"expected_collision_size (table)={list(vis_entry.expected_collision_size)}")
        print(f"collision_pose (table)={list(vis_entry.collision_pose)}")
        print(f"visual_pose_original (table)={list(vis_entry.original_visual_pose)}")
        print(f"visual_pose_runtime_normalized={list(vis_entry.normalized_visual_pose)}")
        print(
            "visual_follows_operational_rule="
            "mesh frame at link base (0,0,0); cuboide operativo = collision box"
        )
        print(f"visual_normalization_notes={vis_entry.notes}")

    if spec is not None:
        l_m, w_m, h_m = spec.dims_lwh_m
        off = spec.model_origin_to_geometry_center_offset_xyz
        print(f"KnownBoxGtSpec.dims_xyz_m={list(spec.dims_xyz_m)}")
        print(f"KnownBoxGtSpec.dims_lwh_m=({l_m}, {w_m}, {h_m})")
        print(
            "model_origin_to_geometry_center_offset_xyz="
            f"{list(off)}"
        )
        print(
            "semantic_center_rule="
            "gazebo_model_origin + Rz(yaw) * offset  "
            f"(offset from collision: {list(off)})"
        )
        print(
            "top_face_rule="
            "semantic_center + [0, 0, H/2]  "
            f"(H={spec.height_m})"
        )
        print(f"local_length_axis={spec.local_length_axis}")
        print(f"local_width_axis={spec.local_width_axis}")
        print(f"yaw_offset_rad={spec.yaw_offset_rad}")
        print(f"overlay_top_face_inset_length_m={spec.overlay_top_face_inset_length_m}")
        print(f"overlay_top_face_inset_width_m={spec.overlay_top_face_inset_width_m}")
        if resolve_top_face_dims_lwh is not None:
            ol, ow, oh, src = resolve_top_face_dims_lwh(spec, for_overlay=True)
            print(
                "overlay_dims_lwh=(%.4f,%.4f,%.4f) source=%s"
                % (ol, ow, oh, src)
            )
        sem_off = model_origin_to_semantic_center_offset(spec.height_m)
        print(f"semantic_offset_from_base_center={list(sem_off)}")
    else:
        print("KnownBoxGtSpec=none")

    policy = get_grasp_policy(lb)
    req_w = policy.get("required_width") or policy.get("required_grasp_width_m")
    print(f"required_gripper_width={req_w}")
    print(f"grasp_primary_strategy={policy.get('primary_strategy')}")
    print(f"grasp_mode={_grasp_mode(lb, get_grasp_policy)}")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Valida geometría YCB (colisión vs visual vs RuntimeScene)."
    )
    parser.add_argument(
        "--models-root",
        type=Path,
        default=_default_models_root(),
        help="Directorio gazebo_ycb/models",
    )
    parser.add_argument(
        "--label",
        type=str,
        default="",
        help="Solo esta etiqueta (vacío = las cuatro cajas conocidas)",
    )
    args = parser.parse_args(argv)

    labels = sorted(YCB_VISUAL_NORMALIZATION.keys())
    if args.label.strip():
        lb = _normalize_label(args.label)
        if lb not in YCB_VISUAL_NORMALIZATION:
            print(f"label desconocida: {lb}", file=sys.stderr)
            return 1
        labels = [lb]

    root = Path(args.models_root).expanduser().resolve()
    if not root.is_dir():
        print(f"models-root no existe: {root}", file=sys.stderr)
        return 1

    rsg = _bootstrap_spawn_geometry_modules()
    ogp = sys.modules["panda_vision.grasp.object_grasp_policy"]
    for label in labels:
        _print_box_report(
            label,
            root,
            get_known_box_gt_spec=rsg.get_known_box_gt_spec,
            get_grasp_policy=ogp.get_grasp_policy,
            resolve_top_face_dims_lwh=rsg.resolve_top_face_dims_lwh,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
