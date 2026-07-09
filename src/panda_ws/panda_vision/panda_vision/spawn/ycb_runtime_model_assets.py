"""Copias runtime de modelos YCB: textura única + URIs file:// (sin tocar IDs Collada)."""

from __future__ import annotations

import re
import shutil
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from panda_vision.spawn.ycb_visual_normalization import (
    extract_sdf_collision_visual_geometry,
    get_visual_normalization_entry,
    log_ycb_model_sdf_geometry,
    normalize_runtime_sdf_visual_to_collision_box,
    should_normalize_visual_for_label,
)

DEFAULT_RUNTIME_MODELS_ROOT = Path("/tmp/tfg_runtime_ycb_models")

_TEXTURE_MAP_INIT_FROM = "<init_from>texture_map.png</init_from>"
_MALFORMED_REF_RE = re.compile(r'(?:url|target)="[^"]*_\#[^"]*"', re.IGNORECASE)
_PREFIXED_INTERNAL_ID_RE = re.compile(
    r'\b(id|url|target)="[a-z0-9_]+_[0-9a-f]{6,}_(?:Material|shape0|texture_map)',
    re.IGNORECASE,
)


def _normalize_label(label: str) -> str:
    return str(label).strip().lower().replace(" ", "_")


def find_texture_map_png(model_root: Path) -> Optional[Path]:
    for rel in (
        Path("meshes") / "texture_map.png",
        Path("materials") / "textures" / "texture_map.png",
    ):
        candidate = model_root / rel
        if candidate.is_file():
            return candidate.resolve()
    return None


def _extract_visual_mesh_uri(sdf_text: str) -> Optional[str]:
    m = re.search(r"<uri>\s*([^<]+?)\s*</uri>", sdf_text, flags=re.IGNORECASE)
    return m.group(1).strip() if m else None


def _collect_dae_material_info(dae_text: str) -> Tuple[List[str], List[str], List[str]]:
    material_names: List[str] = []
    texture_paths: List[str] = []
    for m in re.finditer(
        r'<material\s+id="([^"]+)"\s+name="([^"]*)"',
        dae_text,
        flags=re.IGNORECASE,
    ):
        material_names.append(f"{m.group(1)} name={m.group(2)}")
    for m in re.finditer(r"<init_from>\s*([^<]+?)\s*</init_from>", dae_text):
        texture_paths.append(m.group(1).strip())
    image_ids = re.findall(r'<image\s+id="([^"]+)"', dae_text, flags=re.IGNORECASE)
    return material_names, texture_paths, image_ids


def log_ycb_model_asset_check(
    logger: Any,
    *,
    label: str,
    model_path: Path,
) -> Dict[str, Any]:
    """Inspecciona SDF/DAE fuente y emite ``[YCB_MODEL_ASSET_CHECK]``."""
    root = model_path.parent if model_path.name == "model.sdf" else model_path
    sdf_path = root / "model.sdf"
    dae_path = root / "meshes" / "textured.dae"
    info: Dict[str, Any] = {
        "label": _normalize_label(label),
        "model_path": str(root.resolve()),
        "visual_mesh_uri": None,
        "texture_paths": [],
        "material_names": [],
        "image_ids": [],
    }
    if sdf_path.is_file():
        sdf_text = sdf_path.read_text(encoding="utf-8", errors="replace")
        info["visual_mesh_uri"] = _extract_visual_mesh_uri(sdf_text)
    if dae_path.is_file():
        dae_text = dae_path.read_text(encoding="utf-8", errors="replace")
        mats, texs, imgs = _collect_dae_material_info(dae_text)
        info["material_names"] = mats
        info["texture_paths"] = texs
        info["image_ids"] = imgs
    tex_file = find_texture_map_png(root)
    if tex_file is not None:
        info["texture_map_png"] = str(tex_file)
    if logger is not None:
        try:
            logger.info(
                "[YCB_MODEL_ASSET_CHECK] label=%s model_path=%s visual_mesh_uri=%s "
                "texture_paths=%s material_names=%s image_ids=%s texture_map_png=%s"
                % (
                    info["label"],
                    info["model_path"],
                    info.get("visual_mesh_uri"),
                    info.get("texture_paths"),
                    info.get("material_names"),
                    info.get("image_ids"),
                    info.get("texture_map_png", "n/a"),
                )
            )
        except Exception:
            pass
    return info


def patch_dae_texture_only(dae_text: str, unique_tex_filename: str) -> str:
    """Solo cambia ``<init_from>texture_map.png</init_from>`` (referencia de imagen)."""
    count = dae_text.count(_TEXTURE_MAP_INIT_FROM)
    if count != 1:
        raise ValueError(
            "patch_dae_texture_only: se esperaba exactamente 1 "
            f"{_TEXTURE_MAP_INIT_FROM!r}, encontrado {count}"
        )
    return dae_text.replace(
        _TEXTURE_MAP_INIT_FROM,
        f"<init_from>{unique_tex_filename}</init_from>",
        1,
    )


def _rewrite_runtime_sdf_file_uris(
    sdf_text: str,
    *,
    unique_model_name: str,
    runtime_dir: Path,
    source_model_name: str,
) -> str:
    out = sdf_text
    out = re.sub(
        rf'<model\s+name="{re.escape(source_model_name)}"',
        f'<model name="{unique_model_name}"',
        out,
        count=1,
        flags=re.IGNORECASE,
    )
    pattern = re.compile(
        rf'model://{re.escape(source_model_name)}/([^\s<>"\']+)',
        flags=re.IGNORECASE,
    )

    def _uri_repl(m: re.Match[str]) -> str:
        rel = m.group(1)
        return (runtime_dir / rel).resolve().as_uri()

    out = pattern.sub(_uri_repl, out)
    if re.search(r"model://", out, flags=re.IGNORECASE):
        raise ValueError("SDF runtime aún contiene model:// tras reescritura")
    return out


def validate_runtime_assets(
    *,
    label: str,
    uid: str,
    runtime_dir: Path,
    unique_tex_name: str,
    source_dae_text: str,
    logger: Any = None,
    visual_normalized: bool = False,
) -> Tuple[bool, List[str]]:
    """Valida copia runtime; emite ``[YCB_RUNTIME_ASSET_VALIDATE]``."""
    lb = _normalize_label(label)
    errors: List[str] = []
    sdf_path = runtime_dir / "model.sdf"
    dae_path = runtime_dir / "meshes" / "textured.dae"
    tex_path = runtime_dir / "meshes" / unique_tex_name

    checks: Dict[str, bool] = {
        "sdf_exists": sdf_path.is_file(),
        "dae_exists": dae_path.is_file(),
        "texture_exists": tex_path.is_file(),
    }
    if not checks["sdf_exists"]:
        errors.append(f"model.sdf ausente: {sdf_path}")
    if not checks["dae_exists"]:
        errors.append(f"textured.dae ausente: {dae_path}")
    if not checks["texture_exists"]:
        errors.append(f"textura única ausente: {tex_path}")

    dae_init_from_ok = False
    no_model_uri = False
    no_prefixed_ids = True
    no_malformed_refs = True
    dae_ids_unchanged = False
    visual_pose_matches_table = True

    if checks["dae_exists"]:
        dae_text = dae_path.read_text(encoding="utf-8", errors="replace")
        init_froms = re.findall(
            r"<init_from>\s*([^<]+?)\s*</init_from>", dae_text, flags=re.IGNORECASE
        )
        dae_init_from_ok = init_froms.count(unique_tex_name) >= 1 and (
            "texture_map.png" not in init_froms
        )
        if not dae_init_from_ok:
            errors.append(
                f"DAE init_from inválido: init_froms={init_froms} "
                f"esperado al menos uno {unique_tex_name!r} y sin texture_map.png"
            )
        if _MALFORMED_REF_RE.search(dae_text):
            no_malformed_refs = False
            errors.append("DAE contiene referencias mal formadas (p. ej. prefix_#Material)")
        if _PREFIXED_INTERNAL_ID_RE.search(dae_text):
            no_prefixed_ids = False
            errors.append("DAE contiene IDs internas prefijadas (parcheador antiguo)")
        normalized = dae_text.replace(
            f"<init_from>{unique_tex_name}</init_from>",
            _TEXTURE_MAP_INIT_FROM,
            1,
        )
        dae_ids_unchanged = normalized == source_dae_text
        if not dae_ids_unchanged:
            errors.append(
                "DAE modificado más allá del init_from de textura (IDs/refs alterados)"
            )

    if checks["sdf_exists"]:
        sdf_text = sdf_path.read_text(encoding="utf-8", errors="replace")
        no_model_uri = "model://" not in sdf_text.lower()
        if not no_model_uri:
            errors.append("model.sdf contiene model://")
        for uri in re.findall(r"<uri>\s*([^<]+?)\s*</uri>", sdf_text, flags=re.IGNORECASE):
            if not uri.strip().lower().startswith("file://"):
                errors.append(f"URI visual no es file://: {uri.strip()}")
                no_model_uri = False
        if visual_normalized:
            entry = get_visual_normalization_entry(lb)
            parsed = extract_sdf_collision_visual_geometry(sdf_text)
            if entry is None:
                visual_pose_matches_table = False
                errors.append("visual_normalized=true pero label sin tabla de normalización")
            elif parsed.get("original_visual_pose") != entry.normalized_visual_pose:
                visual_pose_matches_table = False
                errors.append(
                    "visual runtime no coincide con normalized_visual_pose: "
                    f"got={parsed.get('original_visual_pose')} "
                    f"expected={entry.normalized_visual_pose}"
                )

    ok = (
        all(checks.values())
        and dae_init_from_ok
        and no_model_uri
        and no_prefixed_ids
        and no_malformed_refs
        and dae_ids_unchanged
        and (visual_pose_matches_table if visual_normalized else True)
    )
    if logger is not None:
        try:
            logger.info(
                "[YCB_RUNTIME_ASSET_VALIDATE] label=%s uid=%s runtime_model_dir=%s "
                "ok=%s sdf_exists=%s dae_exists=%s texture_exists=%s "
                "no_model_uri=%s dae_init_from_unique_texture=%s visual_normalized=%s "
                "dae_init_from_ok=%s no_prefixed_ids=%s "
                "no_malformed_refs=%s dae_ids_unchanged=%s errors=%s"
                % (
                    lb,
                    uid,
                    str(runtime_dir),
                    str(ok).lower(),
                    str(checks["sdf_exists"]).lower(),
                    str(checks["dae_exists"]).lower(),
                    str(checks["texture_exists"]).lower(),
                    str(no_model_uri).lower(),
                    str(dae_init_from_ok).lower(),
                    str(visual_normalized).lower(),
                    str(dae_init_from_ok).lower(),
                    str(no_prefixed_ids).lower(),
                    str(no_malformed_refs).lower(),
                    str(dae_ids_unchanged).lower(),
                    errors if errors else "none",
                )
            )
        except Exception:
            pass
    return ok, errors


def _cleanup_runtime_dir(runtime_dir: Path) -> None:
    if runtime_dir.exists():
        shutil.rmtree(runtime_dir, ignore_errors=True)


def prepare_runtime_spawn_model(
    label: str,
    source_model_dir: Path,
    *,
    runtime_models_root: Path = DEFAULT_RUNTIME_MODELS_ROOT,
    logger: Any = None,
    normalize_visual_to_collision_box: Optional[bool] = None,
) -> Tuple[Path, str, Path]:
    """Copia runtime con textura PNG única; no altera IDs Collada internos.

    Returns:
        (sdf_path, unique_model_name, runtime_model_dir)

    Raises:
        RuntimeError: si el parche o la validación fallan (se borra la copia parcial).
    """
    lb = _normalize_label(label)
    source_model_dir = source_model_dir.resolve()
    source_model_name = source_model_dir.name
    src_sdf = source_model_dir / "model.sdf"
    src_dae = source_model_dir / "meshes" / "textured.dae"
    if not src_sdf.is_file():
        raise FileNotFoundError(f"model.sdf no encontrado: {src_sdf}")
    if not src_dae.is_file():
        raise FileNotFoundError(f"textured.dae no encontrado: {src_dae}")
    tex_src = find_texture_map_png(source_model_dir)
    if tex_src is None:
        raise FileNotFoundError(
            f"texture_map.png no encontrado bajo {source_model_dir}"
        )

    log_ycb_model_asset_check(logger, label=lb, model_path=src_sdf)
    source_sdf_text = src_sdf.read_text(encoding="utf-8", errors="replace")
    log_ycb_model_sdf_geometry(
        logger, label=lb, source_sdf=src_sdf, sdf_text=source_sdf_text
    )
    source_dae_text = src_dae.read_text(encoding="utf-8", errors="replace")

    do_visual_norm = (
        should_normalize_visual_for_label(lb)
        if normalize_visual_to_collision_box is None
        else bool(normalize_visual_to_collision_box)
    )

    uid = uuid.uuid4().hex[:8]
    unique_model_name = f"tfg_{lb}_{uid}"
    unique_tex_name = f"{lb}_{uid}_albedo.png"
    runtime_dir = (runtime_models_root / f"{lb}_{uid}").resolve()
    runtime_models_root.mkdir(parents=True, exist_ok=True)
    if runtime_dir.exists():
        shutil.rmtree(runtime_dir)

    try:
        shutil.copytree(source_model_dir, runtime_dir)

        unique_tex_path = runtime_dir / "meshes" / unique_tex_name
        unique_tex_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(tex_src, unique_tex_path)

        try:
            patched_dae = patch_dae_texture_only(source_dae_text, unique_tex_name)
        except ValueError as exc:
            raise RuntimeError(f"patch_dae_texture_only falló: {exc}") from exc

        (runtime_dir / "meshes" / "textured.dae").write_text(
            patched_dae, encoding="utf-8"
        )

        sdf_text = source_sdf_text
        try:
            sdf_text = _rewrite_runtime_sdf_file_uris(
                sdf_text,
                unique_model_name=unique_model_name,
                runtime_dir=runtime_dir,
                source_model_name=source_model_name,
            )
        except ValueError as exc:
            raise RuntimeError(f"reescritura SDF falló: {exc}") from exc

        out_sdf = runtime_dir / "model.sdf"
        if do_visual_norm:
            try:
                sdf_text, _vn_ok, _old_vis, _new_vis = (
                    normalize_runtime_sdf_visual_to_collision_box(
                        sdf_text,
                        lb,
                        logger=logger,
                        runtime_sdf_path=out_sdf,
                    )
                )
            except ValueError as exc:
                raise RuntimeError(
                    f"normalización visual runtime falló: {exc}"
                ) from exc

        out_sdf.write_text(sdf_text, encoding="utf-8")

        cfg = runtime_dir / "model.config"
        if cfg.is_file():
            cfg_text = cfg.read_text(encoding="utf-8", errors="replace")
            cfg_text = re.sub(
                rf"<name>\s*{re.escape(source_model_name)}\s*</name>",
                f"<name>{unique_model_name}</name>",
                cfg_text,
                count=1,
                flags=re.IGNORECASE,
            )
            cfg.write_text(cfg_text, encoding="utf-8")

        ok, val_errors = validate_runtime_assets(
            label=lb,
            uid=uid,
            runtime_dir=runtime_dir,
            unique_tex_name=unique_tex_name,
            source_dae_text=source_dae_text,
            logger=logger,
            visual_normalized=do_visual_norm,
        )
        if not ok:
            raise RuntimeError(
                "validación runtime falló: " + "; ".join(val_errors)
            )

        if logger is not None:
            try:
                logger.info(
                    "[YCB_RUNTIME_ASSET_COPY] label=%s source=%s runtime_model_dir=%s "
                    "unique_model_name=%s unique_texture=%s patch=texture_only "
                    "normalize_visual_to_collision_box=%s"
                    % (
                        lb,
                        str(source_model_dir),
                        str(runtime_dir),
                        unique_model_name,
                        str(unique_tex_path),
                        str(do_visual_norm).lower(),
                    )
                )
            except Exception:
                pass

        return out_sdf.resolve(), unique_model_name, runtime_dir

    except Exception:
        _cleanup_runtime_dir(runtime_dir)
        raise
