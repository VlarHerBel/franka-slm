"""Regresión: perception_to_pregrasp_test.py debe compilar sin errores de sintaxis."""

from __future__ import annotations

import py_compile
from pathlib import Path


def test_perception_to_pregrasp_test_py_compiles() -> None:
    module_path = (
        Path(__file__).resolve().parents[1]
        / "panda_controller"
        / "perception_to_pregrasp_test.py"
    )
    assert module_path.is_file(), "missing %s" % module_path
    py_compile.compile(str(module_path), doraise=True)
