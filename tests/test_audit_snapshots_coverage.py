"""Targeted branch coverage tests for scripts/audit_against_snapshots.py."""

import json
from pathlib import Path
from types import ModuleType
from typing import Any, Dict
from unittest.mock import MagicMock, patch
import pytest

from scripts.audit_against_snapshots import (
  _flatten_single_framework,
  _resolve_grounding_engine,
  audit_frameworks,
  load_snapshots,
)


def test_resolve_grounding_engine_fallback_paths(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
  """Test _resolve_grounding_engine fallback through parent repository paths and failure.

  Args:
      monkeypatch: Pytest monkeypatch fixture.
      tmp_path: Temporary directory fixture from pytest.
  """
  # 1. Test fallback when parent_src exists
  mock_engine = MagicMock()
  fake_mod = ModuleType("ml_ecosystem_snapshots.grounding.engine")
  setattr(fake_mod, "GroundingEngine", mock_engine)

  def import_side_effect(name: str) -> Any:
    """Mock import side effect for snapshot packages.

    Args:
        name: Name of package to import.

    Returns:
        Fake module.

    Raises:
        ImportError: If name does not match expected prefixes.
    """
    if name.startswith("ml_ecosystem_snapshots") or name.startswith("ml_framework_snapshots"):
      return fake_mod
    raise ImportError(f"No module named {name}")

  # Force initial import_module calls to fail so it reaches the repo loop
  call_count = 0

  def fail_first_two_imports(name: str, *args: Any, **kwargs: Any) -> Any:
    """Mock failing initial imports to exercise fallback loop.

    Args:
        name: Name of module being imported.
        *args: Variable positional arguments.
        **kwargs: Variable keyword arguments.

    Returns:
        Fake module on third call onwards.

    Raises:
        ImportError: On first two calls.
    """
    nonlocal call_count
    call_count += 1
    if call_count <= 2:
      raise ImportError(name)
    return fake_mod

  with patch("importlib.import_module", side_effect=fail_first_two_imports):
    with patch.object(Path, "exists", return_value=True):
      res = _resolve_grounding_engine()
      assert res == mock_engine

  # 2. Test when all imports fail and it returns None
  with patch("importlib.import_module", side_effect=ImportError("Failed")):
    with patch.object(Path, "exists", return_value=False):
      assert _resolve_grounding_engine() is None


def test_flatten_single_framework_operations_branches() -> None:
  """Test _flatten_single_framework across all operation dictionary variations."""
  flat: Dict[str, Dict[str, Any]] = {"test_fw": {}}
  snap = {
    "operations": [
      # Item with only api_path
      {"api_path": "pkg.op_api"},
      # Item with only class_name
      {"class_name": "pkg.OpClass"},
      # Item with only name
      {"name": "op_name"},
      # Item with aliases as list
      {"name": "multi_alias", "aliases": ["alias_1", "alias_2"]},
      # Item with aliases as non-list
      {"name": "bad_alias", "aliases": "not_a_list"},
      # Item not a dict (skipped)
      "not_a_dict_item",
    ]
  }

  _flatten_single_framework("test_fw", snap, flat)

  assert "pkg.op_api" in flat["test_fw"]
  assert "pkg.OpClass" in flat["test_fw"]
  assert "op_name" in flat["test_fw"]
  assert "alias_1" in flat["test_fw"]
  assert "alias_2" in flat["test_fw"]
  assert "multi_alias" in flat["test_fw"]
  assert "bad_alias" in flat["test_fw"]


def test_load_snapshots_stablehlo_exhaustive(tmp_path: Path) -> None:
  """Test load_snapshots flattens stablehlo_exhaustive into both stablehlo and original keys.

  Args:
      tmp_path: Temporary directory fixture from pytest.
  """
  snap_file = tmp_path / "stablehlo_exhaustive.json"
  snap_content = {
    "operations": [{"name": "custom_shlo_op"}],
  }
  snap_file.write_text(json.dumps(snap_content), encoding="utf-8")

  flat = load_snapshots(tmp_path)
  assert "stablehlo_exhaustive" in flat
  assert "stablehlo" in flat
  assert "custom_shlo_op" in flat["stablehlo"]


def test_audit_frameworks_ir_onnx_spec_exception() -> None:
  """Test audit_frameworks IR path when loading canonical onnx ops raises an exception."""
  mock_mgr = MagicMock()
  mock_mgr.data = {
    "mock_ir_op": {
      "variants": {
        "ir": {
          "api": "Add",
          "args": {},
        }
      }
    }
  }

  snapshots: Dict[str, Dict[str, Any]] = {"ir": {}}

  with patch("ml_ecosystem_snapshots.frameworks.onnx_spec._load_onnx_ops", side_effect=RuntimeError("ONNX err")):
    errors = audit_frameworks(mock_mgr, snapshots, framework="ir")
    assert errors == []
