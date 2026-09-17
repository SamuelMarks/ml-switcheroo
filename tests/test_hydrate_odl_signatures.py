"""Tests for ODL signature hydrator script."""

from pathlib import Path
from typing import Any
from unittest.mock import patch
import pytest
import yaml

import scripts.hydrate_odl_signatures as hydrator


def test_simplify_type() -> None:
  """Test simplify_type function across various annotations."""
  assert hydrator.simplify_type(None) == "Any"
  assert hydrator.simplify_type("") == "Any"
  assert hydrator.simplify_type("torch.Tensor") == "Tensor"
  assert hydrator.simplify_type("jax.Array") == "Tensor"
  assert hydrator.simplify_type("np.ndarray") == "Tensor"
  assert hydrator.simplify_type("int") == "int"
  assert hydrator.simplify_type("float") == "float"
  assert hydrator.simplify_type("bool") == "bool"
  assert hydrator.simplify_type("str") == "str"
  assert hydrator.simplify_type("tuple") == "tuple"
  assert hydrator.simplify_type("list") == "list"
  assert hydrator.simplify_type("dict") == "dict"
  assert hydrator.simplify_type("CustomClass") == "Any"


def test_parse_kind() -> None:
  """Test parse_kind function across various ParameterKind values."""
  assert hydrator.parse_kind(None) == ("positional_or_keyword", False)
  assert hydrator.parse_kind("") == ("positional_or_keyword", False)
  assert hydrator.parse_kind("POSITIONAL_OR_KEYWORD") == ("positional_or_keyword", False)
  assert hydrator.parse_kind("KEYWORD_ONLY") == ("keyword_only", False)
  assert hydrator.parse_kind("POSITIONAL_ONLY") == ("positional_only", False)
  assert hydrator.parse_kind("VAR_POSITIONAL") == ("positional_or_keyword", True)
  assert hydrator.parse_kind("VAR_KEYWORD") == ("keyword_only", True)
  assert hydrator.parse_kind("UNKNOWN_KIND") == ("positional_or_keyword", False)


def test_extract_std_args_from_params() -> None:
  """Test extract_std_args_from_params with filtering, mappings, and defaults."""
  params: list[dict[str, Any]] = [
    {"name": "self", "kind": "POSITIONAL_ONLY"},
    {"name": "cls", "kind": "POSITIONAL_ONLY"},
    {
      "name": "x",
      "kind": "POSITIONAL_OR_KEYWORD",
      "annotation": "torch.Tensor",
      "default": None,
    },
    {
      "name": "dim",
      "kind": "KEYWORD_ONLY",
      "annotation": "int",
      "default": -1,
    },
    {
      "name": "ignored_default",
      "kind": "KEYWORD_ONLY",
      "annotation": "str",
      "default": "inspect._empty",
    },
    {
      "name": "extra_args",
      "kind": "VAR_POSITIONAL",
      "annotation": "Any",
    },
  ]

  arg_map = {"axis": "dim"}
  std_args = hydrator.extract_std_args_from_params(params, arg_map=arg_map)

  assert len(std_args) == 4
  assert std_args[0] == {
    "name": "x",
    "kind": "positional_or_keyword",
    "is_variadic": False,
    "type": "Tensor",
  }
  assert std_args[1] == {
    "name": "axis",
    "kind": "keyword_only",
    "is_variadic": False,
    "type": "int",
    "default": -1,
  }
  assert std_args[2] == {
    "name": "ignored_default",
    "kind": "keyword_only",
    "is_variadic": False,
    "type": "str",
  }
  assert std_args[3] == {
    "name": "extra_args",
    "kind": "positional_or_keyword",
    "is_variadic": True,
    "type": "Any",
  }


def test_hydrate_odl_from_snapshots(tmp_path: Path) -> None:
  """Test hydrate_odl_from_snapshots on empty and existing ODL files.

  Args:
      tmp_path: Pytest temporary directory fixture.
  """
  odl_dir = tmp_path / "odl"
  odl_dir.mkdir()

  # File 1: Already has rich std_args -> should not be modified
  f1 = odl_dir / "Op1.yaml"
  f1_data = {
    "operation": "Op1",
    "std_args": [{"name": "x", "type": "Tensor"}],
    "variants": {"torch": {"api": "torch.foo"}},
  }
  f1.write_text(yaml.dump(f1_data))

  # File 2: Empty std_args, matching variant in snapshots
  f2 = odl_dir / "Op2.yaml"
  f2_data = {
    "operation": "Op2",
    "std_args": [],
    "variants": {
      "torch": {
        "api": "torch.bar",
        "args": {"input_tensor": "input"},
      }
    },
  }
  f2.write_text(yaml.dump(f2_data))

  # File 3: Invalid YAML -> skipped
  f3 = odl_dir / "Op3.yaml"
  f3.write_text("invalid: [broken yaml")

  # File 4: Non-dict YAML -> skipped
  f4 = odl_dir / "Op4.yaml"
  f4.write_text("- item1\n- item2\n")

  # File 5: Empty std_args, but no matching variant -> not hydrated
  f5 = odl_dir / "Op5.yaml"
  f5_data = {
    "operation": "Op5",
    "std_args": [],
    "variants": {"torch": {"api": "torch.unknown"}},
  }
  f5.write_text(yaml.dump(f5_data))

  # File 6: variants is not a dict -> skipped
  f6 = odl_dir / "Op6.yaml"
  f6.write_text("operation: Op6\nstd_args: []\nvariants: not_a_dict\n")

  # File 7: API with empty params -> extracted_args is empty list -> not hydrated
  f7 = odl_dir / "Op7.yaml"
  f7_data = {
    "operation": "Op7",
    "std_args": [],
    "variants": {"torch": {"api": "torch.empty_params"}},
  }
  f7.write_text(yaml.dump(f7_data))

  snapshots = {
    "torch": {
      "torch.bar": {
        "params": [
          {"name": "input", "kind": "POSITIONAL_OR_KEYWORD", "annotation": "Tensor"},
          {"name": "dim", "kind": "KEYWORD_ONLY", "annotation": "int", "default": 0},
        ]
      },
      "torch.empty_params": {
        "params": [],
      },
    }
  }

  # Test dry-run: count returned but file unchanged
  dry_count = hydrator.hydrate_odl_from_snapshots(odl_dir, snapshots, dry_run=True, framework_priority=["torch"])
  assert dry_count == 1
  assert yaml.safe_load(f2.read_text())["std_args"] == []

  # Test real run: file hydrated
  count = hydrator.hydrate_odl_from_snapshots(odl_dir, snapshots, dry_run=False)
  assert count == 1

  updated_f2 = yaml.safe_load(f2.read_text())
  assert len(updated_f2["std_args"]) == 2
  assert updated_f2["std_args"][0]["name"] == "input_tensor"
  assert updated_f2["std_args"][1]["name"] == "dim"
  assert updated_f2["std_args"][1]["default"] == 0


def test_main_cli(tmp_path: Path) -> None:
  """Test main CLI entry point.

  Args:
      tmp_path: Pytest temporary directory fixture.
  """
  odl_dir = tmp_path / "odl"
  odl_dir.mkdir()

  f = odl_dir / "Op.yaml"
  f.write_text(yaml.dump({"operation": "Op", "std_args": [], "variants": {}}))

  with patch("scripts.hydrate_odl_signatures.load_snapshots_multi", return_value={}):
    exit_code = hydrator.main(["--odl-dir", str(odl_dir), "--dry-run"])
    assert exit_code == 0


def test_main_entrypoint(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
  """Test __main__ execution block.

  Args:
      monkeypatch: Pytest monkeypatch fixture.
      tmp_path: Pytest temporary directory fixture.
  """
  import sys
  import runpy

  odl_dir = tmp_path / "odl"
  odl_dir.mkdir()

  repo_str = str(hydrator.REPO_ROOT)
  orig_sys_path = list(sys.path)
  sys.path = [p for p in sys.path if p != repo_str]

  try:
    monkeypatch.setattr(
      "sys.argv",
      ["hydrate_odl_signatures.py", "--odl-dir", str(odl_dir), "--dry-run"],
    )
    with patch("scripts.hydrate_odl_signatures.load_snapshots_multi", return_value={}):
      with pytest.raises(SystemExit) as excinfo:
        runpy.run_module("scripts.hydrate_odl_signatures", run_name="__main__")
      assert excinfo.value.code == 0
  finally:
    sys.path = orig_sys_path


def test_extract_std_args_default_filtering() -> None:
  """Test extract_std_args_from_params default None and empty filtering."""
  params: list[dict[str, Any]] = [
    {"name": "a", "kind": "POSITIONAL_OR_KEYWORD", "default": "None"},
    {"name": "b", "kind": "POSITIONAL_OR_KEYWORD", "default": ""},
  ]
  std_args = hydrator.extract_std_args_from_params(params)
  assert len(std_args) == 2
  assert "default" not in std_args[0]
  assert "default" not in std_args[1]


def test_clean_variant_args() -> None:
  """Test clean_variant_args strips receiver parameters and composite names."""
  args_map = {
    "self": "self",
    "cls": "cls",
    "x, y": "x, y",
    "input": "x",
    "other": "y, z",
  }
  snapshot_params: list[dict[str, Any]] = [
    {"name": "x"},
    {"name": "y, 123invalid"},
    {"name": "invalid identifier"},
    "not-a-dict",  # type: ignore
  ]
  cleaned = hydrator.clean_variant_args(args_map, snapshot_params)
  assert cleaned == {"input": "x"}


def test_hydrate_odl_fix_args(tmp_path: Path) -> None:
  """Test hydrate_odl_from_snapshots with fix_args=True.

  Args:
      tmp_path: Temporary directory fixture.
  """
  odl_dir = tmp_path / "odl_fix"
  odl_dir.mkdir()
  test_file = odl_dir / "TestOp.yaml"
  initial_data = {
    "operation": "TestOp",
    "std_args": [{"name": "x", "type": "Tensor"}],
    "variants": {
      "torch": {
        "api": "torch.foo",
        "args": {"self": "self", "x": "x"},
      },
      "jax": "not-a-dict",
      "mlx": {"api": "mlx.bar"},
      "other_fw": {"api": "other.op", "args": {}},
      "noparams": {"api": "noparams.op", "args": {"x": "x"}},
      "unchanged": {"api": "unchanged.op", "args": {"x": "x"}},
    },
  }
  test_file.write_text(yaml.dump(initial_data))

  snapshots = {
    "torch": {
      "torch.foo": {
        "params": [{"name": "self"}, {"name": "x"}],
      }
    },
    "noparams": {
      "noparams.op": {},
    },
    "unchanged": {
      "unchanged.op": {
        "params": [{"name": "x"}],
      },
    },
  }

  count = hydrator.hydrate_odl_from_snapshots(odl_dir, snapshots, dry_run=False, fix_args=True)
  assert count == 1
  with open(test_file) as f:
    updated = yaml.safe_load(f)
  assert updated["variants"]["torch"]["args"] == {"x": "x"}
