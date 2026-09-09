"""Tests for ODL catalog compiler and roundtrip verifier."""

from pathlib import Path
from unittest.mock import patch
import pytest

import scripts.compile_odl_catalog as compiler


def test_compile_catalog_and_roundtrip(tmp_path: Path) -> None:
  """Test compiling discrete ODL files and verifying roundtrip equivalence.

  Args:
      tmp_path: Pytest temporary directory fixture.
  """
  odl_dir = tmp_path / "odl"
  odl_dir.mkdir()

  f1 = odl_dir / "Op1.yaml"
  f1.write_text("operation: Op1\ndescription: Test Op1\nstd_args: []\nvariants: {}\n")

  f2 = odl_dir / "Op2.yaml"
  f2.write_text("operation: Op2\ndescription: Test Op2\nstd_args: []\nvariants: {}\n")

  # Files to skip (broken or non-dict)
  f3 = odl_dir / "Invalid.yaml"
  f3.write_text("broken: [yaml\n")

  f4 = odl_dir / "NonDict.yaml"
  f4.write_text("- item1\n- item2\n")

  out_json = tmp_path / "odl.json"
  count = compiler.compile_catalog(odl_dir, out_json, validate=True)
  assert count == 2
  assert out_json.is_file()

  # Verify roundtrip
  assert compiler.verify_roundtrip(odl_dir, out_json) is True

  # Test compile with validate=False
  assert compiler.compile_catalog(odl_dir, out_json, validate=False) == 2


def test_compile_catalog_validation_failure(tmp_path: Path) -> None:
  """Test compile_catalog raising ValueError on invalid schema.

  Args:
      tmp_path: Pytest temporary directory fixture.
  """
  odl_dir = tmp_path / "odl"
  odl_dir.mkdir()
  out_json = tmp_path / "odl.json"

  with patch("scripts.compile_odl_catalog.SemanticsFile.model_validate", side_effect=ValueError("Invalid")):
    with pytest.raises(ValueError, match="Compiled catalog failed SemanticsFile validation"):
      compiler.compile_catalog(odl_dir, out_json, validate=True)


def test_verify_roundtrip_edge_cases(tmp_path: Path) -> None:
  """Test verify_roundtrip when file is missing, non-dict, or mismatched.

  Args:
      tmp_path: Pytest temporary directory fixture.
  """
  odl_dir = tmp_path / "odl"
  odl_dir.mkdir()
  missing_json = tmp_path / "missing.json"

  # Missing file
  assert compiler.verify_roundtrip(odl_dir, missing_json) is False

  # Non-dict JSON
  non_dict_json = tmp_path / "nondict.json"
  non_dict_json.write_text("[]\n")
  assert compiler.verify_roundtrip(odl_dir, non_dict_json) is False

  # Key mismatch
  mismatch_json = tmp_path / "mismatch.json"
  mismatch_json.write_text('{"OtherOp": {}}\n')
  assert compiler.verify_roundtrip(odl_dir, mismatch_json) is False

  # Content mismatch
  f1 = odl_dir / "Op1.yaml"
  f1.write_text("operation: Op1\nvariants: {}\n")
  diff_content_json = tmp_path / "diff.json"
  diff_content_json.write_text('{"Op1": {"operation": "Op1", "variants": {"torch": {}}}}\n')
  assert compiler.verify_roundtrip(odl_dir, diff_content_json) is False


def test_decompile_catalog(tmp_path: Path) -> None:
  """Test decompiling JSON catalog into discrete YAML files.

  Args:
      tmp_path: Pytest temporary directory fixture.
  """
  json_file = tmp_path / "catalog.json"
  json_file.write_text('{"OpA": {"operation": "OpA"}, "OpB": {"operation": "OpB"}, "Invalid": "not_dict"}\n')

  out_dir = tmp_path / "decompiled"
  count = compiler.decompile_catalog(json_file, out_dir)
  assert count == 2
  assert (out_dir / "OpA.yaml").is_file()
  assert (out_dir / "OpB.yaml").is_file()

  # Non-dict catalog
  bad_json = tmp_path / "bad.json"
  bad_json.write_text("[]\n")
  assert compiler.decompile_catalog(bad_json, out_dir) == 0


def test_main_cli(tmp_path: Path) -> None:
  """Test main CLI entry point.

  Args:
      tmp_path: Pytest temporary directory fixture.
  """
  odl_dir = tmp_path / "odl"
  odl_dir.mkdir()
  (odl_dir / "Op.yaml").write_text("operation: Op\nstd_args: []\nvariants: {}\n")
  out_json = tmp_path / "odl.json"

  # Compile & verify
  exit_code = compiler.main(["--odl-dir", str(odl_dir), "--output-json", str(out_json)])
  assert exit_code == 0

  # Verify-only success
  exit_code = compiler.main(["--odl-dir", str(odl_dir), "--output-json", str(out_json), "--verify-only"])
  assert exit_code == 0

  # Verify-only failure
  missing_json = tmp_path / "none.json"
  exit_code = compiler.main(["--odl-dir", str(odl_dir), "--output-json", str(missing_json), "--verify-only"])
  assert exit_code == 1

  # Roundtrip failure after compilation
  with patch("scripts.compile_odl_catalog.verify_roundtrip", return_value=False):
    exit_code = compiler.main(["--odl-dir", str(odl_dir), "--output-json", str(out_json)])
    assert exit_code == 1


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
  (odl_dir / "Op.yaml").write_text("operation: Op\nstd_args: []\nvariants: {}\n")
  out_json = tmp_path / "odl.json"

  repo_str = str(compiler.REPO_ROOT)
  orig_sys_path = list(sys.path)
  sys.path = [p for p in sys.path if p != repo_str]

  try:
    monkeypatch.setattr(
      "sys.argv",
      ["compile_odl_catalog.py", "--odl-dir", str(odl_dir), "--output-json", str(out_json)],
    )
    with pytest.raises(SystemExit) as excinfo:
      runpy.run_module("scripts.compile_odl_catalog", run_name="__main__")
    assert excinfo.value.code == 0
  finally:
    sys.path = orig_sys_path
