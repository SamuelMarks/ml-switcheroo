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
  f1.write_text(
    """operation: Op1
description: Test Op1
std_args: []
variants: {}
"""
  )

  f2 = odl_dir / "Op2.yaml"
  f2.write_text(
    """operation: Op2
description: Test Op2
std_args: []
variants: {}
"""
  )

  f3 = odl_dir / "Invalid.yaml"
  f3.write_text("broken: [yaml")

  f4 = odl_dir / "NonDict.yaml"
  f4.write_text(
    """- item1
- item2
"""
  )

  out_json = tmp_path / "odl.json"
  count = compiler.compile_catalog(odl_dir, out_json, validate=True)
  assert count == 2
  assert out_json.is_file()
  assert compiler.verify_roundtrip(odl_dir, out_json) is True
  assert compiler.compile_catalog(odl_dir, out_json, validate=False) == 2


def test_export_and_verify_docs_catalog(tmp_path: Path) -> None:
  """Test deterministic docs YAML catalog export and verification branches.

  Args:
      tmp_path: Pytest temporary directory fixture.
  """
  odl_dir = tmp_path / "odl"
  odl_dir.mkdir()
  (odl_dir / "OpB.yaml").write_text(
    """operation: OpB
std_args: []
variants: {}
"""
  )
  (odl_dir / "OpA.yaml").write_text(
    """operation: OpA
std_args: []
variants: {}
"""
  )

  docs_yaml = tmp_path / "docs" / "operations.yaml"
  assert compiler.verify_docs_catalog(odl_dir, docs_yaml) is False

  exported = compiler.export_docs_catalog(odl_dir, docs_yaml)
  assert exported == 2
  assert compiler.verify_docs_catalog(odl_dir, docs_yaml) is True

  non_list_yaml = tmp_path / "nonlist.yaml"
  non_list_yaml.write_text("key: val")
  assert compiler.verify_docs_catalog(odl_dir, non_list_yaml) is False

  broken_yaml = tmp_path / "broken.yaml"
  broken_yaml.write_text("broken: [yaml")
  assert compiler.verify_docs_catalog(odl_dir, broken_yaml) is False


def test_compile_catalog_validation_failure(tmp_path: Path) -> None:
  """Test compile_catalog raising ValueError on invalid schema.

  Args:
      tmp_path: Pytest temporary directory fixture.
  """
  odl_dir = tmp_path / "odl"
  odl_dir.mkdir()
  out_json = tmp_path / "odl.json"

  with (
    patch("scripts.compile_odl_catalog.SemanticsFile.model_validate", side_effect=ValueError("Invalid")),
    pytest.raises(ValueError, match="Compiled catalog failed SemanticsFile validation"),
  ):
    compiler.compile_catalog(odl_dir, out_json, validate=True)


def test_verify_roundtrip_edge_cases(tmp_path: Path) -> None:
  """Test verify_roundtrip when file is missing, non-dict, or mismatched.

  Args:
      tmp_path: Pytest temporary directory fixture.
  """
  odl_dir = tmp_path / "odl"
  odl_dir.mkdir()
  missing_json = tmp_path / "missing.json"

  assert compiler.verify_roundtrip(odl_dir, missing_json) is False

  non_dict_json = tmp_path / "nondict.json"
  non_dict_json.write_text("[]")
  assert compiler.verify_roundtrip(odl_dir, non_dict_json) is False

  mismatch_json = tmp_path / "mismatch.json"
  mismatch_json.write_text('{"OtherOp": {}}')
  assert compiler.verify_roundtrip(odl_dir, mismatch_json) is False

  f1 = odl_dir / "Op1.yaml"
  f1.write_text(
    """operation: Op1
variants: {}
"""
  )
  diff_content_json = tmp_path / "diff.json"
  diff_content_json.write_text('{"Op1": {"operation": "Op1", "variants": {"torch": {}}}}')
  assert compiler.verify_roundtrip(odl_dir, diff_content_json) is False


def test_decompile_catalog(tmp_path: Path) -> None:
  """Test decompiling JSON catalog into discrete YAML files.

  Args:
      tmp_path: Pytest temporary directory fixture.
  """
  json_file = tmp_path / "catalog.json"
  json_file.write_text('{"OpA": {"operation": "OpA"}, "OpB": {"operation": "OpB"}, "Invalid": "not_dict"}')

  out_dir = tmp_path / "decompiled"
  count = compiler.decompile_catalog(json_file, out_dir)
  assert count == 2
  assert (out_dir / "OpA.yaml").is_file()
  assert (out_dir / "OpB.yaml").is_file()

  bad_json = tmp_path / "bad.json"
  bad_json.write_text("[]")
  assert compiler.decompile_catalog(bad_json, out_dir) == 0


def test_main_cli(tmp_path: Path) -> None:
  """Test main CLI entry point including optional --docs-yaml verification.

  Args:
      tmp_path: Pytest temporary directory fixture.
  """
  odl_dir = tmp_path / "odl"
  odl_dir.mkdir()
  (odl_dir / "Op.yaml").write_text(
    """operation: Op
std_args: []
variants: {}
"""
  )
  out_json = tmp_path / "odl.json"
  docs_yaml = tmp_path / "operations.yaml"

  exit_code = compiler.main(["--odl-dir", str(odl_dir), "--output-json", str(out_json), "--docs-yaml", str(docs_yaml)])
  assert exit_code == 0
  assert docs_yaml.is_file()

  exit_code = compiler.main(["--odl-dir", str(odl_dir), "--output-json", str(out_json)])
  assert exit_code == 0

  exit_code = compiler.main(
    ["--odl-dir", str(odl_dir), "--output-json", str(out_json), "--docs-yaml", str(docs_yaml), "--verify-only"]
  )
  assert exit_code == 0

  exit_code = compiler.main(["--odl-dir", str(odl_dir), "--output-json", str(out_json), "--verify-only"])
  assert exit_code == 0

  missing_json = tmp_path / "none.json"
  exit_code = compiler.main(["--odl-dir", str(odl_dir), "--output-json", str(missing_json), "--verify-only"])
  assert exit_code == 1

  with patch("scripts.compile_odl_catalog.verify_roundtrip", return_value=False):
    exit_code = compiler.main(["--odl-dir", str(odl_dir), "--output-json", str(out_json)])
    assert exit_code == 1


def test_main_entrypoint(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
  """Test __main__ execution block.

  Args:
      monkeypatch: Pytest monkeypatch fixture.
      tmp_path: Pytest temporary directory fixture.
  """
  import runpy
  import sys

  odl_dir = tmp_path / "odl"
  odl_dir.mkdir()
  (odl_dir / "Op.yaml").write_text(
    """operation: Op
std_args: []
variants: {}
"""
  )
  out_json = tmp_path / "odl.json"

  repo_str = str(compiler.REPO_ROOT)
  repo_src = str(compiler.REPO_ROOT / "src")
  orig_sys_path = list(sys.path)
  sys.path = [p for p in sys.path if p not in (repo_str, repo_src)]

  try:
    import importlib

    importlib.reload(compiler)
    monkeypatch.setattr(
      "sys.argv",
      ["compile_odl_catalog.py", "--odl-dir", str(odl_dir), "--output-json", str(out_json)],
    )
    with pytest.raises(SystemExit) as excinfo:
      runpy.run_module("scripts.compile_odl_catalog", run_name="__main__")
    assert excinfo.value.code == 0
  finally:
    sys.path = orig_sys_path
