"""Unit and integration tests for `scripts/validate_snapshot_grounding.py` and snapshot loader."""

import json
import runpy
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

import scripts.validate_snapshot_grounding as validator
from scripts.snapshots.loader import (
  compute_snapshot_checksums,
  load_snapshots_multi,
)


def test_validate_snapshot_grounding_exports() -> None:
  """Verify that `scripts/validate_snapshot_grounding.py` exports all canonical audit helpers."""
  for export_name in validator.__all__:
    assert hasattr(validator, export_name), f"Missing export: {export_name}"


def test_validate_snapshot_grounding_main_entrypoint(monkeypatch: pytest.MonkeyPatch) -> None:
  """Verify running `scripts.validate_snapshot_grounding` as `__main__` invokes `main()` and exits cleanly.

  Args:
      monkeypatch: Pytest monkeypatch fixture.
  """
  monkeypatch.setattr(sys, "argv", ["validate_snapshot_grounding.py", "--strict"])
  with patch("scripts.audit_against_snapshots.main", return_value=0) as mock_main:
    with pytest.raises(SystemExit) as excinfo:
      runpy.run_module("scripts.validate_snapshot_grounding", run_name="__main__")
    assert excinfo.value.code == 0
    mock_main.assert_called_once()


def test_load_snapshots_multi_strict_missing_raises(tmp_path: Path) -> None:
  """Verify `load_snapshots_multi(strict=True)` raises FileNotFoundError when directory is empty or missing.

  Args:
      tmp_path: Pytest temporary directory fixture.
  """
  empty_dir = tmp_path / "empty_snapshots"
  empty_dir.mkdir()
  with pytest.raises(FileNotFoundError, match="No valid framework snapshots found"):
    load_snapshots_multi([empty_dir], strict=True)


def test_load_snapshots_multi_frameworks_sibling_dir(tmp_path: Path) -> None:
  """Verify `load_snapshots_multi` automatically discovers sibling `frameworks/` next to `snapshots/`.

  Args:
      tmp_path: Pytest temporary directory fixture.
  """
  pkg_root = tmp_path / "ml_ecosystem_snapshots"
  snap_dir = pkg_root / "snapshots"
  fw_dir = pkg_root / "frameworks"
  snap_dir.mkdir(parents=True)
  fw_dir.mkdir(parents=True)

  (snap_dir / "torch_v2.0.json").write_text(json.dumps({"functions": {"torch.add": {}}}))
  (fw_dir / "optax_v1.0.json").write_text(json.dumps({"functions": {"optax.adam": {}}}))
  (fw_dir / "pax_v1.0.json").write_text(json.dumps({"functions": {"paxml.layer": {}}}))
  (fw_dir / "stablehlo_exhaustive.json").write_text(json.dumps({"functions": {"stablehlo.add": {}}}))

  loaded = load_snapshots_multi([snap_dir], strict=True)
  assert "torch" in loaded
  assert "optax" in loaded
  assert "paxml" in loaded
  assert "stablehlo" in loaded


def test_compute_snapshot_checksums_real_and_default(tmp_path: Path) -> None:
  """Verify `compute_snapshot_checksums` computes SHA-256 digests and skips `_map.json` / empty files.

  Args:
      tmp_path: Pytest temporary directory fixture.
  """
  snap_dir = tmp_path / "snaps"
  snap_dir.mkdir()
  (snap_dir / "torch_v2.0.json").write_text('{"functions": {}}')
  (snap_dir / "torch_map.json").write_text('{"ignored": true}')
  (snap_dir / "empty_v1.0.json").write_text("")

  checksums = compute_snapshot_checksums([snap_dir, tmp_path / "nonexistent"])
  assert "torch_v2.0.json" in checksums
  assert "torch_map.json" not in checksums
  assert "empty_v1.0.json" not in checksums

  default_checksums = compute_snapshot_checksums(None)
  assert isinstance(default_checksums, dict)


def test_audit_against_snapshots_report_generation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
  """Verify `scripts/audit_against_snapshots.py --report` writes a JSON report with checksums.

  Args:
      tmp_path: Pytest temporary directory fixture.
      monkeypatch: Pytest monkeypatch fixture.
  """
  report_file = tmp_path / "audit_report.json"
  snap_dir = tmp_path / "snapshots"
  snap_dir.mkdir()
  (snap_dir / "torch_v2.0.json").write_text(json.dumps({"functions": {"torch.add": {}}}))

  monkeypatch.setattr(
    sys,
    "argv",
    [
      "audit_against_snapshots.py",
      "--snapshots-dir",
      str(snap_dir),
      "--report",
      str(report_file),
    ],
  )
  with patch("scripts.audit_against_snapshots.audit_frameworks", return_value=[]):
    with patch("scripts.audit_against_snapshots.audit_python_ast", return_value=[]):
      with patch("scripts.audit_against_snapshots.audit_inline_snippets", return_value=[]):
        with patch("scripts.audit_against_snapshots.GroundingEngine", side_effect=OSError("bad dir")):
          rc = validator.main()
          assert rc == 0
          assert report_file.exists()
          report_data = json.loads(report_file.read_text(encoding="utf-8"))
          assert "snapshot_checksums" in report_data
