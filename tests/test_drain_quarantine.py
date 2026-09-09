"""Tests for quarantine drainage and triage tool."""

from pathlib import Path
from typing import Any, Dict
from unittest.mock import patch
import pytest
import yaml

import scripts.drain_quarantine as drainer


def test_find_snapshot_api_match() -> None:
  """Test finding matching APIs by exact name, lowercase, and suffix."""
  fw_snapshot: Dict[str, Any] = {
    "ExactMatch": {"api_path": "torch.ExactMatch", "name": "ExactMatch"},
    "lowerop": {"api_path": "torch.lowerop", "name": "lowerop"},
    "different_key": {"api_path": "torch.by_name", "name": "ByNameOp"},
    "torch.nn.SuffixMatch": {"api_path": "torch.nn.SuffixMatch", "name": "SuffixMatch"},
    "torch.nn.lower_suffix": {"api_path": "torch.nn.lower_suffix", "name": "lower_suffix"},
    "torch.nn.OnlyPathMatch": {"api_path": "torch.nn.OnlyPathMatch"},
    "non_dict_entry": "ignored",
  }

  # Exact match
  res = drainer.find_snapshot_api_match("ExactMatch", fw_snapshot)
  assert res is not None
  assert res[0] == "torch.ExactMatch"

  # Lowercase match
  res = drainer.find_snapshot_api_match("LowerOp", fw_snapshot)
  assert res is not None
  assert res[0] == "torch.lowerop"

  # Match by entry['name'] where key is different
  res = drainer.find_snapshot_api_match("ByNameOp", fw_snapshot)
  assert res is not None
  assert res[0] == "torch.by_name"

  # Suffix match
  res = drainer.find_snapshot_api_match("SuffixMatch", fw_snapshot)
  assert res is not None
  assert res[0] == "torch.nn.SuffixMatch"

  # Suffix lowercase match
  res = drainer.find_snapshot_api_match("Lower_Suffix", fw_snapshot)
  assert res is not None
  assert res[0] == "torch.nn.lower_suffix"

  # Path suffix match with missing or different name
  res = drainer.find_snapshot_api_match("OnlyPathMatch", fw_snapshot)
  assert res is not None
  assert res[0] == "torch.nn.OnlyPathMatch"

  # No match
  assert drainer.find_snapshot_api_match("UnknownOp", fw_snapshot) is None


def test_build_variant_entry() -> None:
  """Test constructing variant dictionary with required parameter filtering."""
  api_data = {
    "params": [
      {"name": "self", "kind": "POSITIONAL_ONLY"},
      {"name": "cls", "kind": "POSITIONAL_ONLY"},
      {"name": "input", "kind": "POSITIONAL_OR_KEYWORD", "default": None},
      {"name": "optional_arg", "kind": "POSITIONAL_OR_KEYWORD", "default": 0},
      {"name": "var_args", "kind": "VAR_POSITIONAL"},
      {"name": "kwargs", "kind": "VAR_KEYWORD"},
    ]
  }

  variant = drainer.build_variant_entry("torch.foo", api_data)
  assert variant["api"] == "torch.foo"
  assert variant["args"] == {"input": "input"}


def test_drain_quarantine_data(tmp_path: Path) -> None:
  """Test drain_quarantine_data triage and discrete ODL creation.

  Args:
      tmp_path: Pytest temporary directory fixture.
  """
  odl_dir = tmp_path / "odl"
  odl_dir.mkdir()

  # Create an existing ODL file
  (odl_dir / "AlreadyExists.yaml").write_text("operation: AlreadyExists\nstd_args: []\n")

  quarantine_data: Dict[str, Any] = {
    "AlreadyExists": {"description": "Existing"},
    "MigratableOp": {
      "description": "Will be migrated",
      "std_args": [],
    },
    "TrulyQuarantined": {
      "description": "Internal type",
      "std_args": [],
    },
    "NonDictOp": "invalid_spec",
  }

  snapshots: Dict[str, Dict[str, Any]] = {
    "torch": {
      "MigratableOp": {
        "api_path": "torch.migratable_op",
        "name": "MigratableOp",
        "docstring": "Migrated docstring\nSecond line",
        "params": [
          {"name": "x", "kind": "POSITIONAL_OR_KEYWORD", "annotation": "Tensor"},
        ],
      }
    },
    "jax": {
      "MigratableOp": {
        "api_path": "jax.numpy.migratable_op",
        "name": "MigratableOp",
        "params": [
          {"name": "a", "kind": "POSITIONAL_OR_KEYWORD", "annotation": "Tensor"},
        ],
      }
    },
  }

  # Dry run
  removed, migrated, cleansed = drainer.drain_quarantine_data(
    quarantine_data=quarantine_data,
    odl_dir=odl_dir,
    snapshots=snapshots,
    dry_run=True,
    framework_order=["torch"],
  )
  assert removed == 1
  assert migrated == 1
  assert "TrulyQuarantined" in cleansed
  assert "NonDictOp" in cleansed
  assert not (odl_dir / "MigratableOp.yaml").exists()

  # Real run
  removed, migrated, cleansed = drainer.drain_quarantine_data(
    quarantine_data=quarantine_data,
    odl_dir=odl_dir,
    snapshots=snapshots,
    dry_run=False,
  )
  assert removed == 1
  assert migrated == 1
  assert (odl_dir / "MigratableOp.yaml").exists()

  with open(odl_dir / "MigratableOp.yaml", "r", encoding="utf-8") as f:
    odl_content = yaml.safe_load(f)
  assert odl_content["operation"] == "MigratableOp"
  assert odl_content["description"] == "Migrated docstring"
  assert "torch" in odl_content["variants"]


def test_drain_quarantine_empty_desc_fallback(tmp_path: Path) -> None:
  """Test fallback description when docstring and description are empty.

  Args:
      tmp_path: Pytest temporary directory fixture.
  """
  odl_dir = tmp_path / "odl"
  odl_dir.mkdir()

  quarantine_data = {
    "NoDescOp": {"description": ""},
  }
  snapshots: Dict[str, Dict[str, Any]] = {
    "torch": {
      "NoDescOp": {
        "api_path": "torch.no_desc",
        "name": "NoDescOp",
        "docstring": "\nSecond line",
        "params": [],
      }
    }
  }

  _, migrated, _ = drainer.drain_quarantine_data(
    quarantine_data=quarantine_data,
    odl_dir=odl_dir,
    snapshots=snapshots,
    dry_run=False,
  )
  assert migrated == 1
  with open(odl_dir / "NoDescOp.yaml", "r", encoding="utf-8") as f:
    odl_content = yaml.safe_load(f)
  assert odl_content["description"] == "Standardized definition for NoDescOp."


def test_main_cli(tmp_path: Path) -> None:
  """Test main CLI entry point for drain_quarantine.

  Args:
      tmp_path: Pytest temporary directory fixture.
  """
  q_file = tmp_path / "quarantine.yaml"
  q_file.write_text("Op: {}\n")
  odl_dir = tmp_path / "odl"
  odl_dir.mkdir()

  with patch("scripts.drain_quarantine.load_snapshots_multi", return_value={}):
    exit_code = drainer.main(
      [
        "--quarantine-file",
        str(q_file),
        "--odl-dir",
        str(odl_dir),
        "--dry-run",
      ]
    )
    assert exit_code == 0

    exit_code = drainer.main(
      [
        "--quarantine-file",
        str(q_file),
        "--odl-dir",
        str(odl_dir),
      ]
    )
    assert exit_code == 0


def test_main_entrypoint(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
  """Test __main__ execution block.

  Args:
      monkeypatch: Pytest monkeypatch fixture.
      tmp_path: Pytest temporary directory fixture.
  """
  import sys
  import runpy

  q_file = tmp_path / "quarantine.yaml"
  q_file.write_text("Op: {}\n")
  odl_dir = tmp_path / "odl"
  odl_dir.mkdir()

  repo_str = str(drainer.REPO_ROOT)
  orig_sys_path = list(sys.path)
  sys.path = [p for p in sys.path if p != repo_str]

  try:
    monkeypatch.setattr(
      "sys.argv",
      ["drain_quarantine.py", "--quarantine-file", str(q_file), "--odl-dir", str(odl_dir), "--dry-run"],
    )
    with patch("scripts.drain_quarantine.load_snapshots_multi", return_value={}):
      with pytest.raises(SystemExit) as excinfo:
        runpy.run_module("scripts.drain_quarantine", run_name="__main__")
      assert excinfo.value.code == 0
  finally:
    sys.path = orig_sys_path
