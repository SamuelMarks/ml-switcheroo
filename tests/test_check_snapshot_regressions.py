"""Test suite for snapshot regression and diff checking utility."""

import gzip
import json
from pathlib import Path
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import pytest

from scripts.check_snapshot_regressions import (
  compare_snapshots,
  has_breaking_changes,
  load_snapshot_file,
  main,
  render_markdown_changelog,
  run_regression_check,
)


def test_load_snapshot_file_missing(tmp_path: Path) -> None:
  """Verifies FileNotFoundError when file does not exist."""
  with pytest.raises(FileNotFoundError):
    load_snapshot_file(tmp_path / "nonexistent.json")


def test_load_snapshot_file_dict(tmp_path: Path) -> None:
  """Verifies loading dictionary JSON file."""
  p = tmp_path / "snap.json"
  p.write_text(json.dumps({"version": "1.0", "categories": {}}))
  data = load_snapshot_file(p)
  assert data["version"] == "1.0"


def test_load_snapshot_file_gz(tmp_path: Path) -> None:
  """Verifies loading gzip-compressed JSON file."""
  p = tmp_path / "snap.json.gz"
  with gzip.open(p, "wt", encoding="utf-8") as f:
    json.dump({"target": "torch"}, f)
  data = load_snapshot_file(p)
  assert data["target"] == "torch"


def test_load_snapshot_file_list(tmp_path: Path) -> None:
  """Verifies loading list JSON file wraps items in categories all."""
  p = tmp_path / "snap_list.json"
  p.write_text(json.dumps([{"name": "op1"}]))
  data = load_snapshot_file(p)
  assert "categories" in data
  assert data["categories"]["all"] == [{"name": "op1"}]


def test_load_snapshot_file_invalid(tmp_path: Path) -> None:
  """Verifies ValueError when JSON content is not a dict or list."""
  p = tmp_path / "snap_invalid.json"
  p.write_text(json.dumps("invalid string"))
  with pytest.raises(ValueError):
    load_snapshot_file(p)


def test_compare_snapshots_and_changelog() -> None:
  """Verifies compare_snapshots and render_markdown_changelog."""
  snap1: Dict[str, Any] = {"categories": {"math": [{"name": "foo", "api_path": "test.foo"}]}}
  snap2: Dict[str, Any] = {
    "categories": {"math": [{"name": "foo", "api_path": "test.foo"}, {"name": "bar", "api_path": "test.bar"}]}
  }

  diff = compare_snapshots(snap1, snap2)
  assert diff is not None
  changelog = render_markdown_changelog(diff)
  assert "Added" in changelog or "test.bar" in changelog

  with patch("scripts.check_snapshot_regressions.diff_snapshots", None):
    assert compare_snapshots(snap1, snap2) is None

  with patch("scripts.check_snapshot_regressions.generate_changelog", None):
    assert "Diff engine not available" in render_markdown_changelog(diff)
  assert "Diff engine not available" in render_markdown_changelog(None)


def test_has_breaking_changes() -> None:
  """Verifies breaking change detection logic."""
  assert has_breaking_changes(None) is False

  mock_diff = MagicMock()
  mock_diff.removed = []
  mock_diff.breaking_signature_changed = []
  assert has_breaking_changes(mock_diff) is False

  mock_diff.removed = ["torch.old_op"]
  assert has_breaking_changes(mock_diff) is True

  mock_diff.removed = []
  mock_diff.breaking_signature_changed = ["torch.changed_op"]
  assert has_breaking_changes(mock_diff) is True


def test_run_regression_check_missing_older() -> None:
  """Verifies failure when older snapshot is not provided."""
  assert run_regression_check(older=None) == 1


def test_run_regression_check_load_errors(tmp_path: Path) -> None:
  """Verifies error handling on invalid snapshot files."""
  assert run_regression_check(older=str(tmp_path / "missing.json")) == 1

  older_file = tmp_path / "older.json"
  older_file.write_text(json.dumps({}))
  assert run_regression_check(older=str(older_file), newer=str(tmp_path / "missing_newer.json")) == 1


def test_run_regression_check_no_newer_or_fw(tmp_path: Path) -> None:
  """Verifies error when neither newer nor framework is provided."""
  older_file = tmp_path / "older.json"
  older_file.write_text(json.dumps({}))
  assert run_regression_check(older=str(older_file)) == 1


def test_run_regression_check_framework_extraction(tmp_path: Path) -> None:
  """Verifies framework extraction via isolated child process."""
  older_file = tmp_path / "older.json"
  older_file.write_text(json.dumps({"categories": {}}))

  with patch("scripts.check_snapshot_regressions.extract_snapshot_isolated", None):
    assert run_regression_check(older=str(older_file), framework="torch") == 1

  with patch("scripts.check_snapshot_regressions.extract_snapshot_isolated", return_value={}):
    assert run_regression_check(older=str(older_file), framework="torch") == 1

  with patch(
    "scripts.check_snapshot_regressions.extract_snapshot_isolated",
    return_value={"categories": {}},
  ):
    assert run_regression_check(older=str(older_file), framework="torch") == 0


def test_run_regression_check_success_and_changelog_output(tmp_path: Path) -> None:
  """Verifies successful check writing markdown changelog to file."""
  older_file = tmp_path / "older.json"
  newer_file = tmp_path / "newer.json"
  out_changelog = tmp_path / "changelog.md"

  older_file.write_text(json.dumps({"categories": {"ops": [{"name": "foo", "api_path": "m.foo"}]}}))
  newer_file.write_text(json.dumps({"categories": {"ops": [{"name": "foo", "api_path": "m.foo"}]}}))

  res = run_regression_check(
    older=str(older_file),
    newer=str(newer_file),
    output_changelog=str(out_changelog),
  )
  assert res == 0
  assert out_changelog.exists()
  content = out_changelog.read_text()
  assert "# Changelog Report" in content


def test_run_regression_check_breaking_changes(tmp_path: Path) -> None:
  """Verifies fail_on_breaking behavior when removals exist."""
  older_file = tmp_path / "older.json"
  newer_file = tmp_path / "newer.json"

  older_file.write_text(json.dumps({"categories": {"ops": [{"name": "removed_op", "api_path": "m.removed"}]}}))
  newer_file.write_text(json.dumps({"categories": {"ops": []}}))

  # Without fail_on_breaking -> returns 0
  assert run_regression_check(older=str(older_file), newer=str(newer_file), fail_on_breaking=False) == 0

  # With fail_on_breaking -> returns 1
  assert run_regression_check(older=str(older_file), newer=str(newer_file), fail_on_breaking=True) == 1


def test_main_cli(tmp_path: Path) -> None:
  """Verifies main CLI parser dispatch."""
  older_file = tmp_path / "older.json"
  newer_file = tmp_path / "newer.json"
  older_file.write_text(json.dumps({"categories": {}}))
  newer_file.write_text(json.dumps({"categories": {}}))

  argv = [
    "--older",
    str(older_file),
    "--newer",
    str(newer_file),
    "--fail-on-breaking",
  ]
  assert main(argv) == 0


def test_cli_execution_main() -> None:
  """Verifies executing script module directly."""
  import runpy
  import sys

  with patch.object(sys, "argv", ["scripts/check_snapshot_regressions.py", "--help"]):
    with pytest.raises(SystemExit) as exc:
      runpy.run_module("scripts.check_snapshot_regressions", run_name="__main__")
    assert exc.value.code == 0
