"""Unit tests for `handle_ci` using real `tmp_path` files and mini-configurations (`ml_switcheroo.cli.handlers.verify`)."""

import json
import textwrap
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from pydantic import JsonValue

import ml_switcheroo.cli.handlers.verify as verify_mod
from ml_switcheroo.cli.handlers.verify import handle_ci
from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo.semantics.manager import SemanticsManager

_NL = chr(10)


def _make_mini_semantics() -> SemanticsManager:
  """Build a compact `SemanticsManager` with `ManualOp` and `BrokenOp` definitions.

  Returns:
      Configured in-memory `SemanticsManager`.
  """
  sem = SemanticsManager(load_defaults=False)
  sem.data = {
    "manual_op": {
      "operation": "manual_op",
      "std_args": ["x"],
      "variants": {"numpy": {"api": "numpy.abs"}},
    }
  }
  return sem


def test_handle_ci_real_files_plugins_readme_and_json_report(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
  """Verify `handle_ci` runs real `BatchValidator`, `ReadmeEditor`, plugin loading, and JSON report writing.

  Args:
      tmp_path: Pytest temporary directory fixture.
      monkeypatch: Pytest monkeypatch fixture.
  """
  monkeypatch.chdir(tmp_path)

  # Real `pyproject.toml` with custom `plugin_paths` and manual `tests/` directory
  plug_dir = tmp_path / "extra_plugins"
  plug_dir.mkdir()
  (plug_dir / "ci_plug.py").write_text(
    textwrap.dedent(
      """
      from ml_switcheroo.core.hooks import register_hook

      @register_hook("ci_custom_hook")
      def _hook(node, ctx):
        return node
      """
    ),
    encoding="utf-8",
  )
  (tmp_path / "pyproject.toml").write_text(
    f'[tool.ml_switcheroo]{_NL}plugin_paths = ["extra_plugins"]{_NL}',
    encoding="utf-8",
  )

  tests_dir = tmp_path / "tests"
  tests_dir.mkdir()
  (tests_dir / "test_manual_op.py").write_text("def test_manual_op(): pass", encoding="utf-8")

  readme_path = tmp_path / "README.md"
  readme_path.write_text(f"# Project{_NL}{_NL}## ✅ Compatibility Matrix{_NL}{_NL}Old table{_NL}", encoding="utf-8")
  report_path = tmp_path / "reports" / "ci.json"

  mini_sem = _make_mini_semantics()
  monkeypatch.setattr(verify_mod, "SemanticsManager", lambda: mini_sem)

  assert handle_ci(update_readme=True, readme_path=readme_path, json_report=report_path, repair=False) == 0
  assert json.loads(report_path.read_text(encoding="utf-8")) == {"manual_op": True}
  assert "✅ Passing" in readme_path.read_text(encoding="utf-8")

  # Configured `plugin_paths` returning 0 newly loaded plugins (already loaded)
  with patch.object(verify_mod, "load_plugins", return_value=0):
    assert handle_ci(update_readme=False, readme_path=readme_path, json_report=None, repair=False) == 0

  # Directory without `tests/` and with 0 newly loaded plugins -> `manual_test_dir=None`
  empty_ws = tmp_path / "empty_ws"
  empty_ws.mkdir()
  monkeypatch.chdir(empty_ws)
  assert handle_ci(update_readme=False, readme_path=readme_path, json_report=None, repair=False) == 0

  # Report write error when `json_report` points to an existing directory
  bad_report_dir = tmp_path / "report_is_dir"
  bad_report_dir.mkdir()
  assert handle_ci(update_readme=False, readme_path=readme_path, json_report=bad_report_dir, repair=False) == 1

  # `RuntimeConfig.load` raising an exception logs warning and continues cleanly
  with patch.object(RuntimeConfig, "load", side_effect=RuntimeError("bad toml")):
    assert handle_ci(update_readme=False, readme_path=readme_path, json_report=None, repair=False) == 0


def test_handle_ci_repair_mode_branches(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
  """Verify `handle_ci(..., repair=True)` handles proposed fixes, unfixable ops, and missing definitions.

  Args:
      tmp_path: Pytest temporary directory fixture.
      monkeypatch: Pytest monkeypatch fixture.
  """
  monkeypatch.chdir(tmp_path)
  sem = _make_mini_semantics()
  sem.data["fixable_op"] = {"operation": "fixable_op", "variants": {}}
  sem.data["unfixable_op"] = {"operation": "unfixable_op", "variants": {}}
  updated_calls: list[tuple[str, dict[str, JsonValue]]] = []

  monkeypatch.setattr(verify_mod, "SemanticsManager", lambda: sem)
  monkeypatch.setattr(sem, "update_definition", lambda op_id, d: updated_calls.append((op_id, d)))

  mock_val = MagicMock()
  mock_val.run_all.return_value = {
    "manual_op": True,
    "fixable_op": False,
    "unfixable_op": False,
    "ghost_missing_op": False,
  }
  monkeypatch.setattr(verify_mod, "BatchValidator", lambda _s: mock_val)

  mock_bisector = MagicMock()
  mock_bisector.propose_fix.side_effect = lambda op_id, _defn: {"rtol": 0.1} if op_id == "fixable_op" else None
  monkeypatch.setattr(verify_mod, "SemanticsBisector", lambda _r: mock_bisector)

  assert handle_ci(update_readme=False, readme_path=tmp_path / "README.md", json_report=None, repair=True) == 0
  assert updated_calls == [("fixable_op", {"rtol": 0.1})]

  # All failed ops returning `None` from `propose_fix` (`fixed_count == 0`)
  mock_val.run_all.return_value = {"unfixable_op": False}
  assert handle_ci(update_readme=False, readme_path=tmp_path / "README.md", json_report=None, repair=True) == 0
