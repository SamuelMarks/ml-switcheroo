"""Docstring for test_suggest_patch module."""

import pathlib


def test_suggest_missing_branches(tmp_path: pathlib.Path) -> None:
  """Verify suggest missing branches behavior.

  Args:
      tmp_path: Pytest temporary directory path fixture.
  """
  import importlib
  import inspect
  from unittest.mock import MagicMock, patch

  import ml_switcheroo.cli.handlers.suggest as suggest_mod
  from ml_switcheroo.cli.handlers.suggest import handle_suggest

  # 1. glob mode (.*) with some members and no members
  out_dir = tmp_path / "suggest_out"
  out_dir.mkdir(parents=True, exist_ok=True)

  handle_suggest("math.*", out_dir, 10)  # 101->104 (out_dir exists), 54-65 (glob members)

  # 2. No targets
  with patch.object(importlib, "import_module"):
    # no members
    with patch.object(inspect, "getmembers", return_value=[]):
      handle_suggest("empty_module.*", None, 10)  # 79-80 (not targets)

  # 3. Invalid api_path (no dot)
  handle_suggest("nodots", None, 10)  # 169 (invalid path format)

  # 4. inspect.signature ValueError/TypeError (144-146)
  handle_suggest("math.sqrt", None, 10)

  # 5. Extract metadata with C-extension Exception
  with patch.object(importlib, "import_module"):
    with patch.object(inspect, "getmembers", return_value=[("func", MagicMock())]):
      with patch.object(suggest_mod, "_extract_metadata", side_effect=Exception("Failed")):
        res = handle_suggest("test.*", None, 10)  # 64-65 (except Exception: continue)
  assert res == 1
