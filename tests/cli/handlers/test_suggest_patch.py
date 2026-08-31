"""Docstring for test_suggest_patch module."""


def test_suggest_missing_branches(tmp_path):
  """Docstring."""
  from unittest.mock import MagicMock, patch
  from ml_switcheroo.cli.handlers.suggest import handle_suggest

  # 1. glob mode (.*) with some members and no members
  out_dir = tmp_path / "suggest_out"
  out_dir.mkdir(parents=True, exist_ok=True)

  handle_suggest("math.*", out_dir, 10)  # 101->104 (out_dir exists), 54-65 (glob members)

  # 2. No targets
  with patch("importlib.import_module"):
    # no members
    with patch("inspect.getmembers", return_value=[]):
      handle_suggest("empty_module.*", None, 10)  # 79-80 (not targets)

  # 3. Invalid api_path (no dot)
  handle_suggest("nodots", None, 10)  # 169 (invalid path format)

  # 4. inspect.signature ValueError/TypeError (144-146)
  handle_suggest("math.sqrt", None, 10)

  # 5. Extract metadata with C-extension Exception
  with patch("importlib.import_module"):
    with patch("inspect.getmembers", return_value=[("func", MagicMock())]):
      with patch("ml_switcheroo.cli.handlers.suggest._extract_metadata", side_effect=Exception("Failed")):
        handle_suggest("test.*", None, 10)  # 64-65 (except Exception: continue)
