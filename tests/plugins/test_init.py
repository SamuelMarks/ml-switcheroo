"""Test suite for the Init module."""

from pathlib import Path
from unittest.mock import patch


def test_plugins_init_discovery(tmp_path: Path) -> None:
  """Verifies that importing ml_switcheroo.plugins delegates to load_plugins and skips helper files.

  Args:
      tmp_path: Pytest temporary directory fixture.
  """
  import importlib

  import ml_switcheroo.core.hooks_registry as hr
  import ml_switcheroo.plugins as plugins_pkg

  (tmp_path / "_protected.py").write_text("x = 1\n", encoding="utf-8")
  (tmp_path / "some_utils.py").write_text("x = 1\n", encoding="utf-8")
  (tmp_path / "my_helpers.py").write_text("x = 1\n", encoding="utf-8")
  (tmp_path / "valid_plugin.py").write_text("x = 1\n", encoding="utf-8")
  (tmp_path / "broken_plugin.py").write_text("raise RuntimeError('broken')\n", encoding="utf-8")

  with patch("ml_switcheroo.core.hooks_registry.load_plugins", wraps=hr.load_plugins) as mock_load:
    importlib.reload(plugins_pkg)
    mock_load.assert_called_once()
    assert hr._PLUGINS_LOADED is True

  with patch("logging.Logger.warning") as mock_warn:
    loaded = hr._import_from_dir(tmp_path, base_package=None)
    assert loaded == 1
    mock_warn.assert_called_once()
