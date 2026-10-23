"""Test suite for the Hooks Registry module."""

import pathlib

from pathlib import Path
from unittest.mock import patch

from ml_switcheroo.core.hooks import AutoWireSpec
from ml_switcheroo.core.hooks_registry import (
  _HOOK_METADATA,
  _HOOKS,
  _import_from_dir,
  clear_hooks,
  get_all_hook_metadata,
  get_hook,
  load_plugins,
  register_hook,
)


def test_clear_hooks() -> None:
  """Verifies the behavior of clear hooks."""
  getattr(_HOOKS, "__setitem__")("test", lambda: None)
  _HOOK_METADATA["test"] = AutoWireSpec()
  import ml_switcheroo.core.hooks_registry as hr

  hr._PLUGINS_LOADED = True
  clear_hooks()
  assert len(_HOOKS) == 0
  assert len(_HOOK_METADATA) == 0
  assert hr._PLUGINS_LOADED is False


def test_register_hook() -> None:
  """Verifies the behavior of register hook."""
  clear_hooks()

  @register_hook("my_hook")
  def dummy() -> None:
    """Execute dummy helper logic."""
    pass

  assert "my_hook" in _HOOKS
  assert "my_hook" not in _HOOK_METADATA


def test_register_hook_auto_wire() -> None:
  """Verifies the behavior of register hook auto wire."""
  clear_hooks()

  @register_hook("my_hook", auto_wire={"operation": "Op", "description": "doc", "variants": {}})
  def dummy() -> None:
    """Execute dummy helper logic."""
    pass

  assert "my_hook" in _HOOKS
  assert "my_hook" in _HOOK_METADATA
  assert getattr(_HOOK_METADATA["my_hook"], "operation", None) == "Op"


def test_get_hook_lazy_load() -> None:
  """Gets hook lazy load."""
  clear_hooks()
  import ml_switcheroo.core.hooks_registry as hr

  hr._PLUGINS_LOADED = False
  with patch("ml_switcheroo.core.hooks_registry.load_plugins") as mock_load:
    get_hook("nonexistent")
    mock_load.assert_called_once()
    assert hr._PLUGINS_LOADED is True


def test_get_all_metadata() -> None:
  """Gets all metadata."""
  clear_hooks()

  @register_hook("my_hook", auto_wire={"operation": "Op", "description": "doc", "variants": {}})
  def dummy() -> None:
    """Execute dummy helper logic."""
    pass

  meta = get_all_hook_metadata()
  assert "my_hook" in meta


def test_load_plugins_default(tmp_path: pathlib.Path) -> None:
  """Loads plugins default including non-existent extra directory.

  Args:
      tmp_path: Pytest temporary directory path fixture.
  """
  count = load_plugins(plugins_dir=Path("does_not_exist"), extra_dirs=[tmp_path, Path("does_not_exist_extra")])
  assert count == 0


def test_import_from_dir_exception(tmp_path: pathlib.Path) -> None:
  """Verifies the behavior of import from a directory correctly handling an exception.

  Args:
      tmp_path: Pytest temporary directory path fixture.
  """
  bad_dir = tmp_path / "bad"
  bad_dir.mkdir()
  (bad_dir / "bad_mod.py").write_text("invalid python code [")
  with patch("logging.Logger.warning") as mock_warn:
    count = _import_from_dir(bad_dir, base_package=None)
    assert count == 0
    mock_warn.assert_called()


def test_import_from_dir_reload(tmp_path: pathlib.Path) -> None:
  """Verifies the behavior of import from a directory reload.

  Args:
      tmp_path: Pytest temporary directory path fixture.
  """
  import sys

  mod_dir = tmp_path / "mods"
  mod_dir.mkdir()
  (mod_dir / "good_mod.py").write_text("x = 1")
  _import_from_dir(mod_dir, base_package=None)
  count = _import_from_dir(mod_dir, base_package=None)
  assert count == 1

  # Also cover base_package uncached and cached branches
  sys.modules.pop("ml_switcheroo.plugins.in_top_k_plugin", None)
  pkg_dir = Path("src/ml_switcheroo/plugins")
  assert _import_from_dir(pkg_dir, base_package="ml_switcheroo.plugins") > 0
  assert _import_from_dir(pkg_dir, base_package="ml_switcheroo.plugins") > 0
