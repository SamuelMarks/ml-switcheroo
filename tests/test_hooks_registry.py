"""Test suite for the Hooks Registry module."""

from unittest.mock import patch
from pathlib import Path
from ml_switcheroo.core.hooks_registry import (
  register_hook,
  get_hook,
  get_all_hook_metadata,
  clear_hooks,
  load_plugins,
  _import_from_dir,
  _HOOKS,
  _HOOK_METADATA,
)


def test_clear_hooks():
  """Verifies the behavior of clear hooks."""
  _HOOKS["test"] = lambda: None
  _HOOK_METADATA["test"] = "meta"
  import ml_switcheroo.core.hooks_registry as hr

  hr._PLUGINS_LOADED = True
  clear_hooks()
  assert len(_HOOKS) == 0
  assert len(_HOOK_METADATA) == 0
  assert hr._PLUGINS_LOADED is False


def test_register_hook():
  """Verifies the behavior of register hook."""
  clear_hooks()

  @register_hook("my_hook")
  def dummy():
    pass

  assert "my_hook" in _HOOKS
  assert "my_hook" not in _HOOK_METADATA


def test_register_hook_auto_wire():
  """Verifies the behavior of register hook auto wire."""
  clear_hooks()

  @register_hook("my_hook", auto_wire={"operation": "Op", "description": "doc", "variants": {}})
  def dummy():
    pass

  assert "my_hook" in _HOOKS
  assert "my_hook" in _HOOK_METADATA
  assert _HOOK_METADATA["my_hook"].operation == "Op"


def test_get_hook_lazy_load():
  """Gets hook lazy load."""
  clear_hooks()
  import ml_switcheroo.core.hooks_registry as hr

  hr._PLUGINS_LOADED = False
  with patch("ml_switcheroo.core.hooks_registry.load_plugins") as mock_load:
    get_hook("nonexistent")
    mock_load.assert_called_once()
    assert hr._PLUGINS_LOADED is True


def test_get_all_metadata():
  """Gets all metadata."""
  clear_hooks()

  @register_hook("my_hook", auto_wire={"operation": "Op", "description": "doc", "variants": {}})
  def dummy():
    pass

  meta = get_all_hook_metadata()
  assert "my_hook" in meta


def test_load_plugins_default(tmp_path):
  """Loads plugins default."""
  count = load_plugins(plugins_dir=Path("does_not_exist"))
  assert count == 0


def test_import_from_dir_exception(tmp_path):
  """Verifies the behavior of import from a directory correctly handling an exception."""
  bad_dir = tmp_path / "bad"
  bad_dir.mkdir()
  (bad_dir / "bad_mod.py").write_text("invalid python code [")
  with patch("logging.Logger.warning") as mock_warn:
    count = _import_from_dir(bad_dir, base_package=None)
    assert count == 0
    mock_warn.assert_called()


def test_import_from_dir_reload(tmp_path):
  """Verifies the behavior of import from a directory reload."""
  mod_dir = tmp_path / "mods"
  mod_dir.mkdir()
  (mod_dir / "good_mod.py").write_text("x = 1")
  _import_from_dir(mod_dir, base_package=None)
  count = _import_from_dir(mod_dir, base_package=None)
  assert count == 1
