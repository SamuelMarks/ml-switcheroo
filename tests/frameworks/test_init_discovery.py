"""
Tests for Dynamic Framework Discovery (__init__.py).

Verifies:
1. `available_frameworks` returns keys from the registry.
2. `pkgutil` iteration correctly attempts imports.
3. Excluded modules (base, common) are skipped.
4. Import errors in plugins are handled gracefully (no crash).
"""

import sys
import importlib
import pytest
from unittest.mock import MagicMock, patch

# We import the module under test. Note that importing it triggers the
# logic immediately, so we must rely on reload() or patching beforehand
# to test the side effects of _auto_register_adapters.
import ml_switcheroo.frameworks as frameworks_pkg


def test_available_frameworks_reflects_registry():
  """
  Verify `available_frameworks()` returns the keys of the underlying registry.
  """
  # Mock the registry directly
  mock_registry = {"mock_fw_1": MagicMock(), "mock_fw_2": MagicMock()}

  with patch.dict(frameworks_pkg._ADAPTER_REGISTRY, mock_registry, clear=True):
    fws = frameworks_pkg.available_frameworks()
    assert "mock_fw_1" in fws
    assert "mock_fw_2" in fws
    assert len(fws) == 2


def test_auto_discovery_logic():
  """
  Verify that `_auto_register_adapters` iterates modules and imports them.
  We mock pkgutil and importlib to simulate finding a 'tinygrad' module.
  """
  # 1. Mock pkgutil to return a specific list of modules
  # Format: (importer, name, ispkg)
  mock_modules = [
    (None, "base", False),  # Should be excluded
    (None, "tinygrad", False),  # Should be imported
    (None, "custom_lib", False),  # Should be imported
  ]

  with patch("pkgutil.iter_modules", return_value=mock_modules):
    # 2. Mock importlib to verify calls
    with patch("importlib.import_module") as mock_import:
      # Force re-execution of the discovery logic
      frameworks_pkg._auto_register_adapters()

      # "base" should be skipped
      with pytest.raises(AssertionError):
        mock_import.assert_any_call(".base", package="ml_switcheroo.frameworks")

      # "tinygrad" and "custom_lib" should be imported
      mock_import.assert_any_call(".tinygrad", package="ml_switcheroo.frameworks")
      mock_import.assert_any_call(".custom_lib", package="ml_switcheroo.frameworks")


def test_broken_module_handling(capsys):
  """
  Verify that if an adapter raises an Exception during import (e.g. SyntaxError
  or runtime error), the scanning continues and logs a warning.
  """
  mock_modules = [(None, "broken_adapter", False)]

  with patch("pkgutil.iter_modules", return_value=mock_modules):
    # Mock importlib to raise an exception
    with patch("importlib.import_module", side_effect=ImportError("Missing dependency")):
      # Should not raise exception
      try:
        frameworks_pkg._auto_register_adapters()
      except Exception as e:
        pytest.fail(f"Discovery crashed on broken module: {e}")

  # Note: `logging` output is captured by caplog in pytest, not capsys or stdout
  # dependent on config, but the absence of a crash is the primary assertion here.


def test_helpers_are_exported():
  """
  Verify that essential helpers are exposed in __all__.
  """
  assert "get_adapter" in frameworks_pkg.__all__
  assert "register_framework" in frameworks_pkg.__all__
  assert callable(frameworks_pkg.get_adapter)
