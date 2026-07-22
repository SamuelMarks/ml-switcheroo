"""Test suite for the Init Discovery module."""

import pytest
from unittest.mock import MagicMock, patch
import ml_switcheroo.frameworks as frameworks_pkg


def test_available_frameworks_reflects_registry():
  """Verifies the behavior of available frameworks reflects registry."""
  mock_registry = {"mock_fw_1": MagicMock(), "mock_fw_2": MagicMock()}
  with patch("ml_switcheroo.frameworks.base._ADAPTER_REGISTRY", mock_registry):
    fws = frameworks_pkg.available_frameworks()
  assert "mock_fw_1" in fws
  assert "mock_fw_2" in fws
  assert len(fws) == 2


def test_auto_discovery_logic():
  """Verifies the behavior of auto discovery logic."""
  mock_modules = [(None, "base", False), (None, "tinygrad", False), (None, "custom_lib", False)]
  with patch("pkgutil.iter_modules", return_value=mock_modules):
    with patch("importlib.import_module") as mock_import:
      frameworks_pkg._auto_register_adapters()
      with pytest.raises(AssertionError):
        mock_import.assert_any_call(".base", package="ml_switcheroo.frameworks")
      mock_import.assert_any_call(".tinygrad", package="ml_switcheroo.frameworks")
      mock_import.assert_any_call(".custom_lib", package="ml_switcheroo.frameworks")


def test_broken_module_handling(capsys):
  """Verifies the behavior of broken module handling."""
  mock_modules = [(None, "broken_adapter", False)]
  with patch("pkgutil.iter_modules", return_value=mock_modules):
    with patch("importlib.import_module", side_effect=ImportError("Missing dependency")):
      try:
        frameworks_pkg._auto_register_adapters()
      except Exception as e:
        pytest.fail(f"Discovery crashed on broken module: {e}")


def test_helpers_are_exported():
  """Verifies the behavior of helpers are exported."""
  assert "get_adapter" in frameworks_pkg.__all__
  assert "register_framework" in frameworks_pkg.__all__
  assert callable(frameworks_pkg.get_adapter)
