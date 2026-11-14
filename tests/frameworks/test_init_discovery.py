"""Test suite for the Init Discovery module."""

from unittest.mock import MagicMock, patch

import pytest

import ml_switcheroo.frameworks as frameworks_pkg


def test_available_frameworks_reflects_registry() -> None:
  """Verify `frameworks_pkg.available_frameworks` emits `mock_fw_1` for reflects registry."""
  from ml_switcheroo.frameworks.base import get_default_adapter_registry

  mock_registry = {"mock_fw_1": MagicMock(), "mock_fw_2": MagicMock()}
  with patch.dict(get_default_adapter_registry().adapters, mock_registry, clear=True):
    fws: list[str] = frameworks_pkg.available_frameworks()
  assert "mock_fw_1" in fws
  assert "mock_fw_2" in fws
  assert len(fws) == 2


def test_auto_discovery_logic() -> None:
  """Verify `frameworks_pkg._auto_register_adapters` raises `AssertionError` during auto discovery logic."""
  mock_modules = [
    (None, "base", False),
    (None, "tinygrad", False),
    (None, "custom_lib", False),
  ]
  with patch("pkgutil.iter_modules", return_value=mock_modules), patch("importlib.import_module") as mock_import:
    frameworks_pkg._auto_register_adapters()
    with pytest.raises(AssertionError):
      mock_import.assert_any_call(".base", package="ml_switcheroo.frameworks")
    mock_import.assert_any_call(".tinygrad", package="ml_switcheroo.frameworks")
    mock_import.assert_any_call(".custom_lib", package="ml_switcheroo.frameworks")


def test_broken_module_handling(capsys: pytest.CaptureFixture[str]) -> None:
  """Verify `frameworks_pkg._auto_register_adapters` handles broken module handling as expected.

  Args:
      capsys: Pytest stdout and stderr capture fixture.
  """
  mock_modules = [(None, "broken_adapter", False)]
  with (
    patch("pkgutil.iter_modules", return_value=mock_modules),
    patch("importlib.import_module", side_effect=ImportError("Missing dependency")) as mock_imp,
  ):
    try:
      frameworks_pkg._auto_register_adapters()
    except Exception as e:
      pytest.fail(f"Discovery crashed on broken module: {e}")
    mock_imp.assert_called_once_with(".broken_adapter", package="ml_switcheroo.frameworks")


def test_helpers_are_exported() -> None:
  """Verify `callable` emits `get_adapter` for helpers are exported."""
  assert "get_adapter" in frameworks_pkg.__all__
  assert "register_framework" in frameworks_pkg.__all__
  assert callable(frameworks_pkg.get_adapter)
