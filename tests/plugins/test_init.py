"""Test suite for the Init module."""

from unittest.mock import patch


def test_plugins_init_discovery():
  """Verifies the behavior of plugins initialization discovery."""
  import ml_switcheroo.plugins as plugins_pkg
  import importlib

  mock_modules = [
    (None, "_protected", False),
    (None, "some_utils", False),
    (None, "valid_plugin", False),
    (None, "broken_plugin", False),
  ]
  with patch("pkgutil.iter_modules", return_value=mock_modules):
    with patch("importlib.import_module") as mock_import:

      def side_effect(name, package):
        """Effect."""
        if "broken_plugin" in name:
          raise ImportError("mock error")

      mock_import.side_effect = side_effect
      importlib.reload(plugins_pkg)
      mock_import.assert_any_call(".valid_plugin", package="ml_switcheroo.plugins")
      mock_import.assert_any_call(".broken_plugin", package="ml_switcheroo.plugins")
