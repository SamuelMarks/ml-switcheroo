"""Test suite for the Config Extra Py310 module."""

from unittest import mock


def test_config_tomli_import():
  """Verifies the behavior of configuration tomli import."""
  from ml_switcheroo.config import _import_tomllib

  with mock.patch("sys.version_info", (3, 10, 0, "final", 0)):
    with mock.patch.dict("sys.modules", {"tomli": None, "tomllib": None}):
      assert _import_tomllib() is None
  with mock.patch("sys.version_info", (3, 10, 0, "final", 0)):
    mock_tomli = mock.MagicMock()
    with mock.patch.dict("sys.modules", {"tomli": mock_tomli, "tomllib": None}):
      assert _import_tomllib() is mock_tomli

  with mock.patch("sys.version_info", (3, 11, 0, "final", 0)):
    mock_tomllib = mock.MagicMock()
    with mock.patch.dict("sys.modules", {"tomllib": mock_tomllib}):
      assert _import_tomllib() is mock_tomllib
