from unittest import mock
import importlib


def test_config_tomli_import():
  # Save the original module to restore it
  import ml_switcheroo.config as orig_config

  with mock.patch("sys.version_info", (3, 10)):
    with mock.patch.dict("sys.modules", {"tomli": None, "tomllib": None}):
      import ml_switcheroo.config

      importlib.reload(ml_switcheroo.config)
      assert ml_switcheroo.config.tomllib is None

  with mock.patch("sys.version_info", (3, 10)):
    with mock.patch.dict("sys.modules", {"tomli": mock.MagicMock(), "tomllib": None}):
      import ml_switcheroo.config

      importlib.reload(ml_switcheroo.config)
      assert ml_switcheroo.config.tomllib is not None

  # Restore it!
  importlib.reload(orig_config)
