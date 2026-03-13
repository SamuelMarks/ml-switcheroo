import sys
from unittest.mock import patch, MagicMock

import ml_switcheroo.frameworks.optax_shim as optax_shim
from ml_switcheroo.frameworks.optax_shim import OptaxScanner


def test_optax_shim_no_optax():
  with patch("ml_switcheroo.frameworks.optax_shim.optax", None):
    assert OptaxScanner.scan_optimizers() == []
    assert OptaxScanner.scan_losses() == []


def test_optax_shim_optimizers():
  mock_optax = MagicMock()

  # create some mock functions
  def mock_adam():
    pass

  def mock_custom_optimizer():
    pass

  def mock_internal():
    pass

  # assign names
  mock_adam.__name__ = "adam"
  mock_custom_optimizer.__name__ = "custom_optimizer"
  mock_internal.__name__ = "_internal"

  mock_optax.adam = mock_adam
  mock_optax.custom_optimizer = mock_custom_optimizer
  mock_optax._internal = mock_internal

  # simulate exception inside GhostInspector
  def mock_fail():
    pass

  mock_fail.__name__ = "sgd"
  mock_optax.sgd = mock_fail

  import inspect

  # We need to mock getmembers to return our specific functions
  members = [
    ("adam", mock_adam),
    ("custom_optimizer", mock_custom_optimizer),
    ("_internal", mock_internal),
    ("sgd", mock_fail),
  ]

  with (
    patch("ml_switcheroo.frameworks.optax_shim.optax", mock_optax),
    patch("inspect.getmembers", return_value=members),
    patch("ml_switcheroo.core.ghost.GhostInspector.inspect") as mock_inspect,
  ):
    # sgd will raise an exception
    def side_effect(obj, name):
      if name == "optax.sgd":
        raise Exception("Test error")
      return "GhostRef_Mock"

    mock_inspect.side_effect = side_effect

    results = OptaxScanner.scan_optimizers()

    # adam and custom_optimizer should be scanned and return mock ref
    assert len(results) == 2
    assert results == ["GhostRef_Mock", "GhostRef_Mock"]


def test_optax_shim_losses():
  mock_optax = MagicMock()
  mock_losses = MagicMock()
  mock_optax.losses = mock_losses

  def mock_mse_loss():
    pass

  def mock_internal_loss():
    pass

  def mock_entropy():
    pass

  def mock_fail_loss():
    pass

  members = [
    ("mse_loss", mock_mse_loss),
    ("_internal_loss", mock_internal_loss),
    ("cross_entropy", mock_entropy),
    ("focal_error", mock_fail_loss),
  ]

  with (
    patch("ml_switcheroo.frameworks.optax_shim.optax", mock_optax),
    patch("inspect.getmembers", return_value=members),
    patch("ml_switcheroo.core.ghost.GhostInspector.inspect") as mock_inspect,
  ):

    def side_effect(obj, name):
      if name == "optax.losses.focal_error":
        raise Exception("Test error")
      return "GhostRef_Mock"

    mock_inspect.side_effect = side_effect

    results = OptaxScanner.scan_losses()

    assert len(results) == 2
    assert results == ["GhostRef_Mock", "GhostRef_Mock"]


def test_optax_shim_import_error():
  # simulate the try block at the top of the file
  import importlib

  # We remove optax from sys.modules temporarily if it exists
  original_optax = sys.modules.pop("optax", None)

  with patch.dict("sys.modules", {"optax": None}):
    # reload optax_shim
    importlib.reload(optax_shim)
    assert optax_shim.optax is None

  if original_optax:
    sys.modules["optax"] = original_optax
  # reload again to restore
  importlib.reload(optax_shim)
