"""Test suite for the Optax Shim module."""

import importlib
import sys
from unittest.mock import MagicMock, patch

import pytest


def test_optax_shim_lazy_resolution() -> None:
  """Verifies optax_shim lazy attribute resolution and deprecation warnings."""
  # Mock optax module
  mock_optax = MagicMock(spec=["adam"])
  mock_optax.adam = "adam_optimizer"

  with patch.dict(sys.modules, {"optax": mock_optax}):
    import ml_switcheroo.frameworks.common.optax_shim as shim

    importlib.reload(shim)

    # Test successful proxy with warning
    with pytest.warns(DeprecationWarning, match="Accessing optax attributes via optax_shim.adam is deprecated"):
      res = shim.adam
      assert res == "adam_optimizer"

    # Test missing attribute raises AttributeError
    with pytest.raises(AttributeError, match="has no attribute 'missing_attr'"):
      _ = shim.missing_attr


def test_optax_shim_import_error() -> None:
  """Verifies the behavior of optax shim import correctly handling an error."""
  with patch.dict(sys.modules, {"optax": None}):  # type: ignore
    import ml_switcheroo.frameworks.common.optax_shim as shim

    importlib.reload(shim)
    assert getattr(shim, "optax", None) is None

    # Test adam fallback when optax is missing
    with pytest.warns(DeprecationWarning, match="optax_shim.adam is deprecated"):
      with pytest.raises(AttributeError, match="has no attribute 'adam'"):
        _ = shim.adam

  importlib.reload(shim)
