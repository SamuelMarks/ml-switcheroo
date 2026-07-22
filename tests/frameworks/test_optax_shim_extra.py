"""Test suite for the Optax Shim Extra module."""

import sys
import importlib
from unittest.mock import patch


def test_optax_shim_import_error():
  """Verifies the behavior of optax shim import correctly handling an error."""
  with patch.dict(sys.modules, {"optax": None}):
    import ml_switcheroo.frameworks.common.optax_shim as optax_shim

    importlib.reload(optax_shim)
    assert optax_shim.optax is None
  importlib.reload(optax_shim)
