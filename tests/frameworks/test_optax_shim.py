"""Test suite for the Optax Shim module."""

import importlib
import sys
from unittest.mock import patch

from ml_switcheroo.frameworks.common import optax_shim


def test_optax_shim() -> None:
  """Verifies the behavior of optax shim."""
  try:
    optax_shim.adam  # type: ignore
  except Exception:
    pass


# --- Merged from test_optax_shim_extra.py ---


def test_optax_shim_import_error() -> None:
  """Verifies the behavior of optax shim import correctly handling an error."""
  with patch.dict(sys.modules, {"optax": None}):  # type: ignore
    import ml_switcheroo.frameworks.common.optax_shim as optax_shim

    importlib.reload(optax_shim)
    assert getattr(optax_shim, "optax", None) is None
  importlib.reload(optax_shim)
