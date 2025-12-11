"""
Tests for JAX Ecosystem Crawlers (Optax, Flax, JAX).

Verifies:
1. OptaxScanner finds losses and optimizers.
2. FlaxScanner finds layers.
3. JaxAdapter delegates correctly.
"""

import sys
import types
import pytest
from unittest.mock import MagicMock, patch

from ml_switcheroo.frameworks.jax import JaxAdapter
from ml_switcheroo.frameworks.optax_shim import OptaxScanner
from ml_switcheroo.frameworks.flax_shim import FlaxScanner
from ml_switcheroo.frameworks.base import StandardCategory, GhostRef

# --- Mocks ---


def create_mock_module(name):
  return types.ModuleType(name)


@pytest.fixture
def mock_jax_env():
  """Sets up sys.modules with fake jax, optax, flax."""

  # 1. OPTAX
  # Mock optimizers (functions)
  # Must correctly set __name__ so usage of inspect.getmembers -> GhostInspector works
  def mock_adam(learning_rate):
    pass

  mock_adam.__name__ = "adam"

  def mock_sgd(learning_rate):
    pass

  mock_sgd.__name__ = "sgd"

  # Mock losses module
  mock_losses_mod = create_mock_module("optax.losses")

  def mock_l2_loss(p, t):
    pass

  mock_l2_loss.__name__ = "l2_loss"
  mock_losses_mod.l2_loss = mock_l2_loss

  mock_optax = create_mock_module("optax")
  mock_optax.adam = mock_adam
  mock_optax.sgd = mock_sgd
  mock_optax.losses = mock_losses_mod

  # 2. FLAX
  class Module:
    pass

  class Dense(Module):
    def __init__(self, features):
      pass

  Dense.__name__ = "Dense"

  class Conv(Module):
    pass

  Conv.__name__ = "Conv"

  mock_linen = create_mock_module("flax.linen")
  mock_linen.Module = Module
  mock_linen.Dense = Dense
  mock_linen.Conv = Conv

  mock_flax = create_mock_module("flax")
  mock_flax.linen = mock_linen

  # 3. JAX
  mock_jax = create_mock_module("jax")
  mock_jax.nn = create_mock_module("jax.nn")

  def mock_relu(x):
    pass

  mock_relu.__name__ = "relu"
  mock_jax.nn.relu = mock_relu

  overrides = {
    "optax": mock_optax,
    "optax.losses": mock_losses_mod,
    "flax": mock_flax,
    "flax.linen": mock_linen,
    "jax": mock_jax,
    "jax.nn": mock_jax.nn,
  }

  with patch.dict(sys.modules, overrides):
    # Must reload modules that import these at top level or in try blocks
    import importlib
    import ml_switcheroo.frameworks.optax_shim
    import ml_switcheroo.frameworks.flax_shim
    import ml_switcheroo.frameworks.jax

    importlib.reload(ml_switcheroo.frameworks.optax_shim)
    importlib.reload(ml_switcheroo.frameworks.flax_shim)
    importlib.reload(ml_switcheroo.frameworks.jax)

    yield ml_switcheroo.frameworks.jax.JaxAdapter()


def test_optax_shim_optimizers(mock_jax_env):
  """Verify scanning of optax optimizers."""
  # Use shim directly
  from ml_switcheroo.frameworks.optax_shim import OptaxScanner

  results = OptaxScanner.scan_optimizers()
  names = [r.name for r in results]

  assert "adam" in names
  assert "sgd" in names

  # Check ghost ref properties
  adam_ref = next(r for r in results if r.name == "adam")
  assert adam_ref.has_arg("learning_rate")


def test_optax_shim_losses(mock_jax_env):
  """Verify scanning of optax losses."""
  from ml_switcheroo.frameworks.optax_shim import OptaxScanner

  results = OptaxScanner.scan_losses()
  names = [r.name for r in results]

  assert "l2_loss" in names


def test_flax_shim_layers(mock_jax_env):
  """Verify scanning of flax layers."""
  from ml_switcheroo.frameworks.flax_shim import FlaxScanner

  results = FlaxScanner.scan_layers()
  names = [r.name for r in results]

  assert "Dense" in names
  assert "Conv" in names
  assert "Module" not in names  # Base class skipped


def test_jax_adapter_integration(mock_jax_env):
  """Verify JaxAdapter calls shims."""
  adapter = mock_jax_env

  # LOSS -> Optax
  losses = adapter.collect_api(StandardCategory.LOSS)
  assert any(r.name == "l2_loss" for r in losses)

  # LAYER -> Flax
  layers = adapter.collect_api(StandardCategory.LAYER)
  assert any(r.name == "Dense" for r in layers)

  # ACTIVATION -> JAX NN
  acts = adapter.collect_api(StandardCategory.ACTIVATION)
  assert any(r.name == "relu" for r in acts)


def test_optax_missing_safe_fallback():
  """Verify empty return if optax not installed."""
  with patch.dict(sys.modules, {"optax": None}):
    import importlib
    import ml_switcheroo.frameworks.optax_shim

    importlib.reload(ml_switcheroo.frameworks.optax_shim)

    from ml_switcheroo.frameworks.optax_shim import OptaxScanner

    assert OptaxScanner.scan_optimizers() == []
