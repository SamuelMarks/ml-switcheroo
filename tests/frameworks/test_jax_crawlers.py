"""
Tests for JAX Ecosystem Crawlers (Optax, Flax, JAX).
"""

import sys
import types
import pytest
import importlib
from unittest.mock import MagicMock, patch

# Fix: Import specific adapter that handles layers
from ml_switcheroo.frameworks.flax_nnx import FlaxNNXAdapter


def create_mock_module(name):
  return types.ModuleType(name)


@pytest.fixture
def mock_jax_env():
  """Sets up sys.modules with fake jax, optax, flax."""

  mock_losses_mod = create_mock_module("optax.losses")

  def mock_l2_loss(p, t):
    pass

  mock_l2_loss.__name__ = "l2_loss"
  mock_losses_mod.l2_loss = mock_l2_loss

  mock_optax = create_mock_module("optax")
  mock_optax.losses = mock_losses_mod

  mock_linen = create_mock_module("flax.linen")

  class Module:
    pass

  Module.__name__ = "Module"
  mock_linen.Module = Module

  class Dense(Module):
    pass

  Dense.__name__ = "Dense"
  mock_linen.Dense = Dense

  mock_flax = create_mock_module("flax")
  mock_flax.linen = mock_linen
  # We also mock flax.nnx as checking for it is part of FlaxNNXAdapter logic
  mock_flax.nnx = create_mock_module("flax.nnx")

  mock_jax = create_mock_module("jax")
  mock_jax.nn = create_mock_module("jax.nn")
  mock_jax.numpy = create_mock_module("jax.numpy")

  def relu(x):
    pass

  relu.__name__ = "relu"
  mock_jax.nn.relu = relu

  overrides = {
    "optax": mock_optax,
    "optax.losses": mock_losses_mod,
    "flax": mock_flax,
    "flax.linen": mock_linen,
    "flax.nnx": mock_flax.nnx,
    "jax": mock_jax,
    "jax.nn": mock_jax.nn,
    "jax.numpy": mock_jax.numpy,
  }

  with patch.dict(sys.modules, overrides):
    # Force reload modules
    import ml_switcheroo.frameworks.optax_shim
    import ml_switcheroo.frameworks.flax_shim
    import ml_switcheroo.frameworks.jax

    importlib.reload(ml_switcheroo.frameworks.optax_shim)
    importlib.reload(ml_switcheroo.frameworks.flax_shim)
    importlib.reload(ml_switcheroo.frameworks.jax)

    with patch("inspect.getmembers") as mock_members:

      def get_members(obj):
        if obj == mock_losses_mod:
          return [("l2_loss", mock_l2_loss)]
        if obj == mock_linen:
          # Return Module too so it behaves like real module
          return [("Dense", Dense), ("Module", Module)]
        if obj == mock_jax.nn:
          return [("relu", relu)]
        return []

      mock_members.side_effect = get_members
      # We explicitly use FlaxNNXAdapter which is capable of scanning layers
      yield FlaxNNXAdapter()


def test_jax_adapter_integration(mock_jax_env):
  """Verify FlaxNNXAdapter calls shims correctly."""
  adapter = mock_jax_env
  from ml_switcheroo.frameworks.base import StandardCategory

  # LOSS -> Optax (inherited from core logic)
  losses = adapter.collect_api(StandardCategory.LOSS)
  names = [r.name for r in losses]
  assert "l2_loss" in names, f"Expected l2_loss, found {names}"

  # LAYER -> Flax (specific to Flax adapter)
  layers = adapter.collect_api(StandardCategory.LAYER)
  layer_names = [r.name for r in layers]
  assert "Dense" in layer_names
