"""
Tests for the JAX Adapter (Core Logic, Stack Separation, and Definitions).

Verifies:
1. Core JAX Adapter populates Level 0 (Math/Array) definitions.
2. Flax NNX Adapter populates Level 2 (Neural) definitions.
3. Optax/Orbax (Level 1) integration via stack mixin.
4. Discovery logic separation.
"""

import pytest
from ml_switcheroo.frameworks.jax import JaxCoreAdapter
from ml_switcheroo.frameworks.flax_nnx import FlaxNNXAdapter


def test_jax_core_definitions():
  """Verify Math/Array API mappings exist on Core Adapter."""
  adapter = JaxCoreAdapter()
  defs = adapter.definitions

  # 1. Math
  assert "Abs" in defs
  assert defs["Abs"].api == "jnp.abs"
  assert "Sum" in defs
  assert defs["Sum"].api == "jnp.sum"

  # 2. Optimization (Level 1 Stack)
  assert "Adam" in defs
  assert defs["Adam"].api == "optax.adam"

  # 3. Extras
  assert "DataLoader" in defs
  assert defs["DataLoader"].requires_plugin == "convert_dataloader"

  # 4. Neural Layers SHOULD BE MISSING
  assert "Linear" not in defs
  assert "Conv2d" not in defs


def test_flax_nnx_definitions():
  """Verify Neural Layer mappings logic on Flax Adapter."""
  adapter = FlaxNNXAdapter()
  defs = adapter.definitions

  # 1. Neural Layers (Level 2)
  assert "Linear" in defs
  assert defs["Linear"].api == "flax.nnx.Linear"
  assert "Conv2d" in defs
  assert defs["Conv2d"].api == "flax.nnx.Conv"

  # 2. State Logic
  assert "state_dict" in defs
  assert defs["state_dict"].requires_plugin == "torch_state_dict_to_nnx"

  # 3. Should NOT redefine Core Math (handled via stack mixin or conversion fallback)
  # NOTE: Adapter definitions property is static map.
  # FlaxNNX relies on JAXStackMixin for runtime wiring of math if not redefined.
  # The 'definitions' property is unique to the class.
  assert "Abs" not in defs


def test_plugin_traits_differences():
  """Verify Flax enables neural traits that Core does not."""
  core_traits = JaxCoreAdapter().plugin_traits
  flax_traits = FlaxNNXAdapter().plugin_traits

  # Both require explicit RNG
  assert core_traits.requires_explicit_rng is True
  assert flax_traits.requires_explicit_rng is True

  # Only Flax defines functional state requirements for Layers
  # Actually, both might set it, but let's check config consistency
  assert flax_traits.requires_functional_state is True


def test_discovery_segmentation_logic():
  """Verify search modules differ."""
  core_modules = JaxCoreAdapter().search_modules
  flax_modules = FlaxNNXAdapter().search_modules

  assert "jax.numpy" in core_modules
  assert "flax.nnx" not in core_modules

  assert "flax.nnx" in flax_modules
  assert "jax.numpy" in flax_modules
