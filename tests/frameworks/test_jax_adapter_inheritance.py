"""
Tests for Split JAX / Flax NNX Adapters.

Verifies that:
1. `JaxCoreAdapter` (key='jax') provides Level 0/1 support but NO neural traits.
2. `FlaxNNXAdapter` (key='flax_nnx') provides Level 2 support and inherits Core logic.
3. Wiring logic handles the separation correctly.
"""

import pytest
from ml_switcheroo.frameworks.jax import JaxCoreAdapter
from ml_switcheroo.frameworks.flax_nnx import FlaxNNXAdapter


def test_core_adapter_scope():
  """Verify Jax Core adapter is limited to Math/Opt."""
  adapter = JaxCoreAdapter()

  # 1. Structural Traits should correspond to pure functions
  assert adapter.structural_traits.module_base is None
  assert len(adapter.structural_traits.inject_magic_args) == 0

  # 2. Wiring should set up Math/Optax but NOT state_dict logic
  snap = {}
  adapter.apply_wiring(snap)
  mappings = snap["mappings"]

  assert mappings["Abs"]["api"] == "jnp.abs"
  assert mappings["Adam"]["api"] == "optax.adam"

  # Should NOT have neural container mappings
  assert "state_dict" not in mappings


def test_flax_nnx_inheritance():
  """Verify Flax NNX adapter inherits logic and adds Neural layer."""
  adapter = FlaxNNXAdapter()

  # 1. Check Inheritance Metadata
  assert adapter.inherits_from == "jax"

  # 2. Traits must be Neural (NNX)
  assert adapter.structural_traits.module_base == "nnx.Module"
  assert "rngs" in adapter.structural_traits.inject_magic_args[0]

  # 3. Wiring should be additive (Stack + NNX)
  snap = {}
  adapter.apply_wiring(snap)
  mappings = snap["mappings"]

  # Inherited L0 (Math)
  assert mappings["Abs"]["api"] == "jnp.abs"

  # Inherited L1 (Optax explicitly re-wired via stack mixin)
  assert mappings["Adam"]["api"] == "optax.adam"

  # Specific L2 (State)
  assert mappings["state_dict"]["requires_plugin"] == "torch_state_dict_to_nnx"


def test_discovery_segmentation():
  """Verify adapters look for different things."""
  core = JaxCoreAdapter()
  flax = FlaxNNXAdapter()

  # Core should not look for Linen/NNX
  assert "flax.nnx" not in core.search_modules

  # Flax should look for both (or at least NNX)
  assert "flax.nnx" in flax.search_modules
  assert "jax.numpy" in flax.search_modules
