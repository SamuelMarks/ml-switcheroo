"""
Tests for the JAX Stack Mixin (Level 0/1 Logic).
"""

import pytest
from ml_switcheroo.frameworks.common.jax_stack import JAXStackMixin


class MockHighLevelAdapter(JAXStackMixin):
  """A dummy adapter imitating Flax/Pax behavior."""

  pass


def test_device_syntax_cuda():
  """Verify CUDA maps to GPU backend index 0."""
  adapter = MockHighLevelAdapter()
  syntax = adapter.get_device_syntax("'cuda'")
  assert "jax.devices('gpu')[0]" == syntax


def test_device_syntax_cpu_index():
  """Verify CPU mapping with explicit index."""
  adapter = MockHighLevelAdapter()
  syntax = adapter.get_device_syntax("'cpu'", "1")
  assert "jax.devices('cpu')[1]" == syntax


def test_serialization_syntax_orbax():
  """Verify Orbax syntax generation."""
  adapter = MockHighLevelAdapter()

  # Imports
  assert "import orbax.checkpoint" in adapter.get_serialization_imports()[0]

  # Save
  save_code = adapter.get_serialization_syntax("save", "dir", "model_state")
  assert "PyTreeCheckpointer().save" in save_code
  assert "directory=dir" in save_code
  assert "item=model_state" in save_code

  # Load
  load_code = adapter.get_serialization_syntax("load", "dir")
  assert "PyTreeCheckpointer().restore(dir)" in load_code


def test_wiring_injection():
  """Verify Level 1 Optax/JNP mappings are injected."""
  adapter = MockHighLevelAdapter()
  snapshot = {}

  adapter._apply_stack_wiring(snapshot)

  mappings = snapshot.get("mappings", {})

  # Check Math (L0)
  assert mappings["Abs"]["api"] == "jnp.abs"
  assert mappings["size"]["requires_plugin"] == "method_to_property"

  # Check Optimizer (L1)
  assert mappings["Adam"]["api"] == "optax.adam"
  assert mappings["step"]["requires_plugin"] == "optimizer_step"
