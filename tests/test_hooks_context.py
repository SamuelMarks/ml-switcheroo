"""
Tests for Extended HookContext Functionality.

Verifies that:
1. lookup_api queries the semantics manager correctly.
2. lookup_signature retrieves argument lists properly.
3. HookContext handles missing keys logic gracefully.
"""

import pytest
from ml_switcheroo.core.hooks import HookContext
from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo.semantics.manager import SemanticsManager


class MockSemantics(SemanticsManager):
  def __init__(self):
    # Override to provide deterministic data
    self.data = {
      "add": {
        "std_args": ["x1", "x2"],
        "variants": {"jax": {"api": "jax.numpy.add"}, "numpy": {"api": "numpy.add"}, "torch": {"api": "torch.add"}},
      },
      "abs": {
        "std_args": [("x", "Array")],  # Typed tuple format
        "variants": {},
      },
      "complex": {
        "variants": {
          "jax": {"requires_plugin": "magic"}
          # No explicit API string
        }
      },
    }

  def get_known_apis(self):
    return self.data


@pytest.fixture
def mock_semantics():
  return MockSemantics()


def test_lookup_api_success(mock_semantics):
  """Verify lookup returns the correct API string for JAX."""
  config = RuntimeConfig(target_framework="jax")
  ctx = HookContext(mock_semantics, config)

  result = ctx.lookup_api("add")
  assert result == "jax.numpy.add"


def test_lookup_api_different_target(mock_semantics):
  """Verify lookup uses the configured target framework."""
  config = RuntimeConfig(target_framework="numpy")
  ctx = HookContext(mock_semantics, config)

  result = ctx.lookup_api("add")
  assert result == "numpy.add"


def test_lookup_api_missing_variant(mock_semantics):
  """Verify lookup returns None if target variant is missing."""
  config = RuntimeConfig(target_framework="tensorflow")  # Not in mock
  ctx = HookContext(mock_semantics, config)

  result = ctx.lookup_api("add")
  assert result is None


def test_lookup_api_missing_op(mock_semantics):
  """Verify lookup returns None for unknown operations."""
  config = RuntimeConfig(target_framework="jax")
  ctx = HookContext(mock_semantics, config)

  result = ctx.lookup_api("unknown_logic")
  assert result is None


def test_lookup_api_plugin_variant(mock_semantics):
  """
  Verify lookup returns None if the variant exists but has no 'api' key.
  (e.g., pure plugin logic).
  """
  config = RuntimeConfig(target_framework="jax")
  ctx = HookContext(mock_semantics, config)

  result = ctx.lookup_api("complex")
  assert result is None


def test_lookup_signature_standard_list(mock_semantics):
  """Verify lookup returns standard arguments list."""
  config = RuntimeConfig(target_framework="jax")
  ctx = HookContext(mock_semantics, config)

  sig = ctx.lookup_signature("add")
  assert sig == ["x1", "x2"]


def test_lookup_signature_typed_tuples(mock_semantics):
  """Verify tuples [('x', 'type')] are flattened to ['x']."""
  config = RuntimeConfig(target_framework="jax")
  ctx = HookContext(mock_semantics, config)

  sig = ctx.lookup_signature("abs")
  assert sig == ["x"]


def test_lookup_signature_unknown_returns_empty(mock_semantics):
  """Verify unknown op returns empty list."""
  config = RuntimeConfig(target_framework="jax")
  ctx = HookContext(mock_semantics, config)

  sig = ctx.lookup_signature("ghost_op")
  assert sig == []
