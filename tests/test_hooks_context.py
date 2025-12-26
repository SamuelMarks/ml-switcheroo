"""
Tests for Extended HookContext Functionality.

Verifies that:
1. lookup_api queries the semantics manager correctly.
2. lookup_signature retrieves argument lists properly.
3. HookContext handles missing keys logic gracefully.
"""

import pytest
from unittest.mock import MagicMock
from ml_switcheroo.core.hooks import HookContext
from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo.semantics.manager import SemanticsManager


# Use MagicMock instead of partial class mock to ensure attributes exist
@pytest.fixture
def mock_semantics():
  mgr = MagicMock(spec=SemanticsManager)

  # Data definitions
  data = {
    "add": {
      "std_args": ["x1", "x2"],
      "variants": {"jax": {"api": "jax.numpy.add"}, "numpy": {"api": "numpy.add"}},
    },
    "abs": {
      "std_args": [("x", "Array")],
      "variants": {},
    },
    "complex": {"variants": {"jax": {"requires_plugin": "magic"}}},
  }

  # Wire methods
  def resolve(aid, fw):
    if aid in data and fw in data[aid]["variants"]:
      return data[aid]["variants"][fw]
    return None

  mgr.resolve_variant.side_effect = resolve
  mgr.get_definition_by_id.side_effect = lambda aid: data.get(aid)

  # Ensure attributes accessed by context properties don't crash
  mgr.get_framework_config.return_value = {}

  return mgr


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
