"""Test suite for the Hooks Context module."""

import pytest
from unittest.mock import MagicMock
from ml_switcheroo.core.hooks import HookContext
from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo.semantics.manager import SemanticsManager


@pytest.fixture
def mock_semantics():
  """Provides a mock semantics for testing."""
  mgr = MagicMock(spec=SemanticsManager)
  data = {
    "add": {"std_args": ["x1", "x2"], "variants": {"jax": {"api": "jax.numpy.add"}, "numpy": {"api": "numpy.add"}}},
    "abs": {"std_args": [("x", "Array")], "variants": {}},
    "complex": {"variants": {"jax": {"requires_plugin": "magic"}}},
  }

  def resolve(aid, fw):
    """Resolves ."""
    if aid in data and fw in data[aid]["variants"]:
      return data[aid]["variants"][fw]
    return None

  mgr.resolve_variant.side_effect = resolve
  mgr.get_definition_by_id.side_effect = lambda aid: data.get(aid)
  mgr.get_framework_config.return_value = {}
  return mgr


def test_lookup_api_success(mock_semantics):
  """Verifies the behavior of lookup API successfully."""
  config = RuntimeConfig(target_framework="jax")
  ctx = HookContext(mock_semantics, config)
  result = ctx.lookup_api("add")
  assert result == "jax.numpy.add"


def test_lookup_api_different_target(mock_semantics):
  """Verifies the behavior of lookup API different target."""
  config = RuntimeConfig(target_framework="numpy")
  ctx = HookContext(mock_semantics, config)
  result = ctx.lookup_api("add")
  assert result == "numpy.add"


def test_lookup_api_missing_variant(mock_semantics):
  """Verifies the behavior of lookup API missing variant."""
  config = RuntimeConfig(target_framework="tensorflow")
  ctx = HookContext(mock_semantics, config)
  result = ctx.lookup_api("add")
  assert result is None


def test_lookup_api_missing_op(mock_semantics):
  """Verifies the behavior of lookup API missing op."""
  config = RuntimeConfig(target_framework="jax")
  ctx = HookContext(mock_semantics, config)
  result = ctx.lookup_api("unknown_logic")
  assert result is None


def test_lookup_api_plugin_variant(mock_semantics):
  """Verifies the behavior of lookup API plugin variant."""
  config = RuntimeConfig(target_framework="jax")
  ctx = HookContext(mock_semantics, config)
  result = ctx.lookup_api("complex")
  assert result is None


def test_lookup_signature_standard_list(mock_semantics):
  """Verifies the behavior of lookup signature standard list."""
  config = RuntimeConfig(target_framework="jax")
  ctx = HookContext(mock_semantics, config)
  sig = ctx.lookup_signature("add")
  assert sig == ["x1", "x2"]


def test_lookup_signature_typed_tuples(mock_semantics):
  """Verifies the behavior of lookup signature typed tuples."""
  config = RuntimeConfig(target_framework="jax")
  ctx = HookContext(mock_semantics, config)
  sig = ctx.lookup_signature("abs")
  assert sig == ["x"]


def test_lookup_signature_unknown_returns_empty(mock_semantics):
  """Verifies the behavior of lookup signature unknown returns empty."""
  config = RuntimeConfig(target_framework="jax")
  ctx = HookContext(mock_semantics, config)
  sig = ctx.lookup_signature("ghost_op")
  assert sig == []
