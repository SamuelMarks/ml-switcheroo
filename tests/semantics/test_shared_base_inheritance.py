"""Test suite for the Shared Base Inheritance module."""

from unittest.mock import patch
import pytest
from ml_switcheroo.semantics.manager import SemanticsManager


class MockAdapterWithParent:
  """Mock Adapter With Parent class for testing purposes."""

  def __init__(self, parent):
    """Initializes the MockAdapterWithParent instance."""
    self.inherits_from = parent


@pytest.fixture
def manager():
  """Provides a mock manager for testing."""
  mgr = SemanticsManager()
  mgr._reverse_index = {}
  mgr.data = {}
  return mgr


def test_immediate_fallback(manager):
  """Verifies the behavior of immediate fallback."""
  manager.data["abs"] = {"variants": {"jax": {"api": "jax.numpy.abs"}}}
  mock_adapter = MockAdapterWithParent("jax")
  with patch("ml_switcheroo.semantics.manager.get_adapter", return_value=mock_adapter):
    variant = manager.resolve_variant("abs", "paxml")
  assert variant is not None
  assert variant["api"] == "jax.numpy.abs"


def test_explicit_override_precedence(manager):
  """Verifies the behavior of explicit override precedence."""
  manager.data["Linear"] = {"variants": {"jax": {"api": "flax.nnx.Linear"}, "paxml": {"api": "praxis.layers.Linear"}}}
  mock_adapter = MockAdapterWithParent("jax")
  with patch("ml_switcheroo.semantics.manager.get_adapter", return_value=mock_adapter):
    variant = manager.resolve_variant("Linear", "paxml")
  assert variant["api"] == "praxis.layers.Linear"


def test_deep_inheritance_chain(manager):
  """Verifies the behavior of deep inheritance chain."""
  manager.data["op"] = {"variants": {"parent": {"api": "found_in_parent"}}}

  def mock_get_adapter(name):
    """Provides a mock get adapter for testing."""
    if name == "grandchild":
      return MockAdapterWithParent("child")
    if name == "child":
      return MockAdapterWithParent("parent")
    return None

  with patch("ml_switcheroo.semantics.manager.get_adapter", side_effect=mock_get_adapter):
    variant = manager.resolve_variant("op", "grandchild")
  assert variant is not None
  assert variant["api"] == "found_in_parent"


def test_circular_inheritance_safety(manager):
  """Verifies the behavior of circular inheritance safety."""
  manager.data["op"] = {"variants": {}}

  def mock_circular_adapter(name):
    """Provides a mock circular adapter for testing."""
    if name == "A":
      return MockAdapterWithParent("B")
    if name == "B":
      return MockAdapterWithParent("A")
    return None

  with patch("ml_switcheroo.semantics.manager.get_adapter", side_effect=mock_circular_adapter):
    variant = manager.resolve_variant("op", "A")
  assert variant is None


def test_integration_with_json_confg_fallback(manager):
  """Verifies the behavior of integration with JSON confg fallback."""
  manager.data["abs"] = {"variants": {"jax": {"api": "jnp.abs"}}}
  manager.framework_configs["legacy_fw"] = {"extends": "jax"}
  with patch("ml_switcheroo.semantics.manager.get_adapter", return_value=None):
    variant = manager.resolve_variant("abs", "legacy_fw")
  assert variant is not None
  assert variant["api"] == "jnp.abs"


def test_json_overrides_adapter_inheritance(manager):
  """Verifies the behavior of JSON overrides adapter inheritance."""
  manager.data["op"] = {"variants": {"parent_B": {"api": "found_In_B"}}}

  class MockAdapterA:
    """Mock Adapter A class for testing purposes."""

    inherits_from = "parent_A"

  manager.framework_configs["child"] = {"extends": "parent_B"}
  with patch("ml_switcheroo.semantics.manager.get_adapter", return_value=MockAdapterA()):
    variant = manager.resolve_variant("op", "child")
  assert variant is not None
  assert variant["api"] == "found_In_B"
