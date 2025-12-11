"""
Tests for Shared Base Inheritance Resolution.

Verifies that SemanticsManager correctly walks the framework hierarchy
defined by adapters to find operation mappings.
"""

from unittest.mock import MagicMock, patch
import pytest

from ml_switcheroo.semantics.manager import SemanticsManager


class MockAdapterWithParent:
  def __init__(self, parent):
    self.inherits_from = parent


@pytest.fixture
def manager():
  # Initialize with clean state
  mgr = SemanticsManager()
  mgr._reverse_index = {}
  mgr.data = {}
  return mgr


def test_immediate_fallback(manager):
  """
  Scenario: 'paxml' has no mapping for 'abs', but inherits from 'jax'. 'jax' has mappings.
  Expectation: resolve_variant('paxml') returns 'jax' variant.
  """
  # 1. Setup Data: 'abs' defined for JAX only
  manager.data["abs"] = {"variants": {"jax": {"api": "jax.numpy.abs"}}}

  # 2. Mock Registry: 'paxml' adapter inherits from 'jax'
  mock_adapter = MockAdapterWithParent("jax")

  with patch("ml_switcheroo.semantics.manager.get_adapter", return_value=mock_adapter):
    variant = manager.resolve_variant("abs", "paxml")

  assert variant is not None
  assert variant["api"] == "jax.numpy.abs"


def test_explicit_override_precedence(manager):
  """
  Scenario: 'paxml' defines 'Linear' specifically. 'jax' also has it.
  Expectation: resolve_variant('paxml') returns 'paxml' variant (Override).
  """
  manager.data["Linear"] = {"variants": {"jax": {"api": "flax.nnx.Linear"}, "paxml": {"api": "praxis.layers.Linear"}}}

  mock_adapter = MockAdapterWithParent("jax")

  with patch("ml_switcheroo.semantics.manager.get_adapter", return_value=mock_adapter):
    variant = manager.resolve_variant("Linear", "paxml")

  assert variant["api"] == "praxis.layers.Linear"


def test_deep_inheritance_chain(manager):
  """
  Scenario: GrandChild -> Child -> Parent.
  Operation defined only in Parent.
  Request for GrandChild should traverse up to Parent.
  """
  manager.data["op"] = {"variants": {"parent": {"api": "found_in_parent"}}}

  # Mock get_adapter to handle dynamic lookups
  def mock_get_adapter(name):
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
  """
  Scenario: A -> B -> A.
  Operation exists in C (unreachable) or nowhere.
  Expectation: Stops safely at recursion limit, returns None.
  """
  manager.data["op"] = {"variants": {}}

  def mock_circular_adapter(name):
    if name == "A":
      return MockAdapterWithParent("B")
    if name == "B":
      return MockAdapterWithParent("A")
    return None

  with patch("ml_switcheroo.semantics.manager.get_adapter", side_effect=mock_circular_adapter):
    variant = manager.resolve_variant("op", "A")

  assert variant is None


def test_integration_with_json_confg_fallback(manager):
  """
  Scenario: Adapter not registered, but JSON config defined 'extends'.
  Expectation: Manager falls back to framework_configs dict.
  """
  manager.data["abs"] = {"variants": {"jax": {"api": "jnp.abs"}}}

  # Inject config manually simulating loaded JSON
  manager.framework_configs["legacy_fw"] = {"extends": "jax"}

  # Registry returns None for 'legacy_fw'
  with patch("ml_switcheroo.semantics.manager.get_adapter", return_value=None):
    variant = manager.resolve_variant("abs", "legacy_fw")

  assert variant is not None
  assert variant["api"] == "jnp.abs"


def test_json_overrides_adapter_inheritance(manager):
  """
  Scenario: Adapter says inherits from 'parent_A'.
            JSON config says inherits from 'parent_B'.
            Operation is defined in 'parent_B'.
  Expectation: Manager follows JSON config to 'parent_B'.
  This confirms the 'JSON-First' architecture requirement.
  """
  manager.data["op"] = {"variants": {"parent_B": {"api": "found_In_B"}}}

  # 1. Setup Mock Adapter (Code: Claims A)
  class MockAdapterA:
    inherits_from = "parent_A"

  # 2. Setup Config (JSON: Claims B)
  manager.framework_configs["child"] = {"extends": "parent_B"}

  with patch("ml_switcheroo.semantics.manager.get_adapter", return_value=MockAdapterA()):
    variant = manager.resolve_variant("op", "child")

  assert variant is not None
  assert variant["api"] == "found_In_B"
