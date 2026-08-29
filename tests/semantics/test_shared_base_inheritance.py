"""Test suite for the Shared Base Inheritance module."""

from typing import Any, Dict, Generator, Optional
from unittest.mock import patch

import pytest

from ml_switcheroo.semantics.manager import SemanticsManager


class MockAdapterWithParent:
  """Docstring."""

  def __init__(self, parent: str) -> None:
    """Initializes the MockAdapterWithParent instance."""
    self.inherits_from: str = parent


@pytest.fixture
def manager() -> Generator[SemanticsManager, None, None]:
  """Provides a mock manager for testing.

  Yields:
      SemanticsManager: A semantics manager instance.
  """
  mgr: SemanticsManager = SemanticsManager()
  mgr._reverse_index = {}
  mgr.data = {}
  yield mgr


def test_immediate_fallback(manager: SemanticsManager) -> None:
  """Verifies the behavior of immediate fallback.

  Args:
      manager (SemanticsManager): The manager fixture.
  """
  manager.data["abs"] = {"variants": {"jax": {"api": "jax.numpy.abs"}}}
  mock_adapter: MockAdapterWithParent = MockAdapterWithParent("jax")
  with patch("ml_switcheroo.semantics.manager.get_adapter", return_value=mock_adapter):
    variant: Optional[Dict[str, Any]] = manager.resolve_variant("abs", "paxml")
  assert variant is not None
  assert variant["api"] == "jax.numpy.abs"


def test_explicit_override_precedence(manager: SemanticsManager) -> None:
  """Verifies the behavior of explicit override precedence.

  Args:
      manager (SemanticsManager): The manager fixture.
  """
  manager.data["Linear"] = {"variants": {"jax": {"api": "flax.nnx.Linear"}, "paxml": {"api": "praxis.layers.Linear"}}}
  mock_adapter: MockAdapterWithParent = MockAdapterWithParent("jax")
  with patch("ml_switcheroo.semantics.manager.get_adapter", return_value=mock_adapter):
    variant: Optional[Dict[str, Any]] = manager.resolve_variant("Linear", "paxml")
  assert variant is not None
  assert variant["api"] == "praxis.layers.Linear"


def test_deep_inheritance_chain(manager: SemanticsManager) -> None:
  """Verifies the behavior of deep inheritance chain.

  Args:
      manager (SemanticsManager): The manager fixture.
  """
  manager.data["op"] = {"variants": {"parent": {"api": "found_in_parent"}}}

  def mock_get_adapter(name: str) -> Optional[MockAdapterWithParent]:
    """Provides a mock get adapter for testing.

    Args:
        name (str): Adapter name.

    Returns:
        Optional[MockAdapterWithParent]: Mock adapter instance.
    """
    if name == "grandchild":
      return MockAdapterWithParent("child")
    if name == "child":
      return MockAdapterWithParent("parent")
    return None

  with patch("ml_switcheroo.semantics.manager.get_adapter", side_effect=mock_get_adapter):
    variant: Optional[Dict[str, Any]] = manager.resolve_variant("op", "grandchild")
  assert variant is not None
  assert variant["api"] == "found_in_parent"


def test_circular_inheritance_safety(manager: SemanticsManager) -> None:
  """Verifies the behavior of circular inheritance safety.

  Args:
      manager (SemanticsManager): The manager fixture.
  """
  manager.data["op"] = {"variants": {}}

  def mock_circular_adapter(name: str) -> Optional[MockAdapterWithParent]:
    """Provides a mock circular adapter for testing.

    Args:
        name (str): Adapter name.

    Returns:
        Optional[MockAdapterWithParent]: Mock adapter instance.
    """
    if name == "A":
      return MockAdapterWithParent("B")
    if name == "B":
      return MockAdapterWithParent("A")
    return None

  with patch("ml_switcheroo.semantics.manager.get_adapter", side_effect=mock_circular_adapter):
    variant: Optional[Dict[str, Any]] = manager.resolve_variant("op", "A")
  assert variant is None


def test_integration_with_json_confg_fallback(manager: SemanticsManager) -> None:
  """Verifies the behavior of integration with JSON confg fallback.

  Args:
      manager (SemanticsManager): The manager fixture.
  """
  manager.data["abs"] = {"variants": {"jax": {"api": "jnp.abs"}}}
  manager.framework_configs["legacy_fw"] = {"extends": "jax"}
  with patch("ml_switcheroo.semantics.manager.get_adapter", return_value=None):
    variant: Optional[Dict[str, Any]] = manager.resolve_variant("abs", "legacy_fw")
  assert variant is not None
  assert variant["api"] == "jnp.abs"


def test_json_overrides_adapter_inheritance(manager: SemanticsManager) -> None:
  """Verifies the behavior of JSON overrides adapter inheritance.

  Args:
      manager (SemanticsManager): The manager fixture.
  """
  manager.data["op"] = {"variants": {"parent_B": {"api": "found_In_B"}}}

  class MockAdapterA:
    inherits_from: str = "parent_A"

  manager.framework_configs["child"] = {"extends": "parent_B"}
  with patch("ml_switcheroo.semantics.manager.get_adapter", return_value=MockAdapterA()):
    variant: Optional[Dict[str, Any]] = manager.resolve_variant("op", "child")
  assert variant is not None
  assert variant["api"] == "found_In_B"
