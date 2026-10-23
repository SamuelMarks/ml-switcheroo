"""Test suite for the Rewriter Functional Unwrap module."""

from ml_switcheroo.enums import RuntimeTensorOrValue

from pydantic import JsonValue

import libcst as cst
import pytest

from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo.semantics.manager import SemanticsManager
from tests.conftest import TestRewriter


class MockUnwrapSemantics(SemanticsManager):
  """Test helper class MockUnwrapSemantics."""

  def __init__(self) -> None:
    """Initializes the MockUnwrapSemantics instance."""
    self.data = {}
    self._reverse_index = {}
    self._key_origins: dict[str, str] = {}
    self.import_data: dict[str, RuntimeTensorOrValue] = {}
    self.framework_configs = {"jax": {"traits": {"functional_execution_method": "apply"}}}

  def get_framework_config(self, framework: str) -> dict[str, JsonValue]:
    """Mock get config.

    Args:
        framework: The framework (str) for get framework config.

    Returns:
        Configuration dictionary for the specified framework.
    """
    return self.framework_configs.get(framework, {})

  def get_definition(self, name: str) -> tuple[str, dict[str, JsonValue]] | None:
    """Mock get def.

    Args:
        name: Identifier or operation name string.

    Returns:
        Tuple of abstract operation ID and definition metadata, or None.
    """
    return None

  def resolve_variant(self, abstract_id: str, fw: str) -> dict[str, JsonValue] | None:
    """Mock.

    Args:
        abstract_id: Canonical abstract operation identifier.
        fw: Framework identifier string (e.g., torch, jax, mlx).

    Returns:
        Target framework variant dictionary, or None if unsupported.
    """
    return None


@pytest.fixture
def rewriter() -> TestRewriter:
  """Execute rewriter helper logic.

  Returns:
      Configured `rewriter` fixture value (TestRewriter).
  """
  semantics = MockUnwrapSemantics()
  config = RuntimeConfig(source_framework="jax", target_framework="jax", strict_mode=False)
  return TestRewriter(semantics, config)


def rewrite_code(rewriter: TestRewriter, code: str) -> str:
  """Rewrites code.

  Args:
      code: Python or DSL source code string to parse or transpile.
      rewriter: Configured CST rewriter test helper instance.

  Returns:
      Resulting `str` from rewrite code.
  """
  tree = cst.parse_module(code)
  try:
    new_tree = rewriter.convert(tree)
    return str(new_tree.code)
  except Exception as e:
    pytest.fail(f"Rewrite failed: {e}")


def test_unwrap_call_only(rewriter: TestRewriter) -> None:
  """Verifies the behavior of unwrap call only.

  Args:
      rewriter: Configured CST rewriter test helper instance.
  """
  code: str = "z = self.layer.apply(variables, x) + 1"
  result: str = rewrite_code(rewriter, code)
  assert "self.layer(x)" in result
  assert "apply" not in result
  assert "variables" not in result


def test_unwrap_assignment_tuple(rewriter: TestRewriter) -> None:
  """Verifies the behavior of unwrap assignment tuple.

  Args:
      rewriter: Configured CST rewriter test helper instance.
  """
  code: str = "y, updates = self.layer.apply(vars, x)"
  result: str = rewrite_code(rewriter, code)
  assert "y = self.layer(x)" in result
  assert "updates" not in result
