"""Test suite for the Rewriter Functional Unwrap module."""

import typing

import libcst as cst
import pytest

from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo.semantics.manager import SemanticsManager
from tests.conftest import TestRewriter


class MockUnwrapSemantics(SemanticsManager):
  """Docstring."""

  def __init__(self) -> None:
    """Initializes the MockUnwrapSemantics instance."""
    self.data: dict[str, typing.Any] = {}
    self._reverse_index: dict[str, tuple[str, dict[str, typing.Any]]] = {}
    self._key_origins: dict[str, str] = {}
    self.import_data: dict[str, typing.Any] = {}
    self.framework_configs: dict[str, typing.Any] = {"jax": {"traits": {"functional_execution_method": "apply"}}}

  def get_framework_config(self, framework: str) -> dict[str, typing.Any]:
    """Mock get config."""
    return self.framework_configs.get(framework, {})

  def get_definition(self, name: str) -> typing.Optional[tuple[str, dict[str, typing.Any]]]:
    """Mock get def."""
    return None

  def resolve_variant(self, abstract_id: str, fw: str) -> typing.Any:
    """Mock."""
    return None


@pytest.fixture
def rewriter() -> TestRewriter:
  """Docstring."""
  semantics = MockUnwrapSemantics()
  config = RuntimeConfig(source_framework="jax", target_framework="jax", strict_mode=False)
  return TestRewriter(semantics, config)


def rewrite_code(rewriter: TestRewriter, code: str) -> str:
  """Rewrites code."""
  tree = cst.parse_module(code)
  try:
    new_tree: typing.Any = rewriter.convert(tree)
    return typing.cast(str, new_tree.code)
  except Exception as e:
    pytest.fail(f"Rewrite failed: {e}")


def test_unwrap_call_only(rewriter: TestRewriter) -> None:
  """Verifies the behavior of unwrap call only."""
  code: str = "z = self.layer.apply(variables, x) + 1"
  result: str = rewrite_code(rewriter, code)
  assert "self.layer(x)" in result
  assert "apply" not in result
  assert "variables" not in result


def test_unwrap_assignment_tuple(rewriter: TestRewriter) -> None:
  """Verifies the behavior of unwrap assignment tuple."""
  code: str = "y, updates = self.layer.apply(vars, x)"
  result: str = rewrite_code(rewriter, code)
  assert "y = self.layer(x)" in result
  assert "updates" not in result
