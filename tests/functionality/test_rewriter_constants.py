"""Test suite for the Rewriter Constants module."""

import typing

import libcst as cst
import pytest

from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo.semantics.manager import SemanticsManager
from tests.conftest import TestRewriter


class MockSemantics(SemanticsManager):
  """Test helper class MockSemantics."""

  def __init__(self) -> None:
    """Initializes the MockSemantics instance."""
    self.data: dict[str, typing.Any] = {}
    self._reverse_index: dict[str, tuple[str, dict[str, typing.Any]]] = {}
    self._key_origins: dict[str, str] = {}
    self.import_data: dict[str, typing.Any] = {}
    self.framework_configs: dict[str, typing.Any] = {}
    self._inject_const("float32", {"torch": "torch.float32", "jax": "jax.numpy.float32"})
    self._inject_func("abs", {"torch": "torch.abs", "jax": "jax.numpy.abs"})

  def get_framework_config(self, framework: str) -> typing.Any:
    """Mock implementation of get framework configuration.

    Returns:
        Computed test result or fixture value.
    """
    return self.framework_configs.get(framework, {})

  def _inject_const(self, name: str, mapping: dict[str, str]) -> None:
    """Mock implementation of  inject const.

    Args:
        name: Test argument or fixture.
        mapping: Test argument or fixture.
    """
    self.data[name] = {"variants": {}}
    for fw, api in mapping.items():
      self.data[name]["variants"][fw] = {"api": api}
      self._reverse_index[api] = (name, self.data[name])

  def _inject_func(self, name: str, mapping: dict[str, str]) -> None:
    """Mock implementation of  inject function.

    Args:
        name: Test argument or fixture.
        mapping: Test argument or fixture.
    """
    self.data[name] = {"variants": {}, "std_args": ["x"]}
    for fw, api in mapping.items():
      self.data[name]["variants"][fw] = {"api": api}
      self._reverse_index[api] = (name, self.data[name])

  def get_definition(self, name: str) -> typing.Optional[tuple[str, dict[str, typing.Any]]]:
    """Mock get_definition.

    Args:
        name: Test argument or fixture.

    Returns:
        Computed test result or fixture value.
    """
    return self._reverse_index.get(name)

  def resolve_variant(self, abstract_id: str, fw: str) -> typing.Any:
    """Mock resolve_variant.

    Args:
        abstract_id: Test argument or fixture.
        fw: Test argument or fixture.

    Returns:
        Computed test result or fixture value.
    """
    return self.data.get(abstract_id, {}).get("variants", {}).get(fw)


@pytest.fixture
def rewriter() -> TestRewriter:
  """Execute rewriter helper logic.

  Returns:
      Computed test result or fixture value.
  """
  config = RuntimeConfig(source_framework="torch", target_framework="jax")
  return TestRewriter(MockSemantics(), config)


def rewrite(rewriter: TestRewriter, code: str) -> str:
  """Rewrites .

  Args:
      rewriter: Test argument or fixture.
      code: Test argument or fixture.

  Returns:
      Computed test result or fixture value.
  """
  tree = cst.parse_module(code)
  return typing.cast(str, rewriter.convert(tree).code)


def test_constant_rewrite_assignment(rewriter: TestRewriter) -> None:
  """Verifies the behavior of constant rewrite assignment.

  Args:
      rewriter: Test argument or fixture.
  """
  code: str = "x = torch.float32"
  res: str = rewrite(rewriter, code)
  assert "jax.numpy.float32" in res
  assert "torch.float32" not in res


def test_constant_rewrite_argument(rewriter: TestRewriter) -> None:
  """Verifies the behavior of constant rewrite argument.

  Args:
      rewriter: Test argument or fixture.
  """
  code: str = "y = init(dtype=torch.float32)"
  res: str = rewrite(rewriter, code)
  assert "jax.numpy.float32" in res


def test_function_call_rewrite(rewriter: TestRewriter) -> None:
  """Verifies the behavior of function call rewrite.

  Args:
      rewriter: Test argument or fixture.
  """
  code: str = "y = torch.abs(x)"
  res: str = rewrite(rewriter, code)
  assert "jax.numpy.abs(x)" in res
  assert "torch.abs" not in res
