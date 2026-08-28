"""Test suite for the Rewriter Constants module."""

import pytest
import typing
import libcst as cst
from tests.conftest import TestRewriter
from ml_switcheroo.semantics.manager import SemanticsManager
from ml_switcheroo.config import RuntimeConfig


class MockSemantics(SemanticsManager):
  """Mock Semantics class for testing purposes."""

  def __init__(self) -> None:
    """Initializes the MockSemantics instance."""
    self.data: dict[str, typing.Any] = {}
    self._reverse_index: dict[str, tuple[str, dict[str, typing.Any]]] = {}
    self._key_origins: dict[str, str] = {}
    self.import_data: dict[str, typing.Any] = {}
    self.framework_configs: dict[str, typing.Any] = {}
    self._inject_const("float32", {"torch": "torch.float32", "jax": "jax.numpy.float32"})
    self._inject_func("abs", {"torch": "torch.abs", "jax": "jax.numpy.abs"})

  def get_framework_config(self, framework: str) -> dict[str, typing.Any]:
    """Mock implementation of get framework configuration."""
    return self.framework_configs.get(framework, {})

  def _inject_const(self, name: str, mapping: dict[str, str]) -> None:
    """Mock implementation of  inject const."""
    self.data[name] = {"variants": {}}
    for fw, api in mapping.items():
      self.data[name]["variants"][fw] = {"api": api}
      self._reverse_index[api] = (name, self.data[name])

  def _inject_func(self, name: str, mapping: dict[str, str]) -> None:
    """Mock implementation of  inject function."""
    self.data[name] = {"variants": {}, "std_args": ["x"]}
    for fw, api in mapping.items():
      self.data[name]["variants"][fw] = {"api": api}
      self._reverse_index[api] = (name, self.data[name])

  def get_definition(self, name: str) -> typing.Optional[tuple[str, dict[str, typing.Any]]]:
    """Mock get_definition."""
    return self._reverse_index.get(name)

  def resolve_variant(self, abstract_id: str, fw: str) -> typing.Any:
    """Mock resolve_variant."""
    return self.data.get(abstract_id, {}).get("variants", {}).get(fw)


@pytest.fixture
def rewriter() -> TestRewriter:
  """Provides a mock rewriter for testing."""
  config = RuntimeConfig(source_framework="torch", target_framework="jax")
  return TestRewriter(MockSemantics(), config)


def rewrite(rewriter: TestRewriter, code: str) -> str:
  """Rewrites ."""
  tree = cst.parse_module(code)
  return typing.cast(str, rewriter.convert(tree).code)


def test_constant_rewrite_assignment(rewriter: TestRewriter) -> None:
  """Verifies the behavior of constant rewrite assignment."""
  code: str = "x = torch.float32"
  res: str = rewrite(rewriter, code)
  assert "jax.numpy.float32" in res
  assert "torch.float32" not in res


def test_constant_rewrite_argument(rewriter: TestRewriter) -> None:
  """Verifies the behavior of constant rewrite argument."""
  code: str = "y = init(dtype=torch.float32)"
  res: str = rewrite(rewriter, code)
  assert "jax.numpy.float32" in res


def test_function_call_rewrite(rewriter: TestRewriter) -> None:
  """Verifies the behavior of function call rewrite."""
  code: str = "y = torch.abs(x)"
  res: str = rewrite(rewriter, code)
  assert "jax.numpy.abs(x)" in res
  assert "torch.abs" not in res
