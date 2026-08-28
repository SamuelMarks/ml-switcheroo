"""Test suite for the Rewriter module."""

import pytest
import libcst as cst
import typing
from tests.conftest import TestRewriter
from ml_switcheroo.semantics.manager import SemanticsManager
from ml_switcheroo.config import RuntimeConfig


class MockSemantics(SemanticsManager):
  """Mock Semantics class for testing purposes."""

  def __init__(self) -> None:
    """Initializes the MockSemantics instance."""
    self.data: dict[str, typing.Any] = {}
    self.import_data: dict[str, typing.Any] = {}
    self._reverse_index: dict[str, tuple[str, dict[str, typing.Any]]] = {}
    self._key_origins: dict[str, str] = {}
    self.framework_configs: dict[str, typing.Any] = {}
    self._inject("abs", ["x"], "torch.abs", "jax.numpy.abs")
    self._inject("sum", ["x"], "torch.sum", "jax.numpy.sum", s_args={"x": "input"}, t_args={"x": "a"})
    self._inject("neg", ["x"], "torch.neg", "jax.numpy.negative")
    self._inject("add", ["x", "y"], "torch.add", "jax.numpy.add")

  def get_framework_config(self, framework: str) -> dict[str, typing.Any]:
    """Mock implementation of get framework configuration."""
    return self.framework_configs.get(framework, {})

  def _inject(
    self,
    name: str,
    std_args: list[str],
    s_api: str,
    t_api: str,
    s_args: typing.Optional[dict[str, str]] = None,
    t_args: typing.Optional[dict[str, str]] = None,
  ) -> None:
    """Mock implementation of  inject."""
    s_def: dict[str, typing.Any] = {"api": s_api}
    if s_args:
      s_def["args"] = s_args
    t_def: dict[str, typing.Any] = {"api": t_api}
    if t_args:
      t_def["args"] = t_args
    self.data[name] = {"std_args": std_args, "variants": {"torch": s_def, "jax": t_def}}
    self._reverse_index[s_api] = (name, self.data[name])

  def get_definition(self, name: str) -> typing.Optional[tuple[str, dict[str, typing.Any]]]:
    """Mock implementation of get definition."""
    return self._reverse_index.get(name)

  def resolve_variant(self, abstract_id: str, fw: str) -> typing.Any:
    """Mock implementation of resolve variant."""
    return self.data.get(abstract_id, {}).get("variants", {}).get(fw)


@pytest.fixture
def rewriter() -> TestRewriter:
  """Provides a mock rewriter for testing."""
  semantics = MockSemantics()
  config = RuntimeConfig(source_framework="torch", target_framework="jax", strict_mode=False)
  return TestRewriter(semantics, config)


def rewrite(rewriter: TestRewriter, code: str) -> str:
  """Rewrites ."""
  tree = cst.parse_module(code)
  new_tree: typing.Any = rewriter.convert(tree)
  return typing.cast(str, new_tree.code)


def test_simple_api_swap(rewriter: TestRewriter) -> None:
  """Verifies the behavior of simple API swap."""
  code: str = "y = torch.abs(x)"
  result: str = rewrite(rewriter, code)
  assert "jax.numpy.abs(x)" in result


def test_argument_renaming(rewriter: TestRewriter) -> None:
  """Verifies the behavior of argument renaming."""
  code: str = "y = torch.sum(input=t)"
  result: str = rewrite(rewriter, code)
  assert "jax.numpy.sum(a=t)" in result


def test_nested_calls_recursive(rewriter: TestRewriter) -> None:
  """Verifies the behavior of nested calls recursive."""
  code: str = "y = torch.abs(torch.neg(x))"
  result: str = rewrite(rewriter, code)
  assert "jax.numpy.abs" in result
  assert "jax.numpy.negative(x)" in result
  assert "torch" not in result


def test_complex_nested_structure(rewriter: TestRewriter) -> None:
  """Verifies the behavior of complex nested structure."""
  code: str = "y = torch.add(torch.abs(a), torch.neg(b))"
  result: str = rewrite(rewriter, code)
  assert "jax.numpy.add" in result
  assert "jax.numpy.abs(a)" in result
  assert "jax.numpy.negative(b)" in result


def test_return_statement_rewrite(rewriter: TestRewriter) -> None:
  """Verifies the behavior of return statement rewrite."""
  code: str = "def f(x):\n    return torch.abs(x)"
  result: str = rewrite(rewriter, code)
  assert "return jax.numpy.abs(x)" in result


def test_function_arg_rewrite(rewriter: TestRewriter) -> None:
  """Verifies the behavior of function argument rewrite."""
  code: str = "print(torch.abs(x))"
  result: str = rewrite(rewriter, code)
  assert "jax.numpy.abs(x)" in result


def test_list_element_rewrite(rewriter: TestRewriter) -> None:
  """Verifies the behavior of list element rewrite."""
  code: str = "l = [torch.abs(x), torch.neg(y)]"
  result: str = rewrite(rewriter, code)
  assert "jax.numpy.abs(x)" in result
  assert "jax.numpy.negative(y)" in result


def test_dict_value_rewrite(rewriter: TestRewriter) -> None:
  """Verifies the behavior of dictionary value rewrite."""
  code: str = "d = {'val': torch.abs(x)}"
  result: str = rewrite(rewriter, code)
  assert "{'val': jax.numpy.abs(x)}" in result


def test_pass_through_unknown(rewriter: TestRewriter) -> None:
  """Verifies the behavior of pass through unknown."""
  code: str = "y = torch.unknown_func(x)"
  result: str = rewrite(rewriter, code)
  assert "torch.unknown_func(x)" in result


def test_aliased_usage(rewriter: TestRewriter) -> None:
  """Verifies the behavior of aliased usage."""
  code: str = "\nimport torch as t\ny = t.abs(x)\n"
  # To properly test aliased usage, the rewriter context must know about the alias
  rewriter.context.alias_map["t"] = "torch"
  result: str = rewrite(rewriter, code)
  assert "jax.numpy.abs(x)" in result
