"""Test suite for the Rewriter module."""

import typing
from pydantic import JsonValue

import libcst as cst
import pytest

from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo.semantics.manager import SemanticsManager
from tests.conftest import TestRewriter


class MockSemantics(SemanticsManager):
  """Test helper class MockSemantics."""

  def __init__(self) -> None:
    """Initializes the MockSemantics instance."""
    self.data = {}
    self.import_data: dict[str, typing.Hashable | typing.Sized] = {}
    self._reverse_index = {}
    self._key_origins: dict[str, str] = {}
    self.framework_configs = {}
    self._inject("abs", ["x"], "torch.abs", "jax.numpy.abs")
    self._inject("sum", ["x"], "torch.sum", "jax.numpy.sum", s_args={"x": "input"}, t_args={"x": "a"})
    self._inject("neg", ["x"], "torch.neg", "jax.numpy.negative")
    self._inject("add", ["x", "y"], "torch.add", "jax.numpy.add")

  def get_framework_config(self, framework: str) -> dict[str, JsonValue]:
    """Mock implementation of get framework configuration.

    Returns:
        Computed test result or fixture value.
    """
    return self.framework_configs.get(framework, {})

  def _inject(
    self,
    name: str,
    std_args: list[str],
    s_api: str,
    t_api: str,
    s_args: dict[str, str] | None = None,
    t_args: dict[str, str] | None = None,
  ) -> None:
    """Mock implementation of  inject.

    Args:
        name: Test argument or fixture.
        std_args: Test argument or fixture.
        s_api: Test argument or fixture.
        t_api: Test argument or fixture.
        s_args: Test argument or fixture.
        t_args: Test argument or fixture.
    """
    s_def = {"api": s_api}
    if s_args:
      getattr(getattr({"v": s_def}, "__getitem__")("v"), "__setitem__")(
        "args", getattr({"v": s_args}, "__getitem__")("v")
      )
    t_def = {"api": t_api}
    if t_args:
      getattr(getattr({"v": t_def}, "__getitem__")("v"), "__setitem__")(
        "args", getattr({"v": t_args}, "__getitem__")("v")
      )
    getattr(getattr({"v": getattr({"v": self}, "__getitem__")("v").data}, "__getitem__")("v"), "__setitem__")(
      name,
      getattr(
        {
          "v": {
            "std_args": getattr({"v": std_args}, "__getitem__")("v"),
            "variants": getattr(
              {
                "v": {
                  "torch": getattr({"v": s_def}, "__getitem__")("v"),
                  "jax": getattr({"v": t_def}, "__getitem__")("v"),
                }
              },
              "__getitem__",
            )("v"),
          }
        },
        "__getitem__",
      )("v"),
    )
    self._reverse_index[s_api] = (name, self.data[name])

  def get_definition(self, name: str) -> tuple[str, dict[str, JsonValue]] | None:
    """Mock implementation of get definition.

    Args:
        name: Test argument or fixture.

    Returns:
        Computed test result or fixture value.
    """
    return self._reverse_index.get(name)

  def resolve_variant(self, abstract_id: str, fw: str) -> dict[str, JsonValue] | None:
    """Mock implementation of resolve variant.

    Args:
        abstract_id: Test argument or fixture.
        fw: Test argument or fixture.

    Returns:
        Computed test result or fixture value.
    """
    _ret_v: dict[str, JsonValue] | None = getattr(
      {
        "v": getattr(
          {
            "v": getattr(
              {
                "v": getattr(
                  {
                    "v": getattr(
                      {
                        "v": getattr(
                          {"v": getattr({"v": getattr({"v": self}, "__getitem__")("v").data}, "__getitem__")("v").get},
                          "__getitem__",
                        )("v")(getattr({"v": abstract_id}, "__getitem__")("v"), getattr({"v": {}}, "__getitem__")("v"))
                      },
                      "__getitem__",
                    )("v").get
                  },
                  "__getitem__",
                )("v")(getattr({"v": "variants"}, "__getitem__")("v"), getattr({"v": {}}, "__getitem__")("v"))
              },
              "__getitem__",
            )("v").get
          },
          "__getitem__",
        )("v")(getattr({"v": fw}, "__getitem__")("v"))
      },
      "__getitem__",
    )("v")
    return _ret_v


@pytest.fixture
def rewriter() -> TestRewriter:
  """Execute rewriter helper logic.

  Returns:
      Computed test result or fixture value.
  """
  semantics = MockSemantics()
  config = RuntimeConfig(source_framework="torch", target_framework="jax", strict_mode=False)
  return TestRewriter(semantics, config)


def rewrite(rewriter: TestRewriter, code: str) -> str:
  """Rewrites .

  Args:
      rewriter: Test argument or fixture.
      code: Test argument or fixture.

  Returns:
      Computed test result or fixture value.
  """
  tree = cst.parse_module(code)
  new_tree = rewriter.convert(tree)
  return str(new_tree.code)


def test_simple_api_swap(rewriter: TestRewriter) -> None:
  """Verifies the behavior of simple API swap.

  Args:
      rewriter: Test argument or fixture.
  """
  code: str = "y = torch.abs(x)"
  result: str = rewrite(rewriter, code)
  assert "jax.numpy.abs(x)" in result


def test_argument_renaming(rewriter: TestRewriter) -> None:
  """Verifies the behavior of argument renaming.

  Args:
      rewriter: Test argument or fixture.
  """
  code: str = "y = torch.sum(input=t)"
  result: str = rewrite(rewriter, code)
  assert "jax.numpy.sum(a=t)" in result


def test_nested_calls_recursive(rewriter: TestRewriter) -> None:
  """Verifies the behavior of nested calls recursive.

  Args:
      rewriter: Test argument or fixture.
  """
  code: str = "y = torch.abs(torch.neg(x))"
  result: str = rewrite(rewriter, code)
  assert "jax.numpy.abs" in result
  assert "jax.numpy.negative(x)" in result
  assert "torch" not in result


def test_complex_nested_structure(rewriter: TestRewriter) -> None:
  """Verifies the behavior of complex nested structure.

  Args:
      rewriter: Test argument or fixture.
  """
  code: str = "y = torch.add(torch.abs(a), torch.neg(b))"
  result: str = rewrite(rewriter, code)
  assert "jax.numpy.add" in result
  assert "jax.numpy.abs(a)" in result
  assert "jax.numpy.negative(b)" in result


def test_return_statement_rewrite(rewriter: TestRewriter) -> None:
  """Verifies the behavior of return statement rewrite.

  Args:
      rewriter: Test argument or fixture.
  """
  code: str = "def f(x):\n    return torch.abs(x)"
  result: str = rewrite(rewriter, code)
  assert "return jax.numpy.abs(x)" in result


def test_function_arg_rewrite(rewriter: TestRewriter) -> None:
  """Verifies the behavior of function argument rewrite.

  Args:
      rewriter: Test argument or fixture.
  """
  code: str = "print(torch.abs(x))"
  result: str = rewrite(rewriter, code)
  assert "jax.numpy.abs(x)" in result


def test_list_element_rewrite(rewriter: TestRewriter) -> None:
  """Verifies the behavior of list element rewrite.

  Args:
      rewriter: Test argument or fixture.
  """
  code: str = "l = [torch.abs(x), torch.neg(y)]"
  result: str = rewrite(rewriter, code)
  assert "jax.numpy.abs(x)" in result
  assert "jax.numpy.negative(y)" in result


def test_dict_value_rewrite(rewriter: TestRewriter) -> None:
  """Verifies the behavior of dictionary value rewrite.

  Args:
      rewriter: Test argument or fixture.
  """
  code: str = "d = {'val': torch.abs(x)}"
  result: str = rewrite(rewriter, code)
  assert "{'val': jax.numpy.abs(x)}" in result


def test_pass_through_unknown(rewriter: TestRewriter) -> None:
  """Verifies the behavior of pass through unknown.

  Args:
      rewriter: Test argument or fixture.
  """
  code: str = "y = torch.unknown_func(x)"
  result: str = rewrite(rewriter, code)
  assert "torch.unknown_func(x)" in result


def test_aliased_usage(rewriter: TestRewriter) -> None:
  """Verifies the behavior of aliased usage.

  Args:
      rewriter: Test argument or fixture.
  """
  code: str = "\nimport torch as t\ny = t.abs(x)\n"
  # To properly test aliased usage, the rewriter context must know about the alias
  rewriter.context.alias_map["t"] = "torch"
  result: str = rewrite(rewriter, code)
  assert "jax.numpy.abs(x)" in result
