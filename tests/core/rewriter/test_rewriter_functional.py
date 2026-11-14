"""Test suite for the Rewriter module."""

import libcst as cst
import pytest
from pydantic import JsonValue

from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo.enums import RuntimeTensorOrValue
from ml_switcheroo.semantics.manager import SemanticsManager
from tests.conftest import TestRewriter


class MockSemantics(SemanticsManager):
  """Stub `SemanticsManager` subclass implementing `get_framework_config`, `_inject` for rewriter tests."""

  def __init__(self) -> None:
    """Initializes the MockSemantics instance."""
    self.data = {}
    self.import_data: dict[str, RuntimeTensorOrValue] = {}
    self._reverse_index = {}
    self._key_origins: dict[str, str] = {}
    self.framework_configs = {}
    self._inject("abs", ["x"], "torch.abs", "jax.numpy.abs")
    self._inject("sum", ["x"], "torch.sum", "jax.numpy.sum", s_args={"x": "input"}, t_args={"x": "a"})
    self._inject("neg", ["x"], "torch.neg", "jax.numpy.negative")
    self._inject("add", ["x", "y"], "torch.add", "jax.numpy.add")

  def get_framework_config(self, framework: str) -> dict[str, JsonValue]:
    """Mock implementation of get framework configuration.

    Args:
        framework: Framework identifier string.

    Returns:
        Resulting `dict[str, JsonValue]` from get framework config.
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
    """Execute `_inject` on `MockSemantics` for rewriter tests.

    Args:
        name: Identifier or operation name string.
        std_args: Input `std_args` parameter value (`list[str]`).
        s_api: Input `s_api` parameter value (`str`).
        t_api: Input `t_api` parameter value (`str`).
        s_args: Input `s_args` parameter value (`dict[str, str] | None`).
        t_args: Input `t_args` parameter value (`dict[str, str] | None`).
    """
    s_def: dict[str, JsonValue] = {"api": s_api}
    if s_args:
      s_args_json: dict[str, JsonValue] = dict(s_args.items())
      s_def["args"] = s_args_json
    t_def: dict[str, JsonValue] = {"api": t_api}
    if t_args:
      t_args_json: dict[str, JsonValue] = dict(t_args.items())
      t_def["args"] = t_args_json
    std_args_json: list[JsonValue] = list(std_args)
    op_def: dict[str, JsonValue] = {
      "std_args": std_args_json,
      "variants": {
        "torch": s_def,
        "jax": t_def,
      },
    }
    self.data[name] = op_def
    self._reverse_index[s_api] = (name, op_def)

  def get_definition(self, name: str) -> tuple[str, dict[str, JsonValue]] | None:
    """Mock implementation of get definition.

    Args:
        name: Identifier or operation name string.

    Returns:
        Resulting `tuple[str, dict[str, JsonValue]] | None` from get definition.
    """
    return self._reverse_index.get(name)

  def resolve_variant(self, abstract_id: str, fw: str) -> dict[str, JsonValue] | None:
    """Mock implementation of resolve variant.

    Args:
        abstract_id: Canonical abstract operation identifier (`str`).
        fw: Framework identifier string (e.g., torch, jax, mlx).

    Returns:
        Resulting `dict[str, JsonValue] | None` from resolve variant.
    """
    defn = self.data.get(abstract_id)
    if not isinstance(defn, dict):
      return None
    variants = defn.get("variants")
    if not isinstance(variants, dict):
      return None
    target = variants.get(fw)
    return target if isinstance(target, dict) else None


@pytest.fixture
def rewriter() -> TestRewriter:
  """Create a configured `TestRewriter` fixture for rewriter tests.

  Returns:
      Configured `rewriter` fixture value (TestRewriter).
  """
  semantics = MockSemantics()
  config = RuntimeConfig(source_framework="torch", target_framework="jax", strict_mode=False)
  return TestRewriter(semantics, config)


def rewrite(rewriter: TestRewriter, code: str) -> str:
  """Execute `rewriter.convert` to support rewrite in rewriter tests.

  Args:
      rewriter: Configured CST rewriter test helper instance.
      code: Python or DSL source code string to parse or transpile.

  Returns:
      Resulting `str` from rewrite.
  """
  tree = cst.parse_module(code)
  new_tree = rewriter.convert(tree)
  return str(new_tree.code)


def test_simple_api_swap(rewriter: TestRewriter) -> None:
  """Verify `rewrite` emits `jax.numpy.abs(x)` for simple API swap.

  Args:
      rewriter: Configured CST rewriter test helper instance.
  """
  code: str = "y = torch.abs(x)"
  result: str = rewrite(rewriter, code)
  assert "jax.numpy.abs(x)" in result


def test_argument_renaming(rewriter: TestRewriter) -> None:
  """Verify `rewrite` emits `jax.numpy.sum(a=t)` for argument renaming.

  Args:
      rewriter: Configured CST rewriter test helper instance.
  """
  code: str = "y = torch.sum(input=t)"
  result: str = rewrite(rewriter, code)
  assert "jax.numpy.sum(a=t)" in result


def test_nested_calls_recursive(rewriter: TestRewriter) -> None:
  """Verify `rewrite` emits `jax.numpy.abs` for nested calls recursive.

  Args:
      rewriter: Configured CST rewriter test helper instance.
  """
  code: str = "y = torch.abs(torch.neg(x))"
  result: str = rewrite(rewriter, code)
  assert "jax.numpy.abs" in result
  assert "jax.numpy.negative(x)" in result
  assert "torch" not in result


def test_complex_nested_structure(rewriter: TestRewriter) -> None:
  """Verify `rewrite` emits `jax.numpy.add` for complex nested structure.

  Args:
      rewriter: Configured CST rewriter test helper instance.
  """
  code: str = "y = torch.add(torch.abs(a), torch.neg(b))"
  result: str = rewrite(rewriter, code)
  assert "jax.numpy.add" in result
  assert "jax.numpy.abs(a)" in result
  assert "jax.numpy.negative(b)" in result


def test_return_statement_rewrite(rewriter: TestRewriter) -> None:
  """Verify `rewrite` emits `return jax.numpy.abs(x)` for return statement rewrite.

  Args:
      rewriter: Configured CST rewriter test helper instance.
  """
  code: str = "def f(x):\n    return torch.abs(x)"
  result: str = rewrite(rewriter, code)
  assert "return jax.numpy.abs(x)" in result


def test_function_arg_rewrite(rewriter: TestRewriter) -> None:
  """Verify `rewrite` emits `jax.numpy.abs(x)` for function arg rewrite.

  Args:
      rewriter: Configured CST rewriter test helper instance.
  """
  code: str = "print(torch.abs(x))"
  result: str = rewrite(rewriter, code)
  assert "jax.numpy.abs(x)" in result


def test_list_element_rewrite(rewriter: TestRewriter) -> None:
  """Verify `rewrite` emits `jax.numpy.abs(x)` for list element rewrite.

  Args:
      rewriter: Configured CST rewriter test helper instance.
  """
  code: str = "l = [torch.abs(x), torch.neg(y)]"
  result: str = rewrite(rewriter, code)
  assert "jax.numpy.abs(x)" in result
  assert "jax.numpy.negative(y)" in result


def test_dict_value_rewrite(rewriter: TestRewriter) -> None:
  """Verify `rewrite` emits `{'val': jax.numpy.abs(x)}` for dict value rewrite.

  Args:
      rewriter: Configured CST rewriter test helper instance.
  """
  code: str = "d = {'val': torch.abs(x)}"
  result: str = rewrite(rewriter, code)
  assert "{'val': jax.numpy.abs(x)}" in result


def test_pass_through_unknown(rewriter: TestRewriter) -> None:
  """Verify `rewrite` emits `torch.unknown_func(x)` for pass through unknown.

  Args:
      rewriter: Configured CST rewriter test helper instance.
  """
  code: str = "y = torch.unknown_func(x)"
  result: str = rewrite(rewriter, code)
  assert "torch.unknown_func(x)" in result


def test_aliased_usage(rewriter: TestRewriter) -> None:
  """Verify `rewrite` emits `jax.numpy.abs(x)` for aliased usage.

  Args:
      rewriter: Configured CST rewriter test helper instance.
  """
  code: str = "\nimport torch as t\ny = t.abs(x)\n"
  # To properly test aliased usage, the rewriter context must know about the alias
  rewriter.context.alias_map["t"] = "torch"
  result: str = rewrite(rewriter, code)
  assert "jax.numpy.abs(x)" in result
