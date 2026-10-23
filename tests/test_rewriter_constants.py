"""Test suite for the Rewriter Constants module."""

from pydantic import JsonValue

import libcst as cst
import pytest

from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo.semantics.manager import SemanticsManager
from tests.conftest import TestRewriter as PivotRewriter


class MockSemantics(SemanticsManager):
  """Test helper class MockSemantics."""

  def __init__(self) -> None:
    """Initializes the MockSemantics instance."""
    self.data = {}
    self._reverse_index = {}
    self._key_origins = {}
    self.framework_configs = {}
    self._inject_const("float32", {"torch": "torch.float32", "jax": "jax.numpy.float32"})
    self._inject_func("abs", {"torch": "torch.abs", "jax": "jax.numpy.abs"})
    self._inject_const("cpu", {"torch": "torch.cpu", "jax": "jax.devices('cpu')[0]"})

  def get_framework_config(self, framework: str) -> dict[str, JsonValue]:
    """Mock implementation of get framework configuration.

    Returns:
        Configuration dictionary for the specified framework.
    """
    return self.framework_configs.get(framework, {})

  def _inject_const(self, name: str, mapping: dict[str, str]) -> None:
    """Mock implementation of  inject const.

    Args:
        name: Identifier or operation name string.
        mapping: The mapping (dict[str, str]) for  inject const.
    """
    self.data[name] = {"variants": {}}
    for fw, api in mapping.items():
      getattr(
        self.data[name]["variants"],
        "__setitem__",
      )(fw, {"api": api})
      self._reverse_index[api] = (name, self.data[name])

  def _inject_func(self, name: str, mapping: dict[str, str]) -> None:
    """Mock implementation of  inject function.

    Args:
        name: Identifier or operation name string.
        mapping: The mapping (dict[str, str]) for  inject func.
    """
    self.data[name] = {"variants": {}, "std_args": ["x"]}
    for fw, api in mapping.items():
      getattr(
        self.data[name]["variants"],
        "__setitem__",
      )(fw, {"api": api})
      self._reverse_index[api] = (name, self.data[name])


@pytest.fixture
def rewriter() -> PivotRewriter:
  """Execute rewriter helper logic.

  Returns:
      Configured `rewriter` fixture value (PivotRewriter).
  """
  config: RuntimeConfig = RuntimeConfig(source_framework="torch", target_framework="jax")
  return PivotRewriter(MockSemantics(), config)


def rewrite(rewriter: PivotRewriter, code: str) -> str:
  """Rewrites .

  Args:
      rewriter: Configured CST rewriter test helper instance.
      code: Python or DSL source code string to parse or transpile.

  Returns:
      Transpiled source code string or rewritten CST module.
  """
  tree: cst.Module = cst.parse_module(code)
  return rewriter.convert(tree).code


def test_constant_rewrite_assignment(rewriter: PivotRewriter) -> None:
  """Verifies the behavior of constant rewrite assignment.

  Args:
      rewriter: Configured CST rewriter test helper instance.
  """
  code: str = "x = torch.float32"
  res: str = rewrite(rewriter, code)
  assert "jax.numpy.float32" in res
  assert "torch.float32" not in res


def test_constant_rewrite_argument(rewriter: PivotRewriter) -> None:
  """Verifies the behavior of constant rewrite argument.

  Args:
      rewriter: Configured CST rewriter test helper instance.
  """
  code: str = "y = init(dtype=torch.float32)"
  res: str = rewrite(rewriter, code)
  assert "jax.numpy.float32" in res


def test_function_attribute_bypass(rewriter: PivotRewriter) -> None:
  """Verifies the behavior of function attribute bypass.

  Args:
      rewriter: Configured CST rewriter test helper instance.
  """
  code: str = "f = torch.abs"
  res: str = rewrite(rewriter, code)
  assert "torch.abs" in res


def test_function_call_rewrite(rewriter: PivotRewriter) -> None:
  """Verifies the behavior of function call rewrite.

  Args:
      rewriter: Configured CST rewriter test helper instance.
  """
  code: str = "y = torch.abs(x)"
  res: str = rewrite(rewriter, code)
  assert "jax.numpy.abs(x)" in res
  assert "torch.abs" not in res
