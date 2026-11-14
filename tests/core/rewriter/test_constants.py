"""Test suite for Rewriter constant attribute and function call rewriting."""

from __future__ import annotations

import libcst as cst
import pytest
from pydantic import JsonValue

from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo.semantics.manager import SemanticsManager
from tests.conftest import TestRewriter


class MockSemantics(SemanticsManager):
  """Minimal SemanticsManager double registering dtype constants and unary functions."""

  def __init__(self) -> None:
    """Initialize MockSemantics with float32, cpu, and abs mappings."""
    super().__init__(load_defaults=False)
    self._inject_const("float32", {"torch": "torch.float32", "jax": "jax.numpy.float32"})
    self._inject_func("abs", {"torch": "torch.abs", "jax": "jax.numpy.abs"})
    self._inject_const("cpu", {"torch": "torch.cpu", "jax": "jax.devices('cpu')[0]"})

  def get_framework_config(self, framework: str) -> dict[str, JsonValue]:
    """Return stored configuration dictionary for ``framework``.

    Args:
        framework: Framework identifier string.

    Returns:
        Configuration dictionary for ``framework``.
    """
    return self.framework_configs.get(framework, {})

  def _inject_const(self, name: str, mapping: dict[str, str]) -> None:
    """Register a constant/attribute operation across frameworks.

    Args:
        name: Canonical constant identifier.
        mapping: Map of framework ID to concrete attribute path.
    """
    variants: dict[str, JsonValue] = {fw: {"api": api} for fw, api in mapping.items()}
    entry: dict[str, JsonValue] = {"variants": variants}
    self.data[name] = entry
    for api in mapping.values():
      self._reverse_index[api] = (name, entry)

  def _inject_func(self, name: str, mapping: dict[str, str]) -> None:
    """Register a unary function operation across frameworks.

    Args:
        name: Canonical operation identifier.
        mapping: Map of framework ID to concrete function path.
    """
    variants: dict[str, JsonValue] = {fw: {"api": api} for fw, api in mapping.items()}
    entry: dict[str, JsonValue] = {"variants": variants, "std_args": ["x"]}
    self.data[name] = entry
    for api in mapping.values():
      self._reverse_index[api] = (name, entry)


@pytest.fixture
def rewriter() -> TestRewriter:
  """Create a PyTorch-to-JAX TestRewriter backed by MockSemantics.

  Returns:
      Configured ``TestRewriter`` instance.
  """
  config = RuntimeConfig(source_framework="torch", target_framework="jax")
  return TestRewriter(MockSemantics(), config)


def rewrite(rewriter: TestRewriter, code: str) -> str:
  """Parse and transpile ``code`` using ``rewriter``.

  Args:
      rewriter: Configured CST rewriter test helper instance.
      code: Python source code snippet to transpile.

  Returns:
      Transpiled Python source code string.
  """
  tree = cst.parse_module(code)
  return rewriter.convert(tree).code


def test_constant_rewrite_assignment(rewriter: TestRewriter) -> None:
  """Verify standalone dtype constant assignments rewrite to target constants.

  Args:
      rewriter: Configured CST rewriter test helper instance.
  """
  res = rewrite(rewriter, "x = torch.float32")
  assert "jax.numpy.float32" in res
  assert "torch.float32" not in res


def test_constant_rewrite_argument(rewriter: TestRewriter) -> None:
  """Verify dtype constant keyword arguments rewrite to target constants.

  Args:
      rewriter: Configured CST rewriter test helper instance.
  """
  res = rewrite(rewriter, "y = init(dtype=torch.float32)")
  assert "jax.numpy.float32" in res


def test_function_attribute_bypass(rewriter: TestRewriter) -> None:
  """Verify non-constant callable references assigned without call parens are preserved.

  Args:
      rewriter: Configured CST rewriter test helper instance.
  """
  res = rewrite(rewriter, "f = torch.abs")
  assert "torch.abs" in res


def test_function_call_rewrite(rewriter: TestRewriter) -> None:
  """Verify standard function call expressions rewrite to target framework APIs.

  Args:
      rewriter: Configured CST rewriter test helper instance.
  """
  res = rewrite(rewriter, "y = torch.abs(x)")
  assert "jax.numpy.abs(x)" in res
  assert "torch.abs" not in res
