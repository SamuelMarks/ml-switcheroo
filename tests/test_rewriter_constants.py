"""Test suite for the Rewriter Constants module."""

import pytest
import libcst as cst
from tests.conftest import TestRewriter as PivotRewriter
from ml_switcheroo.semantics.manager import SemanticsManager
from ml_switcheroo.config import RuntimeConfig
from typing import Dict, Any


class MockSemantics(SemanticsManager):
  """Mock Semantics class for testing purposes."""

  def __init__(self) -> None:
    """Initializes the MockSemantics instance."""
    self.data: Dict[str, Any] = {}
    self._reverse_index: Dict[str, Any] = {}
    self._key_origins: Dict[str, Any] = {}
    self.framework_configs: Dict[str, Any] = {}
    self._inject_const("float32", {"torch": "torch.float32", "jax": "jax.numpy.float32"})
    self._inject_func("abs", {"torch": "torch.abs", "jax": "jax.numpy.abs"})
    self._inject_const("cpu", {"torch": "torch.cpu", "jax": "jax.devices('cpu')[0]"})

  def get_framework_config(self, framework: str) -> Dict[str, Any]:
    """Mock implementation of get framework configuration."""
    return self.framework_configs.get(framework, {})

  def _inject_const(self, name: str, mapping: Dict[str, str]) -> None:
    """Mock implementation of  inject const."""
    self.data[name] = {"variants": {}}
    for fw, api in mapping.items():
      self.data[name]["variants"][fw] = {"api": api}
      self._reverse_index[api] = (name, self.data[name])

  def _inject_func(self, name: str, mapping: Dict[str, str]) -> None:
    """Mock implementation of  inject function."""
    self.data[name] = {"variants": {}, "std_args": ["x"]}
    for fw, api in mapping.items():
      self.data[name]["variants"][fw] = {"api": api}
      self._reverse_index[api] = (name, self.data[name])


@pytest.fixture
def rewriter() -> PivotRewriter:
  """Provides a mock rewriter for testing."""
  config: RuntimeConfig = RuntimeConfig(source_framework="torch", target_framework="jax")
  return PivotRewriter(MockSemantics(), config)


def rewrite(rewriter: PivotRewriter, code: str) -> str:
  """Rewrites ."""
  tree: cst.Module = cst.parse_module(code)
  return rewriter.convert(tree).code


def test_constant_rewrite_assignment(rewriter: PivotRewriter) -> None:
  """Verifies the behavior of constant rewrite assignment."""
  code: str = "x = torch.float32"
  res: str = rewrite(rewriter, code)
  assert "jax.numpy.float32" in res
  assert "torch.float32" not in res


def test_constant_rewrite_argument(rewriter: PivotRewriter) -> None:
  """Verifies the behavior of constant rewrite argument."""
  code: str = "y = init(dtype=torch.float32)"
  res: str = rewrite(rewriter, code)
  assert "jax.numpy.float32" in res


def test_function_attribute_bypass(rewriter: PivotRewriter) -> None:
  """Verifies the behavior of function attribute bypass."""
  code: str = "f = torch.abs"
  res: str = rewrite(rewriter, code)
  assert "torch.abs" in res


def test_function_call_rewrite(rewriter: PivotRewriter) -> None:
  """Verifies the behavior of function call rewrite."""
  code: str = "y = torch.abs(x)"
  res: str = rewrite(rewriter, code)
  assert "jax.numpy.abs(x)" in res
  assert "torch.abs" not in res
