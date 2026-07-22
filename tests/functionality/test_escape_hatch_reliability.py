"""Test suite for the Escape Hatch Reliability module."""

import pytest
import libcst as cst
from tests.conftest import TestRewriter as PivotRewriter
from ml_switcheroo.semantics.manager import SemanticsManager
from ml_switcheroo.core.escape_hatch import EscapeHatch
from ml_switcheroo.config import RuntimeConfig


class MockSemantics(SemanticsManager):
  """Mock Semantics class for testing purposes."""

  def __init__(self):
    """Initializes the MockSemantics instance."""
    self.data = {}
    self._reverse_index = {}
    self.import_data = {}
    self.framework_configs = {}
    self._inject("good_op", "torch.good", "jax.good")
    self._inject("bad_op", "torch.bad", None)

  def get_framework_config(self, framework: str):
    """Mock implementation of get framework configuration."""
    return self.framework_configs.get(framework, {})

  def _inject(self, name, s_api, t_api):
    """Mock implementation of  inject."""
    variants = {"torch": {"api": s_api}}
    if t_api:
      variants["jax"] = {"api": t_api}
    else:
      variants["jax"] = None
    self.data[name] = {"variants": variants, "std_args": ["x"]}
    self._reverse_index[s_api] = (name, self.data[name])


@pytest.fixture
def rewriter():
  """Provides a mock rewriter for testing."""
  semantics = MockSemantics()
  config = RuntimeConfig(source_framework="torch", target_framework="jax", strict_mode=True)
  return PivotRewriter(semantics, config)


def rewrite_code(rewriter, code):
  """Rewrites code."""
  tree = cst.parse_module(code)
  new_tree = rewriter.convert(tree)
  return new_tree.code


def test_verbatim_preservation_on_partial_failure(rewriter):
  """Verifies the behavior of verbatim preservation on partial successfully handling failure."""
  code = "res = torch.good(x) + torch.bad(y)"
  result = rewrite_code(rewriter, code)
  assert EscapeHatch.START_MARKER in result
  assert EscapeHatch.END_MARKER in result
  assert "torch.good(x)" in result
  assert "jax.good(x)" not in result
  assert "torch.bad(y)" in result


def test_end_marker_presence(rewriter):
  """Verifies the behavior of end marker presence."""
  code = "y = torch.bad(x)"
  result = rewrite_code(rewriter, code)
  assert EscapeHatch.START_MARKER in result
  assert EscapeHatch.END_MARKER in result
  assert "..." in result


def test_nested_call_failure_bubbling(rewriter):
  """Verifies the behavior of nested call successfully handling failure bubbling."""
  code = "y = torch.good(torch.bad(x))"
  result = rewrite_code(rewriter, code)
  assert "torch.good" in result
  assert "jax.good" not in result
  assert EscapeHatch.START_MARKER in result


def test_multiple_statements_handled_independently(rewriter):
  """Verifies the behavior of multiple statements handled independently."""
  code = "\ny = torch.bad(x)\nz = torch.good(x)\n"
  result = rewrite_code(rewriter, code)
  print(result)
  assert "torch.bad" in result
  assert result.count(EscapeHatch.START_MARKER) == 1
  assert "jax.good(x)" in result
  assert "torch.good" not in result
