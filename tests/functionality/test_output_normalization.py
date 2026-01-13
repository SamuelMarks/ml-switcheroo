"""
Tests for Output Normalization Logic.

This module validates the mechanism for reconciling return signature differences
between frameworks using structured indexing (`output_select_index`).
Deprecated string-based lambda testing has been removed.
"""

import pytest
import libcst as cst
from tests.conftest import TestRewriter
from ml_switcheroo.semantics.manager import SemanticsManager
from ml_switcheroo.config import RuntimeConfig


class MockOutputSemantics(SemanticsManager):
  """
  Mock Manager injected with specific output index scenarios.
  """

  def __init__(self):
    self.data = {}
    self._reverse_index = {}
    self._key_origins = {}
    self.import_data = {}
    self.framework_configs = {}

    # Scenario: Structured Field (output_select_index)
    self._inject(
      "split_vals",
      "torch.split",
      "jax.numpy.split",
      select_index=0,
    )

  def get_framework_config(self, framework: str):
    return self.framework_configs.get(framework, {})

  def _inject(
    self,
    name,
    s_api,
    t_api,
    select_index=None,
  ):
    variants = {"torch": {"api": s_api}, "jax": {"api": t_api}}

    target_var = variants["jax"]
    if select_index is not None:
      target_var["output_select_index"] = select_index

    self.data[name] = {"variants": variants, "std_args": ["x", "y"]}
    self._reverse_index[s_api] = (name, self.data[name])


@pytest.fixture
def rewriter():
  """Creates a pre-configured TestRewriter for output testing."""
  semantics = MockOutputSemantics()
  config = RuntimeConfig(source_framework="torch", target_framework="jax", strict_mode=True)
  return TestRewriter(semantics, config)


def rewrite(rewriter, code):
  """Helper to run the rewriter on a code snippet."""
  tree = cst.parse_module(code)
  # Using pipeline conversion via helper
  new_tree = rewriter.convert(tree)
  return new_tree.code


def test_structured_index_wrapping(rewriter):
  """
  Scenario: `res = torch.split(x)`
  Semantics: `jax.numpy.split` accessed via `output_select_index=0`.
  Expectation: `res = jax.numpy.split(x)[0]` (Clean Subscript Syntax)
  """
  code = "res = torch.split(x)"
  result = rewrite(rewriter, code)

  # Should use subscription, not lambda
  assert "jax.numpy.split(x)[0]" in result
  assert "lambda" not in result
