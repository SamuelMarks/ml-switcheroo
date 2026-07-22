"""Test suite for the Output Normalization module."""

import pytest
import libcst as cst
from tests.conftest import TestRewriter
from ml_switcheroo.semantics.manager import SemanticsManager
from ml_switcheroo.config import RuntimeConfig


class MockOutputSemantics(SemanticsManager):
  """Mock Output Semantics class for testing purposes."""

  def __init__(self):
    """Initializes the MockOutputSemantics instance."""
    self.data = {}
    self._reverse_index = {}
    self._key_origins = {}
    self.import_data = {}
    self.framework_configs = {}
    self._inject("split_vals", "torch.split", "jax.numpy.split", select_index=0)

  def get_framework_config(self, framework: str):
    """Mock implementation of get framework configuration."""
    return self.framework_configs.get(framework, {})

  def _inject(self, name, s_api, t_api, select_index=None):
    """Mock implementation of  inject."""
    variants = {"torch": {"api": s_api}, "jax": {"api": t_api}}
    target_var = variants["jax"]
    if select_index is not None:
      target_var["output_select_index"] = select_index
    self.data[name] = {"variants": variants, "std_args": ["x", "y"]}
    self._reverse_index[s_api] = (name, self.data[name])


@pytest.fixture
def rewriter():
  """Provides a mock rewriter for testing."""
  semantics = MockOutputSemantics()
  config = RuntimeConfig(source_framework="torch", target_framework="jax", strict_mode=True)
  return TestRewriter(semantics, config)


def rewrite(rewriter, code):
  """Rewrites ."""
  tree = cst.parse_module(code)
  new_tree = rewriter.convert(tree)
  return new_tree.code


def test_structured_index_wrapping(rewriter):
  """Verifies the behavior of structured index wrapping."""
  code = "res = torch.split(x)"
  result = rewrite(rewriter, code)
  assert "jax.numpy.split(x)[0]" in result
  assert "lambda" not in result
