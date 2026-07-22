"""Test suite for the Output Casting module."""

import pytest
import libcst as cst
from tests.conftest import TestRewriter as PivotRewriter
from ml_switcheroo.semantics.manager import SemanticsManager
from ml_switcheroo.config import RuntimeConfig


class MockCastSemantics(SemanticsManager):
  """Mock Cast Semantics class for testing purposes."""

  def __init__(self):
    """Initializes the MockCastSemantics instance."""
    self.data = {}
    self.import_data = {}
    self.framework_configs = {}
    self._reverse_index = {}
    self._key_origins = {}
    self.test_templates = {}
    self._known_rng_methods = set()
    self._inject("ArgMax", "torch.argmax", "jax.numpy.argmax", output_cast="jnp.int64")
    self._inject("Normalize", "torch.simple_op", "jax.op", output_cast="jnp.float32")

  def get_all_rng_methods(self):
    """Mock implementation of get all rng methods."""
    return set()

  def get_framework_config(self, framework):
    """Mock implementation of get framework configuration."""
    return {}

  def _inject(self, name, s_api, t_api, output_cast=None):
    """Mock implementation of  inject."""
    t_def = {"api": t_api}
    if output_cast:
      t_def["output_cast"] = output_cast
    variants = {"torch": {"api": s_api}, "jax": t_def}
    self.data[name] = {"variants": variants, "std_args": ["x"]}
    self._reverse_index[s_api] = (name, self.data[name])


@pytest.fixture
def rewriter():
  """Provides a mock rewriter for testing."""
  semantics = MockCastSemantics()
  config = RuntimeConfig(source_framework="torch", target_framework="jax", strict_mode=True)
  return PivotRewriter(semantics, config)


def rewrite(rewriter, code):
  """Rewrites ."""
  tree = cst.parse_module(code)
  new_tree = rewriter.convert(tree)
  return new_tree.code


def test_output_cast_injection(rewriter):
  """Verifies the behavior of output cast injection."""
  code = "y = torch.argmax(x)"
  result = rewrite(rewriter, code)
  assert "jax.numpy.argmax(x)" in result
  assert ".astype(jnp.int64)" in result


def test_output_cast_float_conversion(rewriter):
  """Verifies the behavior of output cast float conversion."""
  code = "z = torch.simple_op(x)"
  result = rewrite(rewriter, code)
  assert "jax.op(x)" in result
  assert ".astype(jnp.float32)" in result
