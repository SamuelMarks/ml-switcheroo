"""
Tests for Output Dtype Casting (Feature 12).

Verifies that the rewriter injects `.astype(...)` logic when `output_cast`
is defined in the semantics mapping.
"""

import pytest
import libcst as cst
from unittest.mock import MagicMock

# Fix: Import TestRewriter shim
from tests.conftest import TestRewriter as PivotRewriter

from ml_switcheroo.semantics.manager import SemanticsManager
from ml_switcheroo.config import RuntimeConfig


class MockCastSemantics(SemanticsManager):
  """Mock Manager with Output Cast mappings."""

  def __init__(self):
    self.data = {}
    self.import_data = {}
    self.framework_configs = {}
    self._reverse_index = {}
    self._key_origins = {}
    self.test_templates = {}
    self._known_rng_methods = set()

    # 1. ArgMax -> Needs int64
    self._inject("ArgMax", "torch.argmax", "jax.numpy.argmax", output_cast="jnp.int64")

    # 2. Simple Math -> Cast to float32
    self._inject("Normalize", "torch.simple_op", "jax.op", output_cast="jnp.float32")

  def get_all_rng_methods(self):
    return set()

  def get_framework_config(self, framework):
    return {}

  def _inject(self, name, s_api, t_api, output_cast=None):
    t_def = {"api": t_api}
    if output_cast:
      t_def["output_cast"] = output_cast

    variants = {"torch": {"api": s_api}, "jax": t_def}
    self.data[name] = {"variants": variants, "std_args": ["x"]}
    self._reverse_index[s_api] = (name, self.data[name])


@pytest.fixture
def rewriter():
  semantics = MockCastSemantics()
  config = RuntimeConfig(source_framework="torch", target_framework="jax", strict_mode=True)
  return PivotRewriter(semantics, config)


def rewrite(rewriter, code):
  tree = cst.parse_module(code)
  # Fix: Use pipeline conversion
  new_tree = rewriter.convert(tree)
  return new_tree.code


def test_output_cast_injection(rewriter):
  """
  Scenario: torch.argmax(x) -> jax.numpy.argmax(x).astype(jnp.int64)
  """
  code = "y = torch.argmax(x)"
  result = rewrite(rewriter, code)

  assert "jax.numpy.argmax(x)" in result
  assert ".astype(jnp.int64)" in result


def test_output_cast_float_conversion(rewriter):
  """
  Scenario: torch.simple_op(x) -> jax.op(x).astype(jnp.float32)
  """
  code = "z = torch.simple_op(x)"
  result = rewrite(rewriter, code)

  assert "jax.op(x)" in result
  assert ".astype(jnp.float32)" in result
