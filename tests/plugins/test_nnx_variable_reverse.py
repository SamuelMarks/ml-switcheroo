"""
Tests for Flax NNX to PyTorch Parameter Conversion.

Verifies that:
1. `nnx.Param` is converted to `torch.nn.Parameter`.
2. `nnx.BatchStat` is converted to `torch.nn.Parameter(..., requires_grad=False)`.
3. The logic handles attribute access chains correctly.
"""

import pytest
from unittest.mock import MagicMock
import libcst as cst

from ml_switcheroo.core.rewriter import PivotRewriter
from ml_switcheroo.config import RuntimeConfig
import ml_switcheroo.core.hooks as hooks
from ml_switcheroo.plugins.nnx_to_torch_params import transform_nnx_param


# Helper to executing rewrite on a code snippet
def rewrite_code(rewriter: PivotRewriter, code: str) -> str:
  """
  Parses code, runs the rewriter visitor, and returns generated source.

  Args:
      rewriter: Configured PivotRewriter instance.
      code: Source python code string.

  Returns:
      str: Transformed python code.
  """
  tree = cst.parse_module(code)
  new_tree = tree.visit(rewriter)
  return new_tree.code


@pytest.fixture
def rewriter():
  """
  Creates a PivotRewriter configured with the 'nnx_param_to_torch' hook
  and mock semantics for NNX variables.
  """
  # 1. Register Hook & ensure plugins loaded state is set
  hooks._HOOKS["nnx_param_to_torch"] = transform_nnx_param
  hooks._PLUGINS_LOADED = True

  # 2. Setup Mock Semantics
  mgr = MagicMock()

  # Define behavior lookup
  # We restrict mapping to specific NNX classes to avoid greedily matching
  # inner primitives (like jax.random.normal) during the test.
  def get_def(name):
    if any(x in name for x in ["Param", "BatchStat", "Variable"]):
      base_name = name.split(".")[-1]
      return (
        base_name,
        {
          "variants": {
            "torch": {"requires_plugin": "nnx_param_to_torch"},
            "jax": {"api": name},
          }
        },
      )
    # Pass through unrelated calls (like jax.random.normal or jnp.zeros)
    # returning None prevents the Rewriter from trying to transform them via plugin
    return None

  mgr.get_definition.side_effect = get_def
  mgr.is_verified.return_value = True

  # 3. Config for JAX -> Torch
  cfg = RuntimeConfig(source_framework="jax", target_framework="torch")
  return PivotRewriter(semantics=mgr, config=cfg)


def test_param_conversion(rewriter):
  """
  Scenario: Converting a trainable parameter.
  Input: `self.w = nnx.Param(init_fn(rng, shape))`
  Output: `self.w = torch.nn.Parameter(init_fn(rng, shape))`
  """
  code = "self.w = nnx.Param(jax.random.normal(rng))"
  result = rewrite_code(rewriter, code)

  assert "torch.nn.Parameter" in result
  assert "requires_grad" not in result
  # Inner args should be preserved because mock semantics ignores 'jax.random.normal'
  assert "jax.random.normal(rng)" in result


def test_batch_stat_conversion(rewriter):
  """
  Scenario: Converting a non-trainable state (BatchStat).
  Input: `self.running_mean = nnx.BatchStat(zeros)`
  Output: `self.running_mean = torch.nn.Parameter(zeros, requires_grad=False)`
  """
  code = "self.running_mean = nnx.BatchStat(jnp.zeros(10))"
  result = rewrite_code(rewriter, code)

  assert "torch.nn.Parameter" in result
  assert "requires_grad=False" in result
  assert "jnp.zeros(10)" in result


def test_variable_conversion(rewriter):
  """
  Scenario: Converting generic Variable (assumed non-trainable in this context).
  Input: `self.v = flax.nnx.Variable(x)`
  Output: `self.v = torch.nn.Parameter(x, requires_grad=False)`
  """
  code = "self.v = flax.nnx.Variable(x)"
  result = rewrite_code(rewriter, code)

  assert "torch.nn.Parameter" in result
  assert "requires_grad=False" in result


def test_ignore_wrong_target(rewriter):
  """
  Scenario: Target is not Torch (e.g. converting JAX to NumPy).
  Expectation: No change (Plugin returns original node).
  """
  # Reconfigure rewriter for numpy target
  rewriter.config = RuntimeConfig(source_framework="jax", target_framework="numpy")
  # Need to update rewriter internal state which may cache config
  rewriter.target_fw = "numpy"
  # Update context as well
  rewriter.ctx._runtime_config.target_framework = "numpy"
  rewriter.ctx.target_fw = "numpy"

  code = "self.w = nnx.Param(x)"
  result = rewrite_code(rewriter, code)

  assert "nnx.Param(x)" in result
  assert "torch.nn.Parameter" not in result
