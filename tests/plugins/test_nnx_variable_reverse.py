"""
Tests for Decoupled Parameter Conversion Plugin.
"""

import pytest
import libcst as cst
from unittest.mock import MagicMock
from ml_switcheroo.core.rewriter import PivotRewriter
from ml_switcheroo.config import RuntimeConfig
import ml_switcheroo.core.hooks as hooks
from ml_switcheroo.plugins.nnx_to_torch_params import transform_nnx_param
from ml_switcheroo.frameworks.base import register_framework


def rewrite_code(rewriter, code):
  return cst.parse_module(code).visit(rewriter).code


@pytest.fixture
def rewriter():
  hooks._HOOKS["nnx_param_to_torch"] = transform_nnx_param
  hooks._PLUGINS_LOADED = True
  mgr = MagicMock()

  # Define logic for custom framework
  # Map 'Param' -> 'custom.Parameter'
  op_def = {
    "variants": {
      "custom_fw": {"api": "custom.Parameter", "requires_plugin": "nnx_param_to_torch"},
      "torch": {"api": "torch.nn.Parameter", "requires_plugin": "nnx_param_to_torch"},
    }
  }

  mgr.get_definition.return_value = ("Param", op_def)
  mgr.resolve_variant.side_effect = lambda aid, fw: op_def["variants"].get(fw)

  # --- FIX: Register dummy framework 'custom_fw' ---
  @register_framework("custom_fw")
  class CustomFW:
    pass

  cfg = RuntimeConfig(source_framework="jax", target_framework="custom_fw")

  # Inject current op ID for context lookup
  rw = PivotRewriter(mgr, cfg)
  rw.ctx.current_op_id = "Param"
  return rw


def test_param_conversion_custom(rewriter):
  """Verify plugin respects context lookup for API name."""
  res = rewrite_code(rewriter, "w = nnx.Param(x)")
  assert "custom.Parameter(x)" in res


def test_batch_stat_conversion_custom(rewriter):
  """Verify BatchStat adds requires_grad=False for custom framework."""
  res = rewrite_code(rewriter, "m = nnx.BatchStat(z)")
  assert "custom.Parameter(z" in res
  assert "requires_grad=False" in res


def test_fallback_defaults(rewriter):
  """
  Verify graceful failure if API lookup returns None.

  Previously this defaulted to 'torch.nn.Parameter'.
  Now it should return the Original Node (Pass-through) to comply with decoupling.
  """
  # Prepare direct invocation
  code = "w = nnx.Param(x)"
  module = cst.parse_module(code)
  # Navigate to the value being assigned: w = nnx.Param(x)
  # module.body[0] -> SimpleStatementLine
  # module.body[0].body[0] -> Assign
  # Assign.value -> Call(nnx.Param)
  call_node = module.body[0].body[0].value

  # Force lookup failure in context
  rewriter.ctx.lookup_api = MagicMock(return_value=None)

  # Run hook
  res_node = transform_nnx_param(call_node, rewriter.ctx)

  # Re-wrap to verify string
  res_code = cst.Module(body=[cst.SimpleStatementLine([cst.Expr(res_node)])]).code

  # Expect original code, NOT torch fallback
  assert "nnx.Param(x)" in res_code
  assert "torch.nn.Parameter" not in res_code
