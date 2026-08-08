"""Test suite for the Nnx Variable Reverse module."""

import pytest
import libcst as cst
from unittest.mock import MagicMock
from tests.conftest import TestRewriter as PivotRewriter
from ml_switcheroo.config import RuntimeConfig
import ml_switcheroo.core.hooks as hooks
from ml_switcheroo.plugins.nnx_to_torch_params import transform_nnx_param
from ml_switcheroo.frameworks.base import register_framework


def rewrite_code(rewriter, code):
  """Rewrites code."""
  return rewriter.convert(cst.parse_module(code)).code


@pytest.fixture
def rewriter():
  """Provides a mock rewriter for testing."""
  hooks._HOOKS["nnx_param_to_torch"] = transform_nnx_param
  hooks._PLUGINS_LOADED = True
  mgr = MagicMock()
  op_def = {
    "variants": {
      "custom_fw": {"api": "custom.Parameter", "requires_plugin": "nnx_param_to_torch"},
      "torch": {"api": "torch.nn.Parameter", "requires_plugin": "nnx_param_to_torch"},
    }
  }
  mgr.get_definition.return_value = ("Param", op_def)
  mgr.resolve_variant.side_effect = lambda aid, fw: op_def["variants"].get(fw)
  mgr.get_framework_config.return_value = {}

  @register_framework("custom_fw")
  class CustomFW:
    """Test suite for the Custom F W component."""

    pass

  cfg = RuntimeConfig(source_framework="jax", target_framework="custom_fw")
  rw = PivotRewriter(mgr, cfg)
  rw.ctx.current_op_id = "Param"
  return rw


def test_param_conversion_custom(rewriter):
  """Verifies the behavior of parameter conversion custom."""
  res = rewrite_code(rewriter, "w = nnx.Param(x)")
  assert "custom.Parameter(x)" in res


def test_batch_stat_conversion_custom(rewriter):
  """Verifies the behavior of batch statistic conversion custom."""
  res = rewrite_code(rewriter, "m = nnx.BatchStat(z)")
  assert "custom.Parameter(z" in res
  assert "requires_grad=False" in res


def test_fallback_defaults(rewriter):
  """Verifies the behavior of fallback defaults."""
  code = "w = nnx.Param(x)"
  module = cst.parse_module(code)
  call_node = module.body[0].body[0].value
  rewriter.ctx.lookup_api = MagicMock(return_value=None)
  res_node = transform_nnx_param(call_node, rewriter.ctx)
  res_code = cst.Module(body=[cst.SimpleStatementLine([cst.Expr(res_node)])]).code
  assert "nnx.Param(x)" in res_code
  assert "torch.nn.Parameter" not in res_code


def test_param_conversion_name(rewriter):
  """Verifies conversion when the function is a direct Name (e.g. Param(x))."""
  res = rewrite_code(rewriter, "w = Param(x)")
  assert "custom.Parameter(x)" in res


def test_param_conversion_unsupported_func(rewriter):
  """Verifies that the hook ignores calls with unsupported function types."""
  code = "w = func_list[0](x)"
  call_node = cst.parse_module(code).body[0].body[0].value
  res = transform_nnx_param(call_node, rewriter.ctx)
  assert isinstance(res, cst.Call)
