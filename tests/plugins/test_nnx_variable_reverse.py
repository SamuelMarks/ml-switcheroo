import pytest
import libcst as cst
from unittest.mock import MagicMock
from ml_switcheroo.core.rewriter import PivotRewriter
from ml_switcheroo.config import RuntimeConfig
import ml_switcheroo.core.hooks as hooks
from ml_switcheroo.plugins.nnx_to_torch_params import transform_nnx_param


def rewrite_code(rewriter, code):
  return cst.parse_module(code).visit(rewriter).code


@pytest.fixture
def rewriter():
  hooks._HOOKS["nnx_param_to_torch"] = transform_nnx_param
  hooks._PLUGINS_LOADED = True
  mgr = MagicMock()

  op_def = {"variants": {"torch": {"requires_plugin": "nnx_param_to_torch"}}}

  mgr.get_definition.return_value = ("param", op_def)
  mgr.resolve_variant.side_effect = lambda aid, fw: op_def["variants"].get(fw)

  cfg = RuntimeConfig(source_framework="jax", target_framework="torch")
  return PivotRewriter(mgr, cfg)


def test_param_conversion(rewriter):
  res = rewrite_code(rewriter, "w = nnx.Param(x)")
  assert "torch.nn.Parameter(x)" in res


def test_batch_stat_conversion(rewriter):
  res = rewrite_code(rewriter, "m = nnx.BatchStat(z)")
  assert "torch.nn.Parameter(z" in res
  assert "requires_grad=False" in res


def test_variable_conversion(rewriter):
  res = rewrite_code(rewriter, "v = flax.nnx.Variable(x)")
  assert "torch.nn.Parameter(x" in res


def test_ignore_wrong_target(rewriter):
  rewriter.ctx._runtime_config.target_framework = "numpy"
  rewriter.ctx.target_fw = "numpy"
  rewriter.semantics.resolve_variant.side_effect = lambda a, f: None
  assert "nnx.Param" in rewrite_code(rewriter, "w = nnx.Param(x)")
