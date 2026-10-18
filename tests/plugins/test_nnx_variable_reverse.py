"""Test suite for the Nnx Variable Reverse module."""

import typing

from unittest.mock import MagicMock

import libcst as _cst
import pytest

import ml_switcheroo.core.hooks as hooks
from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo.frameworks.base import register_framework
from ml_switcheroo.plugins.nnx_to_torch_params import transform_nnx_param
from tests.conftest import TestRewriter as PivotRewriter

cst: typing.Any = _cst


def rewrite_code(rewriter: typing.Any, code: typing.Any) -> typing.Any:
  """Rewrites code.

  Args:
      rewriter (PivotRewriter): The rewriter instance.
      code (str): The code to rewrite.

  Returns:
      str: The rewritten code string.
  """
  return rewriter.convert(cst.parse_module(code)).code


@pytest.fixture
def rewriter() -> typing.Any:
  """Provides a mock rewriter for testing.

  Yields:
      PivotRewriter: A mock rewriter instance.
  """
  hooks._HOOKS["nnx_param_to_torch"] = transform_nnx_param
  hooks._PLUGINS_LOADED = True
  mgr: typing.Any = MagicMock()
  op_def: typing.Any = {
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
    """Test helper class CustomFW."""

    pass

  cfg: typing.Any = RuntimeConfig(source_framework="jax", target_framework="custom_fw")
  rw: typing.Any = PivotRewriter(mgr, cfg)
  rw.ctx.current_op_id = "Param"
  yield rw


def test_param_conversion_custom(rewriter: typing.Any) -> None:
  """Verifies the behavior of parameter conversion custom.

  Args:
      rewriter (PivotRewriter): The rewriter instance.
  """
  res: typing.Any = rewrite_code(rewriter, "w = nnx.Param(x)")
  assert "custom.Parameter(x)" in res


def test_batch_stat_conversion_custom(rewriter: typing.Any) -> None:
  """Verifies the behavior of batch statistic conversion custom.

  Args:
      rewriter (PivotRewriter): The rewriter instance.
  """
  res: typing.Any = rewrite_code(rewriter, "m = nnx.BatchStat(z)")
  assert "custom.Parameter(z" in res
  assert "requires_grad=False" in res


def test_fallback_defaults(rewriter: typing.Any) -> None:
  """Verifies the behavior of fallback defaults.

  Args:
      rewriter (PivotRewriter): The rewriter instance.
  """
  code: typing.Any = "w = nnx.Param(x)"
  module: typing.Any = cst.parse_module(code)
  call_node: typing.Any = module.body[0].body[0].value
  rewriter.ctx.lookup_api = MagicMock(return_value=None)
  res_node: typing.Any = transform_nnx_param(call_node, rewriter.ctx)
  res_code: typing.Any = cst.Module(body=[cst.SimpleStatementLine([cst.Expr(res_node)])]).code
  assert "nnx.Param(x)" in res_code
  assert "torch.nn.Parameter" not in res_code


def test_param_conversion_name(rewriter: typing.Any) -> None:
  """Verifies conversion when the function is a direct Name (e.g. Param(x)).

  Args:
      rewriter (PivotRewriter): The rewriter instance.
  """
  res: typing.Any = rewrite_code(rewriter, "w = Param(x)")
  assert "custom.Parameter(x)" in res


def test_param_conversion_unsupported_func(rewriter: typing.Any) -> None:
  """Verifies that the hook ignores calls with unsupported function types.

  Args:
      rewriter (PivotRewriter): The rewriter instance.
  """
  code: typing.Any = "w = func_list[0](x)"
  call_node: typing.Any = cst.parse_module(code).body[0].body[0].value
  res: typing.Any = transform_nnx_param(call_node, rewriter.ctx)
  assert isinstance(res, cst.Call)
