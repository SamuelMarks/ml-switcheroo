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

cst = _cst


def rewrite_code(rewriter: PivotRewriter, code: str) -> str:
  """Rewrites code.

  Args:
      rewriter (PivotRewriter): The rewriter instance.
      code (str): The code to rewrite.

  Returns:
      str: The rewritten code string.
  """
  return rewriter.convert(cst.parse_module(code)).code


@pytest.fixture
def rewriter() -> typing.Iterator[PivotRewriter]:
  """Provides a mock rewriter for testing.

  Yields:
      PivotRewriter: A mock rewriter instance.
  """
  hooks.register_hook("nnx_param_to_torch")(transform_nnx_param)
  hooks.set_plugins_loaded(True)
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

  from ml_switcheroo.frameworks.base import get_default_adapter_registry

  @register_framework("custom_fw")
  class CustomFW:
    """Sentinel `CustomFW` fixture class used in NNX variable reverse tests."""

  try:
    cfg = RuntimeConfig(source_framework="jax", target_framework="custom_fw")
    rw = PivotRewriter(mgr, cfg)
    rw.ctx.current_op_id = "Param"
    yield rw
  finally:
    get_default_adapter_registry().adapters.pop("custom_fw", None)


def test_param_conversion_custom(rewriter: PivotRewriter) -> None:
  """Verify `rewrite_code` emits `custom.Parameter(x)` for param conversion custom.

  Args:
      rewriter (PivotRewriter): The rewriter instance.
  """
  res = rewrite_code(rewriter, "w = nnx.Param(x)")
  assert "custom.Parameter(x)" in res


def test_batch_stat_conversion_custom(rewriter: PivotRewriter) -> None:
  """Verify `rewrite_code` emits `custom.Parameter(z` for batch stat conversion custom.

  Args:
      rewriter (PivotRewriter): The rewriter instance.
  """
  res = rewrite_code(rewriter, "m = nnx.BatchStat(z)")
  assert "custom.Parameter(z" in res
  assert "requires_grad=False" in res


def test_fallback_defaults(rewriter: PivotRewriter, monkeypatch: pytest.MonkeyPatch) -> None:
  """Verify `ensure_type` emits `nnx.Param(x)` for fallback defaults.

  Args:
      rewriter (PivotRewriter): The rewriter instance.
      monkeypatch: Pytest monkeypatch fixture.
  """
  code = "w = nnx.Param(x)"
  module = cst.parse_module(code)
  stmt = cst.ensure_type(module.body[0], cst.SimpleStatementLine)
  assign = cst.ensure_type(stmt.body[0], cst.Assign)
  call_node = cst.ensure_type(assign.value, cst.Call)
  monkeypatch.setattr(rewriter.ctx, "lookup_api", MagicMock(return_value=None), raising=False)
  res_node = transform_nnx_param(call_node, rewriter.ctx)
  res_code = cst.Module(body=[cst.SimpleStatementLine([cst.Expr(res_node)])]).code
  assert "nnx.Param(x)" in res_code
  assert "torch.nn.Parameter" not in res_code


def test_param_conversion_name(rewriter: PivotRewriter) -> None:
  """Verifies conversion when the function is a direct Name (e.g. Param(x)).

  Args:
      rewriter (PivotRewriter): The rewriter instance.
  """
  res = rewrite_code(rewriter, "w = Param(x)")
  assert "custom.Parameter(x)" in res


def test_param_conversion_unsupported_func(rewriter: PivotRewriter) -> None:
  """Verifies that the hook ignores calls with unsupported function types.

  Args:
      rewriter (PivotRewriter): The rewriter instance.
  """
  code = "w = func_list[0](x)"
  stmt = cst.ensure_type(cst.parse_module(code).body[0], cst.SimpleStatementLine)
  assign = cst.ensure_type(stmt.body[0], cst.Assign)
  call_node = cst.ensure_type(assign.value, cst.Call)
  res = transform_nnx_param(call_node, rewriter.ctx)
  assert isinstance(res, cst.Call)
  assert cst.Module([]).code_for_node(res) == "custom.Parameter(x)"
