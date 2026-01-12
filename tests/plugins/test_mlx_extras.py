"""
Tests for MLX Extras Plugin (Decoupled).
"""

import pytest
import libcst as cst
from unittest.mock import MagicMock

# Fix: Import TestRewriter shim
from tests.conftest import TestRewriter as PivotRewriter

from ml_switcheroo.config import RuntimeConfig
import ml_switcheroo.core.hooks as hooks
from ml_switcheroo.plugins.mlx_extras import transform_compiler, transform_synchronize
from ml_switcheroo.frameworks.base import register_framework


@pytest.fixture
def rewriter():
  hooks._HOOKS["mlx_compiler"] = transform_compiler
  hooks._HOOKS["mlx_synchronize"] = transform_synchronize
  hooks._PLUGINS_LOADED = True

  mgr = MagicMock()

  # Mock definitions for a generic "custom_fw" target
  comp_def = {"variants": {"custom_fw": {"api": "custom.jit", "requires_plugin": "mlx_compiler"}}}

  sync_def = {"variants": {"custom_fw": {"requires_plugin": "mlx_synchronize"}}}

  def get_def(name):
    if "compile" in name:
      return "Compile", comp_def
    if "synchronize" in name:
      return "Synchronize", sync_def
    return None

  mgr.get_definition.side_effect = get_def

  def resolve(aid, fw):
    if aid == "Compile":
      return comp_def["variants"].get(fw)
    if aid == "Synchronize":
      return sync_def["variants"].get(fw)
    return None

  mgr.resolve_variant.side_effect = resolve
  mgr.get_framework_config.return_value = {}

  # --- FIX: Register dummy framework ---
  @register_framework("custom_fw")
  class CustomFW:
    pass

  cfg = RuntimeConfig(source_framework="torch", target_framework="custom_fw")
  return PivotRewriter(mgr, cfg)


def rewrite(rewriter, code):
  mod = cst.parse_module(code)
  # Fix: Pipeline execution
  return rewriter.convert(mod).code


def test_compiler_decorator(rewriter):
  """
  Verify decorator replacement using looked up API.
  Input: @torch.compile(args)
  Output: @custom.jit
  """
  code = "@torch.compile(fullgraph=True)\ndef f(x): pass"

  # Manually invoke hook
  module = cst.parse_module(code)
  decorator = module.body[0].decorators[0]

  rewriter.ctx.lookup_api = MagicMock(return_value="custom.jit")

  new_dec = transform_compiler(decorator, rewriter.ctx)

  # Render result
  res = cst.Module(body=[module.body[0].with_changes(decorators=[new_dec])]).code

  assert "@custom.jit" in res
  assert "fullgraph" not in res  # Args stripped


def test_compiler_functional(rewriter):
  """Verify functional wrapper replacement via direct hook."""
  code = "opt_fn = torch.compile(fn)"
  call_node = cst.parse_module(code).body[0].body[0].value

  rewriter.ctx.lookup_api = MagicMock(return_value="custom.jit")

  res_node = transform_compiler(call_node, rewriter.ctx)

  res = cst.Module(body=[cst.SimpleStatementLine([cst.Expr(res_node)])]).code

  assert "custom.jit(fn)" in res


def test_sync_warning(rewriter):
  """Verify sync becomes a print warning."""
  code = "torch.cuda.synchronize()"
  call_node = cst.parse_expression(code)

  res_node = transform_synchronize(call_node, rewriter.ctx)
  res = cst.Module(body=[cst.SimpleStatementLine([cst.Expr(res_node)])]).code

  assert "print(" in res
  assert "Global sync requires explicit" in res
