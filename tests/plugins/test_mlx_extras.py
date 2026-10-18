"""Test suite for the Mlx Extras module."""

import typing

from unittest.mock import MagicMock

import libcst as _cst
import pytest

import ml_switcheroo.core.hooks as hooks
from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo.frameworks.base import register_framework
from ml_switcheroo.plugins.mlx_extras import transform_compiler, transform_synchronize
from tests.conftest import TestRewriter as PivotRewriter

cst: typing.Any = _cst


@pytest.fixture
def rewriter() -> typing.Any:
  """Provides a mock rewriter for testing.

  Yields:
      PivotRewriter: A mock rewriter instance.
  """
  hooks._HOOKS["mlx_compiler"] = transform_compiler
  hooks._HOOKS["mlx_synchronize"] = transform_synchronize
  hooks._PLUGINS_LOADED = True
  mgr: typing.Any = MagicMock()
  comp_def: typing.Any = {"variants": {"custom_fw": {"api": "custom.jit", "requires_plugin": "mlx_compiler"}}}
  sync_def: typing.Any = {"variants": {"custom_fw": {"requires_plugin": "mlx_synchronize"}}}

  def get_def(name: typing.Any) -> typing.Any:
    """Gets def.

    Args:
        name (str): Definition name.

    Returns:
        Optional[Tuple[str, Dict[str, Any]]]: Definition tuple.
    """
    if "compile" in name:
      return ("Compile", comp_def)
    if "synchronize" in name:
      return ("Synchronize", sync_def)
    return None

  mgr.get_definition.side_effect = get_def

  def resolve(aid: typing.Any, fw: typing.Any) -> typing.Any:
    """Resolves variant.

    Args:
        aid (str): Definition ID.
        fw (str): Framework string.

    Returns:
        Optional[Dict[str, Any]]: Variant definition.
    """
    if aid == "Compile":
      return comp_def["variants"].get(fw)
    if aid == "Synchronize":
      return sync_def["variants"].get(fw)
    return None

  mgr.resolve_variant.side_effect = resolve
  mgr.get_framework_config.return_value = {}

  @register_framework("custom_fw")
  class CustomFW:
    """Test helper class CustomFW."""

    pass

  cfg: typing.Any = RuntimeConfig(source_framework="torch", target_framework="custom_fw")
  yield PivotRewriter(mgr, cfg)


def rewrite(rewriter: typing.Any, code: typing.Any) -> typing.Any:
  """Rewrites code.

  Args:
      rewriter (PivotRewriter): The rewriter instance.
      code (str): The code.

  Returns:
      str: The rewritten code string.
  """
  mod: typing.Any = cst.parse_module(code)
  return rewriter.convert(mod).code


def test_compiler_decorator(rewriter: typing.Any) -> None:
  """Verifies the behavior of compiler decorator.

  Args:
      rewriter (PivotRewriter): The rewriter instance.
  """
  code: typing.Any = "@torch.compile(fullgraph=True)\ndef f(x): pass"
  module: typing.Any = cst.parse_module(code)
  decorator: typing.Any = module.body[0].decorators[0]
  rewriter.ctx.lookup_api = MagicMock(return_value="custom.jit")
  new_dec: typing.Any = transform_compiler(decorator, rewriter.ctx)
  res: typing.Any = cst.Module(body=[module.body[0].with_changes(decorators=[new_dec])]).code
  assert "@custom.jit" in res
  assert "fullgraph" not in res


def test_compiler_functional(rewriter: typing.Any) -> None:
  """Verifies the behavior of compiler functional.

  Args:
      rewriter (PivotRewriter): The rewriter instance.
  """
  code: typing.Any = "opt_fn = torch.compile(fn)"
  call_node: typing.Any = cst.parse_module(code).body[0].body[0].value
  rewriter.ctx.lookup_api = MagicMock(return_value="custom.jit")
  res_node: typing.Any = transform_compiler(call_node, rewriter.ctx)
  res: typing.Any = cst.Module(body=[cst.SimpleStatementLine([cst.Expr(res_node)])]).code
  assert "custom.jit(fn)" in res


def test_sync_warning(rewriter: typing.Any) -> None:
  """Verifies the behavior of sync warning.

  Args:
      rewriter (PivotRewriter): The rewriter instance.
  """
  code: typing.Any = "torch.cuda.synchronize()"
  call_node: typing.Any = cst.parse_expression(code)
  res_node: typing.Any = transform_synchronize(call_node, rewriter.ctx)
  res: typing.Any = cst.Module(body=[cst.SimpleStatementLine([cst.Expr(res_node)])]).code
  assert "print(" in res
  assert "Global sync requires explicit" in res


def test_compiler_invalid_node(rewriter: typing.Any) -> None:
  """Verifies that the hook returns the original node if it's not a Decorator or Call.

  Args:
      rewriter (PivotRewriter): The rewriter instance.
  """
  node: typing.Any = cst.Name("torch_compile")
  rewriter.ctx.lookup_api = MagicMock(return_value="custom.jit")
  res_node: typing.Any = transform_compiler(node, rewriter.ctx)
  assert res_node is node
