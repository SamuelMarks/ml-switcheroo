"""Test suite for the Mlx Extras module."""

import typing
from pydantic import JsonValue

from unittest.mock import MagicMock

import libcst as _cst
import pytest

import ml_switcheroo.core.hooks as hooks
from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo.frameworks.base import register_framework
from ml_switcheroo.plugins.mlx_extras import transform_compiler, transform_synchronize
from tests.conftest import TestRewriter as PivotRewriter

cst = _cst


@pytest.fixture
def rewriter() -> PivotRewriter:  # type: ignore[misc]
  """Provides a mock rewriter for testing.

  Yields:
      PivotRewriter: A mock rewriter instance.
  """
  hooks._HOOKS["mlx_compiler"] = transform_compiler
  hooks._HOOKS["mlx_synchronize"] = transform_synchronize  # type: ignore[assignment]
  hooks._PLUGINS_LOADED = True
  mgr = MagicMock()
  comp_def = {"variants": {"custom_fw": {"api": "custom.jit", "requires_plugin": "mlx_compiler"}}}
  sync_def = {"variants": {"custom_fw": {"requires_plugin": "mlx_synchronize"}}}

  def get_def(name: str) -> typing.Optional[tuple[str, dict[str, JsonValue]]]:
    """Gets def.

    Args:
        name (str): Definition name.

    Returns:
        Optional[Tuple[str, dict[str, JsonValue]]]: Definition tuple.
    """
    if "compile" in name:
      return ("Compile", comp_def)  # type: ignore[return-value]
    if "synchronize" in name:
      return ("Synchronize", sync_def)  # type: ignore[return-value]
    return None

  mgr.get_definition.side_effect = get_def

  def resolve(aid: str, fw: str) -> typing.Optional[dict[str, JsonValue]]:
    """Resolves variant.

    Args:
        aid (str): Definition ID.
        fw (str): Framework string.

    Returns:
        Optional[dict[str, JsonValue]]: Variant definition.
    """
    if aid == "Compile":
      return comp_def["variants"].get(fw)  # type: ignore[return-value]
    if aid == "Synchronize":
      return sync_def["variants"].get(fw)  # type: ignore[return-value]
    return None

  mgr.resolve_variant.side_effect = resolve
  mgr.get_framework_config.return_value = {}

  @register_framework("custom_fw")
  class CustomFW:
    """Test helper class CustomFW."""

    pass

  cfg = RuntimeConfig(source_framework="torch", target_framework="custom_fw")
  yield PivotRewriter(mgr, cfg)


def rewrite(rewriter: PivotRewriter, code: str) -> str:
  """Rewrites code.

  Args:
      rewriter (PivotRewriter): The rewriter instance.
      code (str): The code.

  Returns:
      str: The rewritten code string.
  """
  mod = cst.parse_module(code)
  return rewriter.convert(mod).code


def test_compiler_decorator(rewriter: PivotRewriter) -> None:
  """Verifies the behavior of compiler decorator.

  Args:
      rewriter (PivotRewriter): The rewriter instance.
  """
  code = "@torch.compile(fullgraph=True)\ndef f(x): pass"
  module = cst.parse_module(code)
  decorator = module.body[0].decorators[0]  # type: ignore[union-attr]
  setattr(rewriter.ctx, "lookup_api", MagicMock(return_value="custom.jit"))
  new_dec = transform_compiler(decorator, rewriter.ctx)
  res = cst.Module(body=[module.body[0].with_changes(decorators=[new_dec])]).code
  assert "@custom.jit" in res
  assert "fullgraph" not in res


def test_compiler_functional(rewriter: PivotRewriter) -> None:
  """Verifies the behavior of compiler functional.

  Args:
      rewriter (PivotRewriter): The rewriter instance.
  """
  code = "opt_fn = torch.compile(fn)"
  call_node = cst.ensure_type(getattr(getattr(cst.parse_module(code).body[0], "body")[0], "value"), cst.Call)
  setattr(rewriter.ctx, "lookup_api", MagicMock(return_value="custom.jit"))
  res_node = transform_compiler(call_node, rewriter.ctx)
  res = cst.Module(body=[cst.SimpleStatementLine([cst.Expr(res_node)])]).code  # type: ignore[arg-type]
  assert "custom.jit(fn)" in res


def test_sync_warning(rewriter: PivotRewriter) -> None:
  """Verifies the behavior of sync warning.

  Args:
      rewriter (PivotRewriter): The rewriter instance.
  """
  code = "torch.cuda.synchronize()"
  call_node = cst.parse_expression(code)
  assert isinstance(call_node, cst.Call)
  res_node = transform_synchronize(call_node, rewriter.ctx)
  res = cst.Module(body=[cst.SimpleStatementLine([cst.Expr(res_node)])]).code  # type: ignore[arg-type]
  assert "print(" in res
  assert "Global sync requires explicit" in res


def test_compiler_invalid_node(rewriter: PivotRewriter) -> None:
  """Verifies that the hook returns the original node if it's not a Decorator or Call.

  Args:
      rewriter (PivotRewriter): The rewriter instance.
  """
  node = cst.Name("torch_compile")
  setattr(rewriter.ctx, "lookup_api", MagicMock(return_value="custom.jit"))
  res_node = transform_compiler(node, rewriter.ctx)
  assert res_node is node
