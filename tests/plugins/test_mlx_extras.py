"""Test suite for the Mlx Extras module."""

import typing
from unittest.mock import MagicMock

import libcst as _cst
import pytest
from pydantic import JsonValue

import ml_switcheroo.core.hooks as hooks
from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo.frameworks.base import register_framework
from ml_switcheroo.plugins.mlx_extras import transform_compiler, transform_synchronize
from tests.conftest import TestRewriter as PivotRewriter

cst = _cst


@pytest.fixture
def rewriter() -> typing.Iterator[PivotRewriter]:
  """Provides a mock rewriter for testing.

  Yields:
      PivotRewriter: A mock rewriter instance.
  """
  hooks.register_hook("mlx_compiler")(transform_compiler)
  hooks.register_hook("mlx_synchronize")(transform_synchronize)
  hooks.set_plugins_loaded(True)
  mgr = MagicMock()
  comp_variants: dict[str, dict[str, JsonValue]] = {"custom_fw": {"api": "custom.jit", "requires_plugin": "mlx_compiler"}}
  sync_variants: dict[str, dict[str, JsonValue]] = {"custom_fw": {"requires_plugin": "mlx_synchronize"}}
  comp_def: dict[str, JsonValue] = {"variants": dict[str, JsonValue](comp_variants)}
  sync_def: dict[str, JsonValue] = {"variants": dict[str, JsonValue](sync_variants)}

  def get_def(name: str) -> tuple[str, dict[str, JsonValue]] | None:
    """Gets def.

    Args:
        name (str): Definition name.

    Returns:
        Optional[Tuple[str, dict[str, JsonValue]]]: Definition tuple.
    """
    if "compile" in name:
      return ("Compile", comp_def)
    if "synchronize" in name:
      return ("Synchronize", sync_def)
    return None

  mgr.get_definition.side_effect = get_def

  def resolve(aid: str, fw: str) -> dict[str, JsonValue] | None:
    """Resolves variant.

    Args:
        aid (str): Definition ID.
        fw (str): Framework string.

    Returns:
        Optional[dict[str, JsonValue]]: Variant definition.
    """
    if aid == "Compile":
      return comp_variants.get(fw)
    if aid == "Synchronize":
      return sync_variants.get(fw)
    return None

  mgr.resolve_variant.side_effect = resolve
  mgr.get_framework_config.return_value = {}

  from ml_switcheroo.frameworks.base import get_default_adapter_registry

  @register_framework("custom_fw")
  class CustomFW:
    """Sentinel `CustomFW` fixture class used in MLX extras tests."""

  try:
    cfg = RuntimeConfig(source_framework="torch", target_framework="custom_fw")
    yield PivotRewriter(mgr, cfg)
  finally:
    get_default_adapter_registry().adapters.pop("custom_fw", None)


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


def test_compiler_decorator(rewriter: PivotRewriter, monkeypatch: pytest.MonkeyPatch) -> None:
  """Verify `transform_compiler` emits `@custom.jit` for compiler decorator.

  Args:
      rewriter (PivotRewriter): The rewriter instance.
      monkeypatch: Pytest monkeypatch fixture.
  """
  code = "@torch.compile(fullgraph=True)\ndef f(x): pass"
  module = cst.parse_module(code)
  fn_def = cst.ensure_type(module.body[0], cst.FunctionDef)
  decorator = fn_def.decorators[0]
  monkeypatch.setattr(rewriter.ctx, "lookup_api", MagicMock(return_value="custom.jit"), raising=False)
  new_dec = cst.ensure_type(transform_compiler(decorator, rewriter.ctx), cst.Decorator)
  res = cst.Module(body=[fn_def.with_changes(decorators=[new_dec])]).code
  assert "@custom.jit" in res
  assert "fullgraph" not in res


def test_compiler_functional(rewriter: PivotRewriter, monkeypatch: pytest.MonkeyPatch) -> None:
  """Verify `ensure_type` emits `custom.jit(fn)` for compiler functional.

  Args:
      rewriter (PivotRewriter): The rewriter instance.
      monkeypatch: Pytest monkeypatch fixture.
  """
  code = "opt_fn = torch.compile(fn)"
  stmt = cst.ensure_type(cst.parse_module(code).body[0], cst.SimpleStatementLine)
  assign = cst.ensure_type(stmt.body[0], cst.Assign)
  call_node = cst.ensure_type(assign.value, cst.Call)
  monkeypatch.setattr(rewriter.ctx, "lookup_api", MagicMock(return_value="custom.jit"), raising=False)
  res_node = cst.ensure_type(transform_compiler(call_node, rewriter.ctx), cst.Call)
  res = cst.Module(body=[cst.SimpleStatementLine([cst.Expr(res_node)])]).code
  assert "custom.jit(fn)" in res


def test_sync_warning(rewriter: PivotRewriter) -> None:
  """Verify `transform_synchronize` emits `print(` for sync warning.

  Args:
      rewriter (PivotRewriter): The rewriter instance.
  """
  code = "torch.cuda.synchronize()"
  call_node = cst.parse_expression(code)
  assert isinstance(call_node, cst.Call)
  res_node = cst.ensure_type(transform_synchronize(call_node, rewriter.ctx), cst.Call)
  res = cst.Module(body=[cst.SimpleStatementLine([cst.Expr(res_node)])]).code
  assert "print(" in res
  assert "Global sync requires explicit" in res


def test_compiler_invalid_node(rewriter: PivotRewriter, monkeypatch: pytest.MonkeyPatch) -> None:
  """Verifies that the hook returns the original node if it's not a Decorator or Call.

  Args:
      rewriter (PivotRewriter): The rewriter instance.
      monkeypatch: Pytest monkeypatch fixture.
  """
  node = cst.Name("torch_compile")
  monkeypatch.setattr(rewriter.ctx, "lookup_api", MagicMock(return_value="custom.jit"), raising=False)
  res_node = transform_compiler(node, rewriter.ctx)
  assert res_node is node
