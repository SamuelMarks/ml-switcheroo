"""Test suite for the Scatter module."""

from unittest.mock import MagicMock

import libcst as cst
import pytest
from pydantic import JsonValue

import ml_switcheroo.core.hooks as hooks
from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo.core.hooks import HookContext
from ml_switcheroo.plugins.scatter import transform_scatter
from tests.conftest import TestRewriter as PivotRewriter


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
def rewriter() -> PivotRewriter:
  """Provides a mock rewriter for testing.

  Returns:
      PivotRewriter: A mock rewriter instance.
  """
  hooks.register_hook("scatter_indexer")(transform_scatter)
  hooks.set_plugins_loaded(True)
  mgr: MagicMock = MagicMock()
  jax_variant: dict[str, JsonValue] = {"api": "at_set", "requires_plugin": "scatter_indexer"}
  scatter_def: dict[str, JsonValue] = {
    "variants": {
      "torch": {"api": "torch.Tensor.scatter_"},
      "jax": jax_variant,
    }
  }

  def get_def(name: str) -> tuple[str, dict[str, JsonValue]] | None:
    """Gets def.

    Args:
        name (str): Definition name.

    Returns:
        Optional[Tuple[str, dict[str, JsonValue]]]: Definition tuple.
    """
    if "scatter" in name:
      _ret_v: tuple[str, dict[str, JsonValue]] | None = (
        "Scatter",
        scatter_def,
      )
      return _ret_v
    return None

  mgr.get_definition.side_effect = get_def
  mgr.resolve_variant.side_effect = lambda aid, fw: jax_variant if fw == "jax" else None
  mgr.is_verified.return_value = True
  mgr.get_known_apis.return_value = {"Scatter": scatter_def}
  mgr.get_framework_config.return_value = {}
  cfg: RuntimeConfig = RuntimeConfig(source_framework="torch", target_framework="jax")
  return PivotRewriter(mgr, cfg)


def test_scatter_simple_rewrite(rewriter: PivotRewriter) -> None:
  """Verify `rewrite_code` emits `x.at[idx]` for scatter simple rewrite.

  Args:
      rewriter (PivotRewriter): The rewriter instance.
  """
  code: str = "res = x.scatter_(1, idx, src)"
  res: str = rewrite_code(rewriter, code)
  assert "x.at[idx]" in res
  assert ".set(src)" in res
  assert ", 1," not in res
  assert "(1," not in res


def test_scatter_add_rewrite(rewriter: PivotRewriter) -> None:
  """Verify `rewrite_code` emits `x.at[idx]` for scatter add rewrite.

  Args:
      rewriter (PivotRewriter): The rewriter instance.
  """
  code: str = "res = x.scatter_add_(0, idx, val)"
  res: str = rewrite_code(rewriter, code)
  assert "x.at[idx]" in res
  assert ".add(val)" in res


def test_scatter_keywords(rewriter: PivotRewriter) -> None:
  """Verify `rewrite_code` emits `x.at[indices]` for scatter keywords.

  Args:
      rewriter (PivotRewriter): The rewriter instance.
  """
  code: str = "x.scatter_(dim=0, src=updates, index=indices)"
  res: str = rewrite_code(rewriter, code)
  assert "x.at[indices]" in res
  assert ".set(updates)" in res


def test_ignore_tf_target(rewriter: PivotRewriter) -> None:
  """Verify `rewrite_code` handles ignore TensorFlow target as expected.

  Args:
      rewriter (PivotRewriter): The rewriter instance.
  """
  rewriter.context.config.target_framework = "tensorflow"
  rewriter.context.hook_context.target_framework = "tensorflow"
  code: str = "x.scatter_(1, i, v)"
  res: str = rewrite_code(rewriter, code)
  assert ".at[" not in res


def test_missing_attribute_func() -> None:
  """Verifies behavior when node.func is not an Attribute."""
  node: cst.Call = cst.Call(
    func=cst.Name("scatter"), args=[cst.Arg(cst.Integer("1")), cst.Arg(cst.Integer("2")), cst.Arg(cst.Integer("3"))]
  )
  ctx: MagicMock = MagicMock()
  ctx.target_framework = "jax"
  res: cst.CSTNode | cst.Call = transform_scatter(node, ctx)
  assert res is node


def test_missing_args() -> None:
  """Verifies behavior when there are fewer than 3 arguments."""
  node: cst.Call = cst.Call(
    func=cst.Attribute(value=cst.Name("x"), attr=cst.Name("scatter")),
    args=[cst.Arg(cst.Integer("1")), cst.Arg(cst.Name("idx"))],
  )
  ctx: MagicMock = MagicMock()
  ctx.target_framework = "jax"
  res: cst.CSTNode | cst.Call = transform_scatter(node, ctx)
  assert res is node


def test_scatter_too_few_args() -> None:
  """Verify `Call` handles scatter too few args as expected."""
  node: cst.Call = cst.Call(func=cst.Attribute(value=cst.Name("x"), attr=cst.Name("scatter")), args=[])
  ctx: HookContext = HookContext(semantics=MagicMock(), config=MagicMock())
  res: cst.CSTNode | cst.Call = transform_scatter(node, ctx)
  assert res is node


def test_scatter_functional_and_keyword_receiver() -> None:
  """Verify functional 4-arg scatter and keyword input= scatter forms."""
  ctx = HookContext(
    semantics=MagicMock(),
    config=RuntimeConfig(source_framework="torch", target_framework="jax"),
  )

  func_4arg = cst.ensure_type(cst.parse_expression("torch.scatter(x, 1, idx, src, *extra)"), cst.Call)
  res_4arg = transform_scatter(func_4arg, ctx)
  assert cst.Module(body=[]).code_for_node(res_4arg) == "x.at[idx].set(src)"

  kw_recv = cst.ensure_type(cst.parse_expression("scatter(input=x, index=idx, src=val)"), cst.Call)
  res_kw = transform_scatter(kw_recv, ctx)
  assert cst.Module(body=[]).code_for_node(res_kw) == "x.at[idx].set(val)"

  callable_expr = cst.ensure_type(cst.parse_expression("(get_fn())(x, 0, idx, val)"), cst.Call)
  res_call = transform_scatter(callable_expr, ctx)
  assert cst.Module(body=[]).code_for_node(res_call) == "x.at[idx].set(val)"
