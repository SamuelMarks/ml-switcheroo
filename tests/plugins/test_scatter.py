"""Test suite for the Scatter module."""

from typing import Any, Dict, Generator, Optional, Tuple, Union
from unittest.mock import MagicMock

import libcst as cst
import pytest

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
def rewriter() -> Generator[PivotRewriter, None, None]:
  """Provides a mock rewriter for testing.

  Yields:
      PivotRewriter: A mock rewriter instance.
  """
  hooks._HOOKS["scatter_indexer"] = transform_scatter
  hooks._PLUGINS_LOADED = True
  mgr: MagicMock = MagicMock()
  scatter_def: Dict[str, Any] = {
    "variants": {
      "torch": {"api": "torch.Tensor.scatter_"},
      "jax": {"api": "at_set", "requires_plugin": "scatter_indexer"},
    }
  }

  def get_def(name: str) -> Optional[Tuple[str, Dict[str, Any]]]:
    """Gets def.

    Args:
        name (str): Definition name.

    Returns:
        Optional[Tuple[str, Dict[str, Any]]]: Definition tuple.
    """
    if "scatter" in name:
      return ("Scatter", scatter_def)
    return None

  mgr.get_definition.side_effect = get_def
  mgr.resolve_variant.side_effect = lambda aid, fw: scatter_def["variants"]["jax"] if fw == "jax" else None
  mgr.is_verified.return_value = True
  mgr.get_known_apis.return_value = {"Scatter": scatter_def}
  mgr.get_framework_config.return_value = {}
  cfg: RuntimeConfig = RuntimeConfig(source_framework="torch", target_framework="jax")
  yield PivotRewriter(mgr, cfg)


def test_scatter_simple_rewrite(rewriter: PivotRewriter) -> None:
  """Verifies the behavior of scatter simple rewrite.

  Args:
      rewriter (PivotRewriter): The rewriter instance.
  """
  code: str = "res = x.scatter_(1, idx, src)"
  res: str = rewrite_code(rewriter, code)
  assert "x.at[idx]" in res
  assert ".set(src)" in res
  assert ", 1," not in res and "(1," not in res


def test_scatter_add_rewrite(rewriter: PivotRewriter) -> None:
  """Verifies the behavior of scatter add rewrite.

  Args:
      rewriter (PivotRewriter): The rewriter instance.
  """
  code: str = "res = x.scatter_add_(0, idx, val)"
  res: str = rewrite_code(rewriter, code)
  assert "x.at[idx]" in res
  assert ".add(val)" in res


def test_scatter_keywords(rewriter: PivotRewriter) -> None:
  """Verifies the behavior of scatter keywords.

  Args:
      rewriter (PivotRewriter): The rewriter instance.
  """
  code: str = "x.scatter_(dim=0, src=updates, index=indices)"
  res: str = rewrite_code(rewriter, code)
  assert "x.at[indices]" in res
  assert ".set(updates)" in res


def test_ignore_tf_target(rewriter: PivotRewriter) -> None:
  """Verifies the behavior of ignore tf target.

  Args:
      rewriter (PivotRewriter): The rewriter instance.
  """
  rewriter.context.config.target_framework = "tensorflow"
  rewriter.context.hook_context.target_fw = "tensorflow"
  code: str = "x.scatter_(1, i, v)"
  res: str = rewrite_code(rewriter, code)
  assert ".at[" not in res


def test_missing_attribute_func() -> None:
  """Verifies behavior when node.func is not an Attribute."""
  node: cst.Call = cst.Call(
    func=cst.Name("scatter"), args=[cst.Arg(cst.Integer("1")), cst.Arg(cst.Integer("2")), cst.Arg(cst.Integer("3"))]
  )
  ctx: MagicMock = MagicMock()
  ctx.target_fw = "jax"
  res: Union[cst.CSTNode, cst.Call] = transform_scatter(node, ctx)
  assert res is node


def test_missing_args() -> None:
  """Verifies behavior when there are fewer than 3 arguments."""
  node: cst.Call = cst.Call(
    func=cst.Attribute(value=cst.Name("x"), attr=cst.Name("scatter")),
    args=[cst.Arg(cst.Integer("1")), cst.Arg(cst.Name("idx"))],
  )
  ctx: MagicMock = MagicMock()
  ctx.target_fw = "jax"
  res: Union[cst.CSTNode, cst.Call] = transform_scatter(node, ctx)
  assert res is node


# --- Merged from test_scatter_extra.py ---


def test_scatter_too_few_args() -> None:
  """Verifies the behavior of scatter too few arguments."""
  node: cst.Call = cst.Call(func=cst.Attribute(value=cst.Name("x"), attr=cst.Name("scatter")), args=[])
  ctx: HookContext = HookContext(semantics=MagicMock(), config=MagicMock())
  res: Union[cst.CSTNode, cst.Call] = transform_scatter(node, ctx)
  assert res is node
