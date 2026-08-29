"""Docstring."""

from unittest.mock import MagicMock

import libcst as cst

from ml_switcheroo.core.hooks import HookContext
from ml_switcheroo.plugins.topk import _create_dotted_name, transform_topk


def parse_expr(code: str) -> cst.BaseExpression:
  """Docstring."""
  return cst.parse_expression(code)


def test_create_dotted_name() -> None:
  """Docstring."""
  node: cst.BaseExpression = _create_dotted_name("a.b.c")
  code: str = cst.Module(body=[cst.SimpleStatementLine(body=[cst.Expr(value=node)])]).code
  assert code.strip() == "a.b.c"


def test_transform_topk_no_target() -> None:
  """Docstring."""
  call: cst.BaseExpression = parse_expr("torch.topk(x, 5)")
  ctx: MagicMock = MagicMock(spec=HookContext)
  ctx.lookup_api.return_value = None
  res: cst.BaseExpression = transform_topk(call, ctx)
  assert res is call


def test_transform_topk_with_target() -> None:
  """Docstring."""
  call: cst.BaseExpression = parse_expr("torch.topk(x, k=5, largest=True, sorted=False, out=None, other=1)")
  ctx: MagicMock = MagicMock(spec=HookContext)
  ctx.lookup_api.return_value = "jax.lax.top_k"

  res: cst.BaseExpression = transform_topk(call, ctx)

  ctx.lookup_api.assert_called_once_with("TopK")
  ctx.inject_preamble.assert_called_once_with("import collections")

  code: str = cst.Module(body=[cst.SimpleStatementLine(body=[cst.Expr(value=res)])]).code
  assert 'collections.namedtuple("TopK"' in code
  assert '"values"' in code
  assert '"indices"' in code
  assert "*jax.lax.top_k(x, k=5, other=1)" in code
