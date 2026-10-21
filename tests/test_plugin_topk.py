"""Test suite for plugin topk."""

from unittest.mock import MagicMock

import libcst as cst

from ml_switcheroo.core.hooks import HookContext
from ml_switcheroo.plugins.topk import _create_dotted_name, transform_topk


def parse_expr(code: str) -> cst.BaseExpression:
  """Execute parse expr helper logic.

  Args:
      code: Test argument or fixture.

  Returns:
      Computed test result or fixture value.
  """
  return cst.parse_expression(code)


def test_create_dotted_name() -> None:
  """Verify create dotted name behavior."""
  node: cst.BaseExpression = _create_dotted_name("a.b.c")
  code: str = cst.Module(body=[cst.SimpleStatementLine(body=[cst.Expr(value=node)])]).code
  assert code.strip() == "a.b.c"


def test_transform_topk_no_target() -> None:
  """Verify transform topk no target behavior."""
  call: cst.Call = cst.ensure_type(parse_expr("torch.topk(x, 5)"), cst.Call)
  ctx: MagicMock = MagicMock(spec=HookContext)
  ctx.lookup_api.return_value = None
  res: cst.BaseExpression = getattr(
    {
      "v": getattr({"v": transform_topk}, "__getitem__")("v")(
        getattr({"v": call}, "__getitem__")("v"), getattr({"v": ctx}, "__getitem__")("v")
      )
    },
    "__getitem__",
  )("v")
  assert res is call


def test_transform_topk_with_target() -> None:
  """Verify transform topk with target behavior."""
  call: cst.Call = cst.ensure_type(
    parse_expr("torch.topk(x, k=5, largest=True, sorted=False, out=None, other=1)"), cst.Call
  )
  ctx: MagicMock = MagicMock(spec=HookContext)
  ctx.lookup_api.return_value = "jax.lax.top_k"

  res: cst.BaseExpression = getattr(
    {
      "v": getattr({"v": transform_topk}, "__getitem__")("v")(
        getattr({"v": call}, "__getitem__")("v"), getattr({"v": ctx}, "__getitem__")("v")
      )
    },
    "__getitem__",
  )("v")

  ctx.lookup_api.assert_called_once_with("TopK")
  ctx.inject_preamble.assert_called_once_with("import collections")

  code: str = cst.Module(body=[cst.SimpleStatementLine(body=[cst.Expr(value=res)])]).code
  assert 'collections.namedtuple("TopK"' in code
  assert '"values"' in code
  assert '"indices"' in code
  assert "*jax.lax.top_k(x, k=5, other=1)" in code
