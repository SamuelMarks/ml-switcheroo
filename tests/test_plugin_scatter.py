"""Test suite for plugin scatter."""

from unittest.mock import MagicMock

import libcst as cst

from ml_switcheroo.core.hooks import HookContext
from ml_switcheroo.plugins.scatter import transform_scatter


def test_transform_scatter_basic() -> None:
  """Verify transform scatter basic behavior."""
  ctx: MagicMock = MagicMock(spec=HookContext)

  # 3 positional args
  node: cst.Call = cst.ensure_type(cst.parse_expression("tensor.scatter_(dim, index, src)"), cst.Call)
  new_node: cst.BaseExpression = transform_scatter(node, ctx)  # type: ignore[assignment]
  code: str = cst.Module(body=[cst.SimpleStatementLine(body=[cst.Expr(value=new_node)])]).code
  assert code.strip() == "tensor.at[index].set(src)"


def test_transform_scatter_keywords() -> None:
  """Verify transform scatter keywords behavior."""
  ctx: MagicMock = MagicMock(spec=HookContext)

  # with keywords
  node: cst.Call = cst.ensure_type(cst.parse_expression("tensor.scatter(dim=0, index=idx, src=val)"), cst.Call)
  new_node: cst.BaseExpression = transform_scatter(node, ctx)  # type: ignore[assignment]
  code: str = cst.Module(body=[cst.SimpleStatementLine(body=[cst.Expr(value=new_node)])]).code
  assert code.strip() == "tensor.at[idx].set(val)"

  # with value keyword (alias for src)
  node2: cst.Call = cst.ensure_type(cst.parse_expression("tensor.scatter(dim, index=idx, value=val2)"), cst.Call)
  new_node2: cst.BaseExpression = transform_scatter(node2, ctx)  # type: ignore[assignment]
  code2: str = cst.Module(body=[cst.SimpleStatementLine(body=[cst.Expr(value=new_node2)])]).code
  assert code2.strip() == "tensor.at[idx].set(val2)"


def test_transform_scatter_add() -> None:
  """Verify transform scatter add behavior."""
  ctx: MagicMock = MagicMock(spec=HookContext)

  # scatter_add uses .add()
  node: cst.Call = cst.ensure_type(cst.parse_expression("tensor.scatter_add_(dim, index, src)"), cst.Call)
  new_node: cst.BaseExpression = transform_scatter(node, ctx)  # type: ignore[assignment]
  code: str = cst.Module(body=[cst.SimpleStatementLine(body=[cst.Expr(value=new_node)])]).code
  assert code.strip() == "tensor.at[index].add(src)"


def test_transform_scatter_early_returns() -> None:
  """Verify transform scatter early returns behavior."""
  ctx: MagicMock = MagicMock(spec=HookContext)

  # args < 3
  node: cst.Call = cst.ensure_type(cst.parse_expression("tensor.scatter(dim, index)"), cst.Call)
  assert transform_scatter(node, ctx) is node

  # Not an attribute call
  node2: cst.Call = cst.ensure_type(cst.parse_expression("scatter(dim, index, src)"), cst.Call)
  assert transform_scatter(node2, ctx) is node2
