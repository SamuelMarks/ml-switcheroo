"""Docstring."""

import libcst as cst
from unittest.mock import MagicMock
from ml_switcheroo.plugins.scatter import transform_scatter
from ml_switcheroo.core.hooks import HookContext


def test_transform_scatter_basic() -> None:
  """Docstring."""
  ctx: MagicMock = MagicMock(spec=HookContext)

  # 3 positional args
  node: cst.BaseExpression = cst.parse_expression("tensor.scatter_(dim, index, src)")
  new_node: cst.BaseExpression = transform_scatter(node, ctx)
  code: str = cst.Module(body=[cst.SimpleStatementLine(body=[cst.Expr(value=new_node)])]).code
  assert code.strip() == "tensor.at[index].set(src)"


def test_transform_scatter_keywords() -> None:
  """Docstring."""
  ctx: MagicMock = MagicMock(spec=HookContext)

  # with keywords
  node: cst.BaseExpression = cst.parse_expression("tensor.scatter(dim=0, index=idx, src=val)")
  new_node: cst.BaseExpression = transform_scatter(node, ctx)
  code: str = cst.Module(body=[cst.SimpleStatementLine(body=[cst.Expr(value=new_node)])]).code
  assert code.strip() == "tensor.at[idx].set(val)"

  # with value keyword (alias for src)
  node2: cst.BaseExpression = cst.parse_expression("tensor.scatter(dim, index=idx, value=val2)")
  new_node2: cst.BaseExpression = transform_scatter(node2, ctx)
  code2: str = cst.Module(body=[cst.SimpleStatementLine(body=[cst.Expr(value=new_node2)])]).code
  assert code2.strip() == "tensor.at[idx].set(val2)"


def test_transform_scatter_add() -> None:
  """Docstring."""
  ctx: MagicMock = MagicMock(spec=HookContext)

  # scatter_add uses .add()
  node: cst.BaseExpression = cst.parse_expression("tensor.scatter_add_(dim, index, src)")
  new_node: cst.BaseExpression = transform_scatter(node, ctx)
  code: str = cst.Module(body=[cst.SimpleStatementLine(body=[cst.Expr(value=new_node)])]).code
  assert code.strip() == "tensor.at[index].add(src)"


def test_transform_scatter_early_returns() -> None:
  """Docstring."""
  ctx: MagicMock = MagicMock(spec=HookContext)

  # args < 3
  node: cst.BaseExpression = cst.parse_expression("tensor.scatter(dim, index)")
  assert transform_scatter(node, ctx) is node

  # Not an attribute call
  node2: cst.BaseExpression = cst.parse_expression("scatter(dim, index, src)")
  assert transform_scatter(node2, ctx) is node2
