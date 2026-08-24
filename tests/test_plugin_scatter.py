"""Docstring."""

import libcst as cst
from unittest.mock import MagicMock
from ml_switcheroo.plugins.scatter import transform_scatter
from ml_switcheroo.core.hooks import HookContext


def test_transform_scatter_basic():
  """Docstring."""
  ctx = MagicMock(spec=HookContext)

  # 3 positional args
  node = cst.parse_expression("tensor.scatter_(dim, index, src)")
  new_node = transform_scatter(node, ctx)
  code = cst.Module(body=[cst.SimpleStatementLine(body=[cst.Expr(value=new_node)])]).code
  assert code.strip() == "tensor.at[index].set(src)"


def test_transform_scatter_keywords():
  """Docstring."""
  ctx = MagicMock(spec=HookContext)

  # with keywords
  node = cst.parse_expression("tensor.scatter(dim=0, index=idx, src=val)")
  new_node = transform_scatter(node, ctx)
  code = cst.Module(body=[cst.SimpleStatementLine(body=[cst.Expr(value=new_node)])]).code
  assert code.strip() == "tensor.at[idx].set(val)"

  # with value keyword (alias for src)
  node = cst.parse_expression("tensor.scatter(dim, index=idx, value=val2)")
  new_node = transform_scatter(node, ctx)
  code = cst.Module(body=[cst.SimpleStatementLine(body=[cst.Expr(value=new_node)])]).code
  assert code.strip() == "tensor.at[idx].set(val2)"


def test_transform_scatter_add():
  """Docstring."""
  ctx = MagicMock(spec=HookContext)

  # scatter_add uses .add()
  node = cst.parse_expression("tensor.scatter_add_(dim, index, src)")
  new_node = transform_scatter(node, ctx)
  code = cst.Module(body=[cst.SimpleStatementLine(body=[cst.Expr(value=new_node)])]).code
  assert code.strip() == "tensor.at[index].add(src)"


def test_transform_scatter_early_returns():
  """Docstring."""
  ctx = MagicMock(spec=HookContext)

  # args < 3
  node = cst.parse_expression("tensor.scatter(dim, index)")
  assert transform_scatter(node, ctx) is node

  # Not an attribute call
  node = cst.parse_expression("scatter(dim, index, src)")
  assert transform_scatter(node, ctx) is node
