"""Test suite for the Static Unroll Extra module."""

import libcst as cst
from typing import Union
from ml_switcheroo.plugins.static_unroll import unroll_static_loops
from ml_switcheroo.core.hooks import HookContext
from unittest.mock import MagicMock


def test_static_unroll_iter_not_call() -> None:
  """Verifies the behavior of static unroll iter not call."""
  node: cst.For = cst.For(
    target=cst.Name("i"), iter=cst.List([]), body=cst.IndentedBlock(body=[cst.SimpleStatementLine(body=[cst.Pass()])])
  )
  ctx: HookContext = HookContext(semantics=MagicMock(), config=MagicMock())
  res: Union[cst.CSTNode, cst.For] = unroll_static_loops(node, ctx)
  assert res is node


def test_static_unroll_iter_call_not_range() -> None:
  """Verifies the behavior of static unroll iter call not range."""
  node: cst.For = cst.For(
    target=cst.Name("i"),
    iter=cst.Call(func=cst.Name("enumerate"), args=[]),
    body=cst.IndentedBlock(body=[cst.SimpleStatementLine(body=[cst.Pass()])]),
  )
  ctx: HookContext = HookContext(semantics=MagicMock(), config=MagicMock())
  res: Union[cst.CSTNode, cst.For] = unroll_static_loops(node, ctx)
  assert res is node
