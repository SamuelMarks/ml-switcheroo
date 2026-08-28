"""Test suite for the Scatter Extra module."""

import libcst as cst
from typing import Union
from ml_switcheroo.plugins.scatter import transform_scatter
from ml_switcheroo.core.hooks import HookContext
from unittest.mock import MagicMock


def test_scatter_too_few_args() -> None:
  """Verifies the behavior of scatter too few arguments."""
  node: cst.Call = cst.Call(func=cst.Attribute(value=cst.Name("x"), attr=cst.Name("scatter")), args=[])
  ctx: HookContext = HookContext(semantics=MagicMock(), config=MagicMock())
  res: Union[cst.CSTNode, cst.Call] = transform_scatter(node, ctx)
  assert res is node
