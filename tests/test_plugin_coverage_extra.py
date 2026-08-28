"""Module docstring."""

import libcst as cst
from unittest import mock
from ml_switcheroo.plugins.in_top_k_plugin import in_top_k_plugin


def test_in_top_k_plugin() -> None:
  """Test in top k plugin."""
  node: cst.Call = cst.Call(func=cst.Name("in_top_k"), args=[])
  ctx: mock.MagicMock = mock.MagicMock()
  result: cst.CSTNode = in_top_k_plugin(node, ctx)
  assert result is node
