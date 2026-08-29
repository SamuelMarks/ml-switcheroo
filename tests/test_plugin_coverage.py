"""Module docstring."""

from unittest import mock

import libcst as cst

from ml_switcheroo.plugins.in_top_k_plugin import in_top_k_plugin


def test_in_top_k_plugin() -> None:
  """Docstring."""
  node: cst.Call = cst.Call(func=cst.Name("in_top_k"), args=[])
  ctx: mock.MagicMock = mock.MagicMock()
  result: cst.CSTNode = in_top_k_plugin(node, ctx)
  assert result is node
