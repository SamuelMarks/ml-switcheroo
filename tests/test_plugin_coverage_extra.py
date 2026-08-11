"""Module docstring."""

import libcst as cst
from unittest import mock
from ml_switcheroo.plugins.in_top_k_plugin import in_top_k_plugin


def test_in_top_k_plugin():
  """Test in top k plugin."""
  node = cst.Call(func=cst.Name("in_top_k"))
  ctx = mock.MagicMock()
  result = in_top_k_plugin(node, ctx)
  assert result is node
