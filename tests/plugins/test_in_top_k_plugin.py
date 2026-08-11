"""Tests for the in_top_k plugin."""

import libcst as cst

from ml_switcheroo.plugins.in_top_k_plugin import in_top_k_plugin
from ml_switcheroo.core.hooks import HookContext


def test_in_top_k_plugin_basic():
  """Test the basic behavior of the in_top_k_plugin.
  Currently it returns the node untouched.
  """
  code = "tf.math.in_top_k(targets, predictions, k=5)"
  module = cst.parse_module(code)
  call_node = module.body[0].body[0].value  # the cst.Call node

  from ml_switcheroo.config import RuntimeConfig

  config = RuntimeConfig(source="tensorflow", target="torch")
  ctx = HookContext(semantics=None, config=config)

  result = in_top_k_plugin(call_node, ctx)
  assert result is call_node
