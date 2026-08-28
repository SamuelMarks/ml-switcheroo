"""Tests for the in_top_k plugin."""

import libcst as cst
import typing

from ml_switcheroo.plugins.in_top_k_plugin import in_top_k_plugin
from ml_switcheroo.core.hooks import HookContext


def test_in_top_k_plugin_basic() -> None:
  """Test the basic behavior of the in_top_k_plugin.
  Currently it returns the node untouched.
  """
  code: str = "tf.math.in_top_k(targets, predictions, k=5)"
  module = cst.parse_module(code)
  call_node: cst.CSTNode = typing.cast(cst.Expr, typing.cast(cst.SimpleStatementLine, module.body[0]).body[0]).value

  from ml_switcheroo.config import RuntimeConfig

  config = RuntimeConfig(source_framework="tensorflow", target_framework="torch")
  ctx = HookContext(semantics=None, config=config)

  result: cst.CSTNode = in_top_k_plugin(typing.cast(cst.Call, call_node), ctx)
  assert result is call_node
