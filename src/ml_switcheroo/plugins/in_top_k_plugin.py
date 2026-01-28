"""
Generic plugin for `in_top_k`. Checks if target indices are present in `top_k(predictions)`.
"""

import libcst as cst
from ml_switcheroo.core.hooks import register_hook, HookContext


@register_hook("in_top_k_plugin")
def in_top_k_plugin(node: cst.Call, ctx: HookContext) -> cst.CSTNode:
  """
  Plugin Hook: Generic plugin for `in_top_k`. Checks if target indices are present in `top_k(predictions)`.
  """
  # TODO: Implement custom logic
  return node
