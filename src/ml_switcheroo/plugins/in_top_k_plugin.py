"""Perform plugin for in_top_k.

Checks if target indices are present in top_k(predictions).
"""

import libcst as cst

from ml_switcheroo.core.hooks import HookContext, register_hook
from ml_switcheroo.frameworks.base import get_adapter
from ml_switcheroo.plugins.utils import create_dotted_name


@register_hook("in_top_k_plugin")
def in_top_k_plugin(node: cst.Call, ctx: HookContext) -> cst.CSTNode:
  """Plugin Transform Generic plugin for in_top_k.

  Checks if target indices are present in top_k(predictions).
  Typically translates in_top_k(predictions, targets, k) to
  `(targets.unsqueeze(-1) == top_k(predictions, k).indices).any(dim=-1)`.

  Args:
      node: The CST Call node representing the in_top_k operation to be
          processed.
      ctx: The hook context containing metadata, configuration, and translation
          environment details.

  Returns:
      The transformed CST node, or the original node if no transformation was
          applied.
  """
  if len(node.args) < 3:
    return node

  topk_api = ctx.lookup_api("TopK") if hasattr(ctx, "lookup_api") else None
  if not isinstance(topk_api, str) or not topk_api:
    config = getattr(ctx, "_runtime_config", getattr(ctx, "config", None))
    target_framework = getattr(config, "target_framework", getattr(config, "target", None)) if config else None
    if (
      not isinstance(target_framework, str)
      or get_adapter(target_framework) is None
      or ctx.plugin_traits.has_numpy_compatible_arrays
    ):
      return node
    topk_api = f"{target_framework}.topk"

  predictions = node.args[0].value
  targets = node.args[1].value
  k = node.args[2].value

  # Build: (targets.unsqueeze(-1) == <topk_api>(predictions, k).indices).any(dim=-1)
  unsqueeze_call = cst.Call(
    func=cst.Attribute(value=targets, attr=cst.Name("unsqueeze")),
    args=[cst.Arg(value=cst.UnaryOperation(operator=cst.Minus(), expression=cst.Integer("1")))],
  )
  topk_call = cst.Call(
    func=create_dotted_name(topk_api),
    args=[
      cst.Arg(value=predictions, comma=cst.Comma(whitespace_after=cst.SimpleWhitespace(" "))),
      cst.Arg(value=k),
    ],
  )
  topk_indices = cst.Attribute(value=topk_call, attr=cst.Name("indices"))
  comparison = cst.Comparison(
    left=unsqueeze_call,
    comparisons=[
      cst.ComparisonTarget(
        operator=cst.Equal(
          whitespace_before=cst.SimpleWhitespace(" "),
          whitespace_after=cst.SimpleWhitespace(" "),
        ),
        comparator=topk_indices,
      )
    ],
    lpar=[cst.LeftParen()],
    rpar=[cst.RightParen()],
  )
  return cst.Call(
    func=cst.Attribute(value=comparison, attr=cst.Name("any")),
    args=[
      cst.Arg(
        keyword=cst.Name("dim"),
        equal=cst.AssignEqual(
          whitespace_before=cst.SimpleWhitespace(""),
          whitespace_after=cst.SimpleWhitespace(""),
        ),
        value=cst.UnaryOperation(operator=cst.Minus(), expression=cst.Integer("1")),
      )
    ],
  )
