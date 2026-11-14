"""Plugin for Functional Gradient Clipping.

Transforms imperative gradient clipping calls into functional gradient
transformation updates driven strictly by ``PluginTraits.requires_functional_state``
and ``HookContext.current_variant`` without hardcoded target framework paths.
"""

import typing

import libcst as cst

from ml_switcheroo.core.hooks import HookContext, register_hook
from ml_switcheroo.plugins.utils import create_dotted_name, resolve_variant_api

_GRADS_KEYWORDS = frozenset({"parameters", "grads", "gradients", "tensors"})
_NORM_KEYWORDS = frozenset({"max_norm", "clip_norm"})


def _extract_clip_operands(
  args: typing.Sequence[cst.Arg],
) -> tuple[cst.BaseExpression | None, cst.BaseExpression | None]:
  """Extract ``(grads, max_norm)`` from positional or keyword call arguments.

  Args:
      args: Call arguments passed to gradient clipping function.

  Returns:
      Tuple ``(grads_expr, max_norm_expr)`` or ``(None, None)`` if incomplete.
  """
  grads_node: cst.BaseExpression | None = None
  max_norm_node: cst.BaseExpression | None = None
  positional: list[cst.BaseExpression] = []

  for arg in args:
    if arg.star:
      continue
    if arg.keyword is not None:
      kw = arg.keyword.value
      if kw in _GRADS_KEYWORDS:
        grads_node = arg.value
      elif kw in _NORM_KEYWORDS:
        max_norm_node = arg.value
    else:
      positional.append(arg.value)

  if grads_node is None and positional:
    grads_node = positional.pop(0)
  if max_norm_node is None and positional:
    max_norm_node = positional.pop(0)

  return grads_node, max_norm_node


@register_hook("grad_clipper")
def transform_grad_clipping(node: cst.Call, ctx: HookContext) -> cst.CSTNode:
  """Transform imperative gradient clipping into functional state-update clipping.

  Checks ``ctx.plugin_traits.requires_functional_state`` and resolves the clipping
  factory API strictly from ``ctx.current_variant`` or semantic lookup (aborting
  safely if unmapped).

  Args:
      node: Source gradient clipping call node.
      ctx: Execution context providing ``plugin_traits`` and ``current_variant``.

  Returns:
      Transformed functional clipping subscript expression, or ``node`` unchanged.
  """
  if not ctx.plugin_traits.requires_functional_state:
    return node

  clip_api = resolve_variant_api(ctx, ("ClipGradNorm", "ClipGrads", "clip_by_global_norm"))
  if not clip_api:
    return node

  grads_node, max_norm_node = _extract_clip_operands(node.args)
  if grads_node is None or max_norm_node is None:
    return node

  clip_fn = cst.Call(
    func=create_dotted_name(clip_api),
    args=[cst.Arg(value=max_norm_node)],
  )
  update_call = cst.Call(
    func=cst.Attribute(value=clip_fn, attr=cst.Name("update")),
    args=[
      cst.Arg(value=grads_node, comma=cst.Comma(whitespace_after=cst.SimpleWhitespace(" "))),
      cst.Arg(value=cst.Name("None")),
    ],
  )
  return cst.Subscript(
    value=update_call,
    slice=[cst.SubscriptElement(slice=cst.Index(value=cst.Integer("0")))],
  )
