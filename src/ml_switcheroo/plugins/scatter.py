"""Plugin for Functional Scatter/Index-Update Syntax Transformation.

Converts in-place and functional scatter operations across method, functional,
and keyword-argument forms into functional ``.at[index].set(src)`` / ``.add(src)``
updates without source-only positional arity assumptions.
"""

import typing

import libcst as cst

from ml_switcheroo.core.hooks import HookContext, register_hook
from ml_switcheroo.plugins.utils import is_framework_module_node

_RECEIVER_KEYWORDS = frozenset({"input", "tensor", "x", "self"})
_INDEX_KEYWORDS = frozenset({"index", "indices"})
_SRC_KEYWORDS = frozenset({"src", "value", "updates", "values"})


def _extract_method_and_receiver(
  node: cst.Call,
  ctx: HookContext,
) -> tuple[str, cst.BaseExpression | None, list[cst.Arg]]:
  """Resolve scatter method name, receiver tensor, and remaining call arguments.

  Args:
      node: Original scatter ``cst.Call`` node.
      ctx: Active ``HookContext``.

  Returns:
      Tuple of ``(method_name, receiver_expr_or_none, remaining_args)``.
  """
  args = list(node.args)
  if isinstance(node.func, cst.Attribute):
    method_name = node.func.attr.value
    if not is_framework_module_node(node.func.value, ctx):
      return method_name, node.func.value, args
  else:
    method_name = node.func.value if isinstance(node.func, cst.Name) else "scatter"

  for idx, arg in enumerate(args):
    if arg.keyword is not None and arg.keyword.value in _RECEIVER_KEYWORDS:
      remaining = args[:idx] + args[idx + 1 :]
      return method_name, arg.value, remaining

  positional = [a for a in args if a.keyword is None and not a.star]
  if len(positional) >= 4:
    first_pos_idx = next(i for i, a in enumerate(args) if a.keyword is None and not a.star)
    return method_name, args[first_pos_idx].value, args[:first_pos_idx] + args[first_pos_idx + 1 :]

  return method_name, None, args


def _extract_index_and_src(
  args: typing.Sequence[cst.Arg],
) -> tuple[cst.BaseExpression | None, cst.BaseExpression | None]:
  """Extract ``index`` and ``src`` expressions from positional or keyword arguments.

  Args:
      args: Arguments excluding the receiver tensor.

  Returns:
      Tuple ``(index_expr, src_expr)`` or ``(None, None)`` if insufficient arguments.
  """
  index_expr: cst.BaseExpression | None = None
  src_expr: cst.BaseExpression | None = None
  positional: list[cst.BaseExpression] = []

  for arg in args:
    if arg.star:
      continue
    if arg.keyword is not None:
      kw = arg.keyword.value
      if kw in _INDEX_KEYWORDS:
        index_expr = arg.value
      elif kw in _SRC_KEYWORDS:
        src_expr = arg.value
    else:
      positional.append(arg.value)

  has_dim_kw = any(a.keyword is not None and a.keyword.value in {"dim", "axis"} for a in args)
  if not has_dim_kw and positional:
    positional = positional[1:]

  if index_expr is None and positional:
    index_expr = positional.pop(0)
  if src_expr is None and positional:
    src_expr = positional.pop(0)

  return index_expr, src_expr


@register_hook("scatter_indexer")
def transform_scatter(node: cst.Call, ctx: HookContext) -> cst.CSTNode:
  """Transform scatter method or function calls into ``.at[index].set(src)`` syntax.

  Args:
      node: Original scatter ``cst.Call`` node.
      ctx: Active ``HookContext``.

  Returns:
      Transformed index-update call, or ``node`` unchanged if arguments are incomplete.
  """
  method_name, receiver, remaining_args = _extract_method_and_receiver(node, ctx)
  if receiver is None:
    return node

  index_arg, src_arg = _extract_index_and_src(remaining_args)
  if index_arg is None or src_arg is None:
    return node

  at_attr = cst.Attribute(value=receiver, attr=cst.Name("at"))
  at_item = cst.Subscript(value=at_attr, slice=[cst.SubscriptElement(slice=cst.Index(value=index_arg))])
  target_method = "add" if "add" in method_name else "set"
  return cst.Call(
    func=cst.Attribute(value=at_item, attr=cst.Name(target_method)),
    args=[cst.Arg(value=src_arg)],
  )
