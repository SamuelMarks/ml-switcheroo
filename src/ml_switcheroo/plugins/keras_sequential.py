"""Plugin for Sequential Container Argument Packing.

Bridges variadic sequential layer constructors (e.g., ``Sequential(l1, l2)``)
and sequence-taking sequential constructors (e.g., ``Sequential([l1, l2])``)
strictly using ``HookContext.current_variant`` without hardcoded framework APIs.
"""

import libcst as cst

from ml_switcheroo.core.hooks import HookContext, register_hook
from ml_switcheroo.plugins.utils import create_dotted_name, resolve_variant_api

_SEQUENCE_KEYWORDS = frozenset({"layers", "modules", "fns"})


def _create_dotted_name(name_str: str) -> cst.BaseExpression:
  """Create a CST node representing a dotted API path.

  Args:
      name_str: Dot-separated API path string.

  Returns:
      LibCST ``Name`` or nested ``Attribute`` node.
  """
  return create_dotted_name(name_str)


@register_hook("keras_sequential_pack")
def transform_keras_sequential(node: cst.Call, ctx: HookContext) -> cst.Call:
  """Transform Sequential container initialization via ``HookContext.current_variant``.

  Resolves the target Sequential API from ``ctx.current_variant`` or semantic lookup;
  if no mapping exists, returns ``node`` unchanged instead of falling back to a
  hardcoded framework path. Supports variadic positional layers, pre-packed list/tuple
  containers, and keyword layer lists (``layers=[...]``, ``modules=[...]``, ``fns=[...]``).

  Args:
      node: Original Sequential constructor call node.
      ctx: Plugin execution context.

  Returns:
      Transformed ``cst.Call`` node, or ``node`` unchanged if unmapped.
  """
  op_api = resolve_variant_api(ctx, "Sequential")
  if not op_api:
    return node

  new_func = _create_dotted_name(op_api)
  if not node.args:
    return node.with_changes(func=new_func)

  first_arg = node.args[0]
  if first_arg.keyword is None and isinstance(first_arg.value, (cst.List, cst.Tuple)):
    return node.with_changes(func=new_func)

  for arg in node.args:
    if arg.keyword is not None and arg.keyword.value in _SEQUENCE_KEYWORDS:
      clean_seq = arg.with_changes(
        keyword=None,
        equal=cst.MaybeSentinel.DEFAULT,
        comma=cst.MaybeSentinel.DEFAULT,
      )
      return node.with_changes(func=new_func, args=[clean_seq])

  elements = [cst.Element(value=arg.value) for arg in node.args if arg.keyword is None]
  if not elements:
    return node.with_changes(func=new_func, args=[])

  list_node = cst.List(elements=elements)
  return node.with_changes(func=new_func, args=[cst.Arg(value=list_node)])
