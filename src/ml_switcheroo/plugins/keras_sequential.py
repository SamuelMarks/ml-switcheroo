"""
Plugin for Keras Sequential Container.

Handles:
1. Packing `*args` into `[layers]`.
2. Renaming API to `keras.Sequential`.
"""

import libcst as cst
from ml_switcheroo.core.hooks import register_hook, HookContext


def _create_dotted_name(name_str: str) -> cst.BaseExpression:
  parts = name_str.split(".")
  node = cst.Name(parts[0])
  for part in parts[1:]:
    node = cst.Attribute(value=node, attr=cst.Name(part))
  return node


@register_hook("keras_sequential_pack")
def transform_keras_sequential(node: cst.Call, ctx: HookContext) -> cst.Call:
  """
  Plugin Hook: Packs positional layer arguments into a list for Keras Sequential.
  Also renames the function to `keras.Sequential`.

  Decoupling: Executes blindly if wired. Target API renaming relies on context value or defaults.
  """
  # 1. Rename Function (API Swap)
  # We resolve the name "Sequential" to the target API using the context if possible.
  op_api = ctx.lookup_api("Sequential")

  # Safety Check: If lookup returns non-string (e.g. Mock in bad test setup) or None
  if not isinstance(op_api, str):
    op_api = "keras.Sequential"

  new_func = _create_dotted_name(op_api)

  # If args are already empty or it looks like a list is passed, just rename and return
  if not node.args:
    return node.with_changes(func=new_func)

  first_val = node.args[0].value
  if isinstance(first_val, (cst.List, cst.Tuple)):
    return node.with_changes(func=new_func)

  # 2. Pack Arguments
  elements = []
  for arg in node.args:
    # Ignore keyword args if any
    if arg.keyword:
      continue
    val = arg.value
    elements.append(cst.Element(value=val))

  list_node = cst.List(elements=elements)
  new_args = [cst.Arg(value=list_node)]

  return node.with_changes(func=new_func, args=new_args)
