"""
Plugin for Keras Sequential Container Translation.

This plugin bridges the difference between PyTorch's `nn.Sequential(*layers)` (variadic args)
and Keras's `keras.Sequential([layers])` (list input).

It performs two key transformations:
1.  **API Renaming**: Swaps the function name to `keras.Sequential` (or configured API).
2.  **Argument Packing**: Collects all positional arguments (individual layers) into a
    single list argument to match the Keras constructor signature.
"""

import libcst as cst
from ml_switcheroo.core.hooks import register_hook, HookContext


def _create_dotted_name(name_str: str) -> cst.BaseExpression:
  """
  Helper function to create a CST node representing a dotted name (e.g., 'keras.Sequential').

  Args:
      name_str: The dot-separated string representation.

  Returns:
      A LibCST node (Name or nested Attribute).
  """
  parts = name_str.split(".")
  node = cst.Name(parts[0])
  for part in parts[1:]:
    node = cst.Attribute(value=node, attr=cst.Name(part))
  return node


@register_hook("keras_sequential_pack")
def transform_keras_sequential(node: cst.Call, ctx: HookContext) -> cst.Call:
  """
  Plugin Hook: Transforms Sequential container initialization.

  Transformation:
      Input: `Sequential(layer1, layer2, ...)`
      Output: `keras.Sequential([layer1, layer2, ...])`

  This hook is triggered for operations mapped with `requires_plugin="keras_sequential_pack"`.
  It uses `ctx.lookup_api("Sequential")` to determine the target class name, defaulting
  to `keras.Sequential` if lookup fails.

  Args:
      node: The original function call node.
      ctx: The plugin execution context.

  Returns:
      The transformed function call.
  """
  # 1. Rename Function (API Swap)
  # We resolve the name "Sequential" to the target API using the context if possible.
  op_api = ctx.lookup_api("Sequential")

  # Safety Check: If lookup returns non-string (e.g. Mock in bad test setup) or None
  if not isinstance(op_api, str):
    op_api = "keras.Sequential"

  new_func = _create_dotted_name(op_api)

  # If args are already empty or it looks like a list is passed (e.g. Sequential([...])),
  # just rename and return to avoid double-packing.
  if not node.args:
    return node.with_changes(func=new_func)

  first_val = node.args[0].value
  if isinstance(first_val, (cst.List, cst.Tuple)):
    return node.with_changes(func=new_func)

  # 2. Pack Arguments
  elements = []
  for arg in node.args:
    # Ignore keyword args if any (Keras Sequential doesn't typically use kwargs for layers)
    if arg.keyword:
      continue
    val = arg.value
    elements.append(cst.Element(value=val))

  list_node = cst.List(elements=elements)
  new_args = [cst.Arg(value=list_node)]

  return node.with_changes(func=new_func, args=new_args)
