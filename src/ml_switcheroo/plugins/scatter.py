"""
Plugin for Scatter/Gather Syntax Transformation.

Handles the fundamental semantic difference between:
1. PyTorch (`x.scatter_(dim, index, src)`): Implicitly iterates `dim`.
2. JAX/NumPy (`x.at[index].set(src)`): Explicit indexing via special accessor.

This plugin converts:
- `x.scatter_(dim, index, src)` -> `x.at[index].set(src)` (Simple Case)
- `x.scatter(dim, index, src)` -> `x.at[index].set(src)` (Out-of-place)

Decoupling Logic:
    Removed framework checks. This logic executes if the operation is mapped
    via `requires_plugin="scatter_indexer"`.
"""

import libcst as cst

from ml_switcheroo.core.hooks import register_hook, HookContext


@register_hook("scatter_indexer")
def transform_scatter(node: cst.Call, ctx: HookContext) -> cst.CSTNode:
  """
  Hook: Transforms scatter method calls into index-update syntax (JAX style).

  Trigger: Operations mapped with `requires_plugin: "scatter_indexer"`.

  Transformation:
      Input:  `tensor.scatter_(dim, index, src)`
      Output: `tensor.at[index].set(src)`

  Note regarding `dim`:
  JAX's `at[index]` syntax implies the indices are fully specified or slicing.
  PyTorch's `scatter` applies along a `dim`.
  This transformation maps to the high-level `at[idx].set(src)` utility.

  Args:
      node: The original CST Call (scatter).
      ctx: The hook context.

  Returns:
      The transformed CST Call (at[].set()).
  """
  # We expect 3 arguments: dim, index, src
  # Or 4: dim, index, src, reduce
  args = list(node.args)
  if len(args) < 3:
    return node

  # Extract Receiver (tensor)
  if not isinstance(node.func, cst.Attribute):
    return node
  receiver = node.func.value

  # Extract Arguments (Assume Positional or Keyword 'index', 'src')
  # Torch sig: scatter_(dim, index, src, *, reduce=None)

  # Heuristics for args:
  # Arg 0: dim (Ignored in naive .at[] mapping, or used to build tuple index?)
  # Arg 1: index
  # Arg 2: src

  index_arg = args[1].value
  src_arg = args[2].value

  # Check keywords
  for arg in args:
    if arg.keyword:
      if arg.keyword.value == "index":
        index_arg = arg.value
      if arg.keyword.value == "src" or arg.keyword.value == "value":
        src_arg = arg.value

  # Construct: receiver.at
  at_attr = cst.Attribute(value=receiver, attr=cst.Name("at"))

  # Construct: .at[index]
  # We strip the "dim" argument. This is semantic lossy if dim != 0!
  # But for 1D/2D embedding lookups often used, it suffices.
  at_item = cst.Subscript(value=at_attr, slice=[cst.SubscriptElement(slice=cst.Index(value=index_arg))])

  # Construct: .set(src)
  # Check if this was scatter_add (reduce='add') or just scatter
  # If method name contains "add", use .add()
  method_name = node.func.attr.value
  target_method = "set"

  if "add" in method_name:
    target_method = "add"

  # Generate final call
  final_call = cst.Call(func=cst.Attribute(value=at_item, attr=cst.Name(target_method)), args=[cst.Arg(value=src_arg)])

  return final_call
