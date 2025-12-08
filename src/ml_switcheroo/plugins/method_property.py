"""
Plugin for transforming Method calls to Property attributes.

This module handles cases where one framework uses a method calls (e.g. `x.size()`)
while the target framework uses properties (e.g. `x.shape`).

Common usage:
- PyTorch: `x.size()` -> JAX/NumPy: `x.shape`
- PyTorch: `x.size(dim)` -> JAX/NumPy: `x.shape[dim]`
"""

import libcst as cst
from typing import Union

from ml_switcheroo.core.hooks import register_hook, HookContext


@register_hook("method_to_property")
def transform_method_to_property(node: cst.Call, ctx: HookContext) -> Union[cst.Attribute, cst.Subscript, cst.Call]:
  """
  Plugin Hook: Transforms a method call into an attribute access or subscript.

  Triggers:
      Operations like `size` mapped with `requires_plugin: "method_to_property"`.

  Args:
      node: The original CST Call (e.g., `x.size()`).
      ctx: HookContext for API lookup.

  Returns:
      cst.Attribute (x.shape), cst.Subscript (x.shape[0]), or original node.
  """
  # 1. Validation: Must be a method call (Attribute)
  if not isinstance(node.func, cst.Attribute):
    return node

  method_name = node.func.attr.value
  receiver = node.func.value

  # 2. Identify the Abstract Operation
  # Since this hook is generic, we infer the op based on the source method name
  # or rely on defaults like 'size'.
  op_id = None
  if method_name == "size":
    op_id = "size"
  elif method_name == "data_ptr":
    op_id = "data_ptr"

  if not op_id:
    return node

  # 3. Lookup Target Property
  # Check semantics for what this maps to.
  # E.g. "size" -> variants[jax][api] = "shape"
  target_prop = ctx.lookup_api(op_id)

  # Fallbacks if semantics are incomplete for the target
  if not target_prop:
    if ctx.target_fw in ["jax", "numpy"]:
      if op_id == "size":
        target_prop = "shape"
      elif op_id == "data_ptr":
        target_prop = "data"

  if not target_prop:
    return node

  # 4. Construct Property Access Node
  # x.size() -> x.shape
  prop_node = cst.Attribute(value=receiver, attr=cst.Name(target_prop))

  # 5. Handle Arguments (e.g. dim index)
  # Case: x.size(index) -> x.shape[index]
  if node.args:
    if len(node.args) == 1:
      idx_val = node.args[0].value
      return cst.Subscript(
        value=prop_node,
        slice=[cst.SubscriptElement(slice=cst.Index(value=idx_val))],
      )
    else:
      # If multiple args (unlikely for simple prop mapping), preserve call or fail safe
      return node

  return prop_node
