"""
Plugin for Packing Shape Arguments.

This transformation converts variable-argument shape definitions into
explicit tuple arguments required by certain frameworks.

Example:
    Source: `x.view(1, 2, -1)` (PyTorch style)
    Target: `jnp.reshape(x, (1, 2, -1))` (JAX/NumPy style)

Decoupling Logic:
    This plugin does NOT enforce a framework whitelist. It executes unconditionally
    if wired. However, it relies on looking up "Reshape" or "View" in the semantics.
    If those definitions are missing for the target framework, it aborts.
"""

import libcst as cst
from typing import List, Union, Optional

from ml_switcheroo.core.hooks import register_hook, HookContext
from ml_switcheroo.plugins.utils import create_dotted_name, is_framework_module_node


def _create_dotted_name(name_str: str) -> cst.BaseExpression:
  """Helper to create a CST Attribute chain from string."""
  parts = name_str.split(".")
  node = cst.Name(parts[0])
  for part in parts[1:]:
    node = cst.Attribute(value=node, attr=cst.Name(part))
  return node


@register_hook("pack_shape_args")
def transform_shape_packing(node: cst.Call, ctx: HookContext) -> cst.Call:
  """
  Hook: Packs trailing positional arguments into a shape tuple.

  Logic:
  1.  Resolve Target API via "Reshape" or "View". Abort if missing.
  2.  Packs arguments.

  Args:
      node: The original CST Call node.
      ctx: The hook execution context.

  Returns:
      cst.Call: The transformed call with packed shape arguments.
  """
  # 0. Resolve Target API (Strict)
  # Attempt to use specific OP ID set by Rewriter, or fallback
  op_name = ctx.current_op_id or "Reshape"

  target_api = ctx.lookup_api(op_name)
  if not target_api and op_name == "Reshape":
    # Try alternate key
    target_api = ctx.lookup_api("View")

  if not target_api:
    return node

  # 1. Determine Input Tensor & Shape Args
  input_tensor: Optional[cst.BaseExpression] = None
  shape_args: List[cst.Arg] = []

  is_method = False
  if isinstance(node.func, cst.Attribute):
    if is_framework_module_node(node.func.value, ctx):
      is_method = False
    else:
      is_method = True

  if is_method:
    input_tensor = node.func.value
    shape_args = list(node.args)
  else:
    if not node.args:
      return node
    input_tensor = node.args[0].value
    shape_args = list(node.args[1:])

  # 2. Pack Shape Arguments
  packed_shape_val: Union[cst.BaseExpression, None] = None

  if len(shape_args) > 1:
    elements = [cst.Element(value=arg.value) for arg in shape_args]
    packed_shape_val = cst.Tuple(elements=elements)

  elif len(shape_args) == 1:
    val = shape_args[0].value
    if isinstance(val, cst.Integer):
      packed_shape_val = cst.Tuple(
        elements=[cst.Element(value=val, comma=cst.Comma(whitespace_after=cst.SimpleWhitespace(" ")))]
      )
    else:
      packed_shape_val = val
  else:
    return node

  # 3. Construct New Call Arguments
  new_args = [cst.Arg(value=input_tensor, comma=cst.Comma(whitespace_after=cst.SimpleWhitespace(" ")))]

  if packed_shape_val:
    new_args.append(cst.Arg(value=packed_shape_val))

  # 4. Create Call
  new_func = create_dotted_name(target_api)

  return node.with_changes(func=new_func, args=new_args)
