"""
Plugin for Packing Shape Arguments.
"""

import libcst as cst
from typing import List, Union

from ml_switcheroo.core.hooks import register_hook, HookContext
from ml_switcheroo.plugins.utils import create_dotted_name, is_framework_module_node


@register_hook("pack_shape_args")
def transform_shape_packing(node: cst.Call, ctx: HookContext) -> cst.Call:
  if ctx.target_fw not in ["jax", "numpy", "tensorflow", "mlx"]:
    return node

  # 1. Determine Input Tensor & Shape Args
  input_tensor: cst.BaseExpression = None
  shape_args: List[cst.Arg] = []

  is_method = False
  if isinstance(node.func, cst.Attribute):
    # Distinguish x.view() from torch.reshape() via centralized check
    if is_framework_module_node(node.func.value, ctx):
      # It's a function call (torch.reshape)
      is_method = False
    else:
      is_method = True

  if is_method:
    # Case: x.view(...) -> Input is x
    input_tensor = node.func.value
    shape_args = list(node.args)
  else:
    # Case: view(x, ...) -> Input is arg 0
    if not node.args:
      return node
    input_tensor = node.args[0].value
    shape_args = list(node.args[1:])

  # 2. Pack Shape Arguments
  packed_shape_val: Union[cst.BaseExpression, None] = None

  if len(shape_args) > 1:
    # Multiple args: Pack into Tuple -> (arg1, arg2, ...)
    elements = []
    for arg in shape_args:
      elements.append(cst.Element(value=arg.value))
    packed_shape_val = cst.Tuple(elements=elements)

  elif len(shape_args) == 1:
    # Single arg: Check if it needs wrapping logic similar to before
    val = shape_args[0].value
    if isinstance(val, cst.Integer):
      packed_shape_val = cst.Tuple(
        elements=[cst.Element(value=val, comma=cst.Comma(whitespace_after=cst.SimpleWhitespace(" ")))]
      )
    else:
      packed_shape_val = val
  else:
    # No shape args? Illegal. Return original.
    return node

  # 3. Construct New Call Arguments
  new_args = [cst.Arg(value=input_tensor, comma=cst.Comma(whitespace_after=cst.SimpleWhitespace(" ")))]

  if packed_shape_val:
    new_args.append(cst.Arg(value=packed_shape_val))

  # 4. Resolve Target API
  op_name = "Reshape"
  target_api = ctx.lookup_api(op_name) or "jax.numpy.reshape"
  new_func = create_dotted_name(target_api)

  return node.with_changes(func=new_func, args=new_args)
