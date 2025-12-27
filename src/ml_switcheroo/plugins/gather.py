"""
Plugin for Gather Semantics Adaptation.

Addresses the signature mismatch between:
1. PyTorch: `torch.gather(input, dim, index, *, sparse_grad=False, out=None)`
2. Target Framework (e.g. JAX/NumPy): `target.take_along_axis(arr, indices, axis)`

Transformation:
- Reorders positional arguments: `(input, dim, index)` -> `(input, index, dim)`.
- Maps keyword arguments: `dim` -> `axis`, `index` -> `indices`.
- Strips unsupported kwargs like `sparse_grad` or `out`.

Decoupling Logic:
- Removes hardcoded framework checks.
- Strict lookup: If `Gather` is not mapped in semantics, preserves original call.
"""

import libcst as cst
from typing import List

from ml_switcheroo.core.hooks import register_hook, HookContext
from ml_switcheroo.plugins.utils import create_dotted_name, is_framework_module_node


@register_hook("gather_adapter")
def transform_gather(node: cst.Call, ctx: HookContext) -> cst.Call:
  """
  Hook: Adapts gather calls to take_along_axis semantics.

  Target API Convention: `func(input, indices, axis)`.
  Source (Torch) Convention: `func(input, dim, index)`.

  Args:
      node: The original CST Call node.
      ctx: Hook Context containing semantic definitions.

  Returns:
      Transformed Call node if API mapping exists, else original node.
  """
  # 0. API Resolution (Strict)
  target_api = ctx.lookup_api("Gather")
  if not target_api:
    # Fail safe if target framework has no mapping for 'Gather'
    return node

  # 1. Identify Input Wrapper
  # If called as method `x.gather(...)`, input is implicit `x` (Attributes).
  # If called as function `torch.gather(x, ...)`, input is arg 0.

  is_method_call = False
  if isinstance(node.func, cst.Attribute):
    # Distinguish x.gather vs torch.gather using centralized util
    if not is_framework_module_node(node.func.value, ctx):
      is_method_call = True

  # Placeholders
  input_arg = None
  dim_arg = None
  index_arg = None

  args = list(node.args)
  current_idx = 0

  # Handle Input Arg
  if not is_method_call:
    if len(args) > current_idx:
      input_arg = args[current_idx]
      current_idx += 1
  else:
    # Input is the receiver, we will move it to first arg for take_along_axis
    # node.func.value is the object expression 'x'
    input_arg = cst.Arg(value=node.func.value)

  # 2. Parse remaining args (dim, index)
  # Search for keywords first
  for arg in args:
    if arg.keyword:
      kw = arg.keyword.value
      if kw == "dim":
        dim_arg = arg
      elif kw == "index":
        index_arg = arg

  # Fill gaps with remaining positionals
  remaining_pos = [a for a in args[current_idx:] if not a.keyword]

  if len(remaining_pos) > 0 and dim_arg is None:
    dim_arg = remaining_pos[0]
    # Torch `dim` is first pos arg after input

  if len(remaining_pos) > 1 and index_arg is None:
    index_arg = remaining_pos[1]

  # 3. Construct New Argument List for take_along_axis(arr, indices, axis)
  if not input_arg or not index_arg:
    # Cannot safely transform without critical args, fail-open (return original)
    return node

  new_args = []

  # Arg 0: arr (Input)
  # Ensure comma if needed
  clean_input = input_arg.with_changes(
    keyword=None,
    equal=cst.MaybeSentinel.DEFAULT,
    comma=cst.Comma(whitespace_after=cst.SimpleWhitespace(" ")),
  )
  new_args.append(clean_input)

  # Arg 1: indices (Index)
  # Strip keyword 'index' effectively
  clean_index = index_arg.with_changes(
    keyword=None,
    equal=cst.MaybeSentinel.DEFAULT,
    comma=cst.Comma(whitespace_after=cst.SimpleWhitespace(" ")),
  )
  new_args.append(clean_index)

  # Arg 2: axis (Dim)
  if dim_arg:
    # Ensure 'dim' keyword is removed or renamed to 'axis'
    # Since it's positional 3rd arg in take_along_axis, we can drop keyword
    clean_dim = dim_arg.with_changes(
      keyword=None,
      equal=cst.MaybeSentinel.DEFAULT,
      comma=cst.MaybeSentinel.DEFAULT,
      # Last arg, no comma generally
    )
    new_args.append(clean_dim)

  # 4. Change Function Name
  new_func = create_dotted_name(target_api)
  return node.with_changes(func=new_func, args=new_args)
