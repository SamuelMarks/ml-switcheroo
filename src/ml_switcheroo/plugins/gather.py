"""
Plugin for Gather Semantics Adaptation.

Addresses the signature mismatch between:
1. PyTorch: `torch.gather(input, dim, index, *, sparse_grad=False, out=None)`
2. JAX/NumPy: `jax.numpy.take_along_axis(arr, indices, axis)`

Transformation:
- Reorders positional arguments: `(input, dim, index)` -> `(input, index, dim)`.
- Maps keyword arguments: `dim` -> `axis`, `index` -> `indices`.
- Strips unsupported kwargs like `sparse_grad` or `out`.

This ensures that `torch.gather(x, 1, idx)` correctly becomes `jnp.take_along_axis(x, idx, 1)`.
"""

import libcst as cst
from typing import List, Optional

from ml_switcheroo.core.hooks import register_hook, HookContext


def _create_dotted_name(name_str: str) -> cst.BaseExpression:
  parts = name_str.split(".")
  node = cst.Name(parts[0])
  for part in parts[1:]:
    node = cst.Attribute(value=node, attr=cst.Name(part))
  return node


def _is_framework_module(node: cst.CSTNode) -> bool:
  """Detects if a node refers to a framework module rather than a tensor object."""
  if isinstance(node, cst.Name):
    # List of common roots to treat as modules
    return node.value in {"torch", "jax", "tensorflow", "tf", "numpy", "np", "flax", "nn"}
  return False


@register_hook("gather_adapter")
def transform_gather(node: cst.Call, ctx: HookContext) -> cst.Call:
  """
  Hook: Adapts gather calls to take_along_axis semantics.

  Target Frameworks: JAX, NumPy.
  """
  if ctx.target_fw.lower() not in ["jax", "numpy"]:
    return node

  args = list(node.args)

  # 1. Identify Input Wrapper
  # If called as method `x.gather(...)`, input is implicit `x`.
  # If called as function `torch.gather(x, ...)`, input is arg 0.

  is_method_call = False
  if isinstance(node.func, cst.Attribute):
    # Distinguish x.gather vs torch.gather
    if not _is_framework_module(node.func.value):
      is_method_call = True

  # Placeholders
  input_arg = None
  dim_arg = None
  index_arg = None

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
  # They might be positional or keyword

  # Heuristic for positional:
  # torch.gather(dim, index) <-- Method signature usually
  # torch.gather(input, dim, index) <-- Function signature

  # Search for keywords first
  for arg in args:
    if arg.keyword:
      kw = arg.keyword.value
      if kw == "dim":
        dim_arg = arg
      elif kw == "index":
        index_arg = arg

  # Fill gaps with positionals
  remaining_pos = [a for a in args[current_idx:] if not a.keyword]

  if len(remaining_pos) > 0 and dim_arg is None:
    dim_arg = remaining_pos[0]
    # Torch `dim` is first pos arg after input

  if len(remaining_pos) > 1 and index_arg is None:
    index_arg = remaining_pos[1]

  # 3. Construct New Argument List for take_along_axis(arr, indices, axis)
  if not input_arg or not index_arg:
    # Cannot safely transform without critical args
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
  # Strip keyword 'index' effectively, or map to 'indices' if we want to be explicit
  # JAX supports positional here. Let's use positional for safety.
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
      # Last arg, no comma generally unless tuple style
    )
    new_args.append(clean_dim)

  # 4. Change Function Name
  target_api = ctx.lookup_api("Gather") or "jax.numpy.take_along_axis"
  new_func = _create_dotted_name(target_api)

  return node.with_changes(func=new_func, args=new_args)
