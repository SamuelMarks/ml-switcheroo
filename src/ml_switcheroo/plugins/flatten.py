"""
Plugin for Dimension-Range Flattening.

PyTorch's `flatten(start_dim, end_dim)` is a powerful operation that collapses
a specific range of dimensions. JAX and NumPy rely on:
1. `ravel()`: Flattens everything (equivalent to `flatten(0, -1)`).
2. `reshape()`: Flattens dimensions if the new shape is calculated correctly.

Common Use Case:
    `x = torch.flatten(x, 1)` -> Flattens from dim 1 to end (preserving batch).
    Used in almost every CNN classifier head.

Transformation Strategy:
    1. If `start_dim=1` and `end_dim=-1` (or implicit):
       Generate `reshape(x, (x.shape[0], -1))`.
    2. If `start_dim=0` and `end_dim=-1`:
       Generate `ravel(x)`.
    3. Input safety: If the input is a complex expression (e.g. function call),
       we duplicate it in the shape lookup. This is acceptable for pure functions
       but suboptimal. Assuming input is usually a variable (`x`).
"""

import libcst as cst
from typing import List, Optional

from ml_switcheroo.core.hooks import register_hook, HookContext


def _create_dotted_name(name_str: str) -> cst.BaseExpression:
  """Helper to create a CST Attribute chain from string."""
  parts = name_str.split(".")
  node = cst.Name(parts[0])
  for part in parts[1:]:
    node = cst.Attribute(value=node, attr=cst.Name(part))
  return node


@register_hook("flatten_range")
def transform_flatten(node: cst.Call, ctx: HookContext) -> cst.Call:
  """
  Hook: Transforms `flatten(x, start, end)` into `reshape` or `ravel`.

  Target Frameworks: JAX, NumPy, TensorFlow, MLX.
  """
  if ctx.target_fw.lower() == "torch":
    return node

  args = list(node.args)
  if not args:
    return node

  input_arg = args[0]
  input_val = input_arg.value

  # Default values for Torch
  start_dim = 0
  end_dim = -1

  # extract positional args
  if len(args) > 1:
    try:
      # We only handle integer literals for structural rewriting
      if isinstance(args[1].value, cst.Integer):
        start_dim = int(args[1].value.value)
    except ValueError:
      pass  # Keep default 0 implies full flatten? No, partial.

  if len(args) > 2:
    try:
      if isinstance(args[2].value, cst.Integer):
        end_dim = int(args[2].value.value)
    except ValueError:
      pass

  # Check keyword args (override positional if present)
  for arg in args:
    if arg.keyword:
      if arg.keyword.value == "start_dim":
        if isinstance(arg.value, cst.Integer):
          start_dim = int(arg.value.value)
      if arg.keyword.value == "end_dim":
        if isinstance(arg.value, cst.Integer):
          end_dim = int(arg.value.value)

  # Strategy 1: Full Flatten (ravel)
  if start_dim == 0 and end_dim == -1:
    target_api = ctx.lookup_api("flatten_full") or "jax.numpy.ravel"
    new_func = _create_dotted_name(target_api)
    return node.with_changes(func=new_func, args=[input_arg])

  # Strategy 2: Batch-Preserving Flatten (Reshape)
  # Case: flatten(x, 1) -> reshape(x, (x.shape[0], -1))
  if start_dim == 1 and end_dim == -1:
    target_api = ctx.lookup_api("flatten_range") or "jax.numpy.reshape"
    new_func = _create_dotted_name(target_api)

    # Construct shape tuple: (x.shape[0], -1)

    # 1. x.shape[0]
    # We access the 'value' of the input argument directly.
    # Warning: If input_val is 'call()', generated code is 'call().shape[0]'.
    shape_attr = cst.Attribute(value=input_val, attr=cst.Name("shape"))
    batch_dim = cst.Subscript(value=shape_attr, slice=[cst.SubscriptElement(slice=cst.Index(value=cst.Integer("0")))])

    # 2. -1
    neg_one = cst.UnaryOperation(operator=cst.Minus(), expression=cst.Integer("1"))

    shape_tuple = cst.Tuple(
      elements=[
        cst.Element(value=batch_dim, comma=cst.Comma(whitespace_after=cst.SimpleWhitespace(" "))),
        cst.Element(value=neg_one),
      ]
    )

    # New Args: (input, shape_tuple)
    # Ensure comma on input arg
    if input_arg.comma == cst.MaybeSentinel.DEFAULT:
      input_arg = input_arg.with_changes(comma=cst.Comma(whitespace_after=cst.SimpleWhitespace(" ")))

    new_args = [input_arg, cst.Arg(value=shape_tuple)]

    return node.with_changes(func=new_func, args=new_args)

  # Fallback: Untranslatable complex slicing (e.g. flatten(x, 2, 3)) without knowing rank.
  # Return original node to trigger escape hatch if configured, or pass through.
  return node
