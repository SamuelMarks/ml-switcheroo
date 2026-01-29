"""
Plugin for Dimension-Range Flattening.

PyTorch's `flatten(start_dim, end_dim)` collapses a range of dimensions.
Mapping strategies:
1. JAX: `jax.lax.collapse(x, start, stop)` - Most robust for dynamic shapes.
2. NumPy/Default: `x.reshape(...)` or `x.ravel()`.
"""

import libcst as cst

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
  Hook: Transforms `flatten(x, start, end)` into target-specific logic.
  """
  args = list(node.args)
  if not args:
    return node

  input_arg = args[0]
  input_val = input_arg.value

  # Default values for Torch flatten semantics
  start_dim = 0
  end_dim = -1

  # Extract positional args
  if len(args) > 1:
    try:
      if isinstance(args[1].value, cst.Integer):
        start_dim = int(args[1].value.value)
    except ValueError:
      pass

  if len(args) > 2:
    try:
      if isinstance(args[2].value, cst.Integer):
        end_dim = int(args[2].value.value)
    except ValueError:
      pass

  # Extract keyword args
  for arg in args:
    if arg.keyword:
      if arg.keyword.value == "start_dim" and isinstance(arg.value, cst.Integer):
        start_dim = int(arg.value.value)
      if arg.keyword.value == "end_dim" and isinstance(arg.value, cst.Integer):
        end_dim = int(arg.value.value)

  # Lookup the API configured in ODL/Semantics
  # We try "Flatten" first (The definition name), then specific variants
  target_api = ctx.lookup_api("Flatten") or ctx.lookup_api("flatten")
  if not target_api:
    # Fallback to internal/legacy keys just in case
    target_api = ctx.lookup_api("flatten_range") or ctx.lookup_api("flatten_full")

  if not target_api:
    return node

  # --- STRATEGY: JAX collapse ---
  # flatten(x, 1) -> collapse(x, 1, x.ndim)
  # flatten(x, 1, 2) -> collapse(x, 1, 3) (Exclusive stop)
  if "collapse" in target_api:
    new_func = _create_dotted_name(target_api)

    # Arg 1: Input (x)
    arg0 = input_arg.with_changes(comma=cst.Comma(whitespace_after=cst.SimpleWhitespace(" ")))

    # Arg 2: start_dim
    arg1_val = cst.Integer(str(start_dim))
    arg1 = cst.Arg(value=arg1_val, comma=cst.Comma(whitespace_after=cst.SimpleWhitespace(" ")))

    # Arg 3: stop_dimension
    # PyTorch 'end_dim' is inclusive. JAX 'stop_dimension' is exclusive.
    # If end_dim == -1, it means "until the end", which corresponds to x.ndim in JAX.
    if end_dim == -1:
      # Generate: x.ndim
      arg2_val = cst.Attribute(value=input_val, attr=cst.Name("ndim"))
    else:
      # Generate: end_dim + 1
      arg2_val = cst.Integer(str(end_dim + 1))

    arg2 = cst.Arg(value=arg2_val)

    return node.with_changes(func=new_func, args=[arg0, arg1, arg2])

  # --- STRATEGY: Ravel (Full Flatten) ---
  # flatten(x) or flatten(x, 0, -1) -> ravel(x)
  if start_dim == 0 and end_dim == -1:
    if "ravel" in target_api or "flatten" in target_api:
      new_func = _create_dotted_name(target_api)
      return node.with_changes(func=new_func, args=[input_arg])

  # --- STRATEGY: Reshape (Batch Preserving) ---
  # flatten(x, 1) -> reshape(x, (x.shape[0], -1))
  if start_dim == 1 and end_dim == -1:
    new_func = _create_dotted_name(target_api)

    # Construct shape tuple: (x.shape[0], -1)
    shape_attr = cst.Attribute(value=input_val, attr=cst.Name("shape"))
    batch_dim = cst.Subscript(value=shape_attr, slice=[cst.SubscriptElement(slice=cst.Index(value=cst.Integer("0")))])
    neg_one = cst.UnaryOperation(operator=cst.Minus(), expression=cst.Integer("1"))

    shape_tuple = cst.Tuple(
      elements=[
        cst.Element(value=batch_dim, comma=cst.Comma(whitespace_after=cst.SimpleWhitespace(" "))),
        cst.Element(value=neg_one),
      ]
    )

    if input_arg.comma == cst.MaybeSentinel.DEFAULT:
      input_arg = input_arg.with_changes(comma=cst.Comma(whitespace_after=cst.SimpleWhitespace(" ")))

    new_args = [input_arg, cst.Arg(value=shape_tuple)]
    return node.with_changes(func=new_func, args=new_args)

  return node
