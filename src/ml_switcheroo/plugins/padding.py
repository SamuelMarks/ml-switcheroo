"""
Plugin for normalizing Padding arguments.

Addresses the semantic mismatch between:
1. PyTorch (`pad(x, (left, right, top, bottom))`): Pads starting from the last dimension.
2. JAX/NumPy (`pad(x, ((n_b, n_a), (c_b, c_a), ...))`): Explicit per-dimension tuples.

This plugin transforms standard 4D tensor padding (images) into the explicit
tuple-of-tuples format required by XLA compilers.
"""

import libcst as cst

from ml_switcheroo.core.hooks import register_hook, HookContext


def _create_zero_pad() -> cst.Element:
  """Helper to create (0, 0) tuple element."""
  return cst.Element(
    value=cst.Tuple(
      elements=[
        cst.Element(cst.Integer("0"), comma=cst.Comma(whitespace_after=cst.SimpleWhitespace(" "))),
        cst.Element(cst.Integer("0")),
      ]
    )
  )


def _create_dim_pad(before: cst.BaseExpression, after: cst.BaseExpression) -> cst.Element:
  """Helper to create (before, after) tuple element."""
  return cst.Element(
    value=cst.Tuple(
      elements=[cst.Element(before, comma=cst.Comma(whitespace_after=cst.SimpleWhitespace(" "))), cst.Element(after)]
    )
  )


@register_hook("padding_converter")
def transform_padding(node: cst.Call, ctx: HookContext) -> cst.Call:
  """
  Hook: Transforms padding coordinate format.
  Trigger: Operations mapped to 'pad' with `requires_plugin: "padding_converter"`.
  """
  if ctx.target_fw not in ["jax", "numpy", "flax_nnx", "tensorflow"]:
    return node

  args = list(node.args)
  # Standard signature: pad(input, pad, mode=..., value=...)
  if len(args) < 2:
    return node

  input_arg = args[0]
  pad_arg = args[1]

  # We can only perform structural rewrites if the padding is a Tuple literal.
  # Dynamic variables (e.g., pad(x, my_tuple)) must be handled by the runtime or pass-through.
  if not isinstance(pad_arg.value, cst.Tuple):
    return node

  elements = pad_arg.value.elements

  # PyTorch Convention: (left, right, top, bottom) for 4D input NCHW -> pads W then H.
  # Map indices: 0->Left, 1->Right, 2->Top, 3->Bottom

  if len(elements) == 4:
    left = elements[0].value
    right = elements[1].value
    top = elements[2].value
    bottom = elements[3].value

    # JAX/NumPy Convention (NCHW assumption): ((0,0), (0,0), (top, bottom), (left, right))
    # 1. Batch Dim (N) -> (0, 0)
    # 2. Channel Dim (C) -> (0, 0)
    # 3. Height Dim (H) -> (top, bottom)
    # 4. Width Dim (W) -> (left, right)

    new_elements = [
      _create_zero_pad(),  # Batch
      _create_zero_pad(),  # Channel
      _create_dim_pad(top, bottom),  # Height
      _create_dim_pad(left, right),  # Width
    ]

    # Construct ((...), (...), ...)
    new_pad_tuple = cst.Tuple(elements=new_elements)

    # Update argument list
    # Ensure input arg has comma syntax preserved
    if input_arg.comma == cst.MaybeSentinel.DEFAULT:
      args[0] = input_arg.with_changes(comma=cst.Comma(whitespace_after=cst.SimpleWhitespace(" ")))

    # Replace the padding arg
    args[1] = pad_arg.with_changes(value=new_pad_tuple)

    # Resolve Target API Name
    # Fallback to jax.numpy.pad if lookup fails to ensure plugin utility
    target_api = ctx.lookup_api("Pad") or "jax.numpy.pad"

    # Build Dotted Name for function
    parts = target_api.split(".")
    new_func = cst.Name(parts[0])
    for part in parts[1:]:
      new_func = cst.Attribute(value=new_func, attr=cst.Name(part))

    return node.with_changes(func=new_func, args=args)

  # Future: Handle 6-tuple (3D) or 2-tuple (1D) cases
  return node
