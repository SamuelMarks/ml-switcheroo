"""
Plugin for normalizing Padding arguments.

Addresses the semantic mismatch between:
1. PyTorch (`pad(x, (left, right, top, bottom))`): Pads starting from the last dimension.
2. JAX/NumPy (`pad(x, ((n_b, n_a), (c_b, c_a), ...))`): Explicit per-dimension tuples.

This plugin transforms standard 4D tensor padding (images) into the explicit
tuple-of-tuples format required by XLA compilers and NumPy-compatible libraries.
"""

import libcst as cst
from typing import Dict, Any

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
      elements=[
        cst.Element(before, comma=cst.Comma(whitespace_after=cst.SimpleWhitespace(" "))),
        cst.Element(after),
      ]
    )
  )


def _supports_numpy_padding(ctx: HookContext) -> bool:
  """Checks if target supports tuple-of-tuples padding via PluginTraits."""
  if not ctx.semantics:
    return False

  conf: Dict[str, Any] = ctx.semantics.get_framework_config(ctx.target_fw)
  if not conf:
    return False

  traits = conf.get("plugin_traits")
  if not traits:
    return False

  if isinstance(traits, dict):
    return traits.get("has_numpy_compatible_arrays", False)

  if hasattr(traits, "has_numpy_compatible_arrays"):
    return getattr(traits, "has_numpy_compatible_arrays", False)

  return False


@register_hook("padding_converter")
def transform_padding(node: cst.Call, ctx: HookContext) -> cst.Call:
  """
  Hook: Transforms padding coordinate format from Torch style to NumPy style.

  Trigger: Operations mapped to 'pad' with `requires_plugin: "padding_converter"`.

  Decoupling:
      Strictly looks up 'Pad' API. If missing, returns original node.

  Args:
      node: The original CST Call node.
      ctx: Hook Context containing target framework metadata.

  Returns:
      The transformed CST Call node if mapping exists, else original.
  """
  # 0. Capability and API Check
  if not _supports_numpy_padding(ctx):
    return node

  target_api = ctx.lookup_api("Pad")
  if not target_api:
    return node

  args = list(node.args)
  if len(args) < 2:
    return node

  input_arg = args[0]
  pad_arg = args[1]

  # We can only perform structural rewrites if the padding is a Tuple literal.
  if not isinstance(pad_arg.value, cst.Tuple):
    return node

  elements = pad_arg.value.elements

  # PyTorch Convention: (left, right, top, bottom) for 4D input NCHW -> pads W then H.
  if len(elements) == 4:
    left = elements[0].value
    right = elements[1].value
    top = elements[2].value
    bottom = elements[3].value

    # JAX/NumPy Convention (NCHW assumption): ((0,0), (0,0), (top, bottom), (left, right))
    new_elements = [
      _create_zero_pad(),  # Batch
      _create_zero_pad(),  # Channel
      _create_dim_pad(top, bottom),  # Height
      _create_dim_pad(left, right),  # Width
    ]

    new_pad_tuple = cst.Tuple(elements=new_elements)

    # Update argument list
    if input_arg.comma == cst.MaybeSentinel.DEFAULT:
      args[0] = input_arg.with_changes(comma=cst.Comma(whitespace_after=cst.SimpleWhitespace(" ")))

    args[1] = pad_arg.with_changes(value=new_pad_tuple)

    # Build Dotted Name for function
    parts = target_api.split(".")
    new_func = cst.Name(parts[0])
    for part in parts[1:]:
      new_func = cst.Attribute(value=new_func, attr=cst.Name(part))

    return node.with_changes(func=new_func, args=args)

  return node
