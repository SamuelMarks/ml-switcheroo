"""
Plugin for normalizing Padding arguments.

Addresses the semantic mismatch between:
1. PyTorch (`pad(x, (left, right, top, bottom))`): Pads starting from the last dimension.
2. JAX/NumPy (`pad(x, ((n_b, n_a), (c_b, c_a), ...))`): Explicit per-dimension tuples.

This plugin transforms standard 4D tensor padding (images) into the explicit
tuple-of-tuples format required by XLA compilers and NumPy-compatible libraries.
It relies on `PluginTraits` to detect if the target framework requires this format.
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
  """
  Checks if the target framework supports explicit tuple-of-tuples padding
  (NumPy/JAX style) via PluginTraits.

  Decouples the plugin from hardcoded framework lists like ['jax', 'numpy'].

  Args:
      ctx: The hook context containing the semantics manager and target framework.

  Returns:
      bool: True if the target framework declares 'has_numpy_compatible_arrays'.
  """
  if not ctx.semantics:
    return False

  # Retrieve dict configuration for the active target framework
  conf: Dict[str, Any] = ctx.semantics.get_framework_config(ctx.target_fw)
  if not conf:
    return False

  # Navigate: config -> plugin_traits -> has_numpy_compatible_arrays
  # We handle both dict access (from JSON) and object access (if hydrated objects are used)
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

  Trigger:
      Operations mapped to 'pad' with `requires_plugin: "padding_converter"`.

  Transformation:
      Input:  `F.pad(x, (left, right, top, bottom))`
      Output: `jax.numpy.pad(x, ((0, 0), (0, 0), (top, bottom), (left, right)))`

  Args:
      node: The original CST Call node.
      ctx: Hook Context containing target framework metadata.

  Returns:
      The transformed CST Call node if the framework is compatible, else original.
  """
  # 0. Capability Check
  if not _supports_numpy_padding(ctx):
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
