"""
Plugin for packing variable positional arguments into a sequence (tuple/list).

Essential for mapping vararg APIs (like `torch.permute(x, 0, 2, 1)`) to sequence-based
APIs (like `jax.numpy.transpose(x, axes=(0, 2, 1))`).

Decoupling Update:
This plugin no longer checks framework names directly. It reads the `pack_to_tuple`
keyword field from the `Variant` metadata exposed via `HookContext`.
"""

import libcst as cst
from typing import List, Optional

from ml_switcheroo.core.hooks import register_hook, HookContext


def _create_dotted_name(name_str: str) -> cst.BaseExpression:
  """Creates a CST attribute chain from a dotted string."""
  parts = name_str.split(".")
  node = cst.Name(parts[0])
  for part in parts[1:]:
    node = cst.Attribute(value=node, attr=cst.Name(part))
  return node


@register_hook("pack_varargs")
def pack_varargs(node: cst.Call, ctx: HookContext) -> cst.Call:
  """
  Plugin Hook: Packs trailing positional arguments into a keyword tuple.

  Triggers:
      Operations marked with `requires_plugin: "pack_varargs"` or manual hook usage.

  Configuration:
      The keyword name (e.g. 'axes' vs 'perm') is derived from the Semantic Variant definition:
      `{"pack_to_tuple": "perm"}` (TensorFlow) vs `{"pack_to_tuple": "axes"}` (JAX).
      Defaults to "axes" if not specified.

  Args:
      node: The original CST Call node.
      ctx: HookContext giving access to Semantic definitions.

  Returns:
      The transformed CST Call node using keyword tuple arguments.
  """
  # 1. Access Variant Metadata via Context
  # This replaces hardcoded checks like 'if target_fw == tensorflow'
  variant = ctx.current_variant

  # Resolve API name
  target_api = variant.api if variant else None

  # Use context lookup fallback if variant isn't populated (e.g. partial manual hook invocation)
  if not target_api:
    # Identify abstract op associated with this hook mechanism?
    # usually permute_dims.
    target_api = ctx.lookup_api("permute_dims")

  if not target_api:
    return node

  # 2. Determine Keyword Name Data-Driven
  keyword_name = "axes"  # Default standard (JAX/NumPy)
  if variant and variant.pack_to_tuple:
    keyword_name = variant.pack_to_tuple

  # 3. Extract Arguments from Source Call
  # We assume signature `func(input, *dims)`.
  # Arg 0 is the Tensor/Array input.
  # Args 1..N are the dimensions to pack.
  all_args = list(node.args)
  if not all_args:
    return node

  input_arg = all_args[0]
  dims_args = all_args[1:]

  packed_elements: List[cst.BaseElement] = []
  for arg in dims_args:
    # If we hit a keyword arg, we assume packing should stop or ignore it
    if arg.keyword:
      continue

    # Clean the argument values (e.g. remove trailing comma style from original position)
    clean_val = arg.value
    packed_elements.append(cst.Element(value=clean_val))

  # 4. Create Tuple Node for the packed sequence
  # (d0, d1, d2)
  tuple_node = cst.Tuple(elements=packed_elements)

  # 5. Construct New Arguments List
  # Format: [input_arg, {keyword_name}=tuple_node]

  # Ensure input_arg has a comma if it's followed by keyword args
  input_arg_clean = input_arg.with_changes(comma=cst.Comma(whitespace_after=cst.SimpleWhitespace(" ")))

  new_args = [input_arg_clean]

  if packed_elements:
    axes_arg = cst.Arg(
      keyword=cst.Name(keyword_name),
      value=tuple_node,
      equal=cst.AssignEqual(whitespace_before=cst.SimpleWhitespace(""), whitespace_after=cst.SimpleWhitespace("")),
    )
    new_args.append(axes_arg)

  # 6. Construct New Function Name (e.g. jax.numpy.transpose)
  new_func = _create_dotted_name(target_api)

  return node.with_changes(func=new_func, args=new_args)
