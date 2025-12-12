"""
Plugin for packing variable positional arguments into a sequence (tuple/list).

Essential for mapping vararg APIs (like `torch.permute(x, 0, 2, 1)`) to sequence-based
APIs (like `jax.numpy.transpose(x, axes=(0, 2, 1))`).

Strategy:
1.  Identify the operation (e.g. `permute_dims`).
2.  Lookup the target API (e.g. `jax.numpy.transpose`).
3.  Separate the primary input argument (first positional) from the varargs.
4.  Pack the trailing varargs into a `Tuple` node.
5.  Construct the new call with the sequence passed as a keyword argument (e.g. `axes=...`).
"""

import libcst as cst

from ml_switcheroo.core.hooks import register_hook, HookContext


def _create_dotted_name(name_str: str) -> cst.BaseExpression:
  """Creates a CST attribute chain from a string string."""
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
      Operations marked with `requires_plugin: "pack_varargs"`.
      Designed for abstract operations like `permute_dims`.

  Transformation:
      Input:  `torch.permute(x, 0, 2, 1)`
      Output: `jax.numpy.transpose(x, axes=(0, 2, 1))` (if mapped)

  Args:
      node: The original CST Call node.
      ctx: HookContext for API lookup directly from Semantics.

  Returns:
      The transformed CST Call node, or original if target mapping is missing.
  """
  # 1. Determine Target API
  # We infer the abstract op is 'permute_dims' based on this plugin's primary use case.
  op_id = "permute_dims"
  target_api = ctx.lookup_api(op_id)

  # If we don't know the target logic via JSON, return unmodified.
  if not target_api:
    return node

  # 2. Extract Arguments
  # We assume signature `func(input, *dims)`.
  # Arg 0 is the Tensor/Array input.
  # Args 1..N are the dimensions to pack.
  all_args = list(node.args)
  if not all_args:
    return node

  input_arg = all_args[0]
  dims_args = all_args[1:]

  # If no dims provided, or if existing dims are keyword args, the logic might differ.
  # We filter out any keyword matches just in case user did something weird.
  packed_elements = []
  for arg in dims_args:
    # If we hit a keyword arg, packing stops
    if arg.keyword:
      continue
    # Clean the argument (remove trailing comma style from original position)
    clean_val = arg.value
    packed_elements.append(cst.Element(value=clean_val))

  # 3. Create Tuple Node
  # (d0, d1, d2)
  tuple_node = cst.Tuple(elements=packed_elements)

  # 4. Construct New Arguments List
  # [input_arg, axes=tuple_node]
  new_args = [input_arg.with_changes(comma=cst.Comma(whitespace_after=cst.SimpleWhitespace(" ")))]

  if packed_elements:
    axes_arg = cst.Arg(
      keyword=cst.Name("axes"),
      value=tuple_node,
      equal=cst.AssignEqual(whitespace_before=cst.SimpleWhitespace(""), whitespace_after=cst.SimpleWhitespace("")),
    )
    new_args.append(axes_arg)

  # 5. Construct New Function Name
  new_func = _create_dotted_name(target_api)

  return node.with_changes(func=new_func, args=new_args)
