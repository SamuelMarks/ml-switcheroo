"""
Plugin for "View" Semantics and Reshape strictness.

Addresses:
    PyTorch `tensor.view(*shape)` requires contiguous memory and shares data.
    JAX/NumPy `reshape(arr, shape)` works on any array.

Plugin Logic:
    1.  **Strict Mode**: If `config.strict_mode` is enabled, this plugin can inject
        synchronization (e.g. `block_until_ready()`).
    2.  **Argument Packing**: Packs varargs `view(a, b)` -> `reshape((a, b))`.
    3.  **Decoupling**: Strictly relies on lookup API. If `Reshape` or `View` are
        not mapped in semantics for the target framework, returns original node.
"""

import libcst as cst
from typing import List, Union

from ml_switcheroo.core.hooks import register_hook, HookContext


def _create_dotted_name(name_str: str) -> cst.BaseExpression:
  """Helper to create a CST Attribute chain from string."""
  parts = name_str.split(".")
  node = cst.Name(parts[0])
  for part in parts[1:]:
    node = cst.Attribute(value=node, attr=cst.Name(part))
  return node


@register_hook("view_semantics")
def transform_view_semantics(node: cst.Call, ctx: HookContext) -> cst.Call:
  """
  Hook: Maps `view` -> `reshape` with optional strictness injections.

  Transformation:
      Input: `x.view(a, b)`
      Standard Output: `target_api(x, (a, b))`
      Strict Output:   `target_api(x, (a, b)).block_until_ready()` [If defined in Traits]

  Decoupling:
      Lookup precedence: "Reshape" -> "View".
      If lookup fails, aborts transformation.
  """
  # 0. Resolve Target API
  target_api = ctx.lookup_api("Reshape") or ctx.lookup_api("View")
  if not target_api:
    # Fail safe if target framework has no mapping
    return node

  # 1. Normalize Arguments (Pack Varargs -> Tuple)
  input_tensor = None
  orig_args = []

  if isinstance(node.func, cst.Attribute):
    # Method call x.view(...)
    input_tensor = node.func.value
    orig_args = list(node.args)
  else:
    # Function call view(x, ...)
    if not node.args:
      return node
    input_tensor = node.args[0].value
    orig_args = list(node.args[1:])

  # Check if args need packing (multiple args or single int arg)
  needs_tuple = False
  if len(orig_args) > 1:
    needs_tuple = True
  elif len(orig_args) == 1:
    val = orig_args[0].value
    if isinstance(val, cst.Integer):
      needs_tuple = True

  if needs_tuple:
    # Pack
    elements = [cst.Element(value=arg.value) for arg in orig_args]
    shape_arg_val = cst.Tuple(elements=elements)
    new_args = [
      cst.Arg(value=input_tensor, comma=cst.Comma(whitespace_after=cst.SimpleWhitespace(" "))),
      cst.Arg(value=shape_arg_val),
    ]
  else:
    # Args are already likely a tuple or singular var.
    # Just ensure input tensor is first arg if converting method->func
    if isinstance(node.func, cst.Attribute):
      # method x.view(shape) -> func(x, shape)
      new_args = [cst.Arg(value=input_tensor, comma=cst.Comma(whitespace_after=cst.SimpleWhitespace(" ")))]
      if orig_args:
        new_args.append(orig_args[0])
    else:
      new_args = list(node.args)

  # 2. Rename Function
  new_func = _create_dotted_name(target_api)
  reshape_call = node.with_changes(func=new_func, args=new_args)

  # 3. Handle Strict Mode (Optional Blocking)
  # Read strict mode from config and trait method from context
  is_strict = getattr(ctx._runtime_config, "strict_mode", False)

  if is_strict:
    trait_method = ctx.plugin_traits.strict_materialization_method
    if trait_method:
      strict_call = cst.Call(func=cst.Attribute(value=reshape_call, attr=cst.Name(trait_method)), args=[])
      return strict_call

  return reshape_call
