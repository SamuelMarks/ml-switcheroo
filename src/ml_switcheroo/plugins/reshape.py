"""
Plugin for "View" Semantics and Reshape strictness.

Addresses:
    PyTorch `tensor.view(*shape)` requires contiguous memory and shares data.
    JAX `jnp.reshape(arr, shape)` works on any array, copying if necessary, producing immutable output.

Semantic Mismatch:
    In PyTorch, `view` is often used as an assertion of zero-copy reshaping.
    In JAX, copy/view distinction is less relevant for correctness due to immutability,
    but relevant for performance.

Plugin Logic:
    1.  **Strict Mode**: If `config.strict_mode` is enabled, this plugin can inject
        synchronization (`block_until_ready()`) or explicit copies to isolate performance artifacts,
        depending on the configuration policy. (Prompt request: injects block/copy).
    2.  **Argument Packing**: Handles the conversion from varargs `view(a, b)` to tuple `reshape((a, b))`
        if not already handled by a prior pass.
"""

import libcst as cst
from typing import List, Union

from ml_switcheroo.core.hooks import register_hook, HookContext


def _create_dotted_name(name_str: str) -> cst.BaseExpression:
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
      Standard Output: `jax.numpy.reshape(x, (a, b))`
      Strict Output:   `jax.numpy.reshape(x, (a, b)).block_until_ready()`
  """
  if ctx.target_fw not in ["jax", "numpy", "flax"]:
    return node

  # 1. Normalize Arguments (Pack Varargs)
  # Similar logic to shape_packing.py, but specific here to ensure View works standalone
  input_tensor = None
  shape_elements = []

  if isinstance(node.func, cst.Attribute):
    # Method call x.view(...)
    input_tensor = node.func.value
    orig_args = node.args
  else:
    # Function call view(x, ...) - uncommon for view, but standard for reshape
    if not node.args:
      return node
    input_tensor = node.args[0].value
    orig_args = node.args[1:]

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
    # Already likely a tuple or var
    # Just ensure input tensor is first arg if converting method->func
    if isinstance(node.func, cst.Attribute):
      new_args = [cst.Arg(value=input_tensor, comma=cst.Comma(whitespace_after=cst.SimpleWhitespace(" "))), orig_args[0]]
    else:
      new_args = list(node.args)

  # 2. Rename Function
  target_api = "jax.numpy.reshape"
  new_func = _create_dotted_name(target_api)

  reshape_call = node.with_changes(func=new_func, args=new_args)

  # 3. Handle Strict Mode
  # If config says be strict about execution (or mimicking sync behavior of some Torch ops contexts)
  # The prompt specifically asked for ".block_until_ready() or copies"

  # We check a hypothetical boolean in runtime config.
  # Current RuntimeConfig might not have 'strict_mode' field defined in types yet,
  # but Python allows attribute access or dictionary lookup if we treat it dynamically.
  is_strict = getattr(ctx._runtime_config, "strict_mode", False)

  if is_strict:
    # Append .block_until_ready()
    # Allows for precise timing benchmarks matching "contiguity" checks overhead?
    # Or simply forces materialization.
    strict_call = cst.Call(func=cst.Attribute(value=reshape_call, attr=cst.Name("block_until_ready")), args=[])
    return strict_call

  return reshape_call
