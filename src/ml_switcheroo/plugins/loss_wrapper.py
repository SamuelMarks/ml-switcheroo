"""
Plugin for Loss Reduction Semantics.

Addresses the mismatch between:
1. PyTorch: `loss = F.cross_entropy(..., reduction='mean')` (Scalar output by default).
2. Functional Frameworks (JAX/Optax): `loss = optax.softmax_cross_entropy(x, y)`
   (Vector output per batch element).

JAX libraries typically return the loss *per sample* to support vmap/pmap flexibility.
PyTorch defaults to averaging (`mean`) immediately.

Transformation:
1. Detects `reduction` keyword argument.
2. Strips the argument (as Optax/JAX funcs don't usually accept it).
3. Wraps the function call in `Mean(x)` or `Sum(x)`.
   - Dynamically looks up "Mean" or "Sum" API from the Semantic Knowledge Base.
   - Supports any target framework definition (e.g. `tf.reduce_mean`, `jnp.mean`, `mx.mean`).
4. If `reduction='none'`, leaves the vector output alone.
"""

import libcst as cst
from typing import Optional

from ml_switcheroo.core.hooks import register_hook, HookContext


def _create_dotted_name(name_str: str) -> cst.BaseExpression:
  """Helper to create CST attribute chain."""
  parts = name_str.split(".")
  node = cst.Name(parts[0])
  for part in parts[1:]:
    node = cst.Attribute(value=node, attr=cst.Name(part))
  return node


@register_hook("loss_reduction")
def transform_loss_reduction(node: cst.Call, ctx: HookContext) -> cst.CSTNode:
  """
  Hook: Wraps loss functions to apply reduction.

  Trigger: Operations mapped with `requires_plugin: "loss_reduction"`.
  Target: Frameworks requiring explicit reduction (JAX, Flax).

  Args:
      node: The original CST Call node.
      ctx: HookContext for API lookup.

  Returns:
      Transformed Call node (wrapped or unwrapped).
  """
  args = list(node.args)

  # 1. Determine Reduction Mode
  # Default PyTorch behavior is 'mean'
  reduction_mode = "mean"
  reduction_arg_index = -1

  for i, arg in enumerate(args):
    if arg.keyword and arg.keyword.value == "reduction":
      reduction_arg_index = i
      # Check value
      if isinstance(arg.value, cst.SimpleString):
        val = arg.value.value.strip("'").strip('"')
        reduction_mode = val
      elif isinstance(arg.value, cst.Name):
        # If variable passed (e.g. reduction=my_mode), we can't statically wrap.
        # We assume standard string literals for now.
        pass
      break

  # 2. Modify Arguments (Strip reduction arg)
  if reduction_arg_index != -1:
    del args[reduction_arg_index]

    # 3. Resolve Inner Function Name from Context
  # We reconstruct the inner call using the mapped API for the current operation.
  # If op_id is unknown (generic hook usage), fallback to 'cross_entropy' heuristic logic?
  loss_op_id = ctx.current_op_id

  # Fallback heuristic if context is empty (e.g. raw test usage)
  if not loss_op_id:
    # Assume CrossEntropy as default test case
    loss_op_id = "CrossEntropyLoss"

  target_loss_api = ctx.lookup_api(loss_op_id)

  if not target_loss_api:
    # Fallback if lookup failed (e.g. unknown op wired to this plugin)
    # Return unmodified node to avoid breaking valid code
    return node

  func_node = _create_dotted_name(target_loss_api)

  # Ensure standard args are comma separated properly after deletion
  # LibCST nodes are immutable; deletion shifts indices.
  # We must ensure the last arg doesn't have a trailing comma inside the wrapper call if strict.
  # But inside a wrapper call it's fine.

  inner_call = node.with_changes(func=func_node, args=args)

  # 4. Apply Wrapper
  wrapper_api = None

  if reduction_mode == "mean":
    # Dynamic Lookup: Get the API for "Mean" in the target framework
    # Removed JAX hardcoding: Only rely on lookup
    wrapper_api = ctx.lookup_api("Mean")

  elif reduction_mode == "sum":
    # Dynamic Lookup: Get the API for "Sum"
    wrapper_api = ctx.lookup_api("Sum")

  elif reduction_mode == "none":
    # No wrapper needed
    return inner_call

  if not wrapper_api:
    # Fallback if reduction requested but API lookup failed
    # Return unwrapped call to preserve functionality as best effort
    return inner_call

  wrapper_func = _create_dotted_name(wrapper_api)

  wrapper_call = cst.Call(func=wrapper_func, args=[cst.Arg(value=inner_call)])

  return wrapper_call
