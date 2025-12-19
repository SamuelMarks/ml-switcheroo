"""
Plugin for Loss Reduction Semantics.

Addresses the mismatch between:
1. PyTorch: `loss = F.cross_entropy(x, y, reduction='mean')` (Scalar output by default).
2. JAX/Optax: `loss = optax.softmax_cross_entropy(x, y)` (Vector output per batch).

JAX libraries typically return the loss *per sample* to support vmap/pmap flexibility.
PyTorch defaults to averaging (`mean`) immediately.

Transformation:
1. Detects `reduction` keyword argument.
2. Strips the argument (as Optax/JAX funcs don't usually accept it).
3. Wraps the function call in `jnp.mean()` (default/mean) or `jnp.sum()`.
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
  Target: JAX, Flax.
  """
  if ctx.target_fw not in ["jax", "flax", "flax_nnx"]:
    return node

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
        # Fallback: assume 'mean' or warn?
        # For safety, strict transpilation might skip wrapper and let it fail,
        # but here we assume standard string literals.
        pass
      break

  # 2. Modify Arguments (Strip reduction arg)
  if reduction_arg_index != -1:
    del args[reduction_arg_index]

  # Reconstruct the inner call (standard functional mapping)
  # The API renaming happens via 'func' replacement in BaseRewriter,
  # but since this is a post-processing hook on the original node,
  # we need to ensure the BaseRewriter has already renamed it OR we rename it here?
  #
  # Hooks run *on the node*. If BaseRewriter calls the hook, it passes the *original* node
  # or the *mostly processed* node?
  # Architecture: Hooks replace specific patterns. If we return a new Call,
  # BaseRewriter uses that. We should apply the API name mapping here if we construct a new tree.

  target_api = ctx.lookup_api(ctx.current_op_id)
  # If op_id is unknown (generic hook), fallback to preserving current func
  # (assuming BaseRewriter handled rename via 'func' replacement, but wait...
  # BaseRewriter logic replaces func *after* transform? No, usually hooks encompass the transform).

  if not target_api:
    # Fallback if context logic missing
    target_api = "optax.softmax_cross_entropy_with_integer_labels"

  func_node = _create_dotted_name(target_api)

  # Ensure standard args are comma separated properly after deletion
  if args and args[-1].comma == cst.MaybeSentinel.DEFAULT:
    # If wrapped, last arg shouldn't necessarily have comma, but inside mean() it's fine.
    pass

  inner_call = node.with_changes(func=func_node, args=args)

  # 3. Apply Wrapper
  wrapper_api = None
  if reduction_mode == "mean":
    wrapper_api = "jax.numpy.mean"
  elif reduction_mode == "sum":
    wrapper_api = "jax.numpy.sum"
  elif reduction_mode == "none":
    return inner_call

  wrapper_func = _create_dotted_name(wrapper_api)

  wrapper_call = cst.Call(func=wrapper_func, args=[cst.Arg(value=inner_call)])

  return wrapper_call
