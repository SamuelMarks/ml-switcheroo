"""
Plugin for Gradient Clipping.

Addresses the mismatch between:
1. PyTorch: `torch.nn.utils.clip_grad_norm_(parameters, max_norm)` (In-place, returns norm).
2. JAX/Optax: `optax.clip_by_global_norm(max_norm).update(grads, state)` (Functional, returns updates).

Decoupling Update:
Checks `traits.requires_functional_state` logic.
"""

import libcst as cst
from ml_switcheroo.core.hooks import register_hook, HookContext


@register_hook("grad_clipper")
def transform_grad_clipping(node: cst.Call, ctx: HookContext) -> cst.CSTNode:
  """
  Hook: Transforms imperative clipping to Optax functional clipping.

  Trigger: `clip_grad_norm_` operation.
  Target: Frameworks with `requires_functional_state=True` (e.g. JAX, Flax).

  Transformation:
      Input:  `clip_grad_norm_(grads, 1.0)`
      Output: `optax.clip_by_global_norm(1.0).update(grads, None)[0]`

  Args:
      node (cst.Call): The source call.
      ctx (HookContext): Execution context with traits.

  Returns:
      The transformed node or original if trait not met.
  """
  # 0. Capability Check
  if not ctx.plugin_traits.requires_functional_state:
    return node

  args = list(node.args)
  if len(args) < 2:
    return node

  # Torch Signature: (parameters, max_norm, norm_type=2)
  # Arg 0: Gradients (parameters in Torch, but grads in JAX training loop)
  grads_node = args[0].value

  # Arg 1: Max Norm
  max_norm_node = args[1].value

  # We ignore Arg 2 (norm_type) as Optax defaults to L2 global norm usually.

  # Construct: optax.clip_by_global_norm(max_norm)
  clip_fn = cst.Call(
    func=cst.Attribute(value=cst.Name("optax"), attr=cst.Name("clip_by_global_norm")), args=[cst.Arg(value=max_norm_node)]
  )

  # Construct: .update(grads, None)
  # Note: clip_by_global_norm returns a GradientTransformation (init, update).

  update_attr = cst.Attribute(value=clip_fn, attr=cst.Name("update"))

  update_call = cst.Call(
    func=update_attr,
    args=[
      cst.Arg(value=grads_node, comma=cst.Comma(whitespace_after=cst.SimpleWhitespace(" "))),
      cst.Arg(value=cst.Name("None")),  # State is None for stateless clipping
    ],
  )

  # Construct: [0] to get the updates (gradients) and discard the empty state
  result_node = cst.Subscript(value=update_call, slice=[cst.SubscriptElement(slice=cst.Index(value=cst.Integer("0")))])

  return result_node
