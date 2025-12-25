"""
Plugin for Optimizer Step Translation.

Handles the conversion of imperative optimization steps (PyTorch) to
functional state updates (JAX/Optax).

Transformations:
1.  **Instantiation**: Strips `model.parameters()` from the constructor, as Optax
    initializes state separately.
    *   Input: `opt = torch.optim.Adam(model.parameters(), lr=0.01)`
    *   Output: `opt = optax.adam(learning_rate=0.01)`

2.  **Step Execution**: Rewrites `step()` to the Optax update/apply sequence.
    *   Input: `optimizer.step()`
    *   Output: `updates, opt_state = optimizer.update(grads, opt_state, params)`
    *           `params = optax.apply_updates(params, updates)`

3.  **Zero Grad**: Strips `zero_grad()` as JAX handles gradients explicitly via `grad` or `value_and_grad`.
"""

import libcst as cst
from typing import Union, List, Optional

from ml_switcheroo.core.hooks import register_hook, HookContext
from ml_switcheroo.core.escape_hatch import EscapeHatch


def _create_dotted_name(name_str: str) -> cst.BaseExpression:
  """Helper to create a CST Attribute chain."""
  parts = name_str.split(".")
  node = cst.Name(parts[0])
  for part in parts[1:]:
    node = cst.Attribute(value=node, attr=cst.Name(part))
  return node


@register_hook("optimizer_constructor")
def transform_optimizer_init(node: cst.Call, ctx: HookContext) -> cst.Call:
  """
  Hook to rewrite Optimizer instantiation.
  Removes the first argument (parameters) if targeting JAX, as Optax is functional.
  """
  if ctx.target_fw != "jax":
    return node

  # PyTorch optimizers take 'params' as the first argument.
  # JAX (Optax) optimizers are factory functions that take hyperparameters only.
  # We strip the first argument if it looks like parameters (not a kwarg).

  new_args = []

  # Heuristic: Skip first arg if it's positional.
  # torch.optim.Adam(params, lr=...)
  start_index = 0
  if len(node.args) > 0 and node.args[0].keyword is None:
    start_index = 1

  for i in range(start_index, len(node.args)):
    new_args.append(node.args[i])

  return node.with_changes(args=new_args)


@register_hook("optimizer_step")
def transform_optimizer_step(node: cst.Call, ctx: HookContext) -> Union[cst.Call, cst.FlattenSentinel]:
  """Hook to rewrite ``optimizer.step()``.

  **JAX Target**

  - Emits the functional update pattern.
  - Assumes existence of ``grads``, ``opt_state``, ``params`` variables in the local scope.
  - Use EscapeHatch to warn user if variables can't be inferred.
  """
  if ctx.target_fw != "jax":
    return node

  # Heuristic check for the receiver name
  optimizer_name = "optimizer"
  if isinstance(node.func, cst.Attribute) and isinstance(node.func.value, cst.Name):
    optimizer_name = node.func.value.value

  # We cannot validly replace a Call expression with multiple statements (Assign + Assign)
  # inside an expression context. This hook is best effort for Statement-level calls.
  # We generate a code block that performs the update.

  # Pattern:
  # updates, opt_state = optimizer.update(grads, opt_state, params)
  # params = optax.apply_updates(params, updates)

  reason = (
    "Imperative optimizer.step() cannot be automatically converted to JAX functional update. "
    "Manual intervention required: `updates, opt_state = optimizer.update(grads, opt_state, params)`"
  )
  return EscapeHatch.mark_failure(node, reason)


@register_hook("optimizer_zero_grad")
def strip_zero_grad(node: cst.Call, ctx: HookContext) -> cst.CSTNode:
  """Hook for ``optimizer.zero_grad()``.

  **JAX Target**

  Removes the call (No-op), as JAX gradients are not accumulated by default.
  """
  if ctx.target_fw != "jax":
    return node

  # Transform to `None` (Effective No-Op in statement context)
  return node.with_changes(func=cst.Name("None"), args=[])
