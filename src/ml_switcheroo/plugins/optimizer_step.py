"""
Plugin for Optimizer Step Translation.

Handles the conversion of imperative optimization steps (e.g., PyTorch) to
functional state updates (e.g., JAX/Optax).

This logic is **Wired-Only**: It executes blindly if the semantic map requests it.

Transformations:
1.  **Instantiation (`optimizer_constructor`)**:
    - Strips the first argument (commonly `model.parameters()` in Torch) because
      functional optimizers (Optax) are initialized stateless/factory-style.
    - Input: `opt = torch.optim.Adam(model.parameters(), lr=0.01)`
    - Output: `opt = optax.adam(lr=0.01)`

2.  **Step Execution (`optimizer_step`)**:
    - Flags `step()` calls as requiring manual intervention or functional rewrite.
    - Output: An `EscapeHatch` warning block suggesting the update pattern.

3.  **Zero Grad (`optimizer_zero_grad`)**:
    - Strips the call completely (No-Op), as functional gradients don't accumulate state.
"""

import libcst as cst
from typing import Union

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
  Removes the first argument (parameters) to support factory-pattern initialization.
  """
  # Heuristic: Skip first arg if it's positional (params).
  # torch.optim.Adam(params, lr=...) -> optax.adam(lr=...)

  new_args = []
  start_index = 0

  if len(node.args) > 0 and node.args[0].keyword is None:
    start_index = 1

  for i in range(start_index, len(node.args)):
    new_args.append(node.args[i])

  return node.with_changes(args=new_args)


@register_hook("optimizer_step")
def transform_optimizer_step(node: cst.Call, ctx: HookContext) -> Union[cst.Call, cst.FlattenSentinel]:
  """Hook to rewrite ``optimizer.step()``.

  Since `step()` logic implies side-effects on the optimizer state and parameters,
  which doesn't translate 1:1 to functional updates without knowing variable names
  (params, grads, opt_state), this hook emits a specialized Escape Hatch.
  """
  # Pattern:
  # updates, opt_state = optimizer.update(grads, opt_state, params)
  # params = optax.apply_updates(params, updates)

  reason = (
    f"Imperative `{_get_func_name(node)}` cannot be automatically converted to functional update. "
    "Manual intervention required (e.g. `updates, state = opt.update(grads, state)`)."
  )
  return EscapeHatch.mark_failure(node, reason)


@register_hook("optimizer_zero_grad")
def strip_zero_grad(node: cst.Call, ctx: HookContext) -> cst.CSTNode:
  """Hook for ``optimizer.zero_grad()``.

  Removes the call (No-op), as gradient accumulation is generally explicit
  in functional frameworks.
  """
  # Transform to `None` (Effective No-Op in statement context)
  return node.with_changes(func=cst.Name("None"), args=[])


def _get_func_name(node: cst.Call) -> str:
  if isinstance(node.func, cst.Attribute):
    return node.func.attr.value
  return "step"
