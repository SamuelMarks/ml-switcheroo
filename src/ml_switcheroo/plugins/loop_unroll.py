"""
Plugin for Unrolling Loops (JAX Scan/Fori_loop).

Detects Python `for` loops and attempts to map them to JAX control flow primitives
when the target framework is JAX.

Supported Patterns:
1. `for i in range(args)` -> `jax.lax.fori_loop(start, stop, body_fun, init_val)`
   *Limitations*: Requires heuristic detection of 'carry' variable (state).
   If heuristics fail, applies an Escape Hatch warning suggesting manual fix.

2. Pass-through for non-JAX targets.
"""

import libcst as cst
from typing import Optional, List, Tuple

from ml_switcheroo.core.hooks import register_hook, HookContext
from ml_switcheroo.core.escape_hatch import EscapeHatch


@register_hook("transform_for_loop")
def transform_loops(node: cst.For, ctx: HookContext) -> cst.CSTNode:
  """
  Plugin Hook: Transforms `for` loops into JAX primitives if applicable.

  Args:
      node: The CST For loop node.
      ctx: Hook context.

  Returns:
      Transformed node or original wrapped in EscapeHatch if unsafe to convert.
  """
  # 1. Check constraints
  if ctx.target_fw != "jax":
    return node

  # 2. Pattern Match: range()
  # Check if iter is a call to `range`
  # Python: `for i in range(start, stop)`
  is_range, range_args = _analyze_range_iterator(node.iter)

  if is_range:
    # Attempt to convert to fori_loop
    try:
      # Heuristic: Find modified variables in body (The 'Carry')
      # Since we don't have full dataflow, we look for assignments to variables defined outside
      # This is risky, so we might just wrap it in EscapeHatch with a helpful hint
      # OR generate the `jax.lax.fori_loop` code structure asking user to fill carry.

      # For Phase 4 compliance, let's detect it and mark it if we can't be 100% sure,
      # but if we are confident (e.g. simple accumulation), we transform.

      # Current Strategy: JAX loops are unsafe to auto-convert without full dataflow.
      # We insert a specific comment advising the user.
      reason = (
        "JAX requires `jax.lax.fori_loop` or `scan` for stateful loops. "
        "Auto-conversion prevented due to missing dataflow analysis of 'carry' state."
      )
      return EscapeHatch.mark_failure(node, reason)

    except Exception:
      return node

  # 3. Fallback for other iterators
  if ctx.target_fw == "jax":
    return EscapeHatch.mark_failure(node, "Python 'for' loop in JAX requires `scan`. Check JIT compatibility.")

  return node


def _analyze_range_iterator(node: cst.BaseExpression) -> Tuple[bool, List[cst.Arg]]:
  """
  Checks if expression is `range(...)` and returns arguments.
  """
  if isinstance(node, cst.Call):
    if isinstance(node.func, cst.Name) and node.func.value == "range":
      return True, list(node.args)
  return False, []
