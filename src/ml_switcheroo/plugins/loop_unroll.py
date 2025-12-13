"""
Plugin for Unrolling Loops (JAX Scan/Fori_loop).

This module handles the structural transformation of Python control flow into
JAX-compatible primitives. Unlike PyTorch, which supports imperative Python loops,
JAX requires loops to be expressed as functional operators (`jax.lax.scan` or `jax.lax.fori_loop`)
to enable XLA compilation (JIT).

Logic:
1.  **Analysis**: Inspects `for` loops to determine if they iterate over a `range` (candidates for `fori_loop`)
    or an iterable (candidates for `scan`).
2.  **Safety Check**: Because automated loop conversion requires solving the "Carry State" problem
    (identifying which variables are mutated across iterations), this plugin currently defaults to a
    **Safety-First** strategy.
3.  **Transformation**: Instead of hallucinating broken JAX code, it wraps the loop in an `EscapeHatch`.
    This preserves the original logic while explicitly flagging it as a blocker for JIT compliance,
    guiding the user to manually refactor it into a functional pattern.
"""

import libcst as cst
from typing import Optional, Tuple, List, Union

from ml_switcheroo.core.hooks import register_hook, HookContext
from ml_switcheroo.core.escape_hatch import EscapeHatch


def _analyze_range_iterator(node: cst.BaseExpression) -> Tuple[bool, List[cst.Arg]]:
  """
  Determines if an expression is a call to `range(...)`.

  Args:
      node: The CST expression node of the iterator.

  Returns:
      Tuple (bool, args): True if it's a range call, plus the arguments passed.
  """
  if isinstance(node, cst.Call):
    if isinstance(node.func, cst.Name) and node.func.value == "range":
      return True, list(node.args)
  return False, []


@register_hook("transform_for_loop")
def transform_loops(node: cst.For, ctx: HookContext) -> Union[cst.For, cst.FlattenSentinel]:
  """
  Plugin Hook: Transforms `for` loops for JAX compliance.

  Triggered by the `ControlFlowMixin` when visiting `For` nodes.

  Strategy:
      - If Target != JAX: Pass through (Python loops are valid in Torch/TF Eager).
      - If Target == JAX:
          - Analyze the iterator.
          - If `range()`: Flag as candidates for `jax.lax.fori_loop`.
          - Else: Flag as candidates for `jax.lax.scan`.
          - Wrap in `EscapeHatch` to prevent compilation errors in JAX/XLA.

  Args:
      node: The original CST For loop node.
      ctx: Hook context containing framework targets.

  Returns:
      The transformed node or an EscapeHatch sentinel.
  """
  # 1. Check Target Constraints
  # Transformation is only required for JAX; other frameworks support imperative loops.
  if ctx.target_fw != "jax":
    return node

  # 2. Analyze Iterator Pattern
  # Python: `for i in range(start, stop)`
  is_range, _range_args = _analyze_range_iterator(node.iter)

  if is_range:
    # Candidate for jax.lax.fori_loop
    # To auto-convert, we would need to:
    # 1. Identify 'carry' variables (variables mutated in the body).
    # 2. Define a body_fun(i, carry).
    # 3. Rewrite usage.
    # Without full dataflow analysis, this is unsafe.

    warn_msg = (
      "JAX requires `jax.lax.fori_loop` or `scan` for stateful loops. "
      "Auto-conversion prevented due to missing dataflow analysis of 'carry' state."
    )
    return EscapeHatch.mark_failure(node, warn_msg)

  # 3. Fallback for Generic Iterators (Lists, Tensors, Zips)
  # Candidate for jax.lax.scan
  warn_msg = (
    "Python 'for' loop in JAX requires `scan`. Check JIT compatibility. "
    "Unrolling this loop may break if it depends on external state."
  )
  return EscapeHatch.mark_failure(node, warn_msg)
