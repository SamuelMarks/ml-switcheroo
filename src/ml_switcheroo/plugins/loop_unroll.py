"""
Plugin for Unrolling Loops (Functional Control Flow Enforcement).

This module handles the structural transformation of Python control flow into
functional primitives required by XLA-based frameworks. Unlike PyTorch/TensorFlow Eager,
which support imperative Python loops, frameworks targeting XLA (like JAX) require
loops to be expressed as functional operators (`scan` or `fori_loop`) to be compiled.

The strategy is **Safety-First**:

1.  **Analysis**: Inspects `for` loops.
2.  **Trait Check**: Determines if the target framework requires functional control flow.
3.  **Handling**: Wraps imperative loops in an `EscapeHatch` warning rather than
    attempting unsafe auto-conversion (solving the "carry state" problem is often undecidable).
"""

import libcst as cst
from typing import Tuple, List, Union

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
  Plugin Hook: Transforms or Flags `for` loops for functional compliance.

  Triggered by the `ControlFlowMixin` when visiting `For` nodes.

  Strategy:

  - Check `ctx.plugin_traits.requires_functional_control_flow`.
  - If False: Pass through (Imperative loops are valid).
  - If True:
      - Analyze the iterator.
      - If `range()`: Warn that `jax.lax.fori_loop` (or equivalent) is required.
      - Else: Warn that `scan` is required.
      - Wrap in `EscapeHatch` to prevent compilation errors in the target.

  Args:
      node: The original CST For loop node.
      ctx: Hook context containing framework configuration traits.

  Returns:
      The transformed node or an EscapeHatch sentinel.
  """
  # 1. Check Target Constraints (Decoupled check)
  # We rely on the Adapter's declared capabilities rather than the framework name.
  if not ctx.plugin_traits.requires_functional_control_flow:
    return node

  # 2. Analyze Iterator Pattern
  # Python: `for i in range(start, stop)`
  is_range, _range_args = _analyze_range_iterator(node.iter)

  target_fw_label = ctx.target_fw.upper()

  if is_range:
    # Candidate for fori_loop
    # To auto-convert, we would need to:
    # 1. Identify 'carry' variables (variables mutated in the body).
    # 2. Define a body_fun(i, carry).
    # 3. Rewrite usage.
    # Without full dataflow analysis, this is unsafe.

    warn_msg = (
      f"{target_fw_label} requires explicit functional loops (e.g. `fori_loop`) for stateful logic. "
      "Auto-conversion prevented due to missing dataflow analysis of 'carry' state."
    )
    return EscapeHatch.mark_failure(node, warn_msg)

  # 3. Fallback for Generic Iterators (Lists, Tensors, Zips)
  # Candidate for scan
  warn_msg = (
    f"Python 'for' loop in {target_fw_label} requires structural rewrite (e.g. `scan`). "
    "Unrolling this loop may break if it depends on external state."
  )
  return EscapeHatch.mark_failure(node, warn_msg)
