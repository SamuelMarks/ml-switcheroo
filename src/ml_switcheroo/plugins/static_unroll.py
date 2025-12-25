"""
Plugin for Statically Unrolling Loops.

JAX and XLA-based frameworks generally require functional loops (``scan``, ``fori_loop``)
which are difficult to automatically generate from imperative Python ``for`` loops
due to complex variable scoping and state containment rules.

However, many neural network definitions use loops over **fixed constants**
(e.g., ``for i in range(3): layer(x)``). Unrolling these provides a valid,
optimizable graph structure without requiring complex functional rewrite logic.

Usage:
    This plugin registers the hook ``transform_for_loop_static``. It is invoked
    by the ``ControlFlowMixin`` prior to general loop safety scanners.

Process:
    1.  **Analysis**: Detects ``for i in range(N)`` where N is a static integer literal.
    2.  **Safety**: Checks if N is within a reasonable limit to prevent code explosion.
    3.  **Expansion**: Duplicates the loop body N times.
    4.  **Substitution**: Replaces usages of the loop variable (``i``) with the
        literal integer for that iteration (``0``, ``1``, etc.).
    5.  **Output**: Returns a ``cst.FlattenSentinel`` containing the list of statements.
"""

import libcst as cst
from typing import Union

from ml_switcheroo.core.hooks import register_hook, HookContext


class LoopVarReplacer(cst.CSTTransformer):
  """
  Helper visitor to replace loop variable instances with a constant integer.
  """

  def __init__(self, var_name: str, value: int):
    """
    Initializes the replacer.

    Args:
        var_name: The identifier string of the loop variable.
        value: The current iteration index.
    """
    self.var_name = var_name
    self.value = value

  def leave_Name(self, original_node: cst.Name, updated_node: cst.Name) -> cst.BaseExpression:
    """
    Replace occurences of variables matching `var_name` with `Integer(value)`.
    """
    if original_node.value == self.var_name:
      return cst.Integer(str(self.value))
    return updated_node


@register_hook("transform_for_loop_static")
def unroll_static_loops(node: cst.For, ctx: HookContext) -> Union[cst.For, cst.FlattenSentinel]:
  """
  Hook: Unrolls loops with static ranges.

  Triggers:
      Invoked by ``ControlFlowMixin`` via the ``transform_for_loop_static`` key.

  Transformation:
      Input:
          for i in range(2):
              x = f(x, i)
      Output:
          x = f(x, 0)
          x = f(x, 1)

  Args:
      node (cst.For): The original For loop node.
      ctx (HookContext): The execution context (unused in this logic but required by protocol).

  Returns:
      Union[cst.For, cst.FlattenSentinel]:
          - ``FlattenSentinel`` containing unrolled statements if successful.
          - Original ``node`` if the loop is dynamic or too large.
  """
  # 1. Analyze Iterator: range(N)
  # Must be a Call to 'range' with 1 arg which is Integer literal.
  is_static_range = False
  limit = 0

  if isinstance(node.iter, cst.Call):
    func_name = node.iter.func.value if isinstance(node.iter.func, cst.Name) else ""
    if func_name == "range":
      args = node.iter.args
      if len(args) == 1 and isinstance(args[0].value, cst.Integer):
        try:
          limit = int(args[0].value.value)
          # Safety Cap: Don't unroll huge loops logic
          if limit <= 16:  # Arbitrary small constant for safety
            is_static_range = True
        except ValueError:
          pass

  if not is_static_range:
    # Fallback to standard handler or return node
    return node

  # 2. Extract Loop Variable Name
  # loop target must be a simple Name (e.g. 'i'), not Tuple unpacking (e.g. 'i, j')
  if not isinstance(node.target, cst.Name):
    return node

  loop_var = node.target.value

  # 3. Unroll Generation
  unrolled_stmts = []

  body_block = node.body
  # Ensure it's an indented block to iterate statements
  if not isinstance(body_block, cst.IndentedBlock):
    return node

  original_statements = body_block.body

  for i in range(limit):
    # Create a replacer for this iteration index
    replacer = LoopVarReplacer(loop_var, i)

    # Clone and Visit each statement
    # Note: LibCST nodes are immutable, visiting returns new nodes
    for stmt in original_statements:
      new_stmt = stmt.visit(replacer)
      unrolled_stmts.append(new_stmt)

  # 4. Return Flattened List
  return cst.FlattenSentinel(unrolled_stmts)
