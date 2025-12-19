"""
Plugin for Statically Unrolling Loops.

JAX/XLA requires functional loops (`scan`, `fori_loop`) which are hard to
automatically generate from imperative Python `for` loops due to state containment rules.

However, many neural network definitions use loops over *fixed constants*
(e.g., `for i in range(3): layer(x)`).

This plugin:
1. Detects `for i in range(N)` where N is a static integer literal.
2. Unrolls the loop body N times.
3. Replaces the loop variable (`i`) with the literal integer `0, 1, ...`.
4. Returns a flattened list of statements to replace the loop block.
"""

import libcst as cst
from typing import Union, List

from ml_switcheroo.core.hooks import register_hook, HookContext


class LoopVarReplacer(cst.CSTTransformer):
  """
  Helper visitor to replace loop variable 'i' with a constant integer '0'.
  """

  def __init__(self, var_name: str, value: int):
    self.var_name = var_name
    self.value = value

  def leave_Name(self, original_node: cst.Name, updated_node: cst.Name) -> cst.BaseExpression:
    if original_node.value == self.var_name:
      return cst.Integer(str(self.value))
    return updated_node


@register_hook("transform_for_loop_static")
def unroll_static_loops(node: cst.For, ctx: HookContext) -> Union[cst.For, cst.FlattenSentinel]:
  """
  Hook: Unrolls loops with static ranges.

  Triggers:
      Automatically invoked by `ControlFlowMixin` if registered as `transform_for_loop`.
      (Renaming internal registration key to override default behavior).

  Transformation:
      Input:
          for i in range(2):
              x = f(x, i)
      Output:
          x = f(x, 0)
          x = f(x, 1)

  Args:
      node: The For loop node.
      ctx: Hook Context.

  Returns:
      FlattenSentinel containing the unrolled statements, or original node.
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
          # Safety Cap: Don't unroll huge loops
          if limit <= 16:
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
