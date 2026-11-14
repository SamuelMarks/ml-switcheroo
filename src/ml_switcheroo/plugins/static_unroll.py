"""Plugin for Statically Unrolling Loops.

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
    5.  **Output**: Returns a ``cst.FlattenSentinel[cst.BaseStatement]`` containing the list of statements.
"""

import libcst as cst

from ml_switcheroo.core.hooks import HookContext, register_hook
from ml_switcheroo.exceptions import PreambleInjectionError


class LoopVarReplacer(cst.CSTTransformer):
  """Support visitor to replace loop variable instances with a constant integer."""

  def __init__(self, var_name: str, value: int) -> None:
    """Initialize the replacer.

    Args:
        var_name: The identifier string of the loop variable.
        value: The current iteration index.

    """
    self.var_name = var_name
    self.value = value

  def leave_Name(self, original_node: cst.Name, updated_node: cst.Name) -> cst.BaseExpression:
    """Replace occurrences of variables matching `var_name` with `Integer(value)`.

    Args:
        original_node: The original CST Name node before visiting children.
        updated_node: The updated CST Name node after visiting children.

    Returns:
        The replaced Integer node if the variable name matches `var_name`,
        otherwise the updated_node itself.
    """
    if original_node.value == self.var_name:
      return cst.Integer(str(self.value))
    return updated_node


def _extract_static_range_limit(iter_node: cst.BaseExpression) -> int | None:
  """Extract bounded static integer limit ``N`` from ``range(N)`` if ``N <= 16``.

  Args:
      iter_node: ``For.iter`` expression node.

  Returns:
      Static range bound ``N`` when ``0 <= N <= 16``, otherwise ``None``.
  """
  if not isinstance(iter_node, cst.Call):
    return None
  func_name = iter_node.func.value if isinstance(iter_node.func, cst.Name) else ""
  if func_name != "range" or len(iter_node.args) != 1 or not isinstance(iter_node.args[0].value, cst.Integer):
    return None
  try:
    limit = int(iter_node.args[0].value.value)
    return limit if limit <= 16 else None
  except ValueError:
    return None


def _attach_iteration_leading_lines(
  stmt: cst.BaseStatement,
  node: cst.For,
  header_comment: cst.Comment | None,
  *,
  iter_idx: int,
  limit: int,
  on_device: bool,
) -> cst.BaseStatement:
  """Prepend preserved loop comments and optional on-device telemetry markers to the first statement of an iteration.

  Args:
      stmt: First unrolled statement in iteration ``iter_idx``.
      node: Original ``cst.For`` loop node.
      header_comment: Optional trailing comment from the ``for ...:`` header line.
      iter_idx: Current zero-based unroll iteration index.
      limit: Total unrolled iteration count.
      on_device: Whether on-device iteration comments are enabled.

  Returns:
      Updated ``cst.BaseStatement`` carrying combined leading lines.
  """
  extra_leading: list[cst.EmptyLine] = []
  if iter_idx == 0:
    extra_leading.extend(node.leading_lines)
    if header_comment is not None:
      extra_leading.append(cst.EmptyLine(comment=header_comment, newline=cst.Newline()))
  if on_device:
    extra_leading.append(
      cst.EmptyLine(comment=cst.Comment(f"# [on-device unrolled iter={iter_idx}/{limit}]"), newline=cst.Newline())
    )
  existing_leading = (
    list(stmt.leading_lines) if isinstance(stmt, (cst.SimpleStatementLine, cst.BaseCompoundStatement)) else []
  )
  return stmt.with_changes(leading_lines=[*extra_leading, *existing_leading])


@register_hook("transform_for_loop_static")
def unroll_static_loops(node: cst.For, ctx: HookContext) -> cst.For | cst.FlattenSentinel[cst.BaseStatement]:
  """Transform Unrolls loops with static ranges.

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
      Union[cst.For, cst.FlattenSentinel[cst.BaseStatement]]:
          - ``FlattenSentinel`` containing unrolled statements if successful.
          - Original ``node`` if the loop is dynamic or too large.

  Raises:
      PreambleInjectionError: If loop statement expansion yields a non-statement CST node.
  """
  limit = _extract_static_range_limit(node.iter)
  if limit is None or not isinstance(node.target, cst.Name) or not isinstance(node.body, cst.IndentedBlock):
    return node

  loop_var = node.target.value
  original_statements = node.body.body
  header_comment = node.body.header.comment
  try:
    on_device = bool(ctx._runtime_config.enable_on_device_execution)
  except AttributeError:
    on_device = False

  unrolled_stmts: list[cst.BaseStatement] = []
  for i in range(limit):
    replacer = LoopVarReplacer(loop_var, i)
    for stmt_idx, stmt in enumerate(original_statements):
      new_stmt = stmt.visit(replacer)
      if not isinstance(new_stmt, cst.BaseStatement):
        raise PreambleInjectionError(f"Unrolled loop body expected BaseStatement, got {type(new_stmt).__name__}.")
      if stmt_idx == 0:
        new_stmt = _attach_iteration_leading_lines(
          new_stmt, node, header_comment, iter_idx=i, limit=limit, on_device=on_device
        )
      unrolled_stmts.append(new_stmt)

  return cst.FlattenSentinel[cst.BaseStatement](unrolled_stmts)
