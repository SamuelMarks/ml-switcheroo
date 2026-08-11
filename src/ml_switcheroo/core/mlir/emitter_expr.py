"""MLIR Emitter Expression Mixin.

This module provides the `MlirEmitterExprMixin` class, which handles the
translation of standard Python expressions (via LibCST AST representation)
into MLIR operations and values.
"""

import libcst as cst
from typing import Tuple, List, TYPE_CHECKING, Any
from ml_switcheroo.core.mlir.cst import ValueNode, OperationNode, AttributeNode


class MlirEmitterExprMixin:
  """Mixin class for emitting MLIR operations from LibCST expressions.

  This mixin provides capabilities to process, lower, and compile various
  Python expression structures—including name accesses, call operations,
  binary expressions, and numeric constants—into an equivalent list of
  supporting MLIR operations and their final evaluated values.
  """

  if TYPE_CHECKING:
    ctx: Any

    def _flatten_attr(self, attr: Any) -> Any:
      """Flattens a Name or Attribute chain into a dotted string.

      Args:
          self: The mixin instance.
          attr: The LibCST node representing the attribute or name to flatten.

      Returns:
          A dotted string representation of the attribute chain (e.g., "self.layer")
          or None if it cannot be flattened.
      """
      return None

    def _get_binop_str(self, op: Any) -> str:
      """Maps a LibCST binary operator node to its corresponding string identifier.

      Args:
          self: The mixin instance.
          op: The LibCST binary operator node (e.g., cst.Add, cst.Multiply).

      Returns:
          The string identifier (e.g., "add", "mul", "matmul") for the operator.
      """
      return ""

  def _emit_expression(self, expr: cst.BaseExpression) -> Tuple[ValueNode, List[OperationNode]]:
    """Recursively converts an expression into a value node and a list of supporting operations.

    Handles the translation of:
    - Variable references (Names)
    - Function and method calls (including keyword arguments)
    - Binary operations
    - Numeric constants (Integers and Floats)

    Args:
        expr: The LibCST base expression node to be processed.

    Returns:
        A tuple containing:
            - The resulting ValueNode representing the evaluated expression.
            - A list of OperationNode objects generated to evaluate the expression.
    """
    ops = []  # type: ignore
    if isinstance(expr, cst.Name):
      val = self.ctx.lookup(expr.value)
      if not val:
        val = ValueNode(name=f"@{expr.value}")
      return val, ops
    elif isinstance(expr, cst.Call):
      operands = []
      arg_keywords = []  # new feature

      # Process arguments
      for arg in expr.args:
        v, o = self._emit_expression(arg.value)
        ops.extend(o)
        operands.append(v)

        # Capture keyword if present
        kw = ""
        if arg.keyword:
          kw = arg.keyword.value
        arg_keywords.append(kw)

      flat_name = self._flatten_attr(expr.func)
      root_var = flat_name.split(".")[0] if flat_name else ""
      is_static_op = False
      if flat_name and not self.ctx.lookup(root_var):
        is_static_op = True

      common_attrs = []
      # Pack keywords into attribute if any are non-empty
      if any(arg_keywords):
        # AttributeNode needs a list of strings formatted for the printer
        # e.g. ["k=val", ""] -> we just need to store the keys.
        # "arg_keywords" = ["a", "", "b"]
        # We store as list of quoted strings
        kw_vals = [f'"{k}"' for k in arg_keywords]
        common_attrs.append(AttributeNode(name="arg_keywords", value=kw_vals))

      if is_static_op:
        result = self.ctx.allocate_ssa()
        attrs = [AttributeNode(name="type", value=f'"{flat_name}"')] + common_attrs
        op = OperationNode(
          name="sw.op",
          results=[result],
          operands=operands,
          attributes=attrs,
        )
        ops.append(op)
        return result, ops

      if isinstance(expr.func, cst.Attribute):
        obj, o_ops = self._emit_expression(expr.func.value)
        ops.extend(o_ops)
        attr_val = self.ctx.allocate_ssa()
        get_op = OperationNode(
          name="sw.getattr",
          results=[attr_val],
          operands=[obj],
          attributes=[AttributeNode(name="name", value=f'"{expr.func.attr.value}"')],
        )
        ops.append(get_op)
        res_val = self.ctx.allocate_ssa()
        # Attach keywords
        call_op = OperationNode(
          name="sw.call",
          results=[res_val],
          operands=[attr_val] + operands,
          attributes=common_attrs,
        )
        ops.append(call_op)
        return res_val, ops

      if isinstance(expr.func, cst.Name):
        func_val, f_ops = self._emit_expression(expr.func)
        ops.extend(f_ops)
        result = self.ctx.allocate_ssa()
        call_op = OperationNode(name="sw.call", results=[result], operands=[func_val] + operands, attributes=common_attrs)
        ops.append(call_op)
        return result, ops

    elif isinstance(expr, cst.BinaryOperation):
      lhs_val, l_ops = self._emit_expression(expr.left)
      rhs_val, r_ops = self._emit_expression(expr.right)
      ops.extend(l_ops)
      ops.extend(r_ops)

      op_str = self._get_binop_str(expr.operator)
      res_val = self.ctx.allocate_ssa()
      op = OperationNode(
        name="sw.op",
        results=[res_val],
        operands=[lhs_val, rhs_val],
        attributes=[AttributeNode(name="type", value=f'"binop.{op_str}"')],
      )
      ops.append(op)
      return res_val, ops

    elif isinstance(expr, (cst.Integer, cst.Float)):
      val = self.ctx.allocate_ssa(prefix="%cst")
      op = OperationNode(
        name="sw.constant", results=[val], attributes=[AttributeNode(name="value", value=getattr(expr, "value", "0"))]
      )
      ops.append(op)
      return val, ops

    return ValueNode(name="%error"), []

  def _annotation_to_string(self, node: cst.CSTNode) -> str:
    """Flattens a type annotation node into its corresponding string representation.

    Args:
        node: The LibCST node representing the type annotation.

    Returns:
        A string representation of the type annotation (e.g., "int", "Any").
    """
    if isinstance(node, cst.Name):
      return node.value
    elif isinstance(node, cst.Attribute):
      return f"{self._annotation_to_string(node.value)}.{node.attr.value}"
    return "Any"
