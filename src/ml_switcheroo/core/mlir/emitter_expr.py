"""MLIR Expression Emitter Delegate.

Provides ``MlirExprEmitter``, a single-responsibility delegate translating
Python LibCST expressions (names, calls, binary operations, constants, and
type annotations) into MLIR ``ValueNode`` and ``OperationNode`` sequences.
"""

import typing

import libcst as cst

from ml_switcheroo.core.mlir.cst import AttributeNode, OperationNode, ValueNode
from ml_switcheroo.core.mlir.emitter_decl import SSAContext


class MlirExprEmitterHost(typing.Protocol):
  """Structural protocol required by ``MlirExprEmitter``.

  Attributes:
      ctx: Active ``SSAContext`` instance.
  """

  ctx: SSAContext

  def _emit_expression(self, expr: cst.BaseExpression) -> tuple[ValueNode, list[OperationNode]]:
    """Emit ``expr`` (allowing polymorphic subclass dispatch such as ``StableHloEmitter``).

    Args:
        expr: Sub-expression to emit.

    Returns:
        Tuple of ``(ValueNode, list[OperationNode])``.
    """
    ...

  def _flatten_attr(self, node: cst.CSTNode) -> str | None:
    """Flatten a ``Name`` or ``Attribute`` chain into a dotted string.

    Args:
        node: CST node to flatten.

    Returns:
        Dotted string or ``None``.
    """
    ...

  def _get_binop_str(self, operator: cst.BaseBinaryOp) -> str:
    """Map a LibCST binary operator node to its string identifier.

    Args:
        operator: Binary operator CST node.

    Returns:
        Operator code string (e.g. ``'add'``, ``'matmul'``).
    """
    ...


class MlirExprEmitter:
  """Single-responsibility delegate lowering LibCST expressions into MLIR operations.

  Attributes:
      host: Bound ``MlirExprEmitterHost`` instance.
  """

  def __init__(self, host: MlirExprEmitterHost) -> None:
    """Initialize ``MlirExprEmitter`` bound to ``host``.

    Args:
        host: Bound ``MlirExprEmitterHost`` instance.
    """
    self.host: MlirExprEmitterHost = host

  def emit_expression(self, expr: cst.BaseExpression) -> tuple[ValueNode, list[OperationNode]]:
    """Recursively convert ``expr`` into a ``ValueNode`` and supporting ``OperationNode`` list.

    Args:
        expr: LibCST expression node to lower.

    Returns:
        Tuple of ``(result_value, emitted_operations)``.
    """
    ops: list[OperationNode] = []
    if isinstance(expr, cst.Name):
      val = self.host.ctx.lookup(expr.value)
      if not val:
        val = ValueNode(name=f"@{expr.value}")
      return val, ops

    if isinstance(expr, cst.Call):
      operands: list[ValueNode] = []
      arg_keywords: list[str] = []

      for arg in expr.args:
        v, o = self.host._emit_expression(arg.value)
        ops.extend(o)
        operands.append(v)
        arg_keywords.append(arg.keyword.value if arg.keyword else "")

      flat_name = self.host._flatten_attr(expr.func)
      root_var = flat_name.split(".")[0] if flat_name else ""
      is_static_op = bool(flat_name and not self.host.ctx.lookup(root_var))

      common_attrs: list[AttributeNode] = []
      if any(arg_keywords):
        kw_vals = [f'"{k}"' for k in arg_keywords]
        common_attrs.append(AttributeNode(name="arg_keywords", value=kw_vals))

      if is_static_op:
        result = self.host.ctx.allocate_ssa()
        attrs: list[AttributeNode] = [AttributeNode(name="type", value=f'"{flat_name}"'), *common_attrs]
        op = OperationNode(
          name="sw.op",
          results=[result],
          operands=operands,
          attributes=attrs,
        )
        ops.append(op)
        return result, ops

      if isinstance(expr.func, cst.Attribute):
        obj, o_ops = self.host._emit_expression(expr.func.value)
        ops.extend(o_ops)
        attr_val = self.host.ctx.allocate_ssa()
        get_op = OperationNode(
          name="sw.getattr",
          results=[attr_val],
          operands=[obj],
          attributes=[AttributeNode(name="name", value=f'"{expr.func.attr.value}"')],
        )
        ops.append(get_op)
        res_val = self.host.ctx.allocate_ssa()
        call_op = OperationNode(
          name="sw.call",
          results=[res_val],
          operands=[attr_val, *operands],
          attributes=common_attrs,
        )
        ops.append(call_op)
        return res_val, ops

      if isinstance(expr.func, cst.Name):
        func_val, f_ops = self.host._emit_expression(expr.func)
        ops.extend(f_ops)
        result = self.host.ctx.allocate_ssa()
        call_op = OperationNode(name="sw.call", results=[result], operands=[func_val, *operands], attributes=common_attrs)
        ops.append(call_op)
        return result, ops

    elif isinstance(expr, cst.BinaryOperation):
      lhs_val, l_ops = self.host._emit_expression(expr.left)
      rhs_val, r_ops = self.host._emit_expression(expr.right)
      ops.extend(l_ops)
      ops.extend(r_ops)

      op_str = self.host._get_binop_str(expr.operator)
      res_val = self.host.ctx.allocate_ssa()
      op = OperationNode(
        name="sw.op",
        results=[res_val],
        operands=[lhs_val, rhs_val],
        attributes=[AttributeNode(name="type", value=f'"binop.{op_str}"')],
      )
      ops.append(op)
      return res_val, ops

    elif isinstance(expr, (cst.Integer, cst.Float)):
      val = self.host.ctx.allocate_ssa(prefix="%cst")
      op = OperationNode(name="sw.constant", results=[val], attributes=[AttributeNode(name="value", value=expr.value)])
      ops.append(op)
      return val, ops

    return ValueNode(name="%error"), []

  @staticmethod
  def annotation_to_string(node: cst.CSTNode) -> str:
    """Flatten a CST type annotation node into its dotted string representation.

    Args:
        node: LibCST node representing the type annotation.

    Returns:
        Dotted string representation (e.g. ``'int'``, ``'torch.Tensor'``) or ``'Any'``.
    """
    if isinstance(node, cst.Name):
      return node.value
    if isinstance(node, cst.Attribute):
      return f"{MlirExprEmitter.annotation_to_string(node.value)}.{node.attr.value}"
    return "Any"


__all__ = ["MlirExprEmitter", "MlirExprEmitterHost"]
