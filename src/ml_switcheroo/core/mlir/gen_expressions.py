"""
Expression Generation Mixin for MLIR->Python.

Handles converting operation nodes (`sw.op`, `sw.call`) into CST Expressions.
"""

from typing import List
import libcst as cst

from ml_switcheroo.core.mlir.nodes import OperationNode
from ml_switcheroo.core.mlir.gen_base import BaseGeneratorMixin


class ExpressionGeneratorMixin(BaseGeneratorMixin):
  """
  Mixin for generating LibCST Expressions from MLIR Operations.
  Assumes `self._resolve_operand` is available on the host class.
  """

  def _resolve_operand(self, ssa_name: str) -> cst.BaseExpression:
    """
    Abstract placeholder.
    Must be implemented by the main Generator class to resolve variables.
    """
    raise NotImplementedError

  def _expr_sw_constant(self, op: OperationNode) -> cst.BaseExpression:
    """Generates constant literal expression."""
    val_str = self._get_attr(op, "value") or "0"
    try:
      return cst.parse_expression(val_str)
    except Exception:
      return cst.Name(val_str)

  def _expr_sw_getattr(self, op: OperationNode) -> cst.BaseExpression:
    """Generates attribute access (e.g. self.layer)."""
    if not op.operands:
      return cst.Name("error")
    obj_expr = self._resolve_operand(op.operands[0].name)
    attr_name = (self._get_attr(op, "name") or "unknown").strip('"')
    return cst.Attribute(value=obj_expr, attr=cst.Name(attr_name))

  def _expr_sw_call(self, op: OperationNode) -> cst.BaseExpression:
    """Generates function call expression."""
    # Operands: [func, arg0, arg1...]
    if not op.operands:
      return cst.Call(func=cst.Name("unknown"))

    func_expr = self._resolve_operand(op.operands[0].name)
    args = []
    for op_val in op.operands[1:]:
      arg_expr = self._resolve_operand(op_val.name)
      args.append(cst.Arg(value=arg_expr))

    return cst.Call(func=func_expr, args=args)

  def _expr_sw_op(self, op: OperationNode) -> cst.BaseExpression:
    """
    Generates generic operation call (e.g. torch.add).
    Handles specialized binary operation mapping via `binop.` types.
    """
    type_name = self._get_attr(op, "type") or '"unknown"'
    if "binop." in type_name:
      return self._expr_binop(op, type_name)

    dotted_path = type_name.strip('"')
    func_node = self._create_dotted_name(dotted_path)
    args = []
    for op_val in op.operands:
      arg_expr = self._resolve_operand(op_val.name)
      args.append(cst.Arg(value=arg_expr))

    return cst.Call(func=func_node, args=args)

  def _expr_binop(self, op: OperationNode, type_attr: str) -> cst.BaseExpression:
    """Generates binary operation expression (e.g. a + b)."""
    op_code = type_attr.strip('"').replace("binop.", "")
    if len(op.operands) < 2:
      return cst.Name("error_binop")

    lhs_expr = self._resolve_operand(op.operands[0].name)
    rhs_expr = self._resolve_operand(op.operands[1].name)

    op_map = {
      "add": cst.Add(),
      "sub": cst.Subtract(),
      "mul": cst.Multiply(),
      "div": cst.Divide(),
      "floordiv": cst.FloorDivide(),
      "mod": cst.Modulo(),
      "pow": cst.Power(),
      "matmul": cst.MatrixMultiply(),
      "lshift": cst.LeftShift(),
      "rshift": cst.RightShift(),
      "and": cst.BitAnd(),
      "or": cst.BitOr(),
      "xor": cst.BitXor(),
    }
    return cst.BinaryOperation(left=lhs_expr, operator=op_map.get(op_code, cst.Add()), right=rhs_expr)
