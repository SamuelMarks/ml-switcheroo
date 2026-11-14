"""Expression Generation Delegate for MLIR -> Python.

Provides ``MlirExpressionGenerator``, a single-responsibility delegate translating
MLIR expression operations (``sw.op``, ``sw.call``, ``sw.getattr``, ``sw.constant``)
into LibCST expressions.
"""

import ast
import typing

import libcst as cst

from ml_switcheroo.core.mlir.cst import OperationNode
from ml_switcheroo.core.mlir.gen_base import MlirGenHelpers


class MlirExprGeneratorHost(typing.Protocol):
  """Structural protocol required by ``MlirExpressionGenerator``."""

  def _resolve_operand(self, ssa_name: str) -> cst.BaseExpression:
    """Resolve an SSA value identifier to its LibCST expression.

    Args:
        ssa_name: SSA identifier string.

    Returns:
        Resolved ``cst.BaseExpression``.
    """
    ...


class MlirExpressionGenerator:
  """Single-responsibility delegate generating LibCST expressions from MLIR operations.

  Attributes:
      host: Bound ``MlirExprGeneratorHost`` supplying SSA operand resolution.
      helpers: ``MlirGenHelpers`` utility instance.
  """

  def __init__(self, host: MlirExprGeneratorHost, helpers: MlirGenHelpers | None = None) -> None:
    """Initialize ``MlirExpressionGenerator`` bound to ``host``.

    Args:
        host: Bound ``MlirExprGeneratorHost`` instance.
        helpers: Optional ``MlirGenHelpers`` instance.
    """
    self.host: MlirExprGeneratorHost = host
    self.helpers: MlirGenHelpers = helpers if helpers is not None else MlirGenHelpers()

  def parse_keywords(self, op: OperationNode) -> list[str]:
    """Extract ``arg_keywords`` from ``op`` into cleaned keyword strings.

    Args:
        op: ``OperationNode`` to inspect.

    Returns:
        List of keyword names (empty string when positional).
    """
    for attr in op.attributes:
      if attr.name == "arg_keywords":
        if isinstance(attr.value, list):
          return [str(v).strip('"') for v in attr.value]
        if isinstance(attr.value, str):
          try:
            val = ast.literal_eval(attr.value)
            if isinstance(val, list):
              return [str(v).strip('"').strip("'") for v in val]
          except (ValueError, SyntaxError):
            pass
    return []

  def expr_sw_constant(self, op: OperationNode) -> cst.BaseExpression:
    """Generate a Python constant literal expression from an ``sw.constant`` op.

    Args:
        op: ``OperationNode`` representing ``sw.constant``.

    Returns:
        Parsed ``cst.BaseExpression`` or fallback ``cst.Name``.
    """
    val_str = self.helpers.get_attr(op, "value") or "0"
    try:
      return cst.parse_expression(val_str)
    except cst.ParserSyntaxError:
      return cst.Name(val_str)

  def expr_sw_getattr(self, op: OperationNode) -> cst.BaseExpression:
    """Generate a Python attribute access expression from an ``sw.getattr`` op.

    Args:
        op: ``OperationNode`` representing ``sw.getattr``.

    Returns:
        Constructed ``cst.Attribute`` (or ``cst.Name('error')`` if missing operands).
    """
    if not op.operands:
      return cst.Name("error")
    obj_expr = self.host._resolve_operand(op.operands[0].name)
    attr_name = (self.helpers.get_attr(op, "name") or "unknown").strip('"')
    return cst.Attribute(value=obj_expr, attr=cst.Name(attr_name))

  def expr_sw_call(self, op: OperationNode) -> cst.BaseExpression:
    """Generate a Python call expression from an ``sw.call`` op.

    Args:
        op: ``OperationNode`` representing ``sw.call``.

    Returns:
        Constructed ``cst.Call`` expression.
    """
    if not op.operands:
      return cst.Call(func=cst.Name("unknown"))

    func_expr = self.host._resolve_operand(op.operands[0].name)
    keywords = self.parse_keywords(op)
    args: list[cst.Arg] = []

    for i, op_val in enumerate(op.operands[1:]):
      arg_expr = self.host._resolve_operand(op_val.name)
      kw_node = cst.Name(keywords[i]) if (i < len(keywords) and keywords[i]) else None
      args.append(
        cst.Arg(
          value=arg_expr,
          keyword=kw_node,
          equal=cst.AssignEqual(whitespace_before=cst.SimpleWhitespace(""), whitespace_after=cst.SimpleWhitespace(""))
          if kw_node
          else cst.MaybeSentinel.DEFAULT,
        )
      )

    return cst.Call(func=func_expr, args=args)

  def expr_sw_op(self, op: OperationNode) -> cst.BaseExpression:
    """Generate a Python call or binary expression from an ``sw.op`` operation.

    Args:
        op: ``OperationNode`` representing ``sw.op``.

    Returns:
        Constructed ``cst.Call`` or ``cst.BinaryOperation`` expression.
    """
    type_name = self.helpers.get_attr(op, "type") or '"unknown"'
    if "binop." in type_name:
      return self.expr_binop(op, type_name)

    dotted_path = type_name.strip('"')
    func_node = self.helpers.create_dotted_name(dotted_path)
    keywords = self.parse_keywords(op)
    args: list[cst.Arg] = []

    for i, op_val in enumerate(op.operands):
      arg_expr = self.host._resolve_operand(op_val.name)
      kw_node = cst.Name(keywords[i]) if (i < len(keywords) and keywords[i]) else None
      args.append(
        cst.Arg(
          value=arg_expr,
          keyword=kw_node,
          equal=cst.AssignEqual(whitespace_before=cst.SimpleWhitespace(""), whitespace_after=cst.SimpleWhitespace(""))
          if kw_node
          else cst.MaybeSentinel.DEFAULT,
        )
      )

    return cst.Call(func=func_node, args=args)

  def expr_binop(self, op: OperationNode, type_attr: str) -> cst.BaseExpression:
    """Generate a ``cst.BinaryOperation`` from a ``binop.*`` ``sw.op``.

    Args:
        op: Binary ``OperationNode``.
        type_attr: ``type`` attribute string containing ``binop.<code>``.

    Returns:
        Constructed ``cst.BinaryOperation`` (or ``cst.Name('error_binop')`` if <2 operands).
    """
    op_code = type_attr.strip('"').replace("binop.", "")
    if len(op.operands) < 2:
      return cst.Name("error_binop")

    lhs_expr = self.host._resolve_operand(op.operands[0].name)
    rhs_expr = self.host._resolve_operand(op.operands[1].name)

    op_map: dict[str, cst.BaseBinaryOp] = {
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


__all__ = ["MlirExprGeneratorHost", "MlirExpressionGenerator"]
