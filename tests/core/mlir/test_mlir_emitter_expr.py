"""Test suite for ``MlirExprEmitter`` and ``PythonToMlirEmitter`` expression lowering."""

import libcst as cst
from libcst._nodes.internal import CodegenState

from ml_switcheroo.core.mlir.cst import OperationNode, ValueNode
from ml_switcheroo.core.mlir.emitter import PythonToMlirEmitter
from ml_switcheroo.core.mlir.emitter_decl import SSAContext
from ml_switcheroo.core.mlir.emitter_expr import MlirExprEmitter


class DummyExprHost:
  """Typed host implementing ``MlirExprEmitterHost`` for isolated unit tests."""

  def __init__(self) -> None:
    """Initialize ``SSAContext`` with ``foo -> %foo`` bound and attach ``MlirExprEmitter``."""
    self.ctx: SSAContext = SSAContext()
    self.ctx.declare("foo", ValueNode(name="%foo"))
    self.emitter: MlirExprEmitter = MlirExprEmitter(self)

  def _emit_expression(self, expr: cst.BaseExpression) -> tuple[ValueNode, list[OperationNode]]:
    """Delegate recursive sub-expression emission to ``self.emitter``.

    Args:
        expr: LibCST expression node.

    Returns:
        Tuple of ``(ValueNode, list[OperationNode])``.
    """
    return self.emitter.emit_expression(expr)

  def _flatten_attr(self, node: cst.CSTNode) -> str | None:
    """Flatten ``Name`` or ``Attribute`` CST nodes into a dotted string.

    Args:
        node: CST node to flatten.

    Returns:
        Dotted string or ``None``.
    """
    if isinstance(node, cst.Name):
      return node.value
    if isinstance(node, cst.Attribute):
      base = self._flatten_attr(node.value)
      if base:
        return f"{base}.{node.attr.value}"
    return None

  def _get_binop_str(self, operator: cst.BaseBinaryOp) -> str:
    """Map binary operator CST nodes to MLIR ``binop.*`` string codes.

    Args:
        operator: CST binary operator node.

    Returns:
        String code for ``operator``.
    """
    if isinstance(operator, cst.Add):
      return "add"
    if isinstance(operator, cst.Multiply):
      return "mul"
    return "unknown"


def test_emit_names_calls_binops_and_constants() -> None:
  """Verify ``MlirExprEmitter`` lowers names, static/method/function calls, binops, and literals."""
  host = DummyExprHost()

  val, ops = host._emit_expression(cst.Name(value="foo"))
  assert val.name == "%foo"
  assert len(ops) == 0

  val2, ops2 = host._emit_expression(cst.Name(value="bar"))
  assert val2.name == "@bar"
  assert len(ops2) == 0

  static_call = cst.Call(
    func=cst.Attribute(value=cst.Name("tf"), attr=cst.Name("add")),
    args=[cst.Arg(value=cst.Name("foo")), cst.Arg(keyword=cst.Name("y"), value=cst.Name("foo"))],
  )
  val_sc, ops_sc = host._emit_expression(static_call)
  assert val_sc.name == "%0"
  assert len(ops_sc) == 1
  assert ops_sc[0].name == "sw.op"

  method_call = cst.Call(func=cst.Attribute(value=cst.Name("foo"), attr=cst.Name("add")), args=[])
  _, ops_mc = host._emit_expression(method_call)
  assert len(ops_mc) == 2
  assert ops_mc[0].name == "sw.getattr"
  assert ops_mc[1].name == "sw.call"

  func_call = cst.Call(func=cst.Name("foo"), args=[])
  _, ops_fc = host._emit_expression(func_call)
  assert len(ops_fc) == 1
  assert ops_fc[0].name == "sw.call"

  bin_node = cst.BinaryOperation(left=cst.Name("foo"), operator=cst.Add(), right=cst.Name("foo"))
  _, ops_bin = host._emit_expression(bin_node)
  assert len(ops_bin) == 1
  assert any(a.name == "type" and a.value == '"binop.add"' for a in ops_bin[0].attributes)

  _, ops_i = host._emit_expression(cst.Integer("42"))
  _, ops_f = host._emit_expression(cst.Float("3.14"))
  assert len(ops_i) == 1
  assert len(ops_f) == 1

  nested_call = cst.Call(func=cst.Call(func=cst.Name("foo"), args=[]), args=[])
  val_nc, _ = host._emit_expression(nested_call)
  assert val_nc.name == "%error"


def test_emit_fallback_annotations_and_null_telemetry() -> None:
  """Verify unsupported expression fallback, annotation flattening, and null ``runtime_trace`` telemetry."""
  host = DummyExprHost()

  class UnsupportedExpr(cst.BaseExpression):
    """Minimal unsupported ``BaseExpression`` subclass for fallback testing."""

    def _visit_and_replace_children(self, visitor: cst.CSTVisitorT) -> cst.CSTNode:
      """Return ``self`` unchanged during child visitation.

      Args:
          visitor: Active CST visitor.

      Returns:
          Unchanged ``self``.
      """
      return self

    def _codegen_impl(self, state: CodegenState, default_semi: bool = False) -> None:
      """No-op codegen hook for test expression node.

      Args:
          state: LibCST ``CodegenState``.
          default_semi: Whether to emit a trailing semicolon.
      """

  val, _ = host._emit_expression(UnsupportedExpr())
  assert val.name == "%error"

  full_emitter = PythonToMlirEmitter()
  assert full_emitter._match_stmt_telemetry("x") is None
  assert full_emitter._annotation_to_string(cst.Name("int")) == "int"
  assert full_emitter._annotation_to_string(cst.Attribute(value=cst.Name("typing"), attr=cst.Name("Any"))) == "typing.Any"
  assert full_emitter._annotation_to_string(UnsupportedExpr()) == "Any"
