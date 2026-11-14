"""MLIR Emitter Logic.

Provides ``PythonToMlirEmitter``, a compiler front-end translating Python LibCST
trees into the MLIR CST object model via explicit composition of ``MlirDeclEmitter``
and ``MlirExprEmitter`` over ``SSAContext``.
"""

import typing

import libcst as cst

from ml_switcheroo.core.cst.base import Trivia
from ml_switcheroo.core.mlir.cst import AttributeNode, BlockNode, ModuleNode, OperationNode, TypeNode, ValueNode
from ml_switcheroo.core.mlir.emitter_decl import MlirDeclEmitter, SSAContext
from ml_switcheroo.core.mlir.emitter_expr import MlirExprEmitter
from ml_switcheroo.core.mlir.types import format_mlir_tensor_type
from ml_switcheroo.core.runtime_models import NodeRuntimeTelemetry, RuntimeTraceReport
from ml_switcheroo.core.scanners import get_full_name


class _LeadingLineProtocol(typing.Protocol):
  """Protocol for leading line items exposing optional ``comment`` and ``newline``.

  Attributes:
      comment: Optional LibCST comment node.
      newline: Optional LibCST newline node.
  """

  comment: cst.Comment | None
  newline: cst.Newline | None


@typing.runtime_checkable
class _LeadingLinesProtocol(typing.Protocol):
  """Protocol for CST nodes exposing ``leading_lines``.

  Attributes:
      leading_lines: Leading empty/comment lines sequence.
  """

  leading_lines: typing.Sequence[_LeadingLineProtocol]


class PythonToMlirEmitter:
  """Translate Python LibCST modules into MLIR structural nodes via composed delegates.

  Attributes:
      ctx: ``SSAContext`` tracking lexical variable bindings and SSA allocations.
      runtime_trace: Optional ``RuntimeTraceReport`` for shape telemetry interweaving.
      decl_emitter: ``MlirDeclEmitter`` delegate handling class and function declarations.
      expr_emitter: ``MlirExprEmitter`` delegate handling expression lowering.
  """

  def __init__(self, runtime_trace: RuntimeTraceReport | None = None) -> None:
    """Initialize the emitter with a fresh ``SSAContext`` and composed delegates.

    Args:
        runtime_trace: Optional ``RuntimeTraceReport`` for interweaving ranked tensor types and trivia.
    """
    self.ctx: SSAContext = SSAContext()
    self.runtime_trace: RuntimeTraceReport | None = runtime_trace
    self._telemetry_cursor: int = 0
    self._in_inference_func: bool = False
    self.decl_emitter: MlirDeclEmitter = MlirDeclEmitter(self)
    self.expr_emitter: MlirExprEmitter = MlirExprEmitter(self)

  def _emit_class_def(self, node: cst.ClassDef) -> OperationNode:
    """Delegate ``cst.ClassDef`` emission to ``self.decl_emitter``.

    Args:
        node: ``cst.ClassDef`` node to lower.

    Returns:
        Emitted ``OperationNode``.
    """
    return self.decl_emitter.emit_class_def(node)

  def _emit_func_def(self, node: cst.FunctionDef) -> OperationNode:
    """Delegate ``cst.FunctionDef`` emission to ``self.decl_emitter``.

    Args:
        node: ``cst.FunctionDef`` node to lower.

    Returns:
        Emitted ``OperationNode``.
    """
    return self.decl_emitter.emit_func_def(node)

  def _emit_expression(self, expr: cst.BaseExpression) -> tuple[ValueNode, list[OperationNode]]:
    """Delegate expression emission to ``self.expr_emitter``.

    Args:
        expr: LibCST expression node to lower.

    Returns:
        Tuple of ``(ValueNode, list[OperationNode])``.
    """
    return self.expr_emitter.emit_expression(expr)

  def _annotation_to_string(self, node: cst.CSTNode) -> str:
    """Delegate type annotation flattening to ``MlirExprEmitter.annotation_to_string``.

    Args:
        node: CST type annotation node.

    Returns:
        Flattened annotation string.
    """
    return MlirExprEmitter.annotation_to_string(node)

  def _match_stmt_telemetry(self, var_name: str) -> NodeRuntimeTelemetry | None:
    """Match the next ``NodeRuntimeTelemetry`` record from ``self.runtime_trace``.

    Args:
        var_name: Assigned variable name or ``'return'``.

    Returns:
        Matched ``NodeRuntimeTelemetry``, or ``None``.
    """
    if self.runtime_trace is None:
      return None
    items = list(self.runtime_trace.node_telemetry.values())
    for idx in range(self._telemetry_cursor, len(items)):
      if items[idx].var_name == var_name:
        self._telemetry_cursor = idx + 1
        return items[idx]
    return None

  @staticmethod
  def _format_mlir_runtime_comment(tel: NodeRuntimeTelemetry) -> str:
    """Format a ``NodeRuntimeTelemetry`` record as an MLIR ``// [on-device fwd] ...`` comment.

    Args:
        tel: ``NodeRuntimeTelemetry`` for the statement.

    Returns:
        Formatted MLIR comment string starting with ``// ``.
    """
    parts = [f"// [on-device fwd] shape={tuple(tel.output_shape)} dtype={tel.dtype}"]
    if tel.is_dynamic and tel.dynamic_reason:
      parts.append(f"dynamic: {tel.dynamic_reason}")
    if tel.grad_shape is not None and tel.grad_norm is not None:
      parts.append(f"[bwd] grad={tuple(tel.grad_shape)} norm={tel.grad_norm:.4e}")
    return " | ".join(parts)

  def convert(self, node: cst.Module) -> ModuleNode:
    """Convert a Python ``cst.Module`` to an MLIR ``ModuleNode``.

    Args:
        node: Python LibCST ``Module``.

    Returns:
        Resulting MLIR ``ModuleNode`` containing translated operations.
    """
    body_block = self._emit_block(node.body)

    header_trivia: list[Trivia] = []
    for line in node.header:
      if line.comment:
        text = line.comment.value.replace("#", "//", 1)
        header_trivia.append(Trivia(text))
        header_trivia.append(Trivia(chr(10)))
      else:
        header_trivia.append(Trivia(chr(10)))

      if header_trivia and body_block.operations:
        body_block.operations[0].leading_trivia = header_trivia + body_block.operations[0].leading_trivia

    return ModuleNode(body=body_block)

  def _extract_trivia(self, node: cst.CSTNode) -> list[Trivia]:
    """Extract comments and blank lines from ``node``'s leading and trailing trivia.

    Args:
        node: CST node to inspect.

    Returns:
        List of MLIR ``Trivia`` items (with ``#`` translated to ``//``).
    """
    trivia: list[Trivia] = []
    leading_lines = node.leading_lines if isinstance(node, _LeadingLinesProtocol) else ()
    for line in leading_lines:
      if line.comment:
        text = line.comment.value.replace("#", "//", 1)
        trivia.append(Trivia(text))
        trivia.append(Trivia(chr(10)))
      elif line.newline is not None and line.newline.value:
        trivia.append(Trivia(chr(10)))

    if isinstance(node, cst.SimpleStatementLine) and node.trailing_whitespace.comment is not None:
      text = node.trailing_whitespace.comment.value.replace("#", "//", 1)
      trivia.append(Trivia(text))
      trivia.append(Trivia(chr(10)))

    return trivia

  def _emit_block(self, body_enc: cst.BaseSuite | typing.Sequence[cst.CSTNode], label: str = "") -> BlockNode:
    """Convert a CST suite or statement sequence into an MLIR ``BlockNode``.

    Args:
        body_enc: CST Suite or sequence of statements.
        label: Optional block label (e.g. ``^entry``).

    Returns:
        Populated ``BlockNode``.
    """
    block = BlockNode(label=label)
    stmts: typing.Sequence[cst.CSTNode] = []
    if isinstance(body_enc, (cst.IndentedBlock, cst.SimpleStatementSuite, cst.Module)):
      stmts = body_enc.body
    elif isinstance(body_enc, (list, tuple)):
      stmts = body_enc

    for stmt in stmts:
      ops = self._emit_statement(stmt)
      if ops:
        block.operations.extend(ops)
    return block

  def _emit_statement(self, stmt: cst.CSTNode) -> list[OperationNode]:
    """Dispatch statement CST nodes to declaration or small-statement handlers.

    Args:
        stmt: Statement CST node.

    Returns:
        List of generated MLIR ``OperationNode`` instances.
    """
    results: list[OperationNode] = []

    if isinstance(stmt, cst.ClassDef):
      results = [self._emit_class_def(stmt)]
    elif isinstance(stmt, cst.FunctionDef):
      was_in = self._in_inference_func
      if stmt.name.value not in ("__init__", "setup"):
        self._in_inference_func = True
      results = [self._emit_func_def(stmt)]
      self._in_inference_func = was_in
    elif isinstance(stmt, cst.If):
      results = self._emit_if(stmt)
    elif isinstance(stmt, cst.While):
      results = self._emit_while(stmt)
    elif isinstance(stmt, cst.SimpleStatementLine) and len(stmt.body) > 0:
      node = stmt.body[0]
      results = self._dispatch_small_stmt(node)
      if results and self._in_inference_func and self.runtime_trace is not None and self.runtime_trace.executed:
        var_name: str | None = (
          node.targets[0].target.value
          if (isinstance(node, cst.Assign) and node.targets and isinstance(node.targets[0].target, cst.Name))
          else None
        )
        if var_name is not None:
          tel = self._match_stmt_telemetry(var_name)
          if tel is not None:
            ranked_t = format_mlir_tensor_type(tel.output_shape, tel.dtype)
            for op_item in results:
              op_item.result_types = [TypeNode(body=ranked_t)]
            rt_cmt = self._format_mlir_runtime_comment(tel)
            results[0].leading_trivia = [*results[0].leading_trivia, Trivia(rt_cmt), Trivia(chr(10))]

    if results:
      extracted = self._extract_trivia(stmt)
      if extracted:
        results[0].leading_trivia = extracted + results[0].leading_trivia

    return results

  def _dispatch_small_stmt(self, node: cst.CSTNode) -> list[OperationNode]:
    """Handle inner small statements inside ``SimpleStatementLine``.

    Args:
        node: Inner small statement CST node.

    Returns:
        List of resulting ``OperationNode`` instances.
    """
    if isinstance(node, cst.Assign):
      return self._emit_assign(node)
    if isinstance(node, cst.Return):
      return self._emit_return(node)
    if isinstance(node, cst.Expr):
      _, ops = self._emit_expression(node.value)
      return ops
    if isinstance(node, (cst.Import, cst.ImportFrom)):
      return [self._emit_import(node)]
    return []

  def _emit_import(self, node: cst.Import | cst.ImportFrom) -> OperationNode:
    """Convert ``Import`` / ``ImportFrom`` into an ``sw.import`` ``OperationNode``.

    Args:
        node: ``cst.Import`` or ``cst.ImportFrom`` node.

    Returns:
        Constructed ``sw.import`` ``OperationNode``.
    """
    names: list[str] = []
    aliases: list[str] = []
    module_val = ""

    if isinstance(node, cst.ImportFrom) and node.module:
      module_val = get_full_name(node.module)

    if isinstance(node.names, cst.ImportStar):
      names.append("*")
      aliases.append("")
    else:
      for alias in node.names:
        names.append(get_full_name(alias.name))
        if alias.asname:
          aliases.append(alias.asname.name.value if isinstance(alias.asname.name, cst.Name) else "")
        else:
          aliases.append("")

    attrs: list[AttributeNode] = []
    if module_val:
      attrs.append(AttributeNode(name="module", value=f'"{module_val}"'))

    attrs.append(AttributeNode(name="names", value=[f'"{n}"' for n in names]))
    attrs.append(AttributeNode(name="aliases", value=[f'"{a}"' for a in aliases]))

    return OperationNode(name="sw.import", attributes=attrs)

  def _emit_assign(self, node: cst.Assign) -> list[OperationNode]:
    """Convert a Python ``cst.Assign`` statement into expression ops and SSA/attribute bindings.

    Args:
        node: ``cst.Assign`` node.

    Returns:
        List of emitted ``OperationNode`` instances.
    """
    val, ops = self._emit_expression(node.value)

    for target in node.targets:
      t = target.target
      if isinstance(t, cst.Name):
        self.ctx.declare(t.value, val)
      elif isinstance(t, cst.Attribute):
        base_name = self._flatten_attr(t.value)
        if base_name:
          base_val = self.ctx.lookup(base_name)
          if base_val:
            attr_name = t.attr.value
            set_op = OperationNode(
              name="sw.setattr",
              operands=[base_val, val],
              attributes=[AttributeNode(name="name", value=f'"{attr_name}"')],
            )
            ops.append(set_op)

    return ops

  def _emit_while(self, node: cst.While) -> list[OperationNode]:
    """Convert a ``while`` statement (overridden in dialect-specific subclasses).

    Args:
        node: ``cst.While`` node.

    Returns:
        List of generated ``OperationNode`` instances.
    """
    return []

  def _emit_if(self, node: cst.If) -> list[OperationNode]:
    """Convert an ``if`` statement (overridden in dialect-specific subclasses).

    Args:
        node: ``cst.If`` node.

    Returns:
        List of generated ``OperationNode`` instances.
    """
    return []

  def _emit_return(self, node: cst.Return) -> list[OperationNode]:
    """Convert a ``return`` statement into expression ops followed by ``sw.return``.

    Args:
        node: ``cst.Return`` node.

    Returns:
        List of expression evaluation ops and the terminal ``sw.return`` op.
    """
    ops: list[OperationNode] = []
    operands: list[ValueNode] = []
    if node.value:
      val, expr_ops = self._emit_expression(node.value)
      ops.extend(expr_ops)
      operands.append(val)

    ops.append(OperationNode(name="sw.return", operands=operands))
    return ops

  def _flatten_attr(self, node: cst.CSTNode) -> str | None:
    """Flatten a ``Name`` or ``Attribute`` chain into a dotted string.

    Args:
        node: CST node to flatten.

    Returns:
        Dotted string (e.g. ``'self.layer'``) or ``None``.
    """
    if isinstance(node, cst.Name):
      return node.value
    if isinstance(node, cst.Attribute):
      base = self._flatten_attr(node.value)
      if base:
        return f"{base}.{node.attr.value}"
    return None

  def _get_binop_str(self, operator: cst.BaseBinaryOp) -> str:
    """Map LibCST binary operator classes to MLIR ``binop.*`` string codes.

    Args:
        operator: CST binary operator node.

    Returns:
        Operator code string (e.g. ``'add'``, ``'mul'``, ``'matmul'``).
    """
    op_map = {
      cst.Add: "add",
      cst.Subtract: "sub",
      cst.Multiply: "mul",
      cst.Divide: "div",
      cst.FloorDivide: "floordiv",
      cst.Modulo: "mod",
      cst.Power: "pow",
      cst.MatrixMultiply: "matmul",
      cst.LeftShift: "lshift",
      cst.RightShift: "rshift",
      cst.BitAnd: "and",
      cst.BitOr: "or",
      cst.BitXor: "xor",
    }
    return op_map.get(type(operator), "unknown")


__all__ = ["MlirDeclEmitter", "MlirExprEmitter", "PythonToMlirEmitter", "SSAContext"]
