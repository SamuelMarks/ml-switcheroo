"""MLIR to Python (LibCST) Generator.

Provides ``MlirToPythonGenerator``, which consumes the MLIR CST object model and
reconstructs valid Python code via LibCST by composing ``MlirGenHelpers``,
``MlirExpressionGenerator``, and ``MlirStatementGenerator``.
"""

import typing
from collections import defaultdict

import libcst as cst

from ml_switcheroo.core.cst.base import Trivia
from ml_switcheroo.core.mlir.cst import BlockNode, ModuleNode, OperationNode
from ml_switcheroo.core.mlir.gen_base import MlirGenHelpers
from ml_switcheroo.core.mlir.gen_expressions import MlirExpressionGenerator
from ml_switcheroo.core.mlir.gen_statements import MlirStatementGenerator
from ml_switcheroo.core.mlir.naming import NamingContext


@typing.runtime_checkable
class _TriviaContentProtocol(typing.Protocol):
  """Protocol for trivia nodes exposing a ``content`` string attribute."""

  @property
  def content(self) -> str | None:
    """Return raw trivia content text.

    Returns:
        Trivia content string or ``None``.
    """


@typing.runtime_checkable
class _TriviaTextProtocol(typing.Protocol):
  """Protocol for trivia nodes exposing a ``text`` string attribute."""

  @property
  def text(self) -> str | None:
    """Return raw trivia text.

    Returns:
        Trivia text string or ``None``.
    """


StatementLineOrCompound: typing.TypeAlias = cst.SimpleStatementLine | cst.BaseCompoundStatement


class MlirToPythonGenerator:
  """Transpiler back-end reconstructing Python LibCST from MLIR CST via composition.

  Attributes:
      ctx: ``NamingContext`` mapping SSA values to Python identifiers.
      usage_counts: Mapping from SSA value names to operand reference counts.
      usage_consumers: Mapping from SSA value names to consumer ``OperationNode`` instances.
      deferred_exprs: Mapping from SSA value names to folded inline expressions.
      deferred_trivia: Mapping from SSA value names to deferred leading comment lines.
      helpers: ``MlirGenHelpers`` utility component.
      expr_generator: ``MlirExpressionGenerator`` delegate for expression ops.
      stmt_generator: ``MlirStatementGenerator`` delegate for structural statement ops.
  """

  def __init__(self) -> None:
    """Initialize the generator, SSA usage state, and composed delegates."""
    self.ctx: NamingContext = NamingContext()
    self.usage_counts: defaultdict[str, int] = defaultdict(int)
    self.usage_consumers: dict[str, OperationNode] = {}
    self.deferred_exprs: dict[str, cst.BaseExpression] = {}
    self.deferred_trivia: dict[str, list[cst.EmptyLine]] = {}
    self._in_func_def: bool = False
    self.helpers: MlirGenHelpers = MlirGenHelpers()
    self.expr_generator: MlirExpressionGenerator = MlirExpressionGenerator(self, self.helpers)
    self.stmt_generator: MlirStatementGenerator = MlirStatementGenerator(self, self.helpers)

  def _get_attr(self, op: OperationNode, key: str) -> str | None:
    """Delegate MLIR attribute string lookup to ``self.helpers``.

    Args:
        op: ``OperationNode`` to inspect.
        key: Attribute name key.

    Returns:
        Attribute value string or ``None``.
    """
    return self.helpers.get_attr(op, key)

  def _create_dotted_name(self, path: str) -> cst.Name | cst.Attribute:
    """Delegate dotted CST ``Name``/``Attribute`` construction to ``self.helpers``.

    Args:
        path: Dot-separated identifier string.

    Returns:
        Constructed ``cst.Name`` or ``cst.Attribute``.
    """
    return self.helpers.create_dotted_name(path)

  def generate(self, node: ModuleNode) -> cst.Module:
    """Convert an MLIR ``ModuleNode`` into a Python ``cst.Module``.

    Args:
        node: Root MLIR ``ModuleNode``.

    Returns:
        Reconstructed LibCST ``Module``.
    """
    self._analyze_module_usage(node)
    stmt_body = self._convert_block(node.body)
    return cst.Module(body=stmt_body)

  def _analyze_module_usage(self, mod: ModuleNode) -> None:
    """Populate ``self.usage_counts`` and ``self.usage_consumers`` across ``mod``.

    Args:
        mod: Root MLIR ``ModuleNode``.
    """
    self._scan_block_usage(mod.body)

  def _scan_block_usage(self, block: BlockNode) -> None:
    """Recursively scan ``block`` operations and nested regions to count SSA usage.

    Args:
        block: ``BlockNode`` to scan.
    """
    for op in block.operations:
      for operand in op.operands:
        self.usage_counts[operand.name] += 1
        self.usage_consumers[operand.name] = op

      for region in op.regions:
        for b in region.blocks:
          self._scan_block_usage(b)

  def _convert_trivia(
    self,
    trivia: typing.Sequence["Trivia | _TriviaContentProtocol | _TriviaTextProtocol"],
  ) -> list[cst.EmptyLine]:
    """Convert MLIR ``//`` / ``%`` trivia entries into Python ``#`` ``cst.EmptyLine`` comments.

    Args:
        trivia: Sequence of MLIR trivia items.

    Returns:
        List of ``cst.EmptyLine`` comment lines.
    """
    lines: list[cst.EmptyLine] = []

    for t in trivia:
      if isinstance(t, Trivia):
        content = t.content.strip()
      elif isinstance(t, _TriviaContentProtocol):
        content = str(t.content or "").strip()
      else:
        content = str(t.text or "").strip()
      if content.startswith("//"):
        lines.append(cst.EmptyLine(comment=cst.Comment("#" + content[2:]), newline=cst.Newline()))
      elif content.startswith("%"):
        lines.append(cst.EmptyLine(comment=cst.Comment("#" + content), newline=cst.Newline()))
    return lines

  def _convert_block(self, block: BlockNode) -> list[StatementLineOrCompound]:
    """Convert operations in ``block`` into a list of Python statements.

    Args:
        block: MLIR ``BlockNode`` containing operations to convert.

    Returns:
        List of LibCST statements.
    """
    stmts: list[StatementLineOrCompound] = []

    for op in block.operations:
      leading = self._convert_trivia(op.leading_trivia)
      op_name_clean = op.name.strip('"')

      if (
        op_name_clean == "module"
        and not self._get_attr(op, "sym_name")
        and op.regions
        and op.regions[0].blocks
        and any(
          inner_op.name.strip('"') == "func.func"
          and any(
            sub_op.name.strip('"').startswith("stablehlo.")
            for r in inner_op.regions
            for b in r.blocks
            for sub_op in b.operations
          )
          for inner_op in op.regions[0].blocks[0].operations
        )
      ):
        stmts.extend(self._convert_block(op.regions[0].blocks[0]))
        continue

      collected_triv: list[cst.EmptyLine] = []
      for operand in op.operands:
        if operand.name in self.deferred_trivia:
          collected_triv.extend(self.deferred_trivia.pop(operand.name))
      collected_triv.extend(leading)

      expr_node = self._create_expression_from_op(op)

      if expr_node:
        if self._should_inline_expression(op, expr_node):
          res_ssa = op.results[0].name
          self.deferred_exprs[res_ssa] = expr_node
          if collected_triv:
            self.deferred_trivia[res_ssa] = collected_triv
        else:
          stmt_node = self._wrap_as_statement(op, expr_node)
          if collected_triv:
            stmt_node = stmt_node.with_changes(leading_lines=collected_triv)
          stmts.append(stmt_node)
      else:
        stmt_op = self._convert_statement_op(op)
        if stmt_op is not None:
          if collected_triv:
            stmt_op = stmt_op.with_changes(leading_lines=collected_triv)
          stmts.append(stmt_op)

    return stmts

  def _should_inline_expression(self, op: OperationNode, expr: cst.BaseExpression) -> bool:
    """Return whether ``op``'s result expression should be folded inline at its consumer.

    Args:
        op: Candidate ``OperationNode``.
        expr: Reconstructed ``cst.BaseExpression`` for ``op``.

    Returns:
        ``True`` if ``expr`` should be deferred and inlined.
    """
    if not op.results:
      return False

    op_name = op.name.strip('"')
    res_ssa = op.results[0].name
    usage = self.usage_counts[res_ssa]

    if op_name in ("sw.constant", "sw.getattr"):
      return usage > 0

    type_attr = self._get_attr(op, "type")
    if op_name == "sw.op" and type_attr and "super" in type_attr:
      return True

    if usage == 1 and res_ssa in self.usage_consumers:
      consumer = self.usage_consumers[res_ssa]
      if consumer.name.strip('"') in {"sw.setattr", "sw.return"}:
        return True

    return False

  def _resolve_operand(self, ssa_name: str) -> cst.BaseExpression:
    """Resolve ``ssa_name`` to either a deferred expression or a named variable/attribute.

    Args:
        ssa_name: SSA value identifier.

    Returns:
        Resolved ``cst.BaseExpression``.
    """
    if ssa_name in self.deferred_exprs:
      return self.deferred_exprs[ssa_name]

    py_name = self.ctx.lookup(ssa_name)
    if "." in py_name:
      return self._create_dotted_name(py_name)

    return cst.Name(py_name)

  def _create_expression_from_op(self, op: OperationNode) -> cst.BaseExpression | None:
    """Delegate expression-producing MLIR operations to ``self.expr_generator``.

    Args:
        op: ``OperationNode`` to inspect.

    Returns:
        Constructed ``cst.BaseExpression`` or ``None`` if ``op`` is structural.
    """
    op_name = op.name.strip('"')

    if op_name == "sw.op":
      return self.expr_generator.expr_sw_op(op)
    if op_name == "sw.call":
      return self.expr_generator.expr_sw_call(op)
    if op_name == "sw.getattr":
      return self.expr_generator.expr_sw_getattr(op)
    if op_name == "sw.constant":
      return self.expr_generator.expr_sw_constant(op)
    if op_name.startswith("stablehlo.") and op_name != "stablehlo.return" and self._in_func_def:
      return self._expr_stablehlo_op(op)

    return None

  def _expr_stablehlo_op(self, op: OperationNode) -> cst.BaseExpression:
    """Convert a ``stablehlo.*`` ``OperationNode`` into a Python ``cst.Call`` expression.

    Args:
        op: StableHLO ``OperationNode`` to convert.

    Returns:
        Constructed ``cst.Call`` expression invoking ``stablehlo.<op>(...)``.
    """
    op_name = op.name.strip('"')
    if op_name == "stablehlo.custom_call":
      target_attr = self._get_attr(op, "call_target_name") or '"@op"'
      clean_target = target_attr.strip('"').lstrip("@")
      func_node = self._create_dotted_name(f"stablehlo.{clean_target}")
    else:
      func_node = self._create_dotted_name(op_name)
    args = [cst.Arg(value=self._resolve_operand(o.name)) for o in op.operands if not o.name.startswith("@")]
    return cst.Call(func=func_node, args=args)

  def _convert_statement_op(self, op: OperationNode) -> StatementLineOrCompound | None:
    """Delegate structural MLIR operations to ``self.stmt_generator``.

    Args:
        op: ``OperationNode`` to convert.

    Returns:
        Constructed LibCST statement or ``None`` if unrecognized.
    """
    op_name = op.name.strip('"')

    if op_name == "sw.module":
      return self.stmt_generator.convert_class_def(op)
    if op_name in ("sw.func", "func.func"):
      was_in = self._in_func_def
      self._in_func_def = True
      try:
        return self.stmt_generator.convert_func_def(op)
      finally:
        self._in_func_def = was_in
    if op_name in ("sw.return", "func.return", "stablehlo.return", "return"):
      return self.stmt_generator.convert_return(op)
    if op_name == "sw.setattr":
      return self.stmt_generator.convert_setattr(op)
    if op_name == "sw.import":
      return self.stmt_generator.convert_import(op)

    return None

  def _wrap_as_statement(self, op: OperationNode, expr: cst.BaseExpression) -> cst.SimpleStatementLine:
    """Wrap ``expr`` as either an ``Assign`` statement or a standalone ``Expr`` statement.

    Args:
        op: ``OperationNode`` whose result is being wrapped.
        expr: ``cst.BaseExpression`` to wrap.

    Returns:
        Constructed ``cst.SimpleStatementLine``.
    """
    if op.results:
      res_ssa = op.results[0].name

      if self.usage_counts[res_ssa] == 0 or self._is_void_call(expr):
        return cst.SimpleStatementLine(body=[cst.Expr(value=expr)])

      hint: str | None = None
      raw_t = self._get_attr(op, "type")
      if raw_t:
        val = raw_t.strip('"').strip("'")
        hint = val.split(".")[-1].lower()
      elif op.name.strip('"') == "sw.getattr":
        raw_n = self._get_attr(op, "name")
        if raw_n:
          hint = raw_n.strip('"')
      elif op.name.strip('"') == "sw.constant":
        hint = "cst"

      py_target = self.ctx.register(res_ssa, hint=hint)
      target_node = cst.AssignTarget(target=cst.Name(py_target))
      return cst.SimpleStatementLine(body=[cst.Assign(targets=[target_node], value=expr)])

    return cst.SimpleStatementLine(body=[cst.Expr(value=expr)])

  def _is_void_call(self, expr: cst.BaseExpression) -> bool:
    """Return whether ``expr`` matches ``super().__init__(...)`` void call pattern.

    Args:
        expr: ``cst.BaseExpression`` to inspect.

    Returns:
        ``True`` if ``expr`` is a ``super().__init__`` call.
    """
    if isinstance(expr, cst.Call) and isinstance(expr.func, cst.Attribute) and expr.func.attr.value == "__init__":
      receiver = expr.func.value
      if isinstance(receiver, cst.Call) and isinstance(receiver.func, cst.Name) and receiver.func.value == "super":
        return True
    return False


__all__ = [
  "MlirExpressionGenerator",
  "MlirGenHelpers",
  "MlirStatementGenerator",
  "MlirToPythonGenerator",
  "StatementLineOrCompound",
]
