"""
MLIR to Python (LibCST) Generator.

This module provides the `MlirToPythonGenerator` class, which consumes the
MLIR CST object model and reconstructs valid Python code via LibCST.

It performs the inverse operation of the Emitter:

1.  **Dialect Interpretation**: Maps `sw.*` ops back to Python constructs.
2.  **SSA Resolution**: Converts `%id` references into readable Python variable names.
3.  **Trivia Restoration**: Transforms `// comment` back to `# comment`.
4.  **Expression Folding**: Inlines specific single-use intermediates (Atoms, Fused Statements)
    to produce Pythonic code.

Feature Hardening:
- Robustly handles `sw.op` "type" attributes for deep class hierarchies.
- Distinguishes between static operation calls and dynamic calls.
- Ensures valid assignment generation for results.
- **Naming Heuristics**: Preserves variable names based on SSA IDs or Type Hints.
- **Type Reconstruction**: Restores Python type hints from MLIR.
- **Void Assignment Suppression**: Strips unused assignments.
- **Explicit Re-rolling**: Enforces sequential statements for clarity unless fusion triggers.
- **Statement Fusion**: Fuses operations into `setattr` and `return`.
"""

from typing import Dict, List, Optional
from collections import defaultdict
import libcst as cst

from ml_switcheroo.core.mlir.nodes import (
  BlockNode,
  ModuleNode,
  OperationNode,
  TriviaNode,
)

# Import modular components
from ml_switcheroo.core.mlir.naming import NamingContext
from ml_switcheroo.core.mlir.gen_base import BaseGeneratorMixin
from ml_switcheroo.core.mlir.gen_expressions import ExpressionGeneratorMixin
from ml_switcheroo.core.mlir.gen_statements import StatementGeneratorMixin


class MlirToPythonGenerator(ExpressionGeneratorMixin, StatementGeneratorMixin, BaseGeneratorMixin):
  """
  Transpiler back-end: MLIR CST -> Python LibCST.
  Integrates expression and statement generation logic.
  """

  def __init__(self) -> None:
    """
    Initialize the generator.

    Sets up naming context and usage tracking containers.
    """
    self.ctx = NamingContext()

    # Store usage counts for inlining logic: {ssa_name: count}
    self.usage_counts: Dict[str, int] = defaultdict(int)
    # Map of ssa_name -> consumer_op (single consumer context)
    self.usage_consumers: Dict[str, OperationNode] = {}

    # Map of ssa_name -> CST Node (Expression) for deferred inlining
    self.deferred_exprs: Dict[str, cst.BaseExpression] = {}

  def generate(self, node: ModuleNode) -> cst.Module:
    """
    Main entry point. Converts MLIR Module to Python Module.

    Args:
        node: The root MLIR ModuleNode.

    Returns:
        A LibCST Module object representing the Python code.
    """
    # Analyze usage for inlining optimization across the whole module
    self._analyze_module_usage(node)

    # Implicit top-level block
    stmt_body = self._convert_block(node.body)

    # Ensure we return a module with valid body sequence
    return cst.Module(body=stmt_body)

  def _analyze_module_usage(self, mod: ModuleNode) -> None:
    """
    Traverses the MLIR tree to count SSAs usage.
    Populates self.usage_counts.
    """
    self._scan_block_usage(mod.body)

  def _scan_block_usage(self, block: BlockNode) -> None:
    """Recursively scans op operands to update usage counts."""
    for op in block.operations:
      # Count operand usage
      for operand in op.operands:
        self.usage_counts[operand.name] += 1
        # Track last (or only) consumer for fusion analysis
        self.usage_consumers[operand.name] = op

      # Recurse regions
      for region in op.regions:
        for b in region.blocks:
          self._scan_block_usage(b)

  def _convert_trivia(self, trivia: List[TriviaNode]) -> List[cst.EmptyLine]:
    """
    Converts MLIR comments (//) to Python comments (#).
    Ignores layout whitespace as LibCST handles indentation.
    """
    lines = []
    for t in trivia:
      content = t.content.strip()
      # Only process comments, not just newlines
      if t.kind == "comment" or content.startswith("//"):
        if content.startswith("//"):
          content = "#" + content[2:]
        lines.append(cst.EmptyLine(comment=cst.Comment(content), newline=cst.Newline()))
    return lines

  def _convert_block(self, block: BlockNode) -> List[cst.BaseStatement]:
    """
    Converts operations in a block to a list of Python statements.
    Applies expression folding where possible.
    """
    stmts: List[cst.BaseStatement] = []

    for op in block.operations:
      leading = self._convert_trivia(op.leading_trivia)

      # Try to create expression first
      expr_node = self._create_expression_from_op(op)

      if expr_node:
        # Evaluate if we should inline or emit statement
        if self._should_inline_expression(op, expr_node):
          # Defer emission
          res_ssa = op.results[0].name
          self.deferred_exprs[res_ssa] = expr_node
        else:
          # Wrap as statement (Assignment or Expression Stmt)
          stmt_node = self._wrap_as_statement(op, expr_node)
          if hasattr(stmt_node, "with_changes") and leading:
            stmt_node = stmt_node.with_changes(leading_lines=leading)
          stmts.append(stmt_node)
      else:
        # Handle statements that are never expressions (Control Flow, Class Defs, Defs)
        # These are handled by StatementGeneratorMixin
        stmt_node = self._convert_statement_op(op)
        if stmt_node:
          if hasattr(stmt_node, "with_changes") and leading:
            stmt_node = stmt_node.with_changes(leading_lines=leading)
          stmts.append(stmt_node)

    return stmts

  def _should_inline_expression(self, op: OperationNode, expr: cst.BaseExpression) -> bool:
    """
    Determines if the result of an operation should be folded/inlined.

    Revised Logic:
    1.  **Atoms**: Always inline `sw.constant` and `sw.getattr` IF USED.
    2.  **Statement Fusion**: Inline if the consumer is a "Terminal Statement"
        like `sw.setattr` or `sw.return`.
    3.  **Default**: Do not inline (sequential generation).
    """
    if not op.results:
      return False  # No result to inline

    op_name = op.name.strip('"')
    res_ssa = op.results[0].name
    usage = self.usage_counts[res_ssa]

    # 1. ATOMS: Always inline simple values (Constants, Attributes) if they are used.
    if op_name == "sw.constant":
      return usage > 0

    if op_name == "sw.getattr":
      return usage > 0

    # 2. VOID/SUPER: Always inline proxies
    type_attr = self._get_attr(op, "type")
    if op_name == "sw.op" and type_attr and "super" in type_attr:
      return True

    # 3. STATEMENT FUSION:
    if usage == 1 and res_ssa in self.usage_consumers:
      consumer = self.usage_consumers[res_ssa]
      cons_name = consumer.name.strip('"')

      # Target consumers that are definitely statements
      statement_ops = {"sw.setattr", "sw.return"}
      if cons_name in statement_ops:
        return True

    return False

  def _resolve_operand(self, ssa_name: str) -> cst.BaseExpression:
    """
    Resolves an SSA value to a CST Expression.
    If previously deferred, returns the AST node (folding).
    Else returns a Name or Attribute reference.
    """
    if ssa_name in self.deferred_exprs:
      # Retrieve deferred expression
      expr = self.deferred_exprs[ssa_name]
      return expr

    # Fallback to variable name lookup
    py_name = self.ctx.lookup(ssa_name)

    # HARDENING FIX: If name is dotted (e.g. self.layer), return Attribute chain
    if "." in py_name:
      return self._create_dotted_name(py_name)

    return cst.Name(py_name)

  def _create_expression_from_op(self, op: OperationNode) -> Optional[cst.BaseExpression]:
    """
    Attempts to map an Op to a Python Expression (e.g. Call, BinaryOp, Attribute).
    Delegates to ExpressionGeneratorMixin.
    """
    op_name = op.name.strip('"')

    if op_name == "sw.op":
      return self._expr_sw_op(op)
    elif op_name == "sw.call":
      return self._expr_sw_call(op)
    elif op_name == "sw.getattr":
      return self._expr_sw_getattr(op)
    elif op_name == "sw.constant":
      return self._expr_sw_constant(op)

    return None

  def _convert_statement_op(self, op: OperationNode) -> Optional[cst.BaseStatement]:
    """
    Maps structural ops to Statements. Delegates to StatementGeneratorMixin.
    """
    op_name = op.name.strip('"')

    if op_name == "sw.module":
      return self._convert_class_def(op)
    elif op_name == "sw.func":
      return self._convert_func_def(op)
    elif op_name == "sw.return":
      return self._convert_return(op)
    elif op_name == "sw.setattr":
      return self._convert_setattr(op)

    return None

  def _wrap_as_statement(self, op: OperationNode, expr: cst.BaseExpression) -> cst.BaseStatement:
    """
    Wraps an expression into a statement (Assign or Expr).
    Extracts semantic hints from 'type' attribute to produce readable variable names.
    """
    if op.results:
      res_ssa = op.results[0].name

      # Rule 1: Usage Count 0 -> Suppress assignment
      if self.usage_counts[res_ssa] == 0:
        return cst.SimpleStatementLine(body=[cst.Expr(value=expr)])

      # Rule 2: Void Pattern Detection
      if self._is_void_call(expr):
        return cst.SimpleStatementLine(body=[cst.Expr(value=expr)])

      # Semantic Hint Extraction
      hint: Optional[str] = None

      # 1. Check for 'type' attribute (e.g. torch.flatten)
      raw_t = self._get_attr(op, "type")
      if raw_t:
        val = raw_t.strip('"').strip("'")
        hint = val.split(".")[-1].lower()

      # 2. Check for 'name' attribute
      elif op.name.strip('"') == "sw.getattr":
        raw_n = self._get_attr(op, "name")
        if raw_n:
          hint = raw_n.strip('"')

      # 3. Fallback for constants: cst
      elif op.name.strip('"') == "sw.constant":
        hint = "cst"

      py_target = self.ctx.register(res_ssa, hint=hint)
      target_node = cst.AssignTarget(target=cst.Name(py_target))
      return cst.SimpleStatementLine(body=[cst.Assign(targets=[target_node], value=expr)])
    else:
      return cst.SimpleStatementLine(body=[cst.Expr(value=expr)])

  def _is_void_call(self, expr: cst.BaseExpression) -> bool:
    """
    Heuristic detection of calls that return None/Void (like super().__init__).
    """
    if isinstance(expr, cst.Call):
      # Detect super().__init__()
      if isinstance(expr.func, cst.Attribute):
        if expr.func.attr.value == "__init__":
          receiver = expr.func.value
          if isinstance(receiver, cst.Call) and isinstance(receiver.func, cst.Name):
            if receiver.func.value == "super":
              return True
    return False
