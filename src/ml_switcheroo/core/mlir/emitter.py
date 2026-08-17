"""MLIR Emitter Logic.

This module provides the `PythonToMlirEmitter`, a compiler front-end that
transforms Python LibCST trees into the MLIR CST object model.
"""

import libcst as cst
from typing import Dict, List, Optional, Union, Sequence

from ml_switcheroo.core.cst.base import Trivia
from ml_switcheroo.core.mlir.cst import (
  ModuleNode,
  OperationNode,
  BlockNode,
  ValueNode,
  AttributeNode,
)
from ml_switcheroo.core.scanners import get_full_name


class SSAContext:
  """Manages Single Static Assignment (SSA) variable scopes and ID allocation."""

  def __init__(self) -> None:
    """Initialize the context with a root scope."""
    self._scopes: List[Dict[str, ValueNode]] = [{}]
    self._counter: int = 0

  def enter_scope(self) -> None:
    """Push a new variable scope onto the stack."""
    self._scopes.append({})

  def exit_scope(self) -> None:
    """Pop the current variable scope from the stack."""
    if len(self._scopes) > 1:
      self._scopes.pop()

  def declare(self, name: str, value: ValueNode) -> None:
    """Register a variable name in the current scope.

    Args:
        name: The Python variable identifier.
        value: The MLIR ValueNode (SSA value) associated with it.

    """
    self._scopes[-1][name] = value

  def lookup(self, name: str) -> Optional[ValueNode]:
    """Resolve a Python variable name to its current SSA value.

    Searches scopes from innermost to outermost.

    Args:
        name: The Python identifier to look up.

    Returns:
        The associated ValueNode or None if not found.

    """
    for scope in reversed(self._scopes):
      if name in scope:
        return scope[name]
    return None

  def allocate_ssa(self, prefix: str = "%") -> ValueNode:
    """Generate a new unique SSA value.

    Args:
        prefix: String prefix for the ID (default "%").

    Returns:
        A new ValueNode with a unique ID (e.g. "%0", "%1").

    """
    val = ValueNode(name=f"{prefix}{self._counter}")
    self._counter += 1
    return val


from ml_switcheroo.core.mlir.emitter_expr import MlirEmitterExprMixin  # noqa: E402
from ml_switcheroo.core.mlir.emitter_decl import MlirEmitterDeclMixin  # noqa: E402


class PythonToMlirEmitter(MlirEmitterExprMixin, MlirEmitterDeclMixin):
  """Translates Python LibCST modules into MLIR structural nodes."""

  def __init__(self) -> None:
    """Initialize the emitter with a fresh SSA context."""
    self.ctx = SSAContext()

  def convert(self, node: cst.Module) -> ModuleNode:
    """Entry point: Converts a CST Module to an MLIR ModuleNode.

    Args:
        node: The Python LibCST Module.

    Returns:
        The resulting MLIR ModuleNode containing the translated operations.

    """
    body_block = self._emit_block(node.body)

    # Capture module header comments
    header_trivia = []
    header = getattr(node, "header", [])
    for line in header:
      if line.comment:
        text = line.comment.value.replace("#", "//", 1)
        header_trivia.append(Trivia(text))
        header_trivia.append(Trivia("\n"))
      else:
        header_trivia.append(Trivia("\n"))

      # Attach to first op
      if header_trivia and body_block.operations:
        body_block.operations[0].leading_trivia = header_trivia + body_block.operations[0].leading_trivia

    return ModuleNode(body=body_block)

  def _extract_trivia(self, node: cst.CSTNode) -> List[Trivia]:
    """Extract comments and newlines from a CST node's leading lines.

    Args:
        node: The CST node to inspect.

    Returns:
        A list of MLIR Trivias (comments translated to `//` syntax).

    """
    trivia = []
    if hasattr(node, "leading_lines"):  # pragma: no branch
      for line in node.leading_lines:
        if line.comment:
          text = line.comment.value.replace("#", "//", 1)
          trivia.append(Trivia(text))
          trivia.append(Trivia("\n"))
        elif line.newline:
          # Persist empty lines for formatting niceness
          if line.newline.value:
            trivia.append(Trivia("\n"))

    return trivia

  def _emit_block(self, body_enc: Union[cst.BaseSuite, Sequence[cst.CSTNode]], label: str = "") -> BlockNode:
    """Convert a sequence of statements (or a Suite) into an MLIR Block.

    Args:
        body_enc: A CST Suite (IndentedBlock) or list of statements.
        label: Optional block label (e.g. `^entry`).

    Returns:
        A populated BlockNode.

    """
    block = BlockNode(label=label)
    stmts = []  # type: ignore
    if isinstance(body_enc, (cst.IndentedBlock, cst.SimpleStatementSuite, cst.Module)):
      stmts = body_enc.body  # type: ignore
    elif isinstance(body_enc, (list, tuple)):
      stmts = body_enc  # type: ignore

    for stmt in stmts:
      ops = self._emit_statement(stmt)
      if ops:
        block.operations.extend(ops)
    return block

  def _emit_statement(self, stmt: cst.CSTNode) -> List[OperationNode]:
    """Dispatches statement nodes to specific handlers.

    Args:
        stmt: The statement node (ClassDef, FunctionDef, Assign, etc.).

    Returns:
        A list of MLIR OperationNodes generated from the statement.

    """
    results = []

    if isinstance(stmt, cst.ClassDef):
      results = [self._emit_class_def(stmt)]
    elif isinstance(stmt, cst.FunctionDef):
      results = [self._emit_func_def(stmt)]
    elif isinstance(stmt, cst.If):
      results = self._emit_if(stmt)
    elif isinstance(stmt, cst.While):
      results = self._emit_while(stmt)
    elif isinstance(stmt, cst.SimpleStatementLine):
      if len(stmt.body) > 0:
        node = stmt.body[0]
        results = self._dispatch_small_stmt(node)

    if results:
      extracted = self._extract_trivia(stmt)
      if extracted:
        results[0].leading_trivia = extracted + results[0].leading_trivia

    return results

  def _dispatch_small_stmt(self, node: cst.CSTNode) -> List[OperationNode]:
    """Handle small statements inside simple lines (Assign, Return, Expr).

    Args:
        node: The inner statement node.

    Returns:
        List of resulting operations.

    """
    if isinstance(node, cst.Assign):
      return self._emit_assign(node)
    elif isinstance(node, cst.Return):
      return self._emit_return(node)
    elif isinstance(node, cst.Expr):
      _, ops = self._emit_expression(node.value)
      return ops
    elif isinstance(node, (cst.Import, cst.ImportFrom)):
      return [self._emit_import(node)]
    return []

  def _emit_import(self, node: Union[cst.Import, cst.ImportFrom]) -> OperationNode:
    """Convert Import/ImportFrom to `sw.import`.

    Args:
        node: The Import or ImportFrom CST node.

    Returns:
        An OperationNode representing the `sw.import` operation.

    """
    names = []
    aliases = []
    module_val = ""

    if isinstance(node, cst.ImportFrom) and node.module:
      module_val = get_full_name(node.module)

    # Extract names and aliases
    # For ImportFrom, name.name is the object imported.
    # For Import, name.name is the module being imported.
    if isinstance(node.names, cst.ImportStar):
      names.append("*")
      aliases.append("")
    else:
      for alias in node.names:
        names.append(get_full_name(alias.name))
        if alias.asname:
          aliases.append((alias.asname.name.value if isinstance(alias.asname.name, cst.Name) else ""))
        else:
          aliases.append("")

    attrs = []
    if module_val:
      attrs.append(AttributeNode(name="module", value=f'"{module_val}"'))

    # Format list strings properly for MLIR array attribute
    quoted_names = [f'"{n}"' for n in names]
    quoted_aliases = [f'"{a}"' for a in aliases]

    attrs.append(AttributeNode(name="names", value=quoted_names))
    attrs.append(AttributeNode(name="aliases", value=quoted_aliases))

    return OperationNode(name="sw.import", attributes=attrs)

  def _emit_assign(self, node: cst.Assign) -> List[OperationNode]:
    """Convert an assignment statement.

    Emits expression operations and registers the result in the SSA context.
    Handles both variable assignment (`x = y`) and attribute assignment (`x.attr = y`)
    via `sw.setattr`.

    Args:
        node: The Assign node.

    Returns:
        List of operations generated by the assignment expression.

    """
    val, ops = self._emit_expression(node.value)

    for target in node.targets:
      t = target.target

      # Variable Assignment: x = ...
      if isinstance(t, cst.Name):
        self.ctx.declare(t.value, val)

      # Attribute Assignment: self.x = ...
      elif isinstance(t, cst.Attribute):
        # Check if base is known (e.g. self)
        base_name = self._flatten_attr(t.value)
        if base_name:
          base_val = self.ctx.lookup(base_name)
          # Support emitting setattr for object attributes
          # sw.setattr %self "layer" %val

          if base_val:
            attr_name = t.attr.value
            set_op = OperationNode(
              name="sw.setattr", operands=[base_val, val], attributes=[AttributeNode(name="name", value=f'"{attr_name}"')]
            )
            ops.append(set_op)

            # Feature: Register the Attribute name in context for future lookup
            # This allows subsequent `self.layer` access to map back to this val if needed
            # But SSA logic handles lookup by name. self.layer is complex.
            pass

    return ops

  def _emit_while(self, node: cst.While) -> List[OperationNode]:
    """Convert a while statement. Overridden in subclasses for specific dialects.

    Args:
        node: The While CST node.

    Returns:
        List of generated OperationNodes.

    """
    return []

  def _emit_if(self, node: cst.If) -> List[OperationNode]:
    """Convert an if statement. Overridden in subclasses for specific dialects.

    Args:
        node: The If CST node.

    Returns:
        List of generated OperationNodes.

    """
    return []

  def _emit_return(self, node: cst.Return) -> List[OperationNode]:
    """Convert a return statement to `sw.return`.

    Args:
        node: The Return node.

    Returns:
        List containing expression evaluation ops and the return op.

    """
    ops = []
    operands = []
    if node.value:
      val, expr_ops = self._emit_expression(node.value)
      ops.extend(expr_ops)
      operands.append(val)

    # Ensure we attach operands list correctly
    op = OperationNode(name="sw.return", operands=operands)
    ops.append(op)
    return ops

  def _flatten_attr(self, node: cst.CSTNode) -> Optional[str]:
    """Flatten a Name or Attribute chain into a string.

    Args:
        node: CST node.

    Returns:
        Dotted string (e.g. "self.layer") or None.

    """
    if isinstance(node, cst.Name):
      return node.value
    if isinstance(node, cst.Attribute):
      base = self._flatten_attr(node.value)
      if base:
        return f"{base}.{node.attr.value}"
    return None

  def _get_binop_str(self, operator: cst.BaseBinaryOp) -> str:
    """Map LibCST binary operator classes to string codes.

    Args:
        operator: The CST binary operator node.

    Returns:
        String identifier (e.g. "add", "mul", "matmul").

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
